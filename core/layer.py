from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter as scatter
from einops import einsum
from torch import Tensor, nn
from torch_geometric.nn import Linear

from .ssm import InitStrategy


class SSMTemporalGate(nn.Module):
    """An alternative approach to time encoding"""

    def __init__(self,
                 d_input,
                 d_state,
                 init_strategy:
                 Optional[InitStrategy] = None,
                 time_unit: int = 86400
                 ):
        super().__init__()
        self.time_unit = time_unit
        self.init_strategy = init_strategy or InitStrategy()
        # mimo impl
        # log(-A) in the diagonal form
        self.log_nA = nn.Parameter(torch.empty(d_state)) 
        self.B = nn.Parameter(torch.empty(d_input, d_state))
        self.C = nn.Parameter(torch.empty(d_state, d_state))
        self.reset_parameters()
        self.state_map = Linear(d_input, d_state)
        self.output_map = Linear(d_input, d_state)

    def reset_parameters(self):
        self.init_strategy.init(self.log_nA, self.B, self.C)

    def forward(self, delta_t, x):
        """Compute a temporal gate applied to the input

        Args:
            delta_t: time difference of shape [E]
            x: gathered tensor of shape [E, d]

        Returns:
            x * gate(x, delta)
        """
        state = self.state_map(x)
        delta_t = delta_t / self.time_unit
        A = - torch.exp(self.log_nA) 
        A_zoh = torch.exp(einsum(delta_t, A, 'e, d_s -> e d_s'))
        B_x = einsum(delta_t, self.B, x, 'e, d_in d_s, e d_in -> e d_s')
        gate = F.selu((A_zoh * state + B_x) @ self.C)
        return gate * self.output_map(x)


class GatedTemporalGraphAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
        use_std: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        # No time encoder needed, explicitly use temporal gating
        self.query_dim = in_channels
        self.key_dim = in_channels + edge_dim

        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k = SSMTemporalGate(self.key_dim, out_channels)
        self.v = SSMTemporalGate(self.key_dim, out_channels)

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)
        
        self.use_std = use_std
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        if self.std_encoder is not None:
            for lin in self.std_encoder:
                if hasattr(lin, 'reset_parameters'):
                    lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        source_node_vec = x_i
        target_node_vec = x_j
        if edge_attr is not None:
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        q_mat = torch.reshape(self.q_linears(source_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        k_mat = torch.reshape(F.relu(self.k(edge_time, target_node_vec)),
                              [-1, self.heads, self.d_k])
        v_mat = torch.reshape(self.v(edge_time, target_node_vec),
                              [-1, self.heads, self.d_k])

        res_att = torch.sum(torch.multiply(q_mat, k_mat),
                            dim=-1) * self.scale  # [T, N]

        scores = self.scatter_softmax(res_att, node_i)

        v = torch.multiply(torch.unsqueeze(scores, dim=2), v_mat)
        v = torch.reshape(v, [-1, self.out_channels])

        out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        
        if node_std is not None and self.std_encoder is not None:
            # node_std: [N, 1] -> std_feat: [N, out_channels]
            std_feat = self.std_encoder(node_std)
            out = out + std_feat
        
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)

        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EnhancedGatedTemporalAttention(nn.Module):
    """Temporal gated attention with cross attention and bilinear scoring."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        use_std: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        self.d_k = out_channels // heads
        
        assert self.d_k * heads == out_channels, "out_channels must be divisible by heads"
        
        self.W_q = Linear(in_channels, out_channels)
        self.W_k = Linear(in_channels + edge_dim, out_channels)
        self.W_v = Linear(in_channels + edge_dim, out_channels)
        
        self.attn_weight = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
        nn.init.xavier_uniform_(self.attn_weight)
        
        self.ssm_gate = SSMTemporalGate(in_channels + edge_dim, out_channels)
        self.gate_proj = Linear(out_channels, out_channels)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                Linear(edge_dim, out_channels // 2),
                nn.ReLU(),
                Linear(out_channels // 2, out_channels)
            )
        else:
            self.edge_encoder = None
        
        self.output_mlp = nn.Sequential(
            Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )
        
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        self.W_v.reset_parameters()
        nn.init.xavier_uniform_(self.attn_weight)
        self.ssm_gate.reset_parameters()
        self.gate_proj.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        if self.edge_encoder is not None:
            for lin in self.edge_encoder:
                if hasattr(lin, 'reset_parameters'):
                    lin.reset_parameters()
        for lin in self.output_mlp:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        if self.std_encoder is not None:
            for lin in self.std_encoder:
                if hasattr(lin, 'reset_parameters'):
                    lin.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        src = edge_index[0]
        dst = edge_index[1]
        x_src = x[src]
        x_dst = x[dst]
        
        if edge_attr is not None and self.edge_encoder is not None:
            edge_encoded = self.edge_encoder(edge_attr)
            key_input = torch.cat([x_dst, edge_encoded], dim=-1)
            value_input = torch.cat([x_dst, edge_encoded], dim=-1)
        else:
            key_input = x_dst
            value_input = x_dst
        
        q = self.W_q(x_src).view(-1, self.heads, self.d_k)
        k = self.W_k(key_input).view(-1, self.heads, self.d_k)
        v = self.W_v(value_input).view(-1, self.heads, self.d_k)
        
        # q: [E, H, d_k], attn_weight: [H, d_k, d_k], k: [E, H, d_k]
        q_expanded = q.unsqueeze(-2)  # [E, H, 1, d_k]
        k_expanded = k.unsqueeze(-1)  # [E, H, d_k, 1]
        # [E, H, 1, d_k] @ [H, d_k, d_k] -> [E, H, 1, d_k]
        q_transformed = torch.matmul(q_expanded, self.attn_weight)
        # [E, H, 1, d_k] @ [E, H, d_k, 1] -> [E, H, 1, 1]
        attn_scores = torch.matmul(q_transformed, k_expanded).squeeze(-1).squeeze(-1)
        attn_scores = attn_scores / np.sqrt(self.d_k)
        
        attn_weights = self.scatter_softmax(attn_scores, src)
        
        v_weighted = v * attn_weights.unsqueeze(-1)  # [E, H, d_k]
        aggregated = scatter.scatter_add(
            v_weighted.view(-1, self.heads * self.d_k),
            src,
            dim=0,
            dim_size=x.size(0)
        )
        
        ssm_output = self.ssm_gate(edge_time, key_input)
        gate_signal = torch.sigmoid(self.gate_proj(ssm_output))
        gate_aggregated = scatter.scatter_mean(
            gate_signal,
            src,
            dim=0,
            dim_size=x.size(0)
        )
        
        fused = aggregated * gate_aggregated
        
        fused = self.norm1(fused)
        output = self.output_mlp(torch.cat([x, fused], dim=-1))
        output = self.norm2(output + self.residual_alpha * x)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            output = output + std_feat
        
        return output
    
    def scatter_softmax(self, scores: Tensor, index: Tensor) -> Tensor:
        """Softmax grouped by the source node."""
        n_head = self.heads
        result = torch.zeros_like(scores)
        for i in range(n_head):
            result[:, i] = scatter.composite.scatter_softmax(
                scores[:, i], index
            )
        return result
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')



class SSMGraphConv(nn.Module):
    """Graph convolution whose temporal gate comes from the SSM decay matrix."""
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        self.log_nA = nn.Parameter(torch.empty(out_channels))
        self.B = nn.Parameter(torch.empty(in_channels + edge_dim, out_channels))
        
        self.input_proj = Linear(in_channels + edge_dim, out_channels)
        self.output_proj = Linear(out_channels, out_channels)
        
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        if in_channels != out_channels:
            self.res_lin = Linear(in_channels, out_channels)
        else:
            self.res_lin = None
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.log_nA, -1, 0) 
        nn.init.kaiming_uniform_(self.B)
        if self.input_proj: self.input_proj.reset_parameters()
        if self.output_proj: self.output_proj.reset_parameters()
        if self.res_lin: self.res_lin.reset_parameters()
        if self.std_encoder is not None:
            for lin in self.std_encoder:
                if hasattr(lin, 'reset_parameters'): lin.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        node_i = edge_index[0] 
        node_j = edge_index[1] 
        
        x_j = x[node_j]
        if edge_attr is not None:
            x_j = torch.cat([x_j, edge_attr], dim=1)
            
        delta_t = edge_time.view(-1, 1) / self.time_unit
        A_discrete = torch.exp(self.log_nA * delta_t) # [E, out_channels]
        
        projected_input = x_j @ self.B # [E, out_channels]
        gated_msg = A_discrete * projected_input 
        
        aggregated = scatter.scatter_add(gated_msg, node_i, dim=0, dim_size=x.size(0))
        
        out = self.output_proj(aggregated)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x
            
        return out


class MomentEnhancedTemporalConv(nn.Module):
    """Permutation-invariant convolution that also aggregates the neighbour moments."""
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        input_dim = in_channels + edge_dim
        self.phi = nn.Sequential(
            Linear(input_dim, out_channels),
            nn.GELU(),
            Linear(out_channels, out_channels)
        )
        
        self.rho = nn.Sequential(
            Linear(3 * out_channels, out_channels),
            nn.GELU(),
            Linear(out_channels, out_channels)
        )
        
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        if in_channels != out_channels:
            self.res_lin = Linear(in_channels, out_channels)
        else:
            self.res_lin = None
            
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.phi:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        for layer in self.rho:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if self.std_encoder is not None:
            for layer in self.std_encoder:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if self.res_lin: self.res_lin.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        node_i = edge_index[0] 
        node_j = edge_index[1] 
        
        x_j = x[node_j]
        if edge_attr is not None:
            x_j = torch.cat([x_j, edge_attr], dim=1)
            
        h_j = self.phi(x_j) # [E, out_channels]
        
        delta_t = edge_time.view(-1, 1) / self.time_unit
        time_gate = torch.exp(-delta_t) 
        h_j_gated = h_j * time_gate
        
        mean_m = scatter.scatter_mean(h_j_gated, node_i, dim=0, dim_size=x.size(0))
        
        # Var(X) = E[X^2] - (E[X])^2
        sq_h_j = h_j_gated ** 2
        mean_sq = scatter.scatter_mean(sq_h_j, node_i, dim=0, dim_size=x.size(0))
        var_m = F.relu(mean_sq - mean_m ** 2)
        
        # Skew(X) = E[(X - Mean)^3] / Var^(3/2)
        diff = h_j_gated - mean_m[node_i]
        cube_diff = diff ** 3
        skew_m = scatter.scatter_mean(cube_diff, node_i, dim=0, dim_size=x.size(0))
        
        moment_vec = torch.cat([mean_m, var_m, skew_m], dim=1)
        out = self.rho(moment_vec)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x
            
        return out


try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("Please install torchdiffeq: pip install torchdiffeq")

class ODEFunc(nn.Module):
    """Right-hand side of dh/dt = f(h, t)."""
    def __init__(self, in_channels, edge_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_channels + edge_dim, in_channels),
            nn.Tanh(),
            Linear(in_channels, in_channels)
        )
        
    def forward(self, t, h_with_edge, edge_index, num_nodes):
        node_i = edge_index[0]
        
        agg = scatter.scatter_mean(h_with_edge, node_i, dim=0, dim_size=num_nodes)
        
        return self.net(agg)

class DynamicMomentSSMConv(nn.Module):
    """Moment-augmented SSM convolution with a gated fusion of the two branches."""
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, 
                 use_std=False, dropout=0., residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        self.residual = residual
        
        self.log_nA = nn.Parameter(torch.empty(out_channels))
        self.B_mat = nn.Parameter(torch.empty(in_channels + edge_dim, out_channels))
        self.C_mat = nn.Parameter(torch.empty(out_channels, out_channels))
        self.state_map = Linear(in_channels + edge_dim, out_channels)
        
        self.moment_mlp = nn.Sequential(
            Linear(2 * out_channels, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )
        
        self.fusion_gate = nn.Sequential(
            Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        
        self.input_proj = Linear(in_channels + edge_dim, out_channels)
        self.output_proj = Linear(out_channels, out_channels)
        
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        if residual and in_channels != out_channels:
            self.res_lin = Linear(in_channels, out_channels)
        elif residual:
            self.res_lin = None
        else:
            self.res_lin = 'disabled'
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.log_nA, -1, 0) 
        nn.init.kaiming_uniform_(self.B_mat)
        nn.init.kaiming_uniform_(self.C_mat)
        if self.state_map: self.state_map.reset_parameters()
        for layer in self.moment_mlp:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        for layer in self.fusion_gate:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if self.input_proj: self.input_proj.reset_parameters()
        if self.output_proj: self.output_proj.reset_parameters()
        if self.std_encoder is not None:
            for layer in self.std_encoder:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if isinstance(self.res_lin, nn.Module): self.res_lin.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        node_i = edge_index[0] 
        node_j = edge_index[1] 
        
        x_j = x[node_j]
        if edge_attr is not None:
            x_j = torch.cat([x_j, edge_attr], dim=1)
            
        h_j = self.input_proj(x_j)
        state = self.state_map(x_j)
        delta_t = edge_time.view(-1, 1) / self.time_unit
        
        A = -torch.exp(self.log_nA) # [C]
        A_disc = torch.exp(A * delta_t)
        
        B_x = (x_j @ self.B_mat) * delta_t # [E, C]
        
        ssm_msg = F.selu((A_disc * state + B_x) @ self.C_mat)
        ssm_agg = scatter.scatter_add(ssm_msg, node_i, dim=0, dim_size=x.size(0))
        
        mean_m = scatter.scatter_mean(h_j, node_i, dim=0, dim_size=x.size(0))
        sq_h_j = h_j ** 2
        mean_sq = scatter.scatter_mean(sq_h_j, node_i, dim=0, dim_size=x.size(0))
        var_m = F.relu(mean_sq - mean_m ** 2)
        
        moment_feat = self.moment_mlp(torch.cat([mean_m, var_m], dim=1))
        
        gate = self.fusion_gate(ssm_agg)
        out_combined = gate * ssm_agg + (1 - gate) * moment_feat
        
        out = self.output_proj(out_combined)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        if self.residual:
            if self.res_lin is not None:
                out = out + self.res_lin(x)
            else:
                out = out + x
            
        return out


class NeuralOTemporalConv(nn.Module):
    """Temporal convolution built on a neural ODE solver."""
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        self.input_proj = Linear(in_channels, out_channels)
        if edge_dim > 0:
            self.edge_proj = Linear(edge_dim, out_channels)
        else:
            self.edge_proj = None
            
        self.ode_func = ODEFunc(out_channels, edge_dim=0)
        
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        self.output_proj = Linear(out_channels, out_channels)
        
        if in_channels != out_channels:
            self.res_lin = Linear(in_channels, out_channels)
        else:
            self.res_lin = None
            
        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        if self.edge_proj: self.edge_proj.reset_parameters()
        for layer in self.ode_func.net:
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        self.output_proj.reset_parameters()
        if self.std_encoder is not None:
            for layer in self.std_encoder:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if self.res_lin: self.res_lin.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        node_i = edge_index[0]
        node_j = edge_index[1]
        
        h0 = self.input_proj(x) # [N, C]
        
        if edge_attr is not None and self.edge_proj is not None:
            e_feat = self.edge_proj(edge_attr)
            h_j = h0[node_j] + e_feat
        else:
            h_j = h0[node_j]
            
        delta_t = edge_time / self.time_unit
        
        max_delta_t = scatter.scatter_max(delta_t, node_i, dim=0, dim_size=x.size(0))[0]
        max_delta_t = max_delta_t.clamp(min=1e-6)
        
        
        spatial_agg = scatter.scatter_mean(h_j, node_i, dim=0, dim_size=x.size(0))
        
        def dynamics(t, h):
            # h: [N, C]
            diff = spatial_agg - h 
            return self.ode_func.net[1](self.ode_func.net[0](diff))
            
        t_span = torch.linspace(0, 1, steps=10).to(x.device)
        
        solution = odeint(dynamics, h0, t_span, method='rk4') # [Steps, N, C]
        
        out = solution[-1] 
        
        out = self.output_proj(out)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x
            
        return out


class StabilizedGatedTemporalAttention(nn.Module):
    """Gated temporal attention with learnable exponential decay."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
        use_std: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        self.use_std = use_std

        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        self.time_decay = nn.Parameter(torch.ones(heads, 1)) 
        self.time_bias = nn.Parameter(torch.zeros(heads, 1))

        self.q_lin = Linear(in_channels, out_channels)
        self.k_lin = Linear(in_channels + edge_dim, out_channels)
        self.v_lin = Linear(in_channels + edge_dim, out_channels)
        
        self.diff_proj = Linear(in_channels, out_channels)

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.GELU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], 
            norm=None,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.time_decay)
        nn.init.zeros_(self.time_bias)
        
        self.q_lin.reset_parameters()
        self.k_lin.reset_parameters()
        self.v_lin.reset_parameters()
        self.diff_proj.reset_parameters()
        
        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        if self.std_encoder is not None:
            for layer in self.std_encoder:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        node_i = edge_index[0]
        node_j = edge_index[1]
        
        x_i = x[node_i]
        x_j = x[node_j]

        target_vec = x_j
        if edge_attr is not None:
            target_vec = torch.cat([target_vec, edge_attr], dim=1)
        
        k = self.k_lin(target_vec).view(-1, self.heads, self.d_k)
        v = self.v_lin(target_vec).view(-1, self.heads, self.d_k)

        q_base = self.q_lin(x_i)
        diff_feat = self.diff_proj(x_i - x_j)
        
        q_enhanced = (q_base + diff_feat).view(-1, self.heads, self.d_k)

        att_base = torch.sum(q_enhanced * k, dim=-1) * self.scale # [E, heads]

        # time_decay: [heads, 1], edge_time: [E]
        decay_factor = torch.exp(
            -F.softplus(self.time_decay.t()) * edge_time.view(-1, 1) + self.time_bias.t()
        ) # [E, heads]
        
        att_final = att_base * decay_factor

        scores = self.scatter_softmax(att_final, node_i)
        scores = self.dropout(scores)
        
        v_weighted = v * scores.unsqueeze(-1)
        v_weighted = v_weighted.view(-1, self.out_channels)
        
        out = scatter.scatter_add(v_weighted, node_i, dim=0, dim_size=x.size(0))

        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat

        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        
        if self.residual:
            if self.lin_res is not None:
                out = out + self.lin_res(x)
            else:
                out = out + x
                
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)
        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class RobustCorrectedTemporalAttention(nn.Module):
    """Temporal attention with a gated residual injection of the descriptor."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
        use_std: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        self.use_std = use_std

        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        self.time_decay = nn.Parameter(torch.ones(heads, 1)) 
        self.time_bias = nn.Parameter(torch.zeros(heads, 1))

        self.q_lin = Linear(in_channels, out_channels)
        self.k_lin = Linear(in_channels + edge_dim, out_channels)
        self.v_lin = Linear(in_channels + edge_dim, out_channels)
        self.diff_proj = Linear(in_channels, out_channels)

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.GELU(),
                Linear(out_channels, out_channels)
            )
            self.std_gate = nn.Sequential(
                Linear(1, out_channels),
                nn.Sigmoid()
            )
        else:
            self.std_encoder = None
            self.std_gate = None

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], 
            norm=None,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.time_decay)
        nn.init.zeros_(self.time_bias)
        
        self.q_lin.reset_parameters()
        self.k_lin.reset_parameters()
        self.v_lin.reset_parameters()
        self.diff_proj.reset_parameters()
        
        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        if self.std_encoder is not None:
            for layer in self.std_encoder:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        if self.std_gate is not None:
            for layer in self.std_gate:
                if hasattr(layer, 'reset_parameters'): layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        node_i = edge_index[0]
        node_j = edge_index[1]
        
        x_i = x[node_i]
        x_j = x[node_j]

        target_vec = x_j
        if edge_attr is not None:
            target_vec = torch.cat([target_vec, edge_attr], dim=1)
        
        k = self.k_lin(target_vec).view(-1, self.heads, self.d_k)
        v = self.v_lin(target_vec).view(-1, self.heads, self.d_k)

        q_base = self.q_lin(x_i)
        diff_feat = self.diff_proj(x_i - x_j)
        q_enhanced = (q_base + diff_feat).view(-1, self.heads, self.d_k)

        att_base = torch.sum(q_enhanced * k, dim=-1) * self.scale
        decay_factor = torch.exp(
            -F.softplus(self.time_decay.t()) * edge_time.view(-1, 1) + self.time_bias.t()
        )
        att_final = att_base * decay_factor

        scores = self.scatter_softmax(att_final, node_i)
        scores = self.dropout(scores)
        
        v_weighted = v * scores.unsqueeze(-1)
        agg_out = scatter.scatter_add(
            v_weighted.view(-1, self.out_channels), 
            node_i, dim=0, dim_size=x.size(0)
        )

        if node_std is not None and self.std_encoder is not None:
            std_embed = self.std_encoder(node_std)       # [N, out_channels]
            gate_val = self.std_gate(node_std)           # [N, out_channels]
            agg_out = agg_out + gate_val * std_embed

        out = torch.cat([x, agg_out], dim=1)
        out = self.merger(out)
        
        if self.residual:
            if self.lin_res is not None:
                out = out + self.lin_res(x)
            else:
                out = out + x
                
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)
        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


from torch_geometric.nn.conv.gcn_conv import gcn_norm
class GatedTemporalLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
        aggr='gcn',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        self.query_dim = in_channels
        self.key_dim = in_channels + edge_dim
        
        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k = SSMTemporalGate(self.key_dim, out_channels)
        self.v = SSMTemporalGate(self.key_dim, out_channels)
        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)

        self.aggr = aggr # gcn, mean, max

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        source_node_vec = x_i
        target_node_vec = x_j
        if edge_attr is not None: 
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        v_mat = self.v(edge_time, target_node_vec)
        if self.aggr=='gcn':
            edge_index, scores = gcn_norm(edge_index, add_self_loops=False, flow='target_to_source')
            v = torch.multiply(torch.unsqueeze(scores, dim=1), v_mat)
            out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        elif self.aggr=='mean':
            out = scatter.scatter_mean(v_mat, node_i, dim=0, dim_size=x.size(0))
        elif self.aggr=='max':
            out = scatter.scatter_max(v_mat, node_i, dim=0, dim_size=x.size(0))[0]
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    




from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Identity
import torch_scatter as scatter
from torch_geometric.nn import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
class TGATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_encoder: nn.Module,
        time_dim: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.time_dim = time_dim
        self.time_encoder = time_encoder
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)
        self.query_dim = in_channels + time_dim 
        self.key_dim = in_channels + time_dim + edge_dim

        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k_linears = nn.Sequential(Linear(self.key_dim, out_channels),
                                       nn.ReLU())
        self.v_linears = Linear(self.key_dim, out_channels)

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.k_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        self.v_linears.reset_parameters()
        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        src_time = self.time_encoder(x_i.new_zeros(x_i.size(0), 1))
        edge_time = self.time_encoder(edge_time.unsqueeze(1))
        source_node_vec = torch.cat([x_i, src_time], dim=1)
        target_node_vec = torch.cat([x_j, edge_time], dim=1) 
        if edge_attr is not None:
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        q_mat = torch.reshape(self.q_linears(source_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        k_mat = torch.reshape(self.k_linears(target_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        v_mat = torch.reshape(self.v_linears(target_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]

        res_att = torch.sum(torch.multiply(q_mat, k_mat),
                            dim=-1) * self.scale  # [T, N]

        scores = self.scatter_softmax(res_att, node_i)

        v = torch.multiply(torch.unsqueeze(scores, dim=2), v_mat)
        v = torch.reshape(v, [-1, self.out_channels])

        out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)

        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')





class TimeEncoder(nn.Module):
    """Time Encoding proposed by TGAT."""
    def __init__(self, dimension: int, time_unit: int = 86400, trainable=True):
        super().__init__()
        self.dimension = dimension
        self.time_unit = time_unit
        self.w = Linear(1, dimension)
        self.dimension = dimension
        self.trainable = trainable

        self.reset_parameters()

    def reset_parameters(self):
        w = 1 / 10**np.linspace(0, 1.5, self.dimension)
        w = torch.from_numpy(w).view(self.dimension, -1)
        self.w.weight = nn.Parameter(w.to(torch.float))
        self.w.bias = nn.init.zeros_(self.w.bias)
        
        if not self.trainable:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False            

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        out = torch.cos(self.w(t))
        return out



class MLP(torch.nn.Module):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        act_first = act_first or kwargs.get("relu_first", False)

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given "
                                 f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1]  # Do not add norm layer at the end
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return x
class TimeStructBias(nn.Module):
    """Per-pair structural bias added to the attention logits."""
    def __init__(self, num_heads, K=8, t_max=86400 * 30, scale_init=5.0):
        super().__init__()
        log_t_max = torch.log(torch.tensor(float(t_max) + 1.0)).item()
        self.register_buffer('mu', torch.linspace(0.0, log_t_max, K))
        self.register_buffer('sigma', torch.full((K,), log_t_max / max(K - 1, 1)))
        self.w = nn.Parameter(torch.zeros(num_heads, K))
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def reset_parameters(self):
        with torch.no_grad():
            self.w.zero_()

    def forward(self, node_time_dense):
        """Map the dense per-pair times to an additive attention bias."""
        diff = (node_time_dense.unsqueeze(0) - node_time_dense.unsqueeze(1)).abs()  # [N, N]
        log_diff = torch.log1p(diff.clamp(min=0))                                  # [N, N]
        phi = torch.exp(-(log_diff.unsqueeze(-1) - self.mu) ** 2
                        / (2 * self.sigma ** 2 + 1e-9))                             # [N, N, K]
        return self.scale * torch.einsum('nmk,hk->hnm', phi, self.w)              # [num_heads, N, N]

    def __repr__(self):
        return (f'{self.__class__.__name__}(heads={self.w.size(0)}, '
                f'K={self.mu.size(0)})')



class TiGNConv(nn.Module):
    """Local aggregation with per-neighbour softmax selection and a time gate."""
    def __init__(self, in_channels, out_channels, edge_dim=0, dropout=0., time_unit=86400):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit

        key_dim = in_channels + edge_dim

        # Q (source), K (neighbor+edge), V (neighbor+edge)
        self.W_q = Linear(in_channels, out_channels)
        self.W_k = Linear(key_dim, out_channels)
        self.W_v = Linear(key_dim, out_channels)

        self.gamma_mlp = nn.Sequential(
            Linear(in_channels, out_channels // 4),
            nn.ReLU(),
            Linear(out_channels // 4, 1, bias=False)
        )
        self.gamma_bias = nn.Parameter(torch.tensor(-2.0))

        self.output_proj = nn.Sequential(
            Linear(out_channels, out_channels),
            nn.ReLU(),
            Linear(out_channels, out_channels)
        )

        if in_channels != out_channels:
            self.res_lin = Linear(in_channels, out_channels)
        else:
            self.res_lin = None

        self.dropout = nn.Dropout(dropout)
        self.scale = out_channels ** -0.5
        self.reset_parameters()

    def reset_parameters(self):
        self.W_q.reset_parameters()
        self.W_k.reset_parameters()
        self.W_v.reset_parameters()
        for layer in self.gamma_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.gamma_bias.data.fill_(-2.0)
        self.output_proj[0].reset_parameters()
        self.output_proj[2].reset_parameters()
        if self.res_lin is not None:
            self.res_lin.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None):
        """x: [N, C], edge_index: [2, E], edge_time: [E], edge_attr: [E, D] or None."""
        src, dst = edge_index[0], edge_index[1]

        x_src = x[src]                                   # [E, C]
        x_dst = x[dst]                                   # [E, C]

        neighbor_feat = x_dst
        if edge_attr is not None:
            neighbor_feat = torch.cat([neighbor_feat, edge_attr], dim=-1)  # [E, C+D]

        q = self.W_q(x_src)                              # [E, C_out]
        k = self.W_k(neighbor_feat)                      # [E, C_out]
        content_score = (q * k).sum(dim=-1) * self.scale  # [E]

        gamma = F.softplus(self.gamma_mlp(x) + self.gamma_bias)  # [N, 1]
        gamma_src = gamma[src].squeeze(-1)                # [E]

        delta_t = edge_time.float() / self.time_unit
        time_score = -gamma_src * delta_t

        combined = content_score + time_score             # [E]

        scores = scatter_softmax_per_source(combined, src)
        scores = self.dropout(scores)

        v = self.W_v(neighbor_feat)                       # [E, C_out]
        v_weighted = v * scores.unsqueeze(-1)             # [E, C_out]
        out = scatter.scatter_add(v_weighted, src, dim=0, dim_size=x.size(0))  # [N, C_out]

        out = self.output_proj(out)

        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


def scatter_softmax_per_source(scores, src_index):
    """Softmax of the incoming scores of every source node."""
    return scatter.composite.scatter_softmax(scores, src_index)


class GatedSSMNeighborAgg(nn.Module):
    """Aggregate the neighbours in temporal order with a gated SSM scan."""

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400, log_space=False):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit
        self.log_space = log_space

        nbr_dim = channels + edge_dim
        self.nbr_proj = Linear(nbr_dim, channels)

        self.W_g = Linear(channels, channels)
        self.U_g = Linear(channels, channels)
        self.time_enc = TimeEncoder(channels)
        self.T_g = Linear(channels, channels)

        self.W_z = Linear(channels, channels)
        self.U_z = Linear(channels, channels)

        self.W0 = Linear(channels, channels)

        self.output_mlp = nn.Sequential(
            Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels, channels),
        )

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.nbr_proj.reset_parameters()
        self.W_g.reset_parameters()
        self.U_g.reset_parameters()
        self.time_enc.reset_parameters()
        self.T_g.reset_parameters()
        self.W_z.reset_parameters()
        self.U_z.reset_parameters()
        self.W0.reset_parameters()
        for layer in self.output_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None):
        """x: [N, C], edge_index: [2, E], edge_time: [E] (delta_t),
           edge_attr: [E, D] or None"""
        src, dst = edge_index[0], edge_index[1]
        N, C = x.size(0), self.channels

        nbr_feat = x[dst]
        if edge_attr is not None:
            nbr_feat = torch.cat([nbr_feat, edge_attr], dim=-1)
        nbr_proj = self.nbr_proj(nbr_feat)           # [E, C]

        sort_key = src.float() * 1e9 + edge_time.float()
        perm = torch.argsort(sort_key)
        src_ord = src[perm]
        nbr_ord = nbr_proj[perm]
        time_ord = edge_time[perm]

        E = src_ord.size(0)
        counts = scatter.scatter_add(
            torch.ones(E, device=x.device), src_ord, dim=0, dim_size=N).long()
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device),
                         counts.cumsum(0)])
        positions = torch.arange(E, device=x.device) - ptr[src_ord]
        M = int(counts.max().item())

        # source → active
        has_edges = counts > 0
        active = torch.where(has_edges)[0]
        num_active = active.size(0)
        src_to_aid = torch.zeros(N, dtype=torch.long, device=x.device)
        src_to_aid[active] = torch.arange(num_active, device=x.device)
        aid = src_to_aid[src_ord]

        flat_idx = aid * M + positions
        pad_nbr = torch.zeros(num_active * M, C, device=x.device)
        pad_nbr[flat_idx] = nbr_ord
        pad_nbr = pad_nbr.view(num_active, M, C)

        pad_time = torch.zeros(num_active * M, device=x.device)
        pad_time[flat_idx] = time_ord.float()
        pad_time = pad_time.view(num_active, M)

        seq_mask = torch.zeros(num_active * M, dtype=torch.bool, device=x.device)
        seq_mask[flat_idx] = True
        seq_mask = seq_mask.view(num_active, M)

        h0 = torch.tanh(self.W0(x[active]))
        C_chunks = 3
        chunk_size = max(1, (M + C_chunks - 1) // C_chunks)

        h = h0
        for c in range(C_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, M)
            if start >= M:
                break

            c_nbr = pad_nbr[:, start:end, :]
            c_time = pad_time[:, start:end]
            c_mask = seq_mask[:, start:end]

            t_enc = self.time_enc(c_time.unsqueeze(-1))
            g = torch.sigmoid(self.W_g(c_nbr) + self.U_g(h.unsqueeze(1)) + self.T_g(t_enc))
            z = torch.tanh(self.W_z(c_nbr) + self.U_z(h.unsqueeze(1)))

            mask_3d = c_mask.unsqueeze(-1)
            a = torch.where(mask_3d, 1.0 - g, torch.ones_like(g))
            b = torch.where(mask_3d, g * z, torch.zeros_like(z))

            if self.log_space:
                log_a = torch.log(a.clamp(min=1e-12))
                log_cumdecay = torch.cumsum(log_a, dim=1)
                final_log_decay = log_cumdecay[:, -1, :]
                log_weights = final_log_decay.unsqueeze(1) - log_cumdecay
                weights = torch.exp(log_weights)
                h = torch.exp(final_log_decay) * h + (weights * b).sum(dim=1)
            else:
                cumdecay = torch.cumprod(a, dim=1)
                final_decay = cumdecay[:, -1, :]
                weights = final_decay.unsqueeze(1) / cumdecay.clamp(min=1e-8)
                h = final_decay * h + (weights * b).sum(dim=1)

        x_active = x[active]
        fused = self.output_mlp(torch.cat([x_active, h], dim=-1))
        fused = self.dropout(fused)
        fused = self.norm(x_active + fused)

        out = x.clone()
        out[active] = fused
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.channels})'


class SrcAwareGateConv(nn.Module):
    """Per-edge message modulated by the source node, with an independent time gate."""

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit

        self.W_n = Linear(channels, channels)
        self.W_s = Linear(channels, channels)
        self.W_e = Linear(edge_dim, channels) if edge_dim > 0 else None

        self.time_enc = TimeEncoder(channels)
        gate_in = channels * 3
        self.gate_mlp = nn.Sequential(
            Linear(gate_in, channels // 4),
            nn.ReLU(),
            Linear(channels // 4, 1),
        )

        self.output_mlp = nn.Sequential(
            Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels, channels),
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.W_n.reset_parameters()
        self.W_s.reset_parameters()
        for p in self.W_s.parameters():
            nn.init.zeros_(p)
        if self.W_e is not None:
            self.W_e.reset_parameters()
        self.time_enc.reset_parameters()
        for layer in self.gate_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        nn.init.constant_(self.gate_mlp[-1].bias, -1.0)
        for layer in self.output_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        x_src, x_dst = x[src], x[dst]                     # [E, C]

        content = self.W_n(x_dst) + self.W_s(x_src)
        if self.W_e is not None and edge_attr is not None:
            content = content + self.W_e(edge_attr)
        content = F.relu(content)                          # [E, C]

        t_enc = self.time_enc(
            (edge_time.float() / self.time_unit).unsqueeze(1))  # [E, C]
        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([x_src, x_dst, t_enc], dim=-1)))  # [E, 1]

        msg = gate * content                               # [E, C]

        out = scatter.scatter_add(msg, src, dim=0, dim_size=N)  # [N, C]

        out = self.output_mlp(out)
        out = self.dropout(out)
        out = self.norm(x + out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.channels})'


class AnchoredGatedSSMNeighborAgg(nn.Module):
    """Gated SSM scan anchored on the source node, which removes the drift."""

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400, log_space=False):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit
        self.log_space = log_space

        nbr_dim = channels + edge_dim
        self.nbr_proj = Linear(nbr_dim, channels)

        self.W_g = Linear(channels, channels)
        self.U_g = Linear(channels, channels)
        self.time_enc = TimeEncoder(channels)
        self.T_g = Linear(channels, channels)
        self.S_g = Linear(channels, channels)
        self.gate_bias = nn.Parameter(torch.zeros(1))

        self.W_z = Linear(channels, channels)
        self.U_z = Linear(channels, channels)
        self.S_z = Linear(channels, channels)

        self.h_norm = nn.LayerNorm(channels)

        self.W0 = Linear(channels, channels)

        self.output_mlp = nn.Sequential(
            Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels, channels),
        )

        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.nbr_proj.reset_parameters()
        self.W_g.reset_parameters()
        self.U_g.reset_parameters()
        self.time_enc.reset_parameters()
        self.T_g.reset_parameters()
        self.S_g.reset_parameters()
        self.W_z.reset_parameters()
        self.U_z.reset_parameters()
        self.S_z.reset_parameters()
        self.W0.reset_parameters()
        nn.init.zeros_(self.gate_bias)
        self.h_norm.reset_parameters()
        for layer in self.output_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None):
        """x: [N, C], edge_index: [2, E], edge_time: [E] (delta_t),
           edge_attr: [E, D] or None"""
        src, dst = edge_index[0], edge_index[1]
        N, C = x.size(0), self.channels

        nbr_feat = x[dst]
        if edge_attr is not None:
            nbr_feat = torch.cat([nbr_feat, edge_attr], dim=-1)
        nbr_proj = self.nbr_proj(nbr_feat)           # [E, C]

        sort_key = src.float() * 1e9 + edge_time.float()
        perm = torch.argsort(sort_key)
        src_ord = src[perm]
        nbr_ord = nbr_proj[perm]
        time_ord = edge_time[perm]

        E = src_ord.size(0)
        counts = scatter.scatter_add(
            torch.ones(E, device=x.device), src_ord, dim=0, dim_size=N).long()
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device),
                         counts.cumsum(0)])
        positions = torch.arange(E, device=x.device) - ptr[src_ord]
        M = int(counts.max().item())

        has_edges = counts > 0
        active = torch.where(has_edges)[0]
        num_active = active.size(0)
        src_to_aid = torch.zeros(N, dtype=torch.long, device=x.device)
        src_to_aid[active] = torch.arange(num_active, device=x.device)
        aid = src_to_aid[src_ord]

        flat_idx = aid * M + positions
        pad_nbr = torch.zeros(num_active * M, C, device=x.device)
        pad_nbr[flat_idx] = nbr_ord
        pad_nbr = pad_nbr.view(num_active, M, C)

        pad_time = torch.zeros(num_active * M, device=x.device)
        pad_time[flat_idx] = time_ord.float()
        pad_time = pad_time.view(num_active, M)

        seq_mask = torch.zeros(num_active * M, dtype=torch.bool, device=x.device)
        seq_mask[flat_idx] = True
        seq_mask = seq_mask.view(num_active, M)

        x_src = x[active]

        h0 = torch.tanh(self.W0(x_src))
        C_chunks = 3
        chunk_size = max(1, (M + C_chunks - 1) // C_chunks)

        h = h0
        for c in range(C_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, M)
            if start >= M:
                break

            c_nbr = pad_nbr[:, start:end, :]
            c_time = pad_time[:, start:end]
            c_mask = seq_mask[:, start:end]

            t_enc = self.time_enc(c_time.unsqueeze(-1))              # [B, L, C]
            h_normed = self.h_norm(h)
            src_term_g = self.S_g(x_src).unsqueeze(1)
            src_term_z = self.S_z(x_src).unsqueeze(1)

            g = torch.sigmoid(
                self.W_g(c_nbr) + self.U_g(h_normed.unsqueeze(1))
                + self.T_g(t_enc) + src_term_g + self.gate_bias)     # [B, L, C]
            z = torch.tanh(
                self.W_z(c_nbr) + self.U_z(h_normed.unsqueeze(1))
                + src_term_z)                                         # [B, L, C]

            mask_3d = c_mask.unsqueeze(-1)
            a = torch.where(mask_3d, 1.0 - g, torch.ones_like(g))
            b = torch.where(mask_3d, g * z, torch.zeros_like(z))

            if self.log_space:
                log_a = torch.log(a.clamp(min=1e-12))
                log_cumdecay = torch.cumsum(log_a, dim=1)
                final_log_decay = log_cumdecay[:, -1, :]
                log_weights = final_log_decay.unsqueeze(1) - log_cumdecay
                weights = torch.exp(log_weights)
                h = torch.exp(final_log_decay) * h + (weights * b).sum(dim=1)
            else:
                cumdecay = torch.cumprod(a, dim=1)
                final_decay = cumdecay[:, -1, :]
                weights = final_decay.unsqueeze(1) / cumdecay.clamp(min=1e-8)
                h = final_decay * h + (weights * b).sum(dim=1)

        fused = self.output_mlp(torch.cat([x_src, h], dim=-1))
        fused = self.dropout(fused)
        fused = self.norm(x_src + fused)

        out = x.clone()
        out[active] = fused
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.channels})'
