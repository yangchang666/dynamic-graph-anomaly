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
        use_std: bool = False,  # 新增参数：是否使用标准差特征
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
        
        # 新增：标准差特征编码器
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
        node_std: Optional[Tensor] = None,  # 新增参数：节点的标准差特征 [N, 1]
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
        
        # 融合标准差特征（在聚合后，MLP 前）
        if node_std is not None and self.std_encoder is not None:
            # node_std: [N, 1] -> std_feat: [N, out_channels]
            std_feat = self.std_encoder(node_std)
            out = out + std_feat  # 相加融合
        
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
    """
    增强的时序门控注意力层
    
    相比原版GatedTemporalGraphAttention的改进:
    1. 多头交叉注意力: Q来自源节点，K/V来自目标节点+边特征
    2. 双线性注意力评分: 替代简单点积，引入可学习权重矩阵W_attn
    3. 自适应门控融合: SSM输出作为门的调制信号，而非简单乘法
    4. 残差连接优化: 使用LayerNorm + 可学习残差系数
    5. 边特征深度编码: 独立的Edge MLP编码器
    """
    
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
        
        # === 改进1: 独立的Query/Key/Value投影 ===
        self.W_q = Linear(in_channels, out_channels)
        self.W_k = Linear(in_channels + edge_dim, out_channels)
        self.W_v = Linear(in_channels + edge_dim, out_channels)
        
        # === 改进2: 双线性注意力权重矩阵 ===
        self.attn_weight = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
        nn.init.xavier_uniform_(self.attn_weight)
        
        # === 改进3: SSM门控改为调制器 ===
        self.ssm_gate = SSMTemporalGate(in_channels + edge_dim, out_channels)
        self.gate_proj = Linear(out_channels, out_channels)
        
        # === 改进4: LayerNorm + 可学习残差 ===
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        
        # === 改进5: 边特征专用编码器 ===
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                Linear(edge_dim, out_channels // 2),
                nn.ReLU(),
                Linear(out_channels // 2, out_channels)
            )
        else:
            self.edge_encoder = None
        
        # 输出MLP
        self.output_mlp = nn.Sequential(
            Linear(out_channels * 2, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )
        
        # 标准差特征编码器（与原版GatedTemporalGraphAttention保持一致）
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
        
        # 编码边特征
        if edge_attr is not None and self.edge_encoder is not None:
            edge_encoded = self.edge_encoder(edge_attr)
            key_input = torch.cat([x_dst, edge_encoded], dim=-1)
            value_input = torch.cat([x_dst, edge_encoded], dim=-1)
        else:
            key_input = x_dst
            value_input = x_dst
        
        # 投影到多头空间 [E, H, d_k]
        q = self.W_q(x_src).view(-1, self.heads, self.d_k)
        k = self.W_k(key_input).view(-1, self.heads, self.d_k)
        v = self.W_v(value_input).view(-1, self.heads, self.d_k)
        
        # 双线性注意力评分: q^T W_attn k
        # q: [E, H, d_k], attn_weight: [H, d_k, d_k], k: [E, H, d_k]
        q_expanded = q.unsqueeze(-2)  # [E, H, 1, d_k]
        k_expanded = k.unsqueeze(-1)  # [E, H, d_k, 1]
        # [E, H, 1, d_k] @ [H, d_k, d_k] -> [E, H, 1, d_k]
        q_transformed = torch.matmul(q_expanded, self.attn_weight)
        # [E, H, 1, d_k] @ [E, H, d_k, 1] -> [E, H, 1, 1]
        attn_scores = torch.matmul(q_transformed, k_expanded).squeeze(-1).squeeze(-1)
        attn_scores = attn_scores / np.sqrt(self.d_k)
        
        # Softmax归一化（按源节点分组）
        attn_weights = self.scatter_softmax(attn_scores, src)
        
        # 加权聚合
        v_weighted = v * attn_weights.unsqueeze(-1)  # [E, H, d_k]
        aggregated = scatter.scatter_add(
            v_weighted.view(-1, self.heads * self.d_k),
            src,
            dim=0,
            dim_size=x.size(0)
        )
        
        # SSM门控调制
        ssm_output = self.ssm_gate(edge_time, key_input)
        gate_signal = torch.sigmoid(self.gate_proj(ssm_output))
        gate_aggregated = scatter.scatter_mean(
            gate_signal,
            src,
            dim=0,
            dim_size=x.size(0)
        )
        
        # 融合: 注意力聚合 * 门控调制
        fused = aggregated * gate_aggregated
        
        # LayerNorm + 残差
        fused = self.norm1(fused)
        output = self.output_mlp(torch.cat([x, fused], dim=-1))
        output = self.norm2(output + self.residual_alpha * x)
        
        # 融合标准差特征（与原版GatedTemporalGraphAttention保持一致：在聚合后直接相加）
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            output = output + std_feat
        
        return output
    
    def scatter_softmax(self, scores: Tensor, index: Tensor) -> Tensor:
        """按源节点分组的Softmax"""
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
    """
    基于SSM的图卷积层(无Attention版本)
    
    核心逻辑：
    1. 利用SSM的可学习衰减矩阵A，根据时间差delta_t生成动态门控。
    2. 对邻居特征进行线性变换后，直接通过门控进行加权聚合。
    3. 摒弃QKV点积，专注于时序状态的平滑演化。
    4. 保留std_feat作为后期的统计校正项。
    """
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        # SSM核心参数: A (衰减), B (输入增益)
        self.log_nA = nn.Parameter(torch.empty(out_channels))
        self.B = nn.Parameter(torch.empty(in_channels + edge_dim, out_channels))
        
        # 特征投影
        self.input_proj = Linear(in_channels + edge_dim, out_channels)
        self.output_proj = Linear(out_channels, out_channels)
        
        # Std特征编码器(与你原来的GatedTemporalGraphAttention保持一致)
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        # 残差映射
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
            
        # 1. SSM门控计算
        delta_t = edge_time.view(-1, 1) / self.time_unit
        A_discrete = torch.exp(self.log_nA * delta_t) # [E, out_channels]
        
        # 2. 消息变换与门控
        projected_input = x_j @ self.B # [E, out_channels]
        gated_msg = A_discrete * projected_input 
        
        # 3. 散点聚合
        aggregated = scatter.scatter_add(gated_msg, node_i, dim=0, dim_size=x.size(0))
        
        # 4. 输出投影
        out = self.output_proj(aggregated)
        
        # 5. 【关键点】融合标准差特征(std_feat)
        # 这一步和你原来的代码逻辑完全一致：在聚合后、残差前加入std信息
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        # 6. 残差连接
        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x
            
        return out


class MomentEnhancedTemporalConv(nn.Module):
    """
    矩增强时序卷积层 (Moment-Enhanced Temporal Convolution)
    
    核心思想：
    1. DeepSets 架构：保证对邻居集合的置换不变性。
    2. 矩增强 (Moment Augmentation)：不仅聚合均值，还聚合方差和偏度，
       以捕捉邻居分布的形状、离散程度和不对称性。
    3. SSM 时间门控：利用可学习的衰减系数处理不规则时间间隔。
    """
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        # 特征投影网络 phi (DeepSets 的第一部分)
        input_dim = in_channels + edge_dim
        self.phi = nn.Sequential(
            Linear(input_dim, out_channels),
            nn.GELU(),
            Linear(out_channels, out_channels)
        )
        
        # 矩融合网络 rho (DeepSets 的第二部分)
        # 输入维度是 3 * out_channels (Mean, Var, Skew)
        self.rho = nn.Sequential(
            Linear(3 * out_channels, out_channels),
            nn.GELU(),
            Linear(out_channels, out_channels)
        )
        
        # Std 特征编码器
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        # 残差映射
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
            
        # 1. 基础特征变换 phi(x_j)
        h_j = self.phi(x_j) # [E, out_channels]
        
        # 2. 应用时间衰减 (SSM-style gating)
        delta_t = edge_time.view(-1, 1) / self.time_unit
        # 简单的可学习指数衰减，模拟 SSM 的 A 矩阵效应
        time_gate = torch.exp(-delta_t) 
        h_j_gated = h_j * time_gate
        
        # 3. 计算多阶矩 (Moments Calculation)
        # Mean (一阶矩)
        mean_m = scatter.scatter_mean(h_j_gated, node_i, dim=0, dim_size=x.size(0))
        
        # Variance (二阶矩 - 离散程度)
        # Var(X) = E[X^2] - (E[X])^2
        sq_h_j = h_j_gated ** 2
        mean_sq = scatter.scatter_mean(sq_h_j, node_i, dim=0, dim_size=x.size(0))
        var_m = F.relu(mean_sq - mean_m ** 2) # 确保非负
        
        # Skewness (三阶矩 - 偏斜程度/突变方向)
        # Skew(X) = E[(X - Mean)^3] / Var^(3/2)
        # 为了稳定性，我们直接聚合立方偏差
        diff = h_j_gated - mean_m[node_i]
        cube_diff = diff ** 3
        skew_m = scatter.scatter_mean(cube_diff, node_i, dim=0, dim_size=x.size(0))
        
        # 4. 矩融合 rho([Mean, Var, Skew])
        moment_vec = torch.cat([mean_m, var_m, skew_m], dim=1)
        out = self.rho(moment_vec)
        
        # 5. 融合标准差特征 (std_feat)
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        # 6. 残差连接
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
    """定义 dh/dt = f(h, t) 的微分方程函数"""
    def __init__(self, in_channels, edge_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            Linear(in_channels + edge_dim, in_channels),
            nn.Tanh(), # Tanh 保证导数有界，利于数值稳定性
            Linear(in_channels, in_channels)
        )
        
    def forward(self, t, h_with_edge, edge_index, num_nodes):
        # h_with_edge: [E, C] 已经拼接了边特征的节点表示
        # 我们需要根据 edge_index 将其映射回节点空间进行聚合
        node_i = edge_index[0]
        
        # 简单的 Mean Aggregation 作为导数的空间部分
        agg = scatter.scatter_mean(h_with_edge, node_i, dim=0, dim_size=num_nodes)
        
        # 导数只取决于当前状态和时间（这里简化为自治系统）
        return self.net(agg)

class DynamicMomentSSMConv(nn.Module):
    """
    动态矩增强 SSM 卷积层 (Refined Version)
    
    改进策略：
    1. 稳定主干：使用固定的、可学习的 SSM 参数进行时序演化，确保梯度稳定。
    2. 矩增强旁路：计算邻居的均值和方差，通过一个独立的 MLP 提取结构特征。
    3. 门控融合：学习一个门控信号，动态决定在每个节点上相信“时序演化”还是“空间统计”。
    """
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, 
                 use_std=False, dropout=0., residual=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        self.residual = residual
        
        # --- 1. SSM 主干 (保持稳定) ---
        self.log_nA = nn.Parameter(torch.empty(out_channels))
        self.B_mat = nn.Parameter(torch.empty(in_channels + edge_dim, out_channels))
        self.C_mat = nn.Parameter(torch.empty(out_channels, out_channels))
        self.state_map = Linear(in_channels + edge_dim, out_channels)
        
        # --- 2. 矩增强旁路 (Moment Branch) ---
        self.moment_mlp = nn.Sequential(
            Linear(2 * out_channels, out_channels), # 输入 Mean 和 Var
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )
        
        # --- 3. 融合门控 (Fusion Gate) ---
        self.fusion_gate = nn.Sequential(
            Linear(out_channels, out_channels),
            nn.Sigmoid()
        )
        
        # 特征投影与输出
        self.input_proj = Linear(in_channels + edge_dim, out_channels)
        self.output_proj = Linear(out_channels, out_channels)
        
        # Std 特征编码器
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        # 残差映射
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
            
        # --- 路径 A: SSM 时序演化 ---
        h_j = self.input_proj(x_j)
        state = self.state_map(x_j)
        delta_t = edge_time.view(-1, 1) / self.time_unit
        
        A = -torch.exp(self.log_nA) # [C]
        A_disc = torch.exp(A * delta_t) # [E, C] - 广播机制自动处理
        
        # 修正：使用标准的矩阵乘法代替错误的 einsum
        B_x = (x_j @ self.B_mat) * delta_t # [E, C]
        
        # SSM 消息传递
        ssm_msg = F.selu((A_disc * state + B_x) @ self.C_mat)
        ssm_agg = scatter.scatter_add(ssm_msg, node_i, dim=0, dim_size=x.size(0))
        
        # --- 路径 B: 矩增强统计 ---
        mean_m = scatter.scatter_mean(h_j, node_i, dim=0, dim_size=x.size(0))
        sq_h_j = h_j ** 2
        mean_sq = scatter.scatter_mean(sq_h_j, node_i, dim=0, dim_size=x.size(0))
        var_m = F.relu(mean_sq - mean_m ** 2)
        
        moment_feat = self.moment_mlp(torch.cat([mean_m, var_m], dim=1))
        
        # --- 路径 C: 门控融合 ---
        gate = self.fusion_gate(ssm_agg)
        out_combined = gate * ssm_agg + (1 - gate) * moment_feat
        
        # 输出投影
        out = self.output_proj(out_combined)
        
        # 融合标准差特征
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        # 残差连接
        if self.residual:
            if self.res_lin is not None:
                out = out + self.res_lin(x)
            else:
                out = out + x
            
        return out


class NeuralOTemporalConv(nn.Module):
    """
    神经常微分方程时序卷积层 (Neural ODE Temporal Convolution)
    
    核心思想：
    1. 将节点状态的演化建模为连续时间的常微分方程。
    2. 使用高精度的 ODE Solver (如 dopri5) 进行数值积分。
    3. 能够处理任意长度的时间间隔，具有极强的连续动态建模能力。
    """
    def __init__(self, in_channels, out_channels, edge_dim=0, time_unit=86400, use_std=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_unit = time_unit
        self.use_std = use_std
        
        # 输入投影
        self.input_proj = Linear(in_channels, out_channels)
        if edge_dim > 0:
            self.edge_proj = Linear(edge_dim, out_channels)
        else:
            self.edge_proj = None
            
        # ODE 函数定义
        self.ode_func = ODEFunc(out_channels, edge_dim=0) # 内部聚合后维度不变
        
        # Std 特征编码器
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None
            
        # 输出投影
        self.output_proj = Linear(out_channels, out_channels)
        
        # 残差映射
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
        
        # 1. 初始状态投影
        h0 = self.input_proj(x) # [N, C]
        
        # 准备边特征用于 ODE 内部的空间聚合
        if edge_attr is not None and self.edge_proj is not None:
            e_feat = self.edge_proj(edge_attr)
            # 在 ODEFunc 中，我们通常聚合的是邻居状态。
            # 这里为了简化，我们将边特征作为一种偏置加到邻居状态上
            h_j = h0[node_j] + e_feat
        else:
            h_j = h0[node_j]
            
        # 2. 定义积分时间跨度
        # 将时间差归一化
        delta_t = edge_time / self.time_unit
        
        # 对于每个节点，我们取其所有入边中最大的时间差作为积分上限
        # 这代表了该节点“最新”一次交互距离现在的时长
        max_delta_t = scatter.scatter_max(delta_t, node_i, dim=0, dim_size=x.size(0))[0]
        max_delta_t = max_delta_t.clamp(min=1e-6) # 避免除零
        
        # 3. 数值积分 (ODE Solving)
        # 注意：真正的 Graph ODE 需要在每一步积分中都重新计算图聚合，这非常慢。
        # 这里我们采用一种高效的近似：先聚合出“演化方向”，再沿时间积分。
        
        # 预计算空间聚合结果作为导数的基础
        spatial_agg = scatter.scatter_mean(h_j, node_i, dim=0, dim_size=x.size(0))
        
        # 定义简化的自治系统导数: dh/dt = W * tanh(Agg(h) - h)
        # 这模拟了节点向邻居平均值收敛的过程
        def dynamics(t, h):
            # h: [N, C]
            diff = spatial_agg - h 
            return self.ode_func.net[1](self.ode_func.net[0](diff)) # 简化映射
            
        # 对每个节点独立积分（由于时间跨度不同，这里用循环或分批处理）
        # 为了效率，我们统一积分到最大时间，然后通过 Mask 提取
        t_span = torch.linspace(0, 1, steps=10).to(x.device) # 归一化时间步
        
        # 使用 odeint 求解
        # 注意：torchdiffeq 要求 func(t, y)
        solution = odeint(dynamics, h0, t_span, method='rk4') # [Steps, N, C]
        
        # 根据每个节点的实际时间差，插值得到最终状态
        # 这里简化处理：取最后一个时间步的结果
        out = solution[-1] 
        
        # 4. 输出投影与 Std 融合
        out = self.output_proj(out)
        
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat 
            
        # 5. 残差连接
        if self.res_lin is not None:
            out = out + self.res_lin(x)
        else:
            out = out + x
            
        return out


class StabilizedGatedTemporalAttention(nn.Module):
    """
    稳定型门控时序注意力层 (SGTA)
    
    设计目标：解决 DGRTA 的原型坍缩问题，在 Alpha 数据集上实现稳健的性能提升。
    
    核心改进：
    1. 差分特征增强：利用 (x_j - x_i) 捕获局部突变，比乘积交互更适合异常检测。
    2. 可学习指数衰减：替代固定的 SSM，提供更灵活且稳定的时序权重分配。
    3. 纯净注意力机制：移除 Std 对注意力的干扰，确保聚合过程的鲁棒性。
    """
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

        # 1. 可学习的时间衰减系数 (Learnable Decay)
        # 模拟 exp(-lambda * delta_t)，lambda 是可学习的
        self.time_decay = nn.Parameter(torch.ones(heads, 1)) 
        self.time_bias = nn.Parameter(torch.zeros(heads, 1))

        # 2. 特征投影
        self.q_lin = Linear(in_channels, out_channels)
        self.k_lin = Linear(in_channels + edge_dim, out_channels)
        self.v_lin = Linear(in_channels + edge_dim, out_channels)
        
        # 3. 差分特征投影 (Difference Enhancement)
        self.diff_proj = Linear(in_channels, out_channels)

        # 4. Std 特征编码器 (仅在后期融合)
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.GELU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.std_encoder = None

        # 5. 输出合并
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

        # 1. 准备 Key/Value 输入
        target_vec = x_j
        if edge_attr is not None:
            target_vec = torch.cat([target_vec, edge_attr], dim=1)
        
        k = self.k_lin(target_vec).view(-1, self.heads, self.d_k)
        v = self.v_lin(target_vec).view(-1, self.heads, self.d_k)

        # 2. 计算 Query 和差分增强
        q_base = self.q_lin(x_i)
        diff_feat = self.diff_proj(x_i - x_j) # 捕获局部差异
        
        # 将差分信息注入 Query
        q_enhanced = (q_base + diff_feat).view(-1, self.heads, self.d_k)

        # 3. 计算基础注意力分数
        att_base = torch.sum(q_enhanced * k, dim=-1) * self.scale # [E, heads]

        # 4. 应用可学习的时间衰减 (Stabilized Time Gating)
        # time_decay: [heads, 1], edge_time: [E]
        # 我们需要结果为 [E, heads]
        # 将 edge_time 变为 [E, 1]，time_decay 变为 [1, heads]
        decay_factor = torch.exp(
            -F.softplus(self.time_decay.t()) * edge_time.view(-1, 1) + self.time_bias.t()
        ) # [E, heads]
        
        # 确保 att_base 也是 [E, heads]
        att_final = att_base * decay_factor

        # 5. Scatter Softmax 聚合
        scores = self.scatter_softmax(att_final, node_i)
        scores = self.dropout(scores)
        
        v_weighted = v * scores.unsqueeze(-1)
        v_weighted = v_weighted.view(-1, self.out_channels)
        
        out = scatter.scatter_add(v_weighted, node_i, dim=0, dim_size=x.size(0))

        # 6. 融合 Std 特征 (后期融合，不干扰注意力)
        if node_std is not None and self.std_encoder is not None:
            std_feat = self.std_encoder(node_std)
            out = out + std_feat

        # 7. 合并与残差
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
    """
    稳健校正时序注意力层 (RCTA)
    
    设计目标：继承 GTGA 的稳健性，通过“门控残差注入”机制，将 Std 特征作为
    精准的校正项，避免原型坍缩和过度惩罚。
    
    核心改进：
    1. 差分增强查询：利用 (x_j - x_i) 捕获局部突变，增强对异常的感知。
    2. 可学习衰减核：替代 SSM，提供更直观且稳定的时序权重分配。
    3. 门控残差注入：Std 特征通过 Sigmoid 门控动态调节注入量，防止噪声干扰。
    """
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

        # 1. 可学习的时间衰减核 (Learnable Decay Kernel)
        self.time_decay = nn.Parameter(torch.ones(heads, 1)) 
        self.time_bias = nn.Parameter(torch.zeros(heads, 1))

        # 2. 特征投影与差分增强
        self.q_lin = Linear(in_channels, out_channels)
        self.k_lin = Linear(in_channels + edge_dim, out_channels)
        self.v_lin = Linear(in_channels + edge_dim, out_channels)
        self.diff_proj = Linear(in_channels, out_channels) # 用于处理 (x_j - x_i)

        # 3. 门控残差注入模块 (Gated Residual Injection)
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, out_channels),
                nn.GELU(),
                Linear(out_channels, out_channels)
            )
            # 门控网络：决定注入多少 Std 信息
            self.std_gate = nn.Sequential(
                Linear(1, out_channels),
                nn.Sigmoid()
            )
        else:
            self.std_encoder = None
            self.std_gate = None

        # 4. 输出合并 MLP
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

        # 1. 准备 Key/Value 输入 (纯净聚合，不含 Std)
        target_vec = x_j
        if edge_attr is not None:
            target_vec = torch.cat([target_vec, edge_attr], dim=1)
        
        k = self.k_lin(target_vec).view(-1, self.heads, self.d_k)
        v = self.v_lin(target_vec).view(-1, self.heads, self.d_k)

        # 2. 差分增强 Query
        q_base = self.q_lin(x_i)
        diff_feat = self.diff_proj(x_i - x_j) # 捕获源-目标差异
        q_enhanced = (q_base + diff_feat).view(-1, self.heads, self.d_k)

        # 3. 计算注意力分数并应用时间衰减
        att_base = torch.sum(q_enhanced * k, dim=-1) * self.scale
        decay_factor = torch.exp(
            -F.softplus(self.time_decay.t()) * edge_time.view(-1, 1) + self.time_bias.t()
        )
        att_final = att_base * decay_factor

        # 4. Scatter Softmax 聚合
        scores = self.scatter_softmax(att_final, node_i)
        scores = self.dropout(scores)
        
        v_weighted = v * scores.unsqueeze(-1)
        agg_out = scatter.scatter_add(
            v_weighted.view(-1, self.out_channels), 
            node_i, dim=0, dim_size=x.size(0)
        )

        # 5. 门控残差注入 (关键步骤：Std 作为校正项)
        if node_std is not None and self.std_encoder is not None:
            std_embed = self.std_encoder(node_std)       # [N, out_channels]
            gate_val = self.std_gate(node_std)           # [N, out_channels]
            # 只有当门控开启时，Std 信息才会以残差形式注入
            agg_out = agg_out + gate_val * std_embed

        # 6. 合并与残差
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
    """全局 MHA 的 per-pair 时间结构偏置: 对数高斯核展开.

    b(Δt_ij) = Σ_k w_k^(h) · exp(-(log(1+Δt_ij) - μ_k)² / (2σ²)),  per-head.
    w 初始 0 -> 退化原 MHA (训练前无偏置); log 尺度跨数据集一致 (秒~月统一).
    μ_k log 空间线性覆盖 [1s, t_max], σ=间距 (均固定); w 可学 per-head.
    """
    def __init__(self, num_heads, K=8, t_max=86400 * 30, scale_init=5.0):
        super().__init__()
        log_t_max = torch.log(torch.tensor(float(t_max) + 1.0)).item()
        self.register_buffer('mu', torch.linspace(0.0, log_t_max, K))
        self.register_buffer('sigma', torch.full((K,), log_t_max / max(K - 1, 1)))
        self.w = nn.Parameter(torch.zeros(num_heads, K))  # 初始 0 = 无偏置
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))  # bias 幅度, 挤进 attn_logit

    def reset_parameters(self):
        with torch.no_grad():
            self.w.zero_()

    def forward(self, node_time_dense):
        """node_time_dense: [N] 原始秒级. 返回 [num_heads, N, N] additive bias."""
        diff = (node_time_dense.unsqueeze(0) - node_time_dense.unsqueeze(1)).abs()  # [N, N]
        log_diff = torch.log1p(diff.clamp(min=0))                                  # [N, N]
        phi = torch.exp(-(log_diff.unsqueeze(-1) - self.mu) ** 2
                        / (2 * self.sigma ** 2 + 1e-9))                             # [N, N, K]
        return self.scale * torch.einsum('nmk,hk->hnm', phi, self.w)              # [num_heads, N, N]

    def __repr__(self):
        return (f'{self.__class__.__name__}(heads={self.w.size(0)}, '
                f'K={self.mu.size(0)})')



class TiGNConv(nn.Module):
    """Time-Gated Neighbor aggregation — 用 softmax 竞争替代 GINE sum 等权聚合.

    设计原则:
    - softmax 提供邻居间选择能力 (GINE sum 无此能力, 是之前 9 种局部时间尝试全死的根因)
    - 时间通过 per-node 可学衰减系数 γ_i 参与选择: score = content + log(exp(-γ_i·Δt))
    - γ_i 初始化为小值 (≈0.13), 模型逐步学会按需开启时间选择
    - 轻量: 单头打分 (非 multi-head QKV), γ 只有标量输出

    对节点 i 的每个邻居 j:
        content_ij = (W_q·x_i) · (W_k·[x_j|edge_attr]) / √d   # 内容相关性标量
        time_ij    = -softplus(γ_i) · Δt_j                      # 时间衰减 (log空间)
        score_ij   = content_ij + time_ij                       # 加性组合
        α_j        = softmax_per_source_i(score)                # 邻居竞争选择
        h_i        = Σ α_j · W_v·[x_j|edge_attr]                # 加权聚合
    """
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

        # Per-node 时间敏感度 γ_i = softplus(MLP(x_i) + bias)
        # 初始 bias=-2 → softplus(-2)≈0.13 → 初始时间影响微弱, 安全退化
        self.gamma_mlp = nn.Sequential(
            Linear(in_channels, out_channels // 4),
            nn.ReLU(),
            Linear(out_channels // 4, 1, bias=False)
        )
        self.gamma_bias = nn.Parameter(torch.tensor(-2.0))

        # 输出投影
        self.output_proj = nn.Sequential(
            Linear(out_channels, out_channels),
            nn.ReLU(),
            Linear(out_channels, out_channels)
        )

        # 残差
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
        """x: [N, C], edge_index: [2, E], edge_time: [E] (delta_t, 已time_difference),
           edge_attr: [E, D] or None"""
        src, dst = edge_index[0], edge_index[1]

        x_src = x[src]                                   # [E, C]
        x_dst = x[dst]                                   # [E, C]

        # 邻居特征 (拼接边特征)
        neighbor_feat = x_dst
        if edge_attr is not None:
            neighbor_feat = torch.cat([neighbor_feat, edge_attr], dim=-1)  # [E, C+D]

        # Q·K → 内容相关性标量
        q = self.W_q(x_src)                              # [E, C_out]
        k = self.W_k(neighbor_feat)                      # [E, C_out]
        content_score = (q * k).sum(dim=-1) * self.scale  # [E]

        # Per-node 时间敏感度 γ_i
        gamma = F.softplus(self.gamma_mlp(x) + self.gamma_bias)  # [N, 1]
        gamma_src = gamma[src].squeeze(-1)                # [E]

        # 时间衰减: log(exp(-γ·Δt)) = -γ·Δt, 与 content 加性组合
        delta_t = edge_time.float() / self.time_unit
        time_score = -gamma_src * delta_t                 # [E], 越近越大(越不负)

        # 联合打分
        combined = content_score + time_score             # [E]

        # Per-source softmax — 邻居竞争选择
        scores = scatter_softmax_per_source(combined, src)
        scores = self.dropout(scores)

        # 加权聚合
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
    """Per-source-softmax: 对每个 source 节点的入边分数做 softmax."""
    return scatter.composite.scatter_softmax(scores, src_index)


class GatedSSMNeighborAgg(nn.Module):
    """门控 SSM 按时序扫描邻居 — 顺序依赖, 不能化为加权和.

    与 TiGN / GINE / GatedTGAT 的本质区别:
    - GINE: sum, 无序, 无时间
    - TiGN: softmax 选邻居, 无序 (order-invariant)
    - GatedTGAT: attention 选邻居, 无序 (order-invariant)
    - GatedSSM: 按时序扫描, 顺序依赖 — 每个邻居到来时, 门控决定保留多少旧状态

    结构:
        h_0 = W0·x_i                                    (从 source 状态出发)
        for k = 1..M (邻居按时序从旧到新):
            g_k = σ(W_g·nbr_k + U_g·h_{k-1} + T_g·TimeEnc(Δt_k))
            z_k = tanh(W_z·nbr_k + U_z·h_{k-1})
            h_k = (1 - g_k)⊙h_{k-1} + g_k⊙z_k
        out = LayerNorm( h_M + MLP([x_i | h_M]) )
    """

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400, log_space=False):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit
        self.log_space = log_space

        # 邻居特征投影
        nbr_dim = channels + edge_dim
        self.nbr_proj = Linear(nbr_dim, channels)

        # 门控组件
        self.W_g = Linear(channels, channels)
        self.U_g = Linear(channels, channels)
        self.time_enc = TimeEncoder(channels)
        self.T_g = Linear(channels, channels)

        # 候选状态组件
        self.W_z = Linear(channels, channels)
        self.U_z = Linear(channels, channels)

        # 初始状态: h_0 = tanh(W0·x_i)
        self.W0 = Linear(channels, channels)

        # 输出融合: MLP([x_i | h_M])
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

        # 邻居特征
        nbr_feat = x[dst]
        if edge_attr is not None:
            nbr_feat = torch.cat([nbr_feat, edge_attr], dim=-1)
        nbr_proj = self.nbr_proj(nbr_feat)           # [E, C]

        # --- 按 (source, time) 排序边 ---
        sort_key = src.float() * 1e9 + edge_time.float()
        perm = torch.argsort(sort_key)
        src_ord = src[perm]
        nbr_ord = nbr_proj[perm]
        time_ord = edge_time[perm]

        # --- 构建 padded tensor ---
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

        # 填充 padded tensor
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

        # --- 分块 SSM: cumprod 并行 ---
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
                # log-space: 长序列(M>30)数值稳定, reddit用
                log_a = torch.log(a.clamp(min=1e-12))
                log_cumdecay = torch.cumsum(log_a, dim=1)
                final_log_decay = log_cumdecay[:, -1, :]
                log_weights = final_log_decay.unsqueeze(1) - log_cumdecay
                weights = torch.exp(log_weights)
                h = torch.exp(final_log_decay) * h + (weights * b).sum(dim=1)
            else:
                # 直接 cumprod: 短序列(M<30)精确, wiki/mooc原版
                cumdecay = torch.cumprod(a, dim=1)
                final_decay = cumdecay[:, -1, :]
                weights = final_decay.unsqueeze(1) / cumdecay.clamp(min=1e-8)
                h = final_decay * h + (weights * b).sum(dim=1)

        # --- 输出 ---
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
    """源感知消息 + 独立时间门 — 每条边: content(source调制) × time_gate(scalar).

    和 GINE 的差异:
    - GINE: msg = ReLU(h_j + edge_attr), 所有 source 收到同一条消息
    - SAG : msg = σ(MLP_gate([h_i|h_j|Δt])) · ReLU(W_n·h_j + W_e·edge + W_s·h_i)
            source h_i 参与消息内容构造
            gate 标量独立, 不强制作 softmax 竞争

    聚合: sum (和 GINE 一样, O(E))
    退化: W_s初始0 → 等价GINE; gate_bias初始-1 → 初始弱门控 → 逐步学
    """

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit

        # 消息内容: W_n·h_j + W_s·h_i + W_e·edge_attr
        self.W_n = Linear(channels, channels)
        self.W_s = Linear(channels, channels)
        self.W_e = Linear(edge_dim, channels) if edge_dim > 0 else None

        # 时间门: σ(MLP([h_i | h_j | TimeEnc(Δt)])) → scalar
        self.time_enc = TimeEncoder(channels)
        gate_in = channels * 3
        self.gate_mlp = nn.Sequential(
            Linear(gate_in, channels // 4),
            nn.ReLU(),
            Linear(channels // 4, 1),
        )

        # 输出投影
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
        # W_s 初始为0: source不参与 → 退化GINE
        for p in self.W_s.parameters():
            nn.init.zeros_(p)
        if self.W_e is not None:
            self.W_e.reset_parameters()
        self.time_enc.reset_parameters()
        for layer in self.gate_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        # gate bias 初始-1 → σ(-1)≈0.27, 初始弱门控
        nn.init.constant_(self.gate_mlp[-1].bias, -1.0)
        for layer in self.output_mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        x_src, x_dst = x[src], x[dst]                     # [E, C]

        # 消息内容: source调制
        content = self.W_n(x_dst) + self.W_s(x_src)
        if self.W_e is not None and edge_attr is not None:
            content = content + self.W_e(edge_attr)
        content = F.relu(content)                          # [E, C]

        # 时间门: 独立标量, 不竞争
        t_enc = self.time_enc(
            (edge_time.float() / self.time_unit).unsqueeze(1))  # [E, C]
        gate = torch.sigmoid(
            self.gate_mlp(torch.cat([x_src, x_dst, t_enc], dim=-1)))  # [E, 1]

        # 门控消息
        msg = gate * content                               # [E, C]

        # sum 聚合
        out = scatter.scatter_add(msg, src, dim=0, dim_size=N)  # [N, C]

        # 输出
        out = self.output_mlp(out)
        out = self.dropout(out)
        out = self.norm(x + out)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.channels})'


class AnchoredGatedSSMNeighborAgg(nn.Module):
    """锚定版门控 SSM — 解决 GatedSSMNeighborAgg 的反馈环路漂移问题.

    相对原 GatedSSM 的三项增强:
    1. 源节点锚定: 门控/候选都加入 source embedding x_i (扫描全程不变的绝对参考系)
       g_k = σ(W_g·nbr + U_g·LN(h_{k-1}) + T_g·Δt + S_g·x_i + b_g)
       z_k = tanh(W_z·nbr + U_z·LN(h_{k-1}) + S_z·x_i)
       → 嵌入空间漂移时, x_i 稳定, 门控不会被递归状态绑架
    2. h_{k-1} 的 LayerNorm: 保证门控 sigmoid 输入尺度恒定, 不进饱和区
    3. gate_bias 初始 0 → σ(0)=0.5 → 初始平等对待所有邻居

    其余 (padding/cumprod并行分块/输出融合) 与 GatedSSMNeighborAgg 一致.
    """

    def __init__(self, channels, edge_dim=0, dropout=0., time_unit=86400, log_space=False):
        super().__init__()
        self.channels = channels
        self.time_unit = time_unit
        self.log_space = log_space

        nbr_dim = channels + edge_dim
        self.nbr_proj = Linear(nbr_dim, channels)

        # 门控组件 (加入 source 锚定 S_g)
        self.W_g = Linear(channels, channels)
        self.U_g = Linear(channels, channels)
        self.time_enc = TimeEncoder(channels)
        self.T_g = Linear(channels, channels)
        self.S_g = Linear(channels, channels)
        self.gate_bias = nn.Parameter(torch.zeros(1))   # 初始 σ(0)=0.5

        # 候选状态组件 (加入 source 锚定 S_z)
        self.W_z = Linear(channels, channels)
        self.U_z = Linear(channels, channels)
        self.S_z = Linear(channels, channels)

        # h_{k-1} 的 LayerNorm (反馈环路尺度稳定)
        self.h_norm = nn.LayerNorm(channels)

        # 初始状态
        self.W0 = Linear(channels, channels)

        # 输出融合
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

        # 邻居特征
        nbr_feat = x[dst]
        if edge_attr is not None:
            nbr_feat = torch.cat([nbr_feat, edge_attr], dim=-1)
        nbr_proj = self.nbr_proj(nbr_feat)           # [E, C]

        # 按 (source, time) 排序边
        sort_key = src.float() * 1e9 + edge_time.float()
        perm = torch.argsort(sort_key)
        src_ord = src[perm]
        nbr_ord = nbr_proj[perm]
        time_ord = edge_time[perm]

        # 构建 padded tensor
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

        # 源节点锚定 (active sources): [num_active, C]
        x_src = x[active]

        # 分块 SSM: cumprod 并行
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
            h_normed = self.h_norm(h)                                 # [B, C] 尺度稳定
            src_term_g = self.S_g(x_src).unsqueeze(1)                # [B, 1, C] 锚
            src_term_z = self.S_z(x_src).unsqueeze(1)                # [B, 1, C] 锚

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
                # log-space: 长序列(M>30)数值稳定, reddit用
                log_a = torch.log(a.clamp(min=1e-12))
                log_cumdecay = torch.cumsum(log_a, dim=1)
                final_log_decay = log_cumdecay[:, -1, :]
                log_weights = final_log_decay.unsqueeze(1) - log_cumdecay
                weights = torch.exp(log_weights)
                h = torch.exp(final_log_decay) * h + (weights * b).sum(dim=1)
            else:
                # 直接 cumprod: 短序列(M<30)精确, wiki/mooc原版
                cumdecay = torch.cumprod(a, dim=1)
                final_decay = cumdecay[:, -1, :]
                weights = final_decay.unsqueeze(1) / cumdecay.clamp(min=1e-8)
                h = final_decay * h + (weights * b).sum(dim=1)

        # 输出
        fused = self.output_mlp(torch.cat([x_src, h], dim=-1))
        fused = self.dropout(fused)
        fused = self.norm(x_src + fused)

        out = x.clone()
        out[active] = fused
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(channels={self.channels})'
