from typing import Optional

from torch import Tensor, nn
import torch
from torch_geometric.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GPSConv
from torch_geometric.utils import scatter, to_dense_batch
import torch.nn.functional as F

from .layer import MLP, GatedTemporalGraphAttention, GatedTemporalLayer, TimeStructBias
from .layer import TGATConv, TimeEncoder, SSMTemporalGate, TiGNConv, GatedSSMNeighborAgg, AnchoredGatedSSMNeighborAgg
    
class GatedTGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
        use_std: bool = False,
        use_single_proto: bool = False,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        self.convs = nn.ModuleList([
            GatedTemporalGraphAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
                use_std=use_std,
            ) for _ in range(num_layers)
        ])
        # self.convs = nn.ModuleList([
        #     EnhancedGatedTemporalAttention(
        #         in_channels=hidden_channels,
        #         out_channels=hidden_channels,
        #         edge_dim=hidden_channels,
        #         heads=heads,
        #         dropout=dropout,
        #         residual=residual,
        #     ) for _ in range(num_layers)
        # ])

        

        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout)
        self.use_std = use_std
        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))
            })
            self.proto_head = Linear(hidden_channels, hidden_channels)
        else:
            self.std_prototypes = None
            self.proto_head = None
        
        if use_single_proto:
            self.global_proto = nn.Parameter(torch.randn(hidden_channels))
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        else:
            self.global_proto = None
            
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()
        if self.proto_head is not None:
            self.proto_head.reset_parameters()
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            
            high = self.std_prototypes['high']  # [hidden_channels]
            low = self.std_prototypes['low']    # [hidden_channels]
            
            projection = (low @ high) / (high @ high + 1e-8) * high
            
            low_orthogonal = low - projection
            
            low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
            
            self.std_prototypes['low'].data = low_orthogonal

        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        """Update the global prototypes with an exponential moving average."""
        if self.global_proto is None: return
        if mask is not None:
            if mask.sum() == 0: return
            batch_mean = embeddings[mask].mean(dim=0)
        else:
            batch_mean = embeddings.mean(dim=0)
        
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * batch_mean
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze()

        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr, node_std=node_std)
        return x

    def get_proto_embedding(self, out: Tensor) -> Tensor:
        """Return the prototype embedding of the root nodes."""
        if self.proto_head is not None:
            return self.proto_head(out)
        return out

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std)
        x = self.decode(x)
        return x


def time_difference(edge_time: Tensor, edge_index: Tensor,
                    num_nodes: Optional[int] = None) -> Tensor:
    """Get the time difference data on the edge.
    First obtain the minimum timestamp sent by each starting
      point, and then subtract
    the minimum timestamp of the starting point from the
      timestamp of each edge.
    """
    dst_node = edge_index[0]
    min_t = scatter(edge_time.view(-1), dst_node, reduce='min', dim=0,
                    dim_size=num_nodes)
    min_t = min_t[dst_node].view(-1, 1)
    delta_t = edge_time.view_as(min_t) - min_t
    return delta_t


class GatedTGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
        aggr='gcn',
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        self.convs = nn.ModuleList([
            GatedTemporalLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
                aggr=aggr,
            ) for _ in range(num_layers)
        ])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout) 
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze() 

        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr)
        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr)
        x = self.decode(x)
        return x


class TemporalKernelGPSConv(GPSConv):
    """GPS layer whose global attention carries a per-pair temporal bias."""
    def __init__(self, channels, conv, heads, dropout=0., act='relu',
                 norm='layer_norm', attn_type='multihead', time_kernel_K=8, time_kernel_scale=5.0):
        super().__init__(channels=channels, conv=conv, heads=heads,
                         dropout=dropout, act=act, norm=norm, attn_type=attn_type)
        self.time_bias = TimeStructBias(heads, K=time_kernel_K, scale_init=time_kernel_scale)

    def _norm(self, norm_layer, h, batch):
        if norm_layer is None:
            return h
        return norm_layer(h, batch=batch) if self.norm_with_batch else norm_layer(h)

    def reset_parameters(self):
        super().reset_parameters()
        self.time_bias.reset_parameters()

    def forward(self, x, edge_index, batch=None, node_time=None, **kwargs):
        hs = []
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training) + x
            hs.append(self._norm(self.norm1, h, batch))
        h_d, mask = to_dense_batch(x, batch)
        akw = dict(need_weights=False)
        if node_time is not None:
            t_d, _ = to_dense_batch(node_time, batch)
            akw['attn_mask'] = self.time_bias(t_d.squeeze(0))   # [num_heads, maxN, maxN]
        h = self.attn(h_d, h_d, h_d, **akw)[0][mask]
        h = F.dropout(h, p=self.dropout, training=self.training) + x
        hs.append(self._norm(self.norm2, h, batch))
        out = sum(hs)
        out = out + self.mlp(out)
        return self._norm(self.norm3, out, batch)


class StdGPSModel(nn.Module):
    """GraphGPS-style encoder with descriptor injection and the prototype branch."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.,
        use_std: bool = False,
        use_single_proto: bool = False,
        gps_local_type: str = 'GINEConv',
        gps_heads: int = 4,
        gps_dropout: float = 0.0,
        temporal_global_kernel: bool = False,
        time_kernel_K: int = 8,
        time_kernel_scale: float = 5.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, hidden_channels // 4),
                nn.ReLU(),
                Linear(hidden_channels // 4, hidden_channels)
            )
        else:
            self.std_encoder = None

        # === 2. GPS Layers ===
        try:
            from torch_geometric.nn import GPSConv, GINEConv
        except ImportError:
            raise ImportError("Please install torch-geometric>=2.3.0 to use GPS models")
        
        def create_local_conv(conv_type):
            if conv_type == 'GINEConv':
                nn_module = nn.Sequential(
                    Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    Linear(hidden_channels, hidden_channels)
                )
                return GINEConv(nn_module, train_eps=True)
            else:
                raise ValueError(f"Unsupported local conv type: {conv_type}")

        local_conv = create_local_conv(gps_local_type)

        if temporal_global_kernel:
            gps_cls = TemporalKernelGPSConv
        else:
            gps_cls = GPSConv
        gps_kwargs = dict(
            channels=hidden_channels,
            conv=local_conv,
            heads=gps_heads,
            dropout=gps_dropout,
            act='relu',
            norm='layer_norm',
            attn_type='multihead',
        )
        if temporal_global_kernel:
            gps_kwargs['time_kernel_K'] = time_kernel_K
            gps_kwargs['time_kernel_scale'] = time_kernel_scale
        self.gps_layers = nn.ModuleList([
            gps_cls(**gps_kwargs) for _ in range(num_layers)
        ])
        
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)
        
        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))
            })
            self.proto_head = Linear(hidden_channels, hidden_channels)
            self._orthogonalize_prototypes()
        else:
            self.std_prototypes = None
        
        if use_single_proto:
            self.global_proto = nn.Parameter(torch.randn(hidden_channels))
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        else:
            self.global_proto = None

        self.reset_parameters()
    
    def _orthogonalize_prototypes(self):
        """Orthogonalise the two prototypes with Gram-Schmidt."""
        if self.std_prototypes is None:
            return
        high = self.std_prototypes['high']
        low = self.std_prototypes['low']
        
        projection = (low @ high) / (high @ high + 1e-8) * high
        
        low_orthogonal = low - projection
        
        low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
        
        self.std_prototypes['low'].data = low_orthogonal
    
    def reset_parameters(self):
        for layer in self.gps_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.decoder.reset_parameters()
        
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            self._orthogonalize_prototypes()
        
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)
    
    def update_global_proto(self, embeddings, mask=None):
        """Update the global prototypes with an exponential moving average."""
        if self.global_proto is None:
            return
        if mask is not None:
            if mask.sum() == 0:
                return
            batch_mean = embeddings[mask].mean(dim=0)
        else:
            batch_mean = embeddings.mean(dim=0)
        
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * batch_mean
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)
    
    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch through the GPS layers with the descriptor injected."""
        x = self.node_encoder(x)

        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        if node_std is not None and self.use_std and self.std_encoder is not None:
            std_embedding = self.std_encoder(node_std)
            x = x + std_embedding

        edge_kwargs = dict(edge_attr=edge_attr)
        if self.temporal_global_kernel and edge_time is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time, edge_time]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))
            edge_kwargs['node_time'] = node_time
        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, batch=None, **edge_kwargs)

        return x
    
    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        """Decode step, kept identical to the reference model."""
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)
    def get_proto_embedding(self, out: Tensor) -> Tensor:
        """Return the prototype embedding of the root nodes."""
        if self.proto_head is not None:
            return self.proto_head(out)
        return out
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        x = self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std)
        x = self.decode(x)
        return x




    
class TGAT(nn.Module):
    def __init__( 
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        time_encoder = TimeEncoder(dimension=hidden_channels) 
        self.convs = nn.ModuleList([
            TGATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                time_encoder=time_encoder,
                time_dim=hidden_channels, 
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
            ) for _ in range(num_layers)
        ])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze() 
        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr) 

        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr)
        x = self.decode(x)
        return x







class TiGNGPSConv(nn.Module):
    """GPS layer: local TiGN aggregation, global attention, optional kernel temporal bias."""
    def __init__(self, channels, edge_dim, heads=4, dropout=0.,
                 temporal_global_kernel=False, time_kernel_K=8, time_kernel_scale=5.0,
                 no_global_attn: bool = False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.d_k = channels // heads
        self.dropout = dropout
        self.temporal_global_kernel = temporal_global_kernel
        self.no_global_attn = no_global_attn

        self.local_conv = TiGNConv(channels, channels, edge_dim=edge_dim, dropout=dropout)

        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)

        if temporal_global_kernel:
            self.time_bias = TimeStructBias(heads, K=time_kernel_K, scale_init=time_kernel_scale)
        else:
            self.time_bias = None

        # Norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        # FFN
        self.ffn = nn.Sequential(
            Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        self.scale = self.d_k ** -0.5
        self.reset_parameters()

    def reset_parameters(self):
        self.local_conv.reset_parameters()
        if self.time_bias is not None:
            self.time_bias.reset_parameters()
        for layer in self.ffn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time=None, edge_attr=None, node_time=None):
        N = x.size(0)

        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = h_local + x
        h_local = self.norm1(h_local)

        if self.no_global_attn:
            out = self.norm3(h_local)
        else:
            h_d = x.unsqueeze(0)
            akw = dict(need_weights=False)
            if node_time is not None and self.time_bias is not None:
                akw['attn_mask'] = self.time_bias(node_time)
            h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            h_global = self.norm2(h_global)

            out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)

        return out


class TiGNGPSModel(nn.Module):
    """GraphGPS encoder with TiGN local aggregation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.,
        use_std: bool = False,
        use_single_proto: bool = False,
        gps_heads: int = 4,
        gps_dropout: float = 0.0,
        temporal_global_kernel: bool = False,
        time_kernel_K: int = 8,
        time_kernel_scale: float = 5.0,
        no_global_attn: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, hidden_channels // 4),
                nn.ReLU(),
                Linear(hidden_channels // 4, hidden_channels)
            )
        else:
            self.std_encoder = None

        # TiGN-GPS layers
        self.gps_layers = nn.ModuleList([
            TiGNGPSConv(
                channels=hidden_channels,
                edge_dim=hidden_channels if edge_dim is not None else 0,
                heads=gps_heads,
                dropout=gps_dropout,
                temporal_global_kernel=temporal_global_kernel,
                time_kernel_K=time_kernel_K,
                time_kernel_scale=time_kernel_scale,
                no_global_attn=no_global_attn,
            ) for _ in range(num_layers)
        ])

        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)

        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))
            })
            self.proto_head = Linear(hidden_channels, hidden_channels)
            self._orthogonalize_prototypes()
        else:
            self.std_prototypes = None

        if use_single_proto:
            self.global_proto = nn.Parameter(torch.randn(hidden_channels))
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        else:
            self.global_proto = None

        self.reset_parameters()

    def _orthogonalize_prototypes(self):
        if self.std_prototypes is None:
            return
        high = self.std_prototypes['high']
        low = self.std_prototypes['low']
        projection = (low @ high) / (high @ high + 1e-8) * high
        low_orthogonal = low - projection
        low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
        self.std_prototypes['low'].data = low_orthogonal

    def reset_parameters(self):
        for layer in self.gps_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.decoder.reset_parameters()
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            self._orthogonalize_prototypes()
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        if self.global_proto is None:
            return
        if mask is not None:
            if mask.sum() == 0:
                return
            batch_mean = embeddings[mask].mean(dim=0)
        else:
            batch_mean = embeddings.mean(dim=0)
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * batch_mean
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        if node_std is not None and self.use_std and self.std_encoder is not None:
            std_embedding = self.std_encoder(node_std)
            x = x + std_embedding

        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()

        node_time = None
        if self.temporal_global_kernel and edge_time_delta is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time_delta, edge_time_delta]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))

        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, edge_time=edge_time_delta,
                         edge_attr=edge_attr, node_time=node_time)

        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def get_proto_embedding(self, out: Tensor) -> Tensor:
        if self.proto_head is not None:
            return self.proto_head(out)
        return out

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std)
        x = self.decode(x)
        return x


class SSMGPSConv(nn.Module):
    """GPS layer with a gated SSM local aggregator and global attention."""

    def __init__(self, channels, edge_dim, heads=4, dropout=0.,
                 temporal_global_kernel=False, time_kernel_K=8, time_kernel_scale=5.0,
                 no_global_attn: bool = False, log_space: bool = False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.temporal_global_kernel = temporal_global_kernel
        self.no_global_attn = no_global_attn

        self.local_conv = GatedSSMNeighborAgg(channels, edge_dim=edge_dim, dropout=dropout, log_space=log_space)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)

        if temporal_global_kernel:
            self.time_bias = TimeStructBias(heads, K=time_kernel_K, scale_init=time_kernel_scale)
        else:
            self.time_bias = None

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.local_conv.reset_parameters()
        if self.time_bias is not None:
            self.time_bias.reset_parameters()
        for layer in self.ffn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time=None, edge_attr=None, node_time=None):
        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm1(h_local)

        if self.no_global_attn:
            out = self.norm3(h_local)
        else:
            h_d = x.unsqueeze(0)                                  # [1, N, C]
            akw = dict(need_weights=False)
            if node_time is not None and self.time_bias is not None:
                akw['attn_mask'] = self.time_bias(node_time)       # [heads, N, N]
            h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)  # [N, C]
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            h_global = self.norm2(h_global)

            out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)
        return out


class SSMGPSModel(nn.Module):
    """GraphGPS encoder with GatedSSM local aggregation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.,
        use_std: bool = False,
        use_single_proto: bool = False,
        gps_heads: int = 4,
        gps_dropout: float = 0.0,
        temporal_global_kernel: bool = False,
        time_kernel_K: int = 8,
        time_kernel_scale: float = 5.0,
        no_global_attn: bool = False,
        log_space: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, hidden_channels // 4),
                nn.ReLU(),
                Linear(hidden_channels // 4, hidden_channels)
            )
        else:
            self.std_encoder = None

        self.gps_layers = nn.ModuleList([
            SSMGPSConv(
                channels=hidden_channels,
                edge_dim=hidden_channels if edge_dim is not None else 0,
                heads=gps_heads,
                dropout=gps_dropout,
                temporal_global_kernel=temporal_global_kernel,
                time_kernel_K=time_kernel_K,
                time_kernel_scale=time_kernel_scale,
                no_global_attn=no_global_attn,
                log_space=log_space,
            ) for _ in range(num_layers)
        ])

        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)

        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))
            })
            self.proto_head = Linear(hidden_channels, hidden_channels)
            self._orthogonalize_prototypes()
        else:
            self.std_prototypes = None

        if use_single_proto:
            self.global_proto = nn.Parameter(torch.randn(hidden_channels))
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        else:
            self.global_proto = None

        self.reset_parameters()

    def _orthogonalize_prototypes(self):
        if self.std_prototypes is None:
            return
        high = self.std_prototypes['high']
        low = self.std_prototypes['low']
        projection = (low @ high) / (high @ high + 1e-8) * high
        low_orthogonal = low - projection
        low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
        self.std_prototypes['low'].data = low_orthogonal

    def reset_parameters(self):
        for layer in self.gps_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.decoder.reset_parameters()
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            self._orthogonalize_prototypes()
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        if self.global_proto is None:
            return
        if mask is not None:
            if mask.sum() == 0:
                return
            batch_mean = embeddings[mask].mean(dim=0)
        else:
            batch_mean = embeddings.mean(dim=0)
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * batch_mean
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        if node_std is not None and self.use_std and self.std_encoder is not None:
            std_embedding = self.std_encoder(node_std)
            x = x + std_embedding

        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()

        node_time = None
        if self.temporal_global_kernel and edge_time_delta is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time_delta, edge_time_delta]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))

        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, edge_time=edge_time_delta,
                         edge_attr=edge_attr, node_time=node_time)

        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def get_proto_embedding(self, out: Tensor) -> Tensor:
        if self.proto_head is not None:
            return self.proto_head(out)
        return out

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std)
        x = self.decode(x)
        return x


class SAGGPSConv(nn.Module):
    """GPS layer with source-aware gated local aggregation."""

    def __init__(self, channels, edge_dim, heads=4, dropout=0.,
                 temporal_global_kernel=False, time_kernel_K=8, time_kernel_scale=5.0):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.temporal_global_kernel = temporal_global_kernel

        self.local_conv = SrcAwareGateConv(channels, edge_dim=edge_dim, dropout=dropout)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)

        if temporal_global_kernel:
            self.time_bias = TimeStructBias(heads, K=time_kernel_K, scale_init=time_kernel_scale)
        else:
            self.time_bias = None

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            Linear(channels, channels * 2), nn.ReLU(), nn.Dropout(dropout),
            Linear(channels * 2, channels), nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.local_conv.reset_parameters()
        if self.time_bias is not None:
            self.time_bias.reset_parameters()
        for layer in self.ffn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time=None, edge_attr=None, node_time=None):
        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm1(h_local)

        h_d = x.unsqueeze(0)
        akw = dict(need_weights=False)
        if node_time is not None and self.time_bias is not None:
            akw['attn_mask'] = self.time_bias(node_time)
        h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)
        h_global = F.dropout(h_global, p=self.dropout, training=self.training)
        h_global = h_global + x
        h_global = self.norm2(h_global)

        out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)
        return out


class SAGGPSModel(nn.Module):
    """GraphGPS encoder with source-aware gated local aggregation."""

    def __init__(self, in_channels, out_channels=1, hidden_channels=256,
                 edge_dim=None, num_layers=2, dropout=0., use_std=False,
                 use_single_proto=False, gps_heads=4, gps_dropout=0.0,
                 temporal_global_kernel=False, time_kernel_K=8, time_kernel_scale=5.0):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels) if edge_dim is not None else None
        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, hidden_channels // 4), nn.ReLU(),
                Linear(hidden_channels // 4, hidden_channels))
        else:
            self.std_encoder = None
        self.gps_layers = nn.ModuleList([
            SAGGPSConv(channels=hidden_channels,
                       edge_dim=hidden_channels if edge_dim is not None else 0,
                       heads=gps_heads, dropout=gps_dropout,
                       temporal_global_kernel=temporal_global_kernel,
                       time_kernel_K=time_kernel_K, time_kernel_scale=time_kernel_scale)
            for _ in range(num_layers)])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels], norm=None, dropout=dropout)
        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))})
            self.proto_head = Linear(hidden_channels, hidden_channels)
            self._orthogonalize_prototypes()
        else:
            self.std_prototypes = None
        self.global_proto = nn.Parameter(torch.randn(hidden_channels)) if use_single_proto else None
        if use_single_proto:
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        self.reset_parameters()

    def _orthogonalize_prototypes(self):
        if self.std_prototypes is None: return
        h = self.std_prototypes['high']; l = self.std_prototypes['low']
        p = (l @ h) / (h @ h + 1e-8) * h
        o = (l - p); o = o / (o.norm() + 1e-8) * h.norm()
        self.std_prototypes['low'].data = o

    def reset_parameters(self):
        for l in self.gps_layers:
            if hasattr(l, 'reset_parameters'): l.reset_parameters()
        self.decoder.reset_parameters()
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            self._orthogonalize_prototypes()
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        if self.global_proto is None: return
        bm = embeddings[mask].mean(dim=0) if mask is not None and mask.sum() > 0 else embeddings.mean(dim=0)
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + (1-self.proto_momentum) * bm
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        if node_std is not None and self.use_std and self.std_encoder is not None:
            x = x + self.std_encoder(node_std)
        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()
        node_time = None
        if self.temporal_global_kernel and edge_time_delta is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time_delta, edge_time_delta]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))
        for l in self.gps_layers:
            x = l(x, edge_index, edge_time=edge_time_delta, edge_attr=edge_attr, node_time=node_time)
        return x

    def decode(self, src_emb, dst_emb=None):
        return self.decoder(src_emb * dst_emb) if dst_emb is not None else self.decoder(src_emb)

    def get_proto_embedding(self, out):
        return self.proto_head(out) if self.proto_head is not None else out

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        return self.decode(self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std))


class AnchoredSSMGPSConv(nn.Module):
    """GPS layer with anchored gated SSM local aggregation."""

    def __init__(self, channels, edge_dim, heads=4, dropout=0.,
                 temporal_global_kernel=False, time_kernel_K=8, time_kernel_scale=5.0,
                 no_global_attn: bool = False):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.dropout = dropout
        self.temporal_global_kernel = temporal_global_kernel
        self.no_global_attn = no_global_attn

        self.local_conv = AnchoredGatedSSMNeighborAgg(channels, edge_dim=edge_dim, dropout=dropout)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)

        if temporal_global_kernel:
            self.time_bias = TimeStructBias(heads, K=time_kernel_K, scale_init=time_kernel_scale)
        else:
            self.time_bias = None

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.local_conv.reset_parameters()
        if self.time_bias is not None:
            self.time_bias.reset_parameters()
        for layer in self.ffn:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_time=None, edge_attr=None, node_time=None):
        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm1(h_local)

        if self.no_global_attn:
            out = self.norm3(h_local)
        else:
            h_d = x.unsqueeze(0)
            akw = dict(need_weights=False)
            if node_time is not None and self.time_bias is not None:
                akw["attn_mask"] = self.time_bias(node_time)
            h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            h_global = self.norm2(h_global)
            out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)
        return out


class AnchoredSSMGPSModel(nn.Module):
    """GPS variant: local = AnchoredGatedSSM (stable). Switch: --aggr gps --local_ssm_v2"""

    def __init__(self, in_channels, out_channels=1, hidden_channels=256,
                 edge_dim=None, num_layers=2, dropout=0., use_std=False,
                 use_single_proto=False, gps_heads=4, gps_dropout=0.0,
                 temporal_global_kernel=False, time_kernel_K=8,
                 time_kernel_scale=5.0, no_global_attn=False):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels) if edge_dim is not None else None

        if use_std:
            self.std_encoder = nn.Sequential(
                Linear(1, hidden_channels // 4),
                nn.ReLU(),
                Linear(hidden_channels // 4, hidden_channels))
        else:
            self.std_encoder = None

        self.gps_layers = nn.ModuleList([
            AnchoredSSMGPSConv(
                channels=hidden_channels,
                edge_dim=hidden_channels if edge_dim is not None else 0,
                heads=gps_heads, dropout=gps_dropout,
                temporal_global_kernel=temporal_global_kernel,
                time_kernel_K=time_kernel_K, time_kernel_scale=time_kernel_scale,
                no_global_attn=no_global_attn)
            for _ in range(num_layers)])

        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)

        if use_std:
            self.std_prototypes = nn.ParameterDict({
                "high": nn.Parameter(torch.randn(hidden_channels)),
                "low": nn.Parameter(torch.randn(hidden_channels))})
            self.proto_head = Linear(hidden_channels, hidden_channels)
            self._orthogonalize_prototypes()
        else:
            self.std_prototypes = None

        if use_single_proto:
            self.global_proto = nn.Parameter(torch.randn(hidden_channels))
            self.register_buffer("proto_momentum", torch.tensor(0.99))
        else:
            self.global_proto = None

        self.reset_parameters()

    def _orthogonalize_prototypes(self):
        if self.std_prototypes is None:
            return
        high = self.std_prototypes["high"]
        low = self.std_prototypes["low"]
        projection = (low @ high) / (high @ high + 1e-8) * high
        low_o = low - projection
        low_o = low_o / (low_o.norm() + 1e-8) * high.norm()
        self.std_prototypes["low"].data = low_o

    def reset_parameters(self):
        for layer in self.gps_layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.decoder.reset_parameters()
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes["high"], std=0.1)
            nn.init.normal_(self.std_prototypes["low"], std=0.1)
            self._orthogonalize_prototypes()
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        if self.global_proto is None:
            return
        if mask is not None and mask.sum() > 0:
            bm = embeddings[mask].mean(dim=0)
        else:
            bm = embeddings.mean(dim=0)
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * bm
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        if node_std is not None and self.use_std and self.std_encoder is not None:
            x = x + self.std_encoder(node_std)
        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()
        node_time = None
        if self.temporal_global_kernel and edge_time_delta is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time_delta, edge_time_delta]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce="max", dim_size=x.size(0))
        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, edge_time=edge_time_delta,
                         edge_attr=edge_attr, node_time=node_time)
        return x

    def decode(self, src_emb, dst_emb=None):
        return self.decoder(src_emb * dst_emb) if dst_emb is not None else self.decoder(src_emb)

    def get_proto_embedding(self, out):
        return self.proto_head(out) if self.proto_head is not None else out

    def forward(self, x, edge_index, edge_time, edge_attr=None, node_std=None):
        return self.decode(self.encode(x, edge_index, edge_time, edge_attr, node_std=node_std))


# =====================================================================
# =====================================================================


def _make_gps_layers(channels, edge_dim, num_layers, gps_heads, gps_dropout,
                     temporal_global_kernel, time_kernel_K, time_kernel_scale,
                     no_global_attn, local_ssm, ssm_log_space):
    """Build one GPS stack: GINE by default, GatedSSM when --local_ssm is set."""
    from torch_geometric.nn import GPSConv, GINEConv
    layers = []
    for _ in range(num_layers):
        if local_ssm:
            layers.append(SSMGPSConv(
                channels=channels, edge_dim=channels if edge_dim else 0,
                heads=gps_heads, dropout=gps_dropout,
                temporal_global_kernel=temporal_global_kernel,
                time_kernel_K=time_kernel_K, time_kernel_scale=time_kernel_scale,
                no_global_attn=no_global_attn, log_space=ssm_log_space))
        else:
            mlp = nn.Sequential(Linear(channels, channels), nn.ReLU(),
                                Linear(channels, channels))
            local_conv = GINEConv(mlp, train_eps=True)
            gps_cls = TemporalKernelGPSConv if temporal_global_kernel else GPSConv
            kw = dict(channels=channels, conv=local_conv, heads=gps_heads,
                      dropout=gps_dropout, act='relu', norm='layer_norm',
                      attn_type='multihead')
            if temporal_global_kernel:
                kw['time_kernel_K'] = time_kernel_K
                kw['time_kernel_scale'] = time_kernel_scale
            layers.append(gps_cls(**kw))
    return nn.ModuleList(layers)


def _run_gps_layers(gps_layers, x, edge_index, edge_time, edge_attr,
                    temporal_global_kernel, local_ssm):
    """Run one GPS stack, with time differences for the SSM and raw times for GINE."""
    if local_ssm:
        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()
        node_time = None
        if temporal_global_kernel and edge_time_delta is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time_delta, edge_time_delta]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))
        kwargs = dict(edge_attr=edge_attr, edge_time=edge_time_delta)
        if node_time is not None:
            kwargs['node_time'] = node_time
        for l in gps_layers:
            x = l(x, edge_index, **kwargs)
    else:
        kwargs = dict(edge_attr=edge_attr)
        if temporal_global_kernel and edge_time is not None:
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time, edge_time]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))
            kwargs['node_time'] = node_time
        for l in gps_layers:
            x = l(x, edge_index, **kwargs)
    return x


class BlockMultiGPSModel(nn.Module):
    def __init__(self, in_channels, out_channels=1, hidden_channels=156, edge_dim=None,
                 num_layers=2, dropout=0., use_std=False, use_single_proto=False,
                 gps_heads=4, gps_dropout=0.0, temporal_global_kernel=False,
                 time_kernel_K=8, time_kernel_scale=5.0, no_global_attn=False,
                 local_ssm=False, ssm_log_space=False, n_proto=3):
        super().__init__()
        assert hidden_channels % n_proto == 0, f"H={hidden_channels} is not divisible by n_proto={n_proto}"
        d = hidden_channels // n_proto
        assert d % gps_heads == 0, f"d={d} is not divisible by heads={gps_heads}"
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel
        self.local_ssm = local_ssm
        self.n_proto = n_proto
        self.d = d
        self.hidden_channels = hidden_channels

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels) if edge_dim is not None else None
        if use_std:
            self.std_encoders = nn.ModuleList([
                nn.Sequential(Linear(1, d), nn.ReLU(), Linear(d, d))
                for _ in range(n_proto)])
        else:
            self.std_encoders = None
        self.specialist_layers = nn.ModuleList([
            _make_gps_layers(d, edge_dim, num_layers, gps_heads, gps_dropout,
                             temporal_global_kernel, time_kernel_K, time_kernel_scale,
                             no_global_attn, local_ssm, ssm_log_space)
            for _ in range(n_proto)])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout)
        if use_std:
            self.proto_heads = nn.ModuleList([Linear(d, d) for _ in range(n_proto)])
            self.std_prototypes = nn.ParameterDict()
            for i in range(n_proto):
                self.std_prototypes[f'high{i}'] = nn.Parameter(torch.randn(d))
                self.std_prototypes[f'low{i}'] = nn.Parameter(torch.randn(d))
            self._orthogonalize_prototypes()
        else:
            self.proto_heads = None
            self.std_prototypes = None
        self.global_proto = nn.Parameter(torch.randn(hidden_channels)) if use_single_proto else None
        if use_single_proto:
            self.register_buffer('proto_momentum', torch.tensor(0.99))
        self.reset_parameters()

    def _orthogonalize_prototypes(self):
        for i in range(self.n_proto):
            high = self.std_prototypes[f'high{i}']
            low = self.std_prototypes[f'low{i}']
            proj = (low @ high) / (high @ high + 1e-8) * high
            lo = low - proj
            lo = lo / (lo.norm() + 1e-8) * high.norm()
            self.std_prototypes[f'low{i}'].data = lo

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        if self.edge_encoder is not None:
            self.edge_encoder.reset_parameters()
        if self.std_encoders is not None:
            for enc in self.std_encoders:
                for l in enc:
                    if hasattr(l, 'reset_parameters'):
                        l.reset_parameters()
        for spec in self.specialist_layers:
            for l in spec:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters()
        self.decoder.reset_parameters()
        if self.proto_heads is not None:
            for h in self.proto_heads:
                h.reset_parameters()
        if self.std_prototypes is not None:
            for i in range(self.n_proto):
                nn.init.normal_(self.std_prototypes[f'high{i}'], std=0.1)
                nn.init.normal_(self.std_prototypes[f'low{i}'], std=0.1)
            self._orthogonalize_prototypes()
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        if self.global_proto is None:
            return
        bm = embeddings[mask].mean(0) if mask is not None and mask.sum() > 0 else embeddings.mean(0)
        with torch.no_grad():
            self.global_proto.data = self.proto_momentum * self.global_proto.data + \
                                     (1 - self.proto_momentum) * bm
            self.global_proto.data = F.normalize(self.global_proto.data, dim=0)

    def encode(self, x, edge_index, edge_time, edge_attr=None, descs=None):
        """Concatenate the three descriptor blocks into the encoder input."""
        base = self.node_encoder(x)                       # [N, H]
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)      # [E, H]
        blocks = list(base.chunk(self.n_proto, dim=-1))   # 3 × [N, d]
        if edge_attr is not None:
            e_blocks = list(edge_attr.chunk(self.n_proto, dim=-1))
        else:
            e_blocks = [None] * self.n_proto
        outs = []
        for i in range(self.n_proto):
            xi = blocks[i]
            if descs is not None and self.use_std and self.std_encoders is not None:
                xi = xi + self.std_encoders[i](descs[:, i:i + 1])
            out_i = _run_gps_layers(self.specialist_layers[i], xi, edge_index,
                                    edge_time, e_blocks[i],
                                    self.temporal_global_kernel, self.local_ssm)
            outs.append(out_i)                            # [N, d]
        return torch.cat(outs, dim=-1)                    # [N, H]

    def get_proto_embeddings(self, h):
        """Project the root representation with each prototype head."""
        d = self.d
        return [self.proto_heads[i](h[:, i * d:(i + 1) * d]) for i in range(self.n_proto)]

    def decode(self, src_emb, dst_emb=None):
        return self.decoder(src_emb * dst_emb) if dst_emb is not None else self.decoder(src_emb)

    def forward(self, x, edge_index, edge_time, edge_attr=None, descs=None):
        h = self.encode(x, edge_index, edge_time, edge_attr, descs=descs)
        return self.decode(h)


