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
        use_std: bool = False,  # 新增参数：是否使用标准差特征
        use_single_proto: bool = False, # 新增参数：是否使用单原型
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
                use_std=use_std,  # 传递 use_std 参数
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
        #         use_std=use_std,  # 传递 use_std 参数
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
            # 新增：独立的原型头，防止梯度直接回传到共享的 out
            self.proto_head = Linear(hidden_channels, hidden_channels)
        else:
            self.std_prototypes = None
            self.proto_head = None
        
        # --- 单原型学习 (针对 Mooc/Reddit) ---
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
        # ★ 新增：std 原型初始化
        if self.use_std and self.std_prototypes is not None:
            # 步骤 1：初始化 high 原型（随机）
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            
            # 步骤 2：初始化 low 原型（随机）
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            
            # 步骤 3：Gram-Schmidt 正交化
            high = self.std_prototypes['high']  # [hidden_channels]
            low = self.std_prototypes['low']    # [hidden_channels]
            
            # 计算 low 在 high 上的投影
            projection = (low @ high) / (high @ high + 1e-8) * high
            
            # 从 low 中减去投影，得到正交分量
            low_orthogonal = low - projection
            
            # 步骤 4：归一化到相同范数
            low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
            
            # 步骤 5：更新 low 原型
            self.std_prototypes['low'].data = low_orthogonal

        # 单原型初始化
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)

    def update_global_proto(self, embeddings, mask=None):
        """使用 EMA 更新全局原型"""
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
        node_std: Optional[Tensor] = None,  # 新增参数：节点的标准差特征 [N, 1]
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
        """获取用于原型学习的独立表示"""
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
    """全局 MHA 注入 per-pair 时间结构偏置 (对数高斯核, 可学非线性, per-head).

    时间只进 attn_mask (结构偏置, 正交叠加), 不进 K/V. node_time 用原始秒级 (对数核跨尺度).
    局部 MPNN + FFN 不变. 与 TemporalGPSConv(固定线性 -Δt/τ) 并列, 由 temporal_global_kernel 切换.
    """
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
        # ① 局部 MPNN (用 x)
        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training) + x
            hs.append(self._norm(self.norm1, h, batch))
        # ② 全局 MHA + 时间结构偏置 (时间进 attn_mask, 不进 K/V)
        h_d, mask = to_dense_batch(x, batch)
        akw = dict(need_weights=False)
        if node_time is not None:
            t_d, _ = to_dense_batch(node_time, batch)
            akw['attn_mask'] = self.time_bias(t_d.squeeze(0))   # [num_heads, maxN, maxN]
        h = self.attn(h_d, h_d, h_d, **akw)[0][mask]
        h = F.dropout(h, p=self.dropout, training=self.training) + x
        hs.append(self._norm(self.norm2, h, batch))
        # ③ 融合 + FFN
        out = sum(hs)
        out = out + self.mlp(out)
        return self._norm(self.norm3, out, batch)


class StdGPSModel(nn.Module):
    """
    基于GraphGPS的时序异常检测模型
    融合std特征和原型学习，通过全局注意力增强原型对齐质量
    """
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
        # GPS特有参数
        gps_local_type: str = 'GINEConv',  # 局部MPNN类型 (GINEConv 为最终配置的甜点)
        gps_heads: int = 4,  # 注意力头数
        gps_dropout: float = 0.0,  # GPS内部的dropout
        # 全局 MHA 时间结构偏置 (对数高斯核, per-head, 可学非线性)
        temporal_global_kernel: bool = False,
        time_kernel_K: int = 8,  # 高斯核个数
        time_kernel_scale: float = 5.0,  # bias 幅度(挤进 attn_logit)
    ):
        super().__init__()
        self.dropout = dropout
        self.use_std = use_std
        self.temporal_global_kernel = temporal_global_kernel

        # === 1. 编码器 ===
        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        # std特征编码器
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
        
        # 创建局部MPNN
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
        
        # === 3. 解码器（与原模型保持一致）===
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)
        
        # === 4. 原型学习组件（与GatedTGAT完全一致）===
        if use_std:
            self.std_prototypes = nn.ParameterDict({
                'high': nn.Parameter(torch.randn(hidden_channels)),
                'low': nn.Parameter(torch.randn(hidden_channels))
            })
            # Gram-Schmidt正交化初始化
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
        """Gram-Schmidt正交化，确保high/low原型分离"""
        if self.std_prototypes is None:
            return
        high = self.std_prototypes['high']
        low = self.std_prototypes['low']
        
        # 计算low在high上的投影
        projection = (low @ high) / (high @ high + 1e-8) * high
        
        # 从low中减去投影，得到正交分量
        low_orthogonal = low - projection
        
        # 归一化到相同范数
        low_orthogonal = low_orthogonal / (low_orthogonal.norm() + 1e-8) * high.norm()
        
        # 更新low原型
        self.std_prototypes['low'].data = low_orthogonal
    
    def reset_parameters(self):
        for layer in self.gps_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.decoder.reset_parameters()
        
        # 原型初始化（与GatedTGAT一致）
        if self.use_std and self.std_prototypes is not None:
            nn.init.normal_(self.std_prototypes['high'], std=0.1)
            nn.init.normal_(self.std_prototypes['low'], std=0.1)
            self._orthogonalize_prototypes()
        
        if self.global_proto is not None:
            nn.init.normal_(self.global_proto, std=0.1)
    
    def update_global_proto(self, embeddings, mask=None):
        """EMA更新全局原型（与原模型保持一致）"""
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
        edge_time: Tensor,  # GPS不需要时间编码，但保留接口兼容性
        edge_attr: Optional[Tensor] = None,
        node_std: Optional[Tensor] = None,
    ) -> Tensor:
        """
        编码函数：融合std特征并通过GPS layers
        """
        # Step 1: 节点特征编码
        x = self.node_encoder(x)

        # Step 1b: 结构PE (RWPE) 注入输入层 —— 补 GraphGPS 第一件套, 局部+全局都吃
        # Step 2: 边特征编码
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Step 3: 融合std特征
        if node_std is not None and self.use_std and self.std_encoder is not None:
            std_embedding = self.std_encoder(node_std)
            x = x + std_embedding

        # Step 4: 通过GPS layers
        edge_kwargs = dict(edge_attr=edge_attr)
        if self.temporal_global_kernel and edge_time is not None:
            # kernel 全局时间偏置: 各节点最晚活跃时刻 (原始秒级, log1p 自带数值稳定)
            nodes = torch.cat([edge_index[0], edge_index[1]])
            t_rep = torch.cat([edge_time, edge_time]).float()
            node_time = scatter(t_rep, nodes, dim=0, reduce='max', dim_size=x.size(0))
            edge_kwargs['node_time'] = node_time
        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, batch=None, **edge_kwargs)

        return x
    
    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        """解码函数（与原模型保持一致）"""
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)
    def get_proto_embedding(self, out: Tensor) -> Tensor:
        """获取用于原型学习的独立表示"""
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
        """前向传播"""
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
    """GPS 风格混合层: 局部 TiGN (时间门控邻居选择) + 全局 MHA + 可选 kernel 时间偏置.

    与 TemporalKernelGPSConv 的核心区别:
    - 局部: TiGNConv (softmax 邻居竞争 + per-node 时间敏感度) 替代 GINE sum
    - 全局: 手动 MHA (支持 per-head 时间偏置) 替代 PyG MultiheadAttention
    - 不再依赖 PyG GPSConv 基类, 完全自主控制 forward 流程
    """
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

        # 局部 TiGN
        self.local_conv = TiGNConv(channels, channels, edge_dim=edge_dim, dropout=dropout)

        # 全局 MHA (PyG, 与原版 GPSConv 一致)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)

        # 可选 kernel 时间偏置 (全局 MHA)
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

        # ① 局部: TiGN — 时间门控邻居选择
        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = h_local + x
        h_local = self.norm1(h_local)

        if self.no_global_attn:
            # 跳过全局 MHA：仅局部 + FFN
            out = self.norm3(h_local)
        else:
            # ② 全局: MHA (PyG MultiheadAttention, 与原版一致)
            h_d = x.unsqueeze(0)
            akw = dict(need_weights=False)
            if node_time is not None and self.time_bias is not None:
                akw['attn_mask'] = self.time_bias(node_time)
            h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            h_global = self.norm2(h_global)

            # ③ 融合 + FFN
            out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)

        return out


class TiGNGPSModel(nn.Module):
    """GPS 变体: 局部聚合用 TiGN (时间门控邻居选择) 替代 GINE sum.

    与 StdGPSModel 的区别:
    - 局部: TiGNConv (softmax nbr competition + per-edge time gate + per-node γ)
      替代 GINEConv (uniform sum aggregation)
    - 全局: 手动 MHA (同 TiGNGPSConv)
    - 其余组件 (编码器/解码器/原型) 与 StdGPSModel 完全一致

    开关: --aggr gps --local_tign
    """
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

        # 编码器 (与 StdGPSModel 一致)
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

        # 解码器
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                          norm=None, dropout=dropout)

        # 原型组件 (与 StdGPSModel 一致)
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
        # 编码
        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # std 特征融合
        if node_std is not None and self.use_std and self.std_encoder is not None:
            std_embedding = self.std_encoder(node_std)
            x = x + std_embedding

        # 计算 delta_t (TiGN 需要 per-edge 时间差, 与 GatedTGAT 一致)
        edge_time_delta = time_difference(edge_time, edge_index, num_nodes=x.size(0)).squeeze()

        # 全局 kernel 用的 node_time
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
    """GPS 风格混合层: 局部 GatedSSM + 全局 MHA (PyG, 与原版 GPSConv 一致)."""

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
        # ① 局部: GatedSSM
        h_local = self.local_conv(x, edge_index, edge_time, edge_attr)
        h_local = F.dropout(h_local, p=self.dropout, training=self.training)
        h_local = self.norm1(h_local)

        if self.no_global_attn:
            # 跳过全局 MHA：仅局部 + FFN
            out = self.norm3(h_local)
        else:
            # ② 全局: MHA (PyG MultiheadAttention, 与原版 GPSConv 一致)
            h_d = x.unsqueeze(0)                                  # [1, N, C]
            akw = dict(need_weights=False)
            if node_time is not None and self.time_bias is not None:
                akw['attn_mask'] = self.time_bias(node_time)       # [heads, N, N]
            h_global = self.attn(h_d, h_d, h_d, **akw)[0].squeeze(0)  # [N, C]
            h_global = F.dropout(h_global, p=self.dropout, training=self.training)
            h_global = h_global + x
            h_global = self.norm2(h_global)

            # ③ 融合 + FFN
            out = self.norm3(h_local + h_global)
        out = out + self.ffn(out)
        return out


class SSMGPSModel(nn.Module):
    """GPS 变体: 局部聚合用 GatedSSM (顺序依赖时序扫描) 替代 GINE sum.

    与 TiGNGPSModel 的区别:
    - 局部: GatedSSMNeighborAgg (order-dependent序列扫描)
      替代 TiGNConv (order-invariant softmax 选择)
    - 其余完全一致

    开关: --aggr gps --local_ssm
    """
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
    """GPS 混合层: 局部 SrcAwareGateConv + 全局 MHA."""

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
    """GPS 变体: 局部 SrcAwareGateConv. 开关: --aggr gps --local_sag"""

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
    """GPS 风格混合层: 局部 AnchoredGatedSSM (源锚定+LN(h)+gate bias) + 全局 MHA.

    与 SSMGPSConv 的唯一区别: local_conv = AnchoredGatedSSMNeighborAgg.
    解决 SSM 反馈环路漂移 (训练中后期对比基线退化).
    """

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
#  BlockMultiGPSModel: 3×H/3 专才 (一次前向, 三对互不干扰原型)
# =====================================================================
#  共享 node_encoder/edge_encoder; 隐维 H 切 3 块(d=H/3), 每块注入一个信号(src_deg/σv/cv),
#  各跑一条 d 维 GPS 栈(GINE 或 GatedSSM)。三块互不接触 => 等价块对角 => 三对原型结构性
#  互不干扰。encode 返回拼接 h[N,H](含三特征, 对比损失用完整 h); get_proto_embeddings 切片
#  读三块。开销 ≈ 1 个满 H GPSConv(线性 3×(H/3)²=H²/3 更省, 注意力 3×B²·(H/3)=B²H 持平)。


def _make_gps_layers(channels, edge_dim, num_layers, gps_heads, gps_dropout,
                     temporal_global_kernel, time_kernel_K, time_kernel_scale,
                     no_global_attn, local_ssm, ssm_log_space):
    """建一条 GPS 栈(给定 channels)。GINE(默认) 或 GatedSSM(--local_ssm)。"""
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
    """运行一条 GPS 栈(SSM 走 edge_time_delta, GINE 走原始 edge_time)。"""
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
        assert hidden_channels % n_proto == 0, f"H={hidden_channels} 必须整除 n_proto={n_proto}"
        d = hidden_channels // n_proto
        assert d % gps_heads == 0, f"块宽 d={d} 必须整除 heads={gps_heads}"
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
        """返回拼接 h[N, H] = 三块(d=H/3) cat, 含 cv/degree/σv 三特征(各在不相交子空间)。"""
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
                xi = xi + self.std_encoders[i](descs[:, i:i + 1])   # 信号 i 注入块 i
            out_i = _run_gps_layers(self.specialist_layers[i], xi, edge_index,
                                    edge_time, e_blocks[i],
                                    self.temporal_global_kernel, self.local_ssm)
            outs.append(out_i)                            # [N, d]
        return torch.cat(outs, dim=-1)                    # [N, H]

    def get_proto_embeddings(self, h):
        """h: [B, H] -> 三块各过 proto_head[i], 返回 3 × [B, d]。"""
        d = self.d
        return [self.proto_heads[i](h[:, i * d:(i + 1) * d]) for i in range(self.n_proto)]

    def decode(self, src_emb, dst_emb=None):
        return self.decoder(src_emb * dst_emb) if dst_emb is not None else self.decoder(src_emb)

    def forward(self, x, edge_index, edge_time, edge_attr=None, descs=None):
        h = self.encode(x, edge_index, edge_time, edge_attr, descs=descs)
        return self.decode(h)


