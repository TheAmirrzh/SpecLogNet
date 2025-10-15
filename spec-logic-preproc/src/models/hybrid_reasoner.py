# src/models/hybrid_reasoner.py
"""
Hybrid Spectral-Spatial Graph Neural Network for Theorem Proving

Architecture Components:
1. SpectralEncoder: Processes eigenvalue/eigenvector features
2. SpatialEncoder: Standard GNN message passing (GCN/GAT/GIN)
3. SpectralSpatialFusion: Attention-based feature fusion
4. MultiStepDecoder: Predicts next proof step(s)

Key Innovation: Combines global spectral patterns with local spatial structure
for improved logical reasoning performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

try:
    from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
    from torch_geometric.utils import to_dense_batch
except ImportError:
    raise RuntimeError("torch_geometric required. Install: pip install torch_geometric")


# 1. SPECTRAL ENCODER: Processes global graph structure

class SpectralEncoder(nn.Module):
    """
    Encodes spectral features (eigenvectors, eigenvalues) into node representations.
    
    Design choices:
    - Uses learnable projection of eigenvectors
    - Adds positional encoding based on eigenvalue magnitudes
    - Optional spectral convolution in frequency domain
    """
    
    def __init__(self, spectral_dim: int, hidden_dim: int, num_filters: int = 16):
        super().__init__()
        self.spectral_dim = spectral_dim
        self.hidden_dim = hidden_dim
        
        # Project raw eigenvectors to hidden space
        self.eigvec_proj = nn.Linear(spectral_dim, hidden_dim)
        
        # Learnable spectral filters (graph signal processing)
        self.spectral_filters = nn.Parameter(torch.randn(num_filters, spectral_dim))
        self.filter_proj = nn.Linear(num_filters, hidden_dim)
        
        # Positional encoding from eigenvalue spectrum
        self.pos_encoding = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, eigvecs: torch.Tensor, eigvals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            eigvecs: (N, k) node eigenvector features
            eigvals: (k,) eigenvalues (optional)
        Returns:
            spectral_feats: (N, hidden_dim)
        """
        # Basic projection
        h_eigvec = self.eigvec_proj(eigvecs)  # (N, hidden)
        
        # Spectral filtering: X' = X @ diag(learnable_filters)
        # Simulates graph convolution in frequency domain
        filtered = torch.matmul(eigvecs.unsqueeze(1), 
                                self.spectral_filters.T.unsqueeze(0))  # (N, num_filters, k)
        filtered = filtered.mean(dim=-1)  # (N, num_filters)
        h_filter = self.filter_proj(filtered)  # (N, hidden)
        
        # Positional encoding
        if eigvals is not None:
            # Broadcast eigenvalues to all nodes
            eigval_feat = eigvals.unsqueeze(0).expand(eigvecs.size(0), -1)
            h_pos = self.pos_encoding(eigval_feat)
        else:
            h_pos = 0
        
        # Combine
        spectral_feats = self.norm(h_eigvec + h_filter + h_pos)
        return spectral_feats


# 2. SPATIAL ENCODER: Local message passing

class SpatialEncoder(nn.Module):
    """
    Standard spatial GNN with multiple layers.
    Supports GCN, GAT, or GIN architectures.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, 
                 gnn_type: str = "gcn", dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Build GNN layers
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            
            if gnn_type == "gcn":
                conv = GCNConv(in_channels, hidden_dim)
            elif gnn_type == "gat":
                conv = GATConv(in_channels, hidden_dim // num_heads, 
                              heads=num_heads, concat=True, dropout=dropout)
            elif gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINConv(mlp, train_eps=True)
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
        Returns:
            spatial_feats: (N, hidden_dim)
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h

# 3. SPECTRAL-SPATIAL FUSION: Combines both modalities

class SpectralSpatialFusion(nn.Module):
    """
    Fuses spectral and spatial features using:
    - Cross-attention mechanism
    - Gating mechanism
    - Residual connections
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-attention: spatial queries spectral
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, spatial_feats: torch.Tensor, spectral_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_feats: (N, hidden_dim)
            spectral_feats: (N, hidden_dim)
        Returns:
            fused_feats: (N, hidden_dim)
        """
        # Cross-attention: use spatial as query, spectral as key/value
        attn_out, _ = self.cross_attn(
            spatial_feats.unsqueeze(0),  # (1, N, hidden)
            spectral_feats.unsqueeze(0),
            spectral_feats.unsqueeze(0)
        )
        attn_out = attn_out.squeeze(0)  # (N, hidden)
        
        # Gating: decide how much to use each modality
        concat_feats = torch.cat([spatial_feats, spectral_feats], dim=-1)
        gate_weights = self.gate(concat_feats)  # (N, hidden)
        
        gated_spatial = spatial_feats * gate_weights
        gated_spectral = spectral_feats * (1 - gate_weights)
        
        # Combine everything
        combined = torch.cat([gated_spatial + attn_out, gated_spectral], dim=-1)
        fused_feats = self.out_proj(combined)
        
        return fused_feats


# 4. MULTI-STEP DECODER: Predicts next proof step(s)


class MultiStepDecoder(nn.Module):
    """
    Decodes fused node representations to predict:
    - Next rule to apply (node classification)
    - Optionally: k-step lookahead
    """
    
    def __init__(self, hidden_dim: int, num_lookahead: int = 1, use_graph_pooling: bool = True):
        super().__init__()
        self.num_lookahead = num_lookahead
        self.use_graph_pooling = use_graph_pooling
        
        # Graph-level context (optional)
        if use_graph_pooling:
            self.graph_context = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        
        # Per-node scoring heads (one per lookahead step)
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 if use_graph_pooling else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_lookahead)
        ])
    
    def forward(self, node_feats: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            node_feats: (N, hidden_dim)
            batch: (N,) batch assignment for pooling
        Returns:
            scores_list: List of (N,) score tensors, one per lookahead step
        """
        # Compute graph-level context
        if self.use_graph_pooling and batch is not None:
            graph_context = global_mean_pool(node_feats, batch)  # (B, hidden)
            # Broadcast back to nodes
            node_graph_context = graph_context[batch]  # (N, hidden)
            decoder_input = torch.cat([node_feats, node_graph_context], dim=-1)
        else:
            decoder_input = node_feats
        
        # Predict scores for each lookahead step
        scores_list = []
        for head in self.score_heads:
            scores = head(decoder_input).squeeze(-1)  # (N,)
            scores_list.append(scores)
        
        return scores_list


# 5. COMPLETE HYBRID MODEL

class HybridSpectralSpatialReasoner(nn.Module):
    """
    Complete hybrid architecture combining spectral and spatial GNNs
    for automated theorem proving.
    
    Forward pass:
    1. Encode spatial features via GNN
    2. Encode spectral features from eigenvectors
    3. Fuse both modalities with attention
    4. Decode to predict next proof step(s)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        spectral_dim: int = 16,
        num_spatial_layers: int = 3,
        gnn_type: str = "gcn",
        num_heads: int = 4,
        num_lookahead: int = 1,
        dropout: float = 0.1,
        use_spectral: bool = True
    ):
        super().__init__()
        
        self.use_spectral = use_spectral
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            hidden_dim, hidden_dim, num_spatial_layers, gnn_type, dropout, num_heads
        )
        
        # Spectral encoder (optional)
        if use_spectral:
            self.spectral_encoder = SpectralEncoder(spectral_dim, hidden_dim)
            self.fusion = SpectralSpatialFusion(hidden_dim, num_heads)
        
        # Decoder
        self.decoder = MultiStepDecoder(hidden_dim, num_lookahead, use_graph_pooling=True)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        eigvecs: Optional[torch.Tensor] = None,
        eigvals: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices
            eigvecs: (N, k) eigenvector features (optional)
            eigvals: (k,) eigenvalues (optional)
            batch: (N,) batch assignment
        
        Returns:
            scores_list: List of (N,) score tensors per lookahead step
            fused_feats: (N, hidden_dim) final node representations
        """
        # Input projection
        h = self.input_proj(x)
        
        # Spatial encoding
        spatial_feats = self.spatial_encoder(h, edge_index)
        
        # Spectral encoding + fusion
        if self.use_spectral and eigvecs is not None:
            spectral_feats = self.spectral_encoder(eigvecs, eigvals)
            fused_feats = self.fusion(spatial_feats, spectral_feats)
        else:
            fused_feats = spatial_feats
        
        # Decode to scores
        scores_list = self.decoder(fused_feats, batch)
        
        return scores_list, fused_feats


# 6. LOSS FUNCTIONS

class HybridLoss(nn.Module):
    """
    Combined loss for hybrid reasoner:
    - CrossEntropy for step prediction
    - Optional contrastive loss for representation learning
    - Optional regularization on attention weights
    """
    
    def __init__(self, use_contrastive: bool = False, alpha: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.use_contrastive = use_contrastive
        self.alpha = alpha
    
    def forward(self, scores: torch.Tensor, target: torch.Tensor, 
                node_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            scores: (1, N) or (N,) predicted scores
            target: (1,) target node index
            node_feats: (N, hidden) for contrastive loss
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        
        ce_loss = self.ce(scores, target)
        
        # Optional: contrastive loss to separate positive/negative nodes
        if self.use_contrastive and node_feats is not None:
            # Simple contrastive: push target closer to graph center
            target_feat = node_feats[target]
            mean_feat = node_feats.mean(dim=0, keepdim=True)
            contrastive_loss = F.mse_loss(target_feat, mean_feat)
            return ce_loss + self.alpha * contrastive_loss
        
        return ce_loss


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example dimensions
    num_nodes = 50
    in_dim = 32  # type_onehot + derived_flag + other features
    hidden_dim = 128
    spectral_dim = 16
    
    # Create model
    model = HybridSpectralSpatialReasoner(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        spectral_dim=spectral_dim,
        num_spatial_layers=3,
        gnn_type="gcn",
        num_heads=4,
        num_lookahead=1,
        use_spectral=True
    )
    
    # Dummy data
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    eigvecs = torch.randn(num_nodes, spectral_dim)
    eigvals = torch.randn(spectral_dim)
    
    # Forward pass
    scores_list, node_feats = model(x, edge_index, eigvecs, eigvals)
    
    print(f"Model output:")
    print(f"  Scores shape: {scores_list[0].shape}")  # (N,)
    print(f"  Node features shape: {node_feats.shape}")  # (N, hidden)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")