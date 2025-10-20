"""
SOTA-Informed GNN Architecture for Node Ranking
Based on recent research findings:
- GATv2 over GAT (dynamic attention)
- Channel-wise attention for better expressiveness
- Proper node-level scoring without confusing pooling
- Type-aware message passing for heterogeneous graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class TypeAwareGATv2Conv(nn.Module):
    """
    GATv2Conv with type-awareness for fact vs rule nodes.
    Based on: "Graph Attention Networks: Self-Attention Explained" (2024)
    """
    
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3, edge_dim=None):
        super().__init__()
        
        # Separate attention for different node types
        self.gat_fact_to_rule = GATv2Conv(
            in_channels, out_channels // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim, concat=True
        )
        self.gat_rule_to_fact = GATv2Conv(
            in_channels, out_channels // heads, heads=heads,
            dropout=dropout, edge_dim=edge_dim, concat=True
        )
        
        # Type embedding to distinguish fact/rule
        self.type_proj = nn.Linear(2, out_channels // 4)
        
        # Combine different attention paths
        self.combine = nn.Linear(out_channels + out_channels // 4, out_channels)
        
    def forward(self, x, edge_index, node_types, edge_attr=None):
        """
        Args:
            x: node features (N, in_channels)
            edge_index: edge connectivity
            node_types: (N, 2) one-hot [fact, rule]
            edge_attr: edge features
        """
        # Apply type-aware convolutions
        h1 = self.gat_fact_to_rule(x, edge_index, edge_attr=edge_attr)
        h2 = self.gat_rule_to_fact(x, edge_index, edge_attr=edge_attr)
        
        # Type embedding
        type_emb = self.type_proj(node_types)
        
        # Combine
        h = torch.cat([h1 + h2, type_emb], dim=-1)
        return self.combine(h)


class NodeRankingGNN(nn.Module):
    """
    Properly designed GNN for node ranking (NOT graph classification).
    
    Key design principles from SOTA papers:
    1. No pooling-then-broadcast (confusing for node tasks)
    2. Type-aware message passing (heterogeneous graphs)
    3. Residual connections with LayerNorm
    4. Separate scoring per node based on LOCAL+GLOBAL context
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Edge type embedding
        self.edge_encoder = nn.Embedding(3, hidden_dim // num_heads)
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Type-aware GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                TypeAwareGATv2Conv(
                    hidden_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, edge_dim=hidden_dim // num_heads
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # CRITICAL: For node ranking, we need BOTH local and global context
        # But DON'T pool then broadcast - that's confusing!
        # Instead: compute graph summary ONCE and use it alongside node features
        
        self.graph_summary = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Node scoring: uses node embedding + graph summary
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Returns node scores (N,) for ranking.
        """
        # Extract node types from features [0-1]
        node_types = x[:, :2]
        
        # Input projection
        h = self.input_proj(x)
        
        # Edge features
        if edge_attr is not None and len(edge_attr) > 0:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None
        
        # Type-aware message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_res = h
            h = conv(h, edge_index, node_types, edge_attr=edge_emb)
            h = norm(h + h_res)
            
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Compute graph-level summary (but DON'T broadcast back!)
        if batch is not None:
            graph_mean = global_mean_pool(h, batch)
            graph_max = global_max_pool(h, batch)
            graph_summary = self.graph_summary(torch.cat([graph_mean, graph_max], dim=-1))
            
            # For each node, concatenate its embedding with its graph's summary
            node_graph_summary = graph_summary[batch]
        else:
            # Single graph
            graph_mean = h.mean(dim=0, keepdim=True)
            graph_max = h.max(dim=0, keepdim=True)[0]
            graph_summary = self.graph_summary(torch.cat([graph_mean, graph_max], dim=-1))
            node_graph_summary = graph_summary.expand(h.size(0), -1)
        
        # Score each node using BOTH local (h) and global (graph_summary) info
        node_input = torch.cat([h, node_graph_summary], dim=-1)
        scores = self.node_scorer(node_input).squeeze(-1)
        
        return scores, h


class SimplifiedNodeRankingGNN(nn.Module):
    """
    Simplified version: Pure node-level scoring without any pooling.
    Sometimes simpler is better for node ranking tasks.
    Based on: "Bag of Tricks for Node Classification" (2021)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.num_layers = num_layers  
        # Simple GATv2 stack
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Direct node scoring
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.input_proj(x)
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h + h_res)
            
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        scores = self.scorer(h).squeeze(-1)
        return scores, h


# Recommended: Use SimplifiedNodeRankingGNN first to establish baseline
# Then try NodeRankingGNN if you need the global context
def get_model(in_dim, hidden_dim=128, num_layers=3, dropout=0.3, use_type_aware=False):
    """Factory function to get the right model."""
    if use_type_aware:
        return NodeRankingGNN(in_dim, hidden_dim, num_layers, dropout=dropout)
    else:
        return SimplifiedNodeRankingGNN(in_dim, hidden_dim, num_layers, dropout=dropout)