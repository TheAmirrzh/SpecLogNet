# src/models/gcn_step_predictor.py
"""
GCN-based step predictor for SpecLogicNet baseline (Step A).

Model:
 - 2-layer GCN (torch_geometric.nn.GCNConv)
 - per-node scoring head (linear) that outputs a scalar score for each node
 - classification target: the rule-node index that was used in the next proof step
"""

from typing import Optional
import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:
    raise RuntimeError("torch_geometric is required for gcn_step_predictor. Install torch_geometric.") from e


class GCNStepPredictor(torch.nn.Module):
    def __init__(self, in_feats: int, hidden: int = 256, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.convs = torch.nn.ModuleList()
        # first conv
        self.convs.append(GCNConv(in_feats, hidden))
        # intermediate convs (if num_layers > 1)
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout
        # scoring head: projects node embedding -> scalar score
        self.score_head = torch.nn.Linear(hidden, 1)

        # Add layer normalization for better training stability
        self.layer_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layer_norms.append(torch.nn.LayerNorm(hidden))

    def forward(self, x, edge_index, batch = None):
        """
        x: node features (N, F)
        edge_index: (2, E)
        batch: optional node->graph mapping (N,) for batched graphs, not used in cls scoring here
        Returns:
          scores: (N,) per-node scalar scores (un-normalized)
          node_emb: (N, hidden) node embeddings (useful for downstream)
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.layer_norms[i](h)  # Apply layer normalization
            if i != len(self.convs) - 1:
                h = F.gelu(h)  # Changed to GELU for smoother gradients
                h = F.dropout(h, p=self.dropout, training=self.training)
        # scoring
        scores = self.score_head(h).squeeze(-1)  # shape (N,)
        return scores, h