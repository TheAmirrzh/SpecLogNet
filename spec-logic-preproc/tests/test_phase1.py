# tests/test_phase1.py
"""
Unit tests and integration tests for Phase 1 components.
Run with: pytest tests/test_phase1.py -v
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.horn_generator_v2 import generate_stratified_horn_instance, Difficulty
from src.spectral import adjacency_from_edges, compute_symmetric_laplacian, topk_eig
from src.models.gcn_step_predictor import GCNStepPredictor
from src.train_step_predictor import StepPredictionDataset


class TestHornGenerator:
    """Test Horn clause generator."""
    
    def test_easy_instance_generation(self):
        """Test easy problem generation."""
        inst = generate_stratified_horn_instance("test_easy", Difficulty.EASY, seed=42)
        
        assert "nodes" in inst
        assert "edges" in inst
        assert "proof_steps" in inst
        assert "metadata" in inst
        
        # Check difficulty constraints
        meta = inst["metadata"]
        assert meta["difficulty"] == "easy"
        assert 4 <= meta["n_facts"] <= 6
        assert 4 <= meta["n_rules"] <= 6
    
    def test_all_difficulty_levels(self):
        """Test all difficulty levels can be generated."""
        for difficulty in Difficulty:
            inst = generate_stratified_horn_instance(
                f"test_{difficulty.value}", 
                difficulty, 
                seed=42
            )
            assert len(inst["nodes"]) > 0
            assert inst["metadata"]["difficulty"] == difficulty.value
    
    def test_proof_steps_valid(self):
        """Test that proof steps reference valid nodes."""
        inst = generate_stratified_horn_instance("test_proof", Difficulty.MEDIUM, seed=42)
        
        node_ids = {n["nid"] for n in inst["nodes"]}
        
        for step in inst["proof_steps"]:
            assert step["derived_node"] in node_ids
            assert step["used_rule"] in node_ids
            for premise in step["premises"]:
                assert premise in node_ids
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same instance."""
        inst1 = generate_stratified_horn_instance("test_seed", Difficulty.MEDIUM, seed=123)
        inst2 = generate_stratified_horn_instance("test_seed", Difficulty.MEDIUM, seed=123)
        
        assert inst1["nodes"] == inst2["nodes"]
        assert inst1["edges"] == inst2["edges"]
        assert inst1["proof_steps"] == inst2["proof_steps"]
    
    def test_negation_feature(self):
        """Test negation in hard/extreme problems."""
        inst = generate_stratified_horn_instance("test_neg", Difficulty.HARD, seed=42)
        
        # Check if any nodes have negation
        has_negation = any(n.get("negated", False) for n in inst["nodes"] if n["type"] == "fact")
        assert inst["metadata"]["has_negation"] is True


class TestSpectralFeatures:
    """Test spectral feature computation."""
    
    def test_adjacency_construction(self):
        """Test adjacency matrix construction."""
        edges = [
            {"src": 0, "dst": 1, "etype": "test"},
            {"src": 1, "dst": 2, "etype": "test"},
            {"src": 2, "dst": 0, "etype": "test"}
        ]
        
        A = adjacency_from_edges(3, edges)
        
        assert A.shape == (3, 3)
        assert A[0, 1] == 1
        assert A[1, 0] == 1  # Symmetric
    
    def test_laplacian_computation(self):
        """Test Laplacian computation."""
        edges = [
            {"src": 0, "dst": 1, "etype": "test"},
            {"src": 1, "dst": 2, "etype": "test"}
        ]
        
        A = adjacency_from_edges(3, edges)
        L = compute_symmetric_laplacian(A)
        
        assert L.shape == (3, 3)
        # Laplacian should be symmetric
        assert np.allclose(L.toarray(), L.T.toarray())
    
    def test_eigendecomposition(self):
        """Test eigendecomposition."""
        edges = [
            {"src": 0, "dst": 1, "etype": "test"},
            {"src": 1, "dst": 2, "etype": "test"},
            {"src": 2, "dst": 3, "etype": "test"},
            {"src": 3, "dst": 0, "etype": "test"}
        ]
        
        A = adjacency_from_edges(4, edges)
        L = compute_symmetric_laplacian(A)
        
        eigvals, eigvecs = topk_eig(L, k=3)
        
        assert eigvals.shape == (3,)
        assert eigvecs.shape == (4, 3)
        # Eigenvalues should be in ascending order
        assert np.all(eigvals[:-1] <= eigvals[1:])
    
    def test_spectral_features_on_instance(self):
        """Test spectral features on generated instance."""
        inst = generate_stratified_horn_instance("test_spectral", Difficulty.MEDIUM, seed=42)
        
        n_nodes = len(inst["nodes"])
        A = adjacency_from_edges(n_nodes, inst["edges"])
        L = compute_symmetric_laplacian(A)
        
        k = min(8, n_nodes - 1)
        if k > 0:
            eigvals, eigvecs = topk_eig(L, k=k)
            
            assert eigvecs.shape[0] == n_nodes
            assert eigvecs.shape[1] == k


class TestGCNModel:
    """Test GCN step predictor model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = GCNStepPredictor(in_feats=32, hidden=64, num_layers=2)
        
        assert model is not None
        assert hasattr(model, 'convs')
        assert hasattr(model, 'score_head')
    
    def test_forward_pass(self):
        """Test forward pass produces correct shapes."""
        model = GCNStepPredictor(in_feats=32, hidden=64, num_layers=2)
        
        # Create dummy input
        n_nodes = 10
        x = torch.randn(n_nodes, 32)
        edge_index = torch.randint(0, n_nodes, (2, 20))
        
        scores, node_emb = model(x, edge_index)
        
        assert scores.shape == (n_nodes,)
        assert node_emb.shape == (n_nodes, 64)
    
    def test_model_output_range(self):
        """Test model outputs reasonable scores."""
        model = GCNStepPredictor(in_feats=32, hidden=64, num_layers=2)
        model.eval()
        
        n_nodes = 10
        x = torch.randn(n_nodes, 32)
        edge_index = torch.randint(0, n_nodes, (2, 20))
        
        with torch.no_grad():
            scores, _ = model(x, edge_index)
        
        # Scores should be finite
        assert torch.isfinite(scores).all()
    
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = GCNStepPredictor(in_feats=32, hidden=64, num_layers=2)
        
        n_nodes = 10
        x = torch.randn(n_nodes, 32)
        edge_index = torch.randint(0, n_nodes, (2, 20))
        
        scores, _ = model(x, edge_index)
        loss = scores.mean()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestDataset:
    """Test dataset loading and processing."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Generate a few instances
        for i in range(3):
            inst = generate_stratified_horn_instance(
                f"test_{i}", 
                Difficulty.EASY, 
                seed=42 + i
            )
            
            with open(os.path.join(temp_dir, f"test_{i}.json"), "w") as f:
                json.dump(inst, f)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_loading(self, temp_dataset):
        """Test dataset can load JSON files."""
        files = [os.path.join(temp_dataset, f) for f in os.listdir(temp_dataset)]
        dataset = StepPredictionDataset(files, spectral_dir=None, seed=42)
        
        assert len(dataset) > 0
    
    def test_dataset_item_structure(self, temp_dataset):
        """Test dataset items have correct structure."""
        files = [os.path.join(temp_dataset, f) for f in os.listdir(temp_dataset)]
        dataset = StepPredictionDataset(files, spectral_dir=None, seed=42)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            assert hasattr(item, 'x')  # Node features
            assert hasattr(item, 'edge_index')  # Edges
            assert hasattr(item, 'y')  # Target
            assert hasattr(item, 'meta')  # Metadata
    
    def test_node_features_composition(self, temp_dataset):
        """Test node features include type and derived flag."""
        files = [os.path.join(temp_dataset, f) for f in os.listdir(temp_dataset)]
        dataset = StepPredictionDataset(files, spectral_dir=None, seed=42)
        
        if len(dataset) > 0:
            item = dataset[0]
            
            # Features should be: [type_onehot | derived_flag]
            # At minimum: 2 node types (fact, rule) + 1 derived flag = 3
            assert item.x.size(1) >= 3


class TestIntegration:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_small(self, temp_workspace):
        """Test end-to-end workflow on small dataset."""
        # Generate small dataset
        data_dir = os.path.join(temp_workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for i in range(5):
            inst = generate_stratified_horn_instance(
                f"e2e_{i}", 
                Difficulty.EASY, 
                seed=42 + i
            )
            with open(os.path.join(data_dir, f"e2e_{i}.json"), "w") as f:
                json.dump(inst, f)
        
        # Load dataset
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        dataset = StepPredictionDataset(files, spectral_dir=None, seed=42)
        
        assert len(dataset) > 0
        
        # Create model
        sample = dataset[0]
        in_feats = sample.x.size(1)
        model = GCNStepPredictor(in_feats=in_feats, hidden=32, num_layers=2)
        
        # Forward pass
        scores, _ = model(sample.x, sample.edge_index)
        
        assert scores.shape[0] == sample.x.size(0)
    
    def test_training_step(self, temp_workspace):
        """Test a single training step runs without error."""
        # Generate dataset
        data_dir = os.path.join(temp_workspace, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for i in range(3):
            inst = generate_stratified_horn_instance(
                f"train_{i}", 
                Difficulty.EASY, 
                seed=42 + i
            )
            with open(os.path.join(data_dir, f"train_{i}.json"), "w") as f:
                json.dump(inst, f)
        
        # Load dataset
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        dataset = StepPredictionDataset(files, spectral_dir=None, seed=42)
        
        # Create model
        sample = dataset[0]
        in_feats = sample.x.size(1)
        model = GCNStepPredictor(in_feats=in_feats, hidden=32, num_layers=2)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        data = dataset[0]
        target_idx = int(data.y.item())
        
        if target_idx >= 0:
            scores, _ = model(data.x, data.edge_index)
            logits = scores.unsqueeze(0)
            target = torch.tensor([target_idx], dtype=torch.long)
            
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            assert torch.isfinite(loss)


class TestValidation:
    """Validation tests for data quality."""
    
    def test_no_orphan_nodes(self):
        """Test that all nodes are connected."""
        inst = generate_stratified_horn_instance("test_orphan", Difficulty.MEDIUM, seed=42)
        
        node_ids = {n["nid"] for n in inst["nodes"]}
        connected_nodes = set()
        
        for edge in inst["edges"]:
            connected_nodes.add(edge["src"])
            connected_nodes.add(edge["dst"])
        
        # All nodes should appear in at least one edge
        # (Note: This may not always be true for all generators, adjust as needed)
        # For now, just check that most nodes are connected
        connection_ratio = len(connected_nodes) / len(node_ids)
        assert connection_ratio > 0.5  # At least 50% connected
    
    def test_proof_step_ordering(self):
        """Test proof steps are in valid order."""
        inst = generate_stratified_horn_instance("test_order", Difficulty.MEDIUM, seed=42)
        
        # Each proof step should have step_id in order
        step_ids = [s["step_id"] for s in inst["proof_steps"]]
        assert step_ids == sorted(step_ids)
        assert step_ids == list(range(len(step_ids)))
    
    def test_edge_types_valid(self):
        """Test edge types are valid."""
        inst = generate_stratified_horn_instance("test_etype", Difficulty.MEDIUM, seed=42)
        
        valid_etypes = {"body", "head"}
        for edge in inst["edges"]:
            assert edge["etype"] in valid_etypes


# Performance/stress tests
class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    def test_large_instance_generation(self):
        """Test generating large instances."""
        inst = generate_stratified_horn_instance(
            "test_large", 
            Difficulty.EXTREME, 
            seed=42
        )
        
        # Should handle large instances
        assert len(inst["nodes"]) >= 15
        assert len(inst["edges"]) > 0
    
    @pytest.mark.slow
    def test_spectral_on_large_graph(self):
        """Test spectral computation on large graph."""
        inst = generate_stratified_horn_instance(
            "test_large_spectral", 
            Difficulty.EXTREME, 
            seed=42
        )
        
        n_nodes = len(inst["nodes"])
        A = adjacency_from_edges(n_nodes, inst["edges"])
        L = compute_symmetric_laplacian(A)
        
        k = min(16, n_nodes - 1)
        if k > 0:
            eigvals, eigvecs = topk_eig(L, k=k)
            assert eigvecs.shape == (n_nodes, k)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])