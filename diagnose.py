"""
Diagnostic Script - Automatically detect common issues
"""

import json
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

from dataset import StepPredictionDataset, create_split
from model import StepPredictorGNN
from losses import HybridRankingLoss


def check_data_quality(data_dir: str):
    """Check dataset for common issues."""
    print("\n" + "="*70)
    print("DATASET DIAGNOSTICS")
    print("="*70)
    
    issues = []
    
    # Check if data exists
    data_path = Path(data_dir)
    if not data_path.exists():
        issues.append("❌ Data directory not found")
        return issues
    
    # Load sample files
    json_files = list(data_path.rglob("*.json"))
    print(f"✓ Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        issues.append("❌ No JSON files found - run data_generator.py first")
        return issues
    
    # Check file content
    sample_file = json_files[0]
    with open(sample_file) as f:
        inst = json.load(f)
    
    # Check structure
    required_keys = ["id", "nodes", "edges", "proof_steps", "metadata"]
    for key in required_keys:
        if key not in inst:
            issues.append(f"❌ Missing key '{key}' in JSON")
    
    # Check proof steps
    n_proofs = len(inst.get("proof_steps", []))
    if n_proofs == 0:
        issues.append(f"⚠️  File {sample_file.name} has no proof steps")
    else:
        print(f"✓ Sample file has {n_proofs} proof steps")
    
    # Check for duplicates
    instance_ids = []
    for f in json_files[:100]:  # Sample
        with open(f) as fp:
            instance_ids.append(json.load(fp).get("id", ""))
    
    if len(instance_ids) != len(set(instance_ids)):
        issues.append("⚠️  Duplicate instance IDs detected")
    else:
        print(f"✓ No duplicate instance IDs (checked 100 files)")
    
    return issues


def check_dataset_class(data_dir: str):
    """Check dataset loading."""
    print("\n" + "="*70)
    print("DATASET CLASS DIAGNOSTICS")
    print("="*70)
    
    issues = []
    
    try:
        train_files, val_files, _ = create_split(data_dir, seed=42)
        
        # Check split sizes
        if len(train_files) < 10:
            issues.append(f"❌ Too few training files: {len(train_files)}")
        else:
            print(f"✓ Train files: {len(train_files)}")
        
        # Load dataset
        dataset = StepPredictionDataset(train_files[:5], spectral_dir=None)
        
        if len(dataset) == 0:
            issues.append("❌ Dataset is empty - check proof_steps in JSON")
            return issues
        
        print(f"✓ Dataset loaded: {len(dataset)} samples from 5 files")
        
        # Check sample structure
        sample = dataset[0]
        
        print(f"✓ Sample shape: x={sample.x.shape}, edges={sample.edge_index.shape}")
        
        # Check feature dimension
        feature_dim = sample.x.shape[1]
        if feature_dim < 10:
            issues.append(f"⚠️  Feature dim too small: {feature_dim} (expected 20+)")
        else:
            print(f"✓ Feature dimension: {feature_dim}")
        
        # Check target validity
        target = sample.y.item()
        n_nodes = sample.x.shape[0]
        
        if target < 0 or target >= n_nodes:
            issues.append(f"❌ Invalid target: {target} (n_nodes={n_nodes})")
        else:
            print(f"✓ Valid target: {target} < {n_nodes}")
        
    except Exception as e:
        issues.append(f"❌ Dataset loading failed: {e}")
    
    return issues


def check_dataloader(data_dir: str, batch_size: int = 64):
    """Check dataloader configuration."""
    print("\n" + "="*70)
    print("DATALOADER DIAGNOSTICS")
    print("="*70)
    
    issues = []
    
    try:
        train_files, _, _ = create_split(data_dir, seed=42)
        dataset = StepPredictionDataset(train_files[:10], spectral_dir=None)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Check batch size
        if batch_size == 1:
            issues.append("❌ CRITICAL: Batch size = 1 (should be 32-128)")
        elif batch_size < 16:
            issues.append(f"⚠️  Batch size too small: {batch_size} (recommend 64)")
        else:
            print(f"✓ Batch size: {batch_size}")
        
        # Check batch
        batch = next(iter(loader))
        
        print(f"✓ Batch loaded: {batch.num_graphs} graphs, {batch.x.shape[0]} nodes")
        
        # Check batch vector
        if hasattr(batch, 'batch'):
            print(f"✓ Batch vector present: {batch.batch.shape}")
        else:
            issues.append("⚠️  No batch vector (needed for batched training)")
        
    except Exception as e:
        issues.append(f"❌ DataLoader failed: {e}")
    
    return issues


def check_model(data_dir: str):
    """Check model architecture."""
    print("\n" + "="*70)
    print("MODEL DIAGNOSTICS")
    print("="*70)
    
    issues = []
    
    try:
        train_files, _, _ = create_split(data_dir, seed=42)
        dataset = StepPredictionDataset(train_files[:5], spectral_dir=None)
        
        sample = dataset[0]
        in_dim = sample.x.shape[1]
        
        model = StepPredictorGNN(
            in_dim=in_dim,
            hidden_dim=256,
            num_layers=4
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {n_params:,} parameters")
        
        # Test forward pass
        scores, embeddings = model(sample.x, sample.edge_index)
        
        print(f"✓ Forward pass: scores={scores.shape}, embeddings={embeddings.shape}")
        
        # Check output
        if scores.shape[0] != sample.x.shape[0]:
            issues.append(f"❌ Score shape mismatch: {scores.shape} vs {sample.x.shape}")
        
        if torch.isnan(scores).any():
            issues.append("❌ NaN in model output")
        
        if torch.isinf(scores).any():
            issues.append("❌ Inf in model output")
        
    except Exception as e:
        issues.append(f"❌ Model test failed: {e}")
    
    return issues


def check_loss_function(data_dir: str):
    """Check loss computation."""
    print("\n" + "="*70)
    print("LOSS FUNCTION DIAGNOSTICS")
    print("="*70)
    
    issues = []
    
    try:
        train_files, _, _ = create_split(data_dir, seed=42)
        dataset = StepPredictionDataset(train_files[:5], spectral_dir=None)
        
        sample = dataset[0]
        in_dim = sample.x.shape[1]
        
        model = StepPredictorGNN(in_dim=in_dim, hidden_dim=256, num_layers=4)
        criterion = HybridRankingLoss()
        
        # Forward pass
        scores, embeddings = model(sample.x, sample.edge_index)
        target_idx = sample.y.item()
        
        # Compute loss
        loss, margin_l, contrast_l = criterion(scores, embeddings, target_idx)
        
        print(f"✓ Loss computed: total={loss.item():.4f}, margin={margin_l.item():.4f}, contrastive={contrast_l.item():.4f}")
        
        # Check loss validity
        if torch.isnan(loss):
            issues.append("❌ NaN loss")
        
        if loss.item() < 0:
            issues.append(f"❌ Negative loss: {loss.item()}")
        
        if loss.item() > 100:
            issues.append(f"⚠️  Very large loss: {loss.item()} (might be OK initially)")
        
        # Test backward pass
        loss.backward()
        
        print("✓ Backward pass successful")
        
    except Exception as e:
        issues.append(f"❌ Loss computation failed: {e}")
    
    return issues


def check_training_config(config_path: str = None):
    """Check training configuration."""
    print("\n" + "="*70)
    print("TRAINING CONFIG DIAGNOSTICS")
    print("="*70)
    
    issues = []
    recommendations = []
    
    # Default config
    config = {
        "batch_size": 64,
        "lr": 3e-4,
        "epochs": 100,
        "hidden_dim": 256,
        "num_layers": 4,
        "dropout": 0.3
    }
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Check batch size
    if config["batch_size"] == 1:
        issues.append("❌ CRITICAL: batch_size=1 (change to 64)")
    elif config["batch_size"] < 32:
        recommendations.append(f"⚠️  batch_size={config['batch_size']} is small (try 64)")
    else:
        print(f"✓ batch_size: {config['batch_size']}")
    
    # Check learning rate
    if config["lr"] > 1e-3:
        recommendations.append(f"⚠️  lr={config['lr']} is high (try 3e-4)")
    else:
        print(f"✓ lr: {config['lr']}")
    
    # Check model capacity
    if config["hidden_dim"] < 128:
        recommendations.append(f"⚠️  hidden_dim={config['hidden_dim']} is small (try 256)")
    else:
        print(f"✓ hidden_dim: {config['hidden_dim']}")
    
    if config["num_layers"] < 3:
        recommendations.append(f"⚠️  num_layers={config['num_layers']} is shallow (try 4)")
    else:
        print(f"✓ num_layers: {config['num_layers']}")
    
    return issues, recommendations


def run_diagnostics(data_dir: str = "data/horn", config_path: str = None):
    """Run all diagnostics."""
    print("\n" + "="*70)
    print("SPECLOGICNET DIAGNOSTICS")
    print("="*70)
    print("Checking for common issues that cause poor Hit@1 performance...")
    
    all_issues = []
    all_recommendations = []
    
    # Run checks
    all_issues.extend(check_data_quality(data_dir))
    all_issues.extend(check_dataset_class(data_dir))
    all_issues.extend(check_dataloader(data_dir, batch_size=64))
    all_issues.extend(check_model(data_dir))
    all_issues.extend(check_loss_function(data_dir))
    
    issues, recommendations = check_training_config(config_path)
    all_issues.extend(issues)
    all_recommendations.extend(recommendations)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if len(all_issues) == 0:
        print("✅ No critical issues found!")
        print("   Training should achieve Hit@1 > 70%")
    else:
        print(f"❌ Found {len(all_issues)} critical issues:")
        for issue in all_issues:
            print(f"   {issue}")
    
    if len(all_recommendations) > 0:
        print(f"\n💡 {len(all_recommendations)} recommendations:")
        for rec in all_recommendations:
            print(f"   {rec}")
    
    print("\n" + "="*70)
    
    return len(all_issues) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose common issues")
    parser.add_argument("--data-dir", default="data/horn")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    
    success = run_diagnostics(args.data_dir, args.config)
    
    exit(0 if success else 1)