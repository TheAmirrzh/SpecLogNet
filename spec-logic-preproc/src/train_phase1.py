# src/train_phase1.py
"""
Complete Phase 1 training script with:
- Experiment tracking
- Comprehensive metrics
- Checkpoint management
- Visualization
"""

import os
import json
import glob
import time
import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

from src.train_step_predictor import StepPredictionDataset, evaluate_model, hit_at_k
from src.models.gcn_step_predictor import GCNStepPredictor


class ExperimentTracker:
    """Simple experiment tracker (no external dependencies)."""
    
    def __init__(self, exp_name: str, log_dir: str = "experiments"):
        self.exp_name = exp_name
        self.exp_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.exp_dir, "metrics.jsonl")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        self.log_file = os.path.join(self.exp_dir, "train.log")
        
        self.start_time = time.time()
    
    def log_config(self, config: Dict):
        """Save experiment configuration."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, epoch: int, metrics: Dict):
        """Append metrics to JSONL file."""
        metrics["epoch"] = epoch
        metrics["timestamp"] = time.time() - self.start_time
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def log_message(self, message: str):
        """Log a message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics
        }
        
        # Save latest
        ckpt_path = os.path.join(self.exp_dir, "checkpoint_latest.pt")
        torch.save(ckpt, ckpt_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoint_best.pt")
            torch.save(ckpt, best_path)
            self.log_message(f"Saved best checkpoint (epoch {epoch})")
    
    def load_metrics_history(self) -> List[Dict]:
        """Load all metrics from JSONL file."""
        if not os.path.exists(self.metrics_file):
            return []
        
        history = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                history.append(json.loads(line))
        return history


def compute_detailed_metrics(model, dataloader, device, dataset_name="val"):
    """Compute comprehensive evaluation metrics."""
    model.eval()
    
    metrics = {
        f"{dataset_name}_loss": 0.0,
        f"{dataset_name}_hit1": 0.0,
        f"{dataset_name}_hit3": 0.0,
        f"{dataset_name}_hit5": 0.0,
        f"{dataset_name}_hit10": 0.0,
        f"{dataset_name}_mrr": 0.0,  # Mean Reciprocal Rank
        f"{dataset_name}_samples": 0
    }
    
    ce = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            
            # Handle batched data
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            scores, _ = model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            
            if batch_size > 1:
                # Process each graph in the batch
                for i in range(batch_size):
                    mask = (data.batch == i)
                    graph_scores = scores[mask]
                    target_idx = int(data.y[i].item())
                    
                    if target_idx < 0 or target_idx >= len(graph_scores):
                        continue
                    
                    logits = graph_scores.unsqueeze(0)
                    targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                    
                    # Loss
                    loss = ce(logits, targ)
                    metrics[f"{dataset_name}_loss"] += float(loss.item())
                    
                    # Hit@K metrics
                    metrics[f"{dataset_name}_hit1"] += hit_at_k(graph_scores, target_idx, 1)
                    metrics[f"{dataset_name}_hit3"] += hit_at_k(graph_scores, target_idx, 3)
                    metrics[f"{dataset_name}_hit5"] += hit_at_k(graph_scores, target_idx, 5)
                    metrics[f"{dataset_name}_hit10"] += hit_at_k(graph_scores, target_idx, 10)
                    
                    # MRR: 1 / rank of correct answer
                    sorted_scores = torch.argsort(graph_scores, descending=True)
                    rank = (sorted_scores == target_idx).nonzero(as_tuple=True)[0].item() + 1
                    metrics[f"{dataset_name}_mrr"] += 1.0 / rank
                    
                    metrics[f"{dataset_name}_samples"] += 1
            else:
                # Single graph case
                target_idx = int(data.y.view(-1).item())
                
                if target_idx < 0:
                    continue
                
                logits = scores.unsqueeze(0)
                targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                
                # Loss
                loss = ce(logits, targ)
                metrics[f"{dataset_name}_loss"] += float(loss.item())
                
                # Hit@K metrics
                metrics[f"{dataset_name}_hit1"] += hit_at_k(scores, target_idx, 1)
                metrics[f"{dataset_name}_hit3"] += hit_at_k(scores, target_idx, 3)
                metrics[f"{dataset_name}_hit5"] += hit_at_k(scores, target_idx, 5)
                metrics[f"{dataset_name}_hit10"] += hit_at_k(scores, target_idx, 10)
                
                # MRR: 1 / rank of correct answer
                sorted_scores = torch.argsort(scores, descending=True)
                rank = (sorted_scores == target_idx).nonzero(as_tuple=True)[0].item() + 1
                metrics[f"{dataset_name}_mrr"] += 1.0 / rank
                
                metrics[f"{dataset_name}_samples"] += 1
    
    # Normalize
    n = max(1, metrics[f"{dataset_name}_samples"])
    for key in metrics:
        if key != f"{dataset_name}_samples":
            metrics[key] /= n
    
    return metrics


def train_phase1(args):
    """Main training function for Phase 1."""
    
    # Initialize experiment tracker
    # Check if exp_name already has a timestamp pattern (to avoid double-timestamping)
    if args.exp_name.startswith("phase1_") and any(char.isdigit() for char in args.exp_name[-15:]):
        # Already has phase1_ prefix and likely a timestamp, use as-is
        exp_name = args.exp_name
    else:
        # Add prefix and timestamp
        exp_name = f"phase1_{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = ExperimentTracker(exp_name, args.log_dir)
    
    # Log configuration
    config = vars(args)
    tracker.log_config(config)
    tracker.log_message(f"Starting Phase 1 training: {exp_name}")
    tracker.log_message(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Device setup
    device = torch.device(args.device)
    tracker.log_message(f"Using device: {device}")
    
    # Load data
    tracker.log_message("Loading datasets...")
    files = sorted(glob.glob(os.path.join(args.json_dir, "**/*.json"), recursive=True))
    
    if len(files) == 0:
        tracker.log_message(f"ERROR: No JSON files found in {args.json_dir}")
        return
    
    tracker.log_message(f"Found {len(files)} JSON files")
    
    # Split data
    np.random.seed(args.seed)
    np.random.shuffle(files)
    n = len(files)
    n_test = max(1, int(n * args.test_fraction))
    n_val = max(1, int(n * args.val_fraction))
    
    train_files = files[: n - n_val - n_test]
    val_files = files[n - n_val - n_test: n - n_test]
    test_files = files[n - n_test:]
    
    tracker.log_message(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Create datasets
    train_ds = StepPredictionDataset(train_files, spectral_dir=args.spectral_dir, seed=args.seed)
    val_ds = StepPredictionDataset(val_files, spectral_dir=args.spectral_dir, seed=args.seed + 1)
    test_ds = StepPredictionDataset(test_files, spectral_dir=args.spectral_dir, seed=args.seed + 2)
    
    tracker.log_message(f"Dataset sizes: {len(train_ds)} train samples, {len(val_ds)} val, {len(test_ds)} test")
    
    # Create dataloaders
    train_loader = GeometricDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = GeometricDataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Determine input features
    sample0 = train_ds[0]
    in_feats = sample0.x.size(1)
    tracker.log_message(f"Input features dimension: {in_feats}")
    
    # Build model
    model = GCNStepPredictor(
        in_feats=in_feats,
        hidden=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    tracker.log_message(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    ce_loss = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_val_hit1 = -1.0
    patience_counter = 0
    
    tracker.log_message("\n" + "="*80)
    tracker.log_message("Starting training...")
    tracker.log_message("="*80 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Handle batched data
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            
            optimizer.zero_grad()
            scores, _ = model(data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None)
            
            # For batched graphs, we need to handle each graph separately
            if batch_size > 1:
                # Get per-graph scores and targets
                batch_losses = []
                for i in range(batch_size):
                    # Get nodes belonging to graph i
                    mask = (data.batch == i)
                    graph_scores = scores[mask]
                    target_idx = int(data.y[i].item())
                    
                    if target_idx < 0 or target_idx >= len(graph_scores):
                        continue
                    
                    logits = graph_scores.unsqueeze(0)
                    targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                    batch_losses.append(ce_loss(logits, targ))
                
                if len(batch_losses) == 0:
                    continue
                    
                loss = torch.stack(batch_losses).mean()
            else:
                # Single graph case
                target_idx = int(data.y.view(-1).item())
                if target_idx < 0:
                    continue
                    
                logits = scores.unsqueeze(0)
                targ = torch.tensor([target_idx], dtype=torch.long, device=device)
                loss = ce_loss(logits, targ)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            train_losses.append(float(loss.item()))
            
            # Progress logging
            if batch_idx % 50 == 0:
                tracker.log_message(
                    f"Epoch {epoch}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute epoch metrics
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        
        # Compute training metrics (for overfitting detection)
        train_metrics = compute_detailed_metrics(model, train_loader, device, "train")
        
        # Validation
        val_metrics = compute_detailed_metrics(model, val_loader, device, "val")
        
        # Calculate overfitting gap
        overfit_gap = train_metrics["train_hit1"] - val_metrics["val_hit1"]
        
        # Learning rate scheduling
        scheduler.step(val_metrics["val_hit1"])
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        
        log_msg = (
            f"\nEpoch {epoch}/{args.epochs} completed in {epoch_time:.1f}s\n"
            f"  Train Loss:    {train_loss:.4f}\n"
            f"  Train Hit@1:   {train_metrics['train_hit1']:.4f}\n"
            f"  Val Loss:      {val_metrics['val_loss']:.4f}\n"
            f"  Val Hit@1:     {val_metrics['val_hit1']:.4f}\n"
            f"  Val Hit@3:     {val_metrics['val_hit3']:.4f}\n"
            f"  Val Hit@10:    {val_metrics['val_hit10']:.4f}\n"
            f"  Val MRR:       {val_metrics['val_mrr']:.4f}\n"
            f"  Overfit Gap:   {overfit_gap:.4f}\n"
        )
        tracker.log_message(log_msg)
        
        # Save metrics
        epoch_metrics = {
            "train_loss": train_loss,
            **train_metrics,
            **val_metrics,
            "overfit_gap": overfit_gap,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time
        }
        tracker.log_metrics(epoch, epoch_metrics)
        
        # Checkpoint saving
        is_best = val_metrics["val_hit1"] > best_val_hit1
        if is_best:
            best_val_hit1 = val_metrics["val_hit1"]
            patience_counter = 0
        else:
            patience_counter += 1
        
        tracker.save_checkpoint(model, optimizer, epoch, epoch_metrics, is_best)
        
        # Early stopping
        if patience_counter >= args.patience:
            tracker.log_message(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Final testing
    tracker.log_message("\n" + "="*80)
    tracker.log_message("Running final test evaluation...")
    tracker.log_message("="*80 + "\n")
    
    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(tracker.exp_dir, "checkpoint_best.pt"))
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_metrics = compute_detailed_metrics(model, test_loader, device, "test")
    
    test_log = (
        f"\nFINAL TEST RESULTS:\n"
        f"  Test Loss:   {test_metrics['test_loss']:.4f}\n"
        f"  Test Hit@1:  {test_metrics['test_hit1']:.4f}\n"
        f"  Test Hit@3:  {test_metrics['test_hit3']:.4f}\n"
        f"  Test Hit@10: {test_metrics['test_hit10']:.4f}\n"
        f"  Test MRR:    {test_metrics['test_mrr']:.4f}\n"
        f"  Test Samples: {test_metrics['test_samples']:.0f}\n"
    )
    tracker.log_message(test_log)
    
    # Save final results
    final_results = {
        "exp_name": exp_name,
        "config": config,
        "best_val_metrics": best_ckpt["metrics"],
        "test_metrics": test_metrics,
        "total_epochs": epoch,
        "training_time": time.time() - tracker.start_time
    }
    
    results_file = os.path.join(tracker.exp_dir, "final_results.json")
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    tracker.log_message(f"\nTraining complete! Results saved to: {tracker.exp_dir}")
    
    return model, test_metrics, tracker.exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Training Script")
    
    # Data
    parser.add_argument("--json-dir", type=str, required=True, help="Directory with JSON instances")
    parser.add_argument("--spectral-dir", type=str, default=None, help="Directory with spectral features")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test split fraction")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    # Experiment
    parser.add_argument("--exp-name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--log-dir", type=str, default="experiments", help="Log directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run training
    train_phase1(args)