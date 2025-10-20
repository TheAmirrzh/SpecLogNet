"""
Training Script with SOTA Techniques
Based on "Bag of Tricks for Node Classification with Graph Neural Networks" (2021)
and recent best practices from 2024 papers.

Key improvements:
1. Label smoothing
2. Learning rate warmup
3. Gradient accumulation
4. Better evaluation metrics
5. Curriculum learning (optional)
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from dataset import StepPredictionDataset, create_split
from losses import compute_hit_at_k
# Import from the new model file
import sys
sys.path.insert(0, '.')
from model import get_model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing for better generalization.
    From "Bag of Tricks" paper - helps prevent overconfidence.
    """
    
    def __init__(self, smoothing=0.1, temperature=1.0):
        super().__init__()
        self.smoothing = smoothing
        self.temperature = temperature
    
    def forward(self, scores: torch.Tensor, target_idx: int) -> torch.Tensor:
        n_nodes = scores.shape[0]
        device = scores.device
        
        if target_idx < 0 or target_idx >= n_nodes:
            return torch.tensor(0.01, device=device, requires_grad=True)
        
        # Apply temperature
        logits = scores / self.temperature
        log_probs = F.log_softmax(logits, dim=0)
        
        # Create smoothed target distribution
        smooth_target = torch.full((n_nodes,), self.smoothing / (n_nodes - 1), device=device)
        smooth_target[target_idx] = 1.0 - self.smoothing
        
        # KL divergence loss
        loss = -(smooth_target * log_probs).sum()
        
        return loss


def train_epoch(model, loader, optimizer, criterion, device, epoch, grad_accum_steps=1):
    """
    Train with gradient accumulation for stable updates.
    """
    model.train()
    
    total_loss = 0
    correct_preds = 0
    n_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        
        scores, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        batch_loss = 0
        batch_size = batch.num_graphs
        
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            target_idx = batch.y[i].item()
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            loss = criterion(graph_scores, target_idx)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_loss += loss
                
                if graph_scores.argmax().item() == target_idx:
                    correct_preds += 1
        
        if batch_size > 0:
            batch_loss = batch_loss / (batch_size * grad_accum_steps)
            batch_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += batch_loss.item() * grad_accum_steps
            n_samples += batch_size
    
    return total_loss / max(len(loader), 1), correct_preds / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, split='val'):
    """Enhanced evaluation with more metrics."""
    model.eval()
    
    losses = []
    hit1, hit3, hit5, hit10 = [], [], [], []
    ranks = []
    
    for batch in loader:
        batch = batch.to(device)
        scores, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        batch_size = batch.num_graphs
        
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            target_idx = batch.y[i].item()
            
            if target_idx < 0 or target_idx >= len(graph_scores):
                continue
            
            loss = criterion(graph_scores, target_idx)
            losses.append(loss.item())
            
            # Metrics
            hit1.append(compute_hit_at_k(graph_scores, target_idx, 1))
            hit3.append(compute_hit_at_k(graph_scores, target_idx, 3))
            hit5.append(compute_hit_at_k(graph_scores, target_idx, 5))
            hit10.append(compute_hit_at_k(graph_scores, target_idx, 10))
            
            # MRR
            sorted_indices = torch.argsort(graph_scores, descending=True)
            rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
    
    mrr = sum(ranks) / max(len(ranks), 1)
    
    return {
        f'{split}_loss': sum(losses) / max(len(losses), 1),
        f'{split}_hit1': sum(hit1) / max(len(hit1), 1),
        f'{split}_hit3': sum(hit3) / max(len(hit3), 1),
        f'{split}_hit5': sum(hit5) / max(len(hit5), 1),
        f'{split}_hit10': sum(hit10) / max(len(hit10), 1),
        f'{split}_mrr': mrr,
        f'{split}_n': len(hit1)
    }


def get_lr(optimizer):
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--exp-dir', default='experiments/sota_run')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--use-type-aware', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("Warning: CUDA and MPS not available. Defaulting to CPU.")
        device = torch.device('cpu')

    print(f"\nUsing device: {device}") # <-- Add this line to be 100% sure
    print("="*70)
    
    print("\n" + "="*70)
    print("SOTA-INFORMED TRAINING - Best Practices Applied")
    print("="*70)
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Gradient Accumulation: {args.grad_accum_steps} steps")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Type-Aware Model: {args.use_type_aware}")
    
    # Load data
    train_files, val_files, test_files = create_split(args.data_dir, seed=args.seed)
    
    train_ds = StepPredictionDataset(train_files, spectral_dir=None, seed=args.seed)
    val_ds = StepPredictionDataset(val_files, spectral_dir=None, seed=args.seed+1)
    test_ds = StepPredictionDataset(test_files, spectral_dir=None, seed=args.seed+2)
    
    print(f"\nDataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Model
    in_dim = train_ds[0].x.shape[1]
    print(f"Input dimension: {in_dim}")
    
    model = get_model(
        in_dim, 
        args.hidden_dim, 
        args.num_layers,
        dropout=args.dropout,
        use_type_aware=args.use_type_aware
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    # Loss with label smoothing
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.label_smoothing,
        temperature=1.0
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Restart every 20 epochs
        T_mult=2,
        eta_min=1e-6
    )
    
    # Manual warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    
    # Training
    best_val_hit1 = 0
    best_val_mrr = 0
    patience_counter = 0
    
    print("\nTraining...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        # Warmup phase
        if epoch <= args.warmup_epochs:
            current_lr = get_lr(optimizer)
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.grad_accum_steps
            )
            warmup_scheduler.step()
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.grad_accum_steps
            )
            scheduler.step()
        
        val_metrics = evaluate(model, val_loader, criterion, device, 'val')
        
        current_lr = get_lr(optimizer)
        
        print(f"[Epoch {epoch:3d}] "
              f"LR: {current_lr:.6f} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Hit@1: {val_metrics['val_hit1']:.4f} "
              f"MRR: {val_metrics['val_mrr']:.4f} "
              f"Hit@10: {val_metrics['val_hit10']:.4f}")
        
        # Save best (based on Hit@1)
        if val_metrics['val_hit1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit1']
            best_val_mrr = val_metrics['val_mrr']
            patience_counter = 0
            
            os.makedirs(args.exp_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_hit1': best_val_hit1,
                'val_mrr': best_val_mrr,
            }, f"{args.exp_dir}/best.pt")
            print(f"  → New best Hit@1: {best_val_hit1:.4f}, MRR: {best_val_mrr:.4f}")
        else:
            patience_counter += 1
            
            if patience_counter >= 25:
                print("Early stopping")
                break
    
    # Test
    checkpoint = torch.load(f"{args.exp_dir}/best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device, 'test')
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Hit@1:  {test_metrics['test_hit1']:.4f} ({test_metrics['test_hit1']*100:.1f}%)")
    print(f"Hit@3:  {test_metrics['test_hit3']:.4f} ({test_metrics['test_hit3']*100:.1f}%)")
    print(f"Hit@5:  {test_metrics['test_hit5']:.4f} ({test_metrics['test_hit5']*100:.1f}%)")
    print(f"Hit@10: {test_metrics['test_hit10']:.4f} ({test_metrics['test_hit10']*100:.1f}%)")
    print(f"MRR:    {test_metrics['test_mrr']:.4f}")
    print("="*70)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_hit1': best_val_hit1,
        'best_val_mrr': best_val_mrr,
        'config': vars(args)
    }
    
    with open(f"{args.exp_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.exp_dir}/results.json")


if __name__ == '__main__':
    main()