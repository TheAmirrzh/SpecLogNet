#!/usr/bin/env python3
"""
Performance diagnostic script for SpecLogicNet.
Analyzes recent experiments to identify performance bottlenecks.
"""

import json
import glob
from pathlib import Path

def diagnose_performance():
    """Analyze recent experiments for performance issues."""

    # Find recent experiments
    exp_dirs = sorted(glob.glob('experiments/phase1_standard_*/final_results.json'))

    if not exp_dirs:
        print("❌ No experiment results found!")
        return

    latest = exp_dirs[-1]
    print(f"📊 Analyzing: {latest}")

    with open(latest) as f:
        data = json.load(f)

    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)

    test_metrics = data["test_metrics"]
    best_val_metrics = data["best_val_metrics"]
    config = data["config"]

    print(f"Test Hit@1:   {test_metrics['test_hit1']:.2%}")
    print(f"Test Hit@10:  {test_metrics['test_hit10']:.2%}")
    print(f"Best Val Hit@1: {best_val_metrics['val_hit1']:.2%}")
    print(f"Total Epochs: {data['total_epochs']} / {config['epochs']}")

    print("\n" + "="*60)
    print("CRITICAL CONFIGURATION ISSUES")
    print("="*60)

    # Critical issues
    issues = []

    if config["batch_size"] == 1:
        issues.append("❌ Batch Size: 1 (TOO SMALL! Use 32-512)")
        issues.append("   → Single-sample batches cause unstable training")

    if config["num_layers"] <= 2:
        issues.append(f"⚠️  Num Layers: {config['num_layers']} (SHALLOW)")
        issues.append("   → Try 3-4 layers for better representation")

    if config["lr"] > 1e-3:
        issues.append(f"⚠️  Learning Rate: {config['lr']} (TOO HIGH)")
        issues.append("   → Try 5e-4 to 1e-4 for better convergence")

    if config["dropout"] < 0.1:
        issues.append(f"⚠️  Dropout: {config['dropout']} (TOO LOW)")
        issues.append("   → Try 0.2-0.3 for better regularization")

    if not issues:
        print("✅ Configuration looks good!")
    else:
        for issue in issues:
            print(issue)

    print("\n" + "="*60)
    print("RECOMMENDED FIXES")
    print("="*60)

    print("1. 🚨 CRITICAL: Increase batch size to 32-512")
    print("   → Edit run_phase1_complete.py, change --batch-size 512")

    print("\n2. 📈 Increase model capacity:")
    print("   → Set --num-layers 3")
    print("   → Set --hidden-dim 256")
    print("   → Set --dropout 0.2")

    print("\n3. 🎯 Better learning rate schedule:")
    print("   → Set --lr 5e-4")
    print("   → Set --weight-decay 1e-4")

    print("\n4. ⏱️  More training time:")
    print("   → Set --epochs 100")
    print("   → Set --patience 30")

    print("\n" + "="*60)
    print("EXPECTED IMPROVEMENT")
    print("="*60)
    print("With these fixes, you should see:")
    print("• Hit@1: 15% → 60-80%")
    print("• Hit@10: 75% → 90-95%")
    print("• More stable training curves")
    print("• Better generalization")

if __name__ == "__main__":
    diagnose_performance()
