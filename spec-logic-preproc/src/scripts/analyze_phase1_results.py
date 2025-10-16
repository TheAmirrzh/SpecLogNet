# src/scripts/analyze_phase1_results.py
"""
Analyze and visualize Phase 1 experimental results.
Creates plots, tables, and summary reports.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_experiment_results(exp_dir: str) -> Dict:
    """Load all results from an experiment directory."""
    results = {
        "config": None,
        "metrics_history": [],
        "final_results": None
    }
    
    # Load config
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results["config"] = json.load(f)
    
    # Load metrics history
    metrics_path = os.path.join(exp_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                results["metrics_history"].append(json.loads(line))
    
    # Load final results
    final_path = os.path.join(exp_dir, "final_results.json")
    if os.path.exists(final_path):
        with open(final_path) as f:
            results["final_results"] = json.load(f)
    
    return results


def plot_training_curves(results: Dict, save_dir: str):
    """Plot training curves (loss, metrics over epochs)."""
    history = results["metrics_history"]
    if not history:
        print("No training history found")
        return
    
    epochs = [m["epoch"] for m in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, [m["train_loss"] for m in history], label="Train Loss", marker='o', markersize=3)
    ax.plot(epochs, [m["val_loss"] for m in history], label="Val Loss", marker='s', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hit@1 curve
    ax = axes[0, 1]
    ax.plot(epochs, [m["val_hit1"] for m in history], label="Val Hit@1", 
            marker='o', markersize=3, color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit@1")
    ax.set_title("Validation Hit@1 over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Hit@K comparison
    ax = axes[1, 0]
    ax.plot(epochs, [m["val_hit1"] for m in history], label="Hit@1", marker='o', markersize=2)
    ax.plot(epochs, [m["val_hit3"] for m in history], label="Hit@3", marker='s', markersize=2)
    ax.plot(epochs, [m["val_hit10"] for m in history], label="Hit@10", marker='^', markersize=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit@K")
    ax.set_title("Hit@K Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # MRR curve
    ax = axes[1, 1]
    ax.plot(epochs, [m.get("val_mrr", 0) for m in history], 
            label="Val MRR", marker='o', markersize=3, color='purple')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_learning_rate_schedule(results: Dict, save_dir: str):
    """Plot learning rate over epochs."""
    history = results["metrics_history"]
    if not history:
        return
    
    epochs = [m["epoch"] for m in history]
    lrs = [m.get("lr", 0) for m in history]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, marker='o', markersize=3, color='red')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "lr_schedule.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved LR schedule to {save_path}")
    plt.close()


def create_results_table(results: Dict, save_dir: str):
    """Create formatted results table."""
    final = results.get("final_results")
    if not final:
        print("No final results found")
        return
    
    test_metrics = final.get("test_metrics", {})
    best_val = final.get("best_val_metrics", {})
    
    # Create table
    table_data = {
        "Metric": [
            "Loss",
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR",
            "Samples"
        ],
        "Validation (Best)": [
            f"{best_val.get('val_loss', 0):.4f}",
            f"{best_val.get('val_hit1', 0):.4f}",
            f"{best_val.get('val_hit3', 0):.4f}",
            f"{best_val.get('val_hit5', 0):.4f}",
            f"{best_val.get('val_hit10', 0):.4f}",
            f"{best_val.get('val_mrr', 0):.4f}",
            f"{best_val.get('val_samples', 0):.0f}"
        ],
        "Test": [
            f"{test_metrics.get('test_loss', 0):.4f}",
            f"{test_metrics.get('test_hit1', 0):.4f}",
            f"{test_metrics.get('test_hit3', 0):.4f}",
            f"{test_metrics.get('test_hit5', 0):.4f}",
            f"{test_metrics.get('test_hit10', 0):.4f}",
            f"{test_metrics.get('test_mrr', 0):.4f}",
            f"{test_metrics.get('test_samples', 0):.0f}"
        ]
    }
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(save_dir, "results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    
    # Save as Markdown
    md_path = os.path.join(save_dir, "results_table.md")
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"Saved markdown table to {md_path}")


def generate_summary_report(results: Dict, save_dir: str):
    """Generate comprehensive text summary report."""
    config = results.get("config", {})
    final = results.get("final_results", {})
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PHASE 1 EXPERIMENT SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # New: Handle missing final results
    if final is None:
        report_lines.append("WARNING: No final results found. Training may have failed or produced no samples.")
        report_lines.append("Check train.log for details (e.g., empty dataset).")
        report_lines.append("")
    else:
        # Config summary
        report_lines.append("EXPERIMENT CONFIGURATION")
        report_lines.append("-" * 80)
        report_lines.append(f"  Experiment Name: {final.get('exp_name', 'N/A')}")
        report_lines.append(f"  Dataset: {config.get('json_dir', 'N/A')}")
        report_lines.append(f"  Spectral Features: {config.get('spectral_dir', 'None')}")
        report_lines.append(f"  Model: GCN with {config.get('num_layers', 2)} layers, hidden_dim={config.get('hidden_dim', 128)}")
        report_lines.append(f"  Training: {config.get('epochs', 50)} epochs, lr={config.get('lr', 0.001)}, batch_size={config.get('batch_size', 1)}")
        report_lines.append(f"  Total Training Time: {final.get('training_time', 0) / 3600:.2f} hours")
        report_lines.append("")
        
        # Best validation results
        best_val = final.get("best_val_metrics", {})
        if best_val:
            report_lines.append("BEST VALIDATION RESULTS")
            report_lines.append("-" * 80)
            report_lines.append(f"  Loss: {best_val.get('val_loss', 0):.4f}")
            report_lines.append(f"  Hit@1: {best_val.get('val_hit1', 0):.4f} ({best_val.get('val_hit1', 0)*100:.2f}%)")
            report_lines.append(f"  Hit@3: {best_val.get('val_hit3', 0):.4f} ({best_val.get('val_hit3', 0)*100:.2f}%)")
            report_lines.append(f"  Hit@10: {best_val.get('val_hit10', 0):.4f} ({best_val.get('val_hit10', 0)*100:.2f}%)")
            report_lines.append(f"  MRR: {best_val.get('val_mrr', 0):.4f}")
            report_lines.append("")
        
        # Test results
        test_metrics = final.get("test_metrics", {})
        if test_metrics:
            report_lines.append("FINAL TEST RESULTS")
            report_lines.append("-" * 80)
            report_lines.append(f"  Loss: {test_metrics.get('test_loss', 0):.4f}")
            report_lines.append(f"  Hit@1: {test_metrics.get('test_hit1', 0):.4f} ({test_metrics.get('test_hit1', 0)*100:.2f}%)")
            report_lines.append(f"  Hit@3: {test_metrics.get('test_hit3', 0):.4f} ({test_metrics.get('test_hit3', 0)*100:.2f}%)")
            report_lines.append(f"  Hit@10: {test_metrics.get('test_hit10', 0):.4f} ({test_metrics.get('test_hit10', 0)*100:.2f}%)")
            report_lines.append(f"  MRR: {test_metrics.get('test_mrr', 0):.4f}")
            report_lines.append(f"  Test samples: {test_metrics.get('test_samples', 0):.0f}")
            report_lines.append("")
        
        # Key findings
        report_lines.append("KEY FINDINGS")
        report_lines.append("-" * 80)
        
        if test_metrics:
            hit1 = test_metrics.get('test_hit1', 0)
            hit10 = test_metrics.get('test_hit10', 0)
            
            if hit1 >= 0.8:
                report_lines.append("  ✅ EXCELLENT: Test Hit@1 >= 80% - Strong baseline performance")
            elif hit1 >= 0.6:
                report_lines.append("  ✓ GOOD: Test Hit@1 >= 60% - Solid baseline")
            elif hit1 >= 0.4:
                report_lines.append("  ⚠ FAIR: Test Hit@1 >= 40% - Room for improvement")
            else:
                report_lines.append("  ⚠ NEEDS WORK: Test Hit@1 < 40% - Requires tuning")
            
            report_lines.append(f"  • Top-1 accuracy: {hit1*100:.1f}%")
            report_lines.append(f"  • Top-10 accuracy: {hit10*100:.1f}%")
            
            gap = hit10 - hit1
            report_lines.append(f"  • Hit@10 - Hit@1 gap: {gap*100:.1f}% (larger gap = more room for ranking improvements)")
    
    report_lines.append("")
    report_lines.append("NEXT STEPS FOR PHASE 2")
    report_lines.append("-" * 80)
    report_lines.append("  1. Implement spectral feature integration")
    report_lines.append("  2. Run ablation: GCN vs GCN+Spectral")
    report_lines.append("  3. Analyze which problem types benefit most from spectral features")
    report_lines.append("  4. Tune spectral encoder architecture (k value, filter types)")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_path = os.path.join(save_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nSummary report saved to {report_path}")


def analyze_experiment(exp_dir: str, output_dir: str = None):
    """Complete analysis of an experiment."""
    if output_dir is None:
        output_dir = os.path.join(exp_dir, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing experiment: {exp_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Load results
    results = load_experiment_results(exp_dir)
    
    # Generate all plots and reports
    plot_training_curves(results, output_dir)
    create_results_table(results, output_dir)
    plot_learning_rate_schedule(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print(f"\n✅ Analysis complete! All outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Phase 1 Results")
    parser.add_argument("--exp-dir", type=str, required=True,
                        help="Experiment directory to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis (default: exp_dir/analysis)")
    
    args = parser.parse_args()
    
    analyze_experiment(args.exp_dir, args.output_dir)