# run_phase1_complete.py
"""
Master script to run complete Phase 1 workflow:
1. Generate dataset
2. Validate dataset
3. Train baseline model
4. Analyze results
5. Generate report

Usage:
    python run_phase1_complete.py --quick       # Quick test (50 instances, 10 epochs)
    python run_phase1_complete.py --standard    # Standard run (450 instances, 50 epochs)
    python run_phase1_complete.py --full        # Full dataset (1000+ instances, 100 epochs)
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


class Phase1Runner:
    def __init__(self, mode="standard", device="cpu", seed=42):
        self.mode = mode
        self.device = device
        self.seed = seed
        # Use current date/time to avoid conflicts
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set parameters based on mode
        self.configs = {
            "quick": {
                "n_easy": 20,
                "n_medium": 20,
                "n_hard": 10,
                "n_extreme": 1, 
                "epochs": 10,
                "description": "Quick test run"
            },
            "standard": {
                "n_easy": 150,
                "n_medium": 150,
                "n_hard": 100,
                "n_extreme": 50,
                "epochs": 50,
                "description": "Standard Phase 1 baseline"
            },
            "full": {
                "n_easy": 300,
                "n_medium": 300,
                "n_hard": 250,
                "n_extreme": 150,
                "epochs": 100,
                "description": "Full dataset for publication"
            }
        }
        
        self.config = self.configs[mode]
        
        # Paths
        # self.base_dir is implicitly the current working directory (spec-logic-preproc)
        self.base_dir = Path(".") 
        self.data_dir = self.base_dir / "data_processed" / f"phase1_{mode}_{self.timestamp}"
        self.exp_dir = self.base_dir / "experiments" / f"phase1_{mode}_{self.timestamp}" # Full desired experiment path
        
        print("=" * 80)
        print(f"PHASE 1 COMPLETE WORKFLOW - {mode.upper()} MODE")
        print("=" * 80)
        print(f"Description: {self.config['description']}")
        print(f"Data directory: {self.data_dir}")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print("=" * 80 + "\n")
    
    def run_command(self, cmd, step_name):
        """Run a command and handle errors."""
        print(f"\n{'='*80}")
        print(f"STEP: {step_name}")
        print(f"{'='*80}")
        # Print command clearly for debugging
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            # Set cwd to base_dir (e.g., spec-logic-preproc)
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n✅ {step_name} completed successfully\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {step_name} failed with error code {e.returncode}")
            print(f"Error: {e}\n")
            return False
    
    def step1_generate_data(self):
        """Step 1: Generate dataset with spectral features."""
        cmd = [
            sys.executable, "-m", "src.scripts.generate_phase1_data",
            "--out-dir", str(self.data_dir),
            "--n-easy", str(self.config["n_easy"]),
            "--n-medium", str(self.config["n_medium"]),
            "--n-hard", str(self.config["n_hard"]),
            "--n-extreme", str(self.config["n_extreme"]),
            "--spectral-k", "16",
            "--seed", str(self.seed),
            "--validate"
        ]
        
        return self.run_command(cmd, "Data Generation")
    
    def step2_train_baseline(self):
        """Step 2: Train baseline GCN model."""
        
        # Manually ensure the target experiment directory exists
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # We need to extract the experiment name (the last part) and the log directory (the parent)
        exp_name = self.exp_dir.name  # e.g., phase1_quick_20251015_134237
        # The parent is 'experiments' relative to the base_dir (.)
        log_dir = self.exp_dir.parent 
        
        # DIAGNOSTIC CHECKPOINT
        print(f"*** CHECKPOINT 1: Training directory created at absolute path: {self.exp_dir.resolve()}")
        
        cmd = [
            sys.executable, "-m", "src.train_phase1",
            "--json-dir", str(self.data_dir),
            "--spectral-dir", str(self.data_dir / "spectral"),
            "--exp-name", exp_name,
            "--log-dir", str(log_dir),
            "--hidden-dim", "128",
            "--num-layers", "3",
            "--dropout", "0.1",
            "--epochs", str(self.config["epochs"]),
            "--batch-size", "32",
            "--lr", "0.001",
            "--weight-decay", "1e-5",
            "--grad-clip", "1.0",
            "--patience", "15",
            "--device", self.device,
            "--seed", str(self.seed)
        ]
        
        return self.run_command(cmd, "Baseline Training")
    
    def step3_analyze_results(self):
        """Step 3: Analyze training results."""
        cmd = [
            sys.executable, "-m", "src.scripts.analyze_phase1_results",
            "--exp-dir", str(self.exp_dir)
        ]
        return self.run_command(cmd, "Results Analysis")
    
    def step4_generate_summary(self):
        """Step 4: Generate final summary document."""
        
        latest_exp = self.exp_dir
        final_results_path = latest_exp / "final_results.json"
        
        if not final_results_path.exists():
            print("❌ Final results not found")
            return False
        
        with open(final_results_path) as f:
            results = json.load(f)
        
        # Create markdown summary
        summary_md = self._create_markdown_summary(results, latest_exp)
        
        summary_path = latest_exp / "PHASE1_SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write(summary_md)
        
        print(f"\n✅ Summary document created: {summary_path}\n")
        return True
    
    def _create_markdown_summary(self, results, exp_dir):
        """Create markdown summary document."""
        test_metrics = results.get("test_metrics", {})
        config = results.get("config", {})
        
        # Handle cases where keys might be nested differently, e.g., 'test_hit1' or 'hit1'
        get_metric = lambda k, default=0: test_metrics.get(f"test_{k}", test_metrics.get(k, default))

        md = f"""# Phase 1 Baseline Results

## Experiment Information

- **Mode**: {self.mode}
- **Date**: {self.timestamp}
- **Experiment Directory**: `{exp_dir}`
- **Dataset**: {self.config['n_easy'] + self.config['n_medium'] + self.config['n_hard'] + self.config['n_extreme']} total instances

## Configuration

```json
{json.dumps(config, indent=2)}
```

## Results Summary

### Test Set Performance

| Metric | Value | Percentage |
|--------|-------|------------|
| Loss | {get_metric('loss'):.4f} | - |
| Hit@1 | {get_metric('hit1'):.4f} | {get_metric('hit1')*100:.2f}% |
| Hit@3 | {get_metric('hit3'):.4f} | {get_metric('hit3')*100:.2f}% |
| Hit@10 | {get_metric('hit10'):.4f} | {get_metric('hit10')*100:.2f}% |
| Samples | {get_metric('n', 0):.0f} | - |

### Key Findings

- **Baseline Performance**: Hit@1 = {get_metric('hit1')*100:.1f}%
- **Ranking Quality**: Hit@10 - Hit@1 = {(get_metric('hit10') - get_metric('hit1'))*100:.1f}%

### Status

{'✅ **PHASE 1 COMPLETE** - Ready for Phase 2' if get_metric('hit1', 0) >= 0.4 else '⚠️ **NEEDS IMPROVEMENT** - Consider hyperparameter tuning'}

## Visualizations

- Training curves: `{exp_dir}/analysis/training_curves.png` (Requires Step 3 to run)
- Full report: `{exp_dir}/analysis/summary_report.txt` (Requires Step 3 to run)

## Files

- Checkpoint: `{exp_dir}/checkpoint_best.pt`
- Config/Metrics: `{exp_dir}/final_results.json`

---

*Generated automatically by Phase 1 runner*
"""
        return md
    
    def run_all(self):
        """Run complete Phase 1 workflow."""
        start_time = datetime.now()
        
        steps = [
            ("Data Generation", self.step1_generate_data),
            ("Baseline Training", self.step2_train_baseline),
            ("Results Analysis", self.step3_analyze_results),
            ("Summary Generation", self.step4_generate_summary)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            success = step_func()
            results[step_name] = success
            
            if not success:
                print(f"\n❌ Phase 1 workflow stopped at: {step_name}")
                return False
        
        end_time = datetime.now()
        # Duration is in hours, assuming it's a long process.
        duration = (end_time - start_time).total_seconds() / 3600 
        
        print("\n" + "="*80)
        print("🎉 PHASE 1 COMPLETE!")
        print("="*80)
        print(f"Total time: {duration:.2f} hours")
        print(f"\nAll results saved to: {self.exp_dir}")
        print("\nNext: Review results and proceed to Phase 2")
        print("="*80 + "\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Run complete Phase 1 workflow")
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--quick", action="store_true",
                            help="Quick test mode (50 instances, 10 epochs)")
    mode_group.add_argument("--standard", action="store_true",
                            help="Standard mode (450 instances, 50 epochs)")
    mode_group.add_argument("--full", action="store_true",
                            help="Full mode (1000+ instances, 100 epochs)")
    
    # Other options
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.standard:
        mode = "standard"
    else:
        mode = "full"
    
    # Create runner and execute
    runner = Phase1Runner(mode=mode, device=args.device, seed=args.seed)
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
