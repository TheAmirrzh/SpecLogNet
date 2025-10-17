# run_phase1_enhanced.py
"""
Enhanced script to run Phase 1 with optimized loss, activation, optimizer.
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
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        
        self.base_dir = Path(".")
        self.data_dir = self.base_dir / "data_processed" / f"phase1_{mode}_{self.timestamp}"
        self.exp_dir = self.base_dir / "experiments" / f"phase1_{mode}_{self.timestamp}"
        
        print("=" * 80)
        print(f"PHASE 1 ENHANCED WORKFLOW - {mode.upper()} MODE")
        print("=" * 80)
        print(f"Description: {self.config['description']}")
        print(f"Data directory: {self.data_dir}")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {device}")
        print(f"Random seed: {seed}")
        print("=" * 80 + "\n")
    
    def run_command(self, cmd, step_name):
        print(f"\n{'='*80}")
        print(f"STEP: {step_name}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n✅ {step_name} completed successfully\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {step_name} failed with error code {e.returncode}")
            print(f"Error: {e}\n")
            return False
    
    def step1_generate_data(self):
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
        os.makedirs(self.exp_dir, exist_ok=True)
        
        exp_name = self.exp_dir.name
        log_dir = self.exp_dir.parent
        
        print(f"*** CHECKPOINT 1: Training directory created at absolute path: {self.exp_dir.resolve()}")
        
        cmd = [
            sys.executable, "-m", "src.train_phase1",
            "--json-dir", str(self.data_dir),
            "--spectral-dir", str(self.data_dir / "spectral"),
            "--exp-name", exp_name,
            "--log-dir", str(log_dir),
            "--hidden-dim", "128",
            "--num-layers", "3",
            "--dropout", "0.2",
            "--epochs", str(self.config["epochs"]),
            "--batch-size", "32",
            "--lr", "0.01",
            "--weight-decay", "1e-6",
            "--grad-clip", "2.0",
            "--patience", "15",
            "--device", self.device,
            "--seed", str(self.seed)
        ]
        return self.run_command(cmd, "Baseline Training")
    
    def step3_analyze_results(self):
        cmd = [
            sys.executable, "-m", "src.scripts.analyze_phase1_results",
            "--exp-dir", str(self.exp_dir)
        ]
        return self.run_command(cmd, "Results Analysis")
    
    def run_all(self):
        start_time = datetime.now()
        
        steps = [
            ("Data Generation", self.step1_generate_data),
            ("Baseline Training", self.step2_train_baseline),
            ("Results Analysis", self.step3_analyze_results)
        ]
        
        for step_name, step_func in steps:
            success = step_func()
            if not success:
                print(f"\n❌ Workflow stopped at: {step_name}")
                return False
        
        end_time = datetime.now()
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
    parser = argparse.ArgumentParser(description="Run enhanced Phase 1 workflow")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--quick", action="store_true")
    mode_group.add_argument("--standard", action="store_true")
    mode_group.add_argument("--full", action="store_true")
    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.quick:
        mode = "quick"
    elif args.standard:
        mode = "standard"
    else:
        mode = "full"
    
    runner = Phase1Runner(mode=mode, device=args.device, seed=args.seed)
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()