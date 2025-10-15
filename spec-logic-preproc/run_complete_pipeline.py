# run_complete_pipeline.py
"""
Complete training pipeline for hybrid spectral-spatial reasoning model.

Supports both:
1. Synthetic Horn clause data (Phase 1)
2. Real TPTP data (Phase 2)
3. Combined training (Phase 3)

Usage:
    # Phase 1: Synthetic only
    python run_complete_pipeline.py --mode synthetic --quick

    # Phase 2: TPTP only  
    python run_complete_pipeline.py --mode tptp --tptp-dir data_raw/tptp

    # Phase 3: Both datasets
    python run_complete_pipeline.py --mode combined
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


class CompletePipeline:
    """Orchestrates complete training pipeline across all datasets."""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Paths
        self.base_dir = Path(".")
        self.synthetic_data_dir = self.base_dir / "data_processed" / "phase1"
        self.tptp_raw_dir = Path(args.tptp_dir) if args.tptp_dir else None
        self.tptp_converted_dir = self.base_dir / "data_processed" / "tptp_converted"
        self.exp_dir = self.base_dir / "experiments" / f"complete_{args.mode}_{self.timestamp}"
        
        print("="*80)
        print(f"COMPLETE TRAINING PIPELINE - {args.mode.upper()} MODE")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Device: {args.device}")
        print("="*80 + "\n")
    
    def run_command(self, cmd, step_name):
        """Run command and handle errors."""
        print(f"\n{'='*80}")
        print(f"STEP: {step_name}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✅ {step_name} completed\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {step_name} failed: {e}\n")
            return False
    
    def phase1_synthetic(self):
        """Phase 1: Train on synthetic data."""
        print("\n" + "="*80)
        print("PHASE 1: SYNTHETIC HORN CLAUSES")
        print("="*80 + "\n")
        
        # Generate synthetic data
        if not self.synthetic_data_dir.exists():
            cmd = [
                sys.executable, "-m", "src.scripts.generate_phase1_data",
                "--out-dir", str(self.synthetic_data_dir),
                "--n-easy", str(self.args.n_easy),
                "--n-medium", str(self.args.n_medium),
                "--n-hard", str(self.args.n_hard),
                "--n-extreme", str(self.args.n_extreme),
                "--seed", str(self.args.seed),
                "--validate"
            ]
            
            if not self.run_command(cmd, "Synthetic Data Generation"):
                return False
        else:
            print(f"ℹ️  Using existing synthetic data at {self.synthetic_data_dir}\n")
        
        # Train on synthetic
        cmd = [
            sys.executable, "-m", "src.train_phase1",
            "--json-dir", str(self.synthetic_data_dir),
            "--spectral-dir", str(self.synthetic_data_dir / "spectral"),
            "--exp-name", f"synthetic_{self.timestamp}",
            "--log-dir", str(self.exp_dir.parent),
            "--epochs", str(self.args.epochs),
            "--device", self.args.device,
            "--seed", str(self.args.seed)
        ]
        
        if not self.run_command(cmd, "Synthetic Training"):
            return False
        
        # Analyze results
        exp_dirs = sorted(self.exp_dir.parent.glob(f"*synthetic_{self.timestamp}*"))
        if exp_dirs:
            cmd = [
                sys.executable, "-m", "src.scripts.analyze_phase1_results",
                "--exp-dir", str(exp_dirs[0])
            ]
            self.run_command(cmd, "Synthetic Analysis")
            
            # Load results
            with open(exp_dirs[0] / "final_results.json") as f:
                self.results["synthetic"] = json.load(f)
        
        return True
    
    def phase2_tptp(self):
        """Phase 2: Train on TPTP data."""
        print("\n" + "="*80)
        print("PHASE 2: TPTP REAL-WORLD DATA")
        print("="*80 + "\n")
        
        if not self.tptp_raw_dir or not self.tptp_raw_dir.exists():
            print(f"❌ TPTP directory not found: {self.tptp_raw_dir}")
            print("Please provide --tptp-dir with TPTP problems\n")
            return False
        
        # Convert TPTP to canonical format
        if not self.tptp_converted_dir.exists():
            cmd = [
                sys.executable, "-m", "src.parsers.tptp_parser_full",
                "--tptp-dir", str(self.tptp_raw_dir),
                "--output-dir", str(self.tptp_converted_dir),
                "--limit", str(self.args.tptp_limit) if self.args.tptp_limit else "1000"
            ]
            
            if not self.run_command(cmd, "TPTP Conversion"):
                return False
        else:
            print(f"ℹ️  Using existing converted TPTP data at {self.tptp_converted_dir}\n")
        
        # Train on TPTP
        cmd = [
            sys.executable, "-m", "src.train_tptp",
            "--json-dir", str(self.tptp_converted_dir),
            "--mode", "axiom_selection",
            "--epochs", str(self.args.epochs),
            "--device", self.args.device,
            "--seed", str(self.args.seed)
        ]
        
        if not self.run_command(cmd, "TPTP Training"):
            return False
        
        # Store results (TPTP uses different format)
        self.results["tptp"] = {
            "test_accuracy": "See train_tptp output",
            "mode": "axiom_selection"
        }
        
        return True
    
    def phase3_combined(self):
        """Phase 3: Combined training on both datasets."""
        print("\n" + "="*80)
        print("PHASE 3: COMBINED SYNTHETIC + TPTP")
        print("="*80 + "\n")
        
        # First train on synthetic
        if not self.phase1_synthetic():
            print("❌ Synthetic training failed, skipping combined")
            return False
        
        # Then evaluate on TPTP
        if not self.phase2_tptp():
            print("⚠️  TPTP training failed, but synthetic succeeded")
            return True  # Partial success
        
        return True
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        report_path = self.exp_dir / "PIPELINE_SUMMARY.md"
        
        lines = [
            "# Complete Pipeline Summary\n",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Mode**: {self.args.mode}",
            f"**Device**: {self.args.device}\n",
            "## Results\n"
        ]
        
        if "synthetic" in self.results:
            syn = self.results["synthetic"].get("test_metrics", {})
            lines.extend([
                "### Synthetic Data Results",
                f"- Test Hit@1: {syn.get('test_hit1', 0):.4f}",
                f"- Test Hit@3: {syn.get('test_hit3', 0):.4f}",
                f"- Test Hit@10: {syn.get('test_hit10', 0):.4f}",
                f"- Test MRR: {syn.get('test_mrr', 0):.4f}\n"
            ])
        
        if "tptp" in self.results:
            lines.extend([
                "### TPTP Data Results",
                "- See TPTP training output for accuracy",
                "- Mode: Axiom Selection\n"
            ])
        
        lines.extend([
            "## Files Generated\n",
            f"- Synthetic data: `{self.synthetic_data_dir}`",
            f"- TPTP data: `{self.tptp_converted_dir}`",
            f"- Experiments: `{self.exp_dir.parent}`",
            f"- Checkpoints: `checkpoints/`\n",
            "## Next Steps\n",
            "1. Review results in experiment directories",
            "2. Compare synthetic vs TPTP performance",
            "3. Document findings for paper Section 5",
            "4. Run ablation studies with spectral features\n"
        ])
        
        report_text = "\n".join(lines)
        
        os.makedirs(self.exp_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(report_text)
        print(f"\nFull report: {report_path}\n")
    
    def run(self):
        """Run the complete pipeline."""
        start_time = datetime.now()
        
        success = False
        
        if self.args.mode == "synthetic":
            success = self.phase1_synthetic()
        elif self.args.mode == "tptp":
            success = self.phase2_tptp()
        elif self.args.mode == "combined":
            success = self.phase3_combined()
        else:
            print(f"❌ Unknown mode: {self.args.mode}")
            return False
        
        if success:
            self.generate_final_report()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        print(f"\n{'='*80}")
        print(f"Total time: {duration:.2f} hours")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete training pipeline")
    parser.add_argument("--mode", choices=["synthetic", "tptp", "combined"], required=True, help="Mode to run")
    parser.add_argument("--tptp-dir", help="Path to TPTP directory")
    parser.add_argument("--tptp-limit", type=int, help="Limit number of TPTP files")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--device", default="cpu", help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    pipeline = CompletePipeline(args)
    pipeline.run()