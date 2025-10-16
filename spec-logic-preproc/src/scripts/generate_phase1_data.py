# src/scripts/generate_phase1_data.py
"""
Generate complete Phase 1 dataset with:
- Stratified difficulty levels
- Spectral features pre-computation
- Dataset statistics and validation
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parsers.horn_generator_v2 import generate_stratified_horn_instance, Difficulty
from src.spectral import adjacency_from_edges, compute_symmetric_laplacian, topk_eig


def generate_dataset_with_spectral(
    out_dir: str,
    n_easy: int = 150,
    n_medium: int = 150,
    n_hard: int = 100,
    n_extreme: int = 50,
    spectral_k: int = 16,
    seed: int = 42
):
    """
    Generate complete Phase 1 dataset with pre-computed spectral features.
    
    Dataset structure:
    out_dir/
        easy/
            easy_0.json
            easy_1.json
            ...
        medium/
            medium_0.json
            ...
        hard/
            hard_0.json
            ...
        extreme/
            extreme_0.json
            ...
        spectral/
            easy_0_spectral_k16.npz
            ...
        stats/
            dataset_info.json
            difficulty_distribution.json
    """
    
    os.makedirs(out_dir, exist_ok=True)
    spectral_dir = os.path.join(out_dir, "spectral")
    stats_dir = os.path.join(out_dir, "stats")
    os.makedirs(spectral_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Track statistics
    all_stats = {
        "total_instances": 0,
        "by_difficulty": {},
        "overall_stats": {
            "avg_nodes": [],
            "avg_edges": [],
            "avg_proof_length": [],
            "avg_graph_density": [],
        }
    }
    
    difficulty_counts = {
        Difficulty.EASY: n_easy,
        Difficulty.MEDIUM: n_medium,
        Difficulty.HARD: n_hard,
        Difficulty.EXTREME: n_extreme
    }
    
    print("=" * 80)
    print("PHASE 1 DATASET GENERATION")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Spectral features: k={spectral_k}")
    print(f"Random seed: {seed}")
    print()
    
    for difficulty, count in difficulty_counts.items():
        print(f"\nGenerating {difficulty.value.upper()} instances: {count}")
        print("-" * 60)
        
        diff_dir = os.path.join(out_dir, difficulty.value)
        os.makedirs(diff_dir, exist_ok=True)
        
        diff_stats = {
            "count": count,
            "instances": [],
            "avg_nodes": 0,
            "avg_edges": 0,
            "avg_proof_length": 0,
            "successful_proofs": 0
        }
        
        for i in tqdm(range(count), desc=f"{difficulty.value}"):
            instance_id = f"{difficulty.value}_{i}"
            
            # Generate instance
            inst = generate_stratified_horn_instance(
                instance_id=instance_id,
                difficulty=difficulty,
                seed=seed + i + hash(difficulty.value) % 10000
            )
            
            # Save instance
            inst_path = os.path.join(diff_dir, f"{instance_id}.json")
            with open(inst_path, "w") as f:
                json.dump(inst, f, indent=2)
            
            # Compute and save spectral features
            try:
                n_nodes = len(inst["nodes"])
                A = adjacency_from_edges(n_nodes, inst["edges"])
                L = compute_symmetric_laplacian(A)
                k = min(spectral_k, n_nodes - 1)
                
                if k > 0:
                    eigvals, eigvecs = topk_eig(L, k=k)
                    
                    # Pad if necessary
                    if eigvecs.shape[1] < spectral_k:
                        pad_width = spectral_k - eigvecs.shape[1]
                        eigvecs = np.pad(eigvecs, ((0, 0), (0, pad_width)), mode='constant')
                        eigvals = np.pad(eigvals, (0, pad_width), mode='constant')
                    
                    spectral_path = os.path.join(spectral_dir, f"{instance_id}_spectral_k{spectral_k}.npz")
                    np.savez_compressed(
                        spectral_path,
                        eigvals=eigvals,
                        eigvecs=eigvecs,
                        n_nodes=n_nodes
                    )
            except Exception as e:
                print(f"Warning: Failed to compute spectral features for {instance_id}: {e}")
            
            # Collect statistics
            meta = inst["metadata"]
            diff_stats["instances"].append({
                "id": instance_id,
                "n_nodes": meta["n_nodes"],
                "n_edges": meta["n_edges"],
                "proof_length": meta["proof_length"]
            })
            
            diff_stats["avg_nodes"] += meta["n_nodes"]
            diff_stats["avg_edges"] += meta["n_edges"]
            diff_stats["avg_proof_length"] += meta["proof_length"]
            if meta["proof_length"] > 0:
                diff_stats["successful_proofs"] += 1
            
            all_stats["overall_stats"]["avg_nodes"].append(meta["n_nodes"])
            all_stats["overall_stats"]["avg_edges"].append(meta["n_edges"])
            all_stats["overall_stats"]["avg_proof_length"].append(meta["proof_length"])
            all_stats["overall_stats"]["avg_graph_density"].append(meta["graph_density"])
        
        # Finalize difficulty stats
        if count > 0:  # Avoid ZeroDivisionError
            diff_stats["avg_nodes"] /= count
            diff_stats["avg_edges"] /= count
            diff_stats["avg_proof_length"] /= count
            diff_stats["success_rate"] = diff_stats["successful_proofs"] / count
        
        all_stats["by_difficulty"][difficulty.value] = diff_stats
        all_stats["total_instances"] += count
        
        # --- defensive stats printing: handle empty/missing stats safely ---
        if isinstance(diff_stats, dict):
            avg_nodes = diff_stats.get("avg_nodes", 0.0)
            avg_edges = diff_stats.get("avg_edges", 0.0)
            avg_proof_length = diff_stats.get("avg_proof_length", 0.0)
            success_rate = diff_stats.get("success_rate", 0.0)
        else:
            avg_nodes = avg_edges = avg_proof_length = success_rate = 0.0

        print(f"  Avg nodes: {avg_nodes:.1f}")
        print(f"  Avg edges: {avg_edges:.1f}")
        print(f"  Avg proof length: {avg_proof_length:.1f}")
        print(f"  Success rate: {success_rate:.2%}")
        # --- end defensive printing ---`

    # Compute overall statistics
    total_samples = len(all_stats["overall_stats"]["avg_nodes"])
    if total_samples > 0:
        for key in ["avg_nodes", "avg_edges", "avg_proof_length", "avg_graph_density"]:
            values = all_stats["overall_stats"][key]
            all_stats["overall_stats"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
    
    # Save statistics
    stats_path = os.path.join(stats_dir, "dataset_info.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total instances: {all_stats['total_instances']}")
    print(f"Statistics saved to: {stats_path}")
    print(f"Dataset ready for training!")
    print()
    
    return all_stats


def validate_dataset(data_dir: str):
    """Validate generated dataset."""
    print("\n" + "=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    
    issues = []
    
    # Check directory structure
    required_dirs = ["easy", "medium", "hard", "extreme", "spectral", "stats"]
    for d in required_dirs:
        path = os.path.join(data_dir, d)
        if not os.path.exists(path):
            issues.append(f"Missing directory: {d}")
    
    # Check JSON files
    for difficulty in ["easy", "medium", "hard", "extreme"]:
        diff_dir = os.path.join(data_dir, difficulty)
        if not os.path.exists(diff_dir):
            continue
        
        json_files = list(Path(diff_dir).glob("*.json"))
        print(f"\n{difficulty.upper()}: {len(json_files)} instances")
        
        # Sample validation
        for json_file in json_files[:5]:  # Check first 5
            try:
                with open(json_file) as f:
                    inst = json.load(f)
                
                # Validate required fields
                required_fields = ["id", "nodes", "edges", "proof_steps", "metadata"]
                for field in required_fields:
                    if field not in inst:
                        issues.append(f"{json_file.name}: missing field '{field}'")
                
                # Check spectral file exists
                spectral_file = os.path.join(data_dir, "spectral", f"{inst['id']}_spectral_k16.npz")
                if not os.path.exists(spectral_file):
                    issues.append(f"{json_file.name}: missing spectral features")
                
            except Exception as e:
                issues.append(f"{json_file.name}: validation error - {e}")
    
    # Report
    if issues:
        print("\n⚠️  VALIDATION ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Dataset validation passed!")
    
    return len(issues) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 1 Dataset")
    parser.add_argument("--out-dir", type=str, default="data_processed/phase1",
                        help="Output directory")
    parser.add_argument("--n-easy", type=int, default=150,
                        help="Number of easy instances")
    parser.add_argument("--n-medium", type=int, default=150,
                        help="Number of medium instances")
    parser.add_argument("--n-hard", type=int, default=100,
                        help="Number of hard instances")
    parser.add_argument("--n-extreme", type=int, default=50,
                        help="Number of extreme instances")
    parser.add_argument("--spectral-k", type=int, default=16,
                        help="Number of eigenvectors to compute")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--validate", action="store_true",
                        help="Validate dataset after generation")
    
    args = parser.parse_args()
    
    # Generate dataset
    stats = generate_dataset_with_spectral(
        out_dir=args.out_dir,
        n_easy=args.n_easy,
        n_medium=args.n_medium,
        n_hard=args.n_hard,
        n_extreme=args.n_extreme,
        spectral_k=args.spectral_k,
        seed=args.seed
    )
    
    # Validate if requested
    if args.validate:
        validate_dataset(args.out_dir)