# diagnose_dataset.py
"""
Diagnostic script to inspect StepPredictionDataset.
Prints stats on samples, proof lengths, graph sizes, features, and labels.
"""

import os
import glob
import random
import json
import argparse
import numpy as np
from src.train_step_predictor_fixed import StepPredictionDataset

def diagnose_dataset(json_dir, spectral_dir=None, seed=42, num_samples=10):
    all_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    random.seed(seed)
    random.shuffle(all_files)
    
    print(f"Total JSON files: {len(all_files)}")
    
    dataset = StepPredictionDataset(all_files, spectral_dir, seed)
    
    print(f"Valid samples: {len(dataset)}")
    if len(dataset) == 0:
        print("CRITICAL ISSUE: No valid samples! Check generator for proof_steps.")
        return
    
    # Stats
    proof_lengths = []
    node_counts = []
    edge_counts = []
    feature_dims = []
    y_values = []
    
    for i in range(min(num_samples, len(dataset))):
        data = dataset[i]
        inst = dataset.samples[i][0]  # Access instance
        proof_lengths.append(len(inst.get("proof_steps", [])))
        node_counts.append(data.x.shape[0])
        edge_counts.append(data.edge_index.shape[1] // 2)  # Undirected
        feature_dims.append(data.x.shape[1])
        y_values.append(int(data.y.item()))
        
        print(f"\nSample {i} ({data.meta['instance_id']}, Step {data.meta['step_idx']}):")
        print(f"  Proof Length: {proof_lengths[-1]}")
        print(f"  Nodes/Edges: {node_counts[-1]}/{edge_counts[-1]}")
        print(f"  Feature Dim: {feature_dims[-1]} (Spectral: {'Yes' if feature_dims[-1] > 2 else 'No'})")
        print(f"  Target y: {y_values[-1]} (Valid: {0 <= y_values[-1] < node_counts[-1]})")
    
    # Aggregate
    print("\nAggregate Stats (from samples):")
    print(f"  Avg Proof Length: {np.mean(proof_lengths):.2f} (Min: {np.min(proof_lengths)}, Max: {np.max(proof_lengths)})")
    print(f"  Avg Nodes: {np.mean(node_counts):.2f}")
    print(f"  Avg Edges: {np.mean(edge_counts):.2f}")
    print(f"  Unique Feature Dims: {set(feature_dims)}")
    print(f"  y Distribution: Min {np.min(y_values)}, Max {np.max(y_values)}, Avg {np.mean(y_values):.2f}")
    
    # Critical Issues Check
    issues = []
    if np.mean(proof_lengths) < 3:
        issues.append("Low proof lengths - Increase max_chain in generator")
    if len(set(feature_dims)) > 1 or list(set(feature_dims))[0] == 2:
        issues.append("Inconsistent/Missing spectral features - Check .npz files")
    if any(y >= n for y, n in zip(y_values, node_counts)):
        issues.append("Invalid y indices - Bug in id2idx mapping")
    
    if issues:
        print("\nCRITICAL ISSUES:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo critical issues detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Dataset")
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--spectral-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    diagnose_dataset(args.json_dir, args.spectral_dir, args.seed)