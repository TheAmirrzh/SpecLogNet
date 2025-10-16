import os
import glob
import random
import json
from src.train_step_predictor import StepPredictionDataset

def inspect_dataset(json_dir, spectral_dir=None, seed=42, num_samples_to_check=10):
    all_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    random.seed(seed)
    random.shuffle(all_files)
    
    print(f"Total JSON files found: {len(all_files)}")
    
    # Load full dataset
    dataset = StepPredictionDataset(all_files, spectral_dir, seed)
    
    print(f"Valid samples (with at least one proof_step): {len(dataset)}")
    if len(dataset) == 0:
        print("WARNING: Empty dataset! No instances have usable proof steps. Check generator logs for short/failed proofs.")
        return
    
    # Aggregate stats from metadata (if available in JSON)
    proof_lengths = []
    graph_sizes = []  # (nodes, edges)
    difficulties = {}
    
    for file in all_files[:num_samples_to_check]:  # Sample a few raw JSONs for metadata
        with open(file, 'r') as f:
            inst = json.load(f)
        meta = inst.get("metadata", {})
        proof_lengths.append(meta.get("proof_length", 0))
        graph_sizes.append((meta.get("n_nodes", 0), meta.get("n_edges", 0)))
        diff = meta.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print("\nSample Metadata Stats (from first 10 JSONs):")
    print(f"  Avg Proof Length: {sum(proof_lengths) / len(proof_lengths):.2f}" if proof_lengths else "N/A")
    print(f"  Avg Graph Size (nodes/edges): {sum(n[0] for n in graph_sizes) / len(graph_sizes):.2f} / {sum(n[1] for n in graph_sizes) / len(graph_sizes):.2f}")
    print(f"  Difficulty Distribution: {difficulties}")
    
    # Check loaded PyG Data samples
    print("\nLoaded Sample Details (first 5):")
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        print(f"  Sample {i} (ID: {data.meta['instance_id']}, Step: {data.meta['step_idx']}):")
        print(f"    Num Nodes: {data.x.shape[0]}")
        print(f"    Num Edges: {data.edge_index.shape[1] // 2} (undirected)")
        print(f"    Feature Dim: {data.x.shape[1]} (expected: 2 base + spectral_k=16 if added)")
        print(f"    Target Rule Index (y): {data.y.item()} (should be 0 <= y < num_nodes)")
        if data.x.shape[1] == 2:
            print("    WARNING: No spectral features (dim=2). Check .npz files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect Phase 1 Dataset")
    parser.add_argument("--json-dir", required=True, help="Path to JSON dir")
    parser.add_argument("--spectral-dir", default=None, help="Path to spectral dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inspect_dataset(args.json_dir, args.spectral_dir, args.seed)