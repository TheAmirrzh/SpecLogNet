# verify_fixes.py
import torch
from src.train_step_predictor import StepPredictionDataset
from torch_geometric.loader import DataLoader

# Load a small sample
json_files = ["data_processed/horn/horn_0.json"]
dataset = StepPredictionDataset(json_files, spectral_dir="data_processed/horn/spectral")

print(f"Dataset size: {len(dataset)}")
print(f"Sample features shape: {dataset[0].x.shape}")
print(f"Sample target: {dataset[0].y}")
print(f"Sample metadata: {dataset[0].meta}")

# Test batching
loader = DataLoader(dataset, batch_size=2, shuffle=False)
for batch in loader:
    print(f"\nBatch info:")
    print(f"  Total nodes: {batch.x.shape[0]}")
    print(f"  Batch vector: {batch.batch}")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Targets: {batch.y}")
    
    # Verify masking works
    for i in range(batch.num_graphs):
        mask = (batch.batch == i)
        print(f"  Graph {i}: {mask.sum().item()} nodes, target={batch.y[i].item()}")
    break