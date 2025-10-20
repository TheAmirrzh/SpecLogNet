# File: debug_dataset.py
from dataset import StepPredictionDataset
from pathlib import Path

# 1. Get a list of any of your JSON files
json_dir = "data/horn" 
all_files = list(Path(json_dir).rglob("*.json"))

if not all_files:
    print("Error: No JSON files found in data/horn. Run data_generator.py first.")
else:
    # 2. Create a dataset with just ONE file
    test_ds = StepPredictionDataset(
        json_files=[str(all_files[0])], 
        spectral_dir=None
    )
    
    if len(test_ds) == 0:
        print("Error: Dataset created, but no samples found. (File might have no proof steps).")
    else:
        # 3. Load the very first sample
        first_sample = test_ds[0]
        
        # 4. This is the "True Issue" test:
        print("\n" + "="*30)
        print("  DATASET DEBUGGER")
        print("="*30)
        print(f"Loaded sample from: {all_files[0].name}")
        print(f"Number of nodes in graph: {first_sample.num_nodes}")
        
        # This is the most important line:
        print(f"FEATURE DIMENSION (x.shape): {first_sample.x.shape}")
        
        print("\n" + "="*30)
        print("\n >> If dimension is 24, the data leak is STILL active.")
        print(" >> If dimension is 22, the leak is fixed.")