"""
Pre-processes all graph data to extract unique atoms and generates
high-dimensional semantic embeddings for them using SentenceTransformers.
"""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_unique_atoms(data_dir: Path) -> set:
    """Find every unique atom string in the entire dataset."""
    atom_set = set()
    json_files = list(data_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} graph files to scan...")

    for file_path in tqdm(json_files, desc="Scanning for atoms"):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                nodes = data.get('nodes', [])
                for node in nodes:
                    if node.get('type') == 'fact':
                        atom_set.add(node.get('atom', ''))
                    elif node.get('type') == 'rule':
                        atom_set.add(node.get('head_atom', ''))
                        for body_atom in node.get('body_atoms', []):
                            atom_set.add(body_atom)
            except Exception as e:
                logger.warning(f"Could not parse {file_path}: {e}")

    # Remove any empty strings
    atom_set.discard('')
    return atom_set

def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic embeddings for all atoms in the dataset."
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, 
        help="Directory containing the generated graph JSONs."
    )
    parser.add_argument(
        "--output-file", type=str, required=True, 
        help="Path to save the output atom_embeddings.json file."
    )
    parser.add_argument(
        "--model-name", type=str, default="all-MiniLM-L6-v2", 
        help="SentenceTransformer model to use (e.g., all-MiniLM-L6-v2)."
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output_file)

    # 1. Extract all unique atoms
    unique_atoms = extract_unique_atoms(data_dir)
    atom_list = sorted(list(unique_atoms))
    logger.info(f"Found {len(atom_list)} unique atoms.")

    if not atom_list:
        logger.error("No atoms found. Exiting.")
        return

    # 2. Load pre-trained model
    logger.info(f"Loading SentenceTransformer model: {args.model_name}...")
    model = SentenceTransformer(args.model_name)

    # 3. Encode all atoms in a batch
    logger.info("Encoding atoms...")
    atom_embeddings = model.encode(
        atom_list, 
        show_progress_bar=True, 
        batch_size=128
    )

    # 4. Create the mapping dictionary
    embedding_map = {}
    for atom, embedding in zip(atom_list, atom_embeddings):
        embedding_map[atom] = embedding.tolist()

    # 5. Save to JSON
    logger.info(f"Saving {len(embedding_map)} embeddings to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(embedding_map, f)

    logger.info("✅ Semantic embedding preprocessing complete.")

if __name__ == "__main__":
    main()