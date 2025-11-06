#!/usr/bin/env python3
"""
Diagnostic script to test data_generator.py independently.

This script will:
1. Import the generation function from your data_generator.py.
2. Attempt to generate 2 'easy' and 2 'medium' instances.
3. Report SUCCESS or FAILURE for each attempt.
4. Save successful files to a new 'test_generator_output' directory
   so you can inspect them.
"""

import json
import logging
import shutil
from pathlib import Path
from data_generator import Difficulty, generate_horn_instance_deterministic

# --- Configuration ---
OUTPUT_DIR = Path("./test_generator_output")
TEST_CONFIG = {
    Difficulty.EASY: 2,
    Difficulty.MEDIUM: 2,
    Difficulty.HARD: 1
}
SEED = 42

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info(f"🔬 STARTING DATA GENERATOR TEST")
    logger.info(f"Output will be saved to: {OUTPUT_DIR}")
    logger.info("="*80)

    # Clean up old test runs
    if OUTPUT_DIR.exists():
        logger.warning(f"Removing old test directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    total_generated = 0
    total_failed = 0

    # Loop through the test config
    for difficulty, count in TEST_CONFIG.items():
        logger.info(f"\n--- Testing Difficulty: {difficulty.value} (Target: {count} instances) ---")
        
        for i in range(count):
            instance_id = f"{difficulty.value}_{i}"
            logger.info(f"Attempting to generate: {instance_id}")

            try:
                # --- This is the function we are testing ---
                inst = generate_horn_instance_deterministic(
                    instance_id=instance_id,
                    difficulty=difficulty,
                    seed=SEED + i + hash(difficulty.value)
                )
                
                if inst is None:
                    logger.error(f"❌ FAILURE: Generator returned None for {instance_id}")
                    total_failed += 1
                    continue

                # --- This is the check that's failing in run.sh ---
                proof_steps = inst.get("proof_steps", [])
                
                if len(proof_steps) == 0:
                    logger.error(f"❌ FAILURE: Instance {instance_id} generated WITH 0 PROOF STEPS.")
                    total_failed += 1
                    
                    # Print diagnostic info for the failed instance
                    n_nodes = len(inst.get('nodes', []))
                    n_rules = sum(1 for n in inst.get('nodes', []) if n.get('type') == 'rule')
                    n_initial = sum(1 for n in inst.get('nodes', []) if n.get('is_initial') == True)
                    goal = inst.get('goal', 'NoGoalSpecified')
                    
                    logger.error(f"   ┣ Goal: {goal}")
                    logger.error(f"   ┣ Total Nodes: {n_nodes}")
                    logger.error(f"   ┣ Total Rules: {n_rules}")
                    logger.error(f"   ┗ Initial Facts: {n_initial}")
                    logger.error(f"   This indicates 'backward_chain' failed to find a proof.")

                else:
                    logger.info(f"✅ SUCCESS: Instance {instance_id} generated with {len(proof_steps)} steps.")
                    total_generated += 1
                    
                    # Save the successful file for inspection
                    file_path = OUTPUT_DIR / f"{instance_id}.json"
                    with open(file_path, 'w') as f:
                        json.dump(inst, f, indent=2)

            except Exception as e:
                logger.critical(f"💥 CRITICAL ERROR during generation of {instance_id}: {e}", exc_info=True)
                total_failed += 1

    logger.info("\n" + "="*80)
    logger.info("🔬 TEST COMPLETE")
    logger.info(f"   Total Successful: {total_generated}")
    logger.info(f"   Total Failed: {total_failed}")
    logger.info("="*80)

if __name__ == "__main__":
    main()