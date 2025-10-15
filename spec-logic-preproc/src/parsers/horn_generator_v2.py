# src/parsers/horn_generator_v2.py
"""
Enhanced Horn clause generator with difficulty stratification.
Generates problems of varying complexity for curriculum learning.
"""

import json
import random
import os
from typing import Dict, List, Tuple
from enum import Enum


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


def _next_nid(nid_counter: Dict[str, int]) -> int:
    nid = nid_counter["v"]
    nid_counter["v"] += 1
    return nid


def generate_stratified_horn_instance(
    instance_id: str,
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: int = None,
) -> Dict:
    """
    Generate Horn clause instance with controlled difficulty.
    
    Difficulty parameters:
    - EASY: 4-6 facts, 4-6 rules, chain depth 2-3, body size 1
    - MEDIUM: 6-10 facts, 8-12 rules, chain depth 3-5, body size 1-2
    - HARD: 10-15 facts, 12-20 rules, chain depth 5-8, body size 2-3
    - EXTREME: 15-25 facts, 20-35 rules, chain depth 8-12, body size 2-4
    """
    
    if seed is not None:
        random.seed(seed)
    
    # Set parameters based on difficulty
    params = {
        Difficulty.EASY: {
            "n_facts": random.randint(4, 6),
            "n_rules": random.randint(4, 6),
            "max_chain": random.randint(2, 3),
            "body_size_range": (1, 1),
            "atoms_pool_size": 8,
            "negation_prob": 0.0,
            "disjunction_prob": 0.0,
        },
        Difficulty.MEDIUM: {
            "n_facts": random.randint(6, 10),
            "n_rules": random.randint(8, 12),
            "max_chain": random.randint(3, 5),
            "body_size_range": (1, 2),
            "atoms_pool_size": 15,
            "negation_prob": 0.1,
            "disjunction_prob": 0.0,
        },
        Difficulty.HARD: {
            "n_facts": random.randint(10, 15),
            "n_rules": random.randint(12, 20),
            "max_chain": random.randint(5, 8),
            "body_size_range": (2, 3),
            "atoms_pool_size": 25,
            "negation_prob": 0.2,
            "disjunction_prob": 0.1,
        },
        Difficulty.EXTREME: {
            "n_facts": random.randint(15, 25),
            "n_rules": random.randint(20, 35),
            "max_chain": random.randint(8, 12),
            "body_size_range": (2, 4),
            "atoms_pool_size": 40,
            "negation_prob": 0.3,
            "disjunction_prob": 0.2,
        }
    }
    
    p = params[difficulty]
    atoms_pool = [f"P{i}" for i in range(p["atoms_pool_size"])]
    
    nid_counter = {"v": 0}
    nodes: List[Dict] = []
    edges: List[Dict] = []
    proof_steps: List[Dict] = []
    
    # Create initial facts
    fact_nids = []
    for i in range(p["n_facts"]):
        atom = random.choice(atoms_pool)
        is_negated = random.random() < p["negation_prob"]
        label = f"~{atom}" if is_negated else atom
        
        n = {
            "nid": _next_nid(nid_counter),
            "type": "fact",
            "label": label,
            "atom": atom,
            "negated": is_negated
        }
        nodes.append(n)
        fact_nids.append(n["nid"])
    
    # Create rules with varying complexity
    rule_nids = []
    for r in range(p["n_rules"]):
        body_size = random.randint(*p["body_size_range"])
        body_atoms = random.choices(atoms_pool, k=body_size)
        head_atom = random.choice(atoms_pool)
        
        # Add negations to body
        body_negations = [random.random() < p["negation_prob"] for _ in body_atoms]
        body_labels = [f"~{a}" if neg else a for a, neg in zip(body_atoms, body_negations)]
        
        # Add disjunction (A + B means A OR B in the body)
        if random.random() < p["disjunction_prob"] and body_size > 1:
            body_str = " | ".join(body_labels)  # disjunctive body
            rule_type = "disjunctive"
        else:
            body_str = " & ".join(body_labels)  # conjunctive body
            rule_type = "conjunctive"
        
        label = f"({body_str}) -> {head_atom}"
        
        rule_nid = _next_nid(nid_counter)
        rule_nids.append(rule_nid)
        
        rule_node = {
            "nid": rule_nid,
            "type": "rule",
            "label": label,
            "body_atoms": body_atoms,
            "body_negations": body_negations,
            "head_atom": head_atom,
            "rule_type": rule_type
        }
        nodes.append(rule_node)
        
        # Connect body atoms to rule
        body_nids = []
        for atom, is_neg in zip(body_atoms, body_negations):
            label_str = f"~{atom}" if is_neg else atom
            # Find or create fact node
            found = next((n for n in nodes if n["type"] == "fact" and n["label"] == label_str), None)
            if found:
                b_nid = found["nid"]
            else:
                b_nid = _next_nid(nid_counter)
                nodes.append({
                    "nid": b_nid,
                    "type": "fact",
                    "label": label_str,
                    "atom": atom,
                    "negated": is_neg
                })
            body_nids.append(b_nid)
            edges.append({"src": b_nid, "dst": rule_nid, "etype": "body"})
        
        # Connect rule to head fact
        found_head = next((n for n in nodes if n["type"] == "fact" and n["label"] == head_atom), None)
        if found_head:
            head_nid = found_head["nid"]
        else:
            head_nid = _next_nid(nid_counter)
            nodes.append({
                "nid": head_nid,
                "type": "fact",
                "label": head_atom,
                "atom": head_atom,
                "negated": False
            })
        edges.append({"src": rule_nid, "dst": head_nid, "etype": "head"})
    
    # Forward-chaining proof generation
    derived = set(fact_nids)
    rule_applied = True
    step_id = 0
    iterations = 0
    max_iterations = p["max_chain"] * 100
    
    while rule_applied and iterations < max_iterations and step_id < p["max_chain"]:
        rule_applied = False
        iterations += 1
        
        for rule in [n for n in nodes if n["type"] == "rule"]:
            # For conjunctive rules: all body facts must be derived
            # For disjunctive rules: at least one body fact must be derived
            body_edges = [e for e in edges if e["dst"] == rule["nid"] and e["etype"] == "body"]
            body_nids = [e["src"] for e in body_edges]
            head_edges = [e for e in edges if e["src"] == rule["nid"] and e["etype"] == "head"]
            
            if not head_edges:
                continue
            
            head_nid = head_edges[0]["dst"]
            
            # Check if rule can fire
            can_apply = False
            if rule.get("rule_type") == "disjunctive":
                can_apply = any(b in derived for b in body_nids) and head_nid not in derived
            else:  # conjunctive
                can_apply = all(b in derived for b in body_nids) and head_nid not in derived
            
            if can_apply:
                derived.add(head_nid)
                proof_steps.append({
                    "step_id": step_id,
                    "derived_node": int(head_nid),
                    "used_rule": int(rule["nid"]),
                    "premises": [int(b) for b in body_nids],
                    "rule_type": rule.get("rule_type", "conjunctive")
                })
                step_id += 1
                rule_applied = True
                
                if step_id >= p["max_chain"]:
                    break
    
    # Compute difficulty metrics
    avg_rule_body_size = sum(len([e for e in edges if e["dst"] == r["nid"] and e["etype"] == "body"]) 
                              for r in nodes if r["type"] == "rule") / max(1, len(rule_nids))
    
    graph_density = len(edges) / max(1, len(nodes) * (len(nodes) - 1) / 2)
    
    instance = {
        "id": instance_id,
        "nodes": nodes,
        "edges": edges,
        "proof_steps": proof_steps,
        "metadata": {
            "source": "synthetic_horn_v2",
            "difficulty": difficulty.value,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_facts": len([n for n in nodes if n["type"] == "fact"]),
            "n_rules": len(rule_nids),
            "proof_length": len(proof_steps),
            "avg_body_size": avg_rule_body_size,
            "graph_density": graph_density,
            "has_negation": p["negation_prob"] > 0,
            "has_disjunction": p["disjunction_prob"] > 0,
        }
    }
    
    return instance


def generate_stratified_dataset(
    out_dir: str,
    n_per_difficulty: int = 100,
    seed: int = 42
):
    """Generate a complete stratified dataset."""
    os.makedirs(out_dir, exist_ok=True)
    
    stats = {"total": 0, "by_difficulty": {}}
    
    for difficulty in Difficulty:
        diff_dir = os.path.join(out_dir, difficulty.value)
        os.makedirs(diff_dir, exist_ok=True)
        
        count = 0
        for i in range(n_per_difficulty):
            inst = generate_stratified_horn_instance(
                f"{difficulty.value}_{i}",
                difficulty=difficulty,
                seed=seed + i
            )
            
            out_path = os.path.join(diff_dir, f"{difficulty.value}_{i}.json")
            with open(out_path, "w") as f:
                json.dump(inst, f, indent=2)
            
            count += 1
        
        stats["by_difficulty"][difficulty.value] = count
        stats["total"] += count
        print(f"Generated {count} {difficulty.value} instances")
    
    # Save dataset statistics
    stats_path = os.path.join(out_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nTotal instances generated: {stats['total']}")
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data_processed/horn_stratified", help="Output directory")
    parser.add_argument("--n-per-difficulty", type=int, default=100, help="Instances per difficulty level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generate_stratified_dataset(args.out_dir, args.n_per_difficulty, args.seed)