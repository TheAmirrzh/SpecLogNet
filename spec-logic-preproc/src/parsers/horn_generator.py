"""
Horn clause synthetic generator.
Generates canonical graph dicts with:
- nodes: facts and rule nodes
- edges: fact -> rule (body), rule -> fact (head)
- proof_steps: forward-chaining derivations
"""

import json
import random
import os
from typing import Dict, List, Tuple


def _next_nid(nid_counter: Dict[str, int]) -> int:
    nid = nid_counter["v"]
    nid_counter["v"] += 1
    return nid


def generate_horn_instance(
    instance_id: str,
    seed: int = None,
    n_facts: int = 6,
    n_rules: int = 8,
    max_chain: int = 4,
    atoms_pool: List[str] = None,
) -> Dict:
    """
    Generate a Horn-clause style canonical instance.

    Return format:
    {
      "id": str,
      "nodes": [{"nid": int, "type": "fact|rule", "label": str}],
      "edges": [{"src": int, "dst": int, "etype": "body|head"}],
      "proof_steps": [{"step_id": int, "derived_node": nid, "used_rule": rule_nid, "premises": [nids]}],
      "metadata": {...}
    }
    """
    if seed is not None:
        random.seed(seed)
    if atoms_pool is None:
        atoms_pool = [f"P{i}" for i in range(20)]

    nid_counter = {"v": 0}
    nodes: List[Dict] = []
    edges: List[Dict] = []
    proof_steps: List[Dict] = []

    # create initial fact nodes
    fact_nids = []
    for i in range(n_facts):
        n = {"nid": _next_nid(nid_counter), "type": "fact", "label": random.choice(atoms_pool)}
        nodes.append(n)
        fact_nids.append(n["nid"])

    # create rules
    rule_nids = []
    for r in range(n_rules):
        # randomly choose body size 1 or 2
        body_size = random.choice([1, 2])
        body_atoms = random.choices(atoms_pool, k=body_size)
        head_atom = random.choice(atoms_pool)
        label = f"{'+'.join(body_atoms)} -> {head_atom}"
        rule_nid = _next_nid(nid_counter)
        rule_nids.append(rule_nid)
        nodes.append({"nid": rule_nid, "type": "rule", "label": label, "body_atoms": body_atoms, "head_atom": head_atom})
        # connect body atoms: create intermediate fact nodes for those atoms if not present
        # for simplicity, create unique fact nodes for body atoms if they don't share label
        body_nids = []
        for a in body_atoms:
            # try to find an existing fact with same label to increase connectivity
            found = next((f for f in nodes if f["type"] == "fact" and f["label"] == a), None)
            if found:
                b_nid = found["nid"]
            else:
                b_nid = _next_nid(nid_counter)
                nodes.append({"nid": b_nid, "type": "fact", "label": a})
            body_nids.append(b_nid)
            edges.append({"src": b_nid, "dst": rule_nid, "etype": "body"})
        # create head fact node (may exist or new)
        found_head = next((f for f in nodes if f["type"] == "fact" and f["label"] == head_atom), None)
        if found_head:
            head_nid = found_head["nid"]
        else:
            head_nid = _next_nid(nid_counter)
            nodes.append({"nid": head_nid, "type": "fact", "label": head_atom})
        edges.append({"src": rule_nid, "dst": head_nid, "etype": "head"})

    # now build a forward-chaining proof chain: repeatedly apply rules when body facts exist but head not yet derived
    derived = set(fact_nids)
    rule_applied = True
    step_id = 0
    iterations = 0
    while rule_applied and iterations < 1000 and step_id < max_chain:
        rule_applied = False
        iterations += 1
        for rule in [n for n in nodes if n["type"] == "rule"]:
            # find body nids by scanning edges
            body_edges = [e for e in edges if e["dst"] == rule["nid"] and e["etype"] == "body"]
            body_nids = [e["src"] for e in body_edges]
            head_edges = [e for e in edges if e["src"] == rule["nid"] and e["etype"] == "head"]
            if not head_edges:
                continue
            head_nid = head_edges[0]["dst"]
            # if all body facts are present (in derived) and head not in derived, apply rule
            if all(b in derived for b in body_nids) and head_nid not in derived:
                # derive
                derived.add(head_nid)
                proof_steps.append(
                    {
                        "step_id": step_id,
                        "derived_node": int(head_nid),
                        "used_rule": int(rule["nid"]),
                        "premises": [int(b) for b in body_nids],
                    }
                )
                step_id += 1
                rule_applied = True
                # stop if reached chain length
                if step_id >= max_chain:
                    break
        # loop continues to find new derivations
    # add metadata
    instance = {
        "id": instance_id,
        "nodes": nodes,
        "edges": edges,
        "proof_steps": proof_steps,
        "metadata": {"source": "synthetic_horn", "n_facts": n_facts, "n_rules": n_rules},
    }
    return instance


def save_instance_json(instance: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(instance, f, indent=2)


def load_instance_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # simple run: generate and save 3 instances to local folder
    for i in range(3):
        inst = generate_horn_instance(f"horn_{i}", seed=42 + i, n_facts=6, n_rules=8, max_chain=4)
        save_instance_json(inst, f"data_processed/horn/horn_{i}.json")
    print("Generated sample horn instances in data_processed/horn/")
