# src/parsers/tptp_parser.py
"""
Naive TPTP parser: extracts fof/fof-style clauses and turns them
into simple canonical graph elements.
This parser is intentionally conservative: it extracts predicate symbols
and their arguments, creating simple predicate and term nodes and edges.
For production use, replace with an official TPTP parser.
"""

import re
import json
from typing import List, Dict
import os

ATOM_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9_]*)\s*\(([^)]*)\)")
CLAUSE_END_RE = re.compile(r"\)\s*\.\s*$")


def parse_tptp_clauses(file_path: str) -> List[str]:
    """
    Read a TPTP file and return a list of clause strings (very basic).
    Strips comments (%) and joins lines until a trailing '.' is seen.
    """
    clauses = []
    buf = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            buf += " " + line
            if "." in line:
                # split on '.' but be careful: '.' may be inside other contexts; this is naive
                parts = buf.split(".")
                for p in parts[:-1]:
                    tok = p.strip()
                    if tok:
                        clauses.append(tok)
                buf = parts[-1]
    if buf.strip():
        clauses.append(buf.strip())
    return clauses


def clause_to_canonical(clause: str, start_nid: int = 0) -> Dict:
    """
    Convert one clause string into nodes and edges (naive).
    Returns: {"nodes": [...], "edges": [...], "next_nid": int}
    """
    nodes = []
    edges = []
    nid = start_nid
    # find all atoms like p(a,b)
    for m in ATOM_RE.finditer(clause):
        pred = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        pred_nid = nid
        nodes.append({"nid": pred_nid, "type": "predicate", "label": pred})
        nid += 1
        for arg in args:
            arg_nid = nid
            nodes.append({"nid": arg_nid, "type": "term", "label": arg})
            edges.append({"src": pred_nid, "dst": arg_nid, "etype": "has_arg"})
            nid += 1
    return {"nodes": nodes, "edges": edges, "next_nid": nid}


def tptp_file_to_canonical(file_path: str, out_path: str) -> Dict:
    clauses = parse_tptp_clauses(file_path)
    nodes = []
    edges = []
    nid = 0
    for i, c in enumerate(clauses):
        res = clause_to_canonical(c, start_nid=nid)
        nodes.extend(res["nodes"])
        edges.extend(res["edges"])
        nid = res["next_nid"]
    inst = {"id": os.path.basename(file_path), "nodes": nodes, "edges": edges, "proof_steps": [], "metadata": {"source": "tptp"}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inst, f, indent=2)
    return inst


if __name__ == "__main__":
    # demo: parse a sample file if provided
    sample = "tests/data/sample.tptp"
    import sys
    if os.path.exists(sample):
        print("Parsing sample.tptp -> data_processed/tptp_sample.json")
        tptp_file_to_canonical(sample, "data_processed/tptp_sample.json")
    else:
        print("Put a simple TPTP file at tests/data/sample.tptp to demo.")