# tests/test_tptp_parser.py
from src.parsers.tptp_parser import parse_tptp_clauses, clause_to_canonical, tptp_file_to_canonical
import os
import json

def test_parse_sample(tmp_path):
    sample = tmp_path / "sample.tptp"
    sample.write_text("fof(ax1, axiom, (! [X] : (p(X) => q(X)))).")
    clauses = parse_tptp_clauses(str(sample))
    assert len(clauses) >= 1
    res = clause_to_canonical(clauses[0], start_nid=0)
    assert "nodes" in res and "edges" in res