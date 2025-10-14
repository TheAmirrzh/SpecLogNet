# tests/test_horn_generator.py
import os
from src.parsers.horn_generator import generate_horn_instance, save_instance_json, load_instance_json

def test_generate_and_save(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    inst = generate_horn_instance("test1", seed=123, n_facts=5, n_rules=6, max_chain=3)
    assert "nodes" in inst and isinstance(inst["nodes"], list)
    assert "edges" in inst and isinstance(inst["edges"], list)
    # save and load
    p = out_dir / "t.json"
    save_instance_json(inst, str(p))
    loaded = load_instance_json(str(p))
    assert loaded["id"] == "test1"