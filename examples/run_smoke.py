#!/usr/bin/env python3
"""Small deterministic smoke run for the ICDE03 disclosure engine."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from disclosure.model import Release, latent_count_model
from disclosure.compiled import Compiled
from disclosure.estimate import path
def main():
    model = latent_count_model(6, 1, seed=17)
    release = Release((2.4,), (1.0,))
    engine = Compiled(model, release)
    world = engine.sample_world(np.random.default_rng(1234))
    increments, diagnostics = path(engine, world, tuple(range(1, len(world))), "zero_one")
    result = {"domain_size": len(world), "target_posterior": engine.posterior_target().tolist(),
              "risk": float(engine.risk()), "path_increments": np.asarray(increments).tolist(),
              "diagnostic": bool(diagnostics)}
    print(json.dumps(result, sort_keys=True))
if __name__ == "__main__": main()
