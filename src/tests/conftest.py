import os
import sys
import random
from pathlib import Path

import numpy as np
import pytest

# Make src/main importable before tests are collected
here = Path(__file__).resolve()
repo_root = here.parents[2]
# Ensure `src` is on sys.path so `import main.*` works
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture(autouse=True, scope="session")
def seed_everything() -> None:
    seed = int(os.getenv("TEST_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
