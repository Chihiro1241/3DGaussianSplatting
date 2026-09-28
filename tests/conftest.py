from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The 4D frame-to-frame extension lives outside the package, exactly as
# scripts/train_4d.py imports it.
EXTENSION_ROOT = PROJECT_ROOT / "extensions" / "4dgs"
if EXTENSION_ROOT.is_dir() and str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

