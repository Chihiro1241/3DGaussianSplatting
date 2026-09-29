from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# The 4D frame-to-frame extension lives outside the package, exactly as
# scripts/4DGS/train_4d.py imports it.
EXTENSION_ROOT = PROJECT_ROOT / "extensions" / "4dgs"
if EXTENSION_ROOT.is_dir() and str(EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTENSION_ROOT))

# scripts/3DGS/ is not a valid package name, so its entry points are imported
# by module name (``from train_3d import ...``) from this directory.
SCRIPTS_3DGS_ROOT = PROJECT_ROOT / "scripts" / "3DGS"
if str(SCRIPTS_3DGS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_3DGS_ROOT))

