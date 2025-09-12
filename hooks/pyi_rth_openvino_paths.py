"""
Runtime hook to ensure OpenVINO/OpenVINO GenAI native libraries are on PATH
when running the frozen application on Windows.

PyInstaller sets sys._MEIPASS to the unpack dir (e.g., dist/App/_internal).
We extend PATH so that dependent DLLs (openvino.dll, tbb*.dll, *_genai*.dll)
can be located by the Python runtime.
"""

import os
import sys
from pathlib import Path

base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))

candidate_dirs = [
    base_dir,
    base_dir / "openvino",
    base_dir / "openvino" / "libs",
    base_dir / "openvino_genai",
    base_dir / "openvino_tokenizers",
]

# Also add the COLLECT root (parent of _internal)
dist_root = base_dir.parent
candidate_dirs += [
    dist_root,
    dist_root / "openvino",
    dist_root / "openvino" / "libs",
    dist_root / "openvino_genai",
    dist_root / "openvino_tokenizers",
]

existing = os.environ.get("PATH", "")
parts = [str(p) for p in candidate_dirs if p.exists()]
if parts:
    os.environ["PATH"] = os.pathsep.join(parts + [existing])


