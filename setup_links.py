# setup_links.py
"""
Create links inside notebooks/ so legacy code that expects files in the
same directory keeps working. Works on Windows, macOS, Linux.

- For CSVs: link every file in data/ into notebooks/
- For models: link every .pt in fine_tuned_models/ into notebooks/
Safe to run multiple times.
"""

import os, sys, shutil
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "fine_tuned_models"

NB_DIR.mkdir(exist_ok=True)
if not DATA_DIR.exists():
    print("WARNING: data/ not found.")
if not MODELS_DIR.exists():
    print("WARNING: fine_tuned_models/ not found.")

def link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    try:
        # On Windows, prefer hardlink for files (no admin needed), fallback to copy
        if platform.system() == "Windows":
            if src.is_file():
                import subprocess
                # hardlink works only on same drive
                res = subprocess.run(["cmd", "/c", "mklink", "/H", str(dst), str(src)], capture_output=True, text=True)
                if res.returncode != 0:
                    # fallback: try symlink (works if Developer Mode or admin), else copy
                    res = subprocess.run(["cmd", "/c", "mklink", str(dst), str(src)], capture_output=True, text=True)
                    if res.returncode != 0:
                        shutil.copy2(src, dst)
            else:
                # directory junction for directories
                import subprocess
                subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True)
        else:
            # Unix: symlink
            dst.symlink_to(src)
    except Exception as e:
        print(f"Could not link {src.name}: {e}. Copying instead.")
        if src.is_file():
            shutil.copy2(src, dst)

# link all CSVs and JSONs used by notebooks
for p in DATA_DIR.glob("*"):
    if p.suffix.lower() in {".csv", ".json", ".parquet", ".tsv"}:
        link(p, NB_DIR / p.name)

# link all .pt model files
for p in MODELS_DIR.glob("*.pt"):
    link(p, NB_DIR / p.name)

print("Done: links created in notebooks/. If you see no errors, you can run notebooks unchanged.")