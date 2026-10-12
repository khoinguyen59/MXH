"""Graphify AST Knowledge Graph Extraction Utility.

Maintains two strictly decoupled graphify profiles:
1. 'main': Extracts AST for core services only (src/), outputting to graphify-out/
2. 'all': Extracts AST for full codebase (src/, experiments/, tests/), outputting to graphify-out-all/
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def extract_main_source():
    """Extract AST graph for core source code (src/) only."""
    print(">>> Extracting AST for Core Source (src/)...")
    cmd = "set KMP_DUPLICATE_LIB_OK=TRUE && graphify extract src --code-only --out . --force"
    subprocess.run(f"cmd /c {cmd}", cwd=str(BASE_DIR), shell=True, check=True, env=dict(os.environ))
    
    # Generate tree
    tree_cmd = 'graphify tree --graph graphify-out/graph.json --output graphify-out/GRAPH_TREE.html --label "SNA_C10_Core_Source"'
    subprocess.run(f"cmd /c {tree_cmd}", cwd=str(BASE_DIR), shell=True, check=False, env=dict(os.environ))
    print(">>> Finished: graphify-out/ contains core source AST (src/).")


def extract_all_source():
    """Extract AST graph for the full codebase (src/, experiments/, tests/)."""
    print(">>> Extracting AST for Full Codebase (src, experiments, tests)...")
    temp_dir = BASE_DIR / "temp_graphify_all"
    out_all_dir = BASE_DIR / "graphify-out-all"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    cmd = f"set KMP_DUPLICATE_LIB_OK=TRUE && graphify extract . --code-only --out temp_graphify_all --force"
    subprocess.run(f"cmd /c {cmd}", cwd=str(BASE_DIR), shell=True, check=True, env=dict(os.environ))
    
    temp_out = temp_dir / "graphify-out"
    if temp_out.exists():
        if out_all_dir.exists():
            shutil.rmtree(out_all_dir, ignore_errors=True)
        shutil.move(str(temp_out), str(out_all_dir))
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    # Generate tree
    tree_cmd = 'graphify tree --graph graphify-out-all/graph.json --output graphify-out-all/GRAPH_TREE.html --label "SNA_C10_Full_Codebase"'
    subprocess.run(f"cmd /c {tree_cmd}", cwd=str(BASE_DIR), shell=True, check=False)
    print(">>> Finished: graphify-out-all/ contains full codebase AST.")


def main():
    parser = argparse.ArgumentParser(description="Update Graphify AST knowledge graphs.")
    parser.add_argument("--profile", choices=["main", "all", "both"], default="both",
                        help="Graph profile to extract: 'main' (src/ only), 'all' (full codebase), or 'both'.")
    args = parser.parse_args()

    if args.profile in ["main", "both"]:
        extract_main_source()
    if args.profile in ["all", "both"]:
        extract_all_source()


if __name__ == "__main__":
    main()
