"""Single-layer (单层) prompt generation.

Same CLI as generate_prompt_from_arc_code_comments.py, but defaults to the
single-layer template in this folder and writes runs to /home/lixinyu/snap/a/des_single.
Both defaults can still be overridden with --prompt_template / --out_dir.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from generate_prompt_from_arc_code_comments import main

if __name__ == "__main__":
    if "--prompt_template" not in sys.argv:
        sys.argv += ["--prompt_template", str(HERE / "prompt_single_layer_zh.md")]
    if "--out_dir" not in sys.argv:
        sys.argv += ["--out_dir", "/home/lixinyu/snap/a/des_single"]
    main()
