from __future__ import annotations

import argparse
from pathlib import Path

from build_arc_codegen_prompts_with_similarity import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_ARC_CODE_DIR,
    DEFAULT_TMP_DIR,
    collect_references,
    load_catalog_terms,
    write_reference_index,
)


DEFAULT_OUT = Path("/home/lixinyu/snap/a/code/arc_reference_index.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ARC_code docstrings, annotations, tags, and code features into one JSON index.")
    parser.add_argument("--arc_code_dir", type=Path, default=DEFAULT_ARC_CODE_DIR)
    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--tmp_dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--source", choices=["eval", "train", "both"], default="both")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_terms = load_catalog_terms(args.tmp_dir)
    refs = collect_references(args.arc_code_dir, args.annotation_dir, source=args.source, catalog_terms=catalog_terms)
    write_reference_index(args.out, refs, arc_code_dir=args.arc_code_dir, annotation_dir=args.annotation_dir, source=args.source)
    by_source: dict[str, int] = {}
    for ref in refs:
        by_source[ref.source] = by_source.get(ref.source, 0) + 1
    print(f"[OK] wrote {len(refs)} references to {args.out}")
    print("[OK] by source:", ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))


if __name__ == "__main__":
    main()
