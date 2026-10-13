from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INDEX_PATH = Path("/home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json")
DEFAULT_OUT_DIR = Path("/home/lixinyu/snap/a/key_combo/select")
DEFAULT_GROUP_SIZE = 20


@dataclass(frozen=True)
class PuzzleEntry:
    original_index: int
    puzzle_id: str
    source: str
    docstring: str
    tags: list[str]
    dataset: str | None


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_entries(index_path: Path) -> list[PuzzleEntry]:
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not read index JSON: {index_path}") from exc

    references = data.get("references")
    if not isinstance(references, list):
        raise RuntimeError(f"Index JSON must contain a list field named 'references': {index_path}")

    entries: list[PuzzleEntry] = []
    for idx, item in enumerate(references):
        if not isinstance(item, dict):
            raise RuntimeError(f"Reference at index {idx} is not an object.")

        missing = [key for key in ("puzzle_id", "source", "docstring") if key not in item]
        if missing:
            raise RuntimeError(f"Reference at index {idx} is missing required fields: {', '.join(missing)}")

        dataset = item.get("dataset")
        entries.append(
            PuzzleEntry(
                original_index=idx + 1,
                puzzle_id=str(item["puzzle_id"]),
                source=str(item["source"]),
                docstring=str(item["docstring"]).strip(),
                tags=normalize_tags(item.get("tags")),
                dataset=str(dataset) if dataset is not None else None,
            )
        )

    return entries


def shuffle_entries(entries: list[PuzzleEntry], *, rng_seed: int) -> list[PuzzleEntry]:
    shuffled = list(entries)
    rng = random.Random(str(rng_seed))
    rng.shuffle(shuffled)
    return shuffled


def iter_groups(
    entries: list[PuzzleEntry],
    *,
    group_size: int,
    drop_remainder: bool,
) -> list[tuple[int, int, list[PuzzleEntry]]]:
    if group_size <= 0:
        raise RuntimeError("--group_size must be positive")

    groups: list[tuple[int, int, list[PuzzleEntry]]] = []
    for start in range(0, len(entries), group_size):
        group = entries[start : start + group_size]
        if drop_remainder and len(group) < group_size:
            break
        groups.append((len(groups) + 1, start, group))
    return groups


def code_fence_for(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return fence


def format_group_markdown(
    group: list[PuzzleEntry],
    *,
    group_index: int,
    start_index: int,
    index_path: Path,
) -> str:
    end_index = start_index + len(group)
    lines = [
        f"# ARC Puzzle Group {group_index:03d}",
        "",
        f"- source_index: `{index_path}`",
        f"- shuffled_record_range: `{start_index + 1}-{end_index}`",
        f"- record_count: `{len(group)}`",
        "",
    ]

    for local_index, entry in enumerate(group, 1):
        shuffled_index = start_index + local_index
        lines.extend(
            [
                f"## {local_index}. {entry.puzzle_id}",
                "",
                f"- puzzle_id: `{entry.puzzle_id}`",
                f"- source: `{entry.source}`",
            ]
        )
        if entry.dataset is not None:
            lines.append(f"- dataset: `{entry.dataset}`")

        lines.append(f"- shuffled_index: `{shuffled_index}`")
        lines.append(f"- original_index: `{entry.original_index}`")
        lines.append("- tags:")
        if entry.tags:
            lines.extend(f"  - {tag}" for tag in entry.tags)
        else:
            lines.append("  - None")

        fence = code_fence_for(entry.docstring)
        lines.extend(
            [
                "",
                "### docstring",
                "",
                f"{fence}text",
                entry.docstring,
                fence,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_groups(
    entries: list[PuzzleEntry],
    *,
    out_dir: Path,
    group_size: int,
    drop_remainder: bool,
    index_path: Path,
) -> list[Path]:
    groups = iter_groups(entries, group_size=group_size, drop_remainder=drop_remainder)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in out_dir.glob("[0-9][0-9][0-9].md"):
        old_path.unlink()

    written: list[Path] = []
    for group_index, start_index, group in groups:
        out_path = out_dir / f"{group_index:03d}.md"
        out_path.write_text(
            format_group_markdown(
                group,
                group_index=group_index,
                start_index=start_index,
                index_path=index_path,
            ),
            encoding="utf-8",
        )
        written.append(out_path)

    return written


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an ARC puzzle index JSON into Markdown files of fixed-size groups."
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="input JSON index containing a references list",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory where numbered Markdown files are written",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="number of puzzles per Markdown file",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="random seed used before partitioning puzzles into groups",
    )
    parser.add_argument(
        "--preserve_order",
        action="store_true",
        help="do not shuffle; split references in their original JSON order",
    )
    remainder_group = parser.add_mutually_exclusive_group()
    remainder_group.add_argument(
        "--drop_remainder",
        dest="drop_remainder",
        action="store_true",
        default=True,
        help="drop the final partial group if it has fewer than --group_size puzzles; this is the default",
    )
    remainder_group.add_argument(
        "--keep_remainder",
        dest="drop_remainder",
        action="store_false",
        help="keep the final partial group if it has fewer than --group_size puzzles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    entries = load_entries(args.index_path)
    if not args.preserve_order:
        entries = shuffle_entries(entries, rng_seed=int(args.rng_seed))

    written = write_groups(
        entries,
        out_dir=args.out_dir,
        group_size=int(args.group_size),
        drop_remainder=bool(args.drop_remainder),
        index_path=args.index_path,
    )

    print(f"[OK] Loaded {len(entries)} puzzle entries from {args.index_path}")
    if args.preserve_order:
        print("[OK] Split order: original JSON order")
    else:
        print(f"[OK] Split order: shuffled with rng_seed={args.rng_seed}")
    print(f"[OK] Wrote {len(written)} Markdown files to {args.out_dir}")
    if written:
        print(f"[OK] First file: {written[0]}")
        print(f"[OK] Last file: {written[-1]}")


if __name__ == "__main__":
    main()
