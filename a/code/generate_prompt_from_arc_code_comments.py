from __future__ import annotations

import argparse
import ast
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ARC_CODE_DIR = Path("/home/lixinyu/snap/new/ARC_code")
DEFAULT_ANNOTATION_DIR = Path("/home/lixinyu/snap/new/annotations")
DEFAULT_PROMPT_TEMPLATE = Path("/home/lixinyu/snap/a/prompt/arc_code_comment_description_prompt_zh.md")
DEFAULT_PROMPT_TEMPLATE_NO_ANNOTATION = Path(
    "/home/lixinyu/snap/a/prompt/3.md"
)
DEFAULT_OUT_DIR = Path("/home/lixinyu/snap/a/des")
DEFAULT_OUT_DIR_NO_ANNOTATION = Path("/home/lixinyu/snap/a/des")
SKIP_FILENAMES = {"arc_utils.py", "__init__.py"}
TYPE_ID_TO_ANNOTATION_TYPE = {
    1: "move_along_path",
    2: "transform_in_place",
    3: "move_to_match",
    4: "symmetry/pattern",
    5: "move_in_order",
}


@dataclass(frozen=True)
class PuzzleDescription:
    puzzle_id: str
    source: str
    path: str
    start_line: int | None
    end_line: int | None
    description: str
    annotation_path: str | None
    annotation: str | None
    annotation_type: str | None


def extract_main_docstring(path: Path) -> tuple[str, int | None, int | None] | None:
    """Extract the main() docstring and its line span from one solver file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        print(f"[WARN] Skip unreadable file {path}: {exc}")
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            docstring = ast.get_docstring(node, clean=True)
            if not docstring:
                return None
            first_stmt = node.body[0] if node.body else None
            start_line = getattr(first_stmt, "lineno", None)
            end_line = getattr(first_stmt, "end_lineno", None)
            return docstring.strip(), start_line, end_line
    return None


def load_annotation(annotation_dir: Path, puzzle_id: str) -> tuple[str, str, str | None] | tuple[None, None, None]:
    """Load one structured annotation JSON if it exists."""
    path = annotation_dir / f"{puzzle_id}.json"
    if not path.is_file():
        return None, None, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"[WARN] Skip unreadable annotation {path}: {exc}")
        return None, None, None

    return str(path), json.dumps(data, ensure_ascii=False, indent=2), data.get("type")


def collect_descriptions(
    arc_code_dir: Path,
    annotation_dir: Path,
    source: str,
) -> list[PuzzleDescription]:
    """Collect puzzle descriptions from eval, train, or both."""
    source_dirs = ["eval", "train"] if source == "both" else [source]

    descriptions: list[PuzzleDescription] = []
    for source_dir in source_dirs:
        folder = arc_code_dir / source_dir
        if not folder.is_dir():
            raise RuntimeError(f"Source folder not found: {folder}")

        for path in sorted(folder.glob("*.py")):
            if path.name in SKIP_FILENAMES:
                continue

            extracted = extract_main_docstring(path)
            if extracted is None:
                print(f"[WARN] No main() docstring found in {path}")
                continue

            docstring, start_line, end_line = extracted
            annotation_path, annotation, annotation_type = load_annotation(annotation_dir, path.stem)
            descriptions.append(
                PuzzleDescription(
                    puzzle_id=path.stem,
                    source=source_dir,
                    path=str(path),
                    start_line=start_line,
                    end_line=end_line,
                    description=docstring,
                    annotation_path=annotation_path,
                    annotation=annotation,
                    annotation_type=annotation_type,
                )
            )

    return descriptions


def filter_descriptions_by_type(
    descriptions: list[PuzzleDescription],
    *,
    type_id: int | None,
) -> list[PuzzleDescription]:
    if type_id is None:
        return descriptions

    annotation_type = TYPE_ID_TO_ANNOTATION_TYPE[type_id]
    return [item for item in descriptions if item.annotation_type == annotation_type]


def select_descriptions(
    descriptions: list[PuzzleDescription],
    *,
    count: int,
    rng_seed: str,
) -> list[PuzzleDescription]:
    if count <= 0:
        raise RuntimeError("--num_descriptions must be positive")
    if len(descriptions) < count:
        raise RuntimeError(f"Only {len(descriptions)} descriptions available, cannot sample {count}.")

    rng = random.Random(rng_seed)
    selected = list(descriptions)
    rng.shuffle(selected)
    return selected[:count]


def select_partitioned_description_groups(
    descriptions: list[PuzzleDescription],
    *,
    count: int,
    batch_size: int,
    rng_seed: str,
) -> list[list[PuzzleDescription]]:
    if count <= 0:
        raise RuntimeError("--num_descriptions must be positive")
    if batch_size <= 0:
        raise RuntimeError("--batch_size must be positive")
    max_batch_size = len(descriptions) // count
    if max_batch_size <= 0:
        raise RuntimeError(f"Only {len(descriptions)} descriptions available, cannot make groups of {count}.")
    if batch_size > max_batch_size:
        raise RuntimeError(
            f"shuffle_partition can create at most {max_batch_size} non-overlapping groups "
            f"from {len(descriptions)} descriptions with group size {count}; got --batch_size {batch_size}."
        )

    rng = random.Random(rng_seed)
    shuffled = list(descriptions)
    rng.shuffle(shuffled)
    return [shuffled[index * count : (index + 1) * count] for index in range(batch_size)]


def format_reference_descriptions(
    selected: list[PuzzleDescription],
    *,
    include_annotation: bool,
) -> str:
    blocks: list[str] = []
    for index, item in enumerate(selected, 1):
        line_span = ""
        if item.start_line is not None and item.end_line is not None:
            line_span = f' lines="{item.start_line}-{item.end_line}"'

        lines = [
            f'<reference_puzzle index="{index}" source="{item.source}" puzzle_id="{item.puzzle_id}"{line_span}>',
            "<description>",
            item.description,
            "</description>",
        ]
        if include_annotation:
            lines.extend(
                [
                    "<annotation>",
                    item.annotation or "No matching annotation JSON was found.",
                    "</annotation>",
                ]
            )
        lines.append("</reference_puzzle>")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_prompt(
    template_text: str,
    selected: list[PuzzleDescription],
    *,
    num_generations: int,
    include_annotation: bool,
) -> str:
    return template_text.format(
        reference_descriptions=format_reference_descriptions(
            selected,
            include_annotation=include_annotation,
        ),
        num_descriptions=len(selected),
        num_generations=num_generations,
    )


def extract_generated_descriptions(completion_text: str) -> list[str]:
    """Extract generated new-puzzle descriptions from a model output file."""
    new_puzzle_pattern = re.compile(r"<new_puzzle>\s*(.*?)\s*</new_puzzle>", re.IGNORECASE | re.DOTALL)
    blocks = [match.strip() for match in new_puzzle_pattern.findall(completion_text) if match.strip()]
    if blocks:
        return [f"<new_puzzle>\n{block}\n</new_puzzle>" for block in blocks]

    description_pattern = re.compile(r"<description>\s*(.*?)\s*</description>", re.IGNORECASE | re.DOTALL)
    return [match.strip() for match in description_pattern.findall(completion_text) if match.strip()]


def write_generated_descriptions(path: Path, descriptions: list[str]) -> None:
    """Write only generated puzzle descriptions to des.md."""
    if not descriptions:
        path.write_text("", encoding="utf-8")
        return
    path.write_text("\n\n".join(descriptions) + "\n", encoding="utf-8")


def write_selected_ids(path: Path, selected: list[PuzzleDescription]) -> None:
    """Write the 20 sampled puzzle ids."""
    lines = [f"{item.source}/{item.puzzle_id}" for item in selected]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ARC puzzle-description prompts from main() docstrings in ARC_code."
    )
    parser.add_argument(
        "--arc_code_dir",
        type=Path,
        default=DEFAULT_ARC_CODE_DIR,
        help="root directory containing eval/ and train/ folders",
    )
    parser.add_argument(
        "--annotation_dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
        help="directory containing per-puzzle annotation JSON files",
    )
    parser.add_argument(
        "--source",
        choices=["eval", "train", "both"],
        default="both",
        help="sample only eval, only train, or randomly from both folders",
    )
    parser.add_argument(
        "--num_descriptions",
        "-d",
        type=int,
        default=20,
        help="number of reference descriptions embedded in each prompt",
    )
    parser.add_argument(
        "--num_generations",
        "-n",
        type=int,
        default=5,
        help="number of new puzzle descriptions requested from the model",
    )
    parser.add_argument(
        "--type_id",
        type=int,
        choices=sorted(TYPE_ID_TO_ANNOTATION_TYPE),
        default=None,
        help=(
            "optional annotation type filter: "
            "1=move_along_path, 2=transform_in_place, 3=move_to_match, "
            "4=symmetry/pattern, 5=move_in_order"
        ),
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=None,
        help=(
            "number of prompt folders to create; defaults to all full non-overlapping groups "
            "for shuffle_partition and 1 for random"
        ),
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=["shuffle_partition", "random"],
        default="shuffle_partition",
        help=(
            "shuffle_partition shuffles all candidate puzzles once and splits them into non-overlapping "
            "groups; random samples each prompt independently"
        ),
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=0,
        help="random seed controlling the shuffled puzzle order",
    )
    parser.add_argument(
        "--prompt_template",
        type=Path,
        default=None,
        help="custom prompt template path",
    )
    annotation_group = parser.add_mutually_exclusive_group()
    annotation_group.add_argument(
        "-a",
        "--annotation",
        dest="include_annotation",
        action="store_true",
        default=False,
        help="embed annotation JSON and use the annotation prompt template",
    )
    annotation_group.add_argument(
        "--no_annotation",
        dest="include_annotation",
        action="store_false",
        help="do not embed annotation JSON; kept for compatibility and is the default",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="directory where run folders are written; defaults to /home/lixinyu/snap/a/des",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="folder name created under --out_dir for this run",
    )
    parser.add_argument(
        "--completion_file",
        type=Path,
        default=None,
        help="optional model output file; generated new-puzzle descriptions are written to des.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    prompt_template = args.prompt_template
    if prompt_template is None:
        prompt_template = DEFAULT_PROMPT_TEMPLATE if args.include_annotation else DEFAULT_PROMPT_TEMPLATE_NO_ANNOTATION
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR if args.include_annotation else DEFAULT_OUT_DIR_NO_ANNOTATION

    template_text = prompt_template.read_text(encoding="utf-8")
    descriptions = collect_descriptions(args.arc_code_dir, args.annotation_dir, args.source)
    descriptions = filter_descriptions_by_type(descriptions, type_id=args.type_id)
    if not descriptions:
        type_note = ""
        if args.type_id is not None:
            type_note = f" and type_id={args.type_id} ({TYPE_ID_TO_ANNOTATION_TYPE[args.type_id]})"
        raise RuntimeError(f"No descriptions found under {args.arc_code_dir} for source={args.source}{type_note}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size is None:
        if args.sampling_strategy == "shuffle_partition":
            batch_size = len(descriptions) // int(args.num_descriptions)
        else:
            batch_size = 1
    else:
        batch_size = int(args.batch_size)

    completion_descriptions: list[str] = []
    if args.completion_file is not None:
        completion_text = args.completion_file.read_text(encoding="utf-8")
        completion_descriptions = extract_generated_descriptions(completion_text)
        if not completion_descriptions:
            print(f"[WARN] No generated descriptions found in completion file: {args.completion_file}")

    if args.sampling_strategy == "shuffle_partition":
        selected_groups = select_partitioned_description_groups(
            descriptions,
            count=int(args.num_descriptions),
            batch_size=batch_size,
            rng_seed=str(args.rng_seed),
        )
    else:
        selected_groups = [
            select_descriptions(
                descriptions,
                count=int(args.num_descriptions),
                rng_seed=f"{args.rng_seed}:{batch_idx}",
            )
            for batch_idx in range(batch_size)
        ]

    skipped_count = len(descriptions) - (len(selected_groups) * int(args.num_descriptions))
    if args.sampling_strategy == "shuffle_partition" and skipped_count > 0:
        print(f"[OK] shuffle_partition skipped {skipped_count} leftover descriptions this round.")

    for batch_idx, selected in enumerate(selected_groups):
        prompt_text = build_prompt(
            template_text,
            selected,
            num_generations=int(args.num_generations),
            include_annotation=bool(args.include_annotation),
        )

        run_dir = args.out_dir / args.run_name / str(batch_idx + 1)
        run_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = run_dir / "prompt.md"
        des_path = run_dir / "des.md"
        ids_path = run_dir / "ids.txt"

        prompt_path.write_text(prompt_text, encoding="utf-8")
        write_generated_descriptions(des_path, completion_descriptions)
        write_selected_ids(ids_path, selected)

        selected_ids = ", ".join(f"{item.source}/{item.puzzle_id}" for item in selected)
        print(f"[OK] Wrote prompt to {prompt_path}")
        print(f"[OK] Wrote descriptions to {des_path}")
        print(f"[OK] Wrote selected ids to {ids_path}")
        if args.type_id is not None:
            print(f"[OK] Filtered type: {args.type_id}={TYPE_ID_TO_ANNOTATION_TYPE[args.type_id]}")
        print(f"[OK] Selected: {selected_ids}")


if __name__ == "__main__":
    main()
