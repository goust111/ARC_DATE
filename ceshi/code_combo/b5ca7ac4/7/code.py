# Combo variant of b5ca7ac4 (index 7)
# combo_spec: D5/1+D7/0
# Name: 垂直堆叠+紧密行排列

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(parent / "solvers"))
        break

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Objects stay the same as the seed: scattered 5x5 framed squares, frame color 8
#      or 2, each with a visible 3x3 interior.
#    - D5/1 rotates the relocation axis: teal squares are now pushed upward and red
#      squares downward, so processing order must switch from column-based to row-based,
#      and the overlap test must switch from row spans to column spans.
#    - D7/0 removes the last preserved coordinate. In the D5-only variant a square
#      would still keep its original input column. In this combo it does not: after
#      edge-vs-inward row selection, the squares of each color are packed tightly into
#      output columns 0,5,10,... by processing order. The result is a compact top block
#      for teal and a compact bottom block for red.
#    - Example: three teal squares at input rows [2, 8, 14] with input columns
#      [1, 3, 11]. They are processed in row order. Square 1 occupies top row band at
#      output row 0 and compact column 0. Square 2 column-overlaps square 1, so it goes
#      to the inward row band at output row 5 and compact column 5. Square 3 does not
#      overlap any earlier edge square in columns, so it stays on row 0 and takes compact
#      column 10. The original task would have preserved rows and pushed horizontally;
#      the combo preserves neither original axis and instead forms a compact block.
# b) Complexity calibration:
#    - The second vertical layer must remain visible, so each color group gets at least
#      one inward square.
#    - Since compact columns consume 5 cells per square, each color count is capped at 4
#      so one group fits inside a 22-wide grid.
#    - Interiors remain unique within a frame color so the moved squares are easy to track.
# c) Implicit assumption audit of main():
#    - Window scanning for 5x5 monochrome borders still holds because object shape is
#      unchanged.
#    - Sorting by input column no longer matches the target-side proximity assumption;
#      the combo needs input-row sorting because motion is vertical.
#    - The seed stored row intervals for overlap because relocation was horizontal. That
#      assumption breaks here; overlap must be checked on input column spans.
#    - The D5-only variant would paste at the original column. D7/0 breaks that: the
#      combo reassigns compact columns from processing index, so the generator must keep
#      group size <= 4 to avoid spilling past the right edge.
# d) Placement legality:
#    - Output uses only two teal rows (0 and 5) and two red rows (H-10 and H-5), with
#      compact columns 0,5,10,15. This guarantees no output overlap.
#    - Input placement is generated so rectangles do not overlap, stay in-bounds, and any
#      square intended for the inward row band is processed after at least one overlapping
#      edge square of the same color.
# e) Visual observability:
#    - The rule is visible from geometry alone: teal gathers into a compact block at the
#      top, red into one at the bottom; within each block, column order follows the input
#      top-to-bottom or bottom-to-top order.
#    - No hidden numeric color ranking is used. Only visible row order, visible frame
#      color, and visible horizontal overlap drive the transformation.
# f) Visual cleanliness:
#    - Separating teal and red vertically avoids mixed-color clutter. Limiting each group
#      to two row bands produces clean rectangles rather than noisy staggered layouts.
# g) Change inferability and generator guarantees:
#    - The generator rejects D5-only degeneracies where preserving input columns would
#      accidentally produce the same output as the combo.
#    - A measurable fusion signal is the number of squares whose compact output column
#      differs from the D5-only preserved-column output; samples require at least two such
#      shifts so the compaction behavior is actually present.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange bordered 5x5 squares into compact vertical stacks by frame color.

    Input structure:
    - bg = most common color.
    - Teal (8) and red (2) bordered 5x5 squares with colored interiors are scattered across the grid.

    Rule:
    - Teal squares are processed from top to bottom and red squares from bottom to top.
    - A square goes to its color's outer row band unless its input 5-column span overlaps an earlier outer-band square of the same color; then it goes one square-height inward.
    - Original columns are not preserved: within each color group, output columns are packed tightly as 0, 5, 10, ... in processing order. Teal uses the top two row bands and red uses the bottom two row bands.

    Key insight: D5/1 rotates the grouping logic into vertical motion, and D7/0 removes the last preserved coordinate, so the deciding evidence is input row order plus input column overlap, while the final layout becomes a compact top/bottom rectangle instead of a scattered side push.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    square_size = 5

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(H - square_size + 1):
        for c0 in range(W - square_size + 1):
            patch = crop_bbox(
                g,
                (int(r0), int(c0), int(r0 + square_size - 1), int(c0 + square_size - 1)),
            )
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {2, 8}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted([s for s in squares if int(s["frame"]) == 8], key=lambda s: int(s["r0"]))
    outer_spans: list[tuple[int, int]] = []
    for idx, square in enumerate(teal_squares):
        c0 = int(square["c0"])
        c1 = int(c0 + square_size - 1)
        has_overlap = any(overlaps(c0, c1, pc0, pc1) for pc0, pc1 in outer_spans)
        dest_r = int(square_size) if has_overlap else 0
        dest_c = int(idx * square_size)
        paste(out, square["patch"], dest_r, dest_c, transparent=bg)
        if dest_r == 0:
            outer_spans.append((c0, c1))

    red_squares = sorted([s for s in squares if int(s["frame"]) == 2], key=lambda s: -int(s["r0"]))
    outer_spans = []
    for idx, square in enumerate(red_squares):
        c0 = int(square["c0"])
        c1 = int(c0 + square_size - 1)
        has_overlap = any(overlaps(c0, c1, pc0, pc1) for pc0, pc1 in outer_spans)
        dest_r = int(H - 2 * square_size) if has_overlap else int(H - square_size)
        dest_c = int(idx * square_size)
        paste(out, square["patch"], dest_r, dest_c, transparent=bg)
        if dest_r == int(H - square_size):
            outer_spans.append((c0, c1))

    return out


# --------------- generator (c) ---------------

GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_ASSIGN_RETRIES = 120
GEN_RETRIES = 900


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _spans_overlap(a: int, b: int) -> bool:
    return abs(int(a) - int(b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_input_cols(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_cols = list(range(GRID_W - SQ + 1))
    random.shuffle(all_cols)
    edge_cols: list[int] = []
    for col in all_cols:
        if all(not _spans_overlap(col, other) for other in edge_cols):
            edge_cols.append(int(col))
            if len(edge_cols) == n_edge:
                break
    if len(edge_cols) < n_edge:
        return None

    random.shuffle(all_cols)
    inward_cols: list[int] = []
    for col in all_cols:
        if not any(_spans_overlap(col, edge_col) for edge_col in edge_cols):
            continue
        if any(_spans_overlap(col, other) for other in inward_cols):
            continue
        inward_cols.append(int(col))
        if len(inward_cols) == n_inward:
            break
    if len(inward_cols) < n_inward:
        return None

    return edge_cols, inward_cols


def _build_side_specs(
    edge_cols: list[int],
    inward_cols: list[int],
    colors: list[int],
    frame_color: int,
) -> list[dict[str, int | bool | np.ndarray]]:
    specs: list[dict[str, int | bool | np.ndarray]] = []
    for i, col in enumerate(edge_cols):
        specs.append(
            {
                "input_col": int(col),
                "is_edge": True,
                "patch": _make_patch(frame_color, colors[i]),
            }
        )
    for i, col in enumerate(inward_cols):
        specs.append(
            {
                "input_col": int(col),
                "is_edge": False,
                "patch": _make_patch(frame_color, colors[len(edge_cols) + i]),
            }
        )
    return specs


def _assign_processing_rows(
    ordered_specs: list[dict[str, int | bool | np.ndarray]],
    *,
    ascending: bool,
) -> list[int] | None:
    rows = list(range(GRID_H - SQ + 1))
    assigned: list[int | None] = [None] * len(ordered_specs)

    def backtrack(i: int) -> bool:
        if i == len(ordered_specs):
            return True

        if i == 0:
            candidates = rows[:]
        else:
            prev = int(assigned[i - 1])
            if ascending:
                candidates = [r for r in rows if r > prev]
            else:
                candidates = [r for r in rows if r < prev]
        random.shuffle(candidates)

        current_col = int(ordered_specs[i]["input_col"])
        for row in candidates:
            blocked = False
            for j in range(i):
                prev_row = int(assigned[j])
                prev_col = int(ordered_specs[j]["input_col"])
                if _spans_overlap(current_col, prev_col) and abs(row - prev_row) < SQ:
                    blocked = True
                    break
            if blocked:
                continue
            assigned[i] = int(row)
            if backtrack(i + 1):
                return True
            assigned[i] = None
        return False

    for _ in range(INPUT_ASSIGN_RETRIES):
        for i in range(len(assigned)):
            assigned[i] = None
        if backtrack(0):
            return [int(r) for r in assigned if r is not None]
    return None


def _derive_input_positions(
    side_specs: list[dict[str, int | bool | np.ndarray]],
    *,
    ascending: bool,
) -> list[dict[str, int | bool | np.ndarray]] | None:
    edges = [spec for spec in side_specs if bool(spec["is_edge"])]
    inwards = [spec for spec in side_specs if not bool(spec["is_edge"])]
    inward_deps = [
        [i for i, edge in enumerate(edges) if _spans_overlap(int(inward["input_col"]), int(edge["input_col"]))]
        for inward in inwards
    ]

    for _ in range(INPUT_ASSIGN_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
            latest_dep_pos = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (kind, idx) in enumerate(order)
                    if kind == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_dep_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered_specs = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
        assigned_rows = _assign_processing_rows(ordered_specs, ascending=ascending)
        if assigned_rows is None:
            continue

        result: list[dict[str, int | bool | np.ndarray]] = []
        for spec, row in zip(ordered_specs, assigned_rows):
            result.append(
                {
                    "row": int(row),
                    "input_col": int(spec["input_col"]),
                    "is_edge": bool(spec["is_edge"]),
                    "patch": spec["patch"],
                }
            )

        if not _any_pairwise_overlap([(int(s["row"]), int(s["input_col"]), s["patch"]) for s in result]):
            return result

    return None


def _render_combo_output(
    bg: int,
    teal_ordered: list[dict[str, int | bool | np.ndarray]],
    red_ordered: list[dict[str, int | bool | np.ndarray]],
) -> np.ndarray:
    out = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for idx, spec in enumerate(teal_ordered):
        dest_r = SQ if not bool(spec["is_edge"]) else 0
        dest_c = int(idx * SQ)
        paste(out, spec["patch"], dest_r, dest_c)
    for idx, spec in enumerate(red_ordered):
        dest_r = int(GRID_H - 2 * SQ) if not bool(spec["is_edge"]) else int(GRID_H - SQ)
        dest_c = int(idx * SQ)
        paste(out, spec["patch"], dest_r, dest_c)
    return out


def _render_d5_only_output(
    bg: int,
    teal_ordered: list[dict[str, int | bool | np.ndarray]],
    red_ordered: list[dict[str, int | bool | np.ndarray]],
) -> np.ndarray:
    out = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for spec in teal_ordered:
        dest_r = SQ if not bool(spec["is_edge"]) else 0
        paste(out, spec["patch"], dest_r, int(spec["input_col"]))
    for spec in red_ordered:
        dest_r = int(GRID_H - 2 * SQ) if not bool(spec["is_edge"]) else int(GRID_H - SQ)
        paste(out, spec["patch"], dest_r, int(spec["input_col"]))
    return out


def _fusion_shift_metric(
    teal_ordered: list[dict[str, int | bool | np.ndarray]],
    red_ordered: list[dict[str, int | bool | np.ndarray]],
) -> int:
    moved = 0
    for ordered in (teal_ordered, red_ordered):
        for idx, spec in enumerate(ordered):
            if int(spec["input_col"]) != int(idx * SQ):
                moved += 1
    return moved


def _band_count_metric(
    ordered: list[dict[str, int | bool | np.ndarray]],
) -> int:
    return len({bool(spec["is_edge"]) for spec in ordered})


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_cols = _assign_input_cols(n_azure)
    red_cols = _assign_input_cols(n_red)
    if azure_cols is None or red_cols is None:
        return None
    azure_edge_cols, azure_inward_cols = azure_cols
    red_edge_cols, red_inward_cols = red_cols

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    azure_specs = _build_side_specs(azure_edge_cols, azure_inward_cols, azure_colors, AZURE)
    red_specs = _build_side_specs(red_edge_cols, red_inward_cols, red_colors, RED)

    azure_input = _derive_input_positions(azure_specs, ascending=True)
    red_input = _derive_input_positions(red_specs, ascending=False)
    if azure_input is None or red_input is None:
        return None

    all_positions = (
        [(int(spec["row"]), int(spec["input_col"]), spec["patch"]) for spec in azure_input]
        + [(int(spec["row"]), int(spec["input_col"]), spec["patch"]) for spec in red_input]
    )
    if _any_pairwise_overlap(all_positions):
        return None

    if _band_count_metric(azure_input) < 2 or _band_count_metric(red_input) < 2:
        return None
    if _fusion_shift_metric(azure_input, red_input) < 2:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for spec in azure_input + red_input:
        paste(inp, spec["patch"], int(spec["row"]), int(spec["input_col"]))

    combo_output = _render_combo_output(bg, azure_input, red_input)
    d5_only_output = _render_d5_only_output(bg, azure_input, red_input)
    if np.array_equal(combo_output, d5_only_output):
        return None

    try:
        if np.array_equal(main(inp), combo_output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
