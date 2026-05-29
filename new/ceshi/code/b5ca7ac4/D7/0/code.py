# Variant of b5ca7ac4
# Dimension: D7 — 行位置保持规则 (meta)
# Change: 每个框架在输出中保持其在输入中的行位置不变，仅列位置发生改变（推向边缘或偏移）... → 行位置也紧密堆叠——取消行保持

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D7 variant
# a) Concrete behavior:
#    Input objects stay identical to the seed task: scattered 5x5 framed squares,
#    teal-framed and red-framed, with visible 3x3 interiors. The old solver kept
#    each square's input row in the output and only changed its column. This
#    variant keeps the same edge/inward column rule, but the output row is now
#    assigned by processing order: the first square of a color goes to row 0, the
#    second to row 5, then 10, etc. Example on the teal side:
#    old: input rows [2, 13, 7], sorted by input col -> output rows [2, 13, 7]
#    new: same sorted order -> output rows [0, 5, 10]. The overlap test that
#    decides edge vs inward still uses the original input rows.
# b) Complexity calibration:
#    If every square always landed on the edge, the variant would look too simple.
#    The generator therefore still creates 1-2 inward squares per color group so
#    the two-column structure survives and the new row compaction is visible.
# c) Implicit assumptions from the original main():
#    - Detection assumes every real object is exactly a 5x5 monochrome border
#      around a non-uniform-to-border interior. The generator preserves this.
#    - Sorting by input column is still valid and still drives processing order.
#    - The old paste row `r0` depended on "row preservation". That is the one
#      assumption removed here; row overlap is still computed from the input rows.
#    - Max depth remains 2 columns per side; once a square overlaps an earlier
#      edge square, it is pushed inward by exactly one square width.
# d) Placement legality:
#    The output is still 22x22, so tight stacking must fit vertically. A 5x5
#    stack needs 5*n rows, therefore each color count is limited to 3-4 here
#    so the compact stack fits in 20 rows. Input objects remain non-overlapping.
# e) Visual observability:
#    The change is visible because inputs are scattered in row position while
#    outputs are snapped to rows 0,5,10,... on both sides. The generator rejects
#    degenerate cases where the compact output would match the legacy row-keeping
#    output.
# f) Visual cleanliness:
#    Each side uses only two fixed columns (0/5 for teal, 12/17 for red on a
#    22-wide grid), so the compact result forms readable left/right blocks with
#    no overdraw and no mixed-color clutter.
# g) Color reasoning:
#    No new color mapping is introduced. Colors keep their original role only:
#    frame color classifies the square as teal or red, interior colors stay as
#    identity markers for tracking squares across the transformation.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact 5×5 framed squares into color-specific edge stacks.

    Input structure:
    - bg = most common color.
    - Teal (8) and red (2) bordered 5×5 squares with colored interiors are scattered on the grid.

    Rule:
    - Teal squares are processed by input column ascending and red squares by input column descending.
    - A square goes to its side's edge column unless its original 5-row span overlaps an earlier edge square of the same color, in which case it shifts one square inward.
    - Output rows are no longer preserved: within each color group, squares are stacked tightly from top to bottom at rows 0, 5, 10, ...

    Key insight: Input columns still determine processing order, and original row overlap still decides edge vs inward placement, but the final row is reassigned from order index instead of copied from the input.
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

    azure_squares = sorted([s for s in squares if int(s["frame"]) == 8], key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for idx, s in enumerate(azure_squares):
        r0 = int(s["r0"])
        r1 = int(r0 + square_size - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_r = int(idx * square_size)
        dest_c = square_size if has_overlap else 0
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((r0, r1))

    red_squares = sorted([s for s in squares if int(s["frame"]) == 2], key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for idx, s in enumerate(red_squares):
        r0 = int(s["r0"])
        r1 = int(r0 + square_size - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_r = int(idx * square_size)
        dest_c = int(W - 2 * square_size) if has_overlap else int(W - square_size)
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == int(W - square_size):
            placed_at_default.append((r0, r1))

    return out


# --------------- generator (c) ---------------

GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 4)
MAX_INWARD = 2
INPUT_COL_RETRIES = 80
GEN_RETRIES = 700


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_input_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, er) for er in edge_rows):
            edge_rows.append(r)
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, er) for er in edge_rows):
            continue
        if any(_rows_overlap(r, ir) for ir in inward_rows):
            continue
        inward_rows.append(r)
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None
    return edge_rows, inward_rows


def _build_side_specs(
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
) -> list[dict[str, int | bool | np.ndarray]]:
    specs: list[dict[str, int | bool | np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        specs.append(
            {
                "row": int(r),
                "is_edge": True,
                "patch": _make_patch(frame_color, colors[i]),
            }
        )
    for i, r in enumerate(inward_rows):
        specs.append(
            {
                "row": int(r),
                "is_edge": False,
                "patch": _make_patch(frame_color, colors[len(edge_rows) + i]),
            }
        )
    return specs


def _derive_input_positions(
    side_specs: list[dict[str, int | bool | np.ndarray]],
    ascending: bool,
    output_cols: tuple[int, int],
) -> list[dict[str, int | bool | np.ndarray]] | None:
    edges = [s for s in side_specs if bool(s["is_edge"])]
    inwards = [s for s in side_specs if not bool(s["is_edge"])]
    inward_deps = [
        [i for i, edge in enumerate(edges) if _rows_overlap(int(inward["row"]), int(edge["row"]))]
        for inward in inwards
    ]

    n = len(side_specs)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_order = list(range(len(inwards)))
        random.shuffle(inward_order)
        for iw_idx in inward_order:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_specs = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[dict[str, int | bool | np.ndarray]] = []
        ok = True
        for i, spec in enumerate(ordered_specs):
            dest_c = output_cols[0] if bool(spec["is_edge"]) else output_cols[1]
            input_c = int(chosen_cols[i])
            if input_c == dest_c:
                ok = False
                break
            result.append(
                {
                    "row": int(spec["row"]),
                    "input_col": input_c,
                    "is_edge": bool(spec["is_edge"]),
                    "patch": spec["patch"],
                }
            )
        if not ok:
            continue
        if not _any_pairwise_overlap([(int(s["row"]), int(s["input_col"]), s["patch"]) for s in result]):
            return result

    return None


def _render_output(
    bg: int,
    azure_ordered: list[dict[str, int | bool | np.ndarray]],
    red_ordered: list[dict[str, int | bool | np.ndarray]],
    *,
    compact_rows: bool,
) -> np.ndarray:
    out = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for idx, spec in enumerate(azure_ordered):
        dest_r = int(idx * SQ) if compact_rows else int(spec["row"])
        dest_c = 0 if bool(spec["is_edge"]) else SQ
        paste(out, spec["patch"], dest_r, dest_c)
    for idx, spec in enumerate(red_ordered):
        dest_r = int(idx * SQ) if compact_rows else int(spec["row"])
        dest_c = int(GRID_W - SQ) if bool(spec["is_edge"]) else int(GRID_W - 2 * SQ)
        paste(out, spec["patch"], dest_r, dest_c)
    return out


def _variant_row_shift_metric(
    azure_ordered: list[dict[str, int | bool | np.ndarray]],
    red_ordered: list[dict[str, int | bool | np.ndarray]],
) -> int:
    moved = 0
    for ordered in (azure_ordered, red_ordered):
        for idx, spec in enumerate(ordered):
            if int(spec["row"]) != int(idx * SQ):
                moved += 1
    return moved


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_rows = _assign_input_rows(n_azure)
    red_rows = _assign_input_rows(n_red)
    if azure_rows is None or red_rows is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_rows
    red_edge_rows, red_inward_rows = red_rows

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    azure_specs = _build_side_specs(azure_edge_rows, azure_inward_rows, azure_colors, AZURE)
    red_specs = _build_side_specs(red_edge_rows, red_inward_rows, red_colors, RED)

    azure_input = _derive_input_positions(azure_specs, ascending=True, output_cols=(0, SQ))
    red_input = _derive_input_positions(red_specs, ascending=False, output_cols=(GRID_W - SQ, GRID_W - 2 * SQ))
    if azure_input is None or red_input is None:
        return None

    all_positions = (
        [(int(s["row"]), int(s["input_col"]), s["patch"]) for s in azure_input]
        + [(int(s["row"]), int(s["input_col"]), s["patch"]) for s in red_input]
    )
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for spec in azure_input + red_input:
        paste(inp, spec["patch"], int(spec["row"]), int(spec["input_col"]))

    compact_output = _render_output(bg, azure_input, red_input, compact_rows=True)
    legacy_output = _render_output(bg, azure_input, red_input, compact_rows=False)
    if np.array_equal(compact_output, legacy_output):
        return None
    if _variant_row_shift_metric(azure_input, red_input) < 2:
        return None

    try:
        if np.array_equal(main(inp), compact_output):
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
