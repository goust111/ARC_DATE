# Variant of b5ca7ac4
# Dimension: D14 — 源-目标基数关系 (meta)
# Change: 多对二——N个框架分为两组，每组按各自规则堆叠到左/右边缘（many:2 目标区域）... → 一对多——每个框架复制到多个位置

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis
# ---------------
# design_analysis:
#   source_task: none
#   concrete_behavior:
#     - Objects are still 5x5 bordered squares with frame color 8 (azure) or 2 (red).
#     - The original edge-stacking rule is preserved in the upper half of the output.
#     - D14 changes the target cardinality: each detected source square now produces
#       two output copies:
#         1. one copy in the usual color-dependent edge stack
#         2. one copy at the vertically mirrored row, keeping its original input column
#     - Example:
#         input square at (r=1, c=13), frame 8
#         original task -> only one output copy at (1, 0 or 5)
#         variant task  -> one output copy at (1, 0 or 5) plus a second copy at
#                          (H-5-1, 13)
#   complexity_calibration:
#     - The extra copy is highly visible because it creates a second band of shapes in
#       the lower half; the task does not rely on hidden ids or color arithmetic.
#     - To make the mirrored copy readable and non-overlapping, all input squares are
#       generated in the upper source band and their mirrored copies land in the lower
#       half.
#     - The red count is tightened from 3..5 to 3..4, and each side uses exactly two
#       edge rows. This is the minimal structural adjustment needed to fit both the
#       preserved stacking rule and the new mirrored copies into 22x22 cleanly.
#   implicit_assumptions_review:
#     - Square detection by scanning 5x5 windows still works unchanged because the input
#       still contains isolated 5x5 frames only once each.
#     - The original edge-placement loops assumed one output copy per square; in this
#       variant that would silently drop the second copy, so main() must paste mirrored
#       copies explicitly in addition to the stacked copies.
#     - Blind pasting becomes unsafe if mirrored rows can overlap stacked rows. The new
#       generator therefore guarantees all source rows are in 0..6, so mirrored rows are
#       in 11..17 and never collide with the upper stacked band.
#   placement_legality:
#     - Input squares are non-overlapping and confined to:
#         red columns   0..6
#         azure columns 11..17
#         source rows   0..6
#       so cross-group input overlap is impossible even when rows overlap.
#     - Mirrored copies inherit those separated columns and move to rows 11..17, so they
#       remain disjoint from the upper stacked copies.
#   visual_observability:
#     - Across examples, the solver must notice two stable effects:
#         1. top copies reorganize by the old left/right edge rule
#         2. lower copies appear directly below, at vertically symmetric rows, with the
#            same shapes and colors as the inputs
#     - This is visible from geometry alone; no sorting uses color ids other than the
#       already-visible frame-color grouping from the original task.
#   visual_cleanliness:
#     - The output separates into an upper "stacked" band and a lower "mirrored" band,
#       avoiding multi-layer clutter in the same region.
#   color_inference:
#     - No new color mapping is introduced. Interior colors remain identity-preserved,
#       so the only new inference burden is the duplication pattern itself.
#   solver_generator_contract:
#     - main() assumes every true square is a 5x5 frame with at least one non-frame
#       interior cell and that mirrored copies fit in-bounds.
#     - generate_input() enforces those conditions and also guarantees the mirror/stack
#       separation that keeps overwrite-free placement valid.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 4)
SOURCE_ROW_MAX = GRID_H // 2 - SQ  # 6 -> mirror rows become 11..17
AZURE_INPUT_COLS = list(range(11, GRID_W - SQ + 1))  # 11..17
RED_INPUT_COLS = list(range(0, 7))  # 0..6
INPUT_COL_RETRIES = 120
ROW_ASSIGN_RETRIES = 120
GEN_RETRIES = 1200


def _mirror_row(r0: int, height: int = GRID_H) -> int:
    return int(height - SQ - int(r0))


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _detect_squares(g: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    squares: list[dict[str, int | np.ndarray]] = []
    for r0 in range(g.shape[0] - SQ + 1):
        for c0 in range(g.shape[1] - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {AZURE, RED}:
                continue
            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})
    return squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Duplicate each framed 5×5 square: one copy is edge-stacked by frame color, and one copy is mirrored downward.

    Input structure:
    - bg is the most frequent color.
    - The foreground consists of red (2) and azure (8) 5×5 framed squares with colored 3×3 interiors, placed in an upper source zone.

    Rule:
    - Azure squares are processed by increasing input column and copied to the left side; red squares are processed by decreasing input column and copied to the right side.
    - Each stacked copy keeps the source row, and row-overlap with an already-used edge row pushes the square one square-width inward, exactly as in the original task.
    - Independently of that top stacking, every detected square also appears once at the vertically mirrored row H−5−r, keeping its original input column.

    Key insight:
    - The old nearest-edge-first ordering still decides which top copy lands on the edge versus the inward lane, but the output cardinality changes: every single source square yields two visible copies.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    squares = _detect_squares(g)

    for s in squares:
        paste(out, s["patch"], _mirror_row(int(s["r0"]), H), int(s["c0"]), transparent=bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        dest_c = SQ if any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge) else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        dest_c = int(W - 2 * SQ) if any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge) else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    n_edge = 2
    n_inward = n - n_edge
    source_rows = list(range(SOURCE_ROW_MAX + 1))
    edge_pairs = [(a, b) for a, b in combinations(source_rows, 2) if not _rows_overlap(a, b)]
    if not edge_pairs:
        return None

    for _ in range(ROW_ASSIGN_RETRIES):
        edge_rows = list(random.choice(edge_pairs))
        random.shuffle(edge_rows)

        inward_rows: list[int] = []
        pool = source_rows[:]
        random.shuffle(pool)
        for r in pool:
            if not any(_rows_overlap(r, er) for er in edge_rows):
                continue
            if any(_rows_overlap(r, ir) for ir in inward_rows):
                continue
            inward_rows.append(r)
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows
    return None


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, r in enumerate(edge_rows):
        patch = _make_patch(frame_color, colors[i])
        paste(output, patch, r, edge_col)
        placed.append((r, edge_col, True, patch))
    for i, r in enumerate(inward_rows):
        patch = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, patch, r, inward_col)
        placed.append((r, inward_col, False, patch))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
    allowed_cols: list[int],
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]
    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(sq_list)
    if len(allowed_cols) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for iw_idx in inward_indices:
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

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(allowed_cols, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, patch) in enumerate(ordered):
            ic = int(chosen_cols[i])
            if ic == oc:
                ok = False
                break
            result.append((int(r), ic, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _build_expected_output(
    bg: int,
    azure_out: list[tuple[int, int, bool, np.ndarray]],
    red_out: list[tuple[int, int, bool, np.ndarray]],
    azure_input: list[tuple[int, int, np.ndarray]],
    red_input: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in azure_input + red_input:
        paste(output, patch, _mirror_row(r), c)
    for r, c, _is_edge, patch in azure_out:
        paste(output, patch, r, c)
    for r, c, _is_edge, patch in red_out:
        paste(output, patch, r, c)
    return output


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    red_assign = _assign_output_rows(n_red)
    if azure_assign is None or red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    top_only_output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(top_only_output, azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_out = _place_side_squares(top_only_output, red_edge_rows, red_inward_rows, red_colors, RED, GRID_W - SQ, GRID_W - 2 * SQ)

    azure_input = _derive_input_positions(azure_out, ascending=True, allowed_cols=AZURE_INPUT_COLS)
    red_input = _derive_input_positions(red_out, ascending=False, allowed_cols=RED_INPUT_COLS)
    if azure_input is None or red_input is None:
        return None

    all_input_positions = azure_input + red_input
    if _any_pairwise_overlap(all_input_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_input_positions:
        paste(inp, patch, r, c)

    expected_output = _build_expected_output(bg, azure_out, red_out, azure_input, red_input)
    try:
        if np.array_equal(main(inp), expected_output):
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
