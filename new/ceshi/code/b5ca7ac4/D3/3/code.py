# Variant of b5ca7ac4
# Dimension: D3 — 源对象——框架的结构与颜色组成 (src)
# Change: 每个源对象为5x5方框，外层1像素宽的边框为组标识色（青色8或红色2），内部3x3区域填充统一的其他颜色；所有框架尺寸相同 → 三种边框颜色分三组

from __future__ import annotations

import random
import sys

import numpy as np

SOLVER_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS_DIR not in sys.path:
    sys.path.append(SOLVER_UTILS_DIR)

from arc_utils import crop_bbox, get_bg, paste

# Design analysis
# ---------------
# variant_spec:
#   preserved_structure:
#     grid: still 22x22
#     objects: every source object is still a 5x5 bordered square with a 3x3 uniform interior
#     motion: every square keeps its original top row in the output
#   D3_change:
#     frame_groups:
#       teal_8: stack at the left band
#       red_2: stack at the right band
#       green_3: new third group, stack at the centered 5x5 anchor
#   concrete_output_layout:
#     teal columns: 0 for the edge slot, 5 for the inward slot
#     green columns: 8 only; this is the centered anchor because (22 - 5) // 2 = 8
#     red columns: 17 for the edge slot, 12 for the inward slot
#
# behavior_trace:
#   original_flow:
#     - detect all 5x5 framed squares whose border is teal or red
#     - sort teal by input column ascending and let the earliest square claim col 0
#     - later teal squares whose row-span overlaps an already edge-placed teal square go to col 5
#     - do the mirror image for red on the right side
#   variant_flow:
#     - detect the same 5x5 structure but allow border color 3 as well
#     - teal and red keep the exact old behavior
#     - green squares are recognized as a third category and all move to the central anchor at col 8
#   small_example:
#     - if a teal square starts at row 4 / col 2 and another teal square starts at row 6 / col 14,
#       the first teal claims col 0 and the second teal is pushed to col 5 because their 5-row spans overlap
#     - if a green square starts at row 10 / col 1, it simply moves to row 10 / col 8
#     - compared with the original task, the same input would previously ignore that green frame entirely;
#       the variant makes it participate as a third destination group
#
# complexity_calibration:
#   - The new concept is visible in 2-4 examples because a third border color always appears at least twice.
#   - Only three frame colors are used, so the grouping rule remains low-entropy and human-readable.
#   - The center group intentionally uses one anchor column rather than a second two-column stack:
#     a 22-column board cannot host three independent two-column 5x5 bands without horizontal collisions.
#
# hidden_assumptions_review:
#   - Window scanning assumes fixed 5x5 frames with a uniform 1-cell border; this still holds.
#   - The original solver hard-coded frame colors {2, 8}; the variant must expand that set to {2, 3, 8}.
#   - The left/right placement loops assume only two side groups need row-overlap arbitration; that remains true.
#   - A naive center stack with row-overlapping green squares would overwrite itself because there is only one
#     collision-free middle anchor. The generator therefore guarantees pairwise non-overlapping green rows.
#   - Green squares at col 8 would horizontally collide with side inward bands at cols 5 and 12 if their rows
#     overlapped, so the generator also forbids green rows from overlapping any teal-inward or red-inward row.
#
# placement_legality:
#   - Input squares are non-overlapping 5x5 rectangles.
#   - Output squares are also non-overlapping because:
#       * teal inward rows are disjoint from each other and only overlap teal edge rows
#       * red inward rows are disjoint from each other and only overlap red edge rows
#       * green rows are pairwise disjoint and are kept away from all inward-side rows
#   - Validation still uses output-first generation plus main(input) == output.
#
# observability_and_cleanliness:
#   - The third group is visually obvious: green frames always end up in the middle vertical band.
#   - Ordering never depends on invisible metadata such as color id ranking; side groups still use visible
#     horizontal proximity to their destination side, and the middle group has no hidden ordering dependency.
#   - The output remains tidy because overlapping can happen only within a side group's own edge/inward pair.


GRID_H, GRID_W = 22, 22
SQ = 5
TEAL, RED, GREEN = 8, 2, 3
LEFT_COL = 0
LEFT_INNER_COL = SQ
CENTER_COL = (GRID_W - SQ) // 2
RIGHT_INNER_COL = GRID_W - 2 * SQ
RIGHT_COL = GRID_W - SQ
N_TEAL_RANGE = (2, 3)
N_GREEN_RANGE = (2, 3)
N_RED_RANGE = (2, 3)
MAX_INWARD = 2
INPUT_COL_RETRIES = 60
GEN_RETRIES = 700


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack three border-color groups of 5×5 framed squares into left, center, and right bands.

    Input structure:
    - bg is the most common color in the grid.
    - Teal (8), green (3), and red (2) bordered 5×5 squares are scattered on the canvas.
    - Each square has a uniform-colored 3×3 interior that differs from its own frame color.

    Rule:
    - Teal squares keep their row and stack against the left side: the first non-overlapping teal row-span
      takes col 0, while later teal squares that overlap an already edge-placed teal square take col 5.
    - Red squares mirror that behavior on the right side: the first non-overlapping red row-span takes col W−5,
      while later overlapping red squares take col W−10.
    - Green squares form the third group and move to the centered 5×5 anchor at col (W−5)//2.

    Key insight: The border color now selects one of three destination zones; only the teal and red groups still
    need input-column ordering to decide which overlapping square keeps the true edge slot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    center_col = (W - SQ) // 2

    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {TEAL, GREEN, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted([s for s in squares if int(s["frame"]) == TEAL], key=lambda s: int(s["c0"]))
    placed_left_rows: list[tuple[int, int]] = []
    for s in teal_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_left_rows)
        dest_c = LEFT_INNER_COL if has_overlap else LEFT_COL
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == LEFT_COL:
            placed_left_rows.append((r0, r1))

    green_squares = sorted(
        [s for s in squares if int(s["frame"]) == GREEN],
        key=lambda s: (abs(int(s["c0"]) - center_col), int(s["c0"]), int(s["r0"])),
    )
    for s in green_squares:
        paste(out, s["patch"], int(s["r0"]), center_col, transparent=bg)

    red_squares = sorted([s for s in squares if int(s["frame"]) == RED], key=lambda s: -int(s["c0"]))
    placed_right_rows: list[tuple[int, int]] = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_right_rows)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_right_rows.append((r0, r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


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


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
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


def _assign_center_rows(n: int, forbidden_rows: list[int]) -> list[int] | None:
    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    rows: list[int] = []
    for r in all_rows:
        if any(_rows_overlap(r, fr) for fr in forbidden_rows):
            continue
        if any(_rows_overlap(r, rr) for rr in rows):
            continue
        rows.append(r)
        if len(rows) == n:
            return rows
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
        p = _make_patch(frame_color, colors[i])
        paste(output, p, r, edge_col)
        placed.append((r, edge_col, True, p))
    for i, r in enumerate(inward_rows):
        p = _make_patch(frame_color, colors[len(edge_rows) + i])
        paste(output, p, r, inward_col)
        placed.append((r, inward_col, False, p))
    return placed


def _place_center_squares(
    output: np.ndarray,
    rows: list[int],
    colors: list[int],
    frame_color: int,
    center_col: int,
) -> list[tuple[int, int, np.ndarray]]:
    placed: list[tuple[int, int, np.ndarray]] = []
    for i, r in enumerate(rows):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, r, center_col)
        placed.append((r, center_col, p))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]

    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]
    n = len(sq_list)
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
                    for d in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == d
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = int(chosen_cols[i])
            if ic == oc:
                ok = False
                break
            result.append((r, ic, p))
        if not ok:
            continue
        if not _any_pairwise_overlap(result):
            return result

    return None


def _derive_center_input_positions(
    sq_list: list[tuple[int, int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray]] | None:
    col_range = [c for c in range(0, GRID_W - SQ + 1) if c != CENTER_COL]
    if len(col_range) < len(sq_list):
        return None

    for _ in range(INPUT_COL_RETRIES):
        chosen_cols = random.sample(col_range, len(sq_list))
        result = [(int(r), int(chosen_cols[i]), p) for i, (r, _, p) in enumerate(sq_list)]
        if not _any_pairwise_overlap(result):
            return result
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {TEAL, GREEN, RED}])
    n_teal = random.randint(*N_TEAL_RANGE)
    n_green = random.randint(*N_GREEN_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    teal_assign = _assign_output_rows(n_teal)
    if teal_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    teal_edge_rows, teal_inward_rows = teal_assign
    red_edge_rows, red_inward_rows = red_assign

    green_rows = _assign_center_rows(n_green, forbidden_rows=teal_inward_rows + red_inward_rows)
    if green_rows is None:
        return None

    avail_teal = [c for c in range(10) if c not in {bg, TEAL}]
    avail_green = [c for c in range(10) if c not in {bg, GREEN}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_teal) < n_teal or len(avail_green) < n_green or len(avail_red) < n_red:
        return None

    teal_colors = random.sample(avail_teal, n_teal)
    green_colors = random.sample(avail_green, n_green)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    teal_out = _place_side_squares(
        output,
        teal_edge_rows,
        teal_inward_rows,
        teal_colors,
        TEAL,
        edge_col=LEFT_COL,
        inward_col=LEFT_INNER_COL,
    )
    green_out = _place_center_squares(output, green_rows, green_colors, GREEN, CENTER_COL)
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=RIGHT_COL,
        inward_col=RIGHT_INNER_COL,
    )

    teal_input = _derive_input_positions(teal_out, ascending=True)
    if teal_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    green_input = None
    for _ in range(INPUT_COL_RETRIES):
        candidate = _derive_center_input_positions(green_out)
        if candidate is None:
            return None
        all_positions = teal_input + red_input + candidate
        if not _any_pairwise_overlap(all_positions):
            green_input = candidate
            break
    if green_input is None:
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in teal_input + green_input + red_input:
        paste(inp, p, r, c)

    try:
        if np.array_equal(main(inp), output):
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
