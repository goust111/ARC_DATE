# Variant of b5ca7ac4
# Dimension: D7 — 行位置保持规则 (meta)
# Change: 每个框架在输出中保持其在输入中的行位置不变，仅列位置发生改变（推向边缘或偏移）... → 行位置取反——上下翻转

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D7 variant
# a) Concrete behavior:
#    - Objects stay the same: 5x5 framed squares, frame color 8 or 2, colored 3x3 center.
#    - The original task only changed columns. This variant also mirrors every square's top row:
#      output_r = H - SQ - input_r. With H=22 and SQ=5, row 0 goes to 17, row 3 goes to 14, etc.
#    - Example trace: two teal squares at input rows 2 and 4 still compete for edge vs inward by the
#      same column ordering as before, but they are pasted at output rows 15 and 13 instead of 2 and 4.
#      The overlap relation survives the flip, while the vertical order is reversed.
# b) Complexity calibration:
#    - The task still has only two frame groups and two output columns per side, so the core logic
#      remains compact. The row flip is visually strong because no 5x5 square has a fixed point under
#      r -> 17-r, and unique interior colors let a solver track individual squares.
# c) Hidden assumptions from the original solver:
#    - Window scanning for 5x5 framed squares is unaffected; shape and colors do not change.
#    - The original code pasted at r0 directly. That assumption is invalid here; all paste rows and
#      edge-occupancy spans must use the mirrored row instead.
#    - Row-overlap tests still work after flipping because the transform preserves interval overlap.
# d) Placement legality:
#    - The generator works output-first, chooses valid mirrored output rows, then derives input rows by
#      the inverse mapping input_r = H - SQ - output_r. This keeps every square in-bounds automatically.
#    - Non-overlap is checked after columns are assigned on the input grid, and solver validation
#      confirms the produced input reproduces the intended output exactly.
# e) Visual observability:
#    - Samples are required to contain squares in both the upper and lower parts of the input, so the
#      inversion is visible in both directions rather than looking like a one-sided downward shift.
# f) Visual cleanliness:
#    - Only the same two horizontal landing bands are used per side (edge and one inward column), so
#      the output stays as tidy as the original task; the only new visual change is vertical inversion.
# g) Color inference:
#    - No new color mapping is introduced. Colors still only signal group membership by frame color,
#      which remains directly visible.
# Post-change solver assumptions checked against the generator:
#    - Every valid square is exactly 5x5 with a uniform border of color 2 or 8.
#    - There are at most two landing depths per side; generator constructs edge rows with no mutual
#      overlap and inward rows that overlap at least one edge row.
#    - Mirrored output rows are the intended occupancy rows; generator chooses them explicitly.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def _mirror_row(r0: int, height: int = GRID_H) -> int:
    """Mirror a SQ-tall top row across the vertical axis of the canvas."""
    return int(height - SQ - r0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5x5 squares to side-specific columns, while vertically mirroring their rows.

    Input structure:
    - The background is the most frequent color.
    - Teal (8) and red (2) 5x5 framed squares with non-frame interior colors are scattered on the grid.

    Rule:
    - Teal squares go to the left side and red squares go to the right side, using the same edge/inward
      columns as the original task.
    - Each square's top row is mirrored: a square starting at row r is pasted at row H-5-r.
    - Teal squares are processed by increasing input column; red squares by decreasing input column.
    - If a mirrored row-span overlaps an already placed edge square on that side, the square goes one
      square-width inward instead of staying on the edge.

    Key insight: Border color still chooses the destination side, but the old row-preservation rule is
    replaced by a vertical mirror, so column ordering decides edge vs inward and the row is flipped.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
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
            if frame_col not in {AZURE, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != frame_col).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in azure_squares:
        dest_r = _mirror_row(int(s["r0"]), H)
        dest_r1 = int(dest_r + SQ - 1)
        has_overlap = any(overlaps(dest_r, dest_r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_default.append((dest_r, dest_r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_default = []
    for s in red_squares:
        dest_r = _mirror_row(int(s["r0"]), H)
        dest_r1 = int(dest_r + SQ - 1)
        has_overlap = any(overlaps(dest_r, dest_r1, pr0, pr1) for pr0, pr1 in placed_at_default)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_default.append((dest_r, dest_r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    return abs(r_a - r_b) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Choose mirrored output rows for edge and inward squares on one side."""
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


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render a side's squares into the output grid using already mirrored rows."""
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


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns and inverse-mirrored input rows that reproduce the output."""
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

        ordered_squares = [(edges[idx] if typ == "edge" else inwards[idx]) for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for i, (out_r, oc, p) in enumerate(ordered_squares):
            input_r = _mirror_row(out_r)
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((input_r, ic, p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _has_visible_flip(rows: list[int]) -> bool:
    """Require rows on both sides of the mirror axis so the inversion is easy to see."""
    return any(r <= 8 for r in rows) and any(r >= 9 for r in rows)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_rows(n_red)
    if red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(output, azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        GRID_W - SQ,
        GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    input_rows = [r for r, _, _ in all_positions]
    if not _has_visible_flip(input_rows):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
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
