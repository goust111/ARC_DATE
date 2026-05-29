# Combo variant of b5ca7ac4 (index 16)
# combo_spec: D8/0+D16/0
# Name: 边框统一化+输出背景色变换

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the fused variant:
# - Behavior trace:
#   Original b5ca7ac4 keeps each 5x5 framed square unchanged while packing teal frames to the left
#   and red frames to the right. This combo keeps the same grouping, ordering, row preservation,
#   and overlap-driven inward shift, but changes the rendered result after the move:
#   every moved border becomes gray and the whole output background becomes black.
#   A concrete trace is: teal-framed patch at (r, c) -> still routed by its teal input border,
#   but once pasted into the output it appears as a gray-framed patch on black background.
# - Complexity calibration:
#   The output border color is fixed to a single visible color (gray=5) and the output background
#   is fixed to black (0). Input backgrounds are sampled only from non-black, non-gray colors so the
#   background swap is always visible and cannot cancel out the border recolor.
# - Hidden assumptions audit:
#   The detector still assumes the input contains intact 5x5 windows whose borders are uniformly
#   teal (8) or red (2); the generator preserves that exactly.
#   The mover still assumes only two landing columns per side (edge or one-square inward); the row
#   assignment logic guarantees that no third stack depth is required.
#   Because output recoloring happens after grouping, the solver must never infer groups from output
#   colors; it derives them only from the input frames.
# - Placement legality:
#   The generator remains output-first. It chooses non-overlapping edge rows, inward rows that
#   overlap an edge row, places gray-bordered output patches at legal columns, and then reverse-
#   engineers input columns whose visible left-to-right/right-to-left order reproduces that layout.
# - Observability and tidiness:
#   The fused change is obvious in 2-4 examples: empty space flips from a non-black color to black,
#   and both frame groups lose their original group colors in the output. Inner 3x3 colors stay
#   colored so the objects remain readable even after the border colors collapse to one class.


# --- Configuration ---
GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
UNIFIED_FRAME = 5
OUTPUT_BG = 0
INPUT_BG_CHOICES = (1, 3, 4, 6, 7, 9)
INNER_COLOR_POOL = (1, 3, 4, 6, 7, 9)
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack colored 5x5 frames to the side walls, then erase the original frame coding.

    Input structure:
    - The dominant input color is the background, and it is different from black.
    - Teal (8) and red (2) 5x5 bordered squares are scattered around the grid.
    - Each square has a 3x3 interior that is not part of the border signal.

    Rule:
    - Detect squares by their input border color, keeping each square's original row.
    - Teal squares are processed from smaller input columns to larger ones and move to the left side;
      red squares are processed from larger input columns to smaller ones and move to the right side.
    - For either group, a square goes to the edge column unless its 5-row span overlaps a square
      already placed on that side's edge column, in which case it shifts inward by one square width.
    - The output canvas uses black background, and every moved border is recolored to gray.

    Key insight:
    The red/teal borders matter only while reading the input. After the move, both groups share the
    same gray frame, so the solver must remember the input-side grouping instead of relying on the
    output colors.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    input_bg = int(get_bg(g))

    out = np.full((H, W), OUTPUT_BG, dtype=int)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    def to_output_patch(patch: np.ndarray, frame_color: int) -> np.ndarray:
        recolored = np.array(patch, copy=True)
        recolored[recolored == int(frame_color)] = int(UNIFIED_FRAME)
        return recolored

    squares = []
    for r0 in range(H - SQ + 1):
        for c0 in range(W - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue

            frame_color = int(border[0])
            if frame_color not in {AZURE, RED}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != int(frame_color)).any()):
                continue

            squares.append({"frame": frame_color, "r0": int(r0), "c0": int(c0), "patch": patch})

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_edge_rows: list[tuple[int, int]] = []
    for sq in azure_squares:
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_edge_rows)
        dest_c = SQ if has_overlap else 0
        paste(out, to_output_patch(sq["patch"], AZURE), r0, dest_c, transparent=input_bg)
        if dest_c == 0:
            placed_edge_rows.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_edge_rows = []
    for sq in red_squares:
        r0 = int(sq["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_edge_rows)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, to_output_patch(sq["patch"], RED), r0, dest_c, transparent=input_bg)
        if dest_c == int(W - SQ):
            placed_edge_rows.append((r0, r1))

    return out


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
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


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    max_inward = min(MAX_INWARD, int(n) - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = int(n) - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for r in all_rows:
        if all(not _rows_overlap(r, edge_r) for edge_r in edge_rows):
            edge_rows.append(int(r))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    random.shuffle(all_rows)
    inward_rows: list[int] = []
    for r in all_rows:
        if not any(_rows_overlap(r, edge_r) for edge_r in edge_rows):
            continue
        if any(_rows_overlap(r, inward_r) for inward_r in inward_rows):
            continue
        inward_rows.append(int(r))
        if len(inward_rows) == n_inward:
            break
    if len(inward_rows) < n_inward:
        return None

    return edge_rows, inward_rows


def _place_side_squares(
    output: np.ndarray,
    edge_rows: list[int],
    inward_rows: list[int],
    inner_colors: list[int],
    input_frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for idx, row in enumerate(edge_rows):
        input_patch = _make_patch(input_frame_color, inner_colors[idx])
        output_patch = _make_patch(UNIFIED_FRAME, inner_colors[idx])
        paste(output, output_patch, row, edge_col)
        placed.append((int(row), int(edge_col), True, input_patch))

    offset = len(edge_rows)
    for idx, row in enumerate(inward_rows):
        input_patch = _make_patch(input_frame_color, inner_colors[offset + idx])
        output_patch = _make_patch(UNIFIED_FRAME, inner_colors[offset + idx])
        paste(output, output_patch, row, inward_col)
        placed.append((int(row), int(inward_col), False, input_patch))

    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    edges = [(row, out_col, patch) for row, out_col, is_edge, patch in sq_list if is_edge]
    inwards = [(row, out_col, patch) for row, out_col, is_edge, patch in sq_list if not is_edge]
    inward_deps = [
        [idx for idx, (edge_row, _edge_col, _patch) in enumerate(edges) if _rows_overlap(inward_row, edge_row)]
        for inward_row, _out_col, _patch in inwards
    ]

    n = len(sq_list)
    col_range = list(range(GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
            latest_dep_pos = max(
                (
                    pos
                    for dep_idx in inward_deps[inward_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep_idx
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_dep_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]
        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        valid = True
        for idx, (row, out_col, patch) in enumerate(ordered):
            in_col = int(chosen_cols[idx])
            if in_col == int(out_col):
                valid = False
                break
            result.append((int(row), in_col, patch))
        if not valid:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    input_bg = random.choice(INPUT_BG_CHOICES)
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_rows(n_azure)
    red_assign = _assign_output_rows(n_red)
    if azure_assign is None or red_assign is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_assign
    red_edge_rows, red_inward_rows = red_assign

    avail_inner = [color for color in INNER_COLOR_POOL if color != int(input_bg)]
    if len(avail_inner) < max(n_azure, n_red):
        return None
    azure_colors = random.sample(avail_inner, n_azure)
    red_colors = random.sample(avail_inner, n_red)

    output = np.full((GRID_H, GRID_W), OUTPUT_BG, dtype=int)
    azure_out = _place_side_squares(
        output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        edge_col=0,
        inward_col=SQ,
    )
    red_out = _place_side_squares(
        output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )

    azure_input = _derive_input_positions(azure_out, ascending=True)
    red_input = _derive_input_positions(red_out, ascending=False)
    if azure_input is None or red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), input_bg, dtype=int)
    for row, col, patch in all_positions:
        paste(inp, patch, row, col)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        maybe = _gen_attempt()
        if maybe is not None:
            return maybe
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
