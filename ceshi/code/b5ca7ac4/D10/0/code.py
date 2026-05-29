# Variant of b5ca7ac4
# Dimension: D10 — 输出网格尺寸 (grid_shape)
# Change: 输出网格与输入网格尺寸相同（22x22），框架移动后空出的原位置变为背景色... → 输出紧密裁剪——去除全背景行列

from __future__ import annotations

import random
import numpy as np
from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the D10 variant
# a) Concrete behavior and trace:
#    The movable objects are still 5x5 framed squares with frame color 8 or 2.
#    The solver still relocates teal squares to columns 0 or 5 and red squares
#    to columns W-5 or W-10 while preserving each square's original top row.
#    The only new behavior is after this full 22x22 arrangement is rendered:
#    remove every all-background border row and column, returning the smallest
#    rectangle that contains all moved squares.
#    Example: if the topmost relocated square starts at row 3 and the bottommost
#    square ends at row 18, the legacy solver would keep rows 0..21, while this
#    variant returns rows 3..18 only. Because both color groups still occupy the
#    left and right edges, width usually remains 22; the visible D10 effect is
#    a dynamically cropped height.
# b) Complexity calibration:
#    No new object type, color mapping, or ordering rule is added. This keeps the
#    task at the same conceptual level as the original: detect squares, order them,
#    stack them, then crop. To make the crop visible, the generator reserves at
#    least one blank row above and below the final arrangement.
# c) Implicit-assumption audit of main():
#    - Square detection by scanning 5x5 windows still assumes every object is
#      exactly a 5x5 framed square. The variant does not change this.
#    - Placement columns 0/5 and W-5/W-10 still assume a 22-column canvas and
#      max depth 2; unchanged.
#    - The old unconditional `return out` relied on fixed output size. That is
#      the one D10-sensitive line: it must become `tight_crop(out, bg)`.
#    - Cropping assumes at least one square exists, which the generator enforces.
# d) Placement legality:
#    Input squares remain non-overlapping. Output stacking is unchanged, so teal
#    squares only use columns 0 or 5, red squares only use columns 12 or 17, and
#    overlapping rows across opposite sides are safe because those column bands
#    are disjoint. The crop is computed after legal placement, so it cannot create
#    overlap or out-of-bounds issues.
# e) Visual observability:
#    Humans can see the new rule because the final picture is vertically trimmed.
#    The generator avoids placing any relocated square flush with the top or bottom
#    of the 22x22 canvas, so the crop is almost always non-trivial and not merely
#    an implementation detail.
# f) Visual neatness:
#    The cropped output removes empty top/bottom bands, so the result is actually
#    cleaner than the original full-canvas rendering. The square layout itself is
#    unchanged, preventing any new clutter.
# g) Color-rule audit:
#    No color remapping is introduced. Colors are still only used as visible frame
#    categories (teal vs red), which is directly inferable from the examples.
# New assumptions introduced by the variant:
#    `main()` now assumes the background color stays uniform outside all squares
#    so that edge rows/cols can be deleted safely. `generate_input()` guarantees
#    this by keeping the original canvas background uniform and by validating every
#    generated sample with the new cropped solver.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500
ROW_MARGIN = 1


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Crop away border rows/cols consisting entirely of the background color."""
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _render_full_output(grid: np.ndarray) -> np.ndarray:
    """Apply the original square-moving rule on the full 22x22 canvas."""
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
            if not bool((inner != int(frame_col)).any()):
                continue
            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((r0, r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for s in red_squares:
        r0 = int(s["r0"])
        r1 = int(r0 + SQ - 1)
        has_overlap = any(overlaps(r0, r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((r0, r1))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack bordered 5x5 squares against the side walls, then tight-crop the result.

    Input structure:
    - Background is the most frequent color.
    - Teal-framed (8) and red-framed (2) 5x5 squares with non-frame interiors are scattered on the canvas.

    Rule:
    - Teal squares keep their rows and stack toward the left wall; red squares keep their rows and stack toward the right wall.
    - Teal is processed from smaller input column to larger, red from larger to smaller.
    - A square that row-overlaps an already placed wall square is shifted one square-width inward.
    - After all squares are placed on the full canvas, delete any outer rows and columns that are entirely background.

    Key insight: The move logic is unchanged from the original task; the variant is that the final answer is the minimal non-background crop of that rearranged canvas.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = _render_full_output(g)
    return _tight_crop_non_bg(out, bg)


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Check whether two SQ-tall row spans overlap."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, object]]) -> bool:
    """Return True if any two (r, c, ...) entries overlap as SQxSQ rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward row positions for one side, leaving crop-visible margins."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    # Reserve one empty border row above and below the final arrangement so D10 is visible.
    all_rows = list(range(ROW_MARGIN, GRID_H - SQ - ROW_MARGIN + 1))
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
    """Render edge and inward squares onto output, returning placement metadata."""
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
    """Assign random input columns that reproduce the solver's wall/inward placement."""
    edges = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if is_edge]
    inwards = [(r, oc, p) for (r, oc, is_edge, p) in sq_list if not is_edge]

    inward_deps = [
        [i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)]
        for (ir, _, _) in inwards
    ]

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

        ordered_squares = [
            (edges[idx] if typ == "edge" else inwards[idx])
            for typ, idx in order
        ]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result = []
        ok = True
        for i, (r, oc, p) in enumerate(ordered_squares):
            ic = chosen_cols[i]
            if ic == oc:
                ok = False
                break
            result.append((r, ic, p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the cropped-output variant."""
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

    full_output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_side_squares(
        full_output,
        azure_edge_rows,
        azure_inward_rows,
        azure_colors,
        AZURE,
        edge_col=0,
        inward_col=SQ,
    )
    red_out = _place_side_squares(
        full_output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        edge_col=GRID_W - SQ,
        inward_col=GRID_W - 2 * SQ,
    )
    expected_output = _tight_crop_non_bg(full_output, bg)

    azure_input = _derive_input_positions(azure_out, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_out, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, p in all_positions:
        paste(inp, p, r, c)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected_output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
