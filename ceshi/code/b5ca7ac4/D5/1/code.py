# Variant of b5ca7ac4
# Dimension: D5 — 目标位置——堆叠的边缘方向 (dst)
# Change: 青色组向左边缘堆叠（首个框架左边对齐col 0），红色组向右边缘堆叠（首个框架右边对齐col max）... → 两组都向上/下边缘堆叠

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


# Design analysis for the D5 variant
# ----------------------------------
# concrete_behavior:
#   objects:
#     - The input still contains scattered 5x5 hollow squares with frame color 8 or 2.
#     - Each square still has a visible 3x3 solid interior whose color differs from the frame and bg.
#   original_vs_variant:
#     - Original behavior kept each square's row and changed only its column, so overlap tests were done on rows.
#     - This variant keeps each square's column and changes only its row, so overlap tests must move to columns.
#   trace_example:
#     - Suppose two azure squares start at columns 3 and 5, with input top rows 2 and 9.
#     - Original logic would compare their rows and send them to col 0 or 5.
#     - Variant logic sorts them by top row (2 before 9), sends the first to row 0 at its own column,
#       then checks whether the second square's 5-column span overlaps the first. If yes it goes to row 5,
#       otherwise it also stays on row 0.
#     - Red squares mirror this from the bottom edge: row H-5 for default placement, row H-10 when overlapping.
# complexity_calibration:
#   - Keep the same object type, counts, colors, and 2-layer depth so the task remains comparable to the seed.
#   - Force at least one inward square per color group so the second vertical layer is visible in every sample.
#   - Use sorting by visible geometry only: azure by input row ascending, red by input row descending.
# implicit_assumption_audit:
#   - Detection by scanning 5x5 windows is still valid because object shape is unchanged.
#   - The original `placed_at_default` stored row intervals; that assumption fails after switching to vertical motion.
#     The new solver stores column intervals instead.
#   - The original destination columns (`0`, `5`, `W-5`, `W-10`) assume horizontal relocation.
#     They are replaced with destination rows (`0`, `5`, `H-5`, `H-10`).
#   - The original processing order depended on proximity to the target side along the horizontal axis.
#     After changing the destination edge direction, that order must become vertical: top-first for azure,
#     bottom-first for red.
# placement_legality:
#   - Output construction uses fixed row bands: azure on rows 0 or 5, red on rows H-5 or H-10.
#   - Within one band, squares are assigned non-overlapping columns; inward squares must column-overlap an edge square.
#   - Top and bottom bands are separated by at least two blank rows on a 22x22 grid, so azure/red outputs cannot collide.
# observability_and_cleanliness:
#   - The change is visually obvious because output motion is vertical instead of horizontal.
#   - Columns remain meaningful anchors, making it easy to see that squares slide to the top or bottom while keeping x-position.
#   - Uniform 3x3 interiors are retained; no hidden color-based ordering is introduced.
# generator_guarantees_for_main:
#   - Input rows are chosen to realize the intended processing order while keeping columns fixed.
#   - Any square meant for the inward row is placed after at least one overlapping edge square in row order.
#   - Validation runs `main(input)` against the synthesized output, so any missed assumption rejects the sample.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered 5x5 squares to vertical edge bands chosen by border color.

    Input structure:
    - bg = most common color.
    - Teal (8) and red (2) bordered 5x5 squares with colored interiors are scattered across the grid.

    Rule:
    - Teal squares keep their original column and are pushed toward the top edge; red squares keep their original column and are pushed toward the bottom edge.
    - Process teal squares from top to bottom and red squares from bottom to top.
    - If a square's columns overlap an already-placed default-band square of the same color, place it one square-height inward instead.

    Key insight: The target edge direction has rotated, so the deciding geometry also rotates: row order picks the placement sequence, and column-span overlap decides whether a square stays on the edge band or drops into the second band.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    out = np.full_like(g, bg)

    square_size = 5

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = []
    for r0 in range(H - square_size + 1):
        for c0 in range(W - square_size + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + square_size - 1), int(c0 + square_size - 1)))

            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            if int(border.min()) != int(border.max()):
                continue
            frame_col = int(border[0])
            if frame_col not in {2, 8}:
                continue

            inner = patch[1:-1, 1:-1]
            if not bool((inner != int(frame_col)).any()):
                continue

            squares.append({"frame": frame_col, "r0": int(r0), "c0": int(c0), "patch": patch})

    teal_squares = sorted([s for s in squares if int(s["frame"]) == 8], key=lambda s: int(s["r0"]))
    placed_at_default: list[tuple[int, int]] = []
    for s in teal_squares:
        c0 = int(s["c0"])
        c1 = int(c0 + square_size - 1)
        has_overlap = any(overlaps(c0, c1, pc0, pc1) for pc0, pc1 in placed_at_default)
        dest_r = int(square_size) if has_overlap else 0
        paste(out, s["patch"], dest_r, c0, transparent=bg)
        if dest_r == 0:
            placed_at_default.append((c0, c1))

    red_squares = sorted([s for s in squares if int(s["frame"]) == 2], key=lambda s: -int(s["r0"]))
    placed_at_default = []
    for s in red_squares:
        c0 = int(s["c0"])
        c1 = int(c0 + square_size - 1)
        has_overlap = any(overlaps(c0, c1, pc0, pc1) for pc0, pc1 in placed_at_default)
        dest_r = int(H - 2 * square_size) if has_overlap else int(H - square_size)
        paste(out, s["patch"], dest_r, c0, transparent=bg)
        if dest_r == int(H - square_size):
            placed_at_default.append((c0, c1))

    return out


# --------------- generator (c) ---------------

# --- Configuration ---
GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_ROW_RETRIES = 50
GEN_RETRIES = 500


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with uniform interior."""
    p = np.full((SQ, SQ), frame_color, dtype=int)
    p[1:-1, 1:-1] = inner_color
    return p


def _spans_overlap(a: int, b: int) -> bool:
    """Check whether two SQ-long 1D spans overlap."""
    return abs(a - b) < SQ


def _rects_overlap(r1, c1, r2, c2) -> bool:
    """Check whether two SQxSQ rectangles overlap."""
    return abs(r1 - r2) < SQ and abs(c1 - c2) < SQ


def _any_pairwise_overlap(positions: list[tuple]) -> bool:
    """Return True if any two (r, c, ...) entries overlap as SQxSQ rects."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _assign_output_cols(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge-band and inward-band column positions for one color group."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_cols = list(range(GRID_W - SQ + 1))
    random.shuffle(all_cols)
    edge_cols: list[int] = []
    for c in all_cols:
        if all(not _spans_overlap(c, ec) for ec in edge_cols):
            edge_cols.append(c)
            if len(edge_cols) == n_edge:
                break
    if len(edge_cols) < n_edge:
        return None

    random.shuffle(all_cols)
    inward_cols: list[int] = []
    for c in all_cols:
        if not any(_spans_overlap(c, ec) for ec in edge_cols):
            continue
        if any(_spans_overlap(c, ic) for ic in inward_cols):
            continue
        inward_cols.append(c)
        if len(inward_cols) == n_inward:
            break
    if len(inward_cols) < n_inward:
        return None

    return edge_cols, inward_cols


def _place_band_squares(
    output: np.ndarray,
    edge_cols: list[int],
    inward_cols: list[int],
    colors: list[int],
    frame_color: int,
    edge_row: int,
    inward_row: int,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Render one color group's top/bottom edge and inward bands onto the output."""
    placed: list[tuple[int, int, bool, np.ndarray]] = []
    for i, c in enumerate(edge_cols):
        p = _make_patch(frame_color, colors[i])
        paste(output, p, edge_row, c)
        placed.append((edge_row, c, True, p))
    for i, c in enumerate(inward_cols):
        p = _make_patch(frame_color, colors[len(edge_cols) + i])
        paste(output, p, inward_row, c)
        placed.append((inward_row, c, False, p))
    return placed


def _derive_input_positions(
    sq_list: list[tuple[int, int, bool, np.ndarray]],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign random input rows that reproduce the solver's top/bottom band placement."""
    edges = [(r, c, p) for (r, c, is_edge, p) in sq_list if is_edge]
    inwards = [(r, c, p) for (r, c, is_edge, p) in sq_list if not is_edge]

    inward_deps = [[i for i, (_, ec, _) in enumerate(edges) if _spans_overlap(ic, ec)] for (_, ic, _) in inwards]

    n = len(sq_list)
    row_range = list(range(GRID_H - SQ + 1))
    if len(row_range) < n:
        return None

    for _ in range(INPUT_ROW_RETRIES):
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

        ordered_squares = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_rows = sorted(random.sample(row_range, n))
        if not ascending:
            chosen_rows = chosen_rows[::-1]

        result = []
        ok = True
        for i, (orow, c, p) in enumerate(ordered_squares):
            irow = chosen_rows[i]
            if irow == orow:
                ok = False
                break
            result.append((irow, c, p))
        if not ok:
            continue

        if not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    bg = random.choice([c for c in range(10) if c not in {AZURE, RED}])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_assign = _assign_output_cols(n_azure)
    if azure_assign is None:
        return None
    red_assign = _assign_output_cols(n_red)
    if red_assign is None:
        return None
    azure_edge_cols, azure_inward_cols = azure_assign
    red_edge_cols, red_inward_cols = red_assign

    avail_azure = [c for c in range(10) if c not in {bg, AZURE}]
    avail_red = [c for c in range(10) if c not in {bg, RED}]
    if len(avail_azure) < n_azure or len(avail_red) < n_red:
        return None
    azure_colors = random.sample(avail_azure, n_azure)
    red_colors = random.sample(avail_red, n_red)

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    azure_out = _place_band_squares(output, azure_edge_cols, azure_inward_cols, azure_colors, AZURE, edge_row=0, inward_row=SQ)
    red_out = _place_band_squares(
        output,
        red_edge_cols,
        red_inward_cols,
        red_colors,
        RED,
        edge_row=GRID_H - SQ,
        inward_row=GRID_H - 2 * SQ,
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
