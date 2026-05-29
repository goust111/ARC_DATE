# Combo variant of b5ca7ac4 (index 9)
# combo_spec: D7/1+D10/0
# Name: 行位置镜像+紧密裁剪输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The objects are unchanged: 5x5 framed squares with frame color 8 (teal) or 2 (red),
#      each containing a uniform 3x3 interior color.
#    - The original task only changed columns. Here the full-canvas move is:
#      first determine edge vs inward exactly as before, but paste every square at the mirrored
#      top row `H - SQ - r` instead of `r`; then tight-crop away outer background rows/cols.
#    - Example trace: if a teal square starts at input row 2 and wins the edge slot, the legacy
#      output would place it at row 2, col 0. The combo places it at row 15, col 0 on the full
#      22x22 canvas, and if the occupied rows span only 5..19, the returned output becomes rows
#      5..19 of that canvas. Relative vertical order is therefore reversed before cropping.
# b) Complexity calibration:
#    - The task still uses only two frame groups and two landing depths per side, so it does not
#      balloon into a many-case puzzle. The visible change comes from two orthogonal effects:
#      vertical mirroring changes where each tracked square lands, and tight cropping removes the
#      now-empty top/bottom bands. Unique interior colors within each frame group keep squares
#      distinguishable when their rows reverse.
# c) Hidden-assumption audit of `main()`:
#    - 5x5 window scanning still assumes every object is an exact framed square; unchanged.
#    - The original code assumed row bookkeeping could use the input row directly. That is false
#      now: overlap checks for edge occupancy must use mirrored row spans, because edge vs inward
#      is decided in output space.
#    - The original `return out` assumed fixed 22x22 output. The combo replaces that with a final
#      non-background crop, so full-canvas rendering and returned output shape are decoupled.
#    - Cropping assumes at least one detected square exists; the generator preserves the original
#      task's non-empty object counts and validates every sample with the combo solver.
# d) Placement legality:
#    - The generator remains output-first. It chooses legal mirrored output rows inside a one-row
#      top/bottom margin, renders the full 22x22 destination, then derives input rows by the inverse
#      mirror `input_r = H - SQ - output_r`.
#    - Non-overlap is checked after input columns are assigned. Cross-side vertical overlap is safe
#      because teal occupies columns 0/5 and red occupies columns 17/12 in the full output.
# e) Visual observability:
#    - Samples are forced to contain squares in both upper and lower parts of the input, so the row
#      flip is visible rather than resembling a uniform shift.
#    - Output rows never touch the very top or bottom border of the 22x22 canvas, so cropping is
#      always non-trivial and can be inferred from a few examples.
#    - The generator also rejects degenerate cases where combo output accidentally matches the
#      row-preserving tight-crop baseline on the same input.
# f) Visual cleanliness:
#    - The layout still uses only two column bands per side, preventing clutter. Tight cropping
#      removes empty border bands instead of adding new markings, so the final picture stays tidy.
# g) Color inference:
#    - No new color remapping is introduced. Colors continue to encode only frame group membership
#      and square identity via distinct interiors, which is directly visible.
# Post-change solver assumptions checked against the generator:
#    - Every placed object is a legal 5x5 frame with border color 2 or 8 and a non-frame interior.
#    - There are at most two landing depths per side; inward rows overlap some edge row on that side.
#    - Mirrored output rows lie inside the crop-visible margin band, and inverse mirroring keeps all
#      input rows in-bounds.
#    - Generated samples satisfy both observability metrics: cropped height < 22 and combo output
#      differs from the row-preserving cropped baseline.


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
GEN_RETRIES = 500
ROW_MARGIN = 1


def _mirror_row(r0: int, height: int = GRID_H) -> int:
    """Mirror the top row of a SQxSQ square within a height-sized canvas."""
    return int(height - SQ - r0)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    """Remove border rows/cols consisting entirely of the background color."""
    non_bg = np.argwhere(grid != int(bg))
    if non_bg.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0 = non_bg.min(axis=0)
    r1, c1 = non_bg.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


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


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ framed patch with a uniform inner fill."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _find_squares(grid: np.ndarray) -> list[dict[str, object]]:
    """Detect all valid 5x5 framed squares by scanning windows."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    squares: list[dict[str, object]] = []
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
    return squares


def _render_full_output(grid: np.ndarray, *, mirror_rows: bool) -> np.ndarray:
    """Render the rearranged 22x22 canvas before any cropping."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return not (int(a1) < int(b0) or int(b1) < int(a0))

    squares = _find_squares(g)

    azure_squares = sorted((s for s in squares if int(s["frame"]) == AZURE), key=lambda s: int(s["c0"]))
    placed_at_edge: list[tuple[int, int]] = []
    for s in azure_squares:
        base_r = int(s["r0"])
        dest_r = _mirror_row(base_r, H) if mirror_rows else base_r
        dest_r1 = int(dest_r + SQ - 1)
        has_overlap = any(overlaps(dest_r, dest_r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = SQ if has_overlap else 0
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == 0:
            placed_at_edge.append((dest_r, dest_r1))

    red_squares = sorted((s for s in squares if int(s["frame"]) == RED), key=lambda s: -int(s["c0"]))
    placed_at_edge = []
    for s in red_squares:
        base_r = int(s["r0"])
        dest_r = _mirror_row(base_r, H) if mirror_rows else base_r
        dest_r1 = int(dest_r + SQ - 1)
        has_overlap = any(overlaps(dest_r, dest_r1, pr0, pr1) for pr0, pr1 in placed_at_edge)
        dest_c = int(W - 2 * SQ) if has_overlap else int(W - SQ)
        paste(out, s["patch"], dest_r, dest_c, transparent=bg)
        if dest_c == int(W - SQ):
            placed_at_edge.append((dest_r, dest_r1))

    return out


def _render_row_preserving_cropped(grid: np.ndarray) -> np.ndarray:
    """Baseline used only for generator anti-degeneracy checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    return _tight_crop_non_bg(_render_full_output(g, mirror_rows=False), bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack framed 5x5 squares to the side walls, mirror their rows, then tight-crop.

    Input structure:
    - The background is the most frequent color.
    - Teal-framed (8) and red-framed (2) 5x5 squares with non-frame interiors are scattered on the canvas.

    Rule:
    - Teal squares are processed from smaller input column to larger and move toward the left side; red squares are processed from larger input column to smaller and move toward the right side.
    - Each square's top row is mirrored before placement: a square starting at row r lands at row H-5-r.
    - If a square's mirrored row-span overlaps an already placed edge square of the same side, it is shifted one square-width inward instead of occupying the wall column.
    - After the full 22x22 mirrored arrangement is built, delete every outer row and column that is entirely background.

    Key insight: The same frame-color grouping and near-edge-first stacking rule still decide left/right edge versus inward placement, but the rows are vertically flipped first and the returned answer is the minimal non-background crop of that mirrored layout.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    full_output = _render_full_output(g, mirror_rows=True)
    return _tight_crop_non_bg(full_output, bg)


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Choose legal mirrored output rows, leaving a blank border for cropping."""
    max_inward = min(MAX_INWARD, n - 1)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

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
    """Render a side's squares into the full output and return placement metadata."""
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
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns and inverse-mirrored rows that reproduce the target output."""
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
                    for dep_idx in inward_deps[iw_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep_idx
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
        for i, (out_r, out_c, patch) in enumerate(ordered_squares):
            input_r = _mirror_row(out_r)
            input_c = chosen_cols[i]
            if input_c == out_c:
                ok = False
                break
            result.append((input_r, input_c, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _has_visible_flip(rows: list[int]) -> bool:
    """Require squares on both sides of the mirror split so the inversion is visible."""
    return any(r <= 8 for r in rows) and any(r >= 9 for r in rows)


def _combo_metrics(inp: np.ndarray) -> tuple[int, int]:
    """Return (cropped_rows_removed, differing_cells_vs_row_preserving_crop)."""
    combo = main(inp)
    baseline = _render_row_preserving_cropped(inp)
    cropped_rows_removed = int(GRID_H - combo.shape[0])
    if combo.shape != baseline.shape:
        differing_cells = int(combo.size + baseline.size)
    else:
        differing_cells = int(np.count_nonzero(combo != baseline))
    return cropped_rows_removed, differing_cells


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the combo variant."""
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
    azure_out = _place_side_squares(full_output, azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_out = _place_side_squares(
        full_output,
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        GRID_W - SQ,
        GRID_W - 2 * SQ,
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

    input_rows = [r for r, _, _ in all_positions]
    if not _has_visible_flip(input_rows):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for r, c, patch in all_positions:
        paste(inp, patch, r, c)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected_output):
        return None

    cropped_rows_removed, differing_cells = _combo_metrics(inp)
    if cropped_rows_removed < 2 or differing_cells == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
