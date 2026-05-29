# Combo variant of 136b0064 (index 6)
# combo_spec: D8/2+D12/1
# Name: 交替步进与对角线段

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concretization and trace
#    The input layout is still the original one: two fixed 3x3 glyph columns on
#    the left, a full-height divider at col 7, and one gray start pixel on the
#    right. The fused change happens only while decoding the glyph list.
#    - D8/2 changes the inter-glyph connector from a constant down-step into an
#      alternating state machine: glyph 0 steps down, glyph 1 steps right,
#      glyph 2 steps down, glyph 3 steps right, and so on.
#    - D12/1 changes area 4 from a short vertical stroke into a 3-cell 45-degree
#      descending diagonal. As in the single variant, the tile column supplies
#      the diagonal sign: left glyph column => down-right, right glyph column =>
#      down-left.
#    Concrete trace from start (0,1) with N=2 and areas [5,4,4]:
#      original rule:
#        idx0 area5: step to (1,1), draw right -> end (1,3)
#        idx1 area4: step to (2,3), draw down  -> end (3,3)
#        idx2 area4: step to (4,3), draw down  -> end (5,3)
#      fused rule:
#        idx0 area5: down-step to (1,1), draw right       -> end (1,3)
#        idx1 area4: right-step to (1,4), draw down-right -> end (3,6)
#        idx2 area4: down-step to (4,6), draw down-left   -> end (6,4)
#      The path is no longer a regular vertical staircase; it becomes a jagged
#      walk whose connectors alternate orientation and whose area-4 turns are
#      diagonal.
#
# b) Complexity calibration
#    The variant keeps the original four glyph templates and the original
#    shape->color binding, so the task still has repeated visible evidence.
#    Complexity rises through two small, structured changes rather than through
#    many ad-hoc cases: one 2-phase connector schedule and one special area-4
#    geometry. This is enough to make the output visibly different without
#    making the rule explosion hard to infer from a few examples.
#
# c) Review of old assumptions
#    - Old `r_cur += 1` before every glyph assumed the connector is always
#      vertical. That is now false; connector orientation depends on glyph index.
#    - Old area-4 logic assumed `c_cur` never changes during that segment. That
#      is now false because area 4 may slope left or right.
#    - Old generator could summarize each area by one fixed delta. That is now
#      false twice over: delta depends on both glyph parity (down-step vs
#      right-step) and, for area 4, on tile column.
#    - Old path cleanliness was mostly automatic because connectors only moved
#      downward. With right-steps, left-moving horizontal segments can backtrack
#      into the previous endpoint, so generation must reject reused cells.
#
# d) Placement legality
#    Glyph placement on the input side is unchanged and always uses the original
#    2 x N tile scaffold. Output legality is still checked cell-by-cell: every
#    painted cell from the fused trace must remain in rows 0..H-1 and cols 0..6.
#    Generation forbids arbitrary self-overlap, but it does allow one specific
#    local reuse: a right-step followed by a leftward segment may legitimately
#    pass back through the current endpoint. That single hinge reuse greatly
#    improves feasibility for larger N without turning the output into clutter.
#
# e) Visual observability
#    A human can see both new cues directly:
#    - odd-index connectors create rightward kinks instead of another downward
#      drop;
#    - plus glyphs produce visible diagonals, and their tile column predicts the
#      diagonal direction.
#    The generator enforces at least one area-4 glyph and rejects samples that
#    would match either single-change baseline, so both halves of the fusion are
#    exercised in the training distribution.
#
# f) Visual cleanliness
#    The output is still almost simple: the generator blocks all long-range
#    self-overlap and only permits immediate hinge reuse at the current endpoint.
#    This keeps the path legible while preserving enough freedom for the
#    alternating-step rule to produce larger, varied examples.
#
# g) Inferability of colors and changes
#    No new color mapping is introduced. Colors remain tied to the same visible
#    glyph shapes, so the solver only needs to infer new geometry and new
#    sequencing, not a hidden color rule.
#
# Post-change solver assumptions mirrored by the generator:
#    - exactly one full-height non-bg divider column exists;
#    - glyphs are monochromatic 3x3 components in the two original tile columns;
#    - there is one gray start pixel on the right that shifts into cols 0..6;
#    - glyph order is still visible column-major order;
#    - area 4 uses tile-column position as a visible left/right diagonal cue;
#    - generated traces may reuse only the current endpoint as an immediate
#      hinge; broader self-overlap is still excluded.
# ---------------------------------------------------------------------------


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _step_axis_from_index(idx: int) -> str:
    """Glyph connectors alternate: even index -> down, odd index -> right."""
    return "down" if int(idx) % 2 == 0 else "right"


def _diag_dc_from_glyph_col(glyph_c0: int, div_c: int) -> int:
    """Left tile column => down-right diagonal, right tile column => down-left diagonal."""
    return 1 if int(glyph_c0) < int(div_c // 2) else -1


def _diag_dc_from_index(idx: int, n_rows: int) -> int:
    """Instruction indices 0..N-1 come from the left tile column; N..2N-1 from the right."""
    return 1 if int(idx) < int(n_rows) else -1


def _trace_instruction_cells(
    r_cur: int,
    c_cur: int,
    area: int,
    *,
    step_axis: str,
    area4_mode: str,
    diag_dc: int | None = None,
) -> list[tuple[int, int]]:
    """Return the painted cells for one decoded glyph, including the connector endpoint."""
    rr = int(r_cur)
    cc = int(c_cur)

    if step_axis == "down":
        rr = int(rr + 1)
    else:
        assert step_axis == "right"
        cc = int(cc + 1)

    cells = [(int(rr), int(cc))]

    if int(area) == 4:
        if area4_mode == "vertical":
            rr = int(rr + 1)
            cells.append((int(rr), int(cc)))
            return cells
        assert area4_mode == "diag"
        assert diag_dc in (-1, 1)
        for _ in range(2):
            rr = int(rr + 1)
            cc = int(cc + int(diag_dc))
            cells.append((int(rr), int(cc)))
        return cells

    area_to_len = {5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}
    seg_len = int(area_to_len[int(area)])
    dir_h = int(area_to_dir[int(area)])
    for _ in range(seg_len - 1):
        cc = int(cc + dir_h)
        cells.append((int(rr), int(cc)))
    return cells


def _render_output_from_areas(
    n_rows: int,
    start_col: int,
    areas: list[int],
    *,
    alternate_step: bool,
    diagonal_area4: bool,
) -> np.ndarray:
    """Render an output canvas from a sampled instruction list under a chosen rule set."""
    h = 4 * int(n_rows) - 1
    out = np.zeros((h, LEFT_W), dtype=int)
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    out[int(r_cur), int(c_cur)] = GRAY

    for idx, area in enumerate(areas):
        step_axis = _step_axis_from_index(int(idx)) if alternate_step else "down"
        area4_mode = "diag" if diagonal_area4 else "vertical"
        diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if diagonal_area4 and int(area) == 4 else None
        cells = _trace_instruction_cells(
            int(r_cur),
            int(c_cur),
            int(area),
            step_axis=step_axis,
            area4_mode=area4_mode,
            diag_dc=diag_dc,
        )
        for rr, cc in cells:
            assert 0 <= int(rr) < int(h), f"render exits bottom: rr={rr}, H={h}"
            assert 0 <= int(cc) < LEFT_W, f"render exits canvas: cc={cc}, LEFT_W={LEFT_W}"
            out[int(rr), int(cc)] = int(_AREA_TO_COLOR[int(area)])
        r_cur, c_cur = map(int, cells[-1])

    return out


def _count_segment_type_changes(areas: list[int]) -> int:
    """Count switches between diagonal area-4 segments and non-diagonal horizontal segments."""
    kinds = [int(area) == 4 for area in areas]
    return sum(1 for i in range(1, len(kinds)) if bool(kinds[i]) != bool(kinds[i - 1]))


def _collect_glyphs(left: np.ndarray, *, bg: int, div_color: int | None) -> list[dict[str, int]]:
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        bbox = comp["bbox"]
        if bbox_h(bbox) == 3 and bbox_w(bbox) == 3:
            r0, c0, _r1, _c1 = map(int, bbox)
            glyphs.append({"color": int(color), "r0": int(r0), "c0": int(c0), "area": int(comp["area"])})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode tiled glyph instructions into a polyline with alternating connectors and diagonal turns.

    Input structure:
    - Left of a full-height divider: monochrome 3x3 glyphs arranged in two fixed tile columns.
    - Right of the divider: one gray (5) start pixel that becomes the path origin on the left canvas.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read glyphs in column-major tile order (top-to-bottom in the left tile column, then the right one).
    - Before each glyph, apply a 1-cell connector step whose direction alternates down, right, down, right...
    - Then draw the glyph-colored segment: area 5 -> right 3, area 6 -> left 4, area 7 -> left 2,
      and area 4 -> a 3-cell descending diagonal that slopes right in the left tile column and left
      in the right tile column.

    Key insight: The glyph shape still selects the segment type, but the path now depends on two visible
    context cues at once: global glyph parity controls the connector direction, while the tile column
    controls the plus glyph's diagonal slant.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((h, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg, div_color=div_color)

    for idx, glyph in enumerate(glyphs):
        color = int(glyph["color"])
        area = int(glyph["area"])
        step_axis = _step_axis_from_index(int(idx))
        diag_dc = _diag_dc_from_glyph_col(int(glyph["c0"]), int(div_c)) if area == 4 else None
        cells = _trace_instruction_cells(
            int(r_cur),
            int(c_cur),
            int(area),
            step_axis=step_axis,
            area4_mode="diag" if area == 4 else "vertical",
            diag_dc=diag_dc,
        )

        for rr, cc in cells:
            assert 0 <= int(rr) < int(h), f"segment exits bottom: rr={rr}, H={h}"
            assert 0 <= int(cc) < int(div_c), f"segment exits canvas: cc={cc}, div_c={div_c}"
            out[int(rr), int(cc)] = int(color)

        r_cur, c_cur = map(int, cells[-1])

    return out


# --------------- generator ---------------

# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus    -> diagonal, color 6
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond -> right 3, color 1
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross   -> left 4,  color 3
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U       -> left 2,  color 2
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_SEGMENT_TYPE_CHANGES = 2
MAX_ATTEMPTS = 500


def _segment_cells_from_index(r_cur: int, c_cur: int, area: int, idx: int, n_rows: int) -> list[tuple[int, int]]:
    """Trace one combo-rule glyph by global index instead of by extracted tile metadata."""
    diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if int(area) == 4 else None
    return _trace_instruction_cells(
        int(r_cur),
        int(c_cur),
        int(area),
        step_axis=_step_axis_from_index(int(idx)),
        area4_mode="diag" if int(area) == 4 else "vertical",
        diag_dc=diag_dc,
    )


def _choose_mandatory_diag_indices(n_rows: int) -> set[int]:
    """Force diagonals in both tile columns and across both connector parities."""
    first_half = list(range(0, int(n_rows)))
    second_half = list(range(int(n_rows), 2 * int(n_rows)))

    left_idx = random.choice(first_half)
    opposite_parity = [idx for idx in second_half if idx % 2 != left_idx % 2]
    right_idx = random.choice(opposite_parity if opposite_parity else second_half)
    return {int(left_idx), int(right_idx)}


def _has_fusion_signature(n_rows: int, start_col: int, areas: list[int]) -> bool:
    """
    Reject samples that collapse to either single-change baseline on the same input.

    combo            = alternating step + diagonal area-4
    D8-only baseline = alternating step + original vertical area-4
    D12-only baseline= original fixed down-step + diagonal area-4
    """
    combo = _render_output_from_areas(int(n_rows), int(start_col), list(areas), alternate_step=True, diagonal_area4=True)
    try:
        d8_only = _render_output_from_areas(
            int(n_rows),
            int(start_col),
            list(areas),
            alternate_step=True,
            diagonal_area4=False,
        )
    except Exception:
        d8_only = None
    try:
        d12_only = _render_output_from_areas(
            int(n_rows),
            int(start_col),
            list(areas),
            alternate_step=False,
            diagonal_area4=True,
        )
    except Exception:
        d12_only = None

    d8_differs = True if d8_only is None else (not np.array_equal(combo, d8_only))
    d12_differs = True if d12_only is None else (not np.array_equal(combo, d12_only))
    return d8_differs and d12_differs


def _sample_path(n_rows: int, start_col: int) -> list[int] | None:
    """
    Backtrack a non-self-overlapping combo-rule walk of length 2*N inside the output canvas.

    Every candidate is traced cell-by-cell under the fused rule itself, so legality and fusion
    observability are checked against the exact geometry that `main()` will later draw.
    """
    h = 4 * int(n_rows) - 1
    target_len = 2 * int(n_rows)
    occupied = {(GRAY_START_ROW, int(start_col))}
    areas: list[int] = []
    mandatory_diag = _choose_mandatory_diag_indices(int(n_rows))

    def rec(idx: int, r_cur: int, c_cur: int) -> bool:
        if int(idx) == target_len:
            if 4 not in areas:
                return False
            if len(set(areas)) < 2:
                return False
            if _count_segment_type_changes(areas) < MIN_SEGMENT_TYPE_CHANGES:
                return False
            return _has_fusion_signature(int(n_rows), int(start_col), areas)

        allowed = [4] if int(idx) in mandatory_diag else list(_AREAS)
        random.shuffle(allowed)

        for area in allowed:
            cells = _segment_cells_from_index(int(r_cur), int(c_cur), int(area), int(idx), int(n_rows))
            if any(not (0 <= int(rr) < int(h) and 0 <= int(cc) < LEFT_W) for rr, cc in cells):
                continue
            if any((int(rr), int(cc)) in occupied and (int(rr), int(cc)) != (int(r_cur), int(c_cur)) for rr, cc in cells):
                continue

            areas.append(int(area))
            added: list[tuple[int, int]] = []
            for cell in cells:
                key = (int(cell[0]), int(cell[1]))
                if key not in occupied:
                    occupied.add(key)
                    added.append(key)

            r_next, c_next = map(int, cells[-1])
            if rec(int(idx) + 1, int(r_next), int(c_next)):
                return True

            for cell in added:
                occupied.remove((int(cell[0]), int(cell[1])))
            areas.pop()

        return False

    if rec(0, GRAY_START_ROW, int(start_col)):
        return list(areas)
    return None


def _render_grid(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    h = 4 * int(n_rows) - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[int(i) // int(n_rows)]
        r0 = TILE_ROW_STRIDE * (int(i) % int(n_rows))
        grid[int(r0) : int(r0) + 3, int(tile_col) : int(tile_col) + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_rows = random.randint(N_MIN, N_MAX)
    h = 4 * int(n_rows) - 1

    start_cols = list(range(LEFT_W))
    random.shuffle(start_cols)
    for start_col in start_cols:
        areas = _sample_path(int(n_rows), int(start_col))
        if areas is None:
            continue
        inp = _render_grid(int(n_rows), int(start_col), areas)
        expected = _render_output_from_areas(
            int(n_rows),
            int(start_col),
            areas,
            alternate_step=True,
            diagonal_area4=True,
        )
        if expected.shape != (int(h), LEFT_W):
            continue
        return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
