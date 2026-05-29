# Combo variant of 136b0064 (index 28)
# combo_spec: D8/1+D10/0+D12/1
# Name: 紧凑环绕对角路径

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "solvers" / "arc_utils.py").exists()
)
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg

# ---------------------------------------------------------------------------
# Design analysis for combo D8/1 + D10/0 + D12/1
#
# a) Concretization and trace
#    - Input geometry stays canonical: width 15, divider at col 7, 7-column
#      glyph panel on the left, one gray start pixel on the right, and 2*N
#      fixed 3x3 glyph slots read column-major.
#    - The fused rule changes three stages at once:
#        1) D8 removes the old forced "step down one row before each glyph".
#           Every new segment now starts on the previous endpoint.
#        2) D12 replaces the old area-4 vertical with a 3-cell descending
#           diagonal. The visible cue is still tile-column position:
#           left tile column => down-right, right tile column => down-left.
#        3) D10 applies column wrapping to every horizontal drift, including the
#           diagonal's sideways part. When a step leaves col 0..6, it re-enters
#           on the opposite edge of the same row.
#    - Concrete trace from start (0, 6) with glyph areas [5, 4_left, 6]:
#        old task:   area 5 uses forced anchor (1,6) and then fails trying to
#                    paint cols 6,7,8; wrap/diagonal were impossible.
#        fused task: area 5 paints (0,6),(0,0),(0,1);
#                    area 4_left paints (0,1),(1,2),(2,3);
#                    area 6 paints (2,3),(2,2),(2,1),(2,0).
#      The path is compact because no blank-row stair step is inserted, richer
#      because the plus glyph bends diagonally, and visibly wrapped because the
#      first segment splits across both canvas edges.
#
# b) Complexity calibration
#    - Keep the original four glyph templates and strict shape->color binding,
#      so the task is still "decode a small visible codebook" rather than
#      introducing a second hidden mapping.
#    - The generator forces both diagonal directions to appear by requiring at
#      least one area-4 glyph in each tile column. This keeps the extra cue
#      human-visible and repeated.
#    - It also forces at least one wrap event and at least one non-wrapping
#      horizontal segment, so the new edge behavior is observable by contrast
#      instead of looking like a one-off edge touch.
#
# c) Implicit assumptions that break and must be rewritten
#    - Old `r_cur += 1` before every segment is invalid under D8; the cursor is
#      reused directly and the gray start may be overwritten immediately.
#    - Old `if area == 4` assumed "down 2 with fixed column". Under D12 the
#      column changes by +/-1 on each row advance, and the sign depends on the
#      glyph's visible tile column.
#    - Old horizontal bound checks such as `0 <= end_c < div_c` are invalid
#      under D10 because a segment may cross the edge mid-stroke and still end
#      in range after wrapping.
#    - The three changes interact: because there is no inter-glyph drop, the
#      diagonal now starts exactly on the previous endpoint, and its sideways
#      steps must also wrap modulo width.
#
# d) Placement legality
#    - Input glyph placement is unchanged, so 3x3 components never overlap and
#      can still be recovered by bbox and area.
#    - Output legality is enforced by simulating each candidate segment
#      cell-by-cell with the fused rule itself. Rows may never exceed 0..H-1,
#      and every newly painted cell after the shared endpoint must be unused.
#      This prevents hidden overlaps and keeps the oracle path unambiguous.
#
# e) Visual observability
#    - A human can now see three distinct cues in a few examples:
#        * compact connection: the path does not insert blank downward anchors;
#        * diagonal area-4 behavior: plus glyphs in the two input tile columns
#          produce mirrored down-right / down-left strokes;
#        * wrap: some same-color segments continue from one side of the canvas
#          to the other within the same row progression.
#    - No ordering or branching depends on color ids or other non-visual data;
#      only visible tile position, shape, and path geometry matter.
#
# f) Visual cleanliness
#    - New-cell reuse is forbidden, so even with wrapping and diagonals the
#      output remains a single clean polyline rather than a scribble.
#    - Requiring both wrapped and non-wrapped horizontals avoids samples where
#      every segment hugs the boundary and becomes hard to parse.
#
# g) Inferability of the changed behavior
#    - Colors remain fixed functions of the glyph shapes, so the only inference
#      burden is geometric. That is supported structurally: the same plus glyph
#      recurs in both tile columns, and wrap is shown by repeated same-color
#      continuity across opposite edges.
#
# Post-change solver assumptions checked against the generator:
#    - exactly one full-height non-background divider column exists;
#    - exactly one gray start lies on the right and shifts into columns 0..6;
#    - glyphs are monochromatic 3x3 components in the two canonical tile
#      columns, so glyph order is visible from layout alone;
#    - area-4 diagonal direction is determined only by glyph tile column;
#    - every generated segment is row-valid and only shares its first cell with
#      the previously drawn segment.
# ---------------------------------------------------------------------------


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


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
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # plus
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),  # diamond
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),  # cross
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),  # U
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_HLEN = {5: 3, 6: 4, 7: 2}
_AREA_TO_HDIR = {5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES = 2
MIN_WRAP_EVENTS = 1
MIN_HORIZONTAL_SEGMENTS = 2
MAX_ATTEMPTS = 900
START_COL_WEIGHTS = (4, 2, 1, 1, 1, 2, 4)


def _diag_dc_from_glyph_col(glyph_c0: int, div_c: int) -> int:
    """Left tile column => down-right diagonal, right tile column => down-left diagonal."""
    return 1 if int(glyph_c0) < int(div_c // 2) else -1


def _diag_dc_from_index(idx: int, n_rows: int) -> int:
    """Indices 0..N-1 belong to the left tile column; N..2N-1 belong to the right."""
    return 1 if int(idx) < int(n_rows) else -1


def _trace_segment_cells(
    r_cur: int,
    c_cur: int,
    area: int,
    *,
    width: int,
    diag_dc: int | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """
    Return (cells, wrap_count) for one fused-rule segment.

    The returned cells always include the current endpoint as the first cell,
    because D8 removes the old pre-segment downward step.
    """
    rr = int(r_cur)
    cc = int(c_cur)
    cells = [(int(rr), int(cc))]
    wraps = 0

    if int(area) == 4:
        assert diag_dc in (-1, 1)
        for _ in range(2):
            rr = int(rr + 1)
            nxt = int(cc + diag_dc)
            if not (0 <= nxt < int(width)):
                wraps += 1
            cc = int(nxt % int(width))
            cells.append((int(rr), int(cc)))
        return cells, int(wraps)

    assert int(area) in _AREA_TO_HLEN
    seg_len = int(_AREA_TO_HLEN[int(area)])
    dir_h = int(_AREA_TO_HDIR[int(area)])
    for _ in range(seg_len - 1):
        nxt = int(cc + dir_h)
        if not (0 <= nxt < int(width)):
            wraps += 1
        cc = int(nxt % int(width))
        cells.append((int(rr), int(cc)))
    return cells, int(wraps)


def _direction_changes(areas: list[int]) -> int:
    kinds = [int(area) == 4 for area in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _simulate_path(
    start_col: int,
    areas: list[int],
    n_rows: int,
    *,
    width: int = LEFT_W,
    check_overlap: bool = False,
) -> dict[str, object]:
    """Simulate the fused polyline and collect paint ops plus observability metrics."""
    H = 4 * int(n_rows) - 1
    r = int(GRAY_START_ROW)
    c = int(start_col)
    occupied = {(int(r), int(c))}
    paints: list[tuple[int, int, int]] = [(int(r), int(c), GRAY)]

    wrap_events = 0
    wrap_segments = 0
    diag_segments = 0
    diag_left = 0
    diag_right = 0
    horizontal_segments = 0
    nonwrap_horizontal = 0
    diag_wrap_events = 0

    for idx, area in enumerate(areas):
        area = int(area)
        diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if area == 4 else None
        cells, seg_wraps = _trace_segment_cells(int(r), int(c), area, width=int(width), diag_dc=diag_dc)

        if any(not (0 <= int(rr) < H) for rr, _cc in cells):
            raise AssertionError(f"segment exits bottom: cells={cells}, H={H}")

        new_cells = cells[1:]
        if check_overlap and any((int(rr), int(cc)) in occupied for rr, cc in new_cells):
            raise AssertionError(f"segment overlaps existing path: idx={idx}, cells={cells}")

        color = int(_AREA_TO_COLOR[area])
        for rr, cc in cells:
            paints.append((int(rr), int(cc), color))

        for rr, cc in new_cells:
            occupied.add((int(rr), int(cc)))

        wrap_events += int(seg_wraps)
        if seg_wraps:
            wrap_segments += 1

        if area == 4:
            diag_segments += 1
            diag_wrap_events += int(seg_wraps)
            if int(idx) < int(n_rows):
                diag_left += 1
            else:
                diag_right += 1
        else:
            horizontal_segments += 1
            if seg_wraps == 0:
                nonwrap_horizontal += 1

        r, c = map(int, cells[-1])

    return {
        "paints": paints,
        "end": (int(r), int(c)),
        "wrap_events": int(wrap_events),
        "wrap_segments": int(wrap_segments),
        "diag_segments": int(diag_segments),
        "diag_left": int(diag_left),
        "diag_right": int(diag_right),
        "horizontal_segments": int(horizontal_segments),
        "nonwrap_horizontal": int(nonwrap_horizontal),
        "diag_wrap_events": int(diag_wrap_events),
        "direction_changes": int(_direction_changes([int(a) for a in areas])),
    }


def _render_output(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * int(n_rows) - 1
    out = np.zeros((H, LEFT_W), dtype=int)
    sim = _simulate_path(int(start_col), [int(a) for a in areas], int(n_rows), width=LEFT_W, check_overlap=False)
    for rr, cc, color in sim["paints"]:
        out[int(rr), int(cc)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the tiled glyphs into a compact wrapped polyline on the left canvas.

    Input structure:
    - Left of the full-height divider: monochrome 3x3 glyphs placed in two fixed tile columns on background.
    - Right of the divider: one gray (5) start pixel whose shifted column seeds the drawing.

    Rule:
    - Remove the divider, shift the gray start into the 7-column left canvas, and read glyphs in visible column-major order.
    - Draw each decoded segment directly from the current endpoint with no extra step between glyphs.
    - Area 5 draws right 3, area 6 draws left 4, area 7 draws left 2, and area 4 draws a 3-cell descending diagonal.
    - The area-4 diagonal slopes down-right for glyphs from the left tile column and down-left for glyphs from the right tile column.
    - Any sideways motion that leaves the canvas wraps to the opposite edge on the same row progression.

    Key insight: The glyph shape still names the segment, but the output cursor now persists continuously, the plus glyph uses its tile-column position to choose a diagonal direction, and every horizontal drift is traced modulo the canvas width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    ignore = {int(bg), GRAY}
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

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        diag_dc = _diag_dc_from_glyph_col(int(glyph["c0"]), int(div_c)) if area == 4 else None
        cells, _wraps = _trace_segment_cells(
            int(r_cur),
            int(c_cur),
            int(area),
            width=int(div_c),
            diag_dc=diag_dc,
        )

        for rr, cc in cells:
            assert 0 <= int(rr) < H, f"segment exits bottom: rr={rr}, H={H}"
            assert 0 <= int(cc) < int(div_c), f"wrapped column out of range: cc={cc}, div_c={div_c}"
            out[int(rr), int(cc)] = color

        r_cur, c_cur = map(int, cells[-1])

    return out


def _choose_mandatory_diag_indices(n_rows: int) -> set[int]:
    """Require one plus glyph in each tile column so both diagonal directions are visible."""
    return {
        random.choice(range(0, int(n_rows))),
        random.choice(range(int(n_rows), 2 * int(n_rows))),
    }


def _sample_path(n_rows: int, start_col: int) -> list[int] | None:
    """
    Backtrack a compact wrapped walk of length 2*N under the fused rule itself.

    The generator mirrors the solver exactly: no inter-glyph drop, area-4
    diagonals depend on tile column, and every sideways step wraps modulo 7.
    """
    H = 4 * int(n_rows) - 1
    total = 2 * int(n_rows)
    mandatory_diag = _choose_mandatory_diag_indices(int(n_rows))
    occupied = {(int(GRAY_START_ROW), int(start_col))}
    areas: list[int] = []

    def rec(
        idx: int,
        r: int,
        c: int,
        *,
        wrap_events: int,
        wrap_segments: int,
        diag_count: int,
        horiz_count: int,
        nonwrap_horizontal: int,
    ) -> bool:
        remaining = int(total - idx)
        remaining_after = int(total - idx - 1)
        mandatory_remaining = sum(1 for j in range(idx, total) if j in mandatory_diag)
        optional_slots_after = remaining_after - sum(1 for j in range(idx + 1, total) if j in mandatory_diag)

        if idx == total:
            if diag_count < 2:
                return False
            if horiz_count < MIN_HORIZONTAL_SEGMENTS:
                return False
            if wrap_events < MIN_WRAP_EVENTS:
                return False
            if wrap_segments == 0:
                return False
            if nonwrap_horizontal == 0:
                return False
            if _direction_changes(areas) < MIN_DIRECTION_CHANGES:
                return False
            try:
                sim = _simulate_path(int(start_col), list(areas), int(n_rows), width=LEFT_W, check_overlap=True)
            except AssertionError:
                return False
            if int(sim["diag_left"]) == 0 or int(sim["diag_right"]) == 0:
                return False
            return True

        if diag_count + mandatory_remaining < 2:
            return False
        if horiz_count + (remaining - mandatory_remaining) < MIN_HORIZONTAL_SEGMENTS:
            return False
        if nonwrap_horizontal + (remaining - mandatory_remaining) <= 0:
            return False

        allowed = [4] if idx in mandatory_diag else list(_AREAS)
        candidates: list[tuple[float, int, list[tuple[int, int]], int]] = []

        for area in allowed:
            area = int(area)
            diag_dc = _diag_dc_from_index(int(idx), int(n_rows)) if area == 4 else None
            cells, seg_wraps = _trace_segment_cells(int(r), int(c), area, width=LEFT_W, diag_dc=diag_dc)
            if any(not (0 <= int(rr) < H) for rr, _cc in cells):
                continue
            new_cells = cells[1:]
            if any((int(rr), int(cc)) in occupied for rr, cc in new_cells):
                continue

            next_diag = int(diag_count + (1 if area == 4 else 0))
            next_horiz = int(horiz_count + (1 if area != 4 else 0))
            next_nonwrap = int(nonwrap_horizontal + (1 if area != 4 and seg_wraps == 0 else 0))
            remaining_after_diag_needed = max(0, 2 - next_diag)
            if remaining_after < remaining_after_diag_needed:
                continue
            if next_horiz + optional_slots_after < MIN_HORIZONTAL_SEGMENTS:
                continue

            score = random.random()
            if area == 4:
                score += 2.0
                if seg_wraps:
                    score += 2.5
            else:
                score += 1.0
                if seg_wraps and wrap_events == 0:
                    score += 4.0
                elif seg_wraps:
                    score += 1.5
                if seg_wraps == 0 and nonwrap_horizontal == 0:
                    score += 2.5
                if c in (0, LEFT_W - 1):
                    score += 0.5
            candidates.append((float(score), area, cells, int(seg_wraps)))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _score, area, cells, seg_wraps in candidates:
            new_cells = cells[1:]
            areas.append(int(area))
            for rr, cc in new_cells:
                occupied.add((int(rr), int(cc)))

            next_r, next_c = map(int, cells[-1])
            if rec(
                int(idx + 1),
                int(next_r),
                int(next_c),
                wrap_events=int(wrap_events + seg_wraps),
                wrap_segments=int(wrap_segments + (1 if seg_wraps else 0)),
                diag_count=int(diag_count + (1 if area == 4 else 0)),
                horiz_count=int(horiz_count + (1 if area != 4 else 0)),
                nonwrap_horizontal=int(nonwrap_horizontal + (1 if area != 4 and seg_wraps == 0 else 0)),
            ):
                return True

            for rr, cc in new_cells:
                occupied.remove((int(rr), int(cc)))
            areas.pop()
        return False

    if rec(
        0,
        int(GRAY_START_ROW),
        int(start_col),
        wrap_events=0,
        wrap_segments=0,
        diag_count=0,
        horiz_count=0,
        nonwrap_horizontal=0,
    ):
        return list(areas)
    return None


def _render_grid(n_rows: int, start_col: int, areas: list[int]) -> np.ndarray:
    H = 4 * int(n_rows) - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for idx, area in enumerate(areas):
        tile_col = TILE_COLS[int(idx) // int(n_rows)]
        r0 = TILE_ROW_STRIDE * (int(idx) % int(n_rows))
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    n_rows = random.randint(N_MIN, N_MAX)
    start_cols = random.choices(range(LEFT_W), weights=START_COL_WEIGHTS, k=LEFT_W * 2)

    seen: set[int] = set()
    ordered_cols: list[int] = []
    for col in start_cols:
        col = int(col)
        if col not in seen:
            ordered_cols.append(col)
            seen.add(col)
    for col in range(LEFT_W):
        if int(col) not in seen:
            ordered_cols.append(int(col))

    for start_col in ordered_cols:
        areas = _sample_path(int(n_rows), int(start_col))
        if areas is None:
            continue
        sim = _simulate_path(int(start_col), list(areas), int(n_rows), width=LEFT_W, check_overlap=True)
        if int(sim["wrap_events"]) < MIN_WRAP_EVENTS:
            continue
        if int(sim["diag_left"]) == 0 or int(sim["diag_right"]) == 0:
            continue
        if int(sim["nonwrap_horizontal"]) == 0:
            continue
        inp = _render_grid(int(n_rows), int(start_col), list(areas))
        expected = _render_output(int(n_rows), int(start_col), list(areas))
        return inp, expected, sim
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected, _sim = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
