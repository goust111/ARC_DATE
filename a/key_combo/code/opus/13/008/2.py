from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
MARKER = 5
GEN_RETRIES = 500


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize_cells(cells: set[Coord] | list[Coord]) -> list[Coord]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return sorted((r - min_r, c - min_c) for r, c in pts)


def _shape_dims(shape: list[Coord]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape: list[Coord], top: int, left: int) -> list[Coord]:
    return [(top + r, left + c) for r, c in shape]


def _component_shapes(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bottom_row(shape: list[Coord]) -> int:
    return max(r for r, _ in shape)


def _bottom_columns(shape: list[Coord]) -> set[int]:
    bottom = _bottom_row(shape)
    return {c for r, c in shape if r == bottom}


def _top_row_columns(shape: list[Coord]) -> set[int]:
    top = min(r for r, _ in shape)
    return {c for r, c in shape if r == top}


def _paint(grid: np.ndarray, coords: list[Coord], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _find_drop(comp: list[Coord], occupied: np.ndarray) -> int:
    h, w = occupied.shape
    drop = 0
    while True:
        moved = [(r + drop + 1, c) for r, c in comp]
        if any(r >= h or c < 0 or c >= w or occupied[r, c] for r, c in moved):
            return drop
        drop += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove gray selectors, then drop only the selected red objects downward as rigid bodies until they hit obstacles or the bottom.

    Input structure:
    - The background is black.
    - Several red 4-connected objects are scattered across the grid.
    - Some gray single-pixel markers each touch exactly one red object by 4-neighborhood.

    Rule:
    - Find every red connected component and every gray marker.
    - Any red object touched by at least one gray marker is selected; gray markers are then erased.
    - Unselected red objects stay fixed and act as obstacles.
    - Process selected objects from lower to higher bottom edge, dropping each whole shape straight downward until one more step would hit the grid bottom or an already-fixed red pixel.
    - Return the full-size grid with the unselected obstacles and the dropped selected objects.

    Key insight:
    - The gray pixels do not survive; they only decide which red objects are released into the gravity simulation, while the untouched red objects become the terrain that stops those falling bodies.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == MARKER] = BG

    red_components = _component_shapes(g == RED)
    marker_cells = [tuple(map(int, pos)) for pos in np.argwhere(g == MARKER)]

    selected: list[list[Coord]] = []
    for comp in red_components:
        comp_set = set(comp)
        if any(any((nr, nc) in comp_set for nr, nc in _neighbors4(mr, mc)) for mr, mc in marker_cells):
            selected.append(comp)

    for comp in selected:
        for r, c in comp:
            out[r, c] = BG

    occupied = out == RED
    selected.sort(key=lambda comp: max(r for r, _ in comp), reverse=True)
    for comp in selected:
        drop = _find_drop(comp, occupied)
        landed = [(r + drop, c) for r, c in comp]
        _paint(out, landed, RED)
        for r, c in landed:
            occupied[r, c] = True
    return out


def _random_connected_shape(max_h: int = 4, max_w: int = 4) -> list[Coord]:
    for _ in range(200):
        area = random.randint(3, 6)
        cells: set[Coord] = {(0, 0)}
        while len(cells) < area:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        shape = _normalize_cells(cells)
        h, w = _shape_dims(shape)
        if h > max_h or w > max_w:
            continue
        if h < 2 or w < 2:
            continue
        return shape
    raise RuntimeError("failed to sample shape")


def _sample_selected_shape(lane_width: int) -> list[Coord]:
    max_w = min(4, lane_width)
    for _ in range(200):
        shape = _random_connected_shape(max_h=4, max_w=max_w)
        _h, w = _shape_dims(shape)
        if w <= lane_width:
            return shape
    raise RuntimeError("failed to sample selected shape")


def _can_place(coords: list[Coord], occupied: set[Coord], h: int, w: int, gap: int = 1) -> bool:
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied:
                    return False
    return True


def _first_free_extra_position(
    shape: list[Coord],
    occupied: set[Coord],
    h: int,
    w: int,
    col_lo: int,
    col_hi: int,
) -> tuple[int, int] | None:
    shp_h, shp_w = _shape_dims(shape)
    if col_hi - col_lo + 1 < shp_w:
        return None
    row_start = max(h // 2, 1)
    row_stop = h - shp_h
    for top in range(row_start, row_stop + 1):
        for left in range(col_lo, col_hi - shp_w + 2):
            coords = _translate(shape, top, left)
            if _can_place(coords, occupied, h, w, gap=1):
                return top, left
    return None


def _marker_candidates(shape_cells: list[Coord], all_red: set[Coord], h: int, w: int) -> list[Coord]:
    comp = set(shape_cells)
    candidates: set[Coord] = set()
    for r, c in shape_cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in all_red:
                continue
            touched = {cell for cell in _neighbors4(nr, nc) if cell in all_red}
            if touched and touched.issubset(comp):
                candidates.add((nr, nc))
    return sorted(candidates)


def _make_empty(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=int)


def _build_expected(
    h: int,
    w: int,
    selected_inputs: list[list[Coord]],
    selected_finals: list[list[Coord]],
    unselected_objs: list[list[Coord]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = _make_empty(h, w)
    out = _make_empty(h, w)
    for coords in selected_inputs:
        _paint(inp, coords, RED)
    for coords in unselected_objs:
        _paint(inp, coords, RED)
        _paint(out, coords, RED)
    for coords in selected_finals:
        _paint(out, coords, RED)
    return inp, out


def _analyze_markers(grid: np.ndarray) -> tuple[list[list[Coord]], list[Coord], list[int]]:
    comps = _component_shapes(grid == RED)
    markers = [tuple(map(int, pos)) for pos in np.argwhere(grid == MARKER)]
    selected_idx: list[int] = []
    for idx, comp in enumerate(comps):
        comp_set = set(comp)
        if any(any((nr, nc) in comp_set for nr, nc in _neighbors4(mr, mc)) for mr, mc in markers):
            selected_idx.append(idx)
    return comps, markers, selected_idx


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    support_types: list[str],
    selected_inputs: list[list[Coord]],
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    comps, markers, selected_idx = _analyze_markers(inp)
    if not (4 <= len(comps) <= 6):
        return False
    if not (2 <= len(markers) <= 3):
        return False
    if len(selected_idx) != len(support_types):
        return False

    for mr, mc in markers:
        touched = 0
        for comp in comps:
            comp_set = set(comp)
            if any((nr, nc) in comp_set for nr, nc in _neighbors4(mr, mc)):
                touched += 1
        if touched != 1:
            return False

    intervals = []
    for coords in selected_inputs:
        cols = [c for _, c in coords]
        intervals.append((min(cols), max(cols)))
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            a0, a1 = intervals[i]
            b0, b1 = intervals[j]
            if not (a1 < b0 or b1 < a0):
                return False

    if "obstacle" not in support_types or "bottom" not in support_types:
        return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray, list[str], list[list[Coord]]] | None:
    num_selected = random.choice([2, 3])
    if num_selected == 2:
        support_types = ["obstacle", "bottom"]
    else:
        support_types = ["obstacle", "bottom", random.choice(["obstacle", "bottom"])]
    random.shuffle(support_types)

    obstacle_count = sum(t == "obstacle" for t in support_types)
    min_unselected = max(obstacle_count, 4 - num_selected, 1)
    max_unselected = 6 - num_selected
    num_unselected = random.randint(min_unselected, max_unselected)
    extras = num_unselected - obstacle_count

    reserve_width = random.randint(5, 6) if extras > 0 else 0
    lane_widths = [random.randint(4, 5) for _ in range(num_selected)]
    gaps = [random.randint(1, 2) for _ in range(num_selected - 1)]
    left_width = 1 + sum(lane_widths) + sum(gaps) + 1
    reserve_gap = random.randint(1, 2) if reserve_width > 0 else 0
    min_width = left_width + reserve_gap + reserve_width + (1 if reserve_width > 0 else 0)
    lower_width = max(18, min_width)
    upper_width = min(30, lower_width + 4)
    if lower_width > upper_width:
        return None
    width = random.randint(lower_width, upper_width)
    height = random.randint(16, 21)

    reserve_lo = width - reserve_width - 1 if reserve_width > 0 else width
    lane_starts: list[int] = []
    cur = 1
    for i, lane_w in enumerate(lane_widths):
        lane_starts.append(cur)
        cur += lane_w
        if i < len(gaps):
            cur += gaps[i]
    if reserve_width > 0 and cur + reserve_gap > reserve_lo:
        return None

    occupied_red: set[Coord] = set()
    selected_inputs: list[list[Coord]] = []
    selected_finals: list[list[Coord]] = []
    unselected_objs: list[list[Coord]] = []

    for lane_start, lane_width, support_type in zip(lane_starts, lane_widths, support_types):
        selected_shape = _sample_selected_shape(lane_width)
        sel_h, sel_w = _shape_dims(selected_shape)
        sel_left = random.randint(lane_start, lane_start + lane_width - sel_w)
        bmax = _bottom_row(selected_shape)
        bottom_cols = _bottom_columns(selected_shape)

        if support_type == "bottom":
            final_top = height - 1 - bmax
            max_lift = min(6, final_top)
            if max_lift < 2:
                return None
            lift = random.randint(2, max_lift)
            input_top = final_top - lift
            input_coords = _translate(selected_shape, input_top, sel_left)
            final_coords = _translate(selected_shape, final_top, sel_left)
            if not _can_place(input_coords, occupied_red, height, width, gap=1):
                return None
            occupied_red.update(input_coords)
            selected_inputs.append(input_coords)
            selected_finals.append(final_coords)
            continue

        support_shape = _sample_selected_shape(lane_width)
        supp_h, supp_w = _shape_dims(support_shape)
        top_cols = _top_row_columns(support_shape)
        candidate_cols = []
        for supp_left in range(lane_start, lane_start + lane_width - supp_w + 1):
            if {(supp_left + c) for c in top_cols} & {(sel_left + c) for c in bottom_cols}:
                candidate_cols.append(supp_left)
        if not candidate_cols:
            return None
        supp_left = random.choice(candidate_cols)

        min_support_top = max(height // 2, bmax + 4)
        max_support_top = height - supp_h - 1
        if min_support_top > max_support_top:
            return None
        supp_top = random.randint(min_support_top, max_support_top)
        final_top = supp_top - 1 - bmax
        max_lift = min(6, final_top)
        if max_lift < 2:
            return None
        lift = random.randint(2, max_lift)
        input_top = final_top - lift

        input_coords = _translate(selected_shape, input_top, sel_left)
        final_coords = _translate(selected_shape, final_top, sel_left)
        support_coords = _translate(support_shape, supp_top, supp_left)

        if not _can_place(input_coords, occupied_red, height, width, gap=1):
            return None
        if not _can_place(support_coords, occupied_red, height, width, gap=1):
            return None

        occupied_red.update(input_coords)
        occupied_red.update(support_coords)
        selected_inputs.append(input_coords)
        selected_finals.append(final_coords)
        unselected_objs.append(support_coords)

    if extras > 0:
        reserve_hi = width - 2
        if reserve_lo > reserve_hi:
            return None
        for _ in range(extras):
            placed = False
            for _attempt in range(120):
                shape = _sample_selected_shape(reserve_hi - reserve_lo + 1)
                pos = _first_free_extra_position(shape, occupied_red, height, width, reserve_lo, reserve_hi)
                if pos is None:
                    continue
                top, left = pos
                coords = _translate(shape, top, left)
                occupied_red.update(coords)
                unselected_objs.append(coords)
                placed = True
                break
            if not placed:
                return None

    inp, expected = _build_expected(height, width, selected_inputs, selected_finals, unselected_objs)

    all_red = {tuple(map(int, pos)) for pos in np.argwhere(inp == RED)}
    marker_positions: list[Coord] = []
    for coords in selected_inputs:
        candidates = [pos for pos in _marker_candidates(coords, all_red, height, width) if pos not in marker_positions]
        if not candidates:
            return None
        marker = random.choice(candidates)
        inp[marker] = MARKER
        marker_positions.append(marker)

    if not _semantic_check(inp, expected, support_types, selected_inputs):
        return None
    return inp, expected, support_types, selected_inputs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_attempt()
        if result is not None:
            inp, _expected, _support_types, _selected_inputs = result
            return inp
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
