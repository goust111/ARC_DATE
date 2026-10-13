import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BRANCHES = ("facing", "same", "back")
AXES = ("horizontal", "vertical")
COLORS = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
PAIR_COUNT_RANGE = (3, 5)
GAP_RANGE = (3, 8)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300

HORIZONTAL_OPTIONS = {
    "facing": (("NE", "NW"), ("SE", "SW")),
    "same": (("NW", "NW"), ("NE", "NE"), ("SW", "SW"), ("SE", "SE")),
    "back": (("NW", "NE"), ("SW", "SE")),
}
VERTICAL_OPTIONS = {
    "facing": (("SW", "NW"), ("SE", "NE")),
    "same": (("NW", "NW"), ("NE", "NE"), ("SW", "SW"), ("SE", "SE")),
    "back": (("NW", "SW"), ("NE", "SE")),
}
CORNER_OFFSETS = {
    "NW": (0, 0),
    "NE": (0, 1),
    "SW": (1, 0),
    "SE": (1, 1),
}


@dataclass(frozen=True)
class Marker:
    color: int
    corner: str
    r0: int
    c0: int

    @property
    def r1(self) -> int:
        return self.r0 + 1

    @property
    def c1(self) -> int:
        return self.c0 + 1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1

    def missing_cell(self) -> tuple[int, int]:
        dr, dc = CORNER_OFFSETS[self.corner]
        return self.r0 + dr, self.c0 + dc

    def cells(self) -> set[tuple[int, int]]:
        miss = self.missing_cell()
        pts = set()
        for dr in range(2):
            for dc in range(2):
                pt = (self.r0 + dr, self.c0 + dc)
                if pt != miss:
                    pts.add(pt)
        return pts


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_marker(color: int, coords: list[tuple[int, int]]) -> Marker:
    assert len(coords) == 3, "each L marker must have area 3"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "each marker must occupy a 2x2 box"
    all_cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    missing = list(all_cells - set(coords))
    assert len(missing) == 1, "marker must miss exactly one corner"
    miss = missing[0]
    for corner, (dr, dc) in CORNER_OFFSETS.items():
        if miss == (r0 + dr, c0 + dc):
            return Marker(color=color, corner=corner, r0=r0, c0=c0)
    raise AssertionError("unreachable corner decode")


def _row_overlap(a: Marker, b: Marker) -> int:
    return min(a.r1, b.r1) - max(a.r0, b.r0) + 1


def _col_overlap(a: Marker, b: Marker) -> int:
    return min(a.c1, b.c1) - max(a.c0, b.c0) + 1


def _pair_axis(a: Marker, b: Marker) -> tuple[str, Marker, Marker]:
    if _row_overlap(a, b) > 0 and (a.c1 < b.c0 or b.c1 < a.c0):
        left, right = (a, b) if a.c0 < b.c0 else (b, a)
        return "horizontal", left, right
    if _col_overlap(a, b) > 0 and (a.r1 < b.r0 or b.r1 < a.r0):
        top, bottom = (a, b) if a.r0 < b.r0 else (b, a)
        return "vertical", top, bottom
    raise AssertionError("paired markers must share a row band or a column band")


def _h_inner_endpoint(marker: Marker, side: str) -> tuple[int, int]:
    row = marker.r0 + (1 if marker.corner[0] == "N" else 0)
    col = marker.c1 if side == "right" else marker.c0
    return row, col


def _v_inner_endpoint(marker: Marker, side: str) -> tuple[int, int]:
    col = marker.c0 + (1 if marker.corner[1] == "W" else 0)
    row = marker.r1 if side == "bottom" else marker.r0
    return row, col


def _h_outer_endpoint(marker: Marker, side: str) -> tuple[int, int]:
    row = marker.r0 if marker.corner[0] == "N" else marker.r1
    col = marker.c1 if side == "left" else marker.c0
    return row, col


def _v_outer_endpoint(marker: Marker, side: str) -> tuple[int, int]:
    col = marker.c0 if marker.corner[1] == "W" else marker.c1
    row = marker.r1 if side == "top" else marker.r0
    return row, col


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            assert c0 == c1, "bridge segments must be orthogonal"
            step = 1 if r1 >= r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        if cells and segment[0] == cells[-1]:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _bridge_cells(a: Marker, b: Marker, shape: tuple[int, int]) -> list[tuple[int, int]]:
    axis, first, second = _pair_axis(a, b)
    if axis == "horizontal":
        if first.corner == second.corner:
            start = _h_inner_endpoint(first, "right")
            end = _h_inner_endpoint(second, "left")
            mid_c = (start[1] + end[1]) // 2
            points = [start, (start[0], mid_c), (end[0], mid_c), end]
        elif (first.corner, second.corner) in HORIZONTAL_OPTIONS["facing"]:
            start = _h_inner_endpoint(first, "right")
            end = _h_inner_endpoint(second, "left")
            assert start[0] == end[0], "facing horizontal pair must align for a straight bridge"
            points = [start, end]
        elif (first.corner, second.corner) in HORIZONTAL_OPTIONS["back"]:
            start = _h_outer_endpoint(first, "left")
            end = _h_outer_endpoint(second, "right")
            detour_row = min(first.r0, second.r0) - 1 if first.corner[0] == "N" else max(first.r1, second.r1) + 1
            points = [start, (detour_row, start[1]), (detour_row, end[1]), end]
        else:
            raise AssertionError("unsupported horizontal corner relation")
    else:
        if first.corner == second.corner:
            start = _v_inner_endpoint(first, "bottom")
            end = _v_inner_endpoint(second, "top")
            mid_r = (start[0] + end[0]) // 2
            points = [start, (mid_r, start[1]), (mid_r, end[1]), end]
        elif (first.corner, second.corner) in VERTICAL_OPTIONS["facing"]:
            start = _v_inner_endpoint(first, "bottom")
            end = _v_inner_endpoint(second, "top")
            assert start[1] == end[1], "facing vertical pair must align for a straight bridge"
            points = [start, end]
        elif (first.corner, second.corner) in VERTICAL_OPTIONS["back"]:
            start = _v_outer_endpoint(first, "top")
            end = _v_outer_endpoint(second, "bottom")
            detour_col = min(first.c0, second.c0) - 1 if first.corner[1] == "W" else max(first.c1, second.c1) + 1
            points = [start, (start[0], detour_col), (end[0], detour_col), end]
        else:
            raise AssertionError("unsupported vertical corner relation")
    cells = _polyline(points)
    h, w = shape
    assert all(0 <= r < h and 0 <= c < w for r, c in cells), "bridge must stay inside the grid"
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored 2x2-minus-one-corner L markers and draw a bridge whose shape is
    determined by the relative missing-corner orientations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != 0):
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each non-background color must appear as exactly one marker pair"
        markers = [_parse_marker(color, comp) for comp in comps]
        bridge = _bridge_cells(markers[0], markers[1], g.shape)
        for r, c in bridge:
            out[r, c] = color
    return out


def _branch_multiset(count: int) -> list[str]:
    branches = list(BRANCHES)
    while len(branches) < count:
        branches.append(random.choice(BRANCHES))
    random.shuffle(branches)
    return branches


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _candidate_pair(color: int, branch: str, shape: tuple[int, int]) -> tuple[Marker, Marker, set[tuple[int, int]]] | None:
    h, w = shape
    axis = random.choice(AXES)
    options = HORIZONTAL_OPTIONS[branch] if axis == "horizontal" else VERTICAL_OPTIONS[branch]
    corner_a, corner_b = random.choice(options)
    gap = random.randint(*GAP_RANGE)

    if axis == "horizontal":
        row_delta = random.choice((-1, 1)) if branch == "same" else 0
        total_w = gap + 4
        if total_w > w:
            return None
        c_left = random.randint(0, w - total_w)
        c_right = c_left + gap + 2
        low = max(0, -row_delta)
        high = min(h - 2, h - 2 - row_delta)
        if low > high:
            return None
        r_left = random.randint(low, high)
        r_right = r_left + row_delta
        first = Marker(color=color, corner=corner_a, r0=r_left, c0=c_left)
        second = Marker(color=color, corner=corner_b, r0=r_right, c0=c_right)
    else:
        col_delta = random.choice((-1, 1)) if branch == "same" else 0
        total_h = gap + 4
        if total_h > h:
            return None
        r_top = random.randint(0, h - total_h)
        r_bottom = r_top + gap + 2
        low = max(0, -col_delta)
        high = min(w - 2, w - 2 - col_delta)
        if low > high:
            return None
        c_top = random.randint(low, high)
        c_bottom = c_top + col_delta
        first = Marker(color=color, corner=corner_a, r0=r_top, c0=c_top)
        second = Marker(color=color, corner=corner_b, r0=r_bottom, c0=c_bottom)

    try:
        bridge = _bridge_cells(first, second, shape)
    except AssertionError:
        return None
    marker_cells = first.cells() | second.cells()
    bridge_cells = set(bridge)
    allowed_overlap = {bridge[0], bridge[-1]}
    if bridge_cells & marker_cells - allowed_overlap:
        return None
    footprint = marker_cells | bridge_cells
    return first, second, footprint


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()

    pair_count = random.randint(*PAIR_COUNT_RANGE)
    branches = _branch_multiset(pair_count)
    colors = random.sample(COLORS, pair_count)

    for color, branch in zip(colors, branches):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            candidate = _candidate_pair(color, branch, (h, w))
            if candidate is None:
                continue
            first, second, footprint = candidate
            halo = _expand(footprint, (h, w), radius=1)
            if reserved & halo:
                continue
            for r, c in first.cells() | second.cells():
                grid[r, c] = color
            reserved |= halo
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    unique_inputs = {int(v) for v in np.unique(grid) if int(v) != 0}
    unique_outputs = {int(v) for v in np.unique(out) if int(v) != 0}
    if unique_inputs != unique_outputs:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
