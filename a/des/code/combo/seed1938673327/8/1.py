import random

import numpy as np


BG = 0
PATH = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _turn_idx(direction: int, bend: str) -> int:
    if bend == "L":
        return (direction - 1) % 4
    return (direction + 1) % 4


def _extract_cycle(g: np.ndarray) -> list[int]:
    h, w = g.shape
    assert h >= 2 and w >= 2
    colors: list[int] = []
    c = 0
    started = False
    while c + 1 < w:
        v = int(g[0, c])
        if (
            v != BG
            and int(g[1, c]) == v
            and int(g[0, c + 1]) == v
            and int(g[1, c + 1]) == v
        ):
            colors.append(v)
            started = True
            c += 2
            while c < w and int(g[0, c]) == BG and int(g[1, c]) == BG:
                c += 1
            continue
        if started:
            if int(g[0, c]) == BG and int(g[1, c]) == BG:
                c += 1
                continue
            break
        c += 1
    assert len(colors) >= 3
    return colors


def _path_color(g: np.ndarray, cycle: list[int]) -> int:
    cycle_set = set(cycle)
    candidates = [int(v) for v in np.unique(g) if int(v) not in cycle_set and int(v) != BG]
    assert candidates
    return max(candidates, key=lambda v: int(np.sum(g == v)))


def _marker_data(
    g: np.ndarray, cycle: list[int], path_color: int
) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = g.shape
    cycle_set = set(cycle)
    markers: list[tuple[int, int, int, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v not in cycle_set:
                continue
            same_neighbors = 0
            path_neighbors: list[tuple[int, int]] = []
            for dr, dc in DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if int(g[rr, cc]) == v:
                        same_neighbors += 1
                    if int(g[rr, cc]) == path_color:
                        path_neighbors.append((dr, dc))
            if same_neighbors == 0 and len(path_neighbors) == 1:
                markers.append((r, c, v, path_neighbors[0]))
    assert markers
    markers.sort()
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color particles walking along gray polylines.

    Input structure:
    - Black background.
    - A top-left row of 2x2 monochrome blocks defines the color cycle.
    - Isolated cycle-colored start markers sit next to gray path endpoints.

    Rule:
    - Each marker launches a particle into the adjacent gray path.
    - Every forward step paints the next color in the cycle.
    - The marker color index sets both the initial paint phase and the first turn:
      even index -> first turn right, odd index -> first turn left.
    - After every successful turn, the turn side alternates.
    - If the scheduled turn is blocked or leaves the grid, the particle stops.
    """
    g = np.asarray(grid, dtype=int)
    cycle = _extract_cycle(g)
    path_color = _path_color(g, cycle)
    markers = _marker_data(g, cycle, path_color)

    out = g.copy()
    h, w = g.shape

    for sr, sc, marker_color, (dr, dc) in markers:
        idx = cycle.index(marker_color)
        turn_side = "R" if idx % 2 == 0 else "L"
        phase = idx
        r, c = sr, sc
        seen: set[tuple[int, int, int, int, int, str]] = set()

        while True:
            state = (r, c, dr, dc, phase, turn_side)
            if state in seen:
                break
            seen.add(state)

            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == path_color:
                phase = (phase + 1) % len(cycle)
                out[nr, nc] = cycle[phase]
                r, c = nr, nc
                continue

            if turn_side == "L":
                tdr, tdc = -dc, dr
            else:
                tdr, tdc = dc, -dr
            rr = r + tdr
            cc = c + tdc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == path_color:
                dr, dc = tdr, tdc
                turn_side = "R" if turn_side == "L" else "L"
                continue
            break

    return out


def _opposite(bend: str) -> str:
    return "R" if bend == "L" else "L"


def _scheduled_bends(index: int, count: int) -> list[str]:
    first = "R" if index % 2 == 0 else "L"
    bends = []
    current = first
    for _ in range(count):
        bends.append(current)
        current = _opposite(current)
    return bends


def _relative_shape(
    start_dir: int, bends: list[str], lengths: list[int]
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    directions = [start_dir]
    for bend in bends:
        directions.append(_turn_idx(directions[-1], bend))

    cells = [(0, 0)]
    r = c = 0
    for direction, length in zip(directions, lengths):
        dr, dc = DIRS[direction]
        for _ in range(length - 1):
            r += dr
            c += dc
            cells.append((r, c))

    cell_set = set(cells)
    marker = (-DIRS[start_dir][0], -DIRS[start_dir][1])
    if len(cell_set) != len(cells) or marker in cell_set:
        return None

    deg1 = 0
    for rr, cc in cell_set:
        degree = sum((rr + dr, cc + dc) in cell_set for dr, dc in DIRS)
        if degree == 1:
            deg1 += 1
        elif degree != 2:
            return None
    if deg1 != 2:
        return None
    if sum((marker[0] + dr, marker[1] + dc) in cell_set for dr, dc in DIRS) != 1:
        return None
    return cells, marker


def _translate_shape(
    cells: list[tuple[int, int]],
    marker: tuple[int, int],
    zone: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    zr0, zr1, zc0, zc1 = zone
    all_rows = [r for r, _ in cells] + [marker[0]]
    all_cols = [c for _, c in cells] + [marker[1]]
    min_r = min(all_rows)
    max_r = max(all_rows)
    min_c = min(all_cols)
    max_c = max(all_cols)

    row_lo = zr0 - min_r
    row_hi = zr1 - max_r
    col_lo = zc0 - min_c
    col_hi = zc1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    offsets = [(rr, cc) for rr in range(row_lo, row_hi + 1) for cc in range(col_lo, col_hi + 1)]
    random.shuffle(offsets)
    for off_r, off_c in offsets:
        shifted_cells = [(r + off_r, c + off_c) for r, c in cells]
        shifted_marker = (marker[0] + off_r, marker[1] + off_c)
        return shifted_cells, shifted_marker
    return None


def _sample_shape(index: int, scenario: str, zone: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    zr0, zr1, zc0, zc1 = zone
    zone_h = zr1 - zr0 + 1
    zone_w = zc1 - zc0 + 1
    max_turns = 2 if min(zone_h, zone_w) < 8 else 3

    if scenario == "full":
        turn_count = random.randint(2, max_turns)
    elif scenario == "mismatch2":
        turn_count = random.randint(2, max_turns)
    else:
        turn_count = random.randint(1, max_turns)

    base = _scheduled_bends(index, turn_count)
    if scenario == "full":
        bends = base
    elif scenario == "mismatch1":
        bends = [_opposite(base[0])] + base[1:]
    else:
        bends = [base[0], _opposite(base[1])] + base[2:]

    len_hi = max(3, min(5, max(zone_h, zone_w) - 2))
    seg_hi = max(2, min(4, max(zone_h, zone_w) - 3))

    for _ in range(120):
        start_dir = random.randrange(4)
        lengths = [random.randint(3, len_hi)]
        lengths.extend(random.randint(2, seg_hi) for _ in range(turn_count))
        shape = _relative_shape(start_dir, bends, lengths)
        if shape is None:
            continue
        translated = _translate_shape(shape[0], shape[1], zone)
        if translated is not None:
            return translated
    return None


def _zones(h: int, w: int, template_w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (3, mid_r - 2, max(template_w + 2, mid_c), w - 3),
        (mid_r + 1, h - 3, 1, mid_c - 3),
        (mid_r + 1, h - 3, mid_c + 1, w - 3),
    ]


def _paint_template(grid: np.ndarray, cycle: list[int]) -> int:
    for i, color in enumerate(cycle):
        c0 = 3 * i
        grid[0:2, c0 : c0 + 2] = color
    return 3 * len(cycle) - 1


def _choose_marker_indices(n_markers: int, cycle_len: int) -> list[int]:
    even = [i for i in range(cycle_len) if i % 2 == 0]
    odd = [i for i in range(cycle_len) if i % 2 == 1]
    chosen = [random.choice(even), random.choice(odd)]
    remaining = [i for i in range(cycle_len) if i not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: n_markers - 2])
    random.shuffle(chosen)
    return chosen


def _scenario_list(n_markers: int) -> list[str]:
    if n_markers == 2:
        options = [["full", "mismatch1"], ["full", "mismatch2"]]
        scenarios = random.choice(options)[:]
    else:
        scenarios = ["full", "mismatch1", "mismatch2"]
    random.shuffle(scenarios)
    return scenarios


def _stamp_object(
    grid: np.ndarray,
    path_cells: list[tuple[int, int]],
    marker: tuple[int, int],
    marker_color: int,
) -> None:
    for r, c in path_cells:
        grid[r, c] = PATH
    grid[marker] = marker_color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(24, 32)
    cycle_len = random.choice([3, 4])
    cycle = random.sample(COLOR_POOL, cycle_len)
    grid = np.full((h, w), BG, dtype=int)
    template_w = _paint_template(grid, cycle)

    zones = _zones(h, w, template_w)
    if any(zr0 > zr1 or zc0 > zc1 for zr0, zr1, zc0, zc1 in zones):
        return None

    n_markers = random.choice([2, 3])
    active_zones = random.sample(zones, n_markers)
    marker_indices = _choose_marker_indices(n_markers, cycle_len)
    scenarios = _scenario_list(n_markers)

    used: set[tuple[int, int]] = set()
    for zone, index, scenario in zip(active_zones, marker_indices, scenarios):
        shape = _sample_shape(index, scenario, zone)
        if shape is None:
            return None
        path_cells, marker = shape
        footprint = set(path_cells) | {marker}
        if footprint & used:
            return None
        used |= footprint
        _stamp_object(grid, path_cells, marker, cycle[index])

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
