from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
INPUT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _same_color_component(
    grid: np.ndarray,
    start: tuple[int, int],
    cache: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    if start in cache:
        return cache[start]

    h, w = grid.shape
    color = int(grid[start])
    queue = deque([start])
    seen = {start}
    coords: list[tuple[int, int]] = []

    while queue:
        r, c = queue.popleft()
        coords.append((r, c))
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) in seen:
                continue
            if int(grid[rr, cc]) != color:
                continue
            seen.add((rr, cc))
            queue.append((rr, cc))

    component = tuple(coords)
    for pos in component:
        cache[pos] = component
    return component


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_same_color_marker(
    component: tuple[tuple[int, int], ...],
    pos: tuple[int, int],
) -> bool:
    if len(component) < 3:
        return False

    rest = [cell for cell in component if cell != pos]
    r0, c0, r1, c1 = _bbox(rest)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    if len(rest) != h * w:
        return False
    if min(h, w) not in (1, 2, 3):
        return False
    if h == w:
        return False

    or0, oc0, or1, oc1 = _bbox(component)
    r, c = pos

    if h > w:
        return (
            or0 == r0
            and or1 == r1
            and (oc1 - oc0 + 1) == w + 1
            and r0 <= r <= r1
            and c in (c0 - 1, c1 + 1)
        )

    return (
        oc0 == c0
        and oc1 == c1
        and (or1 - or0 + 1) == h + 1
        and c0 <= c <= c1
        and r in (r0 - 1, r1 + 1)
    )


def _find_markers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    comp_cache: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    markers: list[dict[str, object]] = []

    for r0, c0 in np.argwhere(grid != BLACK):
        r = int(r0)
        c = int(c0)
        neighbors = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if int(grid[rr, cc]) != BLACK]
        if len(neighbors) != 1:
            continue

        wall_r, wall_c = neighbors[0]
        marker_color = int(grid[r, c])
        wall_color = int(grid[wall_r, wall_c])

        if marker_color == wall_color:
            component = _same_color_component(grid, (r, c), comp_cache)
            if not _is_same_color_marker(component, (r, c)):
                continue

        markers.append(
            {
                "pos": (r, c),
                "dir": (wall_r - r, wall_c - c),
                "wall": (wall_r, wall_c),
                "marker_color": marker_color,
                "wall_color": wall_color,
            }
        )

    return markers


def _wall_thickness(grid: np.ndarray, wall: tuple[int, int], dr: int, dc: int) -> int:
    h, w = grid.shape
    r, c = wall
    color = int(grid[r, c])
    thickness = 0

    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color:
        thickness += 1
        r += dr
        c += dc

    return thickness


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove every leaf-like marker pixel. If the adjacent wall has the same color as
    the marker, cross that wall and draw a yellow trace on the far side whose length
    equals the wall thickness in the crossing direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    markers = _find_markers(g)

    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = BLACK

    for marker in markers:
        if marker["marker_color"] != marker["wall_color"]:
            continue

        wall_r, wall_c = marker["wall"]
        dr, dc = marker["dir"]
        thickness = _wall_thickness(g, (wall_r, wall_c), dr, dc)

        trace_r = wall_r + dr * thickness
        trace_c = wall_c + dc * thickness
        for step in range(thickness):
            rr = trace_r + dr * step
            cc = trace_c + dc * step
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = YELLOW

    return out


# --------------- generator ---------------

MIN_SIZE = 18
MAX_SIZE = 27
OUTER_MARGIN = 4
MIN_BARS = 4
MAX_BARS = 6
MIN_LENGTH = 5
BAR_PAD = 1
BAR_RETRIES = 120
GEN_RETRIES = 300


def _rand_int(low: int, high: int) -> int:
    return int(_RNG.integers(low, high + 1))


def _paint_bar(grid: np.ndarray, bar: dict[str, int]) -> None:
    r0 = bar["r0"]
    c0 = bar["c0"]
    h = bar["height"]
    w = bar["width"]
    grid[r0 : r0 + h, c0 : c0 + w] = bar["color"]


def _place_bars(h: int, w: int) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    grid = np.zeros((h, w), dtype=int)
    reserve = np.zeros((h, w), dtype=bool)

    bar_count = _rand_int(MIN_BARS, MAX_BARS)
    thicknesses = [1, 2, 3] + [_rand_int(1, 3) for _ in range(bar_count - 3)]
    orientations = ["h", "v"] + [str(_RNG.choice(("h", "v"))) for _ in range(bar_count - 2)]
    palette = [int(x) for x in _RNG.choice(INPUT_COLORS, size=_rand_int(3, 4), replace=False)]
    colors = palette + [int(_RNG.choice(palette)) for _ in range(bar_count - len(palette))]

    _RNG.shuffle(thicknesses)
    _RNG.shuffle(orientations)
    _RNG.shuffle(colors)

    bars: list[dict[str, int]] = []
    for thickness, orientation, color in zip(thicknesses, orientations, colors):
        if orientation == "h":
            bar_h = thickness
            max_length = w - 2 * OUTER_MARGIN
            if max_length < MIN_LENGTH:
                return None
            bar_w = _rand_int(MIN_LENGTH, min(max_length, 10 + _rand_int(0, 3)))
        else:
            bar_w = thickness
            max_length = h - 2 * OUTER_MARGIN
            if max_length < MIN_LENGTH:
                return None
            bar_h = _rand_int(MIN_LENGTH, min(max_length, 10 + _rand_int(0, 3)))

        placed = False
        for _ in range(BAR_RETRIES):
            r0 = _rand_int(OUTER_MARGIN, h - OUTER_MARGIN - bar_h)
            c0 = _rand_int(OUTER_MARGIN, w - OUTER_MARGIN - bar_w)
            rr0 = max(0, r0 - BAR_PAD)
            rr1 = min(h, r0 + bar_h + BAR_PAD)
            cc0 = max(0, c0 - BAR_PAD)
            cc1 = min(w, c0 + bar_w + BAR_PAD)
            if reserve[rr0:rr1, cc0:cc1].any():
                continue

            reserve[rr0:rr1, cc0:cc1] = True
            bar = {
                "r0": r0,
                "c0": c0,
                "height": bar_h,
                "width": bar_w,
                "thickness": thickness,
                "orientation": 0 if orientation == "h" else 1,
                "color": color,
            }
            _paint_bar(grid, bar)
            bars.append(bar)
            placed = True
            break

        if not placed:
            return None

    return grid, bars


def _site_ok(
    grid: np.ndarray,
    marker_pos: tuple[int, int],
    wall_pos: tuple[int, int],
    trace: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    mr, mc = marker_pos
    if not (0 <= mr < h and 0 <= mc < w):
        return False
    if int(grid[mr, mc]) != BLACK:
        return False

    for rr, cc in trace:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[rr, cc]) != BLACK:
            return False

    nonblack_neighbors = [(rr, cc) for rr, cc in _neighbors4(mr, mc, h, w) if int(grid[rr, cc]) != BLACK]
    return len(nonblack_neighbors) == 1 and nonblack_neighbors[0] == wall_pos


def _candidate_sites(grid: np.ndarray, bar: dict[str, int]) -> list[dict[str, object]]:
    r0 = bar["r0"]
    c0 = bar["c0"]
    height = bar["height"]
    width = bar["width"]
    thickness = bar["thickness"]
    sites: list[dict[str, object]] = []

    if bar["orientation"] == 1:
        rows = list(range(r0 + 1, r0 + height - 1))
        _RNG.shuffle(rows)
        for row in rows:
            marker = (row, c0 - 1)
            wall = (row, c0)
            trace = [(row, c0 + width + offset) for offset in range(thickness)]
            if _site_ok(grid, marker, wall, trace):
                sites.append(
                    {
                        "bar": bar,
                        "marker_pos": marker,
                        "trace": tuple(trace),
                        "footprint": frozenset([marker, *trace]),
                    }
                )

            marker = (row, c0 + width)
            wall = (row, c0 + width - 1)
            trace = [(row, c0 - 1 - offset) for offset in range(thickness)]
            if _site_ok(grid, marker, wall, trace):
                sites.append(
                    {
                        "bar": bar,
                        "marker_pos": marker,
                        "trace": tuple(trace),
                        "footprint": frozenset([marker, *trace]),
                    }
                )
    else:
        cols = list(range(c0 + 1, c0 + width - 1))
        _RNG.shuffle(cols)
        for col in cols:
            marker = (r0 - 1, col)
            wall = (r0, col)
            trace = [(r0 + height + offset, col) for offset in range(thickness)]
            if _site_ok(grid, marker, wall, trace):
                sites.append(
                    {
                        "bar": bar,
                        "marker_pos": marker,
                        "trace": tuple(trace),
                        "footprint": frozenset([marker, *trace]),
                    }
                )

            marker = (r0 + height, col)
            wall = (r0 + height - 1, col)
            trace = [(r0 - 1 - offset, col) for offset in range(thickness)]
            if _site_ok(grid, marker, wall, trace):
                sites.append(
                    {
                        "bar": bar,
                        "marker_pos": marker,
                        "trace": tuple(trace),
                        "footprint": frozenset([marker, *trace]),
                    }
                )

    return sites


def _pick_marker_count() -> int:
    roll = float(_RNG.random())
    if roll < 0.15:
        return 1
    if roll < 0.60:
        return 2
    return 3


def _pick_match_flags(count: int) -> list[bool]:
    if count == 1:
        return [True]
    if count == 2:
        return [True, False]
    return [True, False, bool(_RNG.integers(0, 2))]


def _choose_marker_sites(
    eligible: list[dict[str, object]],
    count: int,
) -> list[dict[str, object]] | None:
    by_thickness: dict[int, list[dict[str, object]]] = {}
    for item in eligible:
        thickness = int(item["bar"]["thickness"])
        by_thickness.setdefault(thickness, []).append(item)

    chosen: list[dict[str, object]] = []
    used_bars: set[int] = set()
    blocked: set[tuple[int, int]] = set()

    for thickness in [int(x) for x in _RNG.permutation([1, 2, 3])]:
        candidates = by_thickness.get(thickness, [])
        _RNG.shuffle(candidates)
        for item in candidates:
            bar_id = id(item["bar"])
            footprint = set(item["footprint"])
            if bar_id in used_bars or blocked & footprint:
                continue
            chosen.append(item)
            used_bars.add(bar_id)
            blocked |= footprint
            break
        if len(chosen) == count:
            return chosen

    leftovers = eligible[:]
    _RNG.shuffle(leftovers)
    for item in leftovers:
        bar_id = id(item["bar"])
        footprint = set(item["footprint"])
        if bar_id in used_bars or blocked & footprint:
            continue
        chosen.append(item)
        used_bars.add(bar_id)
        blocked |= footprint
        if len(chosen) == count:
            return chosen

    return None


def _gen_attempt() -> np.ndarray | None:
    h = _rand_int(MIN_SIZE, MAX_SIZE)
    w = _rand_int(MIN_SIZE, MAX_SIZE)
    placed = _place_bars(h, w)
    if placed is None:
        return None
    grid, bars = placed

    eligible: list[dict[str, object]] = []
    for bar in bars:
        for site in _candidate_sites(grid, bar):
            eligible.append(site)

    marker_count = _pick_marker_count()
    if len(eligible) < marker_count:
        return None

    sites = _choose_marker_sites(eligible, marker_count)
    if sites is None or len(sites) != marker_count:
        return None

    match_flags = _pick_match_flags(marker_count)
    _RNG.shuffle(match_flags)

    palette = sorted({int(bar["color"]) for bar in bars})
    if len(palette) < 2:
        return None

    expected_yellow = 0
    for site, is_match in zip(sites, match_flags):
        bar = site["bar"]
        marker_color = int(bar["color"])
        if not is_match:
            other_colors = [color for color in palette if color != marker_color]
            marker_color = int(_RNG.choice(other_colors))
        mr, mc = site["marker_pos"]
        grid[mr, mc] = marker_color
        if is_match:
            expected_yellow += int(bar["thickness"])

    detected = _find_markers(grid)
    if len(detected) != marker_count:
        return None

    out = main(grid)
    if int(np.count_nonzero(out == YELLOW)) != expected_yellow:
        return None

    for site in sites:
        mr, mc = site["marker_pos"]
        if int(out[mr, mc]) != BLACK:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze input")
