import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    GREY = 5


GRID_MIN = 17
GRID_MAX = 24
BAR_COUNT_CHOICES = (4, 5, 6)
BAR_LENGTHS = (2, 3, 4, 5, 6, 7)
MAX_GENERATION_ATTEMPTS = 500
MIN_DISTINCT_DIRECTIONS = 3
MIN_TURNS = 3


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def find_color_components4(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in neighbors4(int(cr), int(cc)):
                    if in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def bar_info(coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    same_row = bool(np.all(rows == rows[0]))
    same_col = bool(np.all(cols == cols[0]))
    if same_row == same_col:
        raise AssertionError("gray component must be a non-degenerate straight bar")
    length = int(coords.shape[0])
    if same_row:
        if length != c1 - c0 + 1:
            raise AssertionError("horizontal bar must be gap-free")
        orientation = "H"
    else:
        if length != r1 - r0 + 1:
            raise AssertionError("vertical bar must be gap-free")
        orientation = "V"
    return {
        "coords": coords,
        "length": length,
        "orientation": orientation,
        "center_r2": r0 + r1,
        "center_c2": c0 + c1,
    }


def decode_length(length: int) -> tuple[int, int]:
    cycle = (
        (0, 1),   # right
        (1, 0),   # down
        (0, -1),  # left
        (-1, 0),  # up
    )
    if not 2 <= int(length) <= 7:
        raise AssertionError("bar lengths must be between 2 and 7")
    return cycle[(int(length) - 2) % 4]


def direction_name(length: int) -> str:
    labels = ("R", "D", "L", "U")
    if not 2 <= int(length) <= 7:
        raise AssertionError("bar lengths must be between 2 and 7")
    return labels[(int(length) - 2) % 4]


def trace_path(lengths: list[int], start: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int], list[str]]:
    cur_r, cur_c = map(int, start)
    painted: list[tuple[int, int]] = []
    dirs: list[str] = []
    for length in lengths:
        dr, dc = decode_length(int(length))
        dirs.append(direction_name(int(length)))
        for step in range(int(length)):
            painted.append((int(cur_r + dr * step), int(cur_c + dc * step)))
        cur_r += dr * int(length)
        cur_c += dc * int(length)
    return painted, (int(cur_r), int(cur_c)), dirs


def render_expected(grid: np.ndarray, painted: list[tuple[int, int]], endpoint: tuple[int, int], start: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in painted:
        out[r, c] = Color.BLUE
    out[start] = Color.RED
    out[grid == Color.GREY] = Color.GREY
    out[endpoint] = Color.GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a polyline from sorted gray bar lengths and draw it from the red start marker.

    Input structure:
    - A mostly black grid contains 4 to 6 disconnected gray 1-cell-thick straight bars.
    - The gray bars may be horizontal or vertical, but only their pixel counts matter.
    - A single red start pixel appears in the lower-right region on the background.

    Rule:
    - Find every gray 4-connected bar and measure its length.
    - Sort the lengths from shortest to longest.
    - Decode each length into a movement using the repeating direction cycle right, down, left, up, with movement distance equal to the length itself.
    - Starting from the red pixel, draw every segment in blue on all traversed cells including each segment's start and excluding its endpoint, keep the gray bars and red start unchanged, and place a green pixel at the final endpoint.

    Key insight:
    - The bars' locations and orientations are distractions: only relative length order determines the instruction sequence, and those decoded instructions then drive the drawn polyline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_pixels = np.argwhere(g == Color.RED)
    assert start_pixels.shape[0] == 1, "expected exactly one red start pixel"
    start = tuple(map(int, start_pixels[0]))

    comps = find_color_components4(g, Color.GREY)
    assert 4 <= len(comps) <= 6, "expected 4 to 6 gray bars"
    bars = [bar_info(comp) for comp in comps]
    lengths = [bar["length"] for bar in bars]
    assert len(set(lengths)) == len(lengths), "bar lengths must be distinct"

    ordered_lengths = sorted(lengths)
    painted, endpoint, _dirs = trace_path(ordered_lengths, start)

    for r, c in painted + [endpoint]:
        assert in_bounds(g.shape, r, c), "decoded path must stay in bounds"

    out = render_expected(g, painted, endpoint, start)
    return out


def choose_lengths(rng: random.Random, count: int) -> list[int]:
    while True:
        lengths = sorted(rng.sample(BAR_LENGTHS, count))
        if count == 4 and lengths == [2, 3, 4, 5]:
            continue
        if count == 5 and lengths == [2, 3, 4, 5, 6]:
            continue
        return lengths


def feasible_start_ranges(shape: tuple[int, int], rel_cells: list[tuple[int, int]]) -> tuple[range, range] | None:
    h, w = map(int, shape)
    rows = [r for r, _c in rel_cells]
    cols = [c for _r, c in rel_cells]
    sr_min = -min(rows)
    sr_max = h - 1 - max(rows)
    sc_min = -min(cols)
    sc_max = w - 1 - max(cols)

    lower_r_min = max(sr_min, int(np.ceil(h * 0.55)))
    lower_c_min = max(sc_min, int(np.ceil(w * 0.55)))
    if lower_r_min > sr_max or lower_c_min > sc_max:
        return None
    return range(lower_r_min, sr_max + 1), range(lower_c_min, sc_max + 1)


def placement_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not in_bounds(shape, r, c) or (r, c) in occupied:
            return False
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc) and (nr, nc) in occupied and (nr, nc) not in cell_set:
                return False
    return True


def place_bars(
    shape: tuple[int, int],
    lengths: list[int],
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> list[dict] | None:
    h, w = map(int, shape)
    occupied = set(forbidden)
    placed: list[dict] = []
    order = list(lengths)
    rng.shuffle(order)

    for length in order:
        candidates: list[tuple[str, list[tuple[int, int]]]] = []
        for orientation in ("H", "V"):
            if orientation == "H":
                for r in range(h):
                    for c in range(w - int(length) + 1):
                        cells = [(int(r), int(c + dc)) for dc in range(int(length))]
                        if placement_clear(cells, occupied, shape):
                            candidates.append((orientation, cells))
            else:
                for r in range(h - int(length) + 1):
                    for c in range(w):
                        cells = [(int(r + dr), int(c)) for dr in range(int(length))]
                        if placement_clear(cells, occupied, shape):
                            candidates.append((orientation, cells))
        if not candidates:
            return None

        orientation, cells = rng.choice(candidates)
        coords = np.array(cells, dtype=int)
        info = bar_info(coords)
        info["orientation"] = orientation
        placed.append(info)
        occupied.update(cells)
    return placed


def bar_orders(bars: list[dict]) -> tuple[list[int], list[int]]:
    by_row = [bar["length"] for bar in sorted(bars, key=lambda bar: (bar["center_r2"], bar["center_c2"]))]
    by_col = [bar["length"] for bar in sorted(bars, key=lambda bar: (bar["center_c2"], bar["center_r2"]))]
    return by_row, by_col


def validate_example(
    grid: np.ndarray,
    lengths: list[int],
    start: tuple[int, int],
    painted: list[tuple[int, int]],
    endpoint: tuple[int, int],
    bars: list[dict],
) -> None:
    h, w = grid.shape
    assert 0 < h <= 30 and 0 < w <= 30
    assert grid[start] == Color.RED
    assert start[0] >= int(np.ceil(h * 0.55)) and start[1] >= int(np.ceil(w * 0.55))
    assert len(set(lengths)) == len(lengths)
    assert 4 <= len(lengths) <= 6
    assert len(set(painted)) == len(painted), "avoid self-overlapping path"
    assert endpoint not in set(painted), "endpoint should stay distinct from painted cells"

    decoded_dirs = [direction_name(length) for length in lengths]
    assert len(set(decoded_dirs)) >= MIN_DISTINCT_DIRECTIONS
    turns = sum(decoded_dirs[i] != decoded_dirs[i - 1] for i in range(1, len(decoded_dirs)))
    assert turns >= MIN_TURNS

    bar_lengths = sorted(bar["length"] for bar in bars)
    assert bar_lengths == lengths
    assert len({bar["orientation"] for bar in bars}) == 2, "need both bar orientations"
    matches = [
        bar["orientation"] == ("H" if decode_length(bar["length"])[1] else "V")
        for bar in sorted(bars, key=lambda bar: bar["length"])
    ]
    assert any(matches) and not all(matches), "orientation must be an unreliable cue"

    by_row, by_col = bar_orders(bars)
    assert by_row != lengths and by_row[::-1] != lengths, "row order must not reveal instruction order"
    assert by_col != lengths and by_col[::-1] != lengths, "col order must not reveal instruction order"

    alt_painted_row, alt_endpoint_row, _ = trace_path(by_row, start)
    alt_painted_col, alt_endpoint_col, _ = trace_path(by_col, start)
    assert alt_painted_row != painted or alt_endpoint_row != endpoint
    assert alt_painted_col != painted or alt_endpoint_col != endpoint

    grey_cells = set(map(tuple, np.argwhere(grid == Color.GREY)))
    assert not grey_cells.intersection(set(painted) - {start}), "bars should not overlap the drawn path"
    assert endpoint not in grey_cells
    assert endpoint != start

    out = main(grid)
    expected = render_expected(grid, painted, endpoint, start)
    assert np.array_equal(out, expected)
    assert int(np.count_nonzero(out == Color.GREEN)) == 1
    assert int(np.count_nonzero(out == Color.RED)) == 1
    assert np.all(out[grid == Color.GREY] == Color.GREY)
    assert out[start] == Color.RED


def gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    count = rng.choices(BAR_COUNT_CHOICES, weights=[2, 3, 3], k=1)[0]
    lengths = choose_lengths(rng, count)

    rel_painted, rel_endpoint, _dirs = trace_path(lengths, (0, 0))
    if len(set(rel_painted)) != len(rel_painted):
        return None

    rel_all = rel_painted + [rel_endpoint]
    ranges = feasible_start_ranges((h, w), rel_all)
    if ranges is None:
        return None
    row_range, col_range = ranges
    start = (rng.choice(list(row_range)), rng.choice(list(col_range)))

    painted = [(int(r + start[0]), int(c + start[1])) for r, c in rel_painted]
    endpoint = (int(rel_endpoint[0] + start[0]), int(rel_endpoint[1] + start[1]))
    forbidden = set(painted)
    forbidden.add(endpoint)
    for r, c in list(forbidden):
        for nr, nc in neighbors4(r, c):
            if in_bounds((h, w), nr, nc):
                forbidden.add((int(nr), int(nc)))

    bars = place_bars((h, w), lengths, forbidden, rng)
    if bars is None:
        return None

    row_span = max(bar["center_r2"] for bar in bars) - min(bar["center_r2"] for bar in bars)
    col_span = max(bar["center_c2"] for bar in bars) - min(bar["center_c2"] for bar in bars)
    if row_span < 4 or col_span < 4:
        return None

    grid = np.zeros((h, w), dtype=int)
    for bar in bars:
        coords = bar["coords"]
        grid[coords[:, 0], coords[:, 1]] = Color.GREY
    grid[start] = Color.RED

    try:
        validate_example(grid, lengths, start, painted, endpoint, bars)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid length-coded polyline puzzle")
