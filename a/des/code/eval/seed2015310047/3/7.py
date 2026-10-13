import numpy as np
from numpy.random import default_rng


BLACK = 0
GREY = 5
AZURE = 8
LINE_COLORS = (1, 2, 3, 4, 6, 7, 9)

MIN_SIZE = 12
MAX_SIZE = 20
MIN_H_LINES = 2
MAX_H_LINES = 3
MIN_V_LINES = 2
MAX_V_LINES = 3
MIN_GAP = 2
MAX_GENERATION_ATTEMPTS = 300


def _has_adjacent_marker(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == AZURE:
            return True
    return False


def _sample_spaced_positions(
    rng: np.random.Generator,
    count: int,
    low: int,
    high: int,
    min_gap: int,
) -> list[int] | None:
    if low > high:
        return None
    candidates = list(range(low, high + 1))
    for _ in range(100):
        rng.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(value - other) >= min_gap for other in chosen):
                chosen.append(int(value))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _reconstruct_lines(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BLACK, AZURE)]
    horizontals: list[tuple[int, int, int, int]] = []
    verticals: list[tuple[int, int, int, int]] = []

    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if rows.size == 1 and cols.size >= 2:
            row = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert any(_has_adjacent_marker(grid, row, endpoint) for endpoint in (c0, c1))
            horizontals.append((row, c0, c1, color))
        elif cols.size == 1 and rows.size >= 2:
            col = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert any(_has_adjacent_marker(grid, endpoint, col) for endpoint in (r0, r1))
            verticals.append((col, r0, r1, color))
        else:
            raise AssertionError(f"color {color} is not a single straight line")

    assert horizontals and verticals
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each full straight line and mark the hidden crossing cells in grey.

    Every non-marker color belongs to one horizontal or vertical line. If a cell
    lies on a line's full row/column span but is currently occupied by another
    line color, that line is hidden there; the output renders that hidden cell
    as grey. Azure endpoint markers are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    horizontals, verticals = _reconstruct_lines(g)
    h_cover: dict[tuple[int, int], int] = {}
    v_cover: dict[tuple[int, int], int] = {}

    for row, c0, c1, color in horizontals:
        for col in range(c0, c1 + 1):
            pos = (row, col)
            assert pos not in h_cover
            h_cover[pos] = color

    for col, r0, r1, color in verticals:
        for row in range(r0, r1 + 1):
            pos = (row, col)
            assert pos not in v_cover
            v_cover[pos] = color

    out = g.copy()
    out[out == AZURE] = BLACK

    for pos, color in h_cover.items():
        value = int(g[pos])
        if value != color:
            assert pos in v_cover, f"horizontal gap at {pos} is not covered by a vertical line"
            assert value == v_cover[pos], f"unexpected covering color at {pos}: {value}"
            out[pos] = GREY

    for pos, color in v_cover.items():
        value = int(g[pos])
        if value != color:
            assert pos in h_cover, f"vertical gap at {pos} is not covered by a horizontal line"
            assert value == h_cover[pos], f"unexpected covering color at {pos}: {value}"
            out[pos] = GREY

    return out


def _endpoint_marker_candidates(line: dict, endpoint: str) -> list[tuple[int, int]]:
    if line["kind"] == "h":
        row = line["row"]
        if endpoint == "start":
            col = line["c0"]
            return [(row - 1, col), (row + 1, col), (row, col - 1)]
        col = line["c1"]
        return [(row - 1, col), (row + 1, col), (row, col + 1)]

    col = line["col"]
    if endpoint == "start":
        row = line["r0"]
        return [(row, col - 1), (row, col + 1), (row - 1, col)]
    row = line["r1"]
    return [(row, col - 1), (row, col + 1), (row + 1, col)]


def _paint_input(grid: np.ndarray, lines: list[dict], under_map: dict[tuple[int, int], str]) -> None:
    for line in lines:
        if line["kind"] == "h":
            row = line["row"]
            for col in range(line["c0"], line["c1"] + 1):
                if under_map.get((row, col)) == "h":
                    continue
                assert int(grid[row, col]) == BLACK
                grid[row, col] = line["color"]
        else:
            col = line["col"]
            for row in range(line["r0"], line["r1"] + 1):
                if under_map.get((row, col)) == "v":
                    continue
                assert int(grid[row, col]) == BLACK
                grid[row, col] = line["color"]


def _place_markers(rng: np.random.Generator, grid: np.ndarray, lines: list[dict]) -> bool:
    h, w = grid.shape
    for line in lines:
        endpoints = ["start", "end"]
        rng.shuffle(endpoints)
        want = 2 if rng.random() < 0.45 else 1
        placed = 0

        for endpoint in endpoints:
            candidates = _endpoint_marker_candidates(line, endpoint)
            rng.shuffle(candidates)
            for row, col in candidates:
                if not (0 <= row < h and 0 <= col < w):
                    continue
                if int(grid[row, col]) != BLACK:
                    continue
                grid[row, col] = AZURE
                placed += 1
                break
            if placed >= want:
                break

        if placed == 0:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    width = int(rng.integers(MIN_SIZE, MAX_SIZE + 1))
    num_h = int(rng.integers(MIN_H_LINES, MAX_H_LINES + 1))
    num_v = int(rng.integers(MIN_V_LINES, MAX_V_LINES + 1))

    if num_h + num_v > len(LINE_COLORS):
        return None

    rows = _sample_spaced_positions(rng, num_h, 2, height - 3, MIN_GAP)
    cols = _sample_spaced_positions(rng, num_v, 2, width - 3, MIN_GAP)
    if rows is None or cols is None:
        return None

    left_limit = min(cols) - 1
    right_limit = max(cols) + 1
    top_limit = min(rows) - 1
    bottom_limit = max(rows) + 1
    if left_limit < 1 or right_limit > width - 2 or top_limit < 1 or bottom_limit > height - 2:
        return None

    colors = list(map(int, rng.choice(LINE_COLORS, size=num_h + num_v, replace=False)))
    rng.shuffle(colors)

    horizontals: list[dict] = []
    verticals: list[dict] = []
    color_index = 0

    for row in rows:
        c0 = int(rng.integers(1, left_limit + 1))
        c1 = int(rng.integers(right_limit, width - 1))
        horizontals.append({"kind": "h", "row": row, "c0": c0, "c1": c1, "color": colors[color_index]})
        color_index += 1

    for col in cols:
        r0 = int(rng.integers(1, top_limit + 1))
        r1 = int(rng.integers(bottom_limit, height - 1))
        verticals.append({"kind": "v", "col": col, "r0": r0, "r1": r1, "color": colors[color_index]})
        color_index += 1

    cross_choices = rng.integers(0, 2, size=num_h * num_v)
    if 0 not in cross_choices or 1 not in cross_choices:
        return None

    under_map: dict[tuple[int, int], str] = {}
    idx = 0
    for h_line in horizontals:
        row = h_line["row"]
        for v_line in verticals:
            col = v_line["col"]
            under_map[(row, col)] = "h" if int(cross_choices[idx]) == 0 else "v"
            idx += 1

    grid = np.full((height, width), BLACK, dtype=int)
    lines = horizontals + verticals
    _paint_input(grid, lines, under_map)
    if not _place_markers(rng, grid, lines):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(grid == AZURE):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-line input")
