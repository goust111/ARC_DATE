import numpy as np


BACKGROUND = 0
FLOOR = 5
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MIN_HEIGHT = 12
MAX_HEIGHT = 15
MIN_WIDTH = 20
MAX_WIDTH = 28
MIN_OBJECTS = 6
MAX_OBJECTS = 8
MIN_BAR_WIDTH = 2
MAX_BAR_WIDTH = 4
MAX_RETRIES = 400
SELECTION_PERIOD = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_bars(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    mask = (grid != BACKGROUND) & (grid != FLOOR)
    seen = np.zeros((H, W), dtype=bool)
    bars = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            rows = np.unique(coords_arr[:, 0])
            cols = np.sort(coords_arr[:, 1])
            colors = np.unique(grid[coords_arr[:, 0], coords_arr[:, 1]])

            assert len(rows) == 1, "each object must be a single-row bar"
            assert len(colors) == 1, "each object must be monochromatic"
            assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "each bar must be contiguous"

            bars.append(
                {
                    "row": int(rows[0]),
                    "c0": int(cols[0]),
                    "c1": int(cols[-1]),
                    "width": int(cols[-1] - cols[0] + 1),
                    "color": int(colors[0]),
                }
            )

    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split every third bar by order: selected bars drop near the floor, and the others mirror horizontally.

    Input structure:
    - Black cells form the background.
    - The bottom row is a solid grey floor.
    - Above the floor are several single-row horizontal monochromatic bars with pairwise distinct colors and non-overlapping column spans.

    Rule:
    - Order the bars by top row, then by left column.
    - Select ranks 1, 4, 7, ... in that order.
    - Clear all original bars; move each selected bar straight down within its own columns so it lands one empty row above the grey floor, and move each unselected bar to its left-right mirror position across the grid center while keeping its row fixed.

    Key insight:
    - The output mixes two different motions, and the only way to decide which bar uses which motion is the hidden every-third pattern in top-to-bottom order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 2
    assert np.all(g[-1, :] == FLOOR), "expected a grey floor on the bottom row"

    bars = sorted(_find_bars(g), key=lambda bar: (bar["row"], bar["c0"]))
    out = np.full_like(g, BACKGROUND)
    out[-1, :] = FLOOR
    target_row = g.shape[0] - 3

    for index, bar in enumerate(bars):
        if index % SELECTION_PERIOD == 0:
            row = target_row
            c0 = bar["c0"]
        else:
            row = bar["row"]
            c0 = g.shape[1] - 1 - bar["c1"]
        out[row, c0 : c0 + bar["width"]] = bar["color"]

    return out


def _random_widths(n_objects: int, canvas_width: int) -> list[int] | None:
    for _ in range(80):
        widths = np.random.randint(MIN_BAR_WIDTH, MAX_BAR_WIDTH + 1, size=n_objects).tolist()
        if sum(widths) <= canvas_width - 2:
            return [int(w) for w in widths]
    return None


def _intervals_from_widths(widths: list[int], canvas_width: int) -> list[tuple[int, int]]:
    slack = canvas_width - sum(widths)
    gaps = np.random.multinomial(slack, np.full(len(widths) + 1, 1.0 / (len(widths) + 1)))
    intervals = []
    cursor = int(gaps[0])
    for i, width in enumerate(widths):
        c0 = cursor
        c1 = c0 + int(width) - 1
        intervals.append((int(c0), int(c1)))
        cursor = c1 + 1 + int(gaps[i + 1])
    return intervals


def _is_self_mirrored(c0: int, c1: int, canvas_width: int) -> bool:
    return c0 == canvas_width - 1 - c1


def _build_candidate() -> np.ndarray | None:
    H = int(np.random.randint(MIN_HEIGHT, MAX_HEIGHT + 1))
    W = int(np.random.randint(MIN_WIDTH, MAX_WIDTH + 1))
    n_objects = int(np.random.randint(MIN_OBJECTS, MAX_OBJECTS + 1))

    rows = np.random.choice(np.arange(0, H - 3), size=n_objects, replace=False)
    rows.sort()

    widths = _random_widths(n_objects, W)
    if widths is None:
        return None

    intervals = _intervals_from_widths(widths, W)
    perm = np.random.permutation(n_objects)
    intervals = [intervals[int(i)] for i in perm]
    widths = [widths[int(i)] for i in perm]

    for i in range(n_objects):
        c0, c1 = intervals[i]
        if i % SELECTION_PERIOD != 0 and _is_self_mirrored(c0, c1, W):
            return None

    colors = np.random.permutation(OBJECT_COLORS)[:n_objects]
    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[-1, :] = FLOOR

    for row, color, (c0, c1) in zip(rows, colors, intervals):
        grid[int(row), int(c0) : int(c1) + 1] = int(color)

    return grid


def _valid_candidate(grid: np.ndarray) -> bool:
    bars = sorted(_find_bars(grid), key=lambda bar: (bar["row"], bar["c0"]))
    if len(bars) < MIN_OBJECTS:
        return False
    if len({bar["color"] for bar in bars}) != len(bars):
        return False
    if len({bar["row"] for bar in bars}) != len(bars):
        return False
    if any(bar["row"] >= grid.shape[0] - 3 for bar in bars):
        return False

    out = main(grid)
    target_row = grid.shape[0] - 3
    selected = [bar for i, bar in enumerate(bars) if i % SELECTION_PERIOD == 0]
    unselected = [bar for i, bar in enumerate(bars) if i % SELECTION_PERIOD != 0]

    if not selected or not unselected:
        return False
    if not all(np.all(out[target_row, bar["c0"] : bar["c1"] + 1] == bar["color"]) for bar in selected):
        return False
    if not all(
        np.all(
            out[
                bar["row"],
                grid.shape[1] - 1 - bar["c1"] : grid.shape[1] - bar["c0"],
            ]
            == bar["color"]
        )
        for bar in unselected
    ):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_candidate()
        if grid is not None and _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
