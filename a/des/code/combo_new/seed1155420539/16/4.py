import random

import numpy as np


BG = 0
COLORS = list(range(1, 10))

GEN_RETRIES = 400
HEIGHT_RANGE = (8, 24)
BAR_COUNT_RANGE = (2, 6)
MIN_WIDTH = 10
MAX_SIZE = 30


def _marker_lengths(grid: np.ndarray) -> dict[int, int]:
    height = int(grid.shape[0])
    markers: dict[int, int] = {}
    for row in range(height):
        color = int(grid[row, 0])
        if color == BG:
            continue
        assert color not in markers, "expected one ruler mark per color"
        markers[color] = height - row
    return markers


def _bars(grid: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = map(int, grid.shape)
    bars: list[tuple[int, int, int]] = []
    if width < 2:
        return bars
    for row in range(height):
        color = int(grid[row, 1])
        if color == BG:
            continue
        col = 1
        while col < width and int(grid[row, col]) == color:
            col += 1
        bars.append((row, color, col - 1))
    return bars


def _resize_bars(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width >= 2, "expected a ruler column plus interior columns"

    markers = _marker_lengths(g)
    out = g.copy()
    for row, color, _length in _bars(g):
        target = markers.get(color)
        if target is None:
            continue
        assert 1 <= target <= width - 1, "ruler length must fit inside the grid"
        out[row, 1:] = BG
        out[row, 1 : 1 + target] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Resize each horizontal color strip so its length matches the same-color mark on the left ruler.

    Input structure:
    - The background is black.
    - The leftmost column is a vertical ruler with isolated single-pixel color marks.
    - Each interior object is a one-cell-tall horizontal strip that starts in the second column and uses a unique color.

    Rule:
    - For each strip, find the ruler mark with the same color.
    - Convert that mark's row into a length by counting from the bottom, with the bottom row equal to 1.
    - If the strip is longer than that value, cut cells from its right end; if it is shorter, extend it rightward to that value.
    - Keep every strip on its original row and leave the ruler column unchanged.

    Key insight:
    - The ruler encodes a numeric target length through vertical position, not a destination row for moving the strip.
    """
    return _resize_bars(grid)


def _branch_plan(count: int, rng: random.Random) -> list[str]:
    branches = ["extend", "truncate"] * (count // 2)
    if count % 2:
        branches.append(rng.choice(["extend", "truncate", "equal"]))
    rng.shuffle(branches)
    return branches


def _sample_bar_rows(marker_rows: list[int], height: int, rng: random.Random) -> list[int] | None:
    available = [row for row in range(height) if row not in set(marker_rows)]
    bar_rows = rng.sample(available, len(marker_rows))
    for _ in range(80):
        candidate = bar_rows[:]
        rng.shuffle(candidate)
        if len(candidate) < 2:
            return candidate
        has_higher = any(bar_row < marker_row for bar_row, marker_row in zip(candidate, marker_rows))
        has_lower = any(bar_row > marker_row for bar_row, marker_row in zip(candidate, marker_rows))
        if has_higher and has_lower:
            return candidate
    return None


def _build_pair(height: int, width: int, specs: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    out = np.zeros_like(grid)
    for spec in specs:
        color = int(spec["color"])
        target = int(spec["target"])
        marker_row = height - target
        bar_row = int(spec["bar_row"])
        current = int(spec["current"])

        grid[marker_row, 0] = color
        out[marker_row, 0] = color
        grid[bar_row, 1 : 1 + current] = color
        out[bar_row, 1 : 1 + target] = color
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(max(MIN_WIDTH, height + 1), MAX_SIZE)

    max_bars = min(BAR_COUNT_RANGE[1], len(COLORS), height // 2, height - 2)
    if max_bars < BAR_COUNT_RANGE[0]:
        return None
    count = rng.randint(BAR_COUNT_RANGE[0], max_bars)

    colors = rng.sample(COLORS, count)
    targets = rng.sample(list(range(2, height)), count)
    marker_rows = [height - target for target in targets]
    bar_rows = _sample_bar_rows(marker_rows, height, rng)
    if bar_rows is None:
        return None

    branches = _branch_plan(count, rng)
    specs: list[dict[str, int]] = []
    for color, target, bar_row, branch in zip(colors, targets, bar_rows, branches):
        if branch == "extend":
            current = rng.randint(1, target - 1)
        elif branch == "truncate":
            current = rng.randint(target + 1, width - 1)
        else:
            current = target
        specs.append(
            {
                "color": int(color),
                "target": int(target),
                "bar_row": int(bar_row),
                "current": int(current),
            }
        )

    if not any(spec["current"] < spec["target"] for spec in specs):
        return None
    if not any(spec["current"] > spec["target"] for spec in specs):
        return None

    grid, expected = _build_pair(height, width, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
