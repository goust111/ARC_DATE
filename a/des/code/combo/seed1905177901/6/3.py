import random
from collections import Counter

import numpy as np


AZURE = 8
COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    if AZURE in counts:
        return AZURE
    return int(counts.most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int):
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "size": int(arr.shape[0]),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect same-color markers into bottom-aligned vertical collectors.

    Input structure:
    - Azure background.
    - One-pixel-wide vertical bars act as collectors; each is at least 3 cells tall.
    - Same-color singleton markers appear above the collectors.

    Rule:
    - For each collector, count same-color markers in the same column above it.
    - Remove all markers.
    - Grow the collector upward by that count while keeping its column and bottom row fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    comps = _components(g, bg)
    assert comps, "expected at least one non-background object"

    collectors = []
    markers = []
    for comp in comps:
        height = int(comp["r1"] - comp["r0"] + 1)
        width = int(comp["c1"] - comp["c0"] + 1)
        if int(comp["size"]) == 1:
            markers.append(comp)
        else:
            assert width == 1, "collectors must be one cell wide"
            assert int(comp["size"]) == height, "collectors must be solid vertical bars"
            assert height >= 3, "collectors must be at least three cells tall"
            collectors.append({**comp, "height": height, "col": int(comp["c0"])})

    assert collectors, "expected at least one collector"
    baseline_rows = {int(comp["r1"]) for comp in collectors}
    assert len(baseline_rows) == 1, "collectors must share one bottom row"
    baseline = baseline_rows.pop()

    growth = {}
    occupied_cols = set()
    for collector in collectors:
        key = (int(collector["col"]), int(collector["color"]))
        assert key not in growth, "duplicate collector with same column and color"
        assert int(collector["col"]) not in occupied_cols, "expected one collector per column"
        growth[key] = 0
        occupied_cols.add(int(collector["col"]))

    collector_by_key = {(int(comp["col"]), int(comp["color"])): comp for comp in collectors}
    for marker in markers:
        row, col = map(int, marker["coords"][0])
        key = (int(col), int(marker["color"]))
        assert key in collector_by_key, "each marker must match a same-color collector in its column"
        assert row < int(collector_by_key[key]["r0"]), "markers must be above their collector"
        growth[key] += 1

    out = np.full_like(g, bg)
    for collector in collectors:
        key = (int(collector["col"]), int(collector["color"]))
        new_height = int(collector["height"] + growth[key])
        top = int(baseline - new_height + 1)
        assert top >= 0, "collector growth must stay inside the grid"
        out[top : baseline + 1, int(collector["col"])] = int(collector["color"])

    return out


def _sample_spaced_columns(width: int, count: int):
    if count == 0:
        return []
    if count > (width + 1) // 2:
        return None
    for _ in range(80):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _sample_nonadjacent_positions(length: int, count: int):
    if count == 0:
        return []
    if count > (length + 1) // 2:
        return None
    base = sorted(random.sample(range(length - count + 1), count))
    return [int(value + offset) for offset, value in enumerate(base)]


def _gen_attempt():
    height = random.randint(10, 22)
    width = random.randint(8, 20)
    baseline = random.randint(height // 2 + 1, height - 1)

    max_collectors = min(6, (width + 1) // 2)
    if max_collectors < 2:
        return None
    collector_count = random.randint(2, max_collectors)
    columns = _sample_spaced_columns(width, collector_count)
    if columns is None:
        return None

    colors = random.sample(COLORS, collector_count)
    heights = []
    capacities = []
    for _ in range(collector_count):
        max_height = min(6, baseline - 1)
        if max_height < 3:
            return None
        bar_height = random.randint(3, max_height)
        heights.append(bar_height)
        capacities.append((baseline - bar_height + 1 - 1 + 1) // 2)

    if max(capacities) <= 0:
        return None

    marker_counts = []
    for cap in capacities:
        marker_counts.append(random.randint(0, min(cap, 5)))

    positive_indices = [idx for idx, count in enumerate(marker_counts) if count > 0]
    if not positive_indices:
        choices = [idx for idx, cap in enumerate(capacities) if cap > 0]
        idx = random.choice(choices)
        marker_counts[idx] = random.randint(1, min(capacities[idx], 5))
        positive_indices = [idx]

    if collector_count >= 2 and random.random() < 0.6 and all(count > 0 for count in marker_counts):
        idx = random.randrange(collector_count)
        marker_counts[idx] = 0

    grid = np.full((height, width), AZURE, dtype=int)
    for col, color, bar_height, marker_count in zip(columns, colors, heights, marker_counts):
        top = baseline - bar_height + 1
        grid[top : baseline + 1, col] = color

        marker_rows = _sample_nonadjacent_positions(top - 1, marker_count)
        if marker_rows is None:
            return None
        for row in marker_rows:
            grid[row, col] = color

    fg = int(np.count_nonzero(grid != AZURE))
    bg = int(grid.size - fg)
    if bg <= fg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
