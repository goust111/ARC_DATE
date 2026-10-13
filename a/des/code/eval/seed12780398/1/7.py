import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FILTER_COLORS = (RED, BLUE, GREEN)
NONEMPTY_COLOR_SETS = (
    frozenset((RED,)),
    frozenset((BLUE,)),
    frozenset((GREEN,)),
    frozenset((RED, BLUE)),
    frozenset((RED, GREEN)),
    frozenset((BLUE, GREEN)),
    frozenset((RED, BLUE, GREEN)),
)
ALL_RECEIVER_SETS = (frozenset(),) + NONEMPTY_COLOR_SETS

MIX_TO_OUTPUT = {
    frozenset(): GREY,
    frozenset((RED,)): RED,
    frozenset((BLUE,)): BLUE,
    frozenset((GREEN,)): GREEN,
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, GREEN)): YELLOW,
    frozenset((BLUE, GREEN)): AZURE,
    frozenset((RED, BLUE, GREEN)): MAROON,
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _validate_filter_bars(grid: np.ndarray) -> None:
    for color in FILTER_COLORS:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            assert r0 == r1, f"filter color {color} must form a horizontal bar"
            assert coords.shape[0] == c1 - c0 + 1, f"filter color {color} bar must be contiguous"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine receiver colors from downward rays emitted by orange 2x2 towers.

    Rays travel straight down from the left column of each tower's bottom edge.
    Each ray collects the set of filter colors it crosses. Every receiver unions
    the color sets of all rays in its column, then maps that union to a final
    output color. Azure antenna markers are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    unique = set(map(int, np.unique(g)))
    allowed = {BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON}
    assert unique <= allowed

    _validate_filter_bars(g)

    tower_rays: list[tuple[int, int]] = []
    for coords in _components(g == ORANGE):
        r0, c0, r1, c1 = _bbox(coords)
        assert coords.shape[0] == 4, "orange objects must be 2x2 towers"
        assert r1 - r0 == 1 and c1 - c0 == 1, "orange objects must be 2x2 towers"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == ORANGE), "tower must be solid"
        tower_rays.append((r1, c0))

    receivers: list[tuple[int, int]] = []
    antenna_coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        assert rr > 0 and int(g[rr - 1, cc]) == AZURE, "receiver must have an azure marker above it"
        receivers.append((rr, cc))
        antenna_coords.append((rr - 1, cc))

    for r, c in np.argwhere(g == AZURE):
        rr = int(r)
        cc = int(c)
        assert rr + 1 < g.shape[0] and int(g[rr + 1, cc]) == GREY, "azure pixels must be receiver markers"

    ray_union_by_col: dict[int, set[int]] = defaultdict(set)
    for bottom_row, col in tower_rays:
        seen_colors: set[int] = set()
        for r in range(bottom_row + 1, g.shape[0]):
            color = int(g[r, col])
            if color in FILTER_COLORS:
                seen_colors.add(color)
        ray_union_by_col[int(col)].update(seen_colors)

    out = g.copy()
    for r, c in antenna_coords:
        out[r, c] = BLACK

    for r, c in receivers:
        mix = frozenset(ray_union_by_col.get(int(c), set()))
        out[r, c] = MIX_TO_OUTPUT[mix]

    return out


def _sample_tower_columns(width: int, count: int) -> list[int] | None:
    starts = list(range(width - 1))
    for _ in range(300):
        cols = sorted(random.sample(starts, count))
        if all(b - a >= 3 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_filter_runs(sorted_cols: list[int], sorted_sets: list[frozenset[int]]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    for color in FILTER_COLORS:
        i = 0
        while i < len(sorted_cols):
            if color not in sorted_sets[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(sorted_cols) and color in sorted_sets[j + 1]:
                j += 1
            runs.append((color, sorted_cols[i], sorted_cols[j]))
            i = j + 1
    random.shuffle(runs)
    return runs


def _sample_empty_receiver_columns(width: int, tower_cols: list[int], count: int) -> list[int] | None:
    used = set(tower_cols)
    preferred = [c + 1 for c in tower_cols if c + 1 < width and c + 1 not in used]
    other = [c for c in range(width) if c not in used and c not in preferred]
    chosen: list[int] = []

    if count > 0 and preferred and random.random() < 0.75:
        chosen.append(random.choice(preferred))
        used.add(chosen[-1])
        preferred = [c for c in preferred if c not in used]

    remaining = count - len(chosen)
    pool = preferred + other
    if remaining > len(pool):
        return None
    chosen.extend(random.sample(pool, remaining))
    return chosen


def _receiver_specs() -> list[frozenset[int]]:
    for _ in range(100):
        receiver_count = random.randint(3, 7)
        mixes = [random.choice(ALL_RECEIVER_SETS) for _ in range(receiver_count)]
        nonempty = [mix for mix in mixes if mix]
        if not (2 <= len(nonempty) <= 5):
            continue
        if len(set(mixes)) < 2:
            continue
        if not any(len(s) >= 2 for s in nonempty) and random.random() < 0.5:
            continue
        return mixes
    return [frozenset((RED,)), frozenset((BLUE, GREEN)), frozenset()]


def _gen_attempt() -> np.ndarray | None:
    receiver_mixes = _receiver_specs()
    nonempty_sets = [mix for mix in receiver_mixes if mix]
    empty_count = len(receiver_mixes) - len(nonempty_sets)
    tower_count = len(nonempty_sets)

    min_width = max(12, 3 * tower_count + empty_count + 3)
    width = random.randint(min_width, min_width + 7)
    tower_cols = _sample_tower_columns(width, tower_count)
    if tower_cols is None:
        return None

    tower_rows = [random.randint(0, 2) for _ in range(tower_count)]
    ordered = sorted(zip(tower_cols, nonempty_sets, tower_rows), key=lambda item: item[0])
    sorted_cols = [int(col) for col, _mix, _row in ordered]
    sorted_sets = [mix for _col, mix, _row in ordered]
    sorted_rows = [int(row) for _col, _mix, row in ordered]

    filter_runs = _build_filter_runs(sorted_cols, sorted_sets)
    max_tower_bottom = max(row + 1 for row in sorted_rows)

    min_height = max(14, max_tower_bottom + len(filter_runs) + 7)
    height = random.randint(min_height, min_height + 6)
    antenna_row = height - 2
    receiver_row = height - 1
    bar_rows = list(range(max_tower_bottom + 2, height - 3))
    if len(bar_rows) < len(filter_runs):
        return None
    chosen_bar_rows = random.sample(bar_rows, len(filter_runs))

    empty_cols = _sample_empty_receiver_columns(width, tower_cols, empty_count)
    if empty_cols is None:
        return None

    grid = np.zeros((height, width), dtype=int)

    receiver_expectation: dict[int, frozenset[int]] = {}
    for col, mix, row in zip(tower_cols, nonempty_sets, tower_rows):
        assert row + 1 < antenna_row
        assert np.all(grid[row : row + 2, col : col + 2] == BLACK)
        grid[row : row + 2, col : col + 2] = ORANGE
        receiver_expectation[int(col)] = mix

    for (color, c0, c1), row in zip(filter_runs, chosen_bar_rows):
        assert 0 <= row < antenna_row
        assert np.all(grid[row, c0 : c1 + 1] == BLACK)
        grid[row, c0 : c1 + 1] = int(color)

    for col in empty_cols:
        receiver_expectation[int(col)] = frozenset()

    for col in sorted(receiver_expectation):
        assert grid[antenna_row, col] == BLACK and grid[receiver_row, col] == BLACK
        grid[antenna_row, col] = AZURE
        grid[receiver_row, col] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None

    if not any(out[receiver_row, col] != GREY for col in receiver_expectation):
        return None

    for col, mix in receiver_expectation.items():
        if int(out[antenna_row, col]) != BLACK:
            return None
        if int(out[receiver_row, col]) != MIX_TO_OUTPUT[mix]:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
