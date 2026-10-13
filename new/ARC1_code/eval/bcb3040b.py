import numpy as np

from arc_utils import coords_where
import math
import random


def _unit_step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "markers must be queen-aligned"

    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    steps = max(abs(dr), abs(dc))
    assert steps > 0, "expected two distinct marker pixels"
    return int(sr), int(sc), int(steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the straight segment between the two red markers, turning crossed blue pixels green and crossed empty pixels red.

    Input structure
    - The input contains only black background, blue foreground pixels, and exactly two red marker pixels.
    - The two red markers lie on one horizontal, vertical, or 45-degree diagonal line.

    Rule
    - Copy the input grid.
    - Walk the inclusive straight segment from one red marker to the other.
    - Each path cell that was blue becomes green; each path cell that was black or red becomes red.

    Key insight
    - The red marker pair defines a queen-move path, and the output records which original cells on that path were blue versus not blue.
    """
    g = np.asarray(grid, dtype=int)
    assert set(map(int, np.unique(g))) <= {0, 1, 2}, "expected only black, blue, and red in the input"

    marker_coords = coords_where(g == 2)
    assert marker_coords.shape == (2, 2), "expected exactly two red markers"
    start = tuple(map(int, marker_coords[0]))
    end = tuple(map(int, marker_coords[1]))
    sr, sc, steps = _unit_step(start, end)

    out = g.copy()
    r0, c0 = start
    for i in range(steps + 1):
        r = r0 + i * sr
        c = c0 + i * sc
        cell = int(g[r, c])
        assert cell in (0, 1, 2), "path cell has an unexpected color"
        out[r, c] = 3 if cell == 1 else 2

    return out

# --------------- generator (o5x) ---------------

GEN_BACKGROUND, GEN_BLUE, GEN_RED, GEN_GREEN = 0, 1, 2, 3  # Colors.
GEN_SIZE_MIN, GEN_SIZE_MAX = 10, 18  # Square canvas bounds.
GEN_PATH_KINDS = ("horizontal", "vertical", "diag_down", "diag_up")  # Line layout.
GEN_BLUE_DENSITY_MIN, GEN_BLUE_DENSITY_MAX = 0.35, 0.50  # Total blue density.
GEN_PATH_BLUE_MIN_FRAC, GEN_PATH_BLUE_MAX_FRAC = 0.19, 0.56  # Path blue mix.
GEN_COMPONENT_COUNT_MIN, GEN_COMPONENT_COUNT_MAX = 9, 35  # Blue fragmentation.
GEN_LARGEST_COMPONENT_MAX = 28
GEN_MAX_ATTEMPTS = 300

def _gen_path_cells(n: int) -> list[tuple[int, int]]:
    kind = random.choice(GEN_PATH_KINDS)
    if kind == "horizontal":
        r = random.randrange(n)
        return [(r, c) for c in range(n)]
    if kind == "vertical":
        c = random.randrange(n)
        return [(r, c) for r in range(n)]
    if kind == "diag_down":
        return [(i, i) for i in range(n)]
    return [(i, n - 1 - i) for i in range(n)]

def _gen_add_blue_foreground(grid: np.ndarray, path: list[tuple[int, int]]) -> None:
    n = grid.shape[0]
    path_blue_min = math.ceil(GEN_PATH_BLUE_MIN_FRAC * n)
    path_blue_max = min(n - 2, math.floor(GEN_PATH_BLUE_MAX_FRAC * n))
    path_blue_count = random.randint(path_blue_min, path_blue_max)
    for r, c in random.sample(path[1:-1], path_blue_count):
        grid[r, c] = GEN_BLUE

    total_blue_min = math.ceil(GEN_BLUE_DENSITY_MIN * n * n)
    total_blue_max = math.floor(GEN_BLUE_DENSITY_MAX * n * n)
    total_blue_count = random.randint(max(total_blue_min, path_blue_count), total_blue_max)
    path_set = set(path)
    off_path_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in path_set]
    for r, c in random.sample(off_path_cells, total_blue_count - path_blue_count):
        grid[r, c] = GEN_BLUE

def _gen_expected_output(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    rows, cols = np.array(path, dtype=int).T
    out[rows, cols] = np.where(grid[rows, cols] == GEN_BLUE, GEN_GREEN, GEN_RED)
    return out

def _gen_blue_component_stats(grid: np.ndarray) -> tuple[int, int]:
    from arc_utils import find_components

    comps = find_components(grid, bg=GEN_BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    blue_sizes = [int(comp["area"]) for comp in comps if int(comp["color"]) == GEN_BLUE]
    if not blue_sizes:
        return 0, 0
    return len(blue_sizes), max(blue_sizes)

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    grid = np.zeros((n, n), dtype=int)
    path = _gen_path_cells(n)

    grid[path[0]] = GEN_RED
    grid[path[-1]] = GEN_RED
    _gen_add_blue_foreground(grid, path)

    component_count, largest_component = _gen_blue_component_stats(grid)
    if not (
        GEN_COMPONENT_COUNT_MIN <= component_count <= GEN_COMPONENT_COUNT_MAX
        and largest_component <= GEN_LARGEST_COMPONENT_MAX
    ):
        return None

    expected = _gen_expected_output(grid, path)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bcb3040b input")
