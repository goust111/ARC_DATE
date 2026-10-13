import numpy as np

from arc_utils import Color


def _largest_solid_rectangle(g: np.ndarray) -> tuple[int, int, int, int, int]:
    H, W = g.shape
    best_area = -1
    best: set[tuple[int, int, int, int, int]] = set()

    for color in np.unique(g):
        color = int(color)
        heights = np.zeros(W, dtype=int)
        for r in range(H):
            heights = np.where(g[r] == color, heights + 1, 0)
            stack: list[tuple[int, int]] = []
            for c in range(W + 1):
                cur = int(heights[c]) if c < W else 0
                start = c
                while stack and stack[-1][1] > cur:
                    c0, h = stack.pop()
                    w = c - c0
                    if h >= 3 and w >= 3:
                        area = h * w
                        rect = (color, r - h + 1, c0, r, c - 1)
                        if area > best_area:
                            best_area = area
                            best = {rect}
                        elif area == best_area:
                            best.add(rect)
                    start = c0
                if not stack or stack[-1][1] < cur:
                    stack.append((start, cur))

    assert len(best) == 1
    return next(iter(best))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the center of the largest solid empty rectangle with teal.

    Input structure
    The input is a same-size noisy grid with one uniquely largest solid black
    rectangle; its height and width are at least three cells.
    Rule
    Find that rectangle, preserve its one-cell black border, and recolor only
    the strict interior cells to teal.
    Key insight
    The target is the largest axis-aligned solid rectangle, not the whole
    connected black component around it.
    """
    g = np.asarray(grid, dtype=int)
    color, r0, c0, r1, c1 = _largest_solid_rectangle(g)
    assert color == Color.BLACK
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------
# Canvas, rectangle, and retry sampling bounds.
GRID_SIZE_RANGE, RECT_SIDE_RANGE = (15, 22), (4, 7)
RECT_AREA_FRACTION_RANGE = (0.06, 0.10)
RECT_EDGE_MARGIN = 1
GENERATION_RETRY_LIMIT = 300
# Noise palette, density, and optional secondary-color rates.
GEN_COLORS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
COLORED_FRACTION_RANGE = (0.50, 0.65)
SECONDARY_MIN_COLORED, SECONDARY_PROBABILITY = 10, 0.30
SECONDARY_FRACTION_RANGE = (0.05, 0.10)

def _gen_rect_dims(n: int) -> tuple[int, int]:
    dims = [
        (h, w)
        for h in range(*RECT_SIDE_RANGE)
        for w in range(*RECT_SIDE_RANGE)
        if RECT_AREA_FRACTION_RANGE[0] <= (h * w) / (n * n) <= RECT_AREA_FRACTION_RANGE[1]
    ]
    return dims[int(np.random.randint(len(dims)))]


def _paint_noise(grid: np.ndarray, outside: np.ndarray) -> bool:
    outside_coords = np.argwhere(outside)
    min_colored = int(np.ceil(COLORED_FRACTION_RANGE[0] * grid.size))
    max_colored = int(np.floor(COLORED_FRACTION_RANGE[1] * grid.size))
    colored_count = int(np.random.randint(min_colored, max_colored + 1))
    if colored_count > len(outside_coords):
        return False

    primary = int(np.random.choice(GEN_COLORS))
    chosen_idx = np.random.choice(len(outside_coords), size=colored_count, replace=False)
    chosen = outside_coords[chosen_idx]
    grid[chosen[:, 0], chosen[:, 1]] = primary

    if colored_count >= SECONDARY_MIN_COLORED and np.random.random() < SECONDARY_PROBABILITY:
        secondary_choices = GEN_COLORS[GEN_COLORS != primary]
        secondary = int(np.random.choice(secondary_choices))
        secondary_count = int(round(colored_count * np.random.uniform(*SECONDARY_FRACTION_RANGE)))
        secondary_count = max(1, min(secondary_count, len(chosen)))
        secondary_idx = np.random.choice(len(chosen), size=secondary_count, replace=False)
        secondary_cells = chosen[secondary_idx]
        grid[secondary_cells[:, 0], secondary_cells[:, 1]] = secondary

    return True


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    rect_h, rect_w = _gen_rect_dims(n)
    r0 = int(np.random.randint(RECT_EDGE_MARGIN, n - rect_h - RECT_EDGE_MARGIN + 1))
    c0 = int(np.random.randint(RECT_EDGE_MARGIN, n - rect_w - RECT_EDGE_MARGIN + 1))

    grid = np.zeros((n, n), dtype=int)
    outside = np.ones((n, n), dtype=bool)
    outside[r0 : r0 + rect_h, c0 : c0 + rect_w] = False
    if not _paint_noise(grid, outside):
        return None

    expected = grid.copy()
    expected[r0 + 1 : r0 + rect_h - 1, c0 + 1 : c0 + rect_w - 1] = Color.TEAL

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e88171ec input")
