import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


def _rotate_ring(out: np.ndarray, g: np.ndarray, r0: int, c0: int) -> None:
    patch = g[r0 : r0 + 3, c0 : c0 + 3]

    out[r0 + 0, c0 + 0] = patch[0, 2]
    out[r0 + 0, c0 + 1] = patch[1, 0]
    out[r0 + 0, c0 + 2] = patch[2, 2]
    out[r0 + 1, c0 + 0] = patch[2, 1]
    out[r0 + 1, c0 + 2] = patch[0, 1]
    out[r0 + 2, c0 + 0] = patch[0, 0]
    out[r0 + 2, c0 + 1] = patch[1, 2]
    out[r0 + 2, c0 + 2] = patch[2, 0]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the colors around every 3x3 foreground ring, with corners and side-centers cycling in opposite directions.

    Input structure
    - The grid has a dominant background color.
    - Each 4-connected foreground object is a complete 3x3 ring: eight foreground pixels around a background center.

    Rule
    - For each ring, move corner colors one step around the four corners.
    - Move side-center colors one step around the four side centers in the opposite direction.
    - Leave the background center and all other background pixels unchanged.

    Key insight
    - Corners and side centers form two separate four-cycles on the same 3x3 ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert components

    for comp in components:
        bbox = comp["bbox"]
        r0, c0, r1, c1 = map(int, bbox)
        assert bbox_h(bbox) == 3
        assert bbox_w(bbox) == 3
        assert int(comp["area"]) == 8
        assert g[r0 + 1, c0 + 1] == bg

        ring_coords = {
            (r0 + dr, c0 + dc)
            for dr in range(3)
            for dc in range(3)
            if (dr, dc) != (1, 1)
        }
        component_coords = {tuple(map(int, coord)) for coord in comp["coords"]}
        assert component_coords == ring_coords
        assert r1 == r0 + 2
        assert c1 == c0 + 2

        _rotate_ring(out, g, r0, c0)

    return out

# --------------- generator (att2, o5x) ---------------

# Color and canvas bounds.
BG_COLOR = 7
COLOR_IDS = tuple(range(10))
GRID_SIDE_MIN, GRID_SIDE_MAX = 6, 10
MIN_BACKGROUND_PCT = 67

# Ring geometry, counts, and placement spacing.
RING_SIZE, RING_AREA, RING_GAP = 3, 8, 1
RING_COUNT_MIN, RING_COUNT_MAX = 1, 2
RING_UNIQUE_MIN, RING_UNIQUE_MAX = 2, RING_AREA
RING_CELLS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)

# Retry budget for rejection through the solver oracle.
GENERATION_RETRIES = 300


def _bbox_at(r: int, c: int) -> tuple[int, int, int, int]:
    return r, c, r + RING_SIZE - 1, c + RING_SIZE - 1


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + RING_GAP < b[0]
        or b[2] + RING_GAP < a[0]
        or a[3] + RING_GAP < b[1]
        or b[3] + RING_GAP < a[1]
    )


def _valid_position_sets(rows: int, cols: int, count: int) -> list[tuple[tuple[int, int], ...]]:
    positions = [(r, c) for r in range(rows - RING_SIZE + 1) for c in range(cols - RING_SIZE + 1)]
    if count == 1:
        return [(pos,) for pos in positions]

    pairs = []
    for i, first in enumerate(positions):
        box1 = _bbox_at(*first)
        for second in positions[i + 1 :]:
            if _separated(box1, _bbox_at(*second)):
                pairs.append((first, second))
    return pairs


def _compatible_sizes(count: int) -> list[tuple[int, int]]:
    sizes = []
    for rows in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1):
        for cols in range(GRID_SIDE_MIN, GRID_SIDE_MAX + 1):
            area = rows * cols
            if (area - RING_AREA * count) * 100 < MIN_BACKGROUND_PCT * area:
                continue
            if _valid_position_sets(rows, cols, count):
                sizes.append((rows, cols))
    return sizes


_SIZES = {count: _compatible_sizes(count) for count in range(RING_COUNT_MIN, RING_COUNT_MAX + 1)}
_POSITION_SETS = {
    count: {size: _valid_position_sets(*size, count) for size in _SIZES[count]}
    for count in _SIZES
}


def _make_ring(bg: int) -> np.ndarray:
    ring = np.full((RING_SIZE, RING_SIZE), bg, dtype=int)
    palette = [color for color in COLOR_IDS if color != bg]
    unique_count = random.randint(RING_UNIQUE_MIN, RING_UNIQUE_MAX)
    chosen = random.sample(palette, unique_count)
    values = chosen + [random.choice(chosen) for _ in range(RING_AREA - unique_count)]
    random.shuffle(values)
    for (dr, dc), color in zip(RING_CELLS, values):
        ring[dr, dc] = color
    return ring


def _rotated_output(grid: np.ndarray, positions: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = grid.copy()
    for r0, c0 in positions:
        _rotate_ring(out, grid, r0, c0)
    return out


def _render_grid(rows: int, cols: int, positions: tuple[tuple[int, int], ...], bg: int) -> np.ndarray:
    grid = np.full((rows, cols), bg, dtype=int)
    for r0, c0 in positions:
        paste(grid, _make_ring(bg), r0, c0)
    return grid


def _oracle_accepts(grid: np.ndarray, positions: tuple[tuple[int, int], ...]) -> bool:
    expected = _rotated_output(grid, positions)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    bg = BG_COLOR
    count = random.randint(RING_COUNT_MIN, RING_COUNT_MAX)
    rows, cols = random.choice(_SIZES[count])
    positions = random.choice(_POSITION_SETS[count][(rows, cols)])
    grid = _render_grid(rows, cols, positions, bg)
    return grid if _oracle_accepts(grid, positions) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 46c35fc7 input")
