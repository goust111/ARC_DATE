import random

import numpy as np


FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
RING_OFFSETS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
CORNER_OFFSETS = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_OFFSETS = ((0, 1), (1, 2), (2, 1), (1, 0))

# --------------- solver ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely dominant"
    return int(values[order[0]])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in FOUR_NEIGHBORS:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(coords)

    return components


def _rotate_cycle(out: np.ndarray, src: np.ndarray, offsets: tuple[tuple[int, int], ...], direction: int) -> None:
    # Offsets are ordered clockwise.
    # direction=+1 means colors move one step clockwise.
    # direction=-1 means colors move one step counterclockwise.
    size = len(offsets)
    for i, (r, c) in enumerate(offsets):
        sr, sc = offsets[(i - direction) % size]
        out[r, c] = src[sr, sc]


def _rotate_ring_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    _rotate_cycle(out, patch, CORNER_OFFSETS, direction=1)
    _rotate_cycle(out, patch, SIDE_OFFSETS, direction=-1)
    out[1, 1] = patch[1, 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each 3x3 foreground ring by cycling corners clockwise and side-centers counterclockwise."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    components = _find_components(g != bg)
    assert components, "expected at least one foreground ring"

    out = g.copy()
    for coords in components:
        assert len(coords) == 8, "each object must have eight foreground cells"
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 2 and c1 - c0 == 2, "each object must fit a 3x3 box"
        assert int(g[r0 + 1, c0 + 1]) == bg, "ring center must be background"

        expected = {(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS}
        assert set(coords) == expected, "foreground component must be a complete 3x3 ring"

        patch = g[r0 : r0 + 3, c0 : c0 + 3]
        out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_ring_patch(patch)

    return out


# --------------- generator ---------------


COLORS = tuple(range(10))
GRID_HEIGHT_RANGE = (7, 14)
GRID_WIDTH_RANGE = (7, 14)
MAX_RING_COUNT = 5
MIN_RING_GAP = 1
GENERATION_RETRIES = 500


def _bbox_at(r: int, c: int) -> tuple[int, int, int, int]:
    return r, c, r + 2, c + 2


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_RING_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _max_ring_count(rows: int, cols: int) -> int:
    area_bound = max(1, (rows * cols - 1) // 16)
    packing_bound = max(1, ((rows + 1) // 4) * ((cols + 1) // 4))
    return min(MAX_RING_COUNT, area_bound, packing_bound)


def _sample_positions(rows: int, cols: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(rows - 2) for c in range(cols - 2)]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        box = _bbox_at(r, c)
        if all(_separated(box, other) for other in boxes):
            chosen.append((r, c))
            boxes.append(box)
            if len(chosen) == count:
                return chosen

    return None


def _make_ring(bg: int) -> np.ndarray:
    palette = [color for color in COLORS if color != bg]
    for _ in range(100):
        ring = np.full((3, 3), bg, dtype=int)
        unique_count = random.randint(2, len(RING_OFFSETS))
        chosen = random.sample(palette, unique_count)
        values = chosen + [random.choice(chosen) for _ in range(len(RING_OFFSETS) - unique_count)]
        random.shuffle(values)
        for (dr, dc), color in zip(RING_OFFSETS, values):
            ring[dr, dc] = color
        if not np.array_equal(ring, _rotate_ring_patch(ring)):
            return ring
    raise RuntimeError("failed to sample a non-degenerate ring")


def _expected_output(grid: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0 in positions:
        out[r0 : r0 + 3, c0 : c0 + 3] = _rotate_ring_patch(grid[r0 : r0 + 3, c0 : c0 + 3])
    return out


def _render_grid(rows: int, cols: int, bg: int, positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((rows, cols), bg, dtype=int)
    for r0, c0 in positions:
        grid[r0 : r0 + 3, c0 : c0 + 3] = _make_ring(bg)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_HEIGHT_RANGE)
    cols = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(COLORS)

    max_count = _max_ring_count(rows, cols)
    count = random.randint(1, max_count)
    positions = _sample_positions(rows, cols, count)
    if positions is None:
        return None

    grid = _render_grid(rows, cols, bg, positions)
    expected = _expected_output(grid, positions)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
