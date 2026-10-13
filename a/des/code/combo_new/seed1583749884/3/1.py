import random
from collections import deque

import numpy as np


_BG = 0
_OBJECT_COLOR = 5
_LOOKUP_SIZE_OPTIONS = (2, 3)
_GRID_SIDE_MAX = 30
_MAX_OBJECTS = 5
_MAX_GENERATION_ATTEMPTS = 400
_COLOR_POOL = tuple(range(1, 10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_lookup_size(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape

    row_len = 0
    while row_len < w and int(grid[0, row_len]) != bg:
        row_len += 1

    col_len = 0
    while col_len < h and int(grid[col_len, 0]) != bg:
        col_len += 1

    assert row_len == col_len and row_len > 0, "lookup table must be a square anchored at the top-left corner"
    n = row_len
    block = grid[:n, :n]
    assert np.all(block != bg), "lookup table must be fully filled"
    assert len(set(map(int, block.ravel()))) == n * n, "lookup table cells must be distinct"
    if n < h:
        assert np.all(grid[n, :n] == bg), "lookup table must end cleanly below"
    if n < w:
        assert np.all(grid[:n, n] == bg), "lookup table must end cleanly to the right"
    return n


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome square object by copying colors from the top-left lookup table at matching local coordinates.

    Input structure:
    - The background is black.
    - The top-left corner contains an N×N lookup table made of non-background colors, with one color per cell.
    - Elsewhere on the grid there are one or more separate 4-connected monochrome N×N square objects on the background.

    Rule:
    - Detect the lookup table size N from the contiguous non-background square anchored at the top-left corner.
    - For each other foreground component, verify that it is a solid N×N square.
    - Replace each pixel of that square by the lookup-table color at the same local (row, column) position.
    - Leave the lookup table itself unchanged.

    Key insight:
    - The grey objects are not recolored by their absolute position or color id; each one reuses the lookup table as a fixed local-coordinate stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _most_common_color(g)
    n = _infer_lookup_size(g, bg)
    palette = g[:n, :n].copy()

    mask = g != bg
    mask[:n, :n] = False
    components = _components4(mask)
    assert components, "expected at least one recolorable object"

    out = g.copy()
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 == n and c1 - c0 + 1 == n, "each object must match the lookup-table size"
        assert len(coords) == n * n, "each object must be a solid square"
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(window != bg), "object interiors must be filled"
        assert np.unique(window).size == 1, "each object must be monochrome"
        out[r0 : r1 + 1, c0 : c1 + 1] = palette

    return out


def _sample_lookup_table(n: int) -> np.ndarray:
    colors = random.sample(_COLOR_POOL, n * n)
    return np.array(colors, dtype=int).reshape(n, n)


def _max_object_count(n: int, h: int, w: int) -> int:
    usable_h = h - (n + 1)
    usable_w = w - (n + 1)
    if usable_h < n or usable_w < n:
        return 0
    row_slots = usable_h // (n + 1)
    col_slots = usable_w // (n + 1)
    return max(1, row_slots * col_slots)


def _far_enough(a: tuple[int, int], b: tuple[int, int], n: int) -> bool:
    ar, ac = a
    br, bc = b
    return not (abs(ar - br) <= n and abs(ac - bc) <= n)


def _place_objects(n: int, h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    row_candidates = list(range(n + 1, h - n + 1))
    col_candidates = list(range(n + 1, w - n + 1))
    if not row_candidates or not col_candidates:
        return None

    positions: list[tuple[int, int]] = []
    for _ in range(500):
        r = random.choice(row_candidates)
        c = random.choice(col_candidates)
        if all(_far_enough((r, c), other, n) for other in positions):
            positions.append((r, c))
            if len(positions) == count:
                return positions
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_LOOKUP_SIZE_OPTIONS)
    min_side = 2 * n + 3
    h = random.randint(min_side, _GRID_SIDE_MAX)
    w = random.randint(min_side, _GRID_SIDE_MAX)

    max_count = min(_MAX_OBJECTS, _max_object_count(n, h, w))
    if max_count <= 0:
        return None
    count = random.randint(1, max_count)
    positions = _place_objects(n, h, w, count)
    if positions is None:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    lookup = _sample_lookup_table(n)
    grid[:n, :n] = lookup
    for r, c in positions:
        grid[r : r + n, c : c + n] = _OBJECT_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
