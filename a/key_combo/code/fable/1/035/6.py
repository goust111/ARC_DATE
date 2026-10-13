from __future__ import annotations

from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


_FIELD_TOP = 2
_PALETTE = np.array(
    [
        Color.BLUE,
        Color.RED,
        Color.GREEN,
        Color.YELLOW,
        Color.PINK,
        Color.ORANGE,
        Color.TEAL,
        Color.MAROON,
    ],
    dtype=int,
)
_GEN_RNG = np.random.default_rng()
_GEN_MAX_ATTEMPTS = 400
_GEN_MODE_NAMES = ("shadow_only", "sealed", "mixed")
_GEN_MIN_SHADOW_PIXELS = 4
_GEN_MIN_SEALED_PIXELS = 4


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(mask.shape, nr, nc):
                        continue
                    if not mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _legend_colors(grid: np.ndarray) -> tuple[int, int]:
    legend_row = np.asarray(grid[0], dtype=int)
    assert not np.any(legend_row == Color.GREY), "legend row must not use grey"
    legend_mask = legend_row != Color.BLACK
    legend_cols = np.flatnonzero(legend_mask)
    assert legend_cols.size == 2, "expected exactly two colored legend cells"
    left_col, right_col = map(int, legend_cols)
    even_color = int(legend_row[left_col])
    odd_color = int(legend_row[right_col])
    assert even_color != odd_color, "legend colors must differ"
    return even_color, odd_color


def _wave_distances(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=int)
    h, w = map(int, field.shape)
    wall_mask = field == Color.GREY
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if wall_mask[r, c] or dist[r, c] >= 0:
            return
        dist[r, c] = 0
        q.append((r, c))

    for c in range(w):
        seed(0, c)
        seed(h - 1, c)
    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(field.shape, nr, nc):
                continue
            if wall_mask[nr, nc] or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def _solve_field(field: np.ndarray, even_color: int, odd_color: int) -> tuple[np.ndarray, np.ndarray]:
    dist = _wave_distances(field)
    out = np.asarray(field, dtype=int).copy()
    reachable = dist >= 0
    even_mask = reachable & ((dist % 2) == 0)
    odd_mask = reachable & ((dist % 2) == 1)
    out[even_mask] = int(even_color)
    out[odd_mask] = int(odd_color)
    out[field == Color.GREY] = Color.GREY
    return out, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the field by even/odd obstacle-aware distance from the field boundary.

    Input structure:
    - The top row is a legend with exactly two colored cells: the left color is for even distances and the right color is for odd distances.
    - The second row is a solid grey divider wall separating the legend from the field.
    - The remaining rows form a black field containing one-cell-thick grey wall segments.

    Rule:
    - Treat every non-wall cell on the field's outer border as a wave source with distance 0.
    - Compute the shortest 4-connected path length from those boundary sources through non-grey cells only.
    - Recolor each reachable field cell with the even legend color for even distances and the odd legend color for odd distances; keep grey walls, the legend, the divider row, and unreachable sealed cells unchanged.

    Key insight:
    - The stripes come from geodesic wavefront parity, not plain rectangular rings, so walls can shift the phase behind them and can leave sealed pockets black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= _FIELD_TOP + 1 and g.shape[1] >= 3, "grid is too small"
    assert np.all(g[1] == Color.GREY), "second row must be the full grey divider"

    even_color, odd_color = _legend_colors(g)
    field = g[_FIELD_TOP:, :]

    solved_field, _dist = _solve_field(field, even_color, odd_color)
    out = g.copy()
    out[_FIELD_TOP:, :] = solved_field
    return out


def _grow8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    padded = np.pad(mask, 1, constant_values=False)
    grown = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            grown |= padded[dr : dr + h, dc : dc + w]
    return grown


def _paint_wall(field: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        field[r, c] = Color.GREY


def _can_place(walls: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = map(int, walls.shape)
    uniq = {(int(r), int(c)) for r, c in cells}
    if not uniq:
        return False
    for r, c in uniq:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False

    new_mask = np.zeros_like(walls, dtype=bool)
    for r, c in uniq:
        new_mask[r, c] = True

    return not np.any(_grow8(walls) & new_mask)


def _random_segment(shape: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    if h < 5 or w < 5:
        return None

    if int(rng.integers(2)) == 0:
        max_len = min(8, w - 2)
        if max_len < 3:
            return None
        length = int(rng.integers(3, max_len + 1))
        r = int(rng.integers(1, h - 1))
        c0 = int(rng.integers(1, w - length))
        return [(r, c) for c in range(c0, c0 + length)]

    max_len = min(8, h - 2)
    if max_len < 3:
        return None
    length = int(rng.integers(3, max_len + 1))
    r0 = int(rng.integers(1, h - length))
    c = int(rng.integers(1, w - 1))
    return [(r, c) for r in range(r0, r0 + length)]


def _random_l_shape(shape: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    max_h = min(8, h - 2)
    max_w = min(8, w - 2)
    if max_h < 3 or max_w < 3:
        return None

    box_h = int(rng.integers(3, max_h + 1))
    box_w = int(rng.integers(3, max_w + 1))
    r0 = int(rng.integers(1, h - box_h))
    c0 = int(rng.integers(1, w - box_w))
    variant = int(rng.integers(4))

    cells: set[tuple[int, int]] = set()
    if variant == 0:
        cells.update((r0, c) for c in range(c0, c0 + box_w))
        cells.update((r, c0) for r in range(r0, r0 + box_h))
    elif variant == 1:
        cells.update((r0, c) for c in range(c0, c0 + box_w))
        cells.update((r, c0 + box_w - 1) for r in range(r0, r0 + box_h))
    elif variant == 2:
        cells.update((r0 + box_h - 1, c) for c in range(c0, c0 + box_w))
        cells.update((r, c0) for r in range(r0, r0 + box_h))
    else:
        cells.update((r0 + box_h - 1, c) for c in range(c0, c0 + box_w))
        cells.update((r, c0 + box_w - 1) for r in range(r0, r0 + box_h))
    return list(cells)


def _random_frame(
    shape: tuple[int, int], rng: np.random.Generator, *, closed: bool
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    max_h = min(9, h - 2)
    max_w = min(9, w - 2)
    if max_h < 4 or max_w < 4:
        return None

    box_h = int(rng.integers(4, max_h + 1))
    box_w = int(rng.integers(4, max_w + 1))
    r0 = int(rng.integers(1, h - box_h))
    c0 = int(rng.integers(1, w - box_w))
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    cells: set[tuple[int, int]] = set()
    cells.update((r0, c) for c in range(c0, c1 + 1))
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))

    if not closed:
        side = int(rng.integers(4))
        if side == 0:
            cells.remove((r0, int(rng.integers(c0 + 1, c1))))
        elif side == 1:
            cells.remove((r1, int(rng.integers(c0 + 1, c1))))
        elif side == 2:
            cells.remove((int(rng.integers(r0 + 1, r1)), c0))
        else:
            cells.remove((int(rng.integers(r0 + 1, r1)), c1))

    return list(cells)


def _place_random_primitive(
    field: np.ndarray,
    rng: np.random.Generator,
    builders,
    *,
    attempts: int,
) -> bool:
    for _ in range(attempts):
        builder = builders[int(rng.integers(len(builders)))]
        cells = builder(field.shape, rng)
        if cells is None:
            continue
        if not _can_place(field == Color.GREY, cells):
            continue
        _paint_wall(field, cells)
        return True
    return False


def _assemble_grid(field: np.ndarray, even_color: int, odd_color: int) -> np.ndarray:
    h, w = map(int, field.shape)
    grid = np.full((h + _FIELD_TOP, w), Color.BLACK, dtype=int)
    grid[0, 0] = int(even_color)
    grid[0, w - 1] = int(odd_color)
    grid[1, :] = Color.GREY
    grid[_FIELD_TOP:, :] = field
    return grid


def _field_features(field: np.ndarray) -> tuple[int, int, int]:
    dist = _wave_distances(field)
    wall_mask = field == Color.GREY
    h, w = map(int, field.shape)
    rows = np.arange(h, dtype=int)[:, None]
    cols = np.arange(w, dtype=int)[None, :]
    naive = np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols))

    shadow_mask = (dist >= 0) & (~wall_mask) & (((dist - naive) % 2) == 1)
    unreachable_mask = (~wall_mask) & (dist < 0)
    sealed_components = _components_mask(unreachable_mask)
    max_sealed = max((len(comp) for comp in sealed_components), default=0)
    return int(shadow_mask.sum()), int(unreachable_mask.sum()), int(max_sealed)


def _gen_attempt(mode: str) -> np.ndarray | None:
    field_h = int(_GEN_RNG.integers(10, 23))
    field_w = int(_GEN_RNG.integers(10, 23))
    field = np.full((field_h, field_w), Color.BLACK, dtype=int)
    even_color, odd_color = map(int, _GEN_RNG.choice(_PALETTE, size=2, replace=False))

    closed_builder = lambda shape, rng: _random_frame(shape, rng, closed=True)
    open_builder = lambda shape, rng: _random_frame(shape, rng, closed=False)
    extra_builders = [open_builder, _random_segment, _random_l_shape]

    if mode == "shadow_only":
        if not _place_random_primitive(field, _GEN_RNG, [open_builder], attempts=80):
            return None
        extra_count = int(_GEN_RNG.integers(1, 4))
    elif mode == "sealed":
        if not _place_random_primitive(field, _GEN_RNG, [closed_builder], attempts=80):
            return None
        extra_count = int(_GEN_RNG.integers(0, 3))
    else:
        if not _place_random_primitive(field, _GEN_RNG, [closed_builder], attempts=80):
            return None
        if not _place_random_primitive(field, _GEN_RNG, [open_builder], attempts=80):
            return None
        extra_count = int(_GEN_RNG.integers(1, 4))

    for _ in range(extra_count):
        _place_random_primitive(field, _GEN_RNG, extra_builders, attempts=40)

    wall_pixels = int(np.count_nonzero(field == Color.GREY))
    if wall_pixels < 8 or wall_pixels > field.size // 3:
        return None

    shadow_pixels, unreachable_pixels, max_sealed = _field_features(field)
    if mode == "shadow_only":
        if shadow_pixels < _GEN_MIN_SHADOW_PIXELS or unreachable_pixels != 0:
            return None
    elif mode == "sealed":
        if max_sealed < _GEN_MIN_SEALED_PIXELS:
            return None
    else:
        if shadow_pixels < _GEN_MIN_SHADOW_PIXELS or max_sealed < _GEN_MIN_SEALED_PIXELS:
            return None

    grid = _assemble_grid(field, even_color, odd_color)
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        mode = _GEN_MODE_NAMES[int(_GEN_RNG.integers(len(_GEN_MODE_NAMES)))]
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid onion-wave input")
