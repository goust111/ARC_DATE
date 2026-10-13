import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SEPARATOR_COLORS = (BLUE, RED, GREEN)

MAX_GRID_SIDE = 30
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 7
MIN_REGIONS = 2
MAX_REGIONS = 4
MIN_PATTERN_PIXELS = 3
MAX_PATTERN_PIXELS = 7
GENERATION_ATTEMPTS = 400


def _dominant_separator_color(values: np.ndarray) -> int:
    counts = {color: int(np.count_nonzero(values == color)) for color in SEPARATOR_COLORS}
    color, count = max(counts.items(), key=lambda item: item[1])
    assert count > 0, "separator line must contain at least one separator color"
    return color


def _is_separator_line(values: np.ndarray) -> bool:
    return values.size > 0 and bool(np.all(np.isin(values, SEPARATOR_COLORS)))


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave non-empty regions"
        bands.append((start, sep))
        start = sep + 1
    assert start < length, "separator lines cannot consume the grid border"
    bands.append((start, length))
    return bands


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    coord_set = set(coords)
    seen = {coords[0]}
    queue: deque[tuple[int, int]] = deque([coords[0]])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coord_set)


def _transform_coords(
    coords: list[tuple[int, int]],
    height: int,
    width: int,
    *,
    flip_ud: bool,
    flip_lr: bool,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        nr = height - 1 - r if flip_ud else r
        nc = width - 1 - c if flip_lr else c
        out.append((nr, nc))
    out.sort()
    return out


def _parity_flags(colors: list[int]) -> tuple[bool, bool]:
    flip_ud = False
    flip_lr = False
    for color in colors:
        if color == RED:
            flip_ud = not flip_ud
        elif color == BLUE:
            flip_lr = not flip_lr
        elif color == GREEN:
            flip_ud = not flip_ud
            flip_lr = not flip_lr
        else:
            raise AssertionError("unexpected separator color")
    return flip_ud, flip_lr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the grey pattern to every rectangular region separated by colored
    full-row/full-column divider lines. Crossing red flips vertically, crossing
    blue flips horizontally, and crossing green rotates 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height > 0 and width > 0

    allowed = {BLACK, GREY, *SEPARATOR_COLORS}
    extra = set(map(int, np.unique(g))) - allowed
    assert not extra, "grid contains unexpected colors"

    separator_rows = [r for r in range(height) if _is_separator_line(g[r, :])]
    separator_cols = [c for c in range(width) if _is_separator_line(g[:, c])]
    row_bands = _bands(height, separator_rows)
    col_bands = _bands(width, separator_cols)

    row_separator_colors = [_dominant_separator_color(g[r, :]) for r in separator_rows]
    col_separator_colors = [_dominant_separator_color(g[:, c]) for c in separator_cols]

    source_region: tuple[int, int] | None = None
    source_local_coords: list[tuple[int, int]] = []
    source_region_shape: tuple[int, int] | None = None

    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            assert not np.any(np.isin(region, SEPARATOR_COLORS)), "separator colors cannot appear inside regions"
            grey_mask = region == GREY
            if not np.any(grey_mask):
                continue
            assert source_region is None, "exactly one region may contain the source pattern"
            coords = [tuple(map(int, rc)) for rc in np.argwhere(grey_mask)]
            assert _is_connected(coords), "source pattern must be 4-connected"
            source_region = (rr, cc)
            source_local_coords = sorted(coords)
            source_region_shape = region.shape

    assert source_region is not None, "one source region is required"
    assert source_region_shape is not None
    source_h, source_w = map(int, source_region_shape)

    out = np.full_like(g, BLACK)
    separator_mask = np.isin(g, SEPARATOR_COLORS)
    out[separator_mask] = g[separator_mask]

    src_rr, src_cc = source_region
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            crossed = []
            row_lo, row_hi = sorted((src_rr, rr))
            col_lo, col_hi = sorted((src_cc, cc))
            crossed.extend(row_separator_colors[row_lo:row_hi])
            crossed.extend(col_separator_colors[col_lo:col_hi])
            flip_ud, flip_lr = _parity_flags(crossed)
            coords = _transform_coords(
                source_local_coords,
                source_h,
                source_w,
                flip_ud=flip_ud,
                flip_lr=flip_lr,
            )
            region_h = r1 - r0
            region_w = c1 - c0
            for pr, pc in coords:
                assert pr < region_h and pc < region_w, "target region is too small for the propagated pattern"
                out[r0 + pr, c0 + pc] = GREY

    return out


def _random_connected_shape(max_h: int, max_w: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        box_h = random.randint(2, min(4, max_h))
        box_w = random.randint(2, min(4, max_w))
        max_area = min(MAX_PATTERN_PIXELS, box_h * box_w)
        if max_area < MIN_PATTERN_PIXELS:
            continue
        target = random.randint(MIN_PATTERN_PIXELS, max_area)
        coords = {(random.randrange(box_h), random.randrange(box_w))}
        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(tuple(frontier)))
        if len(coords) != target:
            continue
        arr = np.array(sorted(coords), dtype=int)
        r0, c0 = arr.min(axis=0)
        arr[:, 0] -= r0
        arr[:, 1] -= c0
        tight_h = int(arr[:, 0].max()) + 1
        tight_w = int(arr[:, 1].max()) + 1
        if tight_h < 2 or tight_w < 2:
            continue
        bitmap = np.zeros((tight_h, tight_w), dtype=bool)
        bitmap[arr[:, 0], arr[:, 1]] = True
        if (
            np.array_equal(bitmap, np.flipud(bitmap))
            or np.array_equal(bitmap, np.fliplr(bitmap))
            or np.array_equal(bitmap, np.flipud(np.fliplr(bitmap)))
        ):
            continue
        return [tuple(map(int, rc)) for rc in arr.tolist()]
    return None


def _cell_bands(count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int]]:
    bands = []
    separators = []
    start = 0
    for idx in range(count):
        bands.append((start, start + cell_size))
        start += cell_size
        if idx + 1 < count:
            separators.append(start)
            start += 1
    return bands, separators


def _render_grid(
    row_count: int,
    col_count: int,
    cell_h: int,
    cell_w: int,
    row_separator_colors: list[int],
    col_separator_colors: list[int],
    source_region: tuple[int, int],
    source_coords: list[tuple[int, int]],
    *,
    vertical_overrides: bool,
) -> np.ndarray:
    grid_h = row_count * cell_h + (row_count - 1)
    grid_w = col_count * cell_w + (col_count - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    row_bands, row_separator_positions = _cell_bands(row_count, cell_h)
    col_bands, col_separator_positions = _cell_bands(col_count, cell_w)

    def paint_rows() -> None:
        for pos, color in zip(row_separator_positions, row_separator_colors):
            grid[pos, :] = color

    def paint_cols() -> None:
        for pos, color in zip(col_separator_positions, col_separator_colors):
            grid[:, pos] = color

    if vertical_overrides:
        paint_rows()
        paint_cols()
    else:
        paint_cols()
        paint_rows()

    sr, sc = source_region
    base_r = row_bands[sr][0]
    base_c = col_bands[sc][0]
    for r, c in source_coords:
        grid[base_r + r, base_c + c] = GREY
    return grid


def _sample_layout() -> tuple[int, int, int, int] | None:
    if random.random() < 0.25:
        if random.random() < 0.5:
            row_count = 1
            col_count = random.randint(MIN_REGIONS, MAX_REGIONS)
        else:
            row_count = random.randint(MIN_REGIONS, MAX_REGIONS)
            col_count = 1
    else:
        row_count = random.randint(MIN_REGIONS, MAX_REGIONS)
        col_count = random.randint(MIN_REGIONS, MAX_REGIONS)
    cell_h = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    cell_w = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    grid_h = row_count * cell_h + (row_count - 1)
    grid_w = col_count * cell_w + (col_count - 1)
    if grid_h > MAX_GRID_SIDE or grid_w > MAX_GRID_SIDE:
        return None
    return row_count, col_count, cell_h, cell_w


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    row_count, col_count, cell_h, cell_w = layout

    shape = _random_connected_shape(cell_h, cell_w)
    if shape is None:
        return None
    shape_arr = np.array(shape, dtype=int)
    shape_h = int(shape_arr[:, 0].max()) + 1
    shape_w = int(shape_arr[:, 1].max()) + 1

    top = random.randint(0, cell_h - shape_h)
    left = random.randint(0, cell_w - shape_w)
    source_coords = sorted((top + r, left + c) for r, c in shape)

    variants = {
        frozenset(_transform_coords(source_coords, cell_h, cell_w, flip_ud=flip_ud, flip_lr=flip_lr))
        for flip_ud in (False, True)
        for flip_lr in (False, True)
    }
    if len(variants) < 4:
        return None

    source_region = (random.randrange(row_count), random.randrange(col_count))
    row_separator_colors = [random.choice(SEPARATOR_COLORS) for _ in range(row_count - 1)]
    col_separator_colors = [random.choice(SEPARATOR_COLORS) for _ in range(col_count - 1)]
    vertical_overrides = random.random() < 0.5

    grid = _render_grid(
        row_count,
        col_count,
        cell_h,
        cell_w,
        row_separator_colors,
        col_separator_colors,
        source_region,
        source_coords,
        vertical_overrides=vertical_overrides,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_grey = row_count * col_count * len(source_coords)
    if int(np.count_nonzero(out == GREY)) != expected_grey:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
