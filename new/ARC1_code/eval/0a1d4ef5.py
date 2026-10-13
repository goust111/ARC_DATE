from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste as _gen_paste


def _overlap_groups(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    group_items: list[list[int]] = []
    common_los: list[int] = []
    common_his: list[int] = []
    center_sums: list[float] = []
    counts: list[int] = []

    for idx, lo, hi in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        matches = [
            group_idx
            for group_idx in range(len(group_items))
            if max(common_los[group_idx], lo) <= min(common_his[group_idx], hi)
        ]
        assert len(matches) <= 1, "expected each interval to belong to at most one band"
        if matches:
            group_idx = matches[0]
            group_items[group_idx].append(idx)
            common_los[group_idx] = max(common_los[group_idx], lo)
            common_his[group_idx] = min(common_his[group_idx], hi)
            center_sums[group_idx] += (lo + hi) / 2
            counts[group_idx] += 1
        else:
            group_items.append([idx])
            common_los.append(lo)
            common_his.append(hi)
            center_sums.append((lo + hi) / 2)
            counts.append(1)

    assert group_items, "expected at least one interval group"
    ordered = sorted(
        range(len(group_items)),
        key=lambda group_idx: center_sums[group_idx] / counts[group_idx],
    )
    index_to_rank: dict[int, int] = {}
    for rank, group_idx in enumerate(ordered):
        for idx in group_items[group_idx]:
            index_to_rank[idx] = rank
    return index_to_rank, len(group_items)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the lattice of filled rectangular objects into a compact color grid.

    Input structure
    - The input has a dominant background with scattered clutter colors.
    - The meaningful objects are monochromatic 4-connected foreground rectangles, each at least 3x3 and exactly filling its bbox.
    - These rectangles form a complete lattice whose same-row and same-column bbox intervals overlap.

    Rule
    - Keep only the filled rectangular objects.
    - Group their vertical bbox intervals into output rows and their horizontal bbox intervals into output columns.
    - Fill each compact output cell with the color of the rectangle at the corresponding lattice position.

    Key insight
    - The clutter is not a filled 3x3-or-larger rectangle; the rectangles' overlapping bbox bands define the small table axes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = get_bg(g)
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for component in find_components(g, bg=bg, connectivity=4, monochromatic=True, min_area=9):
        box = component["bbox"]
        height = bbox_h(box)
        width = bbox_w(box)
        area = int(component["area"])
        if height >= 3 and width >= 3 and area == height * width:
            rectangles.append((int(component["color"]), box))

    assert rectangles, "expected filled rectangular objects"

    row_by_rect, row_count = _overlap_groups(
        [(idx, box[0], box[2]) for idx, (_color, box) in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, box[1], box[3]) for idx, (_color, box) in enumerate(rectangles)]
    )
    assert len(rectangles) == row_count * col_count, "expected a complete rectangle lattice"

    out = np.full((row_count, col_count), -1, dtype=int)
    for idx, (color, _box) in enumerate(rectangles):
        row = row_by_rect[idx]
        col = col_by_rect[idx]
        assert out[row, col] == -1, "expected one rectangle per lattice cell"
        out[row, col] = color

    assert np.all(out >= 0), "expected every lattice cell to be occupied"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_N = 30
_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))

# Lattice geometry.
_GEN_COLS = 3
_GEN_ROW_COUNT_MIN = 2
_GEN_ROW_COUNT_MAX = 4
_GEN_RECT_SIZE_MIN = 3
_GEN_RECT_SIZE_MAX = 8
_GEN_MIN_GAP = 2

# Rectangle and clutter color assignment.
_GEN_RECT_COLOR_MIN = 5
_GEN_RECT_COLOR_MAX = 6
_GEN_CLUTTER_COLOR_COUNT = 2

# Clutter component volume and shape bounds.
_GEN_CLUTTER_TOTAL_MIN = 260
_GEN_CLUTTER_TOTAL_MAX = 370
_GEN_SMALL_COMPONENT_COUNT_MIN = 170
_GEN_SMALL_COMPONENT_COUNT_MAX = 200
_GEN_SMALL_COMPONENT_AREA_MAX = 8
_GEN_SMALL_EXTRA_PIXELS_MAX = 160
_GEN_LARGE_BLOBS_PER_COLOR_MIN = 1
_GEN_LARGE_BLOBS_PER_COLOR_MAX = 3
_GEN_LARGE_AREA_MIN = 9
_GEN_LARGE_AREA_MAX = 14
_GEN_LARGE_BOX_MIN = 3
_GEN_LARGE_BOX_MAX = 6

# Connectivity, orientation, and retry controls.
_GEN_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_GEN_FLIP_VARIANTS = 2
_GEN_ROTATION_VARIANTS = 4
_GEN_RETRY_LIMIT = 300
_GEN_CLUTTER_VALUES = tuple(range(_GEN_CLUTTER_TOTAL_MIN, _GEN_CLUTTER_TOTAL_MAX + 1))
_GEN_CLUTTER_SEQUENCE: list[int] = []
_GEN_CLUTTER_POS = 0

def _gen_band_extents(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    """Sample disjoint axis bands with enough span for rectangle intervals."""
    available = _GEN_N - _GEN_MIN_GAP * (count - 1)
    lengths: list[int] = []
    used = 0
    for idx in range(count):
        rem = count - idx - 1
        hi = min(_GEN_RECT_SIZE_MAX, available - used - _GEN_RECT_SIZE_MIN * rem)
        length = int(rng.integers(_GEN_RECT_SIZE_MIN, hi + 1))
        lengths.append(length)
        used += length

    slack = available - sum(lengths)
    extras = rng.multinomial(slack, [1 / (count + 1)] * (count + 1)).tolist()
    pos = extras[0]
    extents: list[tuple[int, int]] = []
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        extents.append((start, end))
        pos = end + 1
        if idx < count - 1:
            pos += _GEN_MIN_GAP + extras[idx + 1]
    return extents

def _gen_intervals_in_extent(
    rng: np.random.Generator, extent: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    """Sample intervals inside one band that share a common coordinate."""
    start, end = extent
    span = end - start + 1
    common = int(rng.integers(start, end + 1))
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(_GEN_RECT_SIZE_MIN, min(_GEN_RECT_SIZE_MAX, span) + 1))
        lo_min = max(start, common - length + 1)
        lo_max = min(common, end - length + 1)
        lo = int(rng.integers(lo_min, lo_max + 1))
        intervals.append((lo, lo + length - 1))
    return intervals

def _gen_color_grid(
    rng: np.random.Generator, rows: int
) -> tuple[np.ndarray, list[int], list[int]]:
    colors = list(_GEN_COLORS)
    rng.shuffle(colors)
    cell_count = rows * _GEN_COLS
    rect_color_count = int(rng.integers(_GEN_RECT_COLOR_MIN, min(_GEN_RECT_COLOR_MAX, cell_count) + 1))
    rect_colors = colors[:rect_color_count]
    clutter_colors = colors[rect_color_count : rect_color_count + _GEN_CLUTTER_COLOR_COUNT]

    cells = rect_colors + rng.choice(rect_colors, cell_count - rect_color_count).tolist()
    rng.shuffle(cells)
    return np.array(cells, dtype=int).reshape(rows, _GEN_COLS), rect_colors, clutter_colors

def _gen_add_frontier_cell(
    rng: np.random.Generator,
    coords: set[tuple[int, int]],
    height: int | None = None,
    width: int | None = None,
) -> None:
    frontier = set()
    for r, c in coords:
        for dr, dc in _GEN_DIRS:
            rr, cc = r + dr, c + dc
            if (rr, cc) in coords:
                continue
            if height is not None and not (0 <= rr < height):
                continue
            if width is not None and not (0 <= cc < width):
                continue
            frontier.add((rr, cc))
    choices = sorted(frontier)
    coords.add(choices[int(rng.integers(len(choices)))])

def _gen_polyomino(rng: np.random.Generator, area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        _gen_add_frontier_cell(rng, coords)
    return _gen_coords_to_mask(coords)

def _gen_large_blob(rng: np.random.Generator, area: int) -> np.ndarray:
    options = [
        (height, width)
        for height in range(_GEN_LARGE_BOX_MIN, _GEN_LARGE_BOX_MAX + 1)
        for width in range(_GEN_LARGE_BOX_MIN, _GEN_LARGE_BOX_MAX + 1)
        if height + width - 1 <= area < height * width
    ]
    height, width = options[int(rng.integers(len(options)))]
    coords = {(r, 0) for r in range(height)}
    coords.update((0, c) for c in range(width))
    while len(coords) < area:
        _gen_add_frontier_cell(rng, coords, height, width)

    blob = _gen_coords_to_mask(coords)
    if bool(rng.integers(_GEN_FLIP_VARIANTS)):
        blob = np.flipud(blob)
    if bool(rng.integers(_GEN_FLIP_VARIANTS)):
        blob = np.fliplr(blob)
    return np.rot90(blob, int(rng.integers(_GEN_ROTATION_VARIANTS)))

def _gen_coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(list(coords), dtype=int)
    shifted = arr - arr.min(axis=0)
    height, width = shifted.max(axis=0) + 1
    mask = np.zeros((height, width), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask

def _gen_area_list(
    rng: np.random.Generator, count: int, total: int
) -> list[int] | None:
    if total < count or total > count * _GEN_SMALL_COMPONENT_AREA_MAX:
        return None
    areas = [1] * count
    extra = total - count
    while extra:
        candidates = [idx for idx, area in enumerate(areas) if area < _GEN_SMALL_COMPONENT_AREA_MAX]
        if not candidates:
            return None
        idx = candidates[int(rng.integers(len(candidates)))]
        areas[idx] += 1
        extra -= 1
    rng.shuffle(areas)
    return areas

def _gen_next_clutter_target(rng: np.random.Generator) -> int:
    global _GEN_CLUTTER_POS, _GEN_CLUTTER_SEQUENCE
    if _GEN_CLUTTER_POS >= len(_GEN_CLUTTER_SEQUENCE):
        _GEN_CLUTTER_SEQUENCE = rng.permutation(_GEN_CLUTTER_VALUES).astype(int).tolist()
        _GEN_CLUTTER_POS = 0
    target = _GEN_CLUTTER_SEQUENCE[_GEN_CLUTTER_POS]
    _GEN_CLUTTER_POS += 1
    return int(target)

def _gen_place_shape(
    rng: np.random.Generator, grid: np.ndarray, shape: np.ndarray, color: int
) -> bool:
    origins = _gen_valid_origins(grid, shape, color)
    if len(origins) == 0:
        return False
    r0, c0 = origins[int(rng.integers(len(origins)))]
    patch = np.where(shape, color, _GEN_BG)
    _gen_paste(grid, patch, r0, c0, transparent=_GEN_BG)
    return True

def _gen_valid_origins(grid: np.ndarray, shape: np.ndarray, color: int) -> np.ndarray:
    """Return paste origins that do not overlap occupied cells or touch same color."""
    coords = np.argwhere(shape)
    height, width = shape.shape
    same = grid == color
    touches_same = np.zeros_like(same, dtype=bool)
    touches_same[1:, :] |= same[:-1, :]
    touches_same[:-1, :] |= same[1:, :]
    touches_same[:, 1:] |= same[:, :-1]
    touches_same[:, :-1] |= same[:, 1:]
    allowed = (grid == _GEN_BG) & ~touches_same

    origin_h = _GEN_N - height + 1
    origin_w = _GEN_N - width + 1
    valid = np.ones((origin_h, origin_w), dtype=bool)
    for dr, dc in coords:
        valid &= allowed[int(dr) : int(dr) + origin_h, int(dc) : int(dc) + origin_w]
    return np.argwhere(valid)

def _gen_clutter_items(
    rng: np.random.Generator, clutter_colors: list[int]
) -> list[tuple[int, np.ndarray]] | None:
    """Build clutter masks in the same order they will be placement-attempted."""
    items: list[tuple[int, np.ndarray]] = []
    for color in clutter_colors:
        count = int(rng.integers(_GEN_LARGE_BLOBS_PER_COLOR_MIN, _GEN_LARGE_BLOBS_PER_COLOR_MAX + 1))
        for _ in range(count):
            area = int(rng.integers(_GEN_LARGE_AREA_MIN, _GEN_LARGE_AREA_MAX + 1))
            items.append((color, _gen_large_blob(rng, area)))

    large_total = sum(int(shape.sum()) for _color, shape in items)
    small_total = _gen_next_clutter_target(rng) - large_total
    min_count = max(_GEN_SMALL_COMPONENT_COUNT_MIN, min(_GEN_SMALL_COMPONENT_COUNT_MAX, small_total - _GEN_SMALL_EXTRA_PIXELS_MAX))
    max_count = min(_GEN_SMALL_COMPONENT_COUNT_MAX, small_total)
    if min_count > max_count:
        return None
    small_count = int(rng.integers(min_count, max_count + 1))
    areas = _gen_area_list(rng, small_count, small_total)
    if areas is None:
        return None
    for area in areas:
        color = int(clutter_colors[int(rng.integers(len(clutter_colors)))])
        items.append((color, _gen_polyomino(rng, area)))

    rng.shuffle(items)
    items.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return items

def _gen_add_clutter(
    rng: np.random.Generator, grid: np.ndarray, clutter_colors: list[int]
) -> bool:
    items = _gen_clutter_items(rng, clutter_colors)
    if items is None:
        return False
    for color, shape in items:
        if not _gen_place_shape(rng, grid, shape, color):
            return False
    return True

def _gen_lattice_intervals(
    rng: np.random.Generator, rows: int
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """Sample row and column intervals for every rectangle lattice cell."""
    row_extents = _gen_band_extents(rng, rows)
    col_extents = _gen_band_extents(rng, _GEN_COLS)
    row_intervals = [_gen_intervals_in_extent(rng, extent, _GEN_COLS) for extent in row_extents]
    col_intervals = [_gen_intervals_in_extent(rng, extent, rows) for extent in col_extents]
    return row_intervals, col_intervals

def _gen_render_rectangles(
    grid: np.ndarray,
    expected: np.ndarray,
    row_intervals: list[list[tuple[int, int]]],
    col_intervals: list[list[tuple[int, int]]],
) -> None:
    rows = expected.shape[0]
    for rr in range(rows):
        for cc in range(_GEN_COLS):
            r0, r1 = row_intervals[rr][cc]
            c0, c1 = col_intervals[cc][rr]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(expected[rr, cc])

def _gen_background_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_by_color = dict(zip(values.tolist(), counts.tolist()))
    max_foreground = max(count for color, count in count_by_color.items() if color != _GEN_BG)
    return count_by_color.get(_GEN_BG, 0) > max_foreground

def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    rows = int(rng.integers(_GEN_ROW_COUNT_MIN, _GEN_ROW_COUNT_MAX + 1))
    expected, _rect_colors, clutter_colors = _gen_color_grid(rng, rows)
    grid = np.zeros((_GEN_N, _GEN_N), dtype=int)

    row_intervals, col_intervals = _gen_lattice_intervals(rng, rows)
    _gen_render_rectangles(grid, expected, row_intervals, col_intervals)

    if not _gen_add_clutter(rng, grid, clutter_colors):
        return None

    if not _gen_background_is_dominant(grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0a1d4ef5 input")
