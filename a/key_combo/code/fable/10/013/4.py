from __future__ import annotations

import hashlib
from collections import Counter, deque

import numpy as np


_BG = 0
_PALETTE = tuple(range(1, 10))
_MIN_RECT_SIDE = 3
_MAX_RECT_SIDE = 6
_GRID_MIN = 22
_GRID_MAX = 30
_ROWS_RANGE = (2, 4)
_COLS_RANGE = (2, 4)
_GAP_RANGE = (2, 3)
_BORDER_RANGE = (1, 3)
_SCALE_MIN = 2
_SCALE_MAX = 7
_MAX_ATTEMPTS = 4000

_RNG = np.random.default_rng()
_SEEN_INPUT_SIGNATURES: set[str] = set()
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = map(int, shape)
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    assert k >= 1, "expected a positive scale factor"
    return np.kron(np.asarray(sprite, dtype=int), np.ones((k, k), dtype=int))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(g.shape, rr, cc):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )

    return components


def _overlap_groups(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    groups: list[list[int]] = []
    common_los: list[int] = []
    common_his: list[int] = []
    center_sums: list[float] = []
    counts: list[int] = []

    for idx, lo, hi in sorted(intervals, key=lambda item: (item[1], item[2], item[0])):
        matches = [
            group_idx
            for group_idx in range(len(groups))
            if max(common_los[group_idx], lo) <= min(common_his[group_idx], hi)
        ]
        assert len(matches) <= 1, "expected each interval to belong to at most one band"
        if matches:
            group_idx = matches[0]
            groups[group_idx].append(idx)
            common_los[group_idx] = max(common_los[group_idx], lo)
            common_his[group_idx] = min(common_his[group_idx], hi)
            center_sums[group_idx] += (lo + hi) / 2.0
            counts[group_idx] += 1
        else:
            groups.append([idx])
            common_los.append(lo)
            common_his.append(hi)
            center_sums.append((lo + hi) / 2.0)
            counts.append(1)

    assert groups, "expected at least one interval group"
    ordered = sorted(range(len(groups)), key=lambda i: center_sums[i] / counts[i])
    index_to_rank: dict[int, int] = {}
    for rank, group_idx in enumerate(ordered):
        for idx in groups[group_idx]:
            index_to_rank[idx] = rank
    return index_to_rank, len(groups)


def _extract_rectangles_and_clutter(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, tuple[int, int, int, int]]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    clutter: list[tuple[int, int]] = []

    for comp in _find_components(g, bg):
        box = comp["bbox"]
        height = _bbox_h(box)
        width = _bbox_w(box)
        area = int(comp["area"])
        if height >= 3 and width >= 3 and area == height * width:
            rectangles.append((int(comp["color"]), box))
        else:
            assert area == 1, "expected all non-rectangle foreground components to be singleton clutter"
            clutter.append(tuple(map(int, comp["coords"][0])))

    assert rectangles, "expected at least one filled rectangle"
    assert clutter, "expected at least one singleton clutter pixel"
    return bg, rectangles, clutter


def _compact_rectangles(rectangles: list[tuple[int, tuple[int, int, int, int]]]) -> np.ndarray:
    row_by_rect, row_count = _overlap_groups(
        [(idx, box[0], box[2]) for idx, (_color, box) in enumerate(rectangles)]
    )
    col_by_rect, col_count = _overlap_groups(
        [(idx, box[1], box[3]) for idx, (_color, box) in enumerate(rectangles)]
    )
    assert len(rectangles) == row_count * col_count, "expected a complete rectangle lattice"

    out = np.full((row_count, col_count), -1, dtype=int)
    for idx, (color, _box) in enumerate(rectangles):
        rr = row_by_rect[idx]
        cc = col_by_rect[idx]
        assert out[rr, cc] == -1, "expected exactly one rectangle per compact cell"
        out[rr, cc] = color

    assert np.all(out >= 0), "expected every lattice position to be occupied"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the lattice of filled rectangles into a small color table and enlarge it by the clutter count.

    Input structure
    - The background is the most common color.
    - The meaningful objects are monochromatic filled rectangles, each at least 3x3 and exactly filling its own bbox.
    - These rectangles form a complete lattice when grouped by overlapping bbox row bands and overlapping bbox column bands.
    - Additional foreground pixels are isolated singleton clutter pixels; their colors may coincide with rectangle colors.

    Rule
    - Keep only the filled rectangles and discard the singleton clutter.
    - Collapse the rectangles into a compact grid whose rows and columns are defined by the overlapping bbox bands.
    - Count the singleton clutter pixels, and replace every compact cell by a solid square block of that side length.

    Key insight
    - The clutter has a dual role: it is ignored while recovering the compact rectangle table, but its count is the only source of the final enlargement factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _bg, rectangles, clutter = _extract_rectangles_and_clutter(g)
    compact = _compact_rectangles(rectangles)
    return _scale_sprite(compact, len(clutter))


def _sample_band_extents(
    rng: np.random.Generator,
    axis_size: int,
    count: int,
    min_len: int,
    max_len: int,
    gap: int,
    border: int,
) -> list[tuple[int, int]] | None:
    usable = axis_size - 2 * border - gap * (count - 1)
    if usable < count * min_len:
        return None

    lengths: list[int] = []
    used = 0
    for idx in range(count):
        remaining = count - idx - 1
        hi = min(max_len, usable - used - remaining * min_len)
        if hi < min_len:
            return None
        length = int(rng.integers(min_len, hi + 1))
        lengths.append(length)
        used += length

    slack = usable - sum(lengths)
    extras = rng.multinomial(slack, np.full(count + 1, 1.0 / (count + 1)))
    pos = border + int(extras[0])
    extents: list[tuple[int, int]] = []
    for idx, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        extents.append((start, end))
        pos = end + 1
        if idx < count - 1:
            pos += gap + int(extras[idx + 1])
    return extents


def _sample_intervals_in_extent(
    rng: np.random.Generator,
    extent: tuple[int, int],
    count: int,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]]:
    start, end = map(int, extent)
    span = end - start + 1
    common = int(rng.integers(start, end + 1))
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(min_len, min(max_len, span) + 1))
        lo_min = max(start, common - length + 1)
        lo_max = min(common, end - length + 1)
        lo = int(rng.integers(lo_min, lo_max + 1))
        intervals.append((lo, lo + length - 1))
    return intervals


def _sample_color_table(
    rng: np.random.Generator, rows: int, cols: int
) -> tuple[np.ndarray, list[int], list[int]]:
    palette = np.array(_PALETTE, dtype=int)
    rng.shuffle(palette)
    cell_count = rows * cols
    distinct = int(rng.integers(2, min(6, cell_count) + 1))
    rect_colors = palette[:distinct].tolist()
    remaining = palette[distinct:].tolist()

    cells = np.empty(cell_count, dtype=int)
    cells[:distinct] = rect_colors
    if distinct < cell_count:
        cells[distinct:] = rng.choice(rect_colors, size=cell_count - distinct, replace=True)
    rng.shuffle(cells)
    return cells.reshape(rows, cols), [int(c) for c in rect_colors], [int(c) for c in remaining]


def _build_rectangle_grid(
    rng: np.random.Generator, height: int, width: int, table: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int]]]] | None:
    rows, cols = map(int, table.shape)
    row_gap = int(rng.integers(_GAP_RANGE[0], _GAP_RANGE[1] + 1))
    col_gap = int(rng.integers(_GAP_RANGE[0], _GAP_RANGE[1] + 1))
    row_border = int(rng.integers(_BORDER_RANGE[0], _BORDER_RANGE[1] + 1))
    col_border = int(rng.integers(_BORDER_RANGE[0], _BORDER_RANGE[1] + 1))

    row_extents = _sample_band_extents(
        rng, height, rows, _MIN_RECT_SIDE, _MAX_RECT_SIDE, row_gap, row_border
    )
    col_extents = _sample_band_extents(
        rng, width, cols, _MIN_RECT_SIDE, _MAX_RECT_SIDE, col_gap, col_border
    )
    if row_extents is None or col_extents is None:
        return None

    row_intervals = [
        _sample_intervals_in_extent(rng, extent, cols, _MIN_RECT_SIDE, _MAX_RECT_SIDE)
        for extent in row_extents
    ]
    col_intervals = [
        _sample_intervals_in_extent(rng, extent, rows, _MIN_RECT_SIDE, _MAX_RECT_SIDE)
        for extent in col_extents
    ]

    grid = np.full((height, width), _BG, dtype=int)
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []
    for rr in range(rows):
        for cc in range(cols):
            r0, r1 = row_intervals[rr][cc]
            c0, c1 = col_intervals[cc][rr]
            color = int(table[rr, cc])
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            rectangles.append((color, (r0, c0, r1, c1)))
    return grid, rectangles


def _singleton_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) != bg:
                continue
            if all(int(g[nr, nc]) == bg for nr, nc in _neighbors4(g.shape, r, c)):
                candidates.append((r, c))
    return candidates


def _choose_clutter_palette(
    rng: np.random.Generator, rect_colors: list[int], remaining_colors: list[int]
) -> list[int]:
    weighted_modes: list[str] = []
    if rect_colors:
        weighted_modes.extend(["shared", "shared"])
    if rect_colors and remaining_colors:
        weighted_modes.extend(["mixed", "mixed"])
    if remaining_colors:
        weighted_modes.append("new")
    if not weighted_modes:
        weighted_modes.append("shared")

    mode = weighted_modes[int(rng.integers(0, len(weighted_modes)))]
    if mode == "shared":
        count = int(rng.integers(1, min(3, len(rect_colors)) + 1))
        picked = rng.choice(rect_colors, size=count, replace=False)
        return [int(c) for c in picked.tolist()]

    if mode == "mixed":
        shared = [int(rng.choice(rect_colors))]
        fresh_count = int(rng.integers(1, min(2, len(remaining_colors)) + 1))
        fresh = [int(c) for c in rng.choice(remaining_colors, size=fresh_count, replace=False).tolist()]
        return shared + fresh

    count = int(rng.integers(1, min(3, len(remaining_colors)) + 1))
    picked = rng.choice(remaining_colors, size=count, replace=False)
    return [int(c) for c in picked.tolist()]


def _sample_scale(rng: np.random.Generator, rows: int, cols: int) -> int:
    candidates = [
        scale
        for scale in range(_SCALE_MIN, min(_SCALE_MAX, 30 // max(rows, cols)) + 1)
        if scale not in {rows, cols, rows * cols}
    ]
    assert candidates, "expected at least one feasible scale"
    return int(rng.choice(candidates))


def _scatter_clutter(
    rng: np.random.Generator, grid: np.ndarray, scale: int, palette: list[int]
) -> list[tuple[int, int]]:
    colors = list(palette)
    if len(colors) < scale:
        colors.extend(int(c) for c in rng.choice(colors, size=scale - len(colors), replace=True).tolist())
    rng.shuffle(colors)

    placed: list[tuple[int, int]] = []
    for color in colors[:scale]:
        candidates = _singleton_candidates(grid, _BG)
        if not candidates:
            return []
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        grid[r, c] = int(color)
        placed.append((r, c))
    return placed


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(map(int, grid.shape))).encode()).hexdigest()


def _is_valid_case(grid: np.ndarray, table: np.ndarray, scale: int) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    bg = _get_bg(g)
    if bg != _BG:
        return False

    counts = Counter(map(int, g.reshape(-1)))
    if counts[bg] <= max(v for color, v in counts.items() if color != bg):
        return False

    try:
        _bg, rectangles, clutter = _extract_rectangles_and_clutter(g)
        compact = _compact_rectangles(rectangles)
        actual = main(g)
    except AssertionError:
        return False

    if len(clutter) != scale:
        return False
    if len(rectangles) != table.size:
        return False
    if not np.array_equal(compact, table):
        return False

    expected = _scale_sprite(table, scale)
    if max(expected.shape) > 30 or not np.array_equal(actual, expected):
        return False

    color_count_all = len(np.unique(g))
    color_count_non_bg = len(np.unique(g[g != bg]))
    rect_color_count = len(np.unique(table))
    forbidden = {table.shape[0], table.shape[1], table.size, color_count_all, color_count_non_bg, rect_color_count}
    if scale in forbidden:
        return False

    for r, c in clutter:
        if int(g[r, c]) == bg:
            return False
        if any(int(g[nr, nc]) != bg for nr, nc in _neighbors4(g.shape, r, c)):
            return False

    return True


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    rows = int(rng.integers(_ROWS_RANGE[0], _ROWS_RANGE[1] + 1))
    cols = int(rng.integers(_COLS_RANGE[0], _COLS_RANGE[1] + 1))
    scale = _sample_scale(rng, rows, cols)
    height = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    width = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))

    table, rect_colors, remaining_colors = _sample_color_table(rng, rows, cols)
    built = _build_rectangle_grid(rng, height, width, table)
    if built is None:
        return None

    grid, _rectangles = built
    clutter_palette = _choose_clutter_palette(rng, rect_colors, remaining_colors)
    clutter = _scatter_clutter(rng, grid, scale, clutter_palette)
    if len(clutter) != scale:
        return None

    expected = _scale_sprite(table, scale)
    if not _is_valid_case(grid, table, scale):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(expected)
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None

    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        case = _generate_case(_RNG)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
