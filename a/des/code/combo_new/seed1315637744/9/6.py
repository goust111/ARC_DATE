from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]) and int(row[0]) != bg:
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1, "expected exactly one full-width non-background divider row"
    divider_row, divider_color = candidates[0]
    assert divider_row == grid.shape[0] // 2, "divider row must split the grid into equal halves"
    return divider_row, divider_color


def _find_separator_columns(
    grid: np.ndarray, divider_row: int, bg: int, divider_color: int
) -> tuple[list[int], int]:
    sep_cols: list[int] = []
    sep_colors: set[int] = set()
    for c in range(grid.shape[1]):
        top = grid[:divider_row, c]
        bottom = grid[divider_row + 1 :, c]
        if top.size == 0 or bottom.size == 0:
            continue
        top_color = int(top[0])
        if (
            np.all(top == top_color)
            and np.all(bottom == top_color)
            and top_color not in {bg, divider_color}
            and int(grid[divider_row, c]) == divider_color
        ):
            sep_cols.append(int(c))
            sep_colors.add(top_color)
    assert sep_cols, "expected at least one vertical separator"
    assert len(sep_colors) == 1, "all vertical separators must share one color"
    return sep_cols, next(iter(sep_colors))


def _intervals(width: int, sep_cols: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(c) for c in sep_cols) + [width]
    parts: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = int(left + 1)
        stop = int(right)
        assert stop > start, "separators must leave non-empty intervals"
        parts.append((start, stop))
    return parts


def _extract_shape(patch: np.ndarray, bg: int, sep_color: int) -> tuple[np.ndarray, int]:
    mask = (patch != bg) & (patch != sep_color)
    assert mask.any(), "each upper interval must contain an object"
    colors = np.unique(patch[mask])
    assert len(colors) == 1, "each upper interval must contain exactly one object color"

    rows, cols = np.where(mask)
    r0 = int(rows.min())
    r1 = int(rows.max()) + 1
    c0 = int(cols.min())
    c1 = int(cols.max()) + 1
    cropped_mask = mask[r0:r1, c0:c1]
    return cropped_mask, int(colors[0])


def _tile_region(region: np.ndarray, stamp_mask: np.ndarray, color: int) -> None:
    stamp_h, stamp_w = map(int, stamp_mask.shape)
    region_h, region_w = map(int, region.shape)

    for r0 in range(0, region_h, stamp_h):
        for c0 in range(0, region_w, stamp_w):
            tile_h = min(stamp_h, region_h - r0)
            tile_w = min(stamp_w, region_w - c0)
            block = region[r0 : r0 + tile_h, c0 : c0 + tile_w]
            block[stamp_mask[:tile_h, :tile_w]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each upper partition's sprite as a stamp to tile the matching lower partition.

    Input structure:
    - Background is black.
    - One full-width red row splits the grid into equal upper and lower halves.
    - Grey vertical separators at matching columns divide both halves into corresponding column intervals.
    - Each upper interval contains exactly one small single-color object, while the matching lower interval is empty.

    Rule:
    - For each column interval, extract the object's cropped binary shape from the upper half.
    - Keep the upper half, the red divider, and the grey separators unchanged.
    - In the matching lower interval, repeatedly stamp that cropped shape from the interval's top-left corner in row-major tile order, clipping any overhang at the interval boundary.

    Key insight:
    - The upper objects are not moved targets; each one defines a reusable shape mask that becomes a tiled texture in its mirrored lower compartment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    divider_row, divider_color = _find_divider_row(g, bg)
    sep_cols, sep_color = _find_separator_columns(g, divider_row, bg, divider_color)
    intervals = _intervals(g.shape[1], sep_cols)

    out = g.copy()
    lower = out[divider_row + 1 :, :]

    for start, stop in intervals:
        upper_patch = g[:divider_row, start:stop]
        lower_patch = lower[:, start:stop]
        stamp_mask, color = _extract_shape(upper_patch, bg, sep_color)
        _tile_region(lower_patch, stamp_mask, color)

    return out


_RNG = np.random.default_rng()

_BG = 0
_DIVIDER = 2
_SEPARATOR = 5
_OBJECT_COLORS = (1, 3, 4, 6, 7, 8, 9)

_HALF_HEIGHT_MIN = 5
_HALF_HEIGHT_MAX = 12
_INTERVAL_COUNT_MIN = 2
_INTERVAL_COUNT_MAX = 5
_INTERVAL_WIDTH_MIN = 4
_INTERVAL_WIDTH_MAX = 7
_GRID_MAX = 30
_MAX_GENERATION_ATTEMPTS = 300


def _odd_choices(limit: int) -> list[int]:
    return [size for size in range(3, limit + 1, 2)]


def _make_hline(max_h: int, max_w: int) -> np.ndarray:
    width = int(_RNG.integers(2, max_w + 1))
    return np.ones((1, width), dtype=bool)


def _make_vline(max_h: int, max_w: int) -> np.ndarray:
    height = int(_RNG.integers(2, max_h + 1))
    return np.ones((height, 1), dtype=bool)


def _make_diag(max_h: int, max_w: int) -> np.ndarray:
    size = int(_RNG.integers(2, min(max_h, max_w) + 1))
    return np.eye(size, dtype=bool)


def _make_anti_diag(max_h: int, max_w: int) -> np.ndarray:
    size = int(_RNG.integers(2, min(max_h, max_w) + 1))
    return np.fliplr(np.eye(size, dtype=bool))


def _make_plus(max_h: int, max_w: int) -> np.ndarray:
    choices = _odd_choices(min(max_h, max_w))
    size = int(_RNG.choice(choices))
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    mask[center, :] = True
    mask[:, center] = True
    return mask


def _make_x(max_h: int, max_w: int) -> np.ndarray:
    size = int(_RNG.integers(3, min(max_h, max_w) + 1))
    mask = np.eye(size, dtype=bool)
    mask |= np.fliplr(np.eye(size, dtype=bool))
    return mask


def _make_l(max_h: int, max_w: int) -> np.ndarray:
    height = int(_RNG.integers(2, max_h + 1))
    width = int(_RNG.integers(2, max_w + 1))
    mask = np.zeros((height, width), dtype=bool)
    orientation = int(_RNG.integers(0, 4))
    if orientation == 0:
        mask[0, :] = True
        mask[:, 0] = True
    elif orientation == 1:
        mask[0, :] = True
        mask[:, -1] = True
    elif orientation == 2:
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True
    return mask


def _make_t(max_h: int, max_w: int) -> np.ndarray:
    feasible_orientations: list[int] = []
    if max_h >= 2 and max_w >= 3:
        feasible_orientations.extend([0, 2])
    if max_h >= 3 and max_w >= 2:
        feasible_orientations.extend([1, 3])
    orientation = int(_RNG.choice(feasible_orientations))

    if orientation in {0, 2}:
        height = int(_RNG.integers(2, max_h + 1))
        width = int(_RNG.integers(3, max_w + 1))
        mask = np.zeros((height, width), dtype=bool)
        center = width // 2
        mask[:, center] = True
        if orientation == 0:
            mask[0, :] = True
        else:
            mask[-1, :] = True
        return mask

    height = int(_RNG.integers(3, max_h + 1))
    width = int(_RNG.integers(2, max_w + 1))
    mask = np.zeros((height, width), dtype=bool)
    center = height // 2
    mask[center, :] = True
    if orientation == 1:
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    return mask


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    families: list[str] = []
    if max_w >= 2:
        families.append("hline")
    if max_h >= 2:
        families.append("vline")
    if min(max_h, max_w) >= 2:
        families.extend(["diag", "anti_diag", "l"])
    if _odd_choices(min(max_h, max_w)):
        families.append("plus")
    if min(max_h, max_w) >= 3:
        families.extend(["x", "t"])

    assert families, "each interval must admit at least one shape family"
    family = str(_RNG.choice(families))
    if family == "hline":
        return _make_hline(max_h, max_w)
    if family == "vline":
        return _make_vline(max_h, max_w)
    if family == "diag":
        return _make_diag(max_h, max_w)
    if family == "anti_diag":
        return _make_anti_diag(max_h, max_w)
    if family == "plus":
        return _make_plus(max_h, max_w)
    if family == "x":
        return _make_x(max_h, max_w)
    if family == "l":
        return _make_l(max_h, max_w)
    return _make_t(max_h, max_w)


def _sample_interval_widths(count: int) -> list[int]:
    widths = [_INTERVAL_WIDTH_MIN] * count
    used = sum(widths) + (count - 1)
    max_extra = min(
        _GRID_MAX - used,
        count * (_INTERVAL_WIDTH_MAX - _INTERVAL_WIDTH_MIN),
    )
    extra = int(_RNG.integers(0, max_extra + 1))
    capacities = [_INTERVAL_WIDTH_MAX - _INTERVAL_WIDTH_MIN] * count

    while extra > 0:
        choices = [idx for idx, cap in enumerate(capacities) if cap > 0]
        idx = int(_RNG.choice(choices))
        widths[idx] += 1
        capacities[idx] -= 1
        extra -= 1

    return widths


def _build_input() -> np.ndarray:
    count = int(_RNG.integers(_INTERVAL_COUNT_MIN, _INTERVAL_COUNT_MAX + 1))
    widths = _sample_interval_widths(count)
    half_h = int(_RNG.integers(_HALF_HEIGHT_MIN, _HALF_HEIGHT_MAX + 1))
    height = 2 * half_h + 1
    width = sum(widths) + (count - 1)
    assert height <= _GRID_MAX and width <= _GRID_MAX

    grid = np.full((height, width), _BG, dtype=int)
    grid[half_h, :] = _DIVIDER

    colors = _RNG.choice(_OBJECT_COLORS, size=count, replace=False)

    col = 0
    for idx, interval_width in enumerate(widths):
        if idx > 0:
            sep_col = col - 1
            grid[:half_h, sep_col] = _SEPARATOR
            grid[half_h + 1 :, sep_col] = _SEPARATOR

        max_shape_h = half_h - 1
        max_shape_w = int(interval_width) - 1
        shape = _random_shape(max_shape_h, max_shape_w)
        shape_h, shape_w = map(int, shape.shape)

        row0 = int(_RNG.integers(0, half_h - shape_h + 1))
        col0 = int(col + _RNG.integers(0, int(interval_width) - shape_w + 1))
        patch = grid[row0 : row0 + shape_h, col0 : col0 + shape_w]
        patch[shape] = int(colors[idx])

        col += int(interval_width) + 1

    return grid


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    try:
        bg = _most_common_color(grid)
        divider_row, divider_color = _find_divider_row(grid, bg)
        sep_cols, sep_color = _find_separator_columns(grid, divider_row, bg, divider_color)
        if divider_color != _DIVIDER or sep_color != _SEPARATOR:
            return False

        for start, stop in _intervals(grid.shape[1], sep_cols):
            upper_patch = grid[:divider_row, start:stop]
            lower_patch = grid[divider_row + 1 :, start:stop]
            stamp_mask, color = _extract_shape(upper_patch, bg, sep_color)
            if int(color) in {_BG, _DIVIDER, _SEPARATOR} or not stamp_mask.any():
                return False
            if not np.all(lower_patch == bg):
                return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not np.array_equal(out[: divider_row + 1], grid[: divider_row + 1]):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if _generated_grid_is_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
