import random

import numpy as np


BG = 0
SEP_COLOR = 9
MAX_ATTEMPTS = 400


def _row_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            bands.append((start, idx - 1))
            start = None
    if start is not None:
        bands.append((start, len(mask) - 1))
    return bands


def _outer_ring(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 1:
        return arr[0].copy()
    if arr.shape[1] == 1:
        return arr[:, 0].copy()
    return np.concatenate((arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]))


def _find_separator_row(grid: np.ndarray) -> int:
    h = grid.shape[0]
    candidates = []
    for r in range(1, h - 1):
        value = int(grid[r, 0])
        if value == BG:
            continue
        if not np.all(grid[r] == value):
            continue
        if not np.any(grid[:r] != BG):
            continue
        if not np.any(grid[r + 1 :] != BG):
            continue
        candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full-width separator row"
    return int(candidates[0])


def _extract_filters(top: np.ndarray) -> dict[int, list[int]]:
    filters: dict[int, list[int]] = {}
    for row in top:
        cols = np.flatnonzero(row != BG)
        if cols.size == 0:
            continue
        colors = {int(v) for v in row[cols]}
        assert len(colors) == 1, "each filter bar must be monochrome"
        color = next(iter(colors))
        assert color not in filters, "filter colors must be unique"
        filters[color] = [int(c) for c in cols.tolist()]
    assert filters, "expected at least one filter bar"
    return filters


def _extract_sources(bottom: np.ndarray, row_offset: int) -> list[dict[str, object]]:
    bands = _row_bands(np.any(bottom != BG, axis=1))
    assert bands, "expected at least one source pattern"

    sources: list[dict[str, object]] = []
    for r0, r1 in bands:
        band = bottom[r0 : r1 + 1]
        cols = np.flatnonzero(np.any(band != BG, axis=0))
        assert cols.size > 0
        c0, c1 = int(cols[0]), int(cols[-1])
        crop = band[:, c0 : c1 + 1]
        assert crop.shape[0] > 1 and crop.shape[1] > 1, "source patterns must be rectangles larger than 1x1"
        ring = _outer_ring(crop)
        border_colors = {int(v) for v in ring}
        assert len(border_colors) == 1 and BG not in border_colors, "outer border must be one non-background color"
        color = next(iter(border_colors))
        sources.append(
            {
                "color": color,
                "bbox": (r0 + row_offset, c0, r1 + row_offset, c1),
                "grid": crop.copy(),
            }
        )
    return sources


def _stack_outputs(parts: list[np.ndarray]) -> np.ndarray:
    assert parts
    total_h = sum(part.shape[0] for part in parts) + max(0, len(parts) - 1)
    total_w = max(part.shape[1] for part in parts)
    out = np.full((total_h, total_w), BG, dtype=int)
    row = 0
    for idx, part in enumerate(parts):
        h, w = part.shape
        out[row : row + h, :w] = part
        row += h
        if idx + 1 < len(parts):
            row += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Filter each colored source rectangle by its matching mask row and left-pack the kept columns.

    Input structure:
    - Black is the background.
    - A single full-width maroon separator row splits the grid into an upper filter region and a lower source region.
    - The upper region contains several one-pixel-high monochrome filter bars on different rows; each bar may have black gaps.
    - The lower region contains several colored rectangular source patterns stacked vertically with black rows between them.
    - Each source rectangle has a uniform outer border color, and that border color matches exactly one filter bar color.

    Rule:
    - Find the separator row, then read each upper filter bar as the set of absolute column indices where its color appears.
    - For every lower source rectangle, match it to the filter bar with the same border color.
    - Keep only the source columns whose absolute indices are present in that filter bar; clear the others.
    - Remove the gaps between kept columns by packing the surviving columns to the left.
    - Stack the packed results in the original top-to-bottom source order with one black row between consecutive outputs.

    Key insight:
    - The colored rows above the separator are functional column masks, and color matching reveals which mask applies to each rectangle before the kept columns are compacted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_row = _find_separator_row(g)
    filters = _extract_filters(g[:sep_row])
    sources = _extract_sources(g[sep_row + 1 :], sep_row + 1)
    assert len(filters) == len(sources), "expected a one-to-one match between filters and sources"

    outputs: list[np.ndarray] = []
    for source in sources:
        color = int(source["color"])
        assert color in filters, "every source must have a matching filter bar color"
        _, c0, _, c1 = source["bbox"]
        keep_cols = [col - c0 for col in sorted(set(filters[color])) if c0 <= col <= c1]
        assert keep_cols, "each filter must keep at least one source column"
        crop = np.asarray(source["grid"], dtype=int)
        outputs.append(crop[:, keep_cols].copy())
    return _stack_outputs(outputs)


def _sample_keep_columns(width: int) -> list[int]:
    styles = ["alternating", "two_blocks", "random"]
    style = random.choice(styles)

    if style == "alternating":
        start = random.randint(0, 1)
        cols = list(range(start, width, 2))
        if len(cols) >= 2 and len(cols) < width and any(b - a > 1 for a, b in zip(cols, cols[1:])):
            return cols

    if style == "two_blocks":
        for _ in range(40):
            left_len = random.randint(1, max(1, width // 3))
            right_len = random.randint(1, max(1, width // 3))
            gap = random.randint(1, width - left_len - right_len)
            start = random.randint(0, width - left_len - gap - right_len)
            cols = list(range(start, start + left_len))
            cols.extend(range(start + left_len + gap, start + left_len + gap + right_len))
            if len(cols) < width:
                return cols

    for _ in range(80):
        k = random.randint(2, width - 1)
        cols = sorted(random.sample(range(width), k))
        if any(b - a > 1 for a, b in zip(cols, cols[1:])):
            return cols
    raise RuntimeError("failed to sample a gapped filter")


def _make_inner_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    style = random.choice(["vertical", "horizontal", "checker"])
    inner = np.zeros((height, width), dtype=int)
    colors = palette.copy()
    if BG not in colors and random.choice([False, True]):
        colors = colors + [BG]
    k = len(colors)
    phase = random.randrange(k)

    rr, cc = np.indices((height, width))
    if style == "vertical":
        inner[:] = np.asarray(colors, dtype=int)[(cc + phase) % k]
    elif style == "horizontal":
        inner[:] = np.asarray(colors, dtype=int)[(2 * rr + cc + phase) % k]
    else:
        inner[:] = np.asarray(colors, dtype=int)[(rr + 2 * cc + phase) % k]
    return inner


def _make_source(height: int, width: int, border_color: int, inner_palette: list[int]) -> np.ndarray:
    source = np.full((height, width), border_color, dtype=int)
    if height > 2 and width > 2:
        source[1:-1, 1:-1] = _make_inner_pattern(height - 2, width - 2, inner_palette)
    return source


def _planned_output(sources: list[np.ndarray], keep_columns: list[list[int]]) -> np.ndarray:
    parts = [source[:, cols].copy() for source, cols in zip(sources, keep_columns)]
    return _stack_outputs(parts)


def _gen_attempt() -> np.ndarray | None:
    n_sources = random.randint(2, 4)
    width = random.randint(5, 11)
    height_cap = {2: 7, 3: 6, 4: 5}[n_sources]
    heights = [random.randint(3, height_cap) for _ in range(n_sources)]

    top_margin = random.randint(0, 2)
    between_filter_gaps = [random.randint(0, 1) for _ in range(n_sources - 1)]
    pre_sep_gap = random.randint(0, 1)
    post_sep_gap = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)

    upper_h = top_margin + n_sources + sum(between_filter_gaps) + pre_sep_gap
    lower_h = post_sep_gap + sum(heights) + (n_sources - 1) + bottom_margin
    total_h = upper_h + 1 + lower_h
    if total_h > 30:
        return None

    left_margin = random.randint(1, 5)
    right_margin = random.randint(1, 5)
    total_w = left_margin + width + right_margin
    if total_w > 30:
        return None

    border_colors = random.sample(list(range(1, 9)), n_sources)
    inner_pool = [color for color in range(1, 9) if color not in border_colors]
    if len(inner_pool) < 2:
        return None

    keep_columns = [_sample_keep_columns(width) for _ in range(n_sources)]
    if not any(len(cols) < width - 1 for cols in keep_columns):
        return None

    sources: list[np.ndarray] = []
    for idx in range(n_sources):
        palette_size = random.randint(2, min(3, len(inner_pool)))
        palette = random.sample(inner_pool, palette_size)
        sources.append(_make_source(heights[idx], width, border_colors[idx], palette))

    expected = _planned_output(sources, keep_columns)
    if max(expected.shape) > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)

    filter_rows = []
    row = top_margin
    for idx in range(n_sources):
        filter_rows.append(row)
        row += 1
        if idx < n_sources - 1:
            row += between_filter_gaps[idx]
    sep_row = upper_h
    grid[sep_row, :] = SEP_COLOR

    bar_order = list(range(n_sources))
    random.shuffle(bar_order)
    for row, src_idx in zip(filter_rows, bar_order):
        abs_cols = [left_margin + c for c in keep_columns[src_idx]]
        grid[row, abs_cols] = border_colors[src_idx]

    row = sep_row + 1 + post_sep_gap
    for idx, source in enumerate(sources):
        h = source.shape[0]
        grid[row : row + h, left_margin : left_margin + width] = source
        row += h
        if idx + 1 < len(sources):
            row += 1

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
