from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
OUTPUT_COLORS = [1, 3, 4, 6, 7, 8]  # blue, green, yellow, magenta, orange, azure
INPUT_COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300

_GEN_CALL_COUNT = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_lines = [
        int(r)
        for r in range(grid.shape[0])
        if np.all(grid[r, :] != BG) and np.any(grid[r, :] == RED)
    ]
    col_lines = [
        int(c)
        for c in range(grid.shape[1])
        if np.all(grid[:, c] != BG) and np.any(grid[:, c] == RED)
    ]
    total = len(row_lines) + len(col_lines)
    assert 2 <= total <= 3, f"expected 2 or 3 separator lines, got rows={row_lines}, cols={col_lines}"
    return row_lines, col_lines


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each path by how many red separator lines it crosses.

    Input structure:
    - The background is black.
    - Two or three full red separator rows or columns split the grid into regions.
    - Several 4-connected monochrome paths in distinct colors run through the regions.
    - Whenever a path crosses a separator line, the crossing cell on that line takes the path color instead of red.

    Rule:
    - Detect each monochrome path independently of its original color.
    - Count how many cells of that path lie on the separator rows or columns; this is the path's crossing count.
    - Recolor the whole path to the fixed sequence blue, green, yellow, magenta, orange, azure for crossing counts 0 through 5.
    - Leave untouched background cells black and untouched separator cells red.

    Key insight:
    - The input path colors are distractors; only the number of separator crossings determines the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep_rows, sep_cols = _separator_lines(g)
    sep_rows_set = set(sep_rows)
    sep_cols_set = set(sep_cols)

    out = g.copy()
    path_colors = [int(color) for color in np.unique(g) if int(color) not in (BG, RED)]
    assert path_colors, "expected at least one colored path"

    for color in path_colors:
        for comp in _find_components(g == color):
            crossings = sum(1 for r, c in comp if r in sep_rows_set or c in sep_cols_set)
            assert 0 <= crossings < len(OUTPUT_COLORS), f"unsupported crossing count {crossings}"
            out[[r for r, _ in comp], [c for _, c in comp]] = OUTPUT_COLORS[crossings]

    return out


def _append_straight(cells: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r0, c0 = cells[-1]
    r1, c1 = end
    if r0 == r1 and c0 == c1:
        return
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1 + step, step):
            cells.append((r0, c))
        return
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1 + step, step):
            cells.append((r, c0))
        return
    raise AssertionError("segments must be axis-aligned")


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _region_bounds(widths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    regions: list[tuple[int, int]] = []
    separators: list[int] = []
    col = 0
    for i, width in enumerate(widths):
        regions.append((col, col + width - 1))
        col += width
        if i + 1 < len(widths):
            separators.append(col)
            col += 1
    return regions, separators, col


def _trim_bounds(bounds: tuple[int, int], region_idx: int, region_count: int, width: int) -> tuple[int, int]:
    lo, hi = bounds
    if region_idx == 0:
        lo = max(lo, 1)
    if region_idx == region_count - 1:
        hi = min(hi, width - 2)
    return lo, hi


def _pick_start_col(
    bounds: tuple[int, int],
    region_idx: int,
    region_count: int,
    width: int,
    outgoing_dir: int,
) -> int:
    lo, hi = _trim_bounds(bounds, region_idx, region_count, width)
    assert lo <= hi
    if outgoing_dir > 0:
        return random.randint(lo, max(lo, hi - 2))
    return random.randint(min(hi, lo + 2), hi)


def _pick_end_col(
    bounds: tuple[int, int],
    region_idx: int,
    region_count: int,
    width: int,
    incoming_dir: int,
) -> int:
    lo, hi = _trim_bounds(bounds, region_idx, region_count, width)
    assert lo <= hi
    if incoming_dir > 0:
        return random.randint(min(hi, lo + 2), hi)
    return random.randint(lo, max(lo, hi - 2))


def _make_region_walk(region_count: int, crossings: int) -> list[int]:
    if crossings == 0:
        return [random.randrange(region_count)]

    for _ in range(100):
        region = random.randrange(region_count)
        walk = [region]
        last_dir = 0
        for _step in range(crossings):
            options = []
            if region > 0:
                options.append(-1)
            if region < region_count - 1:
                options.append(1)
            assert options
            if last_dir in options and random.random() < 0.65:
                direction = last_dir
            else:
                direction = random.choice(options)
            region += direction
            walk.append(region)
            last_dir = direction
        if len(walk) == crossings + 1:
            return walk
    raise RuntimeError("failed to build region walk")


def _draw_vertical_path(
    grid: np.ndarray,
    expected: np.ndarray,
    regions: list[tuple[int, int]],
    separators: list[int],
    top: int,
    crossings: int,
    input_color: int,
    output_color: int,
) -> bool:
    region_count = len(regions)
    walk = _make_region_walk(region_count, crossings)
    cells: list[tuple[int, int]]

    if crossings == 0:
        region_idx = walk[0]
        lo, hi = _trim_bounds(regions[region_idx], region_idx, region_count, grid.shape[1])
        if hi - lo < 2:
            return False
        start_col = random.randint(lo, hi - 2)
        end_col = random.randint(start_col + 2, hi)
        cells = [(top, start_col)]
        _append_straight(cells, (top, end_col))
    else:
        first_dir = walk[1] - walk[0]
        start_col = _pick_start_col(
            regions[walk[0]],
            walk[0],
            region_count,
            grid.shape[1],
            first_dir,
        )
        cells = [(top, start_col)]

        for step in range(crossings):
            src_region = walk[step]
            dst_region = walk[step + 1]
            direction = dst_region - src_region
            sep_col = separators[min(src_region, dst_region)]
            pre_col = sep_col - 1 if direction > 0 else sep_col + 1
            post_col = sep_col + 1 if direction > 0 else sep_col - 1
            row = top + step + 1

            _append_straight(cells, (row, cells[-1][1]))
            _append_straight(cells, (row, pre_col))
            _append_straight(cells, (row, sep_col))
            _append_straight(cells, (row, post_col))

        last_dir = walk[-1] - walk[-2]
        end_row = top + crossings + 1
        _append_straight(cells, (end_row, cells[-1][1]))
        end_col = _pick_end_col(
            regions[walk[-1]],
            walk[-1],
            region_count,
            grid.shape[1],
            last_dir,
        )
        _append_straight(cells, (end_row, end_col))

    for r, c in cells:
        if grid[r, c] not in (BG, RED):
            return False
    for r, c in cells:
        grid[r, c] = input_color
        expected[r, c] = output_color
    return True


def _sample_widths(line_count: int) -> list[int]:
    region_count = line_count + 1
    base_width = 5
    extra_total = random.randint(0, 6 if line_count == 2 else 4)
    extras = _random_composition(extra_total, region_count)
    widths = [base_width + extra for extra in extras]
    total_width = sum(widths) + line_count
    assert total_width <= 30
    return widths


def _assign_input_colors(crossings: list[int]) -> list[int]:
    used: set[int] = set()
    colors: list[int] = []
    for k in crossings:
        choices = [c for c in INPUT_COLOR_POOL if c != OUTPUT_COLORS[k] and c not in used]
        assert choices, "not enough distinct path colors"
        color = random.choice(choices)
        used.add(color)
        colors.append(color)
    return colors


def _build_vertical_case(call_index: int) -> tuple[np.ndarray, np.ndarray]:
    line_count = 2 if (call_index // 2) % 2 == 0 else 3
    featured_k = call_index % 6
    path_count = 2 if call_index % 3 == 0 else 3
    crossings = [featured_k] + [random.randint(0, 5) for _ in range(path_count - 1)]
    colors = _assign_input_colors(crossings)

    band_heights = [1 if k == 0 else k + 2 for k in crossings]
    min_height = sum(band_heights) + (path_count - 1) + 2
    height = random.randint(min_height, min(28, min_height + 4))

    extra = height - min_height
    gap_extras = _random_composition(extra, path_count + 1)
    top_margin = 1 + gap_extras[0]
    between_gaps = [1 + extra_gap for extra_gap in gap_extras[1:-1]]
    bottom_margin = 1 + gap_extras[-1]

    widths = _sample_widths(line_count)
    regions, separators, width = _region_bounds(widths)
    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()

    for sep_col in separators:
        grid[:, sep_col] = RED
        expected[:, sep_col] = RED

    row = top_margin
    for idx, (k, input_color) in enumerate(zip(crossings, colors)):
        ok = _draw_vertical_path(
            grid,
            expected,
            regions,
            separators,
            row,
            k,
            input_color,
            OUTPUT_COLORS[k],
        )
        assert ok, "failed to place path"
        row += band_heights[idx]
        if idx + 1 < path_count:
            row += between_gaps[idx]

    assert row + bottom_margin == height
    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _build_vertical_case(call_index)
        if call_index % 2 == 1:
            inp = inp.T.copy()
            expected = expected.T.copy()
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input grid")
