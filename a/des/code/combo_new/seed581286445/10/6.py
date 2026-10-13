import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_indices(grid: np.ndarray, bg: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] != bg)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] != bg)]


def _separator_color(grid: np.ndarray, row_seps: list[int], col_seps: list[int]) -> int:
    samples = []
    for r in row_seps:
        samples.extend(int(v) for v in grid[r, :])
    for c in col_seps:
        samples.extend(int(v) for v in grid[:, c])
    assert samples, "expected separator rows or columns"
    values, counts = np.unique(np.asarray(samples, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _region_spans(separators: list[int], length: int) -> list[tuple[int, int, int, int]]:
    bounds = [-1, *sorted(int(x) for x in separators), int(length)]
    spans = []
    for prev_bound, next_bound in zip(bounds, bounds[1:]):
        start = prev_bound + 1
        end = next_bound - 1
        if start <= end:
            spans.append((start, end, prev_bound, next_bound))
    return spans


def _marker_op(
    grid: np.ndarray,
    sep_color: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    top_bound: int,
    bottom_bound: int,
    left_bound: int,
    right_bound: int,
) -> str | None:
    total_markers = 0
    seen_ops: set[str] = set()

    if top_bound >= 0:
        top = grid[top_bound, c0 : c1 + 1]
        markers = int(np.count_nonzero(top != sep_color))
        total_markers += markers
        if markers:
            seen_ops.add("flipud")
    if bottom_bound < grid.shape[0]:
        bottom = grid[bottom_bound, c0 : c1 + 1]
        markers = int(np.count_nonzero(bottom != sep_color))
        total_markers += markers
        if markers:
            seen_ops.add("flipud")
    if left_bound >= 0:
        left = grid[r0 : r1 + 1, left_bound]
        markers = int(np.count_nonzero(left != sep_color))
        total_markers += markers
        if markers:
            seen_ops.add("fliplr")
    if right_bound < grid.shape[1]:
        right = grid[r0 : r1 + 1, right_bound]
        markers = int(np.count_nonzero(right != sep_color))
        total_markers += markers
        if markers:
            seen_ops.add("fliplr")

    assert total_markers <= 1, "each region may see at most one boundary marker"
    assert len(seen_ops) <= 1, "boundary markers may not request conflicting flips"
    if not seen_ops:
        return None
    return next(iter(seen_ops))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each subregion's interior according to a colored marker placed on one of its separator-line boundaries.

    Input structure:
    - The dominant color is the background.
    - One-pixel-wide full rows and columns of a common separator color split the
      grid into rectangular subregions.
    - Inside each subregion, a small non-background, non-separator pattern sits
      on the background.
    - Some separator-line boundary segments contain a single non-separator
      marker pixel instead of the separator color.

    Rule:
    - Recover the separator rows and columns, then inspect every subregion's
      top, bottom, left, and right boundary segments.
    - A marker on the top or bottom boundary flips that subregion's interior
      upside down.
    - A marker on the left or right boundary flips that subregion's interior
      left to right.
    - If a subregion sees no marker on its boundary segments, keep its interior
      unchanged.
    - In the output, all separator rows and columns are restored to the
      separator color.

    Key insight:
    - The odd-colored pixels on the separator lines are not noise: their
      boundary position alone encodes which mirror flip to apply inside the
      adjacent rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    row_seps = _separator_indices(g, bg, axis=0)
    col_seps = _separator_indices(g, bg, axis=1)
    assert row_seps and col_seps, "expected separator rows and columns"

    row_spans = _region_spans(row_seps, g.shape[0])
    col_spans = _region_spans(col_seps, g.shape[1])
    assert row_spans and col_spans, "expected at least one subregion"
    sep_color = _separator_color(g, row_seps, col_seps)

    out = g.copy()
    for r0, r1, top_bound, bottom_bound in row_spans:
        for c0, c1, left_bound, right_bound in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            op = _marker_op(
                g,
                sep_color,
                r0,
                r1,
                c0,
                c1,
                top_bound,
                bottom_bound,
                left_bound,
                right_bound,
            )
            if op == "flipud":
                out[r0 : r1 + 1, c0 : c1 + 1] = np.flipud(patch)
            elif op == "fliplr":
                out[r0 : r1 + 1, c0 : c1 + 1] = np.fliplr(patch)
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = patch

    out[row_seps, :] = sep_color
    out[:, col_seps] = sep_color
    return out


GEN_ATTEMPTS = 400
LAYOUTS = ((2, 2), (2, 3), (3, 2))
REGION_SIZE_RANGE = (4, 7)
PATTERN_BOX_MIN = 2
PATTERN_BOX_MAX = 4
PATTERN_AREA_MIN = 3
PATTERN_AREA_MAX = 6
MIN_MARKED_REGIONS = 2


def _choose_sizes(count: int) -> list[int]:
    return [random.randint(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1]) for _ in range(count)]


def _separator_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        positions.append(pos)
    return positions


def _grow_connected_cells(height: int, width: int, area: int) -> set[tuple[int, int]]:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return cells


def _make_region_patch(height: int, width: int, bg: int, color: int) -> np.ndarray | None:
    max_box_h = min(PATTERN_BOX_MAX, height)
    max_box_w = min(PATTERN_BOX_MAX, width)
    if max_box_h < PATTERN_BOX_MIN or max_box_w < PATTERN_BOX_MIN:
        return None

    for _ in range(120):
        box_h = random.randint(PATTERN_BOX_MIN, max_box_h)
        box_w = random.randint(PATTERN_BOX_MIN, max_box_w)
        max_area = min(PATTERN_AREA_MAX, box_h * box_w - 1)
        if max_area < PATTERN_AREA_MIN:
            continue
        area = random.randint(PATTERN_AREA_MIN, max_area)
        cells = _grow_connected_cells(box_h, box_w, area)

        patch = np.full((height, width), bg, dtype=int)
        r_off = random.randint(0, height - box_h)
        c_off = random.randint(0, width - box_w)
        for r, c in cells:
            patch[r_off + r, c_off + c] = color

        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    return None


def _region_bounds(row_seps: list[int], col_seps: list[int]) -> list[dict]:
    regions = []
    for i in range(len(row_seps) - 1):
        for j in range(len(col_seps) - 1):
            regions.append(
                {
                    "i": i,
                    "j": j,
                    "r0": row_seps[i] + 1,
                    "r1": row_seps[i + 1] - 1,
                    "c0": col_seps[j] + 1,
                    "c1": col_seps[j + 1] - 1,
                    "top": row_seps[i],
                    "bottom": row_seps[i + 1],
                    "left": col_seps[j],
                    "right": col_seps[j + 1],
                }
            )
    return regions


def _available_sides(region: dict, row_count: int, col_count: int) -> list[str | None]:
    sides: list[str | None] = [None]
    if region["i"] == 0:
        sides.append("top")
    if region["i"] == row_count - 1:
        sides.append("bottom")
    if region["j"] == 0:
        sides.append("left")
    if region["j"] == col_count - 1:
        sides.append("right")
    return sides


def _choose_region_ops(regions: list[dict], row_count: int, col_count: int) -> list[str | None] | None:
    for _ in range(120):
        ops = [random.choice(_available_sides(region, row_count, col_count)) for region in regions]
        marked = [op for op in ops if op is not None]
        if len(marked) < MIN_MARKED_REGIONS:
            continue
        if all(op in ("top", "bottom") for op in marked):
            continue
        if all(op in ("left", "right") for op in marked):
            continue
        return ops
    return None


def _place_markers(grid: np.ndarray, regions: list[dict], ops: list[str | None], marker: int) -> None:
    for region, op in zip(regions, ops):
        if op == "top":
            c = random.randint(region["c0"], region["c1"])
            grid[region["top"], c] = marker
        elif op == "bottom":
            c = random.randint(region["c0"], region["c1"])
            grid[region["bottom"], c] = marker
        elif op == "left":
            r = random.randint(region["r0"], region["r1"])
            grid[r, region["left"]] = marker
        elif op == "right":
            r = random.randint(region["r0"], region["r1"])
            grid[r, region["right"]] = marker


def _expected_output(
    grid: np.ndarray,
    regions: list[dict],
    ops: list[str | None],
    row_seps: list[int],
    col_seps: list[int],
    sep_color: int,
) -> np.ndarray:
    out = grid.copy()
    for region, op in zip(regions, ops):
        patch = grid[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1]
        if op in ("top", "bottom"):
            out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = np.flipud(patch)
        elif op in ("left", "right"):
            out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = np.fliplr(patch)
        else:
            out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = patch

    out[row_seps, :] = sep_color
    out[:, col_seps] = sep_color
    return out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == int(bg)


def _non_separator_lines_have_background(grid: np.ndarray, row_seps: list[int], col_seps: list[int], bg: int) -> bool:
    row_sep_set = set(row_seps)
    col_sep_set = set(col_seps)
    for r in range(grid.shape[0]):
        if r not in row_sep_set and not np.any(grid[r, :] == bg):
            return False
    for c in range(grid.shape[1]):
        if c not in col_sep_set and not np.any(grid[:, c] == bg):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice(LAYOUTS)
    row_sizes = _choose_sizes(row_count)
    col_sizes = _choose_sizes(col_count)
    row_seps = _separator_positions(row_sizes)
    col_seps = _separator_positions(col_sizes)
    height = row_seps[-1] + 1
    width = col_seps[-1] + 1
    if height > 30 or width > 30:
        return None

    region_total = row_count * col_count
    palette = list(range(10))
    random.shuffle(palette)
    bg, sep_color, marker = palette[:3]
    pattern_colors = palette[3 : 3 + region_total]
    if len(pattern_colors) < region_total:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[row_seps, :] = sep_color
    grid[:, col_seps] = sep_color

    regions = _region_bounds(row_seps, col_seps)
    ops = _choose_region_ops(regions, row_count, col_count)
    if ops is None:
        return None

    for region, color in zip(regions, pattern_colors):
        patch = _make_region_patch(
            region["r1"] - region["r0"] + 1,
            region["c1"] - region["c0"] + 1,
            bg,
            color,
        )
        if patch is None:
            return None
        grid[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = patch

    _place_markers(grid, regions, ops, marker)

    if not _background_is_dominant(grid, bg):
        return None
    if not _non_separator_lines_have_background(grid, row_seps, col_seps, bg):
        return None

    expected = _expected_output(grid, regions, ops, row_seps, col_seps, sep_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
