from __future__ import annotations

from functools import lru_cache
import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 8)
LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 2))
MIN_REGION_SIZE = 5
MAX_REGION_SIZE = 9
MAX_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the marked region's motif by the number of maroon dots it contains.

    Input structure:
    - The background is black.
    - Grey partition lines form equal rectangular regions, with one orange pixel replacing one grey border pixel on the uniquely marked region.
    - Each region contains one small colored pattern inside a 2x2 or 3x3 bounding box, plus several maroon singleton dots elsewhere in the region.

    Rule:
    - Find the region whose outer grey border contains the orange replacement pixel.
    - Count the maroon dots inside that marked region; this count is the integer scale factor.
    - Crop the marked region's colored pattern to its tight bounding box and enlarge it by that factor, turning each input pixel into an equal-sized square block.

    Key insight:
    The orange border marker chooses which region matters, while the maroon-dot count inside that same region encodes how much to enlarge its local motif.
    """
    g = np.asarray(grid, dtype=int)
    line_rows, line_cols = _find_partition_lines(g)
    target_row, target_col = _find_marked_region(g, line_rows, line_cols)

    row_spans = [(line_rows[i] + 1, line_rows[i + 1] - 1) for i in range(len(line_rows) - 1)]
    col_spans = [(line_cols[i] + 1, line_cols[i + 1] - 1) for i in range(len(line_cols) - 1)]
    r0, r1 = row_spans[target_row]
    c0, c1 = col_spans[target_col]
    region = g[r0 : r1 + 1, c0 : c1 + 1]

    scale = int(np.count_nonzero(region == MAROON))
    assert 1 <= scale <= 4, "marked region must contain between 1 and 4 maroon dots"

    pattern = _extract_pattern(region)
    return np.repeat(np.repeat(pattern, scale, axis=0), scale, axis=1)


def _find_partition_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    line_rows = [int(r) for r in range(grid.shape[0]) if bool(np.isin(grid[r], (GREY, ORANGE)).all())]
    line_cols = [int(c) for c in range(grid.shape[1]) if bool(np.isin(grid[:, c], (GREY, ORANGE)).all())]
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "grid must contain partition lines"
    assert line_rows[0] == 0 and line_rows[-1] == grid.shape[0] - 1
    assert line_cols[0] == 0 and line_cols[-1] == grid.shape[1] - 1
    return line_rows, line_cols


def _span_index(lines: list[int], coord: int) -> int:
    for i in range(len(lines) - 1):
        if lines[i] < coord < lines[i + 1]:
            return int(i)
    raise AssertionError("marker must sit on a border segment, not on a line intersection")


def _find_marked_region(grid: np.ndarray, line_rows: list[int], line_cols: list[int]) -> tuple[int, int]:
    orange = np.argwhere(grid == ORANGE)
    assert len(orange) == 1, "there must be exactly one orange border marker"
    r, c = map(int, orange[0])

    if r == line_rows[0]:
        return 0, _span_index(line_cols, c)
    if r == line_rows[-1]:
        return len(line_rows) - 2, _span_index(line_cols, c)
    if c == line_cols[0]:
        return _span_index(line_rows, r), 0
    if c == line_cols[-1]:
        return _span_index(line_rows, r), len(line_cols) - 2
    raise AssertionError("orange marker must lie on the outer border of the marked region")


def _extract_pattern(region: np.ndarray) -> np.ndarray:
    mask = (region != BLACK) & (region != MAROON)
    assert bool(mask.any()), "marked region must contain a colored pattern"
    coords = np.argwhere(mask)
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    patch = region[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    patch[patch == MAROON] = BLACK
    return patch


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


@lru_cache(maxsize=None)
def _shape_catalog(size: int) -> tuple[np.ndarray, ...]:
    shapes: list[np.ndarray] = []
    min_cells = 2 if size == 2 else 3
    total = size * size
    for bits in range(1, 1 << total):
        flat = np.array([(bits >> k) & 1 for k in range(total)], dtype=bool)
        mask = flat.reshape(size, size)
        if int(mask.sum()) < min_cells:
            continue
        if not (bool(mask[0].any()) and bool(mask[-1].any()) and bool(mask[:, 0].any()) and bool(mask[:, -1].any())):
            continue
        if not _connected(mask):
            continue
        shapes.append(mask)
    return tuple(shapes)


def _sample_dot_positions(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    box_top: int,
    box_left: int,
    box_size: int,
    count: int,
) -> list[tuple[int, int]]:
    free = [
        (r, c)
        for r in range(region_h)
        for c in range(region_w)
        if not (box_top <= r < box_top + box_size and box_left <= c < box_left + box_size)
    ]
    assert len(free) >= count
    picks = rng.choice(len(free), size=count, replace=False)
    return [free[int(i)] for i in np.atleast_1d(picks)]


def _place_orange_marker(
    rng: np.random.Generator,
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    region_row: int,
    region_col: int,
) -> None:
    sides: list[tuple[str, int]] = []
    if region_row == 0:
        sides.append(("top", line_rows[region_row]))
    if region_row == len(line_rows) - 2:
        sides.append(("bottom", line_rows[region_row + 1]))
    if region_col == 0:
        sides.append(("left", line_cols[region_col]))
    if region_col == len(line_cols) - 2:
        sides.append(("right", line_cols[region_col + 1]))

    side, value = sides[int(rng.integers(0, len(sides)))]
    r0, r1 = line_rows[region_row] + 1, line_rows[region_row + 1] - 1
    c0, c1 = line_cols[region_col] + 1, line_cols[region_col + 1] - 1

    if side in ("top", "bottom"):
        c = int(rng.integers(c0, c1 + 1))
        grid[int(value), c] = ORANGE
    else:
        r = int(rng.integers(r0, r1 + 1))
        grid[r, int(value)] = ORANGE


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    region_rows, region_cols = LAYOUTS[int(rng.integers(0, len(LAYOUTS)))]
    n_regions = int(region_rows * region_cols)
    target_scale = int(rng.integers(1, 5))
    target_index = int(rng.integers(0, n_regions))

    max_region_h = min(MAX_REGION_SIZE, (30 - (region_rows + 1)) // region_rows)
    max_region_w = min(MAX_REGION_SIZE, (30 - (region_cols + 1)) // region_cols)
    if max_region_h < MIN_REGION_SIZE or max_region_w < MIN_REGION_SIZE:
        return None
    region_h = int(rng.integers(MIN_REGION_SIZE, max_region_h + 1))
    region_w = int(rng.integers(MIN_REGION_SIZE, max_region_w + 1))

    H = region_rows * region_h + region_rows + 1
    W = region_cols * region_w + region_cols + 1
    grid = np.full((H, W), BLACK, dtype=int)

    line_rows = [int(i * (region_h + 1)) for i in range(region_rows + 1)]
    line_cols = [int(j * (region_w + 1)) for j in range(region_cols + 1)]
    grid[line_rows, :] = GREY
    grid[:, line_cols] = GREY

    colors = rng.choice(np.array(PATTERN_COLORS, dtype=int), size=n_regions, replace=False)
    other_counts = [count for count in range(1, 5) if count != target_scale]
    rng.shuffle(other_counts)
    dot_counts = [None] * n_regions
    dot_counts[target_index] = int(target_scale)
    fill = iter(other_counts[: n_regions - 1])
    for idx in range(n_regions):
        if idx != target_index:
            dot_counts[idx] = int(next(fill))

    target_patch: np.ndarray | None = None
    for idx in range(n_regions):
        rr, cc = divmod(idx, region_cols)
        start_r = line_rows[rr] + 1
        start_c = line_cols[cc] + 1
        size = int(rng.choice(np.array([2, 3], dtype=int)))
        mask = _shape_catalog(size)[int(rng.integers(0, len(_shape_catalog(size))))]
        top = int(rng.integers(0, region_h - size + 1))
        left = int(rng.integers(0, region_w - size + 1))

        patch = np.full((size, size), BLACK, dtype=int)
        patch[mask] = int(colors[idx])
        grid[start_r + top : start_r + top + size, start_c + left : start_c + left + size] = patch

        for dr, dc in _sample_dot_positions(rng, region_h, region_w, top, left, size, int(dot_counts[idx])):
            grid[start_r + int(dr), start_c + int(dc)] = MAROON

        if idx == target_index:
            target_patch = patch.copy()

    assert target_patch is not None
    target_rr, target_cc = divmod(target_index, region_cols)
    _place_orange_marker(rng, grid, line_rows, line_cols, int(target_rr), int(target_cc))

    expected = np.repeat(np.repeat(target_patch, target_scale, axis=0), target_scale, axis=1)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
