from __future__ import annotations

import random

import numpy as np


BLACK = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRY_LIMIT = 500


def _neighbors(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _connected_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    return components


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask.astype(bool), factor, axis=0), factor, axis=1)


def _scaled_match_factor(mask_a: np.ndarray, mask_b: np.ndarray) -> int | None:
    for factor in (2, 3):
        for small, large in ((mask_a, mask_b), (mask_b, mask_a)):
            sh, sw = small.shape
            lh, lw = large.shape
            if (sh * factor, sw * factor) != (lh, lw):
                continue
            if np.array_equal(_scale_mask(small, factor), large):
                return factor
    return None


def _find_unique_scale_factor(components: list[dict]) -> int:
    matches: list[int] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            factor = _scaled_match_factor(components[i]["mask"], components[j]["mask"])
            if factor is not None:
                matches.append(int(factor))
    assert len(matches) == 1, "expected exactly one scaled-shape pair"
    return matches[0]


def _infer_fixed_square_tile(grid: np.ndarray, size: int, *, require_full: bool) -> np.ndarray | None:
    tile = np.full((size, size), -1, dtype=int)
    seen = np.zeros((size, size), dtype=bool)
    for r, c in np.argwhere(grid != BLACK):
        pr, pc = int(r) % size, int(c) % size
        value = int(grid[r, c])
        if not seen[pr, pc]:
            tile[pr, pc] = value
            seen[pr, pc] = True
        elif tile[pr, pc] != value:
            return None
    if require_full and not bool(np.all(seen)):
        return None
    return tile


def _render_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols]


def _find_divider_column(grid: np.ndarray) -> int:
    divider_cols = np.flatnonzero(np.all(grid == DIVIDER, axis=0))
    assert divider_cols.size == 1, "expected exactly one full-height gray divider"
    return int(divider_cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the left wallpaper by reading the wall-tile size from a scaled-shape match on the right.

    Input structure:
    - A full-height gray vertical divider splits the grid into a left wallpaper region and a right object region.
    - The left region is a periodic non-black wallpaper with some cells replaced by black holes.
    - The right region is a black background with exactly three separated monochrome 4-connected objects.

    Rule:
    - Find the unique pair of right-side objects where one mask is an integer 2x or 3x enlargement of the other, and record that scale as S.
    - Use S as the wallpaper tile width and height, ignoring black cells while assigning each phase (r mod S, c mod S) its unique visible color.
    - Fill every black hole in the left region from that inferred SxS tile and crop away the divider and right region.

    Key insight:
    - The wallpaper period is not discovered by searching the wallpaper alone; it is encoded as the shape-scale ratio of a separate object pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_column(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]

    components = _connected_components(right, bg=BLACK)
    assert len(components) == 3, "expected three right-side objects"
    scale = _find_unique_scale_factor(components)

    tile = _infer_fixed_square_tile(left, scale, require_full=True)
    assert tile is not None, "left wallpaper must determine a unique full tile"
    out = _render_tile(tile, *left.shape)
    assert np.all(out[left != BLACK] == left[left != BLACK]), "filled wallpaper must preserve visible cells"
    return out


def _row_period(tile: np.ndarray) -> int:
    h = tile.shape[0]
    for period in range(1, h + 1):
        if np.array_equal(tile, tile[np.arange(h) % period]):
            return period
    return h


def _col_period(tile: np.ndarray) -> int:
    w = tile.shape[1]
    for period in range(1, w + 1):
        if np.array_equal(tile, tile[:, np.arange(w) % period]):
            return period
    return w


def _build_full_period_tile(size: int, palette: list[int]) -> np.ndarray | None:
    n_colors = random.randint(2, min(3, len(palette)))
    colors = random.sample(palette, n_colors)
    tile = np.array(
        [[random.choice(colors) for _ in range(size)] for _ in range(size)],
        dtype=int,
    )
    if len(set(map(int, tile.flat))) < 2:
        return None
    if _row_period(tile) != size or _col_period(tile) != size:
        return None
    return tile


def _grow_connected_mask(
    height: int,
    width: int,
    area: int,
    *,
    require_full_bbox: bool,
    forbid_full_rectangle: bool,
) -> np.ndarray | None:
    if area < 1 or area > height * width:
        return None
    if forbid_full_rectangle and area == height * width:
        return None

    for _ in range(150):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set(_neighbors(start[0], start[1], height, width))

        while len(cells) < area:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors(cell[0], cell[1], height, width):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        rr, cc = zip(*cells)
        mask[rr, cc] = True
        if require_full_bbox:
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
        if forbid_full_rectangle and bool(np.all(mask)):
            continue
        return mask
    return None


def _sample_base_shape(scale: int) -> np.ndarray | None:
    height_options = (2, 3, 4) if scale == 2 else (2, 3)
    width_options = (2, 3, 4) if scale == 2 else (2, 3)
    for _ in range(150):
        h = random.choice(height_options)
        w = random.choice(width_options)
        min_area = max(3, h + w - 1)
        max_area = h * w - 1
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(
            h,
            w,
            area,
            require_full_bbox=True,
            forbid_full_rectangle=True,
        )
        if mask is not None:
            return mask
    return None


def _sample_distractor(large_mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, large_mask.shape)
    area = int(large_mask.sum())
    for _ in range(250):
        mask = _grow_connected_mask(
            h,
            w,
            area,
            require_full_bbox=True,
            forbid_full_rectangle=False,
        )
        if mask is None or np.array_equal(mask, large_mask):
            continue
        return mask
    return None


def _phase_counts(height: int, width: int, size: int) -> np.ndarray:
    counts = np.zeros((size, size), dtype=int)
    for r in range(height):
        for c in range(width):
            counts[r % size, c % size] += 1
    return counts


def _random_hole_shape(max_h: int, max_w: int) -> np.ndarray | None:
    h = random.randint(1, max_h)
    w = random.randint(1, max_w)
    max_area = min(h * w, 6)
    area = random.randint(1, max_area)
    return _grow_connected_mask(
        h,
        w,
        area,
        require_full_bbox=False,
        forbid_full_rectangle=False,
    )


def _sample_missing_mask(height: int, width: int, size: int) -> np.ndarray | None:
    total = height * width
    target = random.randint(max(3, total // 6), max(4, total // 3))
    phase_left = _phase_counts(height, width, size)
    missing = np.zeros((height, width), dtype=bool)

    for _ in range(300):
        if int(missing.sum()) >= target:
            break
        shape = _random_hole_shape(min(4, height), min(4, width))
        if shape is None:
            continue
        sh, sw = shape.shape
        if sh > height or sw > width:
            continue
        r0 = random.randint(0, height - sh)
        c0 = random.randint(0, width - sw)
        coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(shape)]
        next_phases = phase_left.copy()
        valid = True
        new_cells = 0
        for r, c in coords:
            if missing[r, c]:
                continue
            next_phases[r % size, c % size] -= 1
            if next_phases[r % size, c % size] <= 0:
                valid = False
                break
            new_cells += 1
        if not valid or new_cells == 0:
            continue
        for r, c in coords:
            if not missing[r, c]:
                missing[r, c] = True
                phase_left[r % size, c % size] -= 1

    if int(missing.sum()) < max(2, target // 2):
        return None

    # Top off with scattered single-cell holes when needed.
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    for r, c in cells:
        if int(missing.sum()) >= target:
            break
        if missing[r, c] or phase_left[r % size, c % size] <= 1:
            continue
        missing[r, c] = True
        phase_left[r % size, c % size] -= 1

    if not bool(np.all(phase_left > 0)):
        return None
    return missing


def _place_right_objects(
    masks: list[np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, int] | None:
    order = list(range(len(masks)))
    random.shuffle(order)
    masks = [masks[i] for i in order]
    colors = [colors[i] for i in order]

    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]
    base_gap = [random.randint(1, 2) for _ in range(len(masks) - 1)]
    min_height = sum(heights) + sum(base_gap)
    if min_height > 26:
        return None

    height = random.randint(max(12, min_height), min(26, min_height + 4))
    right_width = max(widths) + random.randint(2, 4)
    slack = height - min_height
    bins = [0] * (len(masks) + 1)
    for _ in range(slack):
        bins[random.randrange(len(bins))] += 1

    canvas = np.zeros((height, right_width), dtype=int)
    r = bins[0]
    for idx, (mask, color) in enumerate(zip(masks, colors)):
        h, w = map(int, mask.shape)
        c = random.randint(0, right_width - w)
        canvas[r : r + h, c : c + w][mask] = int(color)
        r += h
        if idx < len(masks) - 1:
            r += base_gap[idx] + bins[idx + 1]
    return canvas, height


def _build_right_region(scale: int, colors: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    small = _sample_base_shape(scale)
    if small is None:
        return None
    large = _scale_mask(small, scale)
    distractor = _sample_distractor(large)
    if distractor is None:
        return None

    right_region_and_height = _place_right_objects([small, large, distractor], colors)
    if right_region_and_height is None:
        return None
    right_region, _height = right_region_and_height
    components = _connected_components(right_region, bg=BLACK)
    if len(components) != 3:
        return None
    if _find_unique_scale_factor(components) != scale:
        return None
    return right_region, small, large, distractor


def _validate_input(
    grid: np.ndarray,
    expected: np.ndarray,
    scale: int,
    true_tile: np.ndarray,
    small_mask: np.ndarray,
    large_mask: np.ndarray,
    distractor_mask: np.ndarray,
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    divider = _find_divider_column(grid)
    left = grid[:, :divider]
    right = grid[:, divider + 1 :]
    if not np.any(left == BLACK):
        return False
    if np.any(expected == BLACK):
        return False
    if not np.array_equal(_render_tile(true_tile, *expected.shape), expected):
        return False

    inferred = _infer_fixed_square_tile(left, scale, require_full=True)
    if inferred is None or not np.array_equal(inferred, true_tile):
        return False

    other = 5 - scale
    if _infer_fixed_square_tile(left, other, require_full=True) is not None:
        return False

    components = _connected_components(right, bg=BLACK)
    if len(components) != 3:
        return False
    if _find_unique_scale_factor(components) != scale:
        return False

    masks = [component["mask"] for component in components]
    if not any(np.array_equal(mask, small_mask) for mask in masks):
        return False
    if sum(np.array_equal(mask, large_mask) for mask in masks) != 1:
        return False
    if sum(np.array_equal(mask, distractor_mask) for mask in masks) != 1:
        return False
    if large_mask.shape != distractor_mask.shape or int(large_mask.sum()) != int(distractor_mask.sum()):
        return False

    solved = main(grid)
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    scale = random.choice((2, 3))
    right_colors = random.sample(PALETTE, 3)
    left_palette = [color for color in PALETTE if color not in right_colors]
    tile = _build_full_period_tile(scale, left_palette)
    if tile is None:
        return None

    right_result = _build_right_region(scale, right_colors)
    if right_result is None:
        return None
    right, small_mask, large_mask, distractor_mask = right_result
    height = right.shape[0]

    left_min_width = max(7, 2 * scale + 3)
    left_max_width = min(14, 29 - right.shape[1])
    if left_min_width > left_max_width:
        return None
    left_width = random.randint(left_min_width, left_max_width)

    expected = _render_tile(tile, height, left_width)
    missing = _sample_missing_mask(height, left_width, scale)
    if missing is None:
        return None
    observed = expected.copy()
    observed[missing] = BLACK

    grid = np.zeros((height, left_width + 1 + right.shape[1]), dtype=int)
    grid[:, :left_width] = observed
    grid[:, left_width] = DIVIDER
    grid[:, left_width + 1 :] = right

    if not _validate_input(grid, expected, scale, tile, small_mask, large_mask, distractor_mask):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
