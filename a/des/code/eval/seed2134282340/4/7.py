import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
MAROON = 9

REFLECTION_COLORS = {
    YELLOW: ORANGE,
    ORANGE: RED,
    RED: MAROON,
    MAGENTA: MAROON,
}
OBJECT_COLORS = tuple(REFLECTION_COLORS)


def _is_contiguous(indices: np.ndarray) -> bool:
    return indices.size > 0 and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1))


def _find_river(grid: np.ndarray) -> tuple[str, int, int]:
    blue = grid == BLUE
    assert np.any(blue), "expected a blue river"

    rows = np.flatnonzero(blue.any(axis=1))
    cols = np.flatnonzero(blue.any(axis=0))

    horizontal = (
        2 <= rows.size <= 4
        and _is_contiguous(rows)
        and int(blue.sum()) == int(rows.size * grid.shape[1])
        and np.all(blue[rows, :])
    )
    vertical = (
        2 <= cols.size <= 4
        and _is_contiguous(cols)
        and int(blue.sum()) == int(cols.size * grid.shape[0])
        and np.all(blue[:, cols])
    )
    assert horizontal ^ vertical, "river must be a straight full-span band"

    if horizontal:
        return "horizontal", int(rows[0]), int(rows[-1])
    return "vertical", int(cols[0]), int(cols[-1])


def _mirror_point(r: int, c: int, orient: str, axis_lo: int, axis_hi: int) -> tuple[int, int]:
    if orient == "vertical":
        return r, int(axis_lo + axis_hi - c)
    return int(axis_lo + axis_hi - r), c


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orient, axis_lo, axis_hi = _find_river(g)
    out = g.copy()
    object_mask = np.isin(g, OBJECT_COLORS)
    H, W = g.shape

    for r, c in np.argwhere(object_mask):
        rr, cc = _mirror_point(int(r), int(c), orient, axis_lo, axis_hi)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == GREEN:
            out[rr, cc] = REFLECTION_COLORS[int(g[r, c])]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect objects across the straight blue river and recolor the reflection."""
    return _solve(grid)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
RIVER_WIDTHS = (2, 3, 4)
MAX_OBJECT_SPAN = 5
MAX_OBJECT_CELLS = 10
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 80


def _axis_start_candidates(axis_len: int, river_width: int, source_low: bool, truncated: bool) -> list[int]:
    candidates: list[int] = []
    for start in range(3, axis_len - river_width - 2):
        if axis_len - (start + river_width) < 3:
            continue
        mirror_sum = 2 * start + river_width - 1
        if source_low:
            is_truncated = mirror_sum > axis_len - 1
        else:
            is_truncated = mirror_sum < axis_len - 1
        if is_truncated == truncated:
            candidates.append(start)
    return candidates


def _connected_mask(max_h: int, max_w: int) -> np.ndarray:
    while True:
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        if h * w >= 2:
            break

    target = random.randint(2, min(h * w, MAX_OBJECT_CELLS))
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _candidate_positions(
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    orient: str,
    source_low: bool,
    axis_lo: int,
    axis_hi: int,
    role: str,
) -> list[tuple[int, int]]:
    H, W = grid_shape
    h, w = mask.shape

    if orient == "vertical":
        if source_low:
            min_left, max_left = 0, axis_lo - 1 - w
            boundary_left = 0
            near_left = axis_lo - 1 - w
        else:
            min_left, max_left = axis_hi + 2, W - w
            boundary_left = W - w
            near_left = axis_hi + 2
        if min_left > max_left:
            return []
        if role == "boundary":
            lefts = [boundary_left]
        elif role == "near":
            lefts = [near_left]
        else:
            lefts = list(range(min_left, max_left + 1))
        positions = [(top, left) for top in range(H - h + 1) for left in lefts]
    else:
        if source_low:
            min_top, max_top = 0, axis_lo - 1 - h
            boundary_top = 0
            near_top = axis_lo - 1 - h
        else:
            min_top, max_top = axis_hi + 2, H - h
            boundary_top = H - h
            near_top = axis_hi + 2
        if min_top > max_top:
            return []
        if role == "boundary":
            tops = [boundary_top]
        elif role == "near":
            tops = [near_top]
        else:
            tops = list(range(min_top, max_top + 1))
        positions = [(top, left) for top in tops for left in range(W - w + 1)]

    random.shuffle(positions)
    return positions


def _can_place(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    r0 = max(0, top - 1)
    r1 = min(H, top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(W, left + w + 1)
    return np.all(grid[r0:r1, c0:c1] == GREEN)


def _stamp(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _sample_colors(count: int) -> list[int]:
    if count >= 4 and random.random() < 0.5:
        palette = list(OBJECT_COLORS)
    else:
        min_palette = 1 if count == 1 else 2
        palette = random.sample(OBJECT_COLORS, random.randint(min_palette, min(4, count)))
    colors = list(palette)
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors[:count]


def _reflection_stats(grid: np.ndarray) -> tuple[int, int]:
    orient, axis_lo, axis_hi = _find_river(grid)
    H, W = grid.shape
    total = 0
    in_bounds = 0
    object_mask = np.isin(grid, OBJECT_COLORS)
    for r, c in np.argwhere(object_mask):
        rr, cc = _mirror_point(int(r), int(c), orient, axis_lo, axis_hi)
        total += 1
        if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == GREEN:
            in_bounds += 1
    return total, in_bounds


def _valid_input(grid: np.ndarray, expect_truncated: bool) -> bool:
    orient, axis_lo, axis_hi = _find_river(grid)
    object_mask = np.isin(grid, OBJECT_COLORS)
    if not np.any(object_mask):
        return False

    if orient == "vertical":
        left_has = np.any(object_mask[:, :axis_lo])
        right_has = np.any(object_mask[:, axis_hi + 1 :])
        if left_has == right_has:
            return False
        if left_has:
            if np.any(object_mask[:, axis_lo - 1 :]):
                return False
            if not np.all(grid[:, axis_hi + 1 :] == GREEN):
                return False
        else:
            if np.any(object_mask[:, : axis_hi + 2]):
                return False
            if not np.all(grid[:, :axis_lo] == GREEN):
                return False
    else:
        top_has = np.any(object_mask[:axis_lo, :])
        bottom_has = np.any(object_mask[axis_hi + 1 :, :])
        if top_has == bottom_has:
            return False
        if top_has:
            if np.any(object_mask[axis_lo - 1 :, :]):
                return False
            if not np.all(grid[axis_hi + 1 :, :] == GREEN):
                return False
        else:
            if np.any(object_mask[: axis_hi + 2, :]):
                return False
            if not np.all(grid[:axis_lo, :] == GREEN):
                return False

    total, in_bounds = _reflection_stats(grid)
    if in_bounds == 0 or total == 0:
        return False
    if expect_truncated != (in_bounds < total):
        return False

    out = main(grid)
    return out.shape == grid.shape and np.any(out != grid)


def _gen_attempt() -> tuple[np.ndarray, bool] | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    orient = random.choice(("vertical", "horizontal"))
    source_low = random.choice((True, False))
    truncated = random.choice((True, False))
    river_width = random.choice(RIVER_WIDTHS)

    axis_len = W if orient == "vertical" else H
    starts = _axis_start_candidates(axis_len, river_width, source_low, truncated)
    if not starts:
        return None

    axis_lo = random.choice(starts)
    axis_hi = axis_lo + river_width - 1
    grid = np.full((H, W), GREEN, dtype=int)
    if orient == "vertical":
        grid[:, axis_lo : axis_hi + 1] = BLUE
        source_depth = axis_lo - 1 if source_low else W - axis_hi - 2
        span = H
    else:
        grid[axis_lo : axis_hi + 1, :] = BLUE
        source_depth = axis_lo - 1 if source_low else H - axis_hi - 2
        span = W

    area = source_depth * span
    min_count = 2 if truncated else 1
    max_count = min(5, max(min_count, area // 12))
    object_count = random.randint(min_count, max_count)

    roles = ["free"] * object_count
    if truncated:
        roles[:2] = ["boundary", "near"]
        if random.random() < 0.5:
            roles[0], roles[1] = roles[1], roles[0]
    else:
        roles[0] = random.choice(("boundary", "near", "free"))
    colors = _sample_colors(object_count)

    max_h = min(MAX_OBJECT_SPAN, H if orient == "vertical" else source_depth)
    max_w = min(MAX_OBJECT_SPAN, source_depth if orient == "vertical" else W)
    if max_h < 1 or max_w < 1:
        return None

    for role, color in zip(roles, colors):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            mask = _connected_mask(max_h, max_w)
            positions = _candidate_positions(mask, grid.shape, orient, source_low, axis_lo, axis_hi, role)
            for top, left in positions:
                if _can_place(grid, mask, top, left):
                    _stamp(grid, mask, top, left, color)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    if not _valid_input(grid, truncated):
        return None
    return grid, truncated


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _truncated = result
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
