import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _normalize_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r = min(rows)
    min_c = min(cols)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _coords_to_mask(coords):
    norm = _normalize_coords(coords)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in norm:
        mask[r, c] = 1
    return mask


def _degree(point, points):
    r, c = point
    return sum((r + dr, c + dc) in points for dr, dc in DIR4)


def _is_connected(coords):
    points = set(coords)
    start = next(iter(points))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in points and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(points)


def _is_asymmetric(coords):
    mask = _coords_to_mask(coords)
    variants = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    return all(mask.shape != v.shape or not np.array_equal(mask, v) for v in variants)


def _valid_shape(coords):
    points = set(_normalize_coords(coords))
    if len(points) < 4 or len(points) > 8:
        return False
    if not _is_connected(points):
        return False
    rows = {r for r, _ in points}
    cols = {c for _, c in points}
    if len(rows) < 2 or len(cols) < 2:
        return False
    endpoints = sum(_degree(point, points) <= 1 for point in points)
    if endpoints < 2:
        return False
    return _is_asymmetric(points)


def _transform_shape(coords, rng):
    mask = _coords_to_mask(coords)
    mask = np.rot90(mask, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return tuple(sorted(map(tuple, np.argwhere(mask))))


def _random_walk_shape(area, rng):
    for _ in range(120):
        path = [(0, 0)]
        used = {(0, 0)}
        while len(path) < area:
            r, c = path[-1]
            options = [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) not in used]
            if not options:
                break
            nxt = options[int(rng.integers(0, len(options)))]
            path.append(nxt)
            used.add(nxt)
        if len(used) == area:
            shape = _transform_shape(used, rng)
            if _valid_shape(shape):
                return shape
    return None


def _random_branch_shape(area, rng):
    for _ in range(120):
        coords = {(0, 0)}
        while len(coords) < area:
            frontier = set()
            for r, c in coords:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in coords:
                        frontier.add(nxt)
            if not frontier:
                break
            chosen = tuple(list(frontier)[int(rng.integers(0, len(frontier)))])
            coords.add(chosen)
        if len(coords) == area:
            shape = _transform_shape(coords, rng)
            if _valid_shape(shape):
                return shape
    return None


def _random_shape(rng):
    area = int(rng.integers(4, 9))
    makers = [_random_walk_shape, _random_branch_shape]
    for _ in range(200):
        maker = makers[int(rng.integers(0, len(makers)))]
        shape = maker(area, rng)
        if shape is not None and _valid_shape(shape):
            return tuple(sorted(shape))
    raise RuntimeError("failed to build a valid source shape")


def _sparse_marker_columns(width, rng):
    for _ in range(100):
        start = int(rng.integers(0, min(3, width - 2)))
        cols = [start]
        while True:
            nxt = cols[-1] + int(rng.choice([3, 4]))
            if nxt >= width:
                break
            cols.append(nxt)
            if len(cols) >= 6 and rng.random() < 0.5:
                break
        if 3 <= len(cols) <= 6 and cols[-1] - cols[0] >= max(4, width // 2 - 1):
            return cols
    raise RuntimeError("failed to place sparse marker columns")


def _reflected_points(red_points, axis_row):
    return {(2 * axis_row - r, c) for r, c in red_points}


def _endpoint_count(points):
    return sum(_degree(point, points) <= 1 for point in points)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the red motif across the sparse marker row and selectively expand only endpoint pixels of the reflected copy.

    Input structure:
    - The background is black.
    - A sparse grey horizontal marker row sits above the grid midpoint, a single red 4-connected motif lies above it, and the bottom-left corner holds a guide pixel.
    - The guide pixel color is reused for the reflected copy, while the red motif and the grey marker row remain visible.

    Rule:
    - Read the guide color from the bottom-left corner, then erase that guide pixel to black.
    - Reflect every red motif pixel across the grey marker row and paint the reflected copy with the guide color.
    - Within the reflected copy only, treat copied pixels with at most one copied 4-neighbor as endpoints, and expand each endpoint by one cell in the four cardinal directions onto black cells only.
    - Leave the original red motif and the grey marker row unchanged.

    Key insight:
    - Reflection creates the copied shape, but only copied pixels that are local endpoints grow, so the final blue expansion depends on the reflected shape's topology rather than on a fixed dilation stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    guide_color = int(g[-1, 0])
    assert guide_color != BLACK, "expected a non-black guide pixel in the bottom-left corner"

    grey_rows = np.where(np.any(g == GREY, axis=1))[0]
    assert len(grey_rows) >= 1, "expected a sparse grey marker row"
    axis_row = int(grey_rows[np.argmax(np.sum(g == GREY, axis=1)[grey_rows])])

    red_points = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}
    assert red_points, "expected a red source motif"
    reflected = _reflected_points(red_points, axis_row)

    out = g.copy()
    out[-1, 0] = BLACK

    for r, c in reflected:
        assert 0 <= r < h and 0 <= c < w, "reflected copy must stay in bounds"
        out[r, c] = guide_color

    for r, c in reflected:
        if _degree((r, c), reflected) > 1:
            continue
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BLACK:
                out[nr, nc] = guide_color

    return out


def _gen_attempt(rng):
    shape = _random_shape(rng)
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1

    h = int(rng.integers(max(14, 2 * shape_h + 4), 19))
    w = int(rng.integers(max(11, shape_w + 3), 17))

    axis_low = max(4, shape_h + 1)
    axis_high = min(h // 2 + 1, h - shape_h - 3)
    if axis_low > axis_high:
        return None
    axis_row = int(rng.integers(axis_low, axis_high + 1))

    top_low = max(0, 2 * axis_row - h + 2)
    top_high = axis_row - shape_h - 1
    if top_low > top_high:
        return None
    top = int(rng.integers(top_low, top_high + 1))

    left_low = 1
    left_high = w - shape_w - 1
    if left_low > left_high:
        return None
    left = int(rng.integers(left_low, left_high + 1))

    grid = np.zeros((h, w), dtype=int)
    marker_cols = _sparse_marker_columns(w, rng)
    grid[axis_row, marker_cols] = GREY
    grid[-1, 0] = BLUE

    red_points = {(top + r, left + c) for r, c in shape}
    for r, c in red_points:
        grid[r, c] = RED

    reflected = _reflected_points(red_points, axis_row)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in reflected):
        return None
    if red_points & reflected:
        return None
    if _endpoint_count(reflected) < 2:
        return None

    out = main(grid)
    blue_count = int(np.sum(out == BLUE))
    added_growth = blue_count - len(reflected)
    if added_growth < 3:
        return None
    if np.sum(grid == GREY) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
