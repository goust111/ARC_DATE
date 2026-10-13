import numpy as np


def _background_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return colors[np.argmax(counts)]


def _component_bbox(mask, start):
    h, w = mask.shape
    stack = [start]
    seen = {start}
    rows = []
    cols = []
    while stack:
        r, c = stack.pop()
        rows.append(r)
        cols.append(c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _seed_bbox(grid, bg):
    mask = grid != bg
    if not np.any(mask):
        return 0, 0, 0, 0
    if mask[0, 0]:
        return _component_bbox(mask, (0, 0))

    points = np.argwhere(mask)
    scores = points[:, 0] + points[:, 1]
    start = tuple(points[np.argmin(scores)])
    return _component_bbox(mask, start)


def _tile(seed, shape, mode):
    h, w = shape
    sh, sw = seed.shape
    out = np.empty(shape, dtype=seed.dtype)
    for r0 in range(0, h, sh):
        for c0 in range(0, w, sw):
            br = r0 // sh
            bc = c0 // sw
            if mode == "translate":
                tile = seed
            elif mode == "row_horizontal_reflect":
                tile = seed[::-1, :] if br % 2 else seed
            else:
                tile = seed[::-1, ::-1] if (br + bc) % 2 else seed
            rh = min(sh, h - r0)
            cw = min(sw, w - c0)
            out[r0 : r0 + rh, c0 : c0 + cw] = tile[:rh, :cw]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the top-left seed into a full-grid tiling chosen by its symmetry.

    Input structure:
    The most common color is the background.  A compact rectangular seed pattern
    touches the upper-left corner and is made from two foreground colors.  The
    upper-right area contains small 2x2 legend icons with single-pixel labels
    showing the available tiling modes; these legend pixels are separated from
    the seed by background.

    Rule:
    Find the seed as the foreground component nearest the upper-left corner and
    test it for horizontal-axis and vertical-axis reflection symmetry.  If the
    seed is invariant under a horizontal-axis reflection, repeat it by simple
    translation.  Otherwise, if it is invariant under a vertical-axis reflection,
    alternate tile rows between the seed and its horizontal-axis reflection.
    If neither axis is symmetric, alternate 180-degree rotations in a checkerboard
    of tile positions.  The selected tiling fills the entire original grid size,
    cropping partial tiles at the right and bottom edges and covering the legend.

    Key insight:
    The seed's symmetry class selects which legend tiling behavior to use; after
    that choice, the output is just the selected periodic extension of the seed.
    """
    bg = _background_color(grid)
    r0, r1, c0, c1 = _seed_bbox(grid, bg)
    seed = grid[r0:r1, c0:c1]
    if seed.size == 0:
        return grid.copy()

    horizontal = np.array_equal(seed, seed[::-1, :])
    vertical = np.array_equal(seed, seed[:, ::-1])
    if horizontal:
        mode = "translate"
    elif vertical:
        mode = "row_horizontal_reflect"
    else:
        mode = "checker_rotate"
    return _tile(seed, grid.shape, mode)


def _random_seed(rng, shape, colors, branch):
    sh, sw = shape
    a, b = colors
    vals = np.array([a, b], dtype=int)
    for _ in range(200):
        if branch == 0:
            top = rng.choice(vals, size=((sh + 1) // 2, sw))
            seed = np.empty((sh, sw), dtype=int)
            seed[: top.shape[0], :] = top
            seed[sh - top.shape[0] :, :] = top[: sh // 2, :][::-1, :]
        elif branch == 1:
            left = rng.choice(vals, size=(sh, (sw + 1) // 2))
            seed = np.empty((sh, sw), dtype=int)
            seed[:, : left.shape[1]] = left
            seed[:, sw - left.shape[1] :] = left[:, : sw // 2][:, ::-1]
        else:
            seed = rng.choice(vals, size=(sh, sw))

        uses_both = np.any(seed == a) and np.any(seed == b)
        horizontal = np.array_equal(seed, seed[::-1, :])
        vertical = np.array_equal(seed, seed[:, ::-1])
        if branch == 0 and uses_both and horizontal:
            return seed
        if branch == 1 and uses_both and vertical and not horizontal:
            return seed
        if branch == 2 and uses_both and not horizontal and not vertical:
            return seed

    if branch == 0:
        seed = np.tile(np.array([[a, b]], dtype=int), (sh, (sw + 1) // 2))[:, :sw]
        return seed
    if branch == 1:
        seed = np.full((sh, sw), a, dtype=int)
        seed[0, :] = b
        seed[:, :] = seed[:, ::-1]
        if np.array_equal(seed, seed[::-1, :]):
            seed[-1, :] = a
        return seed
    seed = np.tile(np.array([[a, b], [b, b]], dtype=int), ((sh + 1) // 2, (sw + 1) // 2))[:sh, :sw]
    if np.array_equal(seed, seed[::-1, :]) or np.array_equal(seed, seed[:, ::-1]):
        seed[0, 0] = a
        seed[-1, -1] = b
    return seed


def _draw_icon(grid, row, col, color, mode, bg):
    if mode == "translate":
        icon = np.array([[color, color], [color, color]], dtype=grid.dtype)
    elif mode == "row_horizontal_reflect":
        icon = np.array([[color, color], [bg, bg]], dtype=grid.dtype)
    else:
        icon = np.array([[color, bg], [bg, color]], dtype=grid.dtype)
    grid[row : row + 2, col : col + 2] = icon
    grid[row, col + 3] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    branch = int(rng.integers(0, 3))
    sh = int(rng.integers(3, 5))
    sw = int(rng.integers(3, 5))
    h = int(rng.integers(max(12, sh * 3), 31))
    w = int(rng.integers(max(22, sw + 17), 31))

    bg = int(rng.integers(0, 10))
    available = [c for c in range(10) if c != bg]
    chosen = rng.choice(np.array(available), size=5, replace=False)
    fg1, fg2 = int(chosen[0]), int(chosen[1])
    marker_colors = [int(c) for c in chosen[2:5]]

    grid = np.full((h, w), bg, dtype=int)
    seed = _random_seed(rng, (sh, sw), (fg1, fg2), branch)
    grid[:sh, :sw] = seed

    modes = ["translate", "row_horizontal_reflect", "checker_rotate"]
    rng.shuffle(modes)
    start = w - 15
    color_by_mode = dict(zip(modes, marker_colors))
    for i, mode in enumerate(modes):
        _draw_icon(grid, 0, start + i * 5, color_by_mode[mode], mode, bg)

    return grid
