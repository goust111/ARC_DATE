import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _region_has_signal(grid: np.ndarray, bg: int, r0: int, r1: int, c0: int, c1: int) -> bool:
    return bool(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != bg))


def _split_candidates(
    grid: np.ndarray, bg: int, r0: int, r1: int, c0: int, c1: int
) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []

    for r in range(r0 + 1, r1):
        row = grid[r, c0 : c1 + 1]
        color = int(row[0])
        if color == bg or not np.all(row == color):
            continue
        if r - r0 != r1 - r:
            continue
        if np.any(grid[r0:r, c0 : c1 + 1] == color) or np.any(grid[r + 1 : r1 + 1, c0 : c1 + 1] == color):
            continue
        top_has = _region_has_signal(grid, bg, r0, r - 1, c0, c1)
        bottom_has = _region_has_signal(grid, bg, r + 1, r1, c0, c1)
        if top_has == bottom_has:
            continue
        candidates.append(("H", r, "top" if top_has else "bottom"))

    for c in range(c0 + 1, c1):
        col = grid[r0 : r1 + 1, c]
        color = int(col[0])
        if color == bg or not np.all(col == color):
            continue
        if c - c0 != c1 - c:
            continue
        if np.any(grid[r0 : r1 + 1, c0:c] == color) or np.any(grid[r0 : r1 + 1, c + 1 : c1 + 1] == color):
            continue
        left_has = _region_has_signal(grid, bg, r0, r1, c0, c - 1)
        right_has = _region_has_signal(grid, bg, r0, r1, c + 1, c1)
        if left_has == right_has:
            continue
        candidates.append(("V", c, "left" if left_has else "right"))

    return candidates


def _solve_rect(grid: np.ndarray, bg: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    candidates = _split_candidates(grid, bg, r0, r1, c0, c1)
    if not candidates:
        leaf = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.any(leaf != bg), "leaf region must contain the seed pattern"
        return leaf

    assert len(candidates) == 1, "expected exactly one valid nested split in each non-leaf rectangle"
    orient, pos, side = candidates[0]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)

    if orient == "H":
        line_idx = pos - r0
        child_h = line_idx
        out[line_idx, :] = grid[pos, c0 : c1 + 1]
        if side == "top":
            child = _solve_rect(grid, bg, r0, pos - 1, c0, c1)
            assert child.shape == (child_h, c1 - c0 + 1)
            out[:line_idx, :] = child
            out[line_idx + 1 :, :] = np.flipud(child)
        else:
            child = _solve_rect(grid, bg, pos + 1, r1, c0, c1)
            assert child.shape == (child_h, c1 - c0 + 1)
            out[:line_idx, :] = np.flipud(child)
            out[line_idx + 1 :, :] = child
        return out

    line_idx = pos - c0
    child_w = line_idx
    out[:, line_idx] = grid[r0 : r1 + 1, pos]
    if side == "left":
        child = _solve_rect(grid, bg, r0, r1, c0, pos - 1)
        assert child.shape == (r1 - r0 + 1, child_w)
        out[:, :line_idx] = child
        out[:, line_idx + 1 :] = np.fliplr(child)
    else:
        child = _solve_rect(grid, bg, r0, r1, pos + 1, c1)
        assert child.shape == (r1 - r0 + 1, child_w)
        out[:, :line_idx] = np.fliplr(child)
        out[:, line_idx + 1 :] = child
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a seed motif through a nested stack of mirror axes.

    Input structure:
    - The dominant color is the background.
    - Each non-leaf rectangle contains exactly one full-width or full-height
      one-cell-thick split line of a unique color.
    - Every split line bisects its current rectangle into two equal children.
    - Only one deepest leaf rectangle contains a non-background seed pattern;
      every sibling rectangle on the path to the root is blank background.

    Rule:
    - Starting from the seed leaf, mirror the entire current filled rectangle
      across the nearest split line to fill its blank sibling.
    - Continue outward through the nested split lines until the whole canvas is
      reconstructed.
    - Keep every original split line in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    return _solve_rect(g, bg, 0, g.shape[0] - 1, 0, g.shape[1] - 1)


GRID_MIN_SIZE = 8
GRID_MAX_SIZE = 30
LEAF_MIN_SIZE = 2
LEAF_MAX_SIZE = 10
MAX_DEPTH_PER_AXIS = 3
MAX_GENERATION_ATTEMPTS = 400


def _pow2(n: int) -> int:
    return 1 << n


def _leaf_size_bounds(split_count: int) -> tuple[int, int]:
    scale = _pow2(split_count)
    offset = scale - 1
    min_leaf = max(LEAF_MIN_SIZE, (GRID_MIN_SIZE - offset + scale - 1) // scale)
    max_leaf = min(LEAF_MAX_SIZE, (GRID_MAX_SIZE - offset) // scale)
    return min_leaf, max_leaf


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_connected_cells(h: int, w: int, target: int) -> list[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = {
            nbr
            for r, c in cells
            for nbr in _neighbors4(r, c, h, w)
            if nbr not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return list(cells)


def _random_seed_patch(h: int, w: int, bg: int, colors: list[int]) -> np.ndarray | None:
    area = h * w
    min_cells = max(2, (area + 3) // 4)
    max_cells = area if area <= 2 else max(min_cells, area - 1)

    for _ in range(80):
        patch = np.full((h, w), bg, dtype=int)
        target = random.randint(min_cells, max_cells)
        cells = _grow_connected_cells(h, w, target)
        random.shuffle(cells)

        used_colors = colors[: random.randint(1, min(3, len(colors)))]
        random.shuffle(used_colors)
        for idx, (r, c) in enumerate(cells):
            if idx < len(used_colors):
                patch[r, c] = used_colors[idx]
            else:
                patch[r, c] = random.choice(used_colors)

        if _split_candidates(patch, bg, 0, h - 1, 0, w - 1):
            continue
        if np.all(patch == bg):
            continue
        return patch
    return None


def _expand_input(current: np.ndarray, bg: int, orient: str, side: str, line_color: int) -> np.ndarray:
    h, w = current.shape
    if orient == "H":
        out = np.full((2 * h + 1, w), bg, dtype=int)
        out[h, :] = line_color
        if side == "top":
            out[:h, :] = current
        else:
            out[h + 1 :, :] = current
        return out

    out = np.full((h, 2 * w + 1), bg, dtype=int)
    out[:, w] = line_color
    if side == "left":
        out[:, :w] = current
    else:
        out[:, w + 1 :] = current
    return out


def _expand_output(current: np.ndarray, orient: str, side: str, line_color: int) -> np.ndarray:
    h, w = current.shape
    if orient == "H":
        out = np.empty((2 * h + 1, w), dtype=int)
        out[h, :] = line_color
        if side == "top":
            out[:h, :] = current
            out[h + 1 :, :] = np.flipud(current)
        else:
            out[:h, :] = np.flipud(current)
            out[h + 1 :, :] = current
        return out

    out = np.empty((h, 2 * w + 1), dtype=int)
    out[:, w] = line_color
    if side == "left":
        out[:, :w] = current
        out[:, w + 1 :] = np.fliplr(current)
    else:
        out[:, :w] = np.fliplr(current)
        out[:, w + 1 :] = current
    return out


def _apply_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id == 0:
        out = grid
    elif transform_id == 1:
        out = np.rot90(grid, 1)
    elif transform_id == 2:
        out = np.rot90(grid, 2)
    elif transform_id == 3:
        out = np.rot90(grid, 3)
    elif transform_id == 4:
        out = np.flipud(grid)
    elif transform_id == 5:
        out = np.fliplr(grid)
    elif transform_id == 6:
        out = grid.T
    elif transform_id == 7:
        out = np.fliplr(grid.T)
    else:
        raise ValueError(transform_id)
    return np.ascontiguousarray(out)


def _gen_attempt() -> np.ndarray | None:
    split_options = [
        (v_count, h_count)
        for v_count in range(MAX_DEPTH_PER_AXIS + 1)
        for h_count in range(MAX_DEPTH_PER_AXIS + 1)
        if v_count + h_count >= 1
        and _leaf_size_bounds(v_count)[0] <= _leaf_size_bounds(v_count)[1]
        and _leaf_size_bounds(h_count)[0] <= _leaf_size_bounds(h_count)[1]
    ]
    v_count, h_count = random.choice(split_options)
    orientations = ["V"] * v_count + ["H"] * h_count
    random.shuffle(orientations)

    leaf_h_min, leaf_h_max = _leaf_size_bounds(h_count)
    leaf_w_min, leaf_w_max = _leaf_size_bounds(v_count)
    if leaf_h_min > leaf_h_max or leaf_w_min > leaf_w_max:
        return None
    leaf_h = random.randint(leaf_h_min, leaf_h_max)
    leaf_w = random.randint(leaf_w_min, leaf_w_max)

    bg = random.randrange(10)
    available = [c for c in range(10) if c != bg]
    depth = len(orientations)
    if len(available) < depth + 1:
        return None
    random.shuffle(available)
    line_colors = available[:depth]
    seed_colors = available[depth:]
    seed = _random_seed_patch(leaf_h, leaf_w, bg, seed_colors)
    if seed is None:
        return None

    inp = seed.copy()
    out = seed.copy()
    for orient, line_color in zip(orientations, line_colors):
        side = random.choice(("top", "bottom")) if orient == "H" else random.choice(("left", "right"))
        inp = _expand_input(inp, bg, orient, side, line_color)
        out = _expand_output(out, orient, side, line_color)

    transform_id = random.randrange(8)
    inp = _apply_transform(inp, transform_id)
    out = _apply_transform(out, transform_id)

    if _bg_color(inp) != bg:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-mirror sample")
