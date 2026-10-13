import random

import numpy as np


KEEP_RATIO_MIN = 0.60
MAX_GENERATION_ATTEMPTS = 500
COLOR_POOL = (1, 2, 3, 4)
SIZE_BY_LAYER_COUNT = {3: 6, 4: 8}


def _depth_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.minimum(
        np.minimum(rows, h - 1 - rows),
        np.minimum(cols, w - 1 - cols),
    )


def _layer_slots_by_col(depths: np.ndarray, depth: int) -> list[list[int]]:
    h, w = depths.shape
    slots = []
    for c in range(w):
        rows = [r for r in range(h) if int(depths[r, c]) == depth]
        slots.append(rows)
    return slots


def _settle_layer(occupied: np.ndarray, depths: np.ndarray, depth: int) -> np.ndarray:
    out = np.zeros_like(occupied, dtype=bool)
    for c, rows in enumerate(_layer_slots_by_col(depths, depth)):
        count = sum(bool(occupied[r, c]) for r in rows)
        if count:
            for r in rows[-count:]:
                out[r, c] = True
    return out


def _reverse_layer_colors_only(grid: np.ndarray) -> np.ndarray:
    depths = _depth_grid(grid.shape)
    layer_count = int(depths.max()) + 1
    layer_colors = []
    for depth in range(layer_count):
        mask = depths == depth
        colors = sorted(int(v) for v in np.unique(grid[mask]) if int(v) != 0)
        assert len(colors) == 1
        layer_colors.append(colors[0])

    out = np.zeros_like(grid)
    rev = list(reversed(layer_colors))
    for depth in range(layer_count):
        mask = (depths == depth) & (grid != 0)
        out[mask] = rev[depth]
    return out


def _global_drop_occupancy(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for c in range(w):
        count = int(mask[:, c].sum())
        if count:
            out[h - count :, c] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply layer-constrained downward gravity to concentric rings, then reverse the layer-color order.

    Input structure:
    - The grid is a square of concentric one-cell-thick depth layers defined by nearest-edge distance.
    - Every depth layer contains cells of one non-black color, but some cells on each layer are erased to black gaps.
    - Different depths have different colors, and black cells are only missing parts of those layers.

    Rule:
    - For each depth independently, treat only the cells belonging to that depth as legal positions.
    - In every column, let that depth's non-black cells fall downward within its own legal positions only, skipping cells that belong to other depths.
    - After all depths settle, read the surviving layer colors from outermost to innermost, reverse that sequence, and recolor each settled depth with its reversed partner color while leaving black gaps black.

    Key insight:
    - Gravity is constrained by the concentric depth masks, so geometry changes within each layer before a separate depth-wise color reversal is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.size > 0

    depths = _depth_grid(g.shape)
    layer_count = int(depths.max()) + 1

    settled_masks = []
    layer_colors = []
    for depth in range(layer_count):
        layer_mask = depths == depth
        colors = sorted(int(v) for v in np.unique(g[layer_mask]) if int(v) != 0)
        assert len(colors) == 1, "each layer must have exactly one non-black color"
        layer_colors.append(colors[0])
        occupied = (g != 0) & layer_mask
        settled_masks.append(_settle_layer(occupied, depths, depth))

    out = np.zeros_like(g)
    reversed_colors = list(reversed(layer_colors))
    for depth, mask in enumerate(settled_masks):
        out[mask] = reversed_colors[depth]
    return out


def _side_categories(coords: list[tuple[int, int]], n: int, depth: int) -> set[str]:
    cats = set()
    top = depth
    bottom = n - 1 - depth
    left = depth
    right = n - 1 - depth
    for r, c in coords:
        if r == top:
            cats.add("top")
        if r == bottom:
            cats.add("bottom")
        if c == left:
            cats.add("left")
        if c == right:
            cats.add("right")
    return cats


def _sample_layer_input(
    n: int,
    depth: int,
    depths: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, bool]:
    layer_mask = depths == depth
    coords = [tuple(map(int, rc)) for rc in np.argwhere(layer_mask)]
    size = len(coords)
    min_keep = int(np.ceil(size * KEEP_RATIO_MIN))
    max_remove = size - min_keep
    min_remove = 1 if size <= 4 else 2

    for _ in range(200):
        remove_count = rng.randint(min_remove, max_remove)
        removed = rng.sample(coords, remove_count)
        cats = _side_categories(removed, n, depth)
        min_cats = 2 if size <= 4 else 3
        if len(cats) < min_cats:
            continue

        occupied = np.zeros((n, n), dtype=bool)
        for r, c in coords:
            occupied[r, c] = True
        for r, c in removed:
            occupied[r, c] = False

        settled = _settle_layer(occupied, depths, depth)
        moved = bool(np.any(settled != occupied))

        if size > 4 and not moved and rng.random() < 0.8:
            continue
        return occupied, moved

    raise RuntimeError(f"failed to sample layer {depth}")


def _build_candidate(rng: random.Random) -> np.ndarray:
    layer_count = rng.choice((3, 4))
    n = SIZE_BY_LAYER_COUNT[layer_count]
    depths = _depth_grid((n, n))

    colors = list(rng.sample(COLOR_POOL, layer_count))
    grid = np.zeros((n, n), dtype=int)
    moved_layers = 0

    for depth in range(layer_count):
        occupied, moved = _sample_layer_input(n, depth, depths, rng)
        moved_layers += int(moved)
        grid[occupied] = colors[depth]

    if moved_layers < 2:
        return None
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    n = grid.shape[0]
    depths = _depth_grid(grid.shape)
    layer_count = int(depths.max()) + 1

    original_colors = []
    constrained_occ = np.zeros_like(grid, dtype=bool)
    moved_layers = 0
    for depth in range(layer_count):
        layer_mask = depths == depth
        vals = sorted(int(v) for v in np.unique(grid[layer_mask]) if int(v) != 0)
        if len(vals) != 1:
            return False
        original_colors.append(vals[0])

        occupied = (grid != 0) & layer_mask
        if int(occupied.sum()) < int(np.ceil(layer_mask.sum() * KEEP_RATIO_MIN)):
            return False

        missing = [tuple(map(int, rc)) for rc in np.argwhere(layer_mask & (grid == 0))]
        min_cats = 2 if int(layer_mask.sum()) <= 4 else 3
        if len(_side_categories(missing, n, depth)) < min_cats:
            return False

        settled = _settle_layer(occupied, depths, depth)
        constrained_occ |= settled
        moved_layers += int(np.any(settled != occupied))

    if moved_layers < 2:
        return False
    if original_colors == list(reversed(original_colors)):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _reverse_layer_colors_only(grid)):
        return False

    global_occ = _global_drop_occupancy(grid != 0)
    if np.array_equal(global_occ, constrained_occ):
        return False

    if len({int(v) for v in np.unique(grid) if int(v) != 0}) != layer_count:
        return False
    if len({int(v) for v in np.unique(out) if int(v) != 0}) != layer_count:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _build_candidate(rng)
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid concentric-layer gravity input")
