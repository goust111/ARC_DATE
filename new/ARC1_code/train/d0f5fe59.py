from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the sparse scene with a square diagonal marker grid whose side length is the number of foreground objects.

    Input structure:
    - The grid has one dominant background color and exactly one foreground color.
    - The foreground pixels form one or more separated 4-connected objects.

    Rule:
    - Count the 4-connected foreground objects.
    - Return a square grid, filled with the background color, with side length equal to that count.
    - Place the foreground color on the main diagonal from upper left to lower right.

    Key insight:
    - The large foreground shapes are tokens to count; their detailed shapes and positions do not affect the output beyond the object count.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = int(fg_colors[0])

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg])
    n = len(objects)
    assert n > 0, "expected at least one foreground object"

    out = np.full((n, n), bg, dtype=int)
    np.fill_diagonal(out, fg)
    return out

# --------------- generator (o5x) ---------------

# Fixed colors.
BG_COLOR, FG_COLOR = 0, 8

# Canvas dimensions and placement spacing.
ROW_MIN, ROW_MAX, COL_MIN, COL_MAX = 8, 16, 10, 12
DIMENSION_OBJECT_FACTOR, DIMENSION_EXTRA = 2, 2
BORDER_MARGIN, PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 2, 4

# Object count, object size, and foreground density.
OBJECT_COUNT_MIN, OBJECT_COUNT_MAX, OBJECT_SIZE_MIN, OBJECT_SIZE_MAX = 2, 5, 2, 8
DENSITY_MIN_NUM, DENSITY_MIN_DEN, DENSITY_MAX_NUM, DENSITY_MAX_DEN = 1, 10, 15, 100

# Blob growth shape bias.
BLOB_BBOX_MAX, ADJACENCY_WEIGHT, COMPACTNESS_BONUS_BASE = 4, 3.0, 5
THIN_BLOB_SIZE_THRESHOLD, THIN_BLOB_WEIGHT = 4, 0.12
ORTHOGONAL_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Retry limit for full input generation.
GENERATION_RETRIES = 300


def _gen_blob(size: int, fg: int, bg: int) -> np.ndarray:
    target_size = int(size)
    coords = {(0, 0)}
    while len(coords) < target_size:
        candidates, weights = [], []
        for r, c in coords:
            for dr, dc in ORTHOGONAL_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coords:
                    continue
                rs = [p[0] for p in coords] + [nr]
                cs = [p[1] for p in coords] + [nc]
                h = max(rs) - min(rs) + 1
                w = max(cs) - min(cs) + 1
                if h <= BLOB_BBOX_MAX and w <= BLOB_BBOX_MAX:
                    adj = sum((nr + ar, nc + ac) in coords for ar, ac in ORTHOGONAL_NEIGHBORS)
                    weight = 1.0 + ADJACENCY_WEIGHT * adj + max(0, COMPACTNESS_BONUS_BASE - max(h, w))
                    if target_size >= THIN_BLOB_SIZE_THRESHOLD and min(h, w) == 1:
                        weight *= THIN_BLOB_WEIGHT
                    candidates.append((nr, nc))
                    weights.append(weight)
        if not candidates:
            return np.full((1, 1), bg, dtype=int)
        coords.add(random.choices(candidates, weights=weights, k=1)[0])

    norm = np.array(list(coords), dtype=int)
    norm -= norm.min(axis=0)
    h, w = norm.max(axis=0) + 1
    patch = np.full((h, w), bg, dtype=int)
    patch[norm[:, 0], norm[:, 1]] = fg
    return patch


def _gen_sizes(n_objects: int, total: int) -> list[int]:
    sizes = []
    remaining = int(total)
    for slots_left in range(int(n_objects), 0, -1):
        lo = max(OBJECT_SIZE_MIN, remaining - OBJECT_SIZE_MAX * (slots_left - 1))
        hi = min(OBJECT_SIZE_MAX, remaining - OBJECT_SIZE_MIN * (slots_left - 1))
        size = random.randint(lo, hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _choose_dimensions(n_objects: int) -> tuple[int, int, int, int] | None:
    """Pick a feasible grid size and total foreground bounds."""
    dim_choices = []
    min_side = DIMENSION_OBJECT_FACTOR * int(n_objects) + DIMENSION_EXTRA
    for rows in range(max(ROW_MIN, min_side), ROW_MAX + 1):
        for cols in range(max(COL_MIN, min_side), COL_MAX + 1):
            cells = rows * cols
            lo = max(OBJECT_SIZE_MIN * n_objects, (DENSITY_MIN_NUM * cells + DENSITY_MIN_DEN - 1) // DENSITY_MIN_DEN)
            hi = min(OBJECT_SIZE_MAX * n_objects, (DENSITY_MAX_NUM * cells) // DENSITY_MAX_DEN)
            if lo <= hi:
                dim_choices.append((rows, cols, lo, hi))
    return random.choice(dim_choices) if dim_choices else None


def _place_blobs(grid: np.ndarray, sizes: list[int], fg: int, bg: int) -> bool:
    """Generate and place every blob into grid, returning False on rejection."""
    for size in sizes:
        patch = _gen_blob(size, fg, bg)
        if int(np.count_nonzero(patch != bg)) != size:
            return False
        try:
            r0, c0 = random_free_location(grid, patch, transparent=bg, border=BORDER_MARGIN,
                                          padding=PLACEMENT_PADDING, padding_connectivity=PLACEMENT_CONNECTIVITY)
        except ValueError:
            return False
        paste(grid, patch, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    layout = _choose_dimensions(n_objects)
    if layout is None:
        return None
    rows, cols, total_lo, total_hi = layout
    grid = np.full((rows, cols), BG_COLOR, dtype=int)

    sizes = _gen_sizes(n_objects, random.randint(total_lo, total_hi))
    if not _place_blobs(grid, sizes, FG_COLOR, BG_COLOR):
        return None

    expected = np.full((n_objects, n_objects), BG_COLOR, dtype=int)
    np.fill_diagonal(expected, FG_COLOR)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d0f5fe59 input")
