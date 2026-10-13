import random

import numpy as np


BACKGROUND = 0
CREASE = 5
MOTIF_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _canonicalize(grid: np.ndarray) -> tuple[np.ndarray, bool, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    row_creases = [r for r in range(g.shape[0]) if np.all(g[r, :] == CREASE)]
    col_creases = [c for c in range(g.shape[1]) if np.all(g[:, c] == CREASE)]
    if len(col_creases) == 2 and not row_creases:
        return g.copy(), False, (int(col_creases[0]), int(col_creases[1]))
    if len(row_creases) == 2 and not col_creases:
        gt = g.T.copy()
        transposed_creases = [c for c in range(gt.shape[1]) if np.all(gt[:, c] == CREASE)]
        return gt, True, (int(transposed_creases[0]), int(transposed_creases[1]))
    raise AssertionError("expected exactly two parallel full grey crease lines")


def _background_color(grid: np.ndarray, creases: tuple[int, int]) -> int:
    mask = np.ones_like(grid, dtype=bool)
    for crease in creases:
        mask[:, crease] = False
    vals = grid[mask]
    if vals.size == 0:
        return BACKGROUND
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_motif_cells(
    grid: np.ndarray, bg: int, creases: tuple[int, int]
) -> list[tuple[int, int, int]]:
    mask = grid != bg
    for crease in creases:
        mask[:, crease] = False
    coords = np.argwhere(mask)
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _reflect_cells(
    cells: list[tuple[int, int, int]], crease: int
) -> list[tuple[int, int, int]]:
    return [(r, 2 * crease - c, color) for r, c, color in cells]


def _apply_order(
    shape: tuple[int, int],
    motif_cells: list[tuple[int, int, int]],
    creases: tuple[int, int],
    order: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    h, w = map(int, shape)
    crease_set = set(map(int, creases))
    occupied = {(r, c): color for r, c, color in motif_cells}
    current = list(motif_cells)

    for crease in order:
        mirrored = _reflect_cells(current, crease)
        fresh: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c, color in mirrored:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if c in crease_set:
                return None
            if (r, c) in occupied or (r, c) in seen:
                return None
            seen.add((r, c))
            fresh.append((r, c, color))
        for r, c, color in fresh:
            occupied[(r, c)] = color
        current = [(r, c, color) for (r, c), color in occupied.items()]

    if len(occupied) != 4 * len(motif_cells):
        return None
    return occupied


def _valid_orders_canonical(grid: np.ndarray) -> list[tuple[int, int]]:
    canonical, _was_transposed, creases = _canonicalize(grid)
    bg = _background_color(canonical, creases)
    motif_cells = _extract_motif_cells(canonical, bg, creases)
    assert motif_cells, "missing motif"

    orders = [(creases[0], creases[1]), (creases[1], creases[0])]
    valid: list[tuple[int, int]] = []
    for order in orders:
        occupied = _apply_order(canonical.shape, motif_cells, creases, order)
        if occupied is not None:
            valid.append(order)
    return valid


def _render_canonical(
    shape: tuple[int, int],
    bg: int,
    creases: tuple[int, int],
    occupied: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[:, creases[0]] = CREASE
    out[:, creases[1]] = CREASE
    for (r, c), color in occupied.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold the motif across two parallel grey creases in the only order that keeps all four copies valid.

    Input structure:
    - The grid contains exactly two parallel full grey crease lines, either both rows or both columns.
    - All other non-background pixels form one multicolor motif that stays entirely inside one of the three regions split by the creases.

    Rule:
    - Try the two possible unfold orders: mirror-copy the current motif set across one crease, then mirror-copy the enlarged set across the other crease.
    - Reject any order that sends a copy out of bounds, onto a crease, or onto an already occupied motif pixel.
    - Keep the unique legal order, draw all four motif copies, and preserve both grey crease lines.

    Key insight:
    - The two single reflections appear in both orders; only the doubly reflected copy changes side because two parallel reflections compose into a translation.
    """
    canonical, was_transposed, creases = _canonicalize(grid)
    bg = _background_color(canonical, creases)
    motif_cells = _extract_motif_cells(canonical, bg, creases)
    assert motif_cells, "missing motif"

    valid: list[dict[tuple[int, int], int]] = []
    for order in ((creases[0], creases[1]), (creases[1], creases[0])):
        occupied = _apply_order(canonical.shape, motif_cells, creases, order)
        if occupied is not None:
            valid.append(occupied)

    assert len(valid) == 1, "expected exactly one legal unfolding order"
    out = _render_canonical(canonical.shape, bg, creases, valid[0])
    return out.T.copy() if was_transposed else out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    target = random.randint(max(3, (height * width) // 2), height * width)
    occupied = {(random.randrange(height), random.randrange(width))}
    while len(occupied) < target:
        r, c = random.choice(tuple(occupied))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            occupied.add((nr, nc))

    rows = {r for r, _ in occupied}
    cols = {c for _, c in occupied}
    if len(rows) < 2 or len(cols) < 2:
        return _random_connected_mask(height, width)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in occupied:
        mask[r, c] = True
    return mask


def _make_sprite(height: int, width: int) -> np.ndarray:
    mask = _random_connected_mask(height, width)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    n_colors = random.randint(2, min(4, len(coords)))
    palette = random.sample(MOTIF_COLORS, n_colors)
    random.shuffle(coords)

    sprite = np.zeros((height, width), dtype=int)
    for idx, (r, c) in enumerate(coords):
        sprite[r, c] = palette[idx % n_colors]
    return sprite


def _place_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    patch = grid[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    mask = sprite != BACKGROUND
    patch[mask] = sprite[mask]


def _build_left_outer_plus_canonical() -> np.ndarray:
    while True:
        width = random.randint(2, 4)
        height = random.randint(2, 5)
        gap = random.randint(1, 2)
        dist = random.randint(width + gap, min(width + gap + 4, 8))
        left = random.randint(0, min(2 * dist - 1, 3))
        crease0 = left + width - 1 + gap
        crease1 = crease0 + dist
        right_margin = random.randint(1, 3)
        grid_w = left + 2 * width - 1 + 2 * gap + 2 * dist + right_margin
        if grid_w > 30:
            continue

        grid_h = random.randint(max(height + 2, 8), min(20, 30))
        top = random.randint(0, grid_h - height)
        sprite = _make_sprite(height, width)

        grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
        grid[:, crease0] = CREASE
        grid[:, crease1] = CREASE
        _place_sprite(grid, sprite, top, left)
        if _valid_orders_canonical(grid) == [(crease0, crease1)]:
            return grid


def _build_middle_plus_canonical() -> np.ndarray:
    while True:
        width = random.randint(2, 4)
        height = random.randint(2, 5)
        gap_left = random.randint(1, 2)
        left_margin = random.randint(0, 2)
        min_dist = max(gap_left + width, ((left_margin + width - 1 + 2 * gap_left) // 2) + 1)
        max_dist = min(min_dist + 4, 8)
        if min_dist > max_dist:
            continue
        dist = random.randint(min_dist, max_dist)
        crease0 = left_margin + width - 1 + gap_left
        crease1 = crease0 + dist
        left = crease0 + gap_left
        right_margin = random.randint(1, 3)
        grid_w = left + 2 * dist + width + right_margin
        if grid_w > 30:
            continue

        grid_h = random.randint(max(height + 2, 8), min(20, 30))
        top = random.randint(0, grid_h - height)
        sprite = _make_sprite(height, width)

        grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
        grid[:, crease0] = CREASE
        grid[:, crease1] = CREASE
        _place_sprite(grid, sprite, top, left)
        if _valid_orders_canonical(grid) == [(crease0, crease1)]:
            return grid


def generate_input() -> np.ndarray:
    families = ("left_outer_plus", "middle_plus", "middle_minus", "right_outer_minus")
    for _ in range(MAX_GEN_ATTEMPTS):
        family = random.choice(families)
        if family == "left_outer_plus":
            grid = _build_left_outer_plus_canonical()
        elif family == "middle_plus":
            grid = _build_middle_plus_canonical()
        elif family == "middle_minus":
            grid = np.fliplr(_build_middle_plus_canonical()).copy()
        else:
            grid = np.fliplr(_build_left_outer_plus_canonical()).copy()

        if random.random() < 0.5:
            grid = grid.T.copy()

        if len(_valid_orders_canonical(grid)) == 1:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid double-crease unfolding input")
