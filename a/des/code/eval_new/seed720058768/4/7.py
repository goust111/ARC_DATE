from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rule_tile(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int] | None:
    if int(coords.shape[0]) != 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 4):
        return None

    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(block == BG):
        return None

    left = block[:, :2]
    right = block[:, 2:]
    if not np.all(left == left[0, 0]):
        return None
    if not np.all(right == right[0, 0]):
        return None

    return int(left[0, 0]), int(right[0, 0])


def _extract_rules(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    fg_mask = grid != BG
    rule_mask = np.zeros_like(fg_mask, dtype=bool)
    mapping: dict[int, int] = {}
    for coords in _components(fg_mask):
        tile = _rule_tile(grid, coords)
        if tile is None:
            continue
        src, dst = tile
        if src in mapping and mapping[src] != dst:
            raise AssertionError("conflicting rule tiles for one source color")
        mapping[src] = dst
        rule_mask[coords[:, 0], coords[:, 1]] = True
    return mapping, rule_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Infer a color-remapping table from detached 2x4 rule tiles and apply it to the main motif.

    Input structure:
    - Background is black.
    - One large irregular multicolor motif occupies the central part of the grid.
    - Several detached rule tiles lie in the surrounding background; each tile is a full 2x4 rectangle made of two touching 2x2 solid color blocks.

    Rule:
    - Read every rule tile left-to-right: the left 2x2 color A maps to the right 2x2 color B.
    - Remove all rule tiles by turning them into background.
    - Recolor every remaining non-background pixel by looking up its original color in the mapping.
    - Apply all recolorings simultaneously from the original input colors, so cycles do not cascade.

    Key insight:
    - The small legend tiles are the only perfectly filled 2x4 rectangles, so they can be isolated structurally and interpreted as a lookup table for the larger irregular motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, rule_mask = _extract_rules(g)
    assert mapping, "expected at least one rule tile"

    out = g.copy()
    out[rule_mask] = BG

    main_mask = (g != BG) & ~rule_mask
    for src, dst in mapping.items():
        out[main_mask & (g == src)] = dst

    return out


def _random_derangement(colors: list[int]) -> list[int]:
    if len(colors) == 2:
        return colors[::-1]
    while True:
        shuffled = colors[:]
        random.shuffle(shuffled)
        if all(a != b for a, b in zip(colors, shuffled)):
            return shuffled


def _grow_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        cells = np.argwhere(mask)
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((int(r), int(c)))
                    break
        if not frontier:
            break
        r, c = random.choice(frontier)
        neighbors = []
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            mask[nr, nc] = True

    return mask


def _seed_cells(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    for _ in range(60):
        seeds = random.sample(cells, count)
        if count == 1:
            return seeds
        dists = [
            abs(r1 - r2) + abs(c1 - c2)
            for i, (r1, c1) in enumerate(seeds)
            for r2, c2 in seeds[i + 1 :]
        ]
        if min(dists) >= 2:
            return seeds
    return random.sample(cells, count)


def _colorize_mask(mask: np.ndarray, colors: list[int]) -> np.ndarray | None:
    seeds = _seed_cells(mask, len(colors))
    seed_order = list(range(len(colors)))
    random.shuffle(seed_order)

    owner = -np.ones(mask.shape, dtype=int)
    queue: deque[tuple[int, int, int]] = deque()
    for idx in seed_order:
        r, c = seeds[idx]
        owner[r, c] = idx
        queue.append((r, c, idx))

    while queue:
        r, c, idx = queue.popleft()
        neighbors = []
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and owner[nr, nc] == -1:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        for nr, nc in neighbors:
            owner[nr, nc] = idx
            queue.append((nr, nc, idx))

    if np.any(owner[mask] < 0):
        return None

    counts = [int(np.count_nonzero(owner == idx)) for idx in range(len(colors))]
    if min(counts) < 2:
        return None

    colored = np.zeros(mask.shape, dtype=int)
    for idx, color in enumerate(colors):
        colored[owner == idx] = int(color)
    return colored


def _main_object(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    area = height * width
    min_area = max(14, area // 3, len(colors) * 3)
    max_area = min(area - 1, (area * 3) // 5)
    if min_area > max_area:
        return None

    for _ in range(80):
        target_area = random.randint(min_area, max_area)
        mask = _grow_mask(height, width, target_area)
        coords = np.argwhere(mask)
        if int(coords.shape[0]) != target_area:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        colored = _colorize_mask(mask, colors)
        if colored is None:
            continue
        return colored
    return None


def _can_place_rule(occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = map(int, occupied.shape)
    if top < 0 or left < 0 or top + 2 > h or left + 4 > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + 3)
    c0 = max(0, left - 1)
    c1 = min(w, left + 5)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rule_for_side(
    grid: np.ndarray,
    occupied: np.ndarray,
    side: str,
    src: int,
    dst: int,
    main_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = main_bbox

    if side == "top":
        row_range = range(0, max(0, r0 - 2))
        col_range = range(0, w - 3)
    elif side == "bottom":
        row_range = range(r1 + 2, h - 1)
        col_range = range(0, w - 3)
    elif side == "left":
        row_range = range(0, h - 1)
        col_range = range(0, max(0, c0 - 4))
    else:
        row_range = range(0, h - 1)
        col_range = range(c1 + 2, w - 3)

    if not row_range or not col_range:
        return False

    positions = [(r, c) for r in row_range for c in col_range]
    random.shuffle(positions)
    for top, left in positions[:200]:
        if not _can_place_rule(occupied, int(top), int(left)):
            continue
        grid[top : top + 2, left : left + 2] = src
        grid[top : top + 2, left + 2 : left + 4] = dst
        occupied[top : top + 2, left : left + 4] = True
        return True

    return False


def _expected_output(grid: np.ndarray, mapping: dict[int, int], rule_mask: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[rule_mask] = BG
    main_mask = (grid != BG) & ~rule_mask
    for src, dst in mapping.items():
        out[main_mask & (grid == src)] = dst
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(2, 4)
    src_colors = random.sample(PALETTE, color_count)

    cycle_case = bool(random.getrandbits(1))
    if cycle_case:
        dst_colors = _random_derangement(src_colors)
    else:
        dst_colors = random.sample([c for c in PALETTE if c not in src_colors], color_count)
    mapping = dict(zip(src_colors, dst_colors))

    motif_h = random.randint(6, 11)
    motif_w = random.randint(6, 11)
    grid_h = random.randint(max(16, motif_h + 8), 30)
    grid_w = random.randint(max(18, motif_w + 10), 30)

    motif = _main_object(motif_h, motif_w, src_colors)
    if motif is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    base_top = (grid_h - motif_h) // 2
    base_left = (grid_w - motif_w) // 2
    top = min(max(base_top + random.randint(-2, 2), 4), grid_h - motif_h - 4)
    left = min(max(base_left + random.randint(-3, 3), 5), grid_w - motif_w - 5)

    motif_mask = motif != BG
    grid[top : top + motif_h, left : left + motif_w][motif_mask] = motif[motif_mask]
    occupied[top : top + motif_h, left : left + motif_w][motif_mask] = True

    placed_coords = np.argwhere(grid != BG)
    main_bbox = _bbox(placed_coords)

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    chosen_sides = sides[:color_count]
    items = list(mapping.items())
    random.shuffle(items)

    for side, (src, dst) in zip(chosen_sides, items):
        if not _place_rule_for_side(grid, occupied, side, int(src), int(dst), main_bbox):
            return None

    rule_mapping, rule_mask = _extract_rules(grid)
    if rule_mapping != mapping:
        return None

    non_rule_mask = (grid != BG) & ~rule_mask
    comps = _components(non_rule_mask)
    if len(comps) != 1:
        return None

    remaining_colors = set(map(int, np.unique(grid[non_rule_mask])))
    if remaining_colors != set(src_colors):
        return None

    expected = _expected_output(grid, mapping, rule_mask)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
