from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

TEMPLATE_SIZE = 4
LEGEND_SHAPE = (2, 1)
PLACEMENT_PADDING = 2
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _tight_crop(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    b = _bbox(mask)
    assert b is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _expand_mask(seed_mask: np.ndarray, fg_color: int, bg_color: int) -> np.ndarray:
    """One recursive self-similar expansion using the seed mask as both meta and inner pattern."""
    h, w = map(int, seed_mask.shape)
    out = np.zeros((h * h, w * w), dtype=int)
    for mr in range(h):
        for mc in range(w):
            block_color = fg_color if seed_mask[mr, mc] else bg_color
            block = np.zeros((h, w), dtype=int)
            block[seed_mask] = block_color
            out[mr * h : (mr + 1) * h, mc * w : (mc + 1) * w] = block
    return out


def _tile_crop(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    ph, pw = map(int, pattern.shape)
    reps = ((h + ph - 1) // ph, (w + pw - 1) // pw)
    return np.tile(pattern, reps)[:h, :w].copy()


def _find_template(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates: list[tuple[int, int, np.ndarray]] = []
    H, W = map(int, grid.shape)
    for r in range(H - TEMPLATE_SIZE + 1):
        for c in range(W - TEMPLATE_SIZE + 1):
            window = grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
            colors = set(map(int, np.unique(window)))
            if colors - {BLUE, RED}:
                continue
            if BLUE not in colors or RED not in colors:
                continue
            blue_mask = window == BLUE
            if int(blue_mask.sum()) < 4:
                continue
            if not _is_connected(blue_mask):
                continue
            candidates.append((r, c, window.copy()))
    assert len(candidates) == 1, "expected exactly one complete 4x4 red/blue template"
    return candidates[0]


def _find_legend(grid: np.ndarray) -> tuple[int, int]:
    green_cells = np.argwhere(grid == GREEN)
    yellow_cells = np.argwhere(grid == YELLOW)
    assert green_cells.shape == (1, 2), "expected exactly one green legend cell"
    assert yellow_cells.shape == (1, 2), "expected exactly one yellow legend cell"
    gr, gc = map(int, green_cells[0])
    yr, yc = map(int, yellow_cells[0])
    assert gc == yc and yr == gr + 1, "expected a vertical 2x1 legend bar"
    return GREEN, YELLOW


def _find_partial_origins(grid: np.ndarray, support_mask: np.ndarray, template_origin: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = map(int, grid.shape)
    partials: list[tuple[int, int]] = []
    for r in range(H - TEMPLATE_SIZE + 1):
        for c in range(W - TEMPLATE_SIZE + 1):
            if (r, c) == template_origin:
                continue
            window = grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
            if not np.all(window[support_mask] == RED):
                continue
            if not np.all(window[~support_mask] == BACKGROUND):
                continue
            partials.append((r, c))
    assert len(partials) >= 1, "expected at least one incomplete copy"
    return partials


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the full red/blue template's blue mask to generate a recursive fill pattern for every incomplete copy.

    Input structure:
    - A black background contains one complete 4x4 template made only of red and blue cells.
    - The blue cells mark the positions that are missing inside 2-3 other 4x4 red support copies, where those cells are black.
    - A vertical 2x1 legend bar supplies the recursive fill colors with green on top and yellow below.

    Rule:
    - Find the unique complete 4x4 red/blue template and extract its blue-cell mask.
    - Tight-crop that mask, expand it once self-similarly, and color the expanded copies green when the meta cell is blue and yellow when the meta cell is background.
    - Tile/crop the expanded pattern back to 4x4 and write those values only into the missing-mask cells of every incomplete copy.

    Key insight:
    - The blue cells are not output paint by themselves; their shape is a seed that recursively generates the green/yellow pattern used to fill every matching red support copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_r, template_c, template = _find_template(g)
    fg_color, bg_color = _find_legend(g)

    full_mask = template == BLUE
    seed_mask, _seed_offset = _tight_crop(full_mask)
    expanded = _expand_mask(seed_mask, fg_color, bg_color)
    fill_window = _tile_crop(expanded, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    support_mask = template == RED
    partials = _find_partial_origins(g, support_mask, (template_r, template_c))

    out = g.copy()
    for r, c in partials:
        patch = out[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
        patch[full_mask] = fill_window[full_mask]
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    if area <= 4:
        target = 4
    else:
        target = random.randint(4, area - 1)

    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = {start}

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nxt = random.choice(tuple(frontier))
        cells.add(nxt)
        mask[nxt] = True

    if mask.all():
        return None
    return mask


def _build_template() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        seed_h, seed_w = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
        seed_mask = _random_connected_mask(seed_h, seed_w)
        if seed_mask is None or int(seed_mask.sum()) < 4 or not _is_connected(seed_mask):
            continue

        full_mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        r0 = random.randint(0, TEMPLATE_SIZE - seed_h)
        c0 = random.randint(0, TEMPLATE_SIZE - seed_w)
        full_mask[r0 : r0 + seed_h, c0 : c0 + seed_w] = seed_mask

        for _ in range(random.randrange(4)):
            full_mask = np.rot90(full_mask)
        if random.random() < 0.5:
            full_mask = np.fliplr(full_mask)
        if random.random() < 0.5:
            full_mask = np.flipud(full_mask)

        if not _is_connected(full_mask):
            continue
        if int(full_mask.sum()) < 4 or int((~full_mask).sum()) < 4:
            continue

        tight_seed, _ = _tight_crop(full_mask)
        expanded = _expand_mask(tight_seed, GREEN, YELLOW)
        fill_window = _tile_crop(expanded, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        fill_colors = set(map(int, fill_window[full_mask]))
        if GREEN not in fill_colors or YELLOW not in fill_colors:
            continue

        template = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), RED, dtype=int)
        template[full_mask] = BLUE
        return template, fill_window

    raise RuntimeError("failed to build a non-degenerate template")


def _rectangles_far_apart(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _place_box(
    H: int,
    W: int,
    box_h: int,
    box_w: int,
    existing: list[tuple[int, int, int, int]],
    *,
    padding: int,
) -> tuple[int, int] | None:
    positions = [(r, c) for r in range(H - box_h + 1) for c in range(W - box_w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        rect = (r, c, r + box_h - 1, c + box_w - 1)
        if all(_rectangles_far_apart(rect, other, padding) for other in existing):
            return r, c
    return None


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _count_template_windows(grid: np.ndarray) -> int:
    H, W = map(int, grid.shape)
    total = 0
    for r in range(H - TEMPLATE_SIZE + 1):
        for c in range(W - TEMPLATE_SIZE + 1):
            window = grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
            colors = set(map(int, np.unique(window)))
            if colors <= {BLUE, RED} and BLUE in colors and RED in colors:
                mask = window == BLUE
                if int(mask.sum()) >= 4 and _is_connected(mask):
                    total += 1
    return total


def _count_partial_windows(grid: np.ndarray, template: np.ndarray, template_origin: tuple[int, int]) -> int:
    support_mask = template == RED
    H, W = map(int, grid.shape)
    total = 0
    for r in range(H - TEMPLATE_SIZE + 1):
        for c in range(W - TEMPLATE_SIZE + 1):
            if (r, c) == template_origin:
                continue
            window = grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
            if np.all(window[support_mask] == RED) and np.all(window[~support_mask] == BACKGROUND):
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(17, 24)
    W = random.randint(17, 24)
    grid = np.full((H, W), BACKGROUND, dtype=int)

    template, fill_window = _build_template()
    partial_count = random.randint(2, 3)

    legend_r, legend_c = H - LEGEND_SHAPE[0], W - LEGEND_SHAPE[1]
    grid[legend_r, legend_c] = GREEN
    grid[legend_r + 1, legend_c] = YELLOW

    placed_rects = [(legend_r, legend_c, legend_r + LEGEND_SHAPE[0] - 1, legend_c + LEGEND_SHAPE[1] - 1)]
    object_positions: list[tuple[int, int]] = []
    for _ in range(partial_count + 1):
        loc = _place_box(H, W, TEMPLATE_SIZE, TEMPLATE_SIZE, placed_rects, padding=PLACEMENT_PADDING)
        if loc is None:
            return None
        r, c = loc
        placed_rects.append((r, c, r + TEMPLATE_SIZE - 1, c + TEMPLATE_SIZE - 1))
        object_positions.append((r, c))

    random.shuffle(object_positions)
    template_origin = object_positions[0]
    partial_origins = object_positions[1:]

    tr, tc = template_origin
    grid[tr : tr + TEMPLATE_SIZE, tc : tc + TEMPLATE_SIZE] = template

    support_mask = template == RED
    missing_mask = template == BLUE
    partial = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), BACKGROUND, dtype=int)
    partial[support_mask] = RED
    for r, c in partial_origins:
        grid[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE] = partial

    if _count_template_windows(grid) != 1:
        return None
    if _count_partial_windows(grid, template, template_origin) != partial_count:
        return None

    expected = grid.copy()
    for r, c in partial_origins:
        patch = expected[r : r + TEMPLATE_SIZE, c : c + TEMPLATE_SIZE]
        patch[missing_mask] = fill_window[missing_mask]

    try:
        actual = _expected_output(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    output_new_cells = actual[(actual != grid) & (actual != BACKGROUND)]
    if GREEN not in output_new_cells or YELLOW not in output_new_cells:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
