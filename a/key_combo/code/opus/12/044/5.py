from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 5
PANEL_SIZE = 7
GRID_SIZE = PANEL_SIZE * 2
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
PANEL_NAMES = ("UL", "UR", "LL", "LR")
OVERLAY_ORDER = (0, 3, 2, 1)  # UL, LR, LL, UR
TRIVIAL_ORDERS = {
    (0, 1, 2, 3),  # spatial reading order
    OVERLAY_ORDER,  # paint order
    tuple(reversed(OVERLAY_ORDER)),  # top-to-bottom priority
}
GEN_RETRIES = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _panel_slices(index: int) -> tuple[slice, slice]:
    row_block = 0 if index < 2 else 1
    col_block = 0 if index % 2 == 0 else 1
    return (
        slice(row_block * PANEL_SIZE, (row_block + 1) * PANEL_SIZE),
        slice(col_block * PANEL_SIZE, (col_block + 1) * PANEL_SIZE),
    )


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(v) for v in grid.ravel()}
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return sorted(colors)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                cell = (rr, cc)
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _compose_panels(panels: list[np.ndarray], bg: int) -> np.ndarray:
    out = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    for panel in OVERLAY_ORDER:
        mask = panels[panel] != bg
        out[mask] = panels[panel][mask]
    return out


def _visible_counts(panels: list[np.ndarray], colors: list[int], bg: int) -> list[int]:
    composite = _compose_panels(panels, bg)
    return [int(np.count_nonzero(composite == color)) for color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quadrant panels, count the visible color pixels, and compress the ranking into a single column.

    Input structure:
    - The grid is a 2x2 arrangement of equal 7x7 panels on a grey background.
    - Each panel contains exactly one distinct non-background foreground color.
    - Corresponding cells across panels represent the same location in the overlaid scene.

    Rule:
    - Split the input into upper-left, upper-right, lower-left, and lower-right panels.
    - Transparently overlay non-background pixels in the fixed order upper-left, lower-right, lower-left, then upper-right.
    - Count how many visible pixels remain for each of the four panel colors after occlusion.
    - Sort the four colors from largest visible count to smallest and output them as a 4x1 color column.

    Key insight:
    - The counts must be taken after positional layer occlusion, not from the raw panel areas, and those post-overlay counts determine the final order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 14x14 grid made of four 7x7 panels"

    bg = _get_bg(g)
    panels: list[np.ndarray] = []
    colors: list[int] = []
    for index in range(4):
        rs, cs = _panel_slices(index)
        panel = g[rs, cs]
        panel_colors = _colors_present(panel, exclude={bg})
        assert len(panel_colors) == 1, f"{PANEL_NAMES[index]} panel must contain exactly one foreground color"
        panels.append(panel.copy())
        colors.append(panel_colors[0])

    assert len(set(colors)) == 4, "the four panel colors must be distinct"

    counts = _visible_counts(panels, colors, bg)
    assert len(set(counts)) == 4, "visible counts must be unique for a determinate ordering"
    order = sorted(range(4), key=lambda idx: counts[idx], reverse=True)
    return np.array([[colors[idx]] for idx in order], dtype=int)


def _path_between(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    path = [(r, c)]
    if row_first:
        while r != rr:
            r += 1 if rr > r else -1
            path.append((r, c))
        while c != cc:
            c += 1 if cc > c else -1
            path.append((r, c))
    else:
        while c != cc:
            c += 1 if cc > c else -1
            path.append((r, c))
        while r != rr:
            r += 1 if rr > r else -1
            path.append((r, c))
    return path


def _connected_seed(required: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    ordered = list(required)
    rng.shuffle(ordered)
    shape = {ordered[0]}
    for target in ordered[1:]:
        source = min(shape, key=lambda cell: abs(cell[0] - target[0]) + abs(cell[1] - target[1]))
        shape.update(_path_between(source, target, row_first=rng.random() < 0.5))
    return shape


def _grow_mask(
    area: int,
    required: set[tuple[int, int]],
    bias: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    shape = _connected_seed(required, rng)
    if len(shape) > area:
        return None

    while len(shape) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < PANEL_SIZE and 0 <= cc < PANEL_SIZE and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        frontier = sorted(set(frontier))
        weights = []
        for rr, cc in frontier:
            dist = abs(rr - bias[0]) + abs(cc - bias[1])
            edge_bonus = 0.65 if rr in (0, PANEL_SIZE - 1) or cc in (0, PANEL_SIZE - 1) else 0.0
            weights.append(1.0 / (1.0 + dist) + edge_bonus + rng.random() * 0.3)
        shape.add(rng.choices(frontier, weights=weights, k=1)[0])

    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    if not _is_connected(mask):
        return None
    return mask


def _build_input_from_masks(masks: list[np.ndarray], colors: list[int]) -> list[np.ndarray]:
    panels: list[np.ndarray] = []
    for mask, color in zip(masks, colors):
        panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
        panel[mask] = color
        panels.append(panel)
    return panels


def _overlap_pair_count(masks: list[np.ndarray]) -> int:
    total = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if np.any(masks[i] & masks[j]):
                total += 1
    return total


def _semantic_checks(masks: list[np.ndarray], colors: list[int]) -> bool:
    raw_counts = [int(mask.sum()) for mask in masks]
    if len(set(raw_counts)) != 4:
        return False
    if any(not (3 <= count <= 12) for count in raw_counts):
        return False
    if any(not _is_connected(mask) for mask in masks):
        return False

    overlap_pairs = _overlap_pair_count(masks)
    if overlap_pairs < 2:
        return False

    stack = sum(mask.astype(int) for mask in masks)
    if int(np.count_nonzero(stack >= 3)) > 5:
        return False
    if int(np.count_nonzero(stack == 4)) > 1:
        return False

    panels = _build_input_from_masks(masks, colors)
    visible_counts = _visible_counts(panels, colors, BG)
    if len(set(visible_counts)) != 4:
        return False

    hidden_counts = [raw - visible for raw, visible in zip(raw_counts, visible_counts)]
    if sum(hidden > 0 for hidden in hidden_counts) < 2:
        return False

    visible_order = tuple(sorted(range(4), key=lambda idx: visible_counts[idx], reverse=True))
    raw_order = tuple(sorted(range(4), key=lambda idx: raw_counts[idx], reverse=True))
    if visible_order == raw_order:
        return False
    if visible_order in TRIVIAL_ORDERS:
        return False

    union_area = int(np.count_nonzero(stack > 0))
    if not (9 <= union_area <= 22):
        return False

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for index, panel in enumerate(panels):
        rs, cs = _panel_slices(index)
        grid[rs, cs] = panel

    try:
        out = main(grid)
    except AssertionError:
        return False
    expected = np.array([[colors[idx]] for idx in visible_order], dtype=int)
    return np.array_equal(out, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    colors = random.sample(PALETTE, 4)

    shared_pool = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
    rng.shuffle(shared_pool)
    required: list[set[tuple[int, int]]] = [set() for _ in range(4)]

    pair1_cell, pair2_cell = shared_pool[:2]
    required[0].add(pair1_cell)
    required[3].add(pair1_cell)
    required[2].add(pair2_cell)
    required[1].add(pair2_cell)

    optional_pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]
    extra_cells = shared_pool[2:]
    rng.shuffle(optional_pairs)
    for cell, pair in zip(extra_cells[: rng.randint(1, 3)], optional_pairs):
        a, b = pair
        required[a].add(cell)
        required[b].add(cell)

    base_biases = [(2, 2), (2, 4), (4, 2), (4, 4)]
    biases = []
    for r, c in base_biases:
        rr = min(PANEL_SIZE - 1, max(0, r + rng.choice((-1, 0, 1))))
        cc = min(PANEL_SIZE - 1, max(0, c + rng.choice((-1, 0, 1))))
        biases.append((rr, cc))

    areas = rng.sample(list(range(3, 13)), 4)
    masks: list[np.ndarray] = []
    for index in range(4):
        mask = _grow_mask(areas[index], required[index], biases[index], rng)
        if mask is None:
            return None
        masks.append(mask)

    if not _semantic_checks(masks, colors):
        return None

    grid = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    for index, panel in enumerate(_build_input_from_masks(masks, colors)):
        rs, cs = _panel_slices(index)
        grid[rs, cs] = panel
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
