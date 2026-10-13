import random
from collections import deque

import numpy as np


BG = 0
DIVIDER_COLOR = 5
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_colors(right_panel: np.ndarray) -> list[int]:
    comps = _components(right_panel != BG)
    assert comps, "expected legend blocks in the right panel"

    blocks: list[tuple[int, int, int]] = []
    covered = np.zeros_like(right_panel, dtype=bool)
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(right_panel[comp[0]])
        assert color != BG, "legend component must be colored"
        assert len(comp) == 9, "legend block must be 3x3"
        assert r1 - r0 == 2 and c1 - c0 == 2, "legend block must have 3x3 bbox"
        assert np.all(right_panel[r0 : r1 + 1, c0 : c1 + 1] == color), "legend block must be solid"
        blocks.append((r0, c0, color))
        covered[r0 : r1 + 1, c0 : c1 + 1] |= right_panel[r0 : r1 + 1, c0 : c1 + 1] != BG

    assert np.array_equal(covered, right_panel != BG), "right panel should only contain legend blocks"
    blocks.sort()
    colors = [color for _, _, color in blocks]
    assert len(colors) == len(set(colors)), "legend block colors must be distinct"
    return colors


def _find_divider_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        if not np.all(grid[:, c] == grid[0, c]):
            continue
        if int(grid[0, c]) == BG:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0 or not np.any(left != BG):
            continue
        try:
            _legend_colors(right)
        except AssertionError:
            continue
        candidates.append(c)
    assert len(candidates) == 1, "expected a unique vertical divider"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the left panel by the right-side color legend, then cyclically rotate the kept colors.

    Input structure:
    - The grid is split by one full-height grey divider column into left and right panels.
    - The left panel has black background with colored pixels forming patterns and noise.
    - The right panel has black background and several solid 3x3 monochrome blocks.
    - Those right-panel blocks are arranged from top to bottom, and their colors are a subset of the left-panel colors.

    Rule:
    - Read the right-panel 3x3 blocks from top to bottom to obtain an ordered legend list L.
    - Remove every left-panel pixel whose color is not in L by turning it into black.
    - For each remaining left-panel pixel, replace its color with the next legend color in L, wrapping the last color back to the first.
    - Drop the divider and the whole right panel; output only the transformed left panel.

    Key insight:
    - The right panel is both a visibility filter and an ordered cyclic recoloring instruction, with the vertical block order defining the rotation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider_col = _find_divider_column(g)
    left = g[:, :divider_col]
    right = g[:, divider_col + 1 :]
    legend = _legend_colors(right)

    rotation = {color: legend[(idx + 1) % len(legend)] for idx, color in enumerate(legend)}
    out = np.full(left.shape, BG, dtype=int)
    for src, dst in rotation.items():
        out[left == src] = dst
    return out


GEN_ATTEMPTS = 300
LEGEND_COUNT_RANGE = (2, 5)
LEFT_HEIGHT_RANGE = (12, 24)
LEFT_WIDTH_RANGE = (10, 18)
RIGHT_WIDTH_RANGE = (3, 7)
TOP_PAD_RANGE = (0, 3)
BOTTOM_PAD_RANGE = (0, 3)
BLOCK_GAP_RANGE = (1, 3)
EXTRA_COLOR_COUNT_RANGE = (1, 3)
MANDATORY_AREA_RANGE = (3, 8)
EXTRA_AREA_RANGE = (1, 6)
EXTRA_OBJECT_COUNT_RANGE = (2, 7)
MASK_SIDE_RANGE = (1, 4)
PLACEMENT_ATTEMPTS = 80


def _random_connected_mask(min_area: int, max_area: int) -> np.ndarray:
    for _ in range(80):
        h = random.randint(*MASK_SIDE_RANGE)
        w = random.randint(*MASK_SIDE_RANGE)
        upper = min(max_area, h * w)
        if upper < min_area:
            continue
        target = random.randint(min_area, upper)
        cells = {(random.randrange(h), random.randrange(w))}
        frontier = set(cells)
        while len(cells) < target:
            if not frontier:
                break
            cr, cc = random.choice(tuple(frontier))
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.remove((cr, cc))
                continue
            cell = random.choice(neighbors)
            cells.add(cell)
            frontier.add(cell)
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        return mask
    raise RuntimeError("failed to build a connected mask")


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = mask.shape
    if top + h > occupied.shape[0] or left + w > occupied.shape[1]:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _place_object(grid: np.ndarray, occupied: np.ndarray, color: int, mask: np.ndarray) -> bool:
    max_r = grid.shape[0] - mask.shape[0]
    max_c = grid.shape[1] - mask.shape[1]
    positions = [(r, c) for r in range(max_r + 1) for c in range(max_c + 1)]
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_ATTEMPTS]:
        if not _can_place(occupied, mask, top, left):
            continue
        region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
        region[mask] = color
        occupied[top : top + mask.shape[0], left : left + mask.shape[1]] |= mask
        return True
    return False


def _build_left_panel(height: int, width: int, legend_colors: list[int], extra_colors: list[int]) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for color in legend_colors:
        mask = _random_connected_mask(*MANDATORY_AREA_RANGE)
        if not _place_object(grid, occupied, color, mask):
            return None

    for color in extra_colors:
        mask = _random_connected_mask(2, EXTRA_AREA_RANGE[1])
        if not _place_object(grid, occupied, color, mask):
            return None

    all_colors = legend_colors + extra_colors
    extra_count = random.randint(*EXTRA_OBJECT_COUNT_RANGE)
    for _ in range(extra_count):
        color = random.choice(all_colors)
        min_area = EXTRA_AREA_RANGE[0]
        max_area = EXTRA_AREA_RANGE[1]
        mask = _random_connected_mask(min_area, max_area)
        _place_object(grid, occupied, color, mask)

    for color in legend_colors:
        assert np.any(grid == color), "every legend color must appear in the left panel"
    for color in extra_colors:
        assert np.any(grid == color), "every distractor color must appear in the left panel"
    return grid


def _build_right_panel(height: int, width: int, legend_colors: list[int]) -> np.ndarray | None:
    panel = np.full((height, width), BG, dtype=int)
    required = 3 * len(legend_colors) + (len(legend_colors) - 1)
    if required > height:
        return None

    slack = height - required
    top_pad = random.randint(0, min(TOP_PAD_RANGE[1], slack))
    bottom_cap = min(BOTTOM_PAD_RANGE[1], slack - top_pad)
    bottom_pad = random.randint(0, max(0, bottom_cap))
    remaining = height - top_pad - bottom_pad - 3 * len(legend_colors)
    gap_count = len(legend_colors) - 1
    gaps = [1] * gap_count
    for _ in range(remaining - gap_count):
        gaps[random.randrange(gap_count)] += 1

    row = top_pad
    starts = []
    for idx in range(len(legend_colors)):
        starts.append(row)
        row += 3
        if idx < gap_count:
            row += gaps[idx]

    for start, color in zip(starts, legend_colors):
        col = random.randint(0, width - 3)
        panel[start : start + 3, col : col + 3] = color

    _legend_colors(panel)
    return panel


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    palette = random.sample(AVAILABLE_COLORS, legend_count + random.randint(*EXTRA_COLOR_COUNT_RANGE))
    legend_colors = palette[:legend_count]
    extra_colors = palette[legend_count:]

    right_w = random.randint(*RIGHT_WIDTH_RANGE)
    max_left_w = min(LEFT_WIDTH_RANGE[1], 30 - 1 - right_w)
    if max_left_w < LEFT_WIDTH_RANGE[0]:
        return None
    left_w = random.randint(LEFT_WIDTH_RANGE[0], max_left_w)

    min_h = max(LEFT_HEIGHT_RANGE[0], 3 * legend_count + (legend_count - 1))
    max_h = LEFT_HEIGHT_RANGE[1]
    if min_h > max_h:
        return None
    height = random.randint(min_h, max_h)

    left = _build_left_panel(height, left_w, legend_colors, extra_colors)
    if left is None:
        return None
    right = _build_right_panel(height, right_w, legend_colors)
    if right is None:
        return None

    grid = np.full((height, left_w + 1 + right_w), BG, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = DIVIDER_COLOR
    grid[:, left_w + 1 :] = right

    if len(set(legend_colors)) != len(legend_colors):
        return None
    if not all(np.any(left == color) for color in legend_colors):
        return None
    if not any(np.any(left == color) for color in extra_colors):
        return None
    if np.any(left == DIVIDER_COLOR) or np.any(right == DIVIDER_COLOR):
        return None

    expected = np.full(left.shape, BG, dtype=int)
    for idx, color in enumerate(legend_colors):
        expected[left == color] = legend_colors[(idx + 1) % len(legend_colors)]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
