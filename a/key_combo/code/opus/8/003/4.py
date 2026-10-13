from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

RANK_COLORS = [RED, BLUE, GREEN, YELLOW, GREY, PINK, MAROON]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_cells(shape: tuple[int, int], cells: list[tuple[int, int]] | set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    pts = np.argwhere(mask)
    for r, c in pts:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    deg = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            deg += 1
    return deg


def _has_full_2x2(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(mask[r : r + 2, c : c + 2]):
                return True
    return False


def _is_open_path(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if len(comps) != 1:
        return False
    if _has_full_2x2(mask):
        return False
    endpoint_count = 0
    for r, c in comps[0]:
        deg = _degree(mask, r, c)
        if deg not in (1, 2):
            return False
        if deg == 1:
            endpoint_count += 1
    return endpoint_count == 2


def _is_thin_channel(mask: np.ndarray) -> bool:
    comps = _components(mask)
    if len(comps) != 1 or not comps[0]:
        return False
    if _has_full_2x2(mask):
        return False
    for r, c in comps[0]:
        if _degree(mask, r, c) > 2:
            return False
    return True


def _select_top_channel_component(intersection: np.ndarray) -> list[tuple[int, int]]:
    candidates = [comp for comp in _components(intersection) if any(r == 0 for r, _ in comp)]
    if not candidates:
        raise ValueError("no top-connected channel component found")
    candidates.sort(key=lambda comp: (-len(comp), _bbox(comp)[0], _bbox(comp)[1]))
    return candidates[0]


def _extract_channel(path_mask: np.ndarray) -> np.ndarray:
    path_comps = _components(path_mask)
    if len(path_comps) != 2:
        raise ValueError("expected exactly two orange path components")
    a = _mask_from_cells(path_mask.shape, path_comps[0])
    b = _mask_from_cells(path_mask.shape, path_comps[1])
    intersection = _dilate8(a) & _dilate8(b) & (~path_mask)
    channel_cells = _select_top_channel_component(intersection)
    return _mask_from_cells(path_mask.shape, channel_cells)


def _region_order(bg_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components(bg_mask)
    comps.sort(key=lambda comp: (-len(comp), _bbox(comp)[0], _bbox(comp)[1]))
    return comps


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Find the top-connected overlap channel between two orange paths, treat it as a new wall, and recolor the resulting black regions by area rank.

    Input structure:
    - The grid is mostly black background.
    - Exactly two orange one-cell-wide 4-connected path components run roughly from the upper part toward the lower part of the grid with a narrow gap between them.
    - The paths are open and do not form closed loops on their own.

    Rule:
    - Dilate each orange path by one step in the 8-neighborhood and intersect the two dilations.
    - Among the overlap components, select the component that is connected to the top edge and recolor it teal; this teal component then counts as foreground boundary together with the orange paths.
    - Recompute the black 4-connected background regions under this new foreground partition and rank them by area, breaking ties by the upper-left corner of each region's bounding box.
    - Recolor the largest region red, the second blue, the third green, the fourth yellow, the fifth grey, and continue with the remaining fixed rank palette while keeping the orange paths and teal channel unchanged.

    Key insight:
    - The overlap channel is not just a highlight: once promoted to foreground, it changes how the background is partitioned, so the area ranking must be applied after inserting that channel wall.
    """
    grid = np.asarray(input_grid, dtype=int)
    orange_mask = grid == ORANGE
    channel_mask = _extract_channel(orange_mask)

    out = grid.copy()
    out[channel_mask] = TEAL

    black_regions = _region_order(out == BG)
    for idx, comp in enumerate(black_regions):
        color = RANK_COLORS[idx] if idx < len(RANK_COLORS) else RANK_COLORS[-1]
        for r, c in comp:
            out[r, c] = color
    return out


def _path_from_columns(columns: list[int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    h = len(columns)
    for r in range(h - 1):
        c0 = columns[r]
        c1 = columns[r + 1]
        cells.add((r, c0))
        if c1 != c0:
            step = 1 if c1 > c0 else -1
            for c in range(c0, c1 + step, step):
                cells.add((r + 1, c))
    cells.add((h - 1, columns[-1]))
    return cells


def _sample_column_paths(h: int, w: int) -> tuple[list[int], list[int]]:
    left = [random.randint(1, w - 8)]
    last_left_move = 0
    left_cooldown = 0
    for _ in range(h - 1):
        move = 0
        if left_cooldown == 0 and random.random() < 0.35:
            choices = [-1, 1]
            if last_left_move != 0:
                choices = [d for d in choices if d != -last_left_move]
            move = random.choice(choices)
            last_left_move = move
            left_cooldown = 1
        else:
            left_cooldown = max(0, left_cooldown - 1)
        left.append(max(1, min(w - 8, left[-1] + move)))

    gap = random.choice([3, 4, 5])
    right = [left[0] + gap]
    last_gap_move = 0
    gap_cooldown = 0
    for r in range(1, h):
        gap_move = 0
        if gap_cooldown == 0 and random.random() < 0.30:
            choices = [-1, 1]
            if last_gap_move != 0:
                choices = [d for d in choices if d != -last_gap_move]
            gap_move = random.choice(choices)
            last_gap_move = gap_move
            gap_cooldown = 1
        else:
            gap_cooldown = max(0, gap_cooldown - 1)
        gap = max(3, min(5, gap + gap_move))
        if left[r] + gap > w - 2:
            shift = left[r] + gap - (w - 2)
            left = [max(1, c - shift) for c in left]
        right.append(left[r] + gap)
    return left, right


def _valid_generated_grid(grid: np.ndarray) -> bool:
    orange_mask = grid == ORANGE
    orange_comps = _components(orange_mask)
    if len(orange_comps) != 2:
        return False
    if not all(_is_open_path(_mask_from_cells(grid.shape, comp)) for comp in orange_comps):
        return False

    try:
        channel_mask = _extract_channel(orange_mask)
    except ValueError:
        return False
    if not _is_thin_channel(channel_mask):
        return False
    if not np.any(channel_mask[0]):
        return False

    fg = orange_mask | channel_mask
    regions = _region_order(~fg)
    areas = [len(comp) for comp in regions]
    if len(areas) < 3 or len(set(areas)) < 3:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return (
        out.shape == grid.shape
        and np.count_nonzero(out == TEAL) == np.count_nonzero(channel_mask)
        and len(set(np.unique(out)) & {RED, BLUE, GREEN}) == 3
    )


def generate_input() -> np.ndarray:
    for _ in range(1500):
        h = random.randint(16, 24)
        w = random.randint(14, 24)
        left_cols, right_cols = _sample_column_paths(h, w)
        left_path = _path_from_columns(left_cols)
        right_path = _path_from_columns(right_cols)
        if left_path & right_path:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r, c in left_path | right_path:
            grid[r, c] = ORANGE

        if _valid_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
