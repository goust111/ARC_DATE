from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
FG = 7
RANK_COLORS = (1, 2, 3, 4, 5, 6)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _column_groups(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    occupied = np.any(grid != bg, axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < grid.shape[1]:
        if not occupied[c]:
            c += 1
            continue
        start = c
        while c < grid.shape[1] and occupied[c]:
            c += 1
        groups.append(grid[:, start:c].copy())
    return groups


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    degree = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            degree += 1
    return degree


def _endpoint_row(group: np.ndarray, bg: int, side: str) -> int:
    mask = group != bg
    col = 0 if side == "left" else group.shape[1] - 1
    rows = [
        r
        for r in range(group.shape[0])
        if mask[r, col] and _degree(mask, r, col) == 1
    ]
    assert len(rows) == 1, f"expected one {side} endpoint, got {rows}"
    return int(rows[0])


def _shift_group(group: np.ndarray, bg: int, dr: int) -> tuple[np.ndarray, int]:
    coords = np.argwhere(group != bg)
    assert coords.size > 0
    h = group.shape[0]
    min_row = int(coords[:, 0].min())
    max_row = int(coords[:, 0].max())
    actual = int(np.clip(dr, -min_row, h - 1 - max_row))
    shifted = np.full_like(group, bg)
    shifted[coords[:, 0] + actual, coords[:, 1]] = group[coords[:, 0], coords[:, 1]]
    return shifted, actual


def _pack_groups(groups: list[np.ndarray], bg: int) -> np.ndarray:
    assert groups
    h = groups[0].shape[0]
    assert all(group.shape[0] == h for group in groups)
    out = np.full((h, sum(group.shape[1] for group in groups)), bg, dtype=int)

    col = 0
    first = groups[0]
    out[:, : first.shape[1]] = first
    prev_right_row = _endpoint_row(first, bg, "right")
    col += first.shape[1]

    for group in groups[1:]:
        left_row = _endpoint_row(group, bg, "left")
        shifted, _actual_dr = _shift_group(group, bg, prev_right_row - left_row)
        out[:, col : col + group.shape[1]] = shifted
        prev_right_row = _endpoint_row(shifted, bg, "right")
        col += group.shape[1]

    return out


def _component_sort_key(coords: np.ndarray) -> tuple[int, int, int]:
    area = int(coords.shape[0])
    left_col = int(coords[:, 1].min())
    top_row = int(coords[:, 0].min())
    return area, left_col, top_row


def _recolor_ranked_components(packed: np.ndarray, bg: int) -> np.ndarray:
    components = _find_components(packed == FG)
    assert 1 <= len(components) <= len(RANK_COLORS)
    out = packed.copy()
    for color, coords in zip(RANK_COLORS, sorted(components, key=_component_sort_key)):
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove blank separator columns, pack orange path fragments by edge endpoints, then recolor the resulting components by area rank.

    Input structure:
    - The grid is a black horizontal strip containing 4 to 6 orange path fragments.
    - Each fragment occupies a contiguous run of non-background columns, and adjacent runs are separated by all-black columns.
    - Every fragment has exactly one degree-1 orange endpoint in its leftmost column and exactly one degree-1 orange endpoint in its rightmost column.

    Rule:
    - Delete every all-black separator column.
    - Keep the first fragment fixed. Process later fragments from left to right: vertically shift each fragment so its left endpoint row matches the previous placed fragment's right endpoint row, while keeping the fragment inside the strip height, and place it flush to the right.
    - After packing, find the orange 4-connected components. Sort them by area from small to large; break ties by the leftmost occupied column, then by the topmost occupied row.
    - Recolor component ranks 1, 2, 3, 4, 5, 6 as blue, red, green, yellow, gray, and magenta respectively.

    Key insight:
    - Endpoint-guided packing changes the real connectivity: some fragments fuse into larger components while others stay separate because vertical motion is constrained by the strip height, so ranking must be done after packing rather than on the input fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    groups = _column_groups(g, bg)
    assert 4 <= len(groups) <= 6
    packed = _pack_groups(groups, bg)
    return _recolor_ranked_components(packed, bg)


def _make_path_sprite(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        h = int(rng.integers(2, 6))
        w = int(rng.integers(3, 5))
        start_row = int(rng.integers(0, h))
        points = [(start_row, 0)]
        cur_row = start_row
        used_vertical = False

        for c in range(w - 1):
            points.append((cur_row, c + 1))
            if c + 1 == w - 1:
                continue

            if h == 1:
                target_row = cur_row
            elif rng.random() < 0.75:
                choices = [r for r in range(h) if r != cur_row]
                target_row = int(rng.choice(choices))
            else:
                target_row = cur_row

            step = 1 if target_row > cur_row else -1
            while cur_row != target_row:
                cur_row += step
                points.append((cur_row, c + 1))
                used_vertical = True

        if not used_vertical:
            continue

        sprite = np.full((h, w), BG, dtype=int)
        for r, c in points:
            sprite[r, c] = FG

        if not _is_valid_group(sprite):
            continue
        return sprite

    raise RuntimeError("failed to generate a valid path sprite")


def _is_valid_group(group: np.ndarray) -> bool:
    mask = group == FG
    if np.count_nonzero(mask) == 0:
        return False
    comps = _find_components(mask)
    if len(comps) != 1:
        return False
    rows, cols = np.where(mask)
    if int(rows.max() - rows.min() + 1) < 2:
        return False
    if int(cols.max() - cols.min() + 1) < 3:
        return False

    degrees = np.zeros_like(group, dtype=int)
    for r, c in np.argwhere(mask):
        degrees[r, c] = _degree(mask, int(r), int(c))
    if not np.all((degrees[mask] >= 1) & (degrees[mask] <= 2)):
        return False
    if int(np.sum(degrees[mask] == 1)) != 2:
        return False

    try:
        left_row = _endpoint_row(group, BG, "left")
        right_row = _endpoint_row(group, BG, "right")
    except AssertionError:
        return False

    left_count = int(np.count_nonzero(mask[:, 0]))
    right_count = int(np.count_nonzero(mask[:, -1]))
    return left_count == 1 and right_count == 1 and left_row >= 0 and right_row >= 0


def _build_input_grid(
    height: int,
    sprites: list[np.ndarray],
    tops: list[int],
    separators: list[int],
) -> np.ndarray:
    width = sum(sprite.shape[1] for sprite in sprites) + sum(separators)
    grid = np.full((height, width), BG, dtype=int)
    col = 0
    for i, (sprite, top) in enumerate(zip(sprites, tops)):
        h, w = sprite.shape
        grid[top : top + h, col : col + w] = np.maximum(grid[top : top + h, col : col + w], sprite)
        col += w
        if i < len(separators):
            col += separators[i]
    return grid


def _component_runs_after_packing(grid: np.ndarray) -> tuple[int, list[int]]:
    packed = _pack_groups(_column_groups(grid, BG), BG)
    components = _find_components(packed == FG)
    return len(components), sorted(int(comp.shape[0]) for comp in components)


def _input_is_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(np.isin(grid, (BG, FG))):
        return False
    groups = _column_groups(grid, BG)
    if not (4 <= len(groups) <= 6):
        return False
    if any(not _is_valid_group(group) for group in groups):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(7, 11))
    group_count = int(rng.integers(4, 7))
    sprites = [_make_path_sprite(rng) for _ in range(group_count)]

    base_width = sum(sprite.shape[1] for sprite in sprites) + (group_count - 1)
    if base_width > 30:
        return None

    extra_budget = 30 - base_width
    separators = [1] * (group_count - 1)
    for i in rng.permutation(group_count - 1):
        if extra_budget <= 0:
            break
        add = int(rng.integers(0, min(3, extra_budget) + 1))
        separators[i] += add
        extra_budget -= add

    tops = []
    for sprite in sprites:
        slack = height - sprite.shape[0]
        if slack < 0:
            return None
        tops.append(int(rng.integers(0, slack + 1)))

    grid = _build_input_grid(height, sprites, tops, separators)
    if not _input_is_valid(grid):
        return None

    component_count, component_areas = _component_runs_after_packing(grid)
    if component_count < 2 or component_count > group_count:
        return None

    mode = "separate" if rng.random() < 0.25 else "partial"
    if mode == "separate":
        if component_count != group_count or group_count < 5:
            return None
    else:
        if not (2 <= component_count < group_count):
            return None

    if len(set(component_areas)) == 1:
        return None

    out = main(grid)
    used_rank_colors = set(map(int, out[out != BG].ravel()))
    if len(used_rank_colors) != component_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
