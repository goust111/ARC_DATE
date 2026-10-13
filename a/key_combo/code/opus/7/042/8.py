from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MASK_SHAPE = (3, 3)
GRID_HEIGHT_RANGE = (10, 15)
GRID_WIDTH_RANGE = (12, 17)
LEGEND_COUNT_RANGE = (3, 4)
INSTANCE_COUNT_RANGE = (0, 2)
PLACEMENT_PADDING = 1
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_mask_library() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        mask = np.zeros(MASK_SHAPE, dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        cells = int(mask.sum())
        if not 3 <= cells <= 5:
            continue
        if not _is_connected(mask):
            continue
        if any(not mask[r, :].any() for r in range(3)):
            continue
        if any(not mask[:, c].any() for c in range(3)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        masks.append(mask)
    assert masks, "expected a nonempty mask library"
    return masks


MASK_LIBRARY = _build_mask_library()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool)


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = int(color)
    return patch


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _legend_candidates(grid: np.ndarray) -> list[tuple[int, int, list[int]]]:
    candidates: list[tuple[int, int, list[int]]] = []
    for coords in _components(grid != BACKGROUND):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r0 != r1:
            continue
        width = c1 - c0 + 1
        if width not in range(LEGEND_COUNT_RANGE[0], LEGEND_COUNT_RANGE[1] + 1):
            continue
        if len(coords) != width:
            continue
        colors = [int(v) for v in grid[r0, c0 : c1 + 1]]
        if any(color == BACKGROUND for color in colors):
            continue
        if len(set(colors)) != width:
            continue
        candidates.append((r0, c0, colors))
    return candidates


def _find_legend_order(grid: np.ndarray) -> list[int]:
    candidates = _legend_candidates(grid)
    assert candidates, "expected a horizontal legend"
    candidates.sort(key=lambda item: (item[0], item[1]))
    top_row = candidates[0][0]
    top_candidates = [item for item in candidates if item[0] == top_row]
    assert len(top_candidates) == 1, "expected one top legend candidate"
    return top_candidates[0][2]


def _build_output(order: list[int], template_mask: np.ndarray) -> np.ndarray:
    strip = np.concatenate([_paint_mask(template_mask, color) for color in order], axis=0)
    mirrored = np.fliplr(strip)
    return np.concatenate([strip, mirrored, strip, mirrored, strip], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble a legend-ordered sprite strip, then tile it horizontally with alternating left-right flips.

    Input structure:
    - The grid has a black background, one horizontal single-pixel color legend near the top, and one gray template sprite.
    - The legend is a contiguous row of 3 or 4 distinct non-black pixels read left-to-right.
    - The gray template is a 4-connected shape inside a 3x3 box, and optional colored copies of that same shape may also appear elsewhere.

    Rule:
    - Read the legend colors from left to right.
    - Crop the gray template to its bounding box and use that crop as a binary mask.
    - Recolor one copy of the template for each legend color and stack those copies vertically in legend order.
    - Use that stacked strip as a tile and build a five-tile horizontal wallpaper: original, left-right mirror, original, mirror, original.

    Key insight:
    - The legend fixes the vertical assembly order, while the gray template alone defines the tile shape that gets mirrored across alternating wallpaper slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    order = _find_legend_order(g)
    template_mask = _crop_mask(g == TEMPLATE_COLOR)
    assert template_mask.any(), "expected a gray template"
    return _build_output(order, template_mask)


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc0 = max(0, c0 - padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_patch(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    patch = _paint_mask(mask, color)
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w][patch != BACKGROUND] = patch[patch != BACKGROUND]


def _sample_positions(H: int, W: int, count: int, legend_width: int) -> tuple[int, list[tuple[int, int]]] | None:
    occupied = np.zeros((H, W), dtype=bool)
    legend_c = random.randint(0, W - legend_width)
    _mark_bbox(occupied, 0, legend_c, 1, legend_width)

    positions: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [(r, c) for r in range(2, H - 2) for c in range(0, W - 2)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            if _can_place_bbox(occupied, r0, c0, 3, 3, PLACEMENT_PADDING):
                positions.append((r0, c0))
                _mark_bbox(occupied, r0, c0, 3, 3)
                break
        else:
            return None
    return legend_c, positions


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    instance_count = random.randint(*INSTANCE_COUNT_RANGE)
    legend_colors = random.sample(PALETTE, legend_count)
    template_mask = random.choice(MASK_LIBRARY)

    sampled = _sample_positions(H, W, 1 + instance_count, legend_count)
    if sampled is None:
        return None
    legend_c, positions = sampled

    grid = np.full((H, W), BACKGROUND, dtype=int)
    grid[0, legend_c : legend_c + legend_count] = np.array(legend_colors, dtype=int)

    template_r, template_c = positions[0]
    _place_patch(grid, template_mask, TEMPLATE_COLOR, template_r, template_c)

    instance_colors = random.sample(legend_colors, instance_count)
    for (r0, c0), color in zip(positions[1:], instance_colors):
        _place_patch(grid, template_mask, int(color), r0, c0)

    if len(_legend_candidates(grid)) != 1:
        return None
    if not np.array_equal(main(grid), _build_output(legend_colors, template_mask)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid task 8 input")
