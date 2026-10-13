from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _solid_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    solid_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r] == int(grid[r, 0]))
    ]
    solid_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == int(grid[0, c]))
    ]
    matches = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(matches) == 1, "expected exactly one solid same-color separator cross"
    return matches[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _hole_count(mask: np.ndarray) -> int:
    padded_bg = np.pad(~mask, 1, constant_values=True)
    comps = _components(padded_bg)
    outside = None
    holes = 0
    for idx, coords in enumerate(comps):
        if np.any((coords[:, 0] == 0) & (coords[:, 1] == 0)):
            outside = idx
            break
    assert outside is not None, "padded exterior must exist"
    for idx, _coords in enumerate(comps):
        if idx != outside:
            holes += 1
    return holes


def _extract_quadrants(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    bg = _most_common_color(grid)
    cross_r, cross_c, _sep = _solid_cross(grid, bg)
    top_h = cross_r
    left_w = cross_c
    assert top_h > 0 and left_w > 0
    assert grid.shape[0] - cross_r - 1 == top_h
    assert grid.shape[1] - cross_c - 1 == left_w
    quadrants = [
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 :],
        grid[cross_r + 1 :, :cross_c],
        grid[cross_r + 1 :, cross_c + 1 :],
    ]
    return bg, _sep, quadrants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four separator-cross quadrants onto one canvas, using hole count as the layer order.

    Input structure:
    - A solid one-row and one-column separator cross of the same non-background color splits the grid into four equal quadrants.
    - Each quadrant contains background plus one distinct foreground color.
    - The foreground cells in each quadrant form one connected object with one or more enclosed background holes.

    Rule:
    - Remove the separator cross and align the four quadrants onto a single quadrant-sized output canvas.
    - Count the enclosed background holes of each quadrant object.
    - For every output cell, among the quadrants whose aligned cell is foreground, keep the color from the quadrant with the largest hole count; if none are foreground, keep the background.

    Key insight:
    - The visible stacking order is not tied to quadrant position; it is encoded by each object's own number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, _sep, quadrants = _extract_quadrants(g)
    info = []
    for quadrant in quadrants:
        colors = [int(x) for x in np.unique(quadrant) if int(x) != bg]
        assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
        color = colors[0]
        mask = quadrant == color
        comps = _components(mask)
        assert len(comps) == 1, "each quadrant object must be connected"
        holes = _hole_count(mask)
        assert holes > 0, "each quadrant object must contain at least one enclosed hole"
        info.append((holes, color, mask))

    hole_counts = [holes for holes, _color, _mask in info]
    assert len(set(hole_counts)) == 4, "hole counts must be pairwise distinct"

    out = np.full(quadrants[0].shape, bg, dtype=int)
    for holes, color, mask in sorted(info, key=lambda item: item[0]):
        out[mask] = color
    return out


_BG = 0
_SIZE_RANGE = (10, 13)
_MARGIN_CHOICES = (1, 2)
_HOLE_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))
_COLOR_POOL = tuple(range(1, 10))
_MAX_TRIES = 400


def _choose_layout(q: int) -> tuple[int, int, int, int]:
    for _ in range(200):
        top = random.choice(_MARGIN_CHOICES)
        left = random.choice(_MARGIN_CHOICES)
        bottom = random.choice(_MARGIN_CHOICES)
        right = random.choice(_MARGIN_CHOICES)
        h = q - top - bottom
        w = q - left - right
        if h >= 7 and w >= 7:
            return top, left, h, w
    raise RuntimeError("failed to choose a feasible outer rectangle")


def _choose_hole_boxes(h: int, w: int) -> list[tuple[int, int, int, int]]:
    for _ in range(300):
        tl_h, tl_w = random.choice(_HOLE_SHAPES)
        tr_h, tr_w = random.choice(_HOLE_SHAPES)
        bl_h, bl_w = random.choice(_HOLE_SHAPES)
        br_h, br_w = random.choice(_HOLE_SHAPES)

        top_band = max(tl_h, tr_h)
        bottom_band = max(bl_h, br_h)
        left_band = max(tl_w, bl_w)
        right_band = max(tr_w, br_w)
        if h - top_band - bottom_band < 3 or w - left_band - right_band < 3:
            continue

        split_r = random.randint(top_band + 1, h - bottom_band - 2)
        split_c = random.randint(left_band + 1, w - right_band - 2)

        tl_r = random.randint(1, split_r - tl_h)
        tl_c = random.randint(1, split_c - tl_w)
        tr_r = random.randint(1, split_r - tr_h)
        tr_c = random.randint(split_c + 1, w - tr_w - 1)
        bl_r = random.randint(split_r + 1, h - bl_h - 1)
        bl_c = random.randint(1, split_c - bl_w)
        br_r = random.randint(split_r + 1, h - br_h - 1)
        br_c = random.randint(split_c + 1, w - br_w - 1)

        boxes = [
            (tl_r, tl_c, tl_h, tl_w),
            (tr_r, tr_c, tr_h, tr_w),
            (bl_r, bl_c, bl_h, bl_w),
            (br_r, br_c, br_h, br_w),
        ]
        if _boxes_are_separate(boxes):
            return boxes
    raise RuntimeError("failed to choose separated hole boxes")


def _boxes_are_separate(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, h0, w0) in enumerate(boxes):
        a0, a1 = r0, r0 + h0 - 1
        b0, b1 = c0, c0 + w0 - 1
        for r1, c1, h1, w1 in boxes[i + 1 :]:
            c0r, c1r = r1, r1 + h1 - 1
            d0, d1 = c1, c1 + w1 - 1
            if not (a1 + 1 < c0r or c1r + h1 - 1 < a0 or b1 + 1 < c1 or d1 + 1 < b0):
                row_gap = max(0, max(r1 - a1 - 1, r0 - (r1 + h1 - 1) - 1))
                col_gap = max(0, max(c1 - b1 - 1, c0 - d1 - 1))
                if row_gap == 0 and col_gap == 0:
                    return False
    return True


def _base_masks(q: int) -> tuple[dict[int, np.ndarray], list[tuple[int, int, int, int]]]:
    top, left, h, w = _choose_layout(q)
    boxes = _choose_hole_boxes(h, w)
    chain = boxes[:]
    random.shuffle(chain)

    masks: dict[int, np.ndarray] = {}
    for holes in (1, 2, 3, 4):
        mask = np.zeros((q, q), dtype=bool)
        mask[top : top + h, left : left + w] = True
        for box_r, box_c, box_h, box_w in chain[:holes]:
            rr0 = top + box_r
            cc0 = left + box_c
            mask[rr0 : rr0 + box_h, cc0 : cc0 + box_w] = False
        masks[holes] = mask
    return masks, chain


def _appendage_candidates(mask: np.ndarray) -> list[np.ndarray]:
    q = int(mask.shape[0])
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    options: list[np.ndarray] = []

    def add_rect(rr0: int, rr1: int, cc0: int, cc1: int) -> None:
        if rr0 > rr1 or cc0 > cc1:
            return
        patch = np.zeros((q, q), dtype=bool)
        patch[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        options.append(patch)

    for w in range(1, min(4, c1 - c0 + 1) + 1):
        if r0 > 0:
            start = random.randint(c0, c1 - w + 1)
            add_rect(r0 - 1, r0 - 1, start, start + w - 1)
        if r1 < q - 1:
            start = random.randint(c0, c1 - w + 1)
            add_rect(r1 + 1, r1 + 1, start, start + w - 1)
    for h in range(1, min(4, r1 - r0 + 1) + 1):
        if c0 > 0:
            start = random.randint(r0, r1 - h + 1)
            add_rect(start, start + h - 1, c0 - 1, c0 - 1)
        if c1 < q - 1:
            start = random.randint(r0, r1 - h + 1)
            add_rect(start, start + h - 1, c1 + 1, c1 + 1)

    unique = []
    seen = set()
    for patch in options:
        key = patch.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(patch)
    return unique


def _scramble_areas(base_masks: dict[int, np.ndarray]) -> dict[int, np.ndarray] | None:
    masks = {holes: mask.copy() for holes, mask in base_masks.items()}
    target_order = sorted((holes for holes in masks), key=lambda _h: random.random())
    bonus_targets = {holes: idx * 2 + random.randint(0, 3) for idx, holes in enumerate(target_order)}

    for holes in (1, 2, 3, 4):
        needed = bonus_targets[holes]
        candidates = _appendage_candidates(masks[holes])
        random.shuffle(candidates)
        used = np.zeros_like(masks[holes], dtype=bool)
        added = 0
        for patch in candidates:
            patch = patch & ~masks[holes] & ~used
            if not np.any(patch):
                continue
            masks[holes] |= patch
            used |= patch
            added += int(patch.sum())
            if added >= needed:
                break

    areas = {holes: int(mask.sum()) for holes, mask in masks.items()}
    ordered_by_area = [holes for holes, _area in sorted(areas.items(), key=lambda item: item[1])]
    if ordered_by_area == [4, 3, 2, 1] or ordered_by_area == [1, 2, 3, 4]:
        return None
    return masks


def _compose_input(q: int, sep: int, colors_by_quadrant: list[int], masks_by_quadrant: list[np.ndarray]) -> np.ndarray:
    grid = np.full((2 * q + 1, 2 * q + 1), _BG, dtype=int)
    grid[q, :] = sep
    grid[:, q] = sep
    offsets = [(0, 0), (0, q + 1), (q + 1, 0), (q + 1, q + 1)]
    for color, mask, (r0, c0) in zip(colors_by_quadrant, masks_by_quadrant, offsets):
        patch = np.full((q, q), _BG, dtype=int)
        patch[mask] = color
        grid[r0 : r0 + q, c0 : c0 + q] = patch
    return grid


def _semantic_checks(grid: np.ndarray) -> None:
    bg, sep, quadrants = _extract_quadrants(grid)
    assert sep != bg

    hole_counts = []
    masks = []
    colors = []
    for quadrant in quadrants:
        fg = [int(x) for x in np.unique(quadrant) if int(x) != bg]
        assert len(fg) == 1
        color = fg[0]
        colors.append(color)
        mask = quadrant == color
        masks.append(mask)
        assert len(_components(mask)) == 1
        hole_counts.append(_hole_count(mask))
    assert sorted(hole_counts) == [1, 2, 3, 4]
    assert len(set(colors)) == 4

    count_grid = sum(mask.astype(int) for mask in masks)
    for needed in (0, 1, 2, 3, 4):
        assert np.any(count_grid == needed), f"missing overlap class {needed}"

    out = main(grid)
    visible = set(int(x) for x in np.unique(out) if int(x) != bg)
    assert len(visible) == 4, "every layer must be visible in the output"

    ranked = sorted(zip(hole_counts, colors, masks), reverse=True)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            expected = bg
            for _holes, color, mask in ranked:
                if mask[r, c]:
                    expected = color
                    break
            assert int(out[r, c]) == expected


def _attempt() -> np.ndarray | None:
    q = random.randint(*_SIZE_RANGE)
    palette = list(_COLOR_POOL)
    random.shuffle(palette)
    sep = palette[0]
    quadrant_colors = palette[1:5]

    base_masks, _chain = _base_masks(q)
    masks = _scramble_areas(base_masks)
    if masks is None:
        return None

    counts_by_quadrant = [1, 2, 3, 4]
    random.shuffle(counts_by_quadrant)
    masks_by_quadrant = [masks[holes] for holes in counts_by_quadrant]
    grid = _compose_input(q, sep, quadrant_colors, masks_by_quadrant)

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_TRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-layer overlay input")
