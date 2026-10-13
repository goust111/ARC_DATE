from __future__ import annotations

import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]

    matches = []
    for r, row_color in rows:
        for c, col_color in cols:
            if row_color == col_color:
                matches.append((int(r), int(c), int(row_color)))
    assert len(matches) == 1, "expected exactly one full-row/full-column cross"
    return matches[0]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _coords_to_mask(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _classify_mask(mask: np.ndarray) -> tuple[str, bool, bool, bool]:
    has_h = bool(np.array_equal(mask, np.flipud(mask)))
    has_v = bool(np.array_equal(mask, np.fliplr(mask)))
    has_rot = bool(np.array_equal(mask, np.rot90(mask, 2)))
    if has_h and has_v:
        return "both", has_h, has_v, has_rot
    if has_h:
        return "horizontal", has_h, has_v, has_rot
    if has_v:
        return "vertical", has_h, has_v, has_rot
    return "none", has_h, has_v, has_rot


def _transform_mask(mask: np.ndarray, symmetry_type: str) -> np.ndarray:
    if symmetry_type == "both":
        return mask.copy()
    if symmetry_type == "horizontal":
        return np.flipud(mask)
    if symmetry_type == "vertical":
        return np.fliplr(mask)
    if symmetry_type == "none":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown symmetry type: {symmetry_type}")


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _enumerate_connected_shapes(max_area: int = 7) -> dict[int, set[tuple[tuple[int, int], ...]]]:
    shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for area in range(1, max_area):
        next_shapes: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes[area]:
            cells = set(shape)
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            for extra in frontier:
                next_shapes.add(_normalize_shape(cells | {extra}))
        shapes[area + 1] = next_shapes
    return shapes


def _build_shape_pools() -> dict[str, list[np.ndarray]]:
    all_shapes = _enumerate_connected_shapes(7)
    pools: dict[str, list[np.ndarray]] = {
        "horizontal": [],
        "vertical": [],
        "both": [],
        "none": [],
    }
    for area in range(3, 8):
        for shape in all_shapes[area]:
            mask = _coords_to_mask(shape)
            h, w = mask.shape
            if h > 4 or w > 4:
                continue
            if min(h, w) < 2:
                continue
            symmetry_type, _has_h, _has_v, has_rot = _classify_mask(mask)
            if symmetry_type == "none" and has_rot:
                continue
            pools[symmetry_type].append(mask)
    for key, value in pools.items():
        assert value, f"shape pool {key} is empty"
    return pools


SHAPE_POOLS = _build_shape_pools()


def main(grid: np.ndarray) -> np.ndarray:
    """Move each quadrant object to the point-symmetric quadrant using a transformation chosen by its own axis symmetries.

    Input structure:
    - The background is black.
    - A full grey row and a full grey column cross at the exact grid center and split the canvas into four quadrants.
    - Each quadrant contains exactly one small 4-connected monochrome object, with four distinct colors overall.
    - Every object belongs to one of four symmetry classes: horizontal-axis only, vertical-axis only, both axes, or no axis.

    Rule:
    - Remove each colored object from its original quadrant and send it to the quadrant opposite through the cross center.
    - The target bbox is the point-symmetric image of the original bbox with respect to the cross center.
    - The object's internal orientation depends on its symmetry class: objects with no axis are rotated 180 degrees, horizontal-axis-only objects are flipped vertically, vertical-axis-only objects are flipped horizontally, and both-axis objects are left unchanged.
    - The grey cross stays unchanged and all vacated cells become background.

    Key insight:
    - The quadrant swap is only half of the rule; the hidden selector for the orientation change is each object's own mirror symmetry type.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    bg = _most_common_color(g)
    mid_r, mid_c, cross_color = _find_cross(g, bg)
    assert h == 2 * mid_r + 1 and w == 2 * mid_c + 1, "cross must be centered"

    obj_mask = g != bg
    obj_mask[mid_r, :] = False
    obj_mask[:, mid_c] = False
    components = _find_components(obj_mask)
    assert len(components) == 4, "expected exactly four colored objects"

    out = np.full_like(g, bg)
    out[mid_r, :] = cross_color
    out[:, mid_c] = cross_color

    quadrants_seen: set[tuple[int, int]] = set()
    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        quad = (0 if r1 < mid_r else 1, 0 if c1 < mid_c else 1)
        quadrants_seen.add(quad)

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local_mask[r - r0, c - c0] = True
        symmetry_type, _has_h, _has_v, _has_rot = _classify_mask(local_mask)
        moved_mask = _transform_mask(local_mask, symmetry_type)

        target_r0 = 2 * mid_r - r1
        target_c0 = 2 * mid_c - c1
        target_r1 = target_r0 + moved_mask.shape[0] - 1
        target_c1 = target_c0 + moved_mask.shape[1] - 1
        assert 0 <= target_r0 <= target_r1 < h
        assert 0 <= target_c0 <= target_c1 < w
        assert target_r1 < mid_r or target_r0 > mid_r
        assert target_c1 < mid_c or target_c0 > mid_c

        patch = out[target_r0 : target_r1 + 1, target_c0 : target_c1 + 1]
        assert not np.any(patch[moved_mask] != bg), "moved objects must not overlap"
        patch[moved_mask] = color

    assert len(quadrants_seen) == 4, "expected one object per quadrant"
    return out


BG_COLOR = 0
CROSS_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_CHOICES = [13, 15, 17, 19, 21, 23, 25]
MAX_RETRIES = 300


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == BG_COLOR)
    patch[mask] = color


def _sample_shape(symmetry_type: str) -> np.ndarray:
    return random.choice(SHAPE_POOLS[symmetry_type]).copy()


def _generate_attempt() -> np.ndarray | None:
    h = random.choice(GRID_SIDE_CHOICES)
    w = random.choice(GRID_SIDE_CHOICES)
    mid_r, mid_c = h // 2, w // 2

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[mid_r, :] = CROSS_COLOR
    grid[:, mid_c] = CROSS_COLOR

    colors = random.sample(PALETTE, 4)
    symmetry_types = ["horizontal", "vertical", "both", "none"]
    random.shuffle(symmetry_types)
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)

    quadrant_height = mid_r
    quadrant_width = mid_c
    assignments = list(zip(quadrants, symmetry_types, colors, strict=False))

    for (qr, qc), symmetry_type, color in assignments:
        mask = _sample_shape(symmetry_type)
        mh, mw = mask.shape
        if quadrant_height - mh - 1 < 1 or quadrant_width - mw - 1 < 1:
            return None

        local_r = random.randint(1, quadrant_height - mh - 1)
        local_c = random.randint(1, quadrant_width - mw - 1)
        global_r = local_r + (0 if qr == 0 else mid_r + 1)
        global_c = local_c + (0 if qc == 0 else mid_c + 1)
        _place_mask(grid, mask, global_r, global_c, color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
