from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


BG_COLOR = 0
GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 34)
STRUCTURE_COUNT_RANGE = (1, 3)
DEPTH_OPTIONS = (1, 2, 3, 4)
SHAPE_EXTRA_SIDE_MAX = 6
NOTCH_COUNT_MAX = 5
PLACEMENT_PADDING = 2
PLACEMENT_TRIES = 250
GENERATION_ATTEMPTS = 400
LOCAL_SHAPE_ATTEMPTS = 120


def _most_common_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    best = int(np.argmax(counts))
    if int(np.sum(counts == counts[best])) != 1:
        raise AssertionError("background color must be unique")
    return int(colors[best])


def _unique_largest_non_bg_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    candidates = [(color, count) for color, count in counts.items() if color != int(bg)]
    assert candidates, "expected at least one non-background color"
    candidates.sort(key=lambda item: item[1], reverse=True)
    if len(candidates) > 1:
        assert candidates[0][1] > candidates[1][1], "gray ring color must be uniquely largest"
    return int(candidates[0][0])


def _label_mask(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[np.ndarray]]:
    assert connectivity in (4, 8)
    dirs = DIR4 if connectivity == 4 else DIR8
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []

    next_label = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
            next_label += 1

    return labels, components


def _erode4(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    cur = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(steps)):
        if not cur.any():
            break
        up = np.zeros_like(cur)
        down = np.zeros_like(cur)
        left = np.zeros_like(cur)
        right = np.zeros_like(cur)
        up[1:] = cur[:-1]
        down[:-1] = cur[1:]
        left[:, 1:] = cur[:, :-1]
        right[:, :-1] = cur[:, 1:]
        cur = cur & up & down & left & right
    return cur


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & (~_erode4(mask, 1))


def _component_count(mask: np.ndarray, connectivity: int) -> int:
    _labels, components = _label_mask(mask, connectivity)
    return len(components)


def _palette_colors(g: np.ndarray, bg: int, gray: int) -> list[int]:
    coords = np.argwhere((g != int(bg)) & (g != int(gray)))
    assert coords.shape[0] == 4, "palette must have four non-gray pixels"
    order = coords[np.argsort(coords[:, 0])]
    rows = order[:, 0]
    cols = order[:, 1]
    assert len(set(map(int, cols))) == 1, "palette must be vertical"
    assert np.all(np.diff(rows) == 2), "palette entries must be separated by one background row"
    assert int(cols[0]) in {0, g.shape[1] - 1}, "palette must live on a corner column"
    assert int(rows[0]) in {0, g.shape[0] - 7}, "palette must start at a corner row"
    colors = [int(g[int(r), int(c)]) for r, c in order]
    assert len(set(colors)) == 4, "palette colors must be distinct"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the nesting depth of each gray ring stack, then recolor only its innermost ring
    using the palette color that corresponds to that depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _most_common_color(g)
    gray = _unique_largest_non_bg_color(g, bg)
    palette = _palette_colors(g, bg, gray)

    gray_mask = g == int(gray)
    assert gray_mask.any(), "expected at least one gray ring"

    free_labels, free_components = _label_mask(~gray_mask, connectivity=4)
    ring_labels, ring_components = _label_mask(gray_mask, connectivity=8)
    assert ring_components, "expected gray connected components"

    region_to_rings: list[set[int]] = [set() for _ in free_components]
    ring_to_regions: list[set[int]] = []

    h, w = g.shape
    for ring_idx, coords in enumerate(ring_components):
        neighbors: set[int] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    label = int(free_labels[nr, nc])
                    if label > 0:
                        neighbors.add(label - 1)
        assert len(neighbors) == 2, "each ring should separate exactly two free-space regions"
        ring_to_regions.append(neighbors)
        for region_idx in neighbors:
            region_to_rings[region_idx].add(ring_idx)

    edge_region_ids = {
        int(label) - 1
        for label in np.unique(
            np.concatenate(
                [
                    free_labels[0, :],
                    free_labels[-1, :],
                    free_labels[:, 0],
                    free_labels[:, -1],
                ]
            )
        )
        if int(label) > 0
    }
    assert edge_region_ids, "expected an exterior region"

    depths = [-1] * len(free_components)
    q = deque()
    for region_idx in edge_region_ids:
        depths[region_idx] = 0
        q.append(region_idx)

    while q:
        region_idx = q.popleft()
        for ring_idx in region_to_rings[region_idx]:
            for other_region_idx in ring_to_regions[ring_idx]:
                if depths[other_region_idx] == -1:
                    depths[other_region_idx] = depths[region_idx] + 1
                    q.append(other_region_idx)

    assert all(depth >= 0 for depth in depths), "all free-space regions must be reachable from outside"

    out = g.copy()
    for ring_idx, coords in enumerate(ring_components):
        region_a, region_b = tuple(ring_to_regions[ring_idx])
        depth_a = depths[region_a]
        depth_b = depths[region_b]
        assert abs(depth_a - depth_b) == 1, "crossing a ring must change nesting depth by one"
        ring_level = max(depth_a, depth_b)
        inner_region = region_a if depth_a > depth_b else region_b
        if len(region_to_rings[inner_region]) == 1:
            assert 1 <= ring_level <= len(palette), "ring depth must map into the palette"
            out[coords[:, 0], coords[:, 1]] = int(palette[ring_level - 1])

    return out


def _random_shape(depth: int) -> np.ndarray | None:
    min_side = 3 + 4 * (int(depth) - 1)
    for _ in range(LOCAL_SHAPE_ATTEMPTS):
        h = random.randint(min_side, min_side + SHAPE_EXTRA_SIDE_MAX)
        w = random.randint(min_side, min_side + SHAPE_EXTRA_SIDE_MAX)
        mask = np.ones((h, w), dtype=bool)

        notch_count = random.randint(0, NOTCH_COUNT_MAX)
        for _ in range(notch_count):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                if w < 5:
                    continue
                depth_cap = max(1, min(3, h // 3))
                notch_depth = random.randint(1, depth_cap)
                span = random.randint(2, w - 2)
                start = random.randint(1, w - span - 1)
                if side == "top":
                    mask[:notch_depth, start : start + span] = False
                else:
                    mask[h - notch_depth :, start : start + span] = False
            else:
                if h < 5:
                    continue
                depth_cap = max(1, min(3, w // 3))
                notch_depth = random.randint(1, depth_cap)
                span = random.randint(2, h - 2)
                start = random.randint(1, h - span - 1)
                if side == "left":
                    mask[start : start + span, :notch_depth] = False
                else:
                    mask[start : start + span, w - notch_depth :] = False

        if _is_valid_base_shape(mask, depth):
            return mask
    return None


def _is_valid_base_shape(mask: np.ndarray, depth: int) -> bool:
    if not mask.any():
        return False
    cur = np.asarray(mask, dtype=bool)
    for level in range(int(depth)):
        if not cur.any() or _component_count(cur, connectivity=4) != 1:
            return False
        interior = _erode4(cur, 1)
        if not interior.any():
            return False
        ring = cur & (~interior)
        if _component_count(ring, connectivity=8) != 1:
            return False
        if level + 1 < int(depth):
            cur = _erode4(cur, 2)
    return True


def _structure_sprite(depth: int, gray: int) -> tuple[np.ndarray, np.ndarray]:
    shape = _random_shape(depth)
    assert shape is not None, "failed to build a valid nested-ring shape"

    sprite = np.zeros(shape.shape, dtype=int)
    cur = shape.copy()
    innermost_ring = None
    for _ in range(int(depth)):
        ring = _boundary(cur)
        sprite[ring] = int(gray)
        innermost_ring = ring
        cur = _erode4(cur, 2)
    assert innermost_ring is not None
    return sprite, np.asarray(innermost_ring, dtype=bool)


def _palette_layout(height: int, width: int, colors: list[int]) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    corner = random.choice(("tl", "tr", "bl", "br"))
    top = 0 if corner[0] == "t" else int(height) - 7
    col = 0 if corner[1] == "l" else int(width) - 1

    placements = [(top + 2 * i, col, int(color)) for i, color in enumerate(colors)]
    reserved = np.zeros((height, width), dtype=bool)
    r0 = max(0, top - 1)
    r1 = min(height, top + 8)
    if col == 0:
        c0, c1 = 0, min(width, 3)
    else:
        c0, c1 = max(0, width - 3), width
    reserved[r0:r1, c0:c1] = True
    return placements, reserved


def _try_place_sprite(
    occupied: np.ndarray,
    sprite: np.ndarray,
    *,
    padding: int,
) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    sprite_h, sprite_w = sprite.shape
    if sprite_h + 2 > grid_h or sprite_w + 2 > grid_w:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(1, grid_h - sprite_h - 1)
        left = random.randint(1, grid_w - sprite_w - 1)
        r0 = max(0, top - padding)
        r1 = min(grid_h, top + sprite_h + padding)
        c0 = max(0, left - padding)
        c1 = min(grid_w, left + sprite_w + padding)
        if not occupied[r0:r1, c0:c1].any():
            return top, left
    return None


def _place_structures(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    gray: int,
    palette: list[int],
    depths: list[int],
) -> bool:
    sprites: list[tuple[int, np.ndarray, np.ndarray]] = []
    for depth in depths:
        sprite, inner_ring = _structure_sprite(depth, gray)
        sprites.append((int(depth), sprite, inner_ring))

    sprites.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for depth, sprite, inner_ring in sprites:
        location = _try_place_sprite(occupied, sprite, padding=PLACEMENT_PADDING)
        if location is None:
            return False
        top, left = location
        sprite_h, sprite_w = sprite.shape
        region = sprite != BG_COLOR
        grid[top : top + sprite_h, left : left + sprite_w][region] = int(gray)
        expected[top : top + sprite_h, left : left + sprite_w][region] = int(gray)
        ring_region = inner_ring
        expected[top : top + sprite_h, left : left + sprite_w][ring_region] = int(palette[depth - 1])

        r0 = max(0, top - PLACEMENT_PADDING)
        r1 = min(grid.shape[0], top + sprite_h + PLACEMENT_PADDING)
        c0 = max(0, left - PLACEMENT_PADDING)
        c1 = min(grid.shape[1], left + sprite_w + PLACEMENT_PADDING)
        occupied[r0:r1, c0:c1] = True

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        if height < 8 or width < 8:
            continue

        colors = list(range(1, 10))
        random.shuffle(colors)
        gray = int(colors[0])
        palette = [int(color) for color in colors[1:5]]

        grid = np.full((height, width), BG_COLOR, dtype=int)
        expected = grid.copy()

        palette_cells, occupied = _palette_layout(height, width, palette)
        for r, c, color in palette_cells:
            grid[r, c] = color
            expected[r, c] = color

        structure_count = random.randint(*STRUCTURE_COUNT_RANGE)
        depths = [random.choice(DEPTH_OPTIONS) for _ in range(structure_count)]
        area_budget = sum((3 + 4 * (depth - 1)) ** 2 for depth in depths)
        if area_budget > int(height * width * 0.45):
            continue

        if not _place_structures(grid, expected, occupied, gray, palette, depths):
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
