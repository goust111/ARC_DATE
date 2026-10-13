from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
HOLE_SHAPES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _flood_outside(space_mask: np.ndarray) -> np.ndarray:
    h, w = space_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if space_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if space_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and space_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return outside


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        queue = deque([(r, c)])
        seen[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return count


def _find_rectangular_hole(mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    bbox = _bbox_from_coords(np.argwhere(mask))
    r0, c0, r1, c1 = bbox
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    bg_crop = ~crop
    hole_mask = bg_crop & ~_flood_outside(bg_crop)
    hole_coords = np.argwhere(hole_mask)
    assert hole_coords.size > 0, "object must enclose a hole"
    assert _count_components(hole_mask) == 1, "expected exactly one enclosed hole"

    hr0, hc0, hr1, hc1 = _bbox_from_coords(hole_coords)
    hole_box = (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1)
    assert hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1].all(), "hole must be rectangular"
    area = int(hole_mask.sum())
    assert area == (hr1 - hr0 + 1) * (hc1 - hc0 + 1), "hole area must match its rectangle"
    return hole_box, area


def _direction_from_centers(
    outer_box: tuple[int, int, int, int],
    hole_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = outer_box
    hr0, hc0, hr1, hc1 = hole_box
    dr2 = (hr0 + hr1) - (r0 + r1)
    dc2 = (hc0 + hc1) - (c0 + c1)
    assert (dr2 == 0) ^ (dc2 == 0), "hole offset must point along one cardinal axis"
    if dr2 < 0:
        return -1, 0
    if dr2 > 0:
        return 1, 0
    if dc2 < 0:
        return 0, -1
    return 0, 1


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each object by the direction and distance encoded by its enclosed hole.

    Input structure:
    - The grid has one background color and several non-background objects of distinct colors.
    - Each object is a single connected component whose bounding box contains exactly one enclosed rectangular background hole.
    - The hole center is offset from the object bbox center strictly upward, downward, leftward, or rightward.

    Rule:
    - For each colored object, detect its unique enclosed rectangular hole.
    - Use the hole center's offset from the object bbox center to choose the translation direction.
    - Use the hole area as the translation distance, clear the canvas to background, and repaint the object at its translated location.

    Key insight:
    - Each object's internal hole acts as both compass and odometer: its off-center position gives the direction, and its area gives the step count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    out = np.full_like(g, bg)
    occupied = np.zeros((h, w), dtype=bool)
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        mask = g == color
        assert mask.any(), "foreground color must appear"
        assert _count_components(mask) == 1, "each object color must form one component"
        box = _bbox_from_coords(np.argwhere(mask))
        hole_box, distance = _find_rectangular_hole(mask)
        dr, dc = _direction_from_centers(box, hole_box)

        coords = np.argwhere(mask)
        dst = coords + np.array([dr * distance, dc * distance], dtype=int)
        assert (0 <= dst[:, 0]).all() and (dst[:, 0] < h).all()
        assert (0 <= dst[:, 1]).all() and (dst[:, 1] < w).all()
        assert not occupied[dst[:, 0], dst[:, 1]].any(), "translated objects must stay disjoint"

        out[dst[:, 0], dst[:, 1]] = color
        occupied[dst[:, 0], dst[:, 1]] = True

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _make_object_patch(color: int, direction: tuple[int, int]) -> tuple[np.ndarray, int]:
    hole_h, hole_w = random.choice(HOLE_SHAPES)
    if direction[0] != 0:
        extra_h = random.choice([4, 6, 8])
        extra_w = random.choice([2, 4, 6])
        outer_h = hole_h + extra_h
        outer_w = hole_w + extra_w
        centered_left = (outer_w - hole_w) // 2
        center_gap = (outer_h - hole_h) // 2
        delta = random.randint(1, center_gap - 1)
        top = center_gap - delta if direction[0] < 0 else center_gap + delta
        bottom = outer_h - hole_h - top
        left = centered_left
    else:
        extra_h = random.choice([2, 4, 6])
        extra_w = random.choice([4, 6, 8])
        outer_h = hole_h + extra_h
        outer_w = hole_w + extra_w
        centered_top = (outer_h - hole_h) // 2
        center_gap = (outer_w - hole_w) // 2
        delta = random.randint(1, center_gap - 1)
        left = center_gap - delta if direction[1] < 0 else center_gap + delta
        top = centered_top
        bottom = outer_h - hole_h - top

    right = outer_w - hole_w - left
    assert min(top, bottom, left, right) >= 1

    patch = np.full((outer_h, outer_w), color, dtype=int)
    patch[top : top + hole_h, left : left + hole_w] = 0
    return patch, hole_h * hole_w


def _placeable_positions(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    shift: tuple[int, int],
    reserved: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = patch_shape
    dr, dc = shift
    positions: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            src = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            dst = (src[0] + dr, src[1] + dc, src[2] + dr, src[3] + dc)
            if dst[0] < 0 or dst[1] < 0 or dst[2] >= h or dst[3] >= w:
                continue
            sweep = _union_box(src, dst)
            if any(_boxes_overlap(sweep, box, pad=1) for box in reserved):
                continue
            positions.append((r0, c0))
    return positions


def _sample_object_specs() -> list[dict[str, object]]:
    n = random.randint(2, 4)
    directions = random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], k=n)
    colors = random.sample(COLOR_POOL, k=n)
    specs: list[dict[str, object]] = []
    for direction, color in zip(directions, colors):
        patch, distance = _make_object_patch(color, direction)
        shift = (direction[0] * distance, direction[1] * distance)
        specs.append(
            {
                "color": color,
                "patch": patch,
                "shift": shift,
                "footprint": patch.shape[0] * patch.shape[1],
            }
        )
    specs.sort(key=lambda spec: int(spec["footprint"]), reverse=True)
    return specs


def _render_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    specs = _sample_object_specs()
    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    for spec in specs:
        patch = np.asarray(spec["patch"], dtype=int)
        shift = tuple(spec["shift"])
        candidates = _placeable_positions((grid_h, grid_w), patch.shape, shift, reserved)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        ph, pw = patch.shape
        dr, dc = shift
        src = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        dst = (src[0] + dr, src[1] + dc, src[2] + dr, src[3] + dc)
        sweep = _union_box(src, dst)

        src_view = grid[r0 : r0 + ph, c0 : c0 + pw]
        src_view[patch != 0] = patch[patch != 0]
        dst_view = expected[dst[0] : dst[2] + 1, dst[1] : dst[3] + 1]
        dst_view[patch != 0] = patch[patch != 0]
        reserved.append(sweep)

    if len(set(map(int, np.unique(grid))) - {0}) < 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _render_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-compass ARC input")
