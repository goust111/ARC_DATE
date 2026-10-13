from __future__ import annotations

import random

import numpy as np

BBox = tuple[int, int, int, int]

ALL_COLORS = list(range(10))
GEN_ATTEMPTS = 500
PATCH_SIZE_RANGE = (4, 8)
GRID_MAX = 29
PATCH_COLORS_RANGE = (2, 4)
PATCH_MIN_COUNT = 3
GUIDE_POINTS_RANGE = (5, 7)
MAX_CORRUPT_CELLS = 6
PATCH_BUILD_ATTEMPTS = 200


def _bbox(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit(shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = map(int, shape)
    pts = {
        (int(r), int(c)),
        (int(h - 1 - r), int(c)),
        (int(r), int(w - 1 - c)),
        (int(h - 1 - r), int(w - 1 - c)),
    }
    return tuple(sorted(pts))


def _inside_box(box: BBox, r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _is_biaxially_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask))


def _box_has_conflict(grid: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            orbit_colors = {int(grid[rr, cc]) for rr, cc in _orbit(grid.shape, r, c)}
            if len(orbit_colors) > 1:
                return True
    return False


def _repair_crop(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    repaired = np.array(grid, dtype=int, copy=True)
    changed = False
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            outside_colors = {
                int(grid[rr, cc])
                for rr, cc in _orbit(grid.shape, r, c)
                if not _inside_box(box, rr, cc)
            }
            assert outside_colors
            assert len(outside_colors) == 1
            expected = next(iter(outside_colors))
            if int(repaired[r, c]) != expected:
                repaired[r, c] = expected
                changed = True

    assert changed
    return repaired[r0 + 1 : r1, c0 + 1 : c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the true sparse guide frame, repair the symmetry-broken area inside it, and crop that repaired interior.

    Input structure:
    - A multicolor pattern is consistent under both horizontal and vertical whole-grid reflection, except for one sparse guide color and several tampered cells.
    - Two low-frequency colors appear as sparse point sets: one point set is itself biaxially symmetric and belongs to the pattern, while the other sparsely marks the border of a rectangular frame.
    - The guide frame's strict interior contains ordinary pattern colors, but some of those cells were rewritten so they disagree with their mirror orbit partners outside the frame.

    Rule:
    - Among the two rare colors, find the one whose own point set is not biaxially symmetric and whose bounding box interior contains mirror-orbit conflicts.
    - For every cell in that box's strict interior, read the unique consistent color from the same horizontal-and-vertical reflection orbit outside the box and restore the interior cell to that color.
    - Return only the repaired strict interior crop of the true guide box.

    Key insight:
    - The wrong cells use normal palette colors, so color frequency alone cannot repair them; the sparse guide is identified structurally by failing symmetry itself while enclosing the only region whose interior breaks global mirror-orbit consistency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors, counts = np.unique(g, return_counts=True)
    rare_order = sorted((int(count), int(color)) for color, count in zip(colors, counts))
    assert len(rare_order) >= 2
    if len(rare_order) >= 3:
        assert rare_order[1][0] < rare_order[2][0]
    rare_colors = [rare_order[0][1], rare_order[1][1]]

    guide_box = None
    for color in rare_colors:
        mask = g == int(color)
        box = _bbox(mask)
        symmetric = _is_biaxially_symmetric(mask)
        conflicted = _box_has_conflict(g, box)
        if (not symmetric) and conflicted:
            assert guide_box is None
            guide_box = box

    assert guide_box is not None
    return _repair_crop(g, guide_box)


def _make_patch(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(PATCH_BUILD_ATTEMPTS):
        seeds = random.sample(cells, len(colors))
        patch = np.empty((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                best_idx = 0
                best_dist = None
                for idx, (sr, sc) in enumerate(seeds):
                    dist = abs(r - sr) + abs(c - sc)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                patch[r, c] = int(colors[best_idx])

        color_counts = [int(np.sum(patch == color)) for color in colors]
        if min(color_counts) >= PATCH_MIN_COUNT and len(np.unique(patch)) == len(colors):
            return patch
    return None


def _paint_patch_copies(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    height, width = map(int, grid.shape)
    patch_h, patch_w = map(int, patch.shape)
    bottom = height - top - patch_h
    right = width - left - patch_w

    grid[top : top + patch_h, left : left + patch_w] = patch
    grid[bottom : bottom + patch_h, left : left + patch_w] = np.flipud(patch)
    grid[top : top + patch_h, right : right + patch_w] = np.fliplr(patch)
    grid[bottom : bottom + patch_h, right : right + patch_w] = np.flipud(np.fliplr(patch))


def _guide_points(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    points = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    extras = []
    for c in range(c0 + 1, c1):
        extras.append((r0, c))
        extras.append((r1, c))
    for r in range(r0 + 1, r1):
        extras.append((r, c0))
        extras.append((r, c1))
    random.shuffle(extras)

    target = random.randint(GUIDE_POINTS_RANGE[0], min(GUIDE_POINTS_RANGE[1], len(points) + len(extras)))
    while len(points) < target and extras:
        points.add(extras.pop())
    return list(points)


def _central_false_guide_orbit(shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = map(int, shape)
    return _orbit(shape, h // 2 - 1, w // 2 - 1)


def _candidate_dimensions() -> tuple[int, int, int, int] | None:
    patch_h = random.randint(*PATCH_SIZE_RANGE)
    patch_w = random.randint(*PATCH_SIZE_RANGE)

    min_h = 2 * patch_h + 5
    min_w = 2 * patch_w + 5
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    return height, width, patch_h, patch_w


def _attempt() -> np.ndarray | None:
    dims = _candidate_dimensions()
    if dims is None:
        return None
    height, width, patch_h, patch_w = dims

    top_max = (height - 2 * patch_h - 1) // 2
    left_max = (width - 2 * patch_w - 1) // 2
    if top_max < 2 or left_max < 2:
        return None

    top = random.randint(2, top_max)
    left = random.randint(2, left_max)
    guide_box = (top - 1, left - 1, top + patch_h, left + patch_w)

    palette = random.sample(ALL_COLORS, random.randint(PATCH_COLORS_RANGE[0] + 3, PATCH_COLORS_RANGE[1] + 3))
    bg = palette[0]
    guide_color = palette[1]
    symmetric_rare = palette[2]
    patch_colors = palette[3:]

    patch = _make_patch(patch_h, patch_w, patch_colors)
    if patch is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    _paint_patch_copies(grid, patch, top, left)

    false_orbit = _central_false_guide_orbit(grid.shape)
    if any(_inside_box(guide_box, r, c) or int(grid[r, c]) != bg for r, c in false_orbit):
        return None
    for r, c in false_orbit:
        grid[r, c] = symmetric_rare

    guide_points = _guide_points(guide_box)
    for r, c in guide_points:
        if int(grid[r, c]) != bg:
            return None
        grid[r, c] = guide_color

    cells = [(r, c) for r in range(patch_h) for c in range(patch_w)]
    random.shuffle(cells)
    corrupt_count = random.randint(1, min(MAX_CORRUPT_CELLS, len(cells) // 2))
    for r, c in cells[:corrupt_count]:
        true_color = int(patch[r, c])
        replacements = [color for color in patch_colors if color != true_color]
        grid[top + r, left + c] = random.choice(replacements)

    counts = {int(color): int(np.sum(grid == color)) for color in np.unique(grid)}
    order = sorted((count, color) for color, count in counts.items())
    if len(order) < 3:
        return None
    if {order[0][1], order[1][1]} != {guide_color, symmetric_rare}:
        return None
    if order[1][0] >= order[2][0]:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, patch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
