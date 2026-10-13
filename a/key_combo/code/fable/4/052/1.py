from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_legend(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    comps = _components(grid != BLACK)
    assert comps, "expected a non-black legend component"
    legend_coords = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(legend_coords)
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1 and size % 2 == 1

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != BLACK), "legend must be a filled odd square"

    center = size // 2
    mapping: dict[int, int] = {}
    for d in range(center + 1):
        colors = {
            int(patch[rr, cc])
            for rr in range(size)
            for cc in range(size)
            if max(abs(rr - center), abs(cc - center)) == d
        }
        assert len(colors) == 1, "each legend ring must have one color"
        color = next(iter(colors))
        mapping[color] = d

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return legend_mask, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each cross-arm bead onto the legend-prescribed radius.

    Input structure:
    - A black background contains one odd-sized concentric square legend in a corner.
    - The legend is a filled square whose Chebyshev rings each have a unique color.
    - Outside the legend there is one gray center pixel and several colored singleton beads on the same row or column as that center.

    Rule:
    - Read the legend as a color-to-radius codebook: each ring color maps to its Chebyshev distance from the legend center.
    - For every bead, keep its side relative to the gray center and slide it along its current row or column until its distance from the gray center equals the coded radius for its color.
    - Preserve the gray center, the legend, and every untouched background cell.

    Key insight:
    - The corner square is not a transform target; it is a lookup table that converts each bead color into the only radius where that bead belongs.
    """
    g = np.asarray(grid, dtype=int)
    legend_mask, color_to_radius = _extract_legend(g)

    center_positions = np.argwhere((g == GREY) & (~legend_mask))
    assert len(center_positions) == 1, "expected one gray center outside the legend"
    center_r, center_c = map(int, center_positions[0])

    bead_positions = np.argwhere((g != BLACK) & (g != GREY) & (~legend_mask))
    ray_used: dict[tuple[int, int], tuple[int, int]] = {}
    out = g.copy()

    for r, c in bead_positions:
        rr, cc = int(r), int(c)
        color = int(g[rr, cc])
        assert color in color_to_radius
        target_dist = int(color_to_radius[color])
        assert target_dist > 0, "beads cannot use the legend center color"
        same_row = rr == center_r
        same_col = cc == center_c
        assert same_row ^ same_col, "each bead must lie on one axis through the center"
        dr = 0 if same_row else (1 if rr > center_r else -1)
        dc = 0 if same_col else (1 if cc > center_c else -1)
        ray = (dr, dc)
        assert ray not in ray_used, "at most one bead is allowed on each half-axis"
        ray_used[ray] = (rr, cc)
        out[rr, cc] = BLACK

    for (dr, dc), (_rr, _cc) in ray_used.items():
        color = int(g[_rr, _cc])
        dist = int(color_to_radius[color])
        tr = center_r + dr * dist
        tc = center_c + dc * dist
        assert 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1]
        out[tr, tc] = color

    return out


def _legend_patch(ring_colors: list[int]) -> np.ndarray:
    max_radius = len(ring_colors) - 1
    size = 2 * max_radius + 1
    center = max_radius
    patch = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            dist = max(abs(r - center), abs(c - center))
            patch[r, c] = ring_colors[dist]
    return patch


def _legend_bbox(shape: tuple[int, int], size: int, corner: str) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0, size - 1, size - 1
    if corner == "tr":
        return 0, w - size, size - 1, w - 1
    if corner == "bl":
        return h - size, 0, h - 1, size - 1
    return h - size, w - size, h - 1, w - 1


def _place_patch(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _legend_bbox(grid.shape, patch.shape[0], corner)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return r0, c0, r1, c1


def _ray_capacity(shape: tuple[int, int], center: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = center
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _choose_center(
    shape: tuple[int, int],
    legend_box: tuple[int, int, int, int],
    max_radius: int,
) -> tuple[int, int] | None:
    h, w = shape
    margin = max_radius + 2
    lr0, lc0, lr1, lc1 = legend_box
    rows = [r for r in range(margin, h - margin) if not (lr0 <= r <= lr1)]
    cols = [c for c in range(margin, w - margin) if not (lc0 <= c <= lc1)]
    if not rows or not cols:
        return None
    return random.choice(rows), random.choice(cols)


def _choose_status(target_dist: int, capacity: int) -> str:
    options = ["correct"]
    if target_dist >= 2:
        options.append("near")
    if capacity >= target_dist + 1:
        options.append("far")
    return random.choice(options)


def _distance_from_status(target_dist: int, capacity: int, status: str) -> int:
    if status == "correct":
        return target_dist
    if status == "near":
        return random.randint(1, target_dist - 1)
    return random.randint(target_dist + 1, capacity)


def _gen_attempt() -> np.ndarray | None:
    max_radius = random.randint(2, 4)
    legend_size = 2 * max_radius + 1
    h = random.randint(max(16, legend_size + 8), 30)
    w = random.randint(max(16, legend_size + 8), 30)
    ring_colors = random.sample(PALETTE, max_radius + 1)
    corner = random.choice(["tl", "tr", "bl", "br"])

    grid = np.full((h, w), BLACK, dtype=int)
    legend_box = _place_patch(grid, _legend_patch(ring_colors), corner)
    center = _choose_center((h, w), legend_box, max_radius)
    if center is None:
        return None
    center_r, center_c = center
    grid[center_r, center_c] = GREY

    bead_count = random.randint(2, 4)
    directions = random.sample(DIRS4, bead_count)
    usable_radii = list(range(1, max_radius + 1))
    radii = usable_radii[:]
    random.shuffle(radii)
    while len(radii) < bead_count:
        radii.append(random.choice(usable_radii))
    random.shuffle(radii)

    bead_specs: list[tuple[int, int, int, tuple[int, int], int]] = []
    moved = False
    for direction, target_dist in zip(directions, radii):
        capacity = _ray_capacity((h, w), center, direction)
        if capacity < target_dist:
            return None
        status = _choose_status(target_dist, capacity)
        current_dist = _distance_from_status(target_dist, capacity, status)
        if current_dist != target_dist:
            moved = True
        dr, dc = direction
        r = center_r + dr * current_dist
        c = center_c + dc * current_dist
        color = ring_colors[target_dist]
        bead_specs.append((r, c, color, direction, target_dist))
        grid[r, c] = color

    if not moved:
        adjustable = [
            i
            for i, (direction, target_dist) in enumerate(zip(directions, radii))
            if _ray_capacity((h, w), center, direction) >= target_dist + 1
        ]
        if not adjustable:
            return None
        idx = random.choice(adjustable)
        old_r, old_c, color, direction, target_dist = bead_specs[idx]
        grid[old_r, old_c] = BLACK
        current_dist = random.randint(target_dist + 1, _ray_capacity((h, w), center, direction))
        dr, dc = direction
        new_r = center_r + dr * current_dist
        new_c = center_c + dc * current_dist
        bead_specs[idx] = (new_r, new_c, color, direction, target_dist)
        grid[new_r, new_c] = color

    expected = grid.copy()
    for r, c, _color, _direction, _target_dist in bead_specs:
        expected[r, c] = BLACK
    for _r, _c, color, direction, target_dist in bead_specs:
        dr, dc = direction
        tr = center_r + dr * target_dist
        tc = center_c + dc * target_dist
        expected[tr, tc] = color

    if np.count_nonzero(grid == GREY) != 1:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-abacus input")
