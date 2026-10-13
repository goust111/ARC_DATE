from __future__ import annotations

import numpy as np


BG = 0
PROTOTYPE_COLOR = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = (
    frozenset({(1, 0), (0, 1)}),
    frozenset({(1, 0), (0, -1)}),
    frozenset({(-1, 0), (0, 1)}),
    frozenset({(-1, 0), (0, -1)}),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
            comps.append(coords)
    return comps


def _rotate_vec(vec: tuple[int, int], k: int) -> tuple[int, int]:
    dr, dc = vec
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return dr, dc


def _rotate_offsets(offsets: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_rotate_vec(offset, k) for offset in offsets))


def _extract_prototype(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]:
    h, w = grid.shape
    grey_mask = grid == PROTOTYPE_COLOR
    comps = _components(grey_mask)
    assert comps, "expected a grey prototype"

    def touches_corner(coords: list[tuple[int, int]]) -> bool:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )

    corner_comps = [comp for comp in comps if touches_corner(comp)]
    assert len(corner_comps) == 1, "expected exactly one corner prototype"
    coords = corner_comps[0]
    coord_set = set(coords)

    elbow_candidates: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
    for r, c in coords:
        neighbors = []
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in coord_set:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            continue
        has_vertical = any(dr != 0 for dr, _ in neighbors)
        has_horizontal = any(dc != 0 for _, dc in neighbors)
        if has_vertical and has_horizontal:
            elbow_candidates.append(((r, c), frozenset(neighbors)))

    assert len(elbow_candidates) == 1, "prototype corner must be unique"
    (anchor_r, anchor_c), arm_dirs = elbow_candidates[0]
    offsets = tuple(sorted((r - anchor_r, c - anchor_c) for r, c in coords))
    return grey_mask, offsets, arm_dirs


def _decode_locator(coords: np.ndarray) -> tuple[tuple[int, int], frozenset[tuple[int, int]]]:
    points = [tuple(map(int, point)) for point in coords]
    candidates: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
    for r, c in points:
        horiz = []
        vert = []
        for rr, cc in points:
            if (rr, cc) == (r, c):
                continue
            if rr == r and cc != c:
                horiz.append((0, 1 if cc > c else -1))
            if cc == c and rr != r:
                vert.append((1 if rr > r else -1, 0))
        if len(horiz) == 1 and len(vert) == 1:
            candidates.append(((r, c), frozenset({horiz[0], vert[0]})))
    assert len(candidates) == 1, "locator group must have one right-angle vertex"
    return candidates[0]


def _rotation_for_dirs(
    src_dirs: frozenset[tuple[int, int]],
    dst_dirs: frozenset[tuple[int, int]],
) -> int:
    for k in range(4):
        if frozenset(_rotate_vec(vec, k) for vec in src_dirs) == dst_dirs:
            return k
    raise AssertionError("no rotation matches locator orientation")


def main(grid: np.ndarray) -> np.ndarray:
    """Find the grey stamp prototype, decode each three-point locator, and place rotated colored copies.

    Input structure
    - The background is black.
    - A single connected grey prototype sits in one grid corner and contains a unique elbow pixel with two orthogonal grey neighbors.
    - Every other foreground color appears as exactly three isolated pixels forming a right angle, with one pixel aligned horizontally and vertically to the other two.

    Rule
    - Keep the grey prototype unchanged.
    - For each non-grey color, find the right-angle vertex; use it as the stamp anchor.
    - Rotate the prototype so the elbow arms point in the same horizontal and vertical directions as that locator group, then paint the rotated prototype in that color with the elbow placed on the anchor pixel.

    Key insight
    - Each locator trio encodes both translation and rotation: the shared row/column pixel gives the anchor, and the two arm directions determine which 90-degree rotation of the prototype to place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    grey_mask, prototype_offsets, prototype_dirs = _extract_prototype(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) not in {BG, PROTOTYPE_COLOR}]
    assert colors, "expected at least one locator group"
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 3, "each locator color must appear exactly three times"
        anchor, target_dirs = _decode_locator(coords)
        rotation = _rotation_for_dirs(prototype_dirs, target_dirs)
        for dr, dc in _rotate_offsets(prototype_offsets, rotation):
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert int(out[rr, cc]) in {BG, color, PROTOTYPE_COLOR}
            out[rr, cc] = color

    out[grey_mask] = PROTOTYPE_COLOR
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
GRID_SIZE_RANGE = (14, 27)
ARM_LEN_RANGE = (1, 5)
STAMP_COUNT_RANGE = (2, 6)
GENERATION_ATTEMPTS = 400
PLACEMENT_MARGIN = 1
ANCHOR_TRIALS = 200
DIST_TRIALS = 10
CORNERS = ("tl", "tr", "bl", "br")
BASE_PROTOTYPE_DIRS = frozenset({(1, 0), (0, 1)})


def _bounds(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _ in offsets]
    cols = [dc for _, dc in offsets]
    return min(rows), max(rows), min(cols), max(cols)


def _corner_shift(
    offsets: tuple[tuple[int, int], ...],
    corner: str,
    height: int,
    width: int,
) -> tuple[int, int]:
    min_dr, max_dr, min_dc, max_dc = _bounds(offsets)
    if corner == "tl":
        return -min_dr, -min_dc
    if corner == "tr":
        return -min_dr, width - 1 - max_dc
    if corner == "bl":
        return height - 1 - max_dr, -min_dc
    if corner == "br":
        return height - 1 - max_dr, width - 1 - max_dc
    raise ValueError(f"unknown corner {corner}")


def _expand_cells(cells: set[tuple[int, int]], height: int, width: int, margin: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((rr, cc))
    return expanded


def _extent_in_dir(offsets: tuple[tuple[int, int], ...], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr == 1:
        return max(offset_r for offset_r, _ in offsets)
    if dr == -1:
        return max(-offset_r for offset_r, _ in offsets)
    if dc == 1:
        return max(offset_c for _, offset_c in offsets)
    return max(-offset_c for _, offset_c in offsets)


def _steps_to_edge(
    anchor: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> int:
    r, c = anchor
    dr, dc = direction
    if dr == 1:
        return height - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return width - 1 - c
    return c


def _render_stamp(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def _place_group(
    height: int,
    width: int,
    shape_offsets: tuple[tuple[int, int], ...],
    shape_dirs: frozenset[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    target_dirs = ORIENTATIONS[int(_RNG.integers(len(ORIENTATIONS)))]
    rotation = _rotation_for_dirs(shape_dirs, target_dirs)
    stamp_offsets = _rotate_offsets(shape_offsets, rotation)
    min_dr, max_dr, min_dc, max_dc = _bounds(stamp_offsets)
    row_lo, row_hi = -min_dr, height - 1 - max_dr
    col_lo, col_hi = -min_dc, width - 1 - max_dc
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchors = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    order = _RNG.permutation(len(anchors))
    horiz_dir = next(direction for direction in target_dirs if direction[0] == 0)
    vert_dir = next(direction for direction in target_dirs if direction[1] == 0)
    min_horiz = _extent_in_dir(stamp_offsets, horiz_dir) + 2
    min_vert = _extent_in_dir(stamp_offsets, vert_dir) + 2

    for idx in order[:ANCHOR_TRIALS]:
        anchor = anchors[int(idx)]
        stamp_cells = _render_stamp(anchor, stamp_offsets)
        if stamp_cells & blocked:
            continue

        max_horiz = _steps_to_edge(anchor, horiz_dir, height, width)
        max_vert = _steps_to_edge(anchor, vert_dir, height, width)
        if max_horiz < min_horiz or max_vert < min_vert:
            continue

        for _ in range(DIST_TRIALS):
            horiz_dist = int(_RNG.integers(min_horiz, max_horiz + 1))
            vert_dist = int(_RNG.integers(min_vert, max_vert + 1))
            horiz_point = (
                anchor[0] + horiz_dir[0] * horiz_dist,
                anchor[1] + horiz_dir[1] * horiz_dist,
            )
            vert_point = (
                anchor[0] + vert_dir[0] * vert_dist,
                anchor[1] + vert_dir[1] * vert_dist,
            )
            extra_points = {horiz_point, vert_point}
            if len(extra_points) != 2:
                continue
            if extra_points & stamp_cells:
                continue
            if extra_points & blocked:
                continue
            return {
                "anchor": anchor,
                "target_dirs": target_dirs,
                "stamp_cells": stamp_cells,
                "locator_cells": {anchor} | extra_points,
            }
    return None


def _expected_output(
    grid: np.ndarray,
    prototype_cells: set[tuple[int, int]],
    groups: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        color = int(group["color"])
        for r, c in group["stamp_cells"]:
            out[r, c] = color
    for r, c in prototype_cells:
        out[r, c] = PROTOTYPE_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    width = int(_RNG.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))

    horiz_arm = int(_RNG.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1]))
    vert_arm = int(_RNG.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1]))
    base_offsets = tuple(sorted({(0, 0)} | {(0, c) for c in range(1, horiz_arm + 1)} | {(r, 0) for r in range(1, vert_arm + 1)}))

    display_rotation = int(_RNG.integers(4))
    display_offsets = _rotate_offsets(base_offsets, display_rotation)
    display_dirs = frozenset(_rotate_vec(direction, display_rotation) for direction in BASE_PROTOTYPE_DIRS)
    corner = CORNERS[int(_RNG.integers(len(CORNERS)))]
    shift_r, shift_c = _corner_shift(display_offsets, corner, height, width)
    prototype_cells = {(shift_r + dr, shift_c + dc) for dr, dc in display_offsets}

    blocked = _expand_cells(prototype_cells, height, width, PLACEMENT_MARGIN)
    stamp_count = int(_RNG.integers(STAMP_COUNT_RANGE[0], STAMP_COUNT_RANGE[1]))
    if stamp_count > len(COLOR_POOL):
        return None
    colors = [int(color) for color in _RNG.choice(COLOR_POOL, size=stamp_count, replace=False)]

    groups: list[dict[str, object]] = []
    for color in colors:
        placement = _place_group(height, width, display_offsets, display_dirs, blocked)
        if placement is None:
            return None
        placement["color"] = color
        actual_cells = set(placement["stamp_cells"]) | set(placement["locator_cells"])
        blocked |= _expand_cells(actual_cells, height, width, PLACEMENT_MARGIN)
        groups.append(placement)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in prototype_cells:
        grid[r, c] = PROTOTYPE_COLOR
    for group in groups:
        color = int(group["color"])
        for r, c in group["locator_cells"]:
            grid[r, c] = color

    expected = _expected_output(grid, prototype_cells, groups)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
