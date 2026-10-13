import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GEN_MIN_SIDE = 12
GEN_MAX_SIDE = 15
GEN_MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _center(box: tuple[int, int, int, int], axis: int) -> float:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2 if axis == 0 else (c0 + c1) / 2


def _span(box: tuple[int, int, int, int], axis: int) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) if axis == 0 else (c1 - c0 + 1)


def _solid_rect_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == color
    box = _bbox(mask)
    patch = _crop(grid, box)
    assert np.all((patch == BLACK) | (patch == color)), "unexpected extra color inside bbox"
    assert int(mask.sum()) == patch.size, "colored object must be a solid rectangle"
    assert np.all(patch == color), "bbox crop must be fully filled"
    return box


def _moving_patch(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), ORANGE, dtype=int)
    if h >= 3 and w >= 3:
        patch[1:-1, 1:-1] = BLACK
        rr, cc = np.indices((h - 2, w - 2))
        patch[1:-1, 1:-1][(rr + cc) % 2 == 0] = RED
    return patch


def _mirror_quadrants(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the orange rectangle through the green bar to the edge, checker-fill its interior, then mirror the whole scene into four symmetric quadrants.

    Input structure:
    - A black grid contains exactly three solid monochrome rectangles: one blue pusher, one orange moving rectangle, and one green target bar.
    - The green target bar is one cell thick and its long side is perpendicular to the movement axis.
    - The orange rectangle lies between the blue pusher and the green bar along one shared horizontal or vertical axis.

    Rule:
    - Move the orange rectangle away from the blue pusher and toward the green bar until it is flush with the far grid edge.
    - Erase the original orange rectangle and the original green bar, then split the green bar along the orange rectangle's path and shift the two remaining halves one cell outward perpendicular to the motion.
    - Draw the moved orange rectangle at the edge with a one-cell orange border and a red-on-black checkerboard interior anchored at the interior's top-left cell.
    - Treat that updated canvas as the upper-left quadrant, mirror it left-right to form the upper half, then mirror the full upper half top-bottom to make the final output.

    Key insight:
    - The green bar is split by the orange rectangle's full cross-section, while the checkerboard phase is determined by the orange rectangle's final edge-flush position before the whole canvas is mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert set(np.unique(g)).issubset({BLACK, BLUE, GREEN, ORANGE}), "unexpected input colors"

    H, W = map(int, g.shape)
    blue_box = _solid_rect_bbox(g, BLUE)
    source_box = _solid_rect_bbox(g, ORANGE)
    target_box = _solid_rect_bbox(g, GREEN)

    target_h = _span(target_box, 0)
    target_w = _span(target_box, 1)
    assert min(target_h, target_w) == 1, "target must be one cell thick"
    assert target_h != target_w, "target must have a unique long side"

    axis = 0 if target_h == 1 else 1
    perp = 1 - axis

    blue_ctr = _center(blue_box, axis)
    source_ctr = _center(source_box, axis)
    target_ctr = _center(target_box, axis)
    assert min(blue_ctr, target_ctr) < source_ctr < max(blue_ctr, target_ctr), (
        "orange rectangle must lie between blue pusher and green target"
    )
    direction = -1 if target_ctr < source_ctr else 1

    sr0, sc0, sr1, sc1 = source_box
    tr0, tc0, tr1, tc1 = target_box
    source_lo = sc0 if axis == 0 else sr0
    source_hi = sc1 if axis == 0 else sr1
    target_lo = tc0 if axis == 0 else tr0
    target_hi = tc1 if axis == 0 else tr1
    assert target_lo < source_lo <= source_hi < target_hi, "target must extend past both sides of the path"

    out = g.copy()
    out[g == ORANGE] = BLACK
    out[g == GREEN] = BLACK

    left_count = 0
    right_count = 0
    for r, c in np.argwhere(g == GREEN):
        r = int(r)
        c = int(c)
        perp_coord = c if axis == 0 else r
        if perp_coord < source_lo:
            delta = -1
            left_count += 1
        elif perp_coord > source_hi:
            delta = 1
            right_count += 1
        else:
            continue

        nr, nc = r, c
        if axis == 0:
            nc += delta
        else:
            nr += delta
        assert 0 <= nr < H and 0 <= nc < W, "split target must stay on the grid"
        out[nr, nc] = GREEN

    assert left_count > 0 and right_count > 0, "target must split into two visible halves"

    source_h = _span(source_box, 0)
    source_w = _span(source_box, 1)
    moved = _moving_patch(source_h, source_w)
    if axis == 0:
        new_r = 0 if direction < 0 else H - source_h
        new_c = sc0
    else:
        new_r = sr0
        new_c = 0 if direction < 0 else W - source_w

    dest = out[new_r : new_r + source_h, new_c : new_c + source_w]
    assert np.all(dest == BLACK), "moved orange destination must be clear after splitting"
    _paste(out, moved, new_r, new_c)
    return _mirror_quadrants(out)


def _transform_grid(grid: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return grid.copy()
    if code == 1:
        return np.fliplr(grid).copy()
    if code == 2:
        return np.flipud(grid).copy()
    if code == 3:
        return np.rot90(grid, 1).copy()
    if code == 4:
        return np.rot90(grid, -1).copy()
    if code == 5:
        return np.rot90(grid, 2).copy()
    if code == 6:
        return grid.T.copy()
    return np.fliplr(grid.T).copy()


def _build_canonical_case() -> np.ndarray | None:
    H = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    W = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)

    source_h = 4
    source_w = random.choice((4, 5))
    target_extra = 2
    target_len = source_h + target_extra
    if target_len > H - 2:
        return None

    pusher_h = 2
    pusher_w = random.choice((2, 3))
    gap_push = random.randint(1, 3)
    gap_target = random.randint(1, 3)

    target_r0 = random.randint(1, H - target_len - 1)
    source_r0 = target_r0 + target_extra // 2
    pusher_r0 = source_r0 + source_h // 2 - pusher_h // 2
    if not (0 <= pusher_r0 and pusher_r0 + pusher_h <= H):
        return None

    source_c0_max = min(W - source_w - gap_target - 3, W - (2 * source_w + 2))
    source_c0_min = pusher_w + gap_push
    if source_c0_min > source_c0_max:
        return None
    source_c0 = random.randint(source_c0_min, source_c0_max)
    pusher_c0 = source_c0 - gap_push - pusher_w
    target_c0 = source_c0 + source_w + gap_target
    if pusher_c0 < 0 or target_c0 > W - 3:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[pusher_r0 : pusher_r0 + pusher_h, pusher_c0 : pusher_c0 + pusher_w] = BLUE
    grid[source_r0 : source_r0 + source_h, source_c0 : source_c0 + source_w] = ORANGE
    grid[target_r0 : target_r0 + target_len, target_c0] = GREEN

    return _transform_grid(grid, random.randrange(8))


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_canonical_case()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 15 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
