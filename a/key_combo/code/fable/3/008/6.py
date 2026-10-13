from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
SCALE_FACTORS = (2, 3)
COLORS = list(range(1, 10))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))


def _solid_rect_colors(g: np.ndarray, colors: list[int]) -> list[tuple[int, tuple[int, int, int, int]]]:
    out = []
    for color in colors:
        mask = g == color
        b = _bbox(mask)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h >= 2 and w >= 2 and int(mask.sum()) == h * w:
            out.append((int(color), b))
    return out


def _match_positions(
    g: np.ndarray,
    scaled_mask: np.ndarray,
    big_color: int,
    occ_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    height, width = g.shape
    sh, sw = scaled_mask.shape
    occ_r0, occ_c0, occ_r1, occ_c1 = occ_bbox
    big_mask = g == big_color
    big_pts = np.argwhere(big_mask)
    if big_pts.size == 0:
        return []

    positions = []
    true_count = int(scaled_mask.sum())
    visible_count = int(big_mask.sum())
    if visible_count >= true_count:
        return []

    for top_r in range(0, height - sh + 1):
        for top_c in range(0, width - sw + 1):
            ok = True
            for r, c in big_pts:
                rr = int(r) - top_r
                cc = int(c) - top_c
                if rr < 0 or rr >= sh or cc < 0 or cc >= sw or not bool(scaled_mask[rr, cc]):
                    ok = False
                    break
            if not ok:
                continue

            hidden = 0
            for rr, cc in np.argwhere(scaled_mask):
                r = top_r + int(rr)
                c = top_c + int(cc)
                inside_occ = occ_r0 <= r <= occ_r1 and occ_c0 <= c <= occ_c1
                if inside_occ:
                    hidden += 1
                elif int(g[r, c]) != int(big_color):
                    ok = False
                    break
            if not ok or hidden == 0:
                continue
            positions.append((top_r, top_c))
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded patch of a large object by upscaling its intact small blueprint.

    Input structure:
    - The grid has a black background, one large one-color object, one solid rectangular occluder whose color appears only in that rectangle, one smaller same-shape object in another color, and one unrelated distractor object.
    - The small object is an exact one-half or one-third scale copy of the complete large object mask.
    - The occluder overwrites part of the large object, so the large color shows only the visible remainder.

    Rule:
    - Find the unique small object whose mask can be enlarged by factor 2 or 3 and aligned so that every visible large-object cell is explained and every non-occluded enlarged cell matches the large color.
    - Use that unique alignment to infer which cells under the occluder belong to the large object.
    - Return a grid with the occluder bbox shape, painting inferred large-object cells in the large object's color and all other cells black.

    Key insight:
    - The repair is not based on symmetry or the occluder position; the intact smaller object is a cross-scale blueprint for the hidden part of the large object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert int(g.max()) <= 9 and int(g.min()) >= 0

    colors = sorted(int(c) for c in np.unique(g) if int(c) != BLACK)
    assert len(colors) >= 4, "expected large, small, occluder, and distractor colors"

    candidates = []
    for occ_color, occ_bbox in _solid_rect_colors(g, colors):
        other_colors = [c for c in colors if c != occ_color]
        occ_r0, occ_c0, occ_r1, occ_c1 = occ_bbox
        out_h = occ_r1 - occ_r0 + 1
        out_w = occ_c1 - occ_c0 + 1

        for small_color in other_colors:
            small_mask = g == small_color
            small = _crop(small_mask)
            if int(small.sum()) == small.size:
                continue
            for factor in SCALE_FACTORS:
                scaled = _scale(small, factor)
                for big_color in other_colors:
                    if big_color == small_color:
                        continue
                    positions = _match_positions(g, scaled, big_color, occ_bbox)
                    if len(positions) != 1:
                        continue
                    top_r, top_c = positions[0]
                    patch = np.full((out_h, out_w), BLACK, dtype=int)
                    for rr in range(out_h):
                        for cc in range(out_w):
                            sr = occ_r0 + rr - top_r
                            sc = occ_c0 + cc - top_c
                            if 0 <= sr < scaled.shape[0] and 0 <= sc < scaled.shape[1] and bool(scaled[sr, sc]):
                                patch[rr, cc] = int(big_color)
                    if np.any(patch == big_color):
                        candidates.append((occ_color, small_color, big_color, factor, top_r, top_c, patch))

    assert len(candidates) == 1, f"expected one cross-scale repair, got {len(candidates)}"
    return candidates[0][-1].copy()


def _trim(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        fringe = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    fringe.append((nr, nc))
        if not fringe:
            break
        cells.add(random.choice(fringe))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim(mask)


def _make_blueprint(factor: int) -> np.ndarray:
    for _ in range(200):
        if factor == 3:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
        else:
            h = random.randint(4, 7)
            w = random.randint(4, 7)
        area = h * w
        target = random.randint(max(5, area // 3), max(6, int(area * 0.68)))
        mask = _random_connected_mask(h, w, target)
        if min(mask.shape) < 3:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        scaled = _scale(mask, factor)
        if scaled.shape[0] <= 18 and scaled.shape[1] <= 18:
            return mask
    raise RuntimeError("failed to make blueprint")


def _candidate_locations(grid: np.ndarray, mask: np.ndarray, padding: int = 1) -> list[tuple[int, int]]:
    height, width = grid.shape
    h, w = mask.shape
    locs = []
    for r0 in range(0, height - h + 1):
        for c0 in range(0, width - w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(height, r0 + h + padding)
            cc1 = min(width, c0 + w + padding)
            if np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
                continue
            locs.append((r0, c0))
    random.shuffle(locs)
    return locs


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    pts = np.argwhere(mask)
    for rr, cc in pts:
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _all_rectangles_touching_object(
    grid_shape: tuple[int, int],
    top_r: int,
    top_c: int,
    big_mask: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    height, width = grid_shape
    obj = np.zeros(grid_shape, dtype=bool)
    obj[top_r : top_r + big_mask.shape[0], top_c : top_c + big_mask.shape[1]] = big_mask
    obj_pts = np.argwhere(obj)
    choices = []
    for rh in range(2, min(7, height) + 1):
        for rw in range(2, min(7, width) + 1):
            for r0 in range(max(0, top_r - rh + 1), min(height - rh, top_r + big_mask.shape[0] - 1) + 1):
                for c0 in range(max(0, top_c - rw + 1), min(width - rw, top_c + big_mask.shape[1] - 1) + 1):
                    r1 = r0 + rh - 1
                    c1 = c0 + rw - 1
                    rect = np.zeros(grid_shape, dtype=bool)
                    rect[r0 : r1 + 1, c0 : c1 + 1] = True
                    hidden = int((obj & rect).sum())
                    if hidden < 2:
                        continue
                    if hidden >= int(obj_pts.shape[0]) - 3:
                        continue
                    patch_has_bg = hidden < rh * rw
                    if not patch_has_bg:
                        continue
                    choices.append((r0, c0, r1, c1))
    random.shuffle(choices)
    return choices


def _random_distractor(small: np.ndarray, factor: int) -> np.ndarray:
    small_area = int(small.sum())
    for _ in range(200):
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        area = h * w
        target = random.randint(4, max(5, int(area * 0.65)))
        mask = _random_connected_mask(h, w, min(target, area - 1))
        if min(mask.shape) < 2 or int(mask.sum()) == mask.size:
            continue
        if mask.shape == small.shape and np.array_equal(mask, small):
            continue
        if int(mask.sum()) == small_area:
            continue
        if factor in SCALE_FACTORS and mask.shape == _scale(small, factor).shape and np.array_equal(mask, _scale(small, factor)):
            continue
        return mask
    raise RuntimeError("failed to make distractor")


def _expected_patch(
    big_mask: np.ndarray,
    big_top: tuple[int, int],
    occ_bbox: tuple[int, int, int, int],
    big_color: int,
) -> np.ndarray:
    occ_r0, occ_c0, occ_r1, occ_c1 = occ_bbox
    out = np.zeros((occ_r1 - occ_r0 + 1, occ_c1 - occ_c0 + 1), dtype=int)
    top_r, top_c = big_top
    for rr in range(out.shape[0]):
        for cc in range(out.shape[1]):
            br = occ_r0 + rr - top_r
            bc = occ_c0 + cc - top_c
            if 0 <= br < big_mask.shape[0] and 0 <= bc < big_mask.shape[1] and bool(big_mask[br, bc]):
                out[rr, cc] = int(big_color)
    return out


def _object_touch_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid != BLACK))


def _gen_attempt() -> np.ndarray | None:
    factor = random.choice(SCALE_FACTORS)
    small = _make_blueprint(factor)
    big = _scale(small, factor)
    big_h, big_w = big.shape

    min_h = max(20, big_h + small.shape[0] + 7)
    min_w = max(20, big_w + small.shape[1] + 7)
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    if height > 30 or width > 30:
        return None

    big_color, small_color, occ_color, distractor_color = random.sample(COLORS, 4)
    grid = np.zeros((height, width), dtype=int)

    top_r = random.randint(2, height - big_h - 2)
    top_c = random.randint(2, width - big_w - 2)
    _paste_mask(grid, big, top_r, top_c, big_color)

    rects = _all_rectangles_touching_object(grid.shape, top_r, top_c, big)
    if not rects:
        return None
    occ_bbox = random.choice(rects[: min(40, len(rects))])
    occ_r0, occ_c0, occ_r1, occ_c1 = occ_bbox
    grid[occ_r0 : occ_r1 + 1, occ_c0 : occ_c1 + 1] = occ_color

    small_locs = _candidate_locations(grid, small, padding=2)
    if not small_locs:
        return None
    sr, sc = small_locs[0]
    _paste_mask(grid, small, sr, sc, small_color)

    distractor = _random_distractor(small, factor)
    dist_locs = _candidate_locations(grid, distractor, padding=2)
    if not dist_locs:
        return None
    dr, dc = dist_locs[0]
    _paste_mask(grid, distractor, dr, dc, distractor_color)

    expected = _expected_patch(big, (top_r, top_c), occ_bbox, big_color)
    if not (np.any(expected == big_color) and np.any(expected == BLACK)):
        return None
    if int(np.count_nonzero(grid == big_color)) < max(4, int(big.sum()) // 4):
        return None
    if _object_touch_count(grid) <= int(big.sum()) + int(small.sum()):
        return None

    try:
        got = main(grid)
    except (AssertionError, Exception):
        return None
    if not np.array_equal(got, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cross-scale blueprint input")

