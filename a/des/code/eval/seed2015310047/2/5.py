from __future__ import annotations

import random

import numpy as np


RED = 2
YELLOW = 4
GREY = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _sort_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in points)


def _line_pixels(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    ra, ca = a
    rb, cb = b
    pixels: set[tuple[int, int]] = set()

    step_c = 1 if cb >= ca else -1
    for c in range(ca, cb + step_c, step_c):
        pixels.add((ra, c))

    step_r = 1 if rb >= ra else -1
    for r in range(ra, rb + step_r, step_r):
        pixels.add((r, cb))
    return pixels


def _line_counts(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]], height: int, width: int
) -> np.ndarray:
    counts = np.zeros((height, width), dtype=int)
    for a, b in pairs:
        for r, c in _line_pixels(a, b):
            counts[r, c] += 1
    return counts


def _mirror_mid_to_left(point: tuple[int, int], width: int) -> tuple[int, int]:
    r, c = point
    return int(r), int(width - 1 - c)


def _extract_layout(
    grid: np.ndarray,
) -> tuple[int, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 8 and (w - 2) % 3 == 0, "expected three equal vertical regions"

    region_w = (w - 2) // 3
    sep1 = region_w
    sep2 = 2 * region_w + 1
    bg = _most_common_color(g)
    sep_color = int(g[0, sep1])

    assert sep_color != bg
    assert np.all(g[:, sep1] == sep_color) and np.all(
        g[:, sep2] == sep_color
    ), "expected two full separator columns"

    left = g[:, :region_w]
    mid = g[:, sep1 + 1 : sep2]
    right = g[:, sep2 + 1 :]
    assert right.shape == (h, region_w)
    assert np.all(right == bg), "right region must be blank in the input"

    fg_colors = set(map(int, np.unique(g))) - {bg, sep_color}
    assert len(fg_colors) == 1, "expected one drawing color besides background/separators"
    return h, w, region_w, sep1, sep2, bg, left, mid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the left pattern with the mirrored middle pattern, then connect
    mismatching pixels inside the blank right region.

    The grid is split into three equal-width vertical regions by two grey
    separator columns. After mirroring the left region horizontally, compare it
    cell-by-cell with the middle region:
    - A-type mismatch: pixel exists in the left region but not at the mirrored
      middle position. Record the left-region coordinates.
    - B-type mismatch: pixel exists in the middle region but not at the mirrored
      left position. Record the middle-region coordinates.

    Sort A and B points top-to-bottom then left-to-right, pair them in order,
    and for each pair draw an L-shape in the right region: horizontal first,
    then vertical. Use red for normal line pixels and yellow for pixels covered
    by two or more L-shapes.
    """
    h, _w, region_w, _sep1, sep2, bg, left, mid = _extract_layout(grid)
    left_mask = left != bg
    mid_mask = mid != bg

    a_points: list[tuple[int, int]] = []
    b_points: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(region_w):
            left_on = bool(left_mask[r, c])
            mc = region_w - 1 - c
            mid_on = bool(mid_mask[r, mc])
            if left_on and not mid_on:
                a_points.append((r, c))
            elif mid_on and not left_on:
                b_points.append((r, mc))

    a_points = _sort_points(a_points)
    b_points = _sort_points(b_points)
    assert a_points and len(a_points) == len(
        b_points
    ), "difference points must form non-empty one-to-one pairs"

    pairs = list(zip(a_points, b_points))
    counts = _line_counts(pairs, h, region_w)

    out = np.asarray(grid, dtype=int).copy()
    out[:, sep2 + 1 :] = bg
    out[:, sep2 + 1 :][counts == 1] = RED
    out[:, sep2 + 1 :][counts >= 2] = YELLOW
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _place_patch(region_h: int, region_w: int) -> np.ndarray | None:
    patch_h = random.randint(4, max(4, region_h - 1))
    patch_w = random.randint(4, max(4, region_w - 1))
    max_area = max(6, (patch_h * patch_w * 3) // 5)
    min_area = max(6, (patch_h * patch_w) // 4)
    if min_area >= patch_h * patch_w:
        return None
    area = random.randint(min_area, min(max_area, patch_h * patch_w - 1))
    patch = _random_connected_mask(patch_h, patch_w, area)
    if patch is None:
        return None

    top_lo = 1 if region_h - patch_h >= 2 else 0
    top_hi = region_h - patch_h - (1 if region_h - patch_h >= 2 else 0)
    left_lo = 1 if region_w - patch_w >= 2 else 0
    left_hi = region_w - patch_w - (1 if region_w - patch_w >= 2 else 0)
    if top_lo > top_hi or left_lo > left_hi:
        return None

    top = random.randint(top_lo, top_hi)
    left = random.randint(left_lo, left_hi)
    base = np.zeros((region_h, region_w), dtype=bool)
    base[top : top + patch_h, left : left + patch_w] = patch
    return base


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    pts: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    pts.add((r, c))
                    break
    return list(pts)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _sample_pairs(base: np.ndarray, crossing: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = base.shape
    left_front = _frontier(base)
    mid_front = _frontier(np.fliplr(base))
    if not left_front or not mid_front:
        return None

    for _ in range(400):
        max_k = min(3, len(left_front), len(mid_front))
        if max_k < 1:
            return None
        k = 2 if crossing else random.randint(1, max_k)
        if max_k < k:
            continue

        a_pick = random.sample(left_front, k)
        b_candidates = [
            pt
            for pt in mid_front
            if _mirror_mid_to_left(pt, w) not in a_pick and (pt[0], w - 1 - pt[1]) not in a_pick
        ]
        if len(b_candidates) < k:
            continue

        if crossing and k == 2:
            a_sorted = _sort_points(a_pick)
            if a_sorted[0][0] == a_sorted[1][0]:
                continue
            orientation = random.choice(("above", "below"))
            if orientation == "above":
                filtered = [pt for pt in b_candidates if pt[0] < a_sorted[0][0]]
            else:
                filtered = [pt for pt in b_candidates if pt[0] > a_sorted[1][0]]
            if len(filtered) < 2:
                continue
            b_pick = random.sample(filtered, 2)
        else:
            b_pick = random.sample(b_candidates, k)

        a_sorted = _sort_points(a_pick)
        b_sorted = _sort_points(b_pick)
        if any((r, w - 1 - c) in b_sorted for r, c in a_sorted):
            continue

        counts = _line_counts(list(zip(a_sorted, b_sorted)), h, w)
        has_crossing = bool((counts >= 2).any())
        if has_crossing != crossing:
            continue
        return a_sorted, b_sorted
    return None


def _render_input(
    base: np.ndarray,
    a_points: list[tuple[int, int]],
    b_points: list[tuple[int, int]],
    bg: int,
    pattern_color: int,
) -> np.ndarray:
    h, region_w = base.shape
    grid = np.full((h, 3 * region_w + 2), bg, dtype=int)
    grid[:, region_w] = GREY
    grid[:, 2 * region_w + 1] = GREY

    left_mask = base.copy()
    for r, c in a_points:
        left_mask[r, c] = True

    mid_mask = np.fliplr(base)
    for r, c in b_points:
        mid_mask[r, c] = True

    assert _connected(left_mask) and _connected(mid_mask)
    grid[:, :region_w][left_mask] = pattern_color
    grid[:, region_w + 1 : 2 * region_w + 1][mid_mask] = pattern_color
    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray:
    h, _w, region_w, _sep1, sep2, bg, left, mid = _extract_layout(inp)
    a_points: list[tuple[int, int]] = []
    b_points: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(region_w):
            mc = region_w - 1 - c
            left_on = bool(left[r, c] != bg)
            mid_on = bool(mid[r, mc] != bg)
            if left_on and not mid_on:
                a_points.append((r, c))
            elif mid_on and not left_on:
                b_points.append((r, mc))
    counts = _line_counts(list(zip(_sort_points(a_points), _sort_points(b_points))), h, region_w)
    out = inp.copy()
    out[:, sep2 + 1 :] = bg
    out[:, sep2 + 1 :][counts == 1] = RED
    out[:, sep2 + 1 :][counts >= 2] = YELLOW
    return out


def generate_input() -> np.ndarray:
    """
    Randomly generate three-region mirror-comparison tasks.

    Variation axes:
    - grid height and region width
    - background and pattern colors
    - base shape size, footprint, and placement
    - number and location of A/B mismatch points
    - crossing vs non-crossing output lines
    """
    bg_choices = [0, 1, 3, 6, 7, 8, 9]
    pattern_choices = [1, 3, 6, 7, 8, 9]

    for _ in range(500):
        region_h = random.randint(8, 15)
        region_w = random.randint(7, 12)
        base = _place_patch(region_h, region_w)
        if base is None:
            continue

        crossing = bool(random.getrandbits(1))
        sampled = _sample_pairs(base, crossing)
        if sampled is None:
            continue
        a_points, b_points = sampled

        bg = random.choice(bg_choices)
        pattern_pool = [c for c in pattern_choices if c != bg and c not in (RED, YELLOW, GREY)]
        pattern_color = random.choice(pattern_pool)

        inp = _render_input(base, a_points, b_points, bg, pattern_color)
        expected = _expected_output(inp)

        try:
            actual = main(inp)
        except AssertionError:
            continue

        if not np.array_equal(actual, expected):
            continue
        has_yellow = bool(np.any(actual[:, 2 * region_w + 2 :] == YELLOW))
        if has_yellow != crossing:
            continue
        return inp

    raise RuntimeError("failed to generate a valid ARC sample")
