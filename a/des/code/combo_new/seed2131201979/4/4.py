from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5

# ARC palette order from the task description:
# blue, red, green, yellow, grey, magenta, orange, azure, maroon
PALETTE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE_RANK = {color: idx for idx, color in enumerate(PALETTE_ORDER)}
PATH_COLOR_POOL = [color for color in PALETTE_ORDER if color != GREY]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _ordered_pair(color_a: int, color_b: int) -> tuple[int, int]:
    rank_a = PALETTE_RANK.get(int(color_a), len(PALETTE_RANK))
    rank_b = PALETTE_RANK.get(int(color_b), len(PALETTE_RANK))
    if rank_a <= rank_b:
        return int(color_a), int(color_b)
    return int(color_b), int(color_a)


def _crossing_pairs(grid: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    pairs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue

            up = int(grid[r - 1, c]) if r > 0 and int(grid[r - 1, c]) not in (BG, GREY) else None
            down = int(grid[r + 1, c]) if r + 1 < h and int(grid[r + 1, c]) not in (BG, GREY) else None
            left = int(grid[r, c - 1]) if c > 0 and int(grid[r, c - 1]) not in (BG, GREY) else None
            right = int(grid[r, c + 1]) if c + 1 < w and int(grid[r, c + 1]) not in (BG, GREY) else None

            vertical = up if up is not None and up == down else None
            horizontal = left if left is not None and left == right else None
            if vertical is None or horizontal is None or vertical == horizontal:
                continue
            pairs.add(_ordered_pair(vertical, horizontal))
    return pairs


def _enclosed_regions(grid: np.ndarray) -> list[tuple[np.ndarray, set[tuple[int, int]]]]:
    h, w = grid.shape
    regions: list[tuple[np.ndarray, set[tuple[int, int]]]] = []
    for comp in _components(grid == BG):
        pts = [(int(r), int(c)) for r, c in comp]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        comp_set = set(pts)
        border: set[tuple[int, int]] = set()
        valid = True
        for r, c in pts:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in comp_set:
                    continue
                if int(grid[nr, nc]) != GREY:
                    valid = False
                    break
                border.add((int(nr), int(nc)))
            if not valid:
                break
        if valid and border:
            regions.append((comp, border))
    return regions


def _touching_path_colors(grid: np.ndarray, border: set[tuple[int, int]]) -> list[int]:
    h, w = grid.shape
    colors: set[int] = set()
    for r, c in border:
        for nr, nc in _neighbors4(r, c, h, w):
            value = int(grid[nr, nc])
            if value not in (BG, GREY):
                colors.add(value)
    return sorted(colors, key=lambda color: PALETTE_RANK.get(int(color), len(PALETTE_RANK)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey-framed enclosed region based on whether its two touching paths intersect elsewhere.

    Input structure:
    - The background is black.
    - Several one-pixel-wide colored paths run through the grid as orthogonal chains, and true path crossings are marked by grey overlap pixels.
    - Several hollow grey rectangular frames enclose black interior regions, and each region is touched from outside by exactly two differently colored paths.

    Rule:
    - Detect every pair of path colors that shares a true grey crossing pixel, where one color continues vertically through the grey cell and another continues horizontally.
    - Detect each black region completely enclosed by a grey frame and find the two path colors that 4-neighbor its border from outside.
    - If those two path colors intersect anywhere in the grid, fill the region with the earlier color in the fixed ARC palette order; otherwise fill it with the later color.

    Key insight:
    - The local fill decision depends on a global topological fact: whether the two border-touching paths meet at any grey crossing somewhere else in the grid.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    crossings = _crossing_pairs(g)

    for comp, border in _enclosed_regions(g):
        colors = _touching_path_colors(g, border)
        if len(colors) != 2:
            continue
        first, second = _ordered_pair(colors[0], colors[1])
        fill_color = first if (first, second) in crossings else second
        out[comp[:, 0], comp[:, 1]] = int(fill_color)

    return out


def _draw_rectangle_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _apply_random_symmetry(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    a = inp.copy()
    b = out.copy()
    if rng.random() < 0.5:
        a = np.fliplr(a).copy()
        b = np.fliplr(b).copy()
    if rng.random() < 0.5:
        a = np.flipud(a).copy()
        b = np.flipud(b).copy()
    if rng.random() < 0.5:
        a = a.T.copy()
        b = b.T.copy()
    return a, b


def _paint_case(
    h: int,
    w: int,
    frame1: tuple[int, int, int, int],
    frame2: tuple[int, int, int, int],
    cross_geom: tuple[int, int, int, int],
    noncross_geom: tuple[int, int],
    colors: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((h, w), BG, dtype=int)

    r1, c1, fh1, fw1 = frame1
    r2, c2, fh2, fw2 = frame2
    touch_row1, touch_col1, cross_row, cross_col = cross_geom
    touch_row2, touch_col2 = noncross_geom
    color_a, color_b, color_c, color_d = map(int, colors)

    _draw_rectangle_border(grid, r1, c1, fh1, fw1, GREY)
    _draw_rectangle_border(grid, r2, c2, fh2, fw2, GREY)

    # Crossing pair: vertical-then-horizontal and horizontal-then-vertical.
    grid[0 : touch_row1 + 1, cross_col] = color_a
    grid[touch_row1, cross_col : c1] = color_a

    grid[cross_row, 0 : touch_col1 + 1] = color_b
    grid[cross_row : r1, touch_col1] = color_b

    grid[cross_row, cross_col] = GREY

    # Non-crossing pair: one approaches from below, the other from the right.
    grid[r2 + fh2 : h, touch_col2] = color_c
    grid[touch_row2, c2 + fw2 : w] = color_d

    out = grid.copy()
    cross_fill = _ordered_pair(color_a, color_b)[0]
    noncross_fill = _ordered_pair(color_c, color_d)[1]
    out[r1 + 1 : r1 + fh1 - 1, c1 + 1 : c1 + fw1 - 1] = cross_fill
    out[r2 + 1 : r2 + fh2 - 1, c2 + 1 : c2 + fw2 - 1] = noncross_fill
    return grid, out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        h = rng.randint(22, 30)
        w = rng.randint(22, 30)

        fh1 = rng.randint(4, 6)
        fw1 = rng.randint(4, 6)
        fh2 = rng.randint(4, 6)
        fw2 = rng.randint(4, 6)

        r1_min = 4
        r1_max = h // 2 - fh1 - 1
        c1_min = max(8, w // 2)
        c1_max = w - fw1 - 3
        r2_min = h // 2 + 1
        r2_max = h - fh2 - 3
        c2_min = 2
        c2_max = w // 2 - fw2 - 1
        if r1_min > r1_max or c1_min > c1_max or r2_min > r2_max or c2_min > c2_max:
            continue

        r1 = rng.randint(r1_min, r1_max)
        c1 = rng.randint(c1_min, c1_max)
        r2 = rng.randint(r2_min, r2_max)
        c2 = rng.randint(c2_min, c2_max)

        if r2 <= r1 + fh1 + 2 or c2 + fw2 + 2 >= c1:
            continue

        touch_row1 = rng.randint(r1 + 1, r1 + fh1 - 2)
        touch_col1 = rng.randint(c1 + 1, c1 + fw1 - 2)
        cross_row = rng.randint(1, r1 - 3)
        cross_col = rng.randint(1, c1 - 3)

        touch_row2 = rng.randint(r2 + 1, r2 + fh2 - 2)
        touch_col2 = rng.randint(c2 + 1, c2 + fw2 - 2)

        colors = tuple(rng.sample(PATH_COLOR_POOL, 4))
        inp, out = _paint_case(
            h,
            w,
            (r1, c1, fh1, fw1),
            (r2, c2, fh2, fw2),
            (touch_row1, touch_col1, cross_row, cross_col),
            (touch_row2, touch_col2),
            colors,
        )
        inp, out = _apply_random_symmetry(inp, out, rng)

        regions = _enclosed_regions(inp)
        if len(regions) != 2:
            continue

        crossings = _crossing_pairs(inp)
        cross_regions = 0
        noncross_regions = 0
        valid = True
        for _comp, border in regions:
            region_colors = _touching_path_colors(inp, border)
            if len(region_colors) != 2:
                valid = False
                break
            pair = _ordered_pair(region_colors[0], region_colors[1])
            if pair in crossings:
                cross_regions += 1
            else:
                noncross_regions += 1
        if not valid or cross_regions != 1 or noncross_regions != 1:
            continue

        if not np.array_equal(main(inp), out):
            continue
        return inp, out

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    rng = random.Random()
    inp, _ = _generate_case(rng)
    return inp
