from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GREY = 5
AZURE = 8
COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
LEGEND_HEIGHT = 4
LEGEND_WIDTH = 5
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    candidates: list[dict[int, int]] = []
    for comp in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        if GREY not in colors:
            continue

        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_HEIGHT, LEGEND_WIDTH):
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        fill_col = None
        mapping: dict[int, int] = {}
        ok = True
        for idx in range(LEGEND_HEIGHT):
            row = patch[idx]
            color_cols = np.where((row != bg) & (row != GREY))[0]
            grey_cols = np.where(row == GREY)[0]
            if color_cols.size != 1:
                ok = False
                break
            this_fill_col = int(color_cols[0])
            if fill_col is None:
                fill_col = this_fill_col
            elif this_fill_col != fill_col:
                ok = False
                break
            expected_grey = list(range(this_fill_col - (idx + 1), this_fill_col))
            if list(map(int, grey_cols)) != expected_grey:
                ok = False
                break
            if np.any(row[: expected_grey[0]] != bg):
                ok = False
                break
            if np.any(row[this_fill_col + 1 :] != bg):
                ok = False
                break
            mapping[idx + 1] = int(row[this_fill_col])

        if ok and len(mapping) == 4:
            candidates.append(mapping)

    assert candidates, "legend component not found"
    return candidates[0]


def _mono_non_bg_color(values: np.ndarray, bg: int) -> int | None:
    colors = set(map(int, np.unique(values)))
    colors.discard(bg)
    if len(colors) != 1:
        return None
    return int(next(iter(colors)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed rectangular region using the legend entry for its number of border colors.

    Input structure:
    - The background is Azure.
    - The grid contains horizontal and vertical 1-cell-thick colored line segments whose endpoints connect to form some closed rectangles.
    - One corner contains a vertical legend: each row has a single fill-color pixel with 1, 2, 3, or 4 Grey index pixels immediately to its left.

    Rule:
    - Read the corner legend from top to bottom as the mapping for regions bordered by 1, 2, 3, and 4 distinct edge colors.
    - Find every Azure region that is fully enclosed by four axis-aligned line sides and whose interior is a solid rectangle.
    - For each such region, determine the single color on each of its top, right, bottom, and left sides, count how many distinct side colors appear, and fill the Azure interior with the mapped legend color.
    - Leave the boundary lines, the legend, and any not-fully-closed areas unchanged.

    Key insight:
    - The task links two separate observations: the corner marker encodes a count-to-color lookup table, and each enclosed rectangle must first be classified by how many different colors its four sides use.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    legend = _parse_legend(g, bg)
    out = g.copy()

    for comp in _components(g == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue

        top = _mono_non_bg_color(g[r0 - 1, c0 : c1 + 1], bg)
        bottom = _mono_non_bg_color(g[r1 + 1, c0 : c1 + 1], bg)
        left = _mono_non_bg_color(g[r0 : r1 + 1, c0 - 1], bg)
        right = _mono_non_bg_color(g[r0 : r1 + 1, c1 + 1], bg)
        if None in (top, bottom, left, right):
            continue

        fill_color = legend[len({int(top), int(bottom), int(left), int(right)})]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


def _legend_bbox(shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - LEGEND_WIDTH
    elif corner == "bl":
        r0, c0 = h - LEGEND_HEIGHT, 0
    else:
        r0, c0 = h - LEGEND_HEIGHT, w - LEGEND_WIDTH
    return r0, c0, r0 + LEGEND_HEIGHT - 1, c0 + LEGEND_WIDTH - 1


def _paint_legend(grid: np.ndarray, mapping: dict[int, int], corner: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _legend_bbox(grid.shape, corner)
    fill_col = c1
    for count in range(1, 5):
        rr = r0 + count - 1
        grid[rr, fill_col - count : fill_col] = GREY
        grid[rr, fill_col] = mapping[count]
    return r0, c0, r1, c1


def _reserve(occupied: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + box_h + pad)
    cc1 = min(w, c0 + box_w + pad)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _find_placement(occupied: np.ndarray, box_h: int, box_w: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = occupied.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(1, h - box_h):
        for c0 in range(1, w - box_w):
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + box_h + 1
            cc1 = c0 + box_w + 1
            if not np.any(occupied[rr0:rr1, cc0:cc1]):
                spots.append((r0, c0))
    if not spots:
        return None
    return rng.choice(spots)


def _sample_side_colors(distinct_count: int, rng: random.Random) -> tuple[int, int, int, int]:
    chosen = rng.sample(COLOR_POOL, distinct_count)
    sides = [None, None, None, None]
    forced_slots = rng.sample(range(4), distinct_count)
    for color, slot in zip(chosen, forced_slots, strict=True):
        sides[slot] = color
    for idx in range(4):
        if sides[idx] is None:
            sides[idx] = rng.choice(chosen)
    return tuple(int(v) for v in sides)


def _draw_closed_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side_colors: tuple[int, int, int, int],
) -> None:
    top, right, bottom, left = side_colors
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = top
    grid[r1, c0 : c1 + 1] = bottom
    grid[r0 + 1 : r1, c0] = left
    grid[r0 + 1 : r1, c1] = right


def _draw_open_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side_colors: tuple[int, int, int, int],
    gap_side: str,
) -> None:
    top, right, bottom, left = side_colors
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    if gap_side != "top":
        grid[r0, c0 : c1 + 1] = top
    if gap_side != "bottom":
        grid[r1, c0 : c1 + 1] = bottom
    if gap_side != "left":
        grid[r0 + 1 : r1, c0] = left
    if gap_side != "right":
        grid[r0 + 1 : r1, c1] = right


def _expected_output(grid: np.ndarray, rects: list[dict], legend: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        inner_h = int(rect["inner_h"])
        inner_w = int(rect["inner_w"])
        count = int(rect["count"])
        out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = legend[count]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), AZURE, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_colors = rng.sample(COLOR_POOL, 4)
    legend = {count: legend_colors[count - 1] for count in range(1, 5)}
    legend_corner = rng.choice(("tl", "tr", "bl", "br"))
    lr0, lc0, lr1, lc1 = _paint_legend(grid, legend, legend_corner)
    _reserve(occupied, lr0, lc0, lr1 - lr0 + 1, lc1 - lc0 + 1, pad=1)

    rect_count = rng.randint(2, 4)
    closed_counts = rng.sample((1, 2, 3, 4), rect_count)
    rects: list[dict] = []
    for count in closed_counts:
        placed = False
        for _ in range(120):
            inner_h = rng.randint(2, 6)
            inner_w = rng.randint(2, 6)
            box_h = inner_h + 2
            box_w = inner_w + 2
            pos = _find_placement(occupied, box_h, box_w, rng)
            if pos is None:
                break
            r0, c0 = pos
            if not _reserve(occupied, r0, c0, box_h, box_w, pad=1):
                continue
            side_colors = _sample_side_colors(int(count), rng)
            _draw_closed_rectangle(grid, r0, c0, inner_h, inner_w, side_colors)
            rects.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "count": count,
                }
            )
            placed = True
            break
        if not placed:
            return None

    open_count = rng.randint(0, 2) if rng.random() < 0.75 else 0
    for _ in range(open_count):
        for _ in range(80):
            inner_h = rng.randint(2, 6)
            inner_w = rng.randint(2, 6)
            box_h = inner_h + 2
            box_w = inner_w + 2
            pos = _find_placement(occupied, box_h, box_w, rng)
            if pos is None:
                break
            r0, c0 = pos
            if not _reserve(occupied, r0, c0, box_h, box_w, pad=1):
                continue
            side_colors = _sample_side_colors(rng.randint(1, 4), rng)
            gap_side = rng.choice(("top", "right", "bottom", "left"))
            _draw_open_rectangle(grid, r0, c0, inner_h, inner_w, side_colors, gap_side)
            break

    expected = _expected_output(grid, rects, legend)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid input after retries")
