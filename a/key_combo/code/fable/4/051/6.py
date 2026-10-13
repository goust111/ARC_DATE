from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            border = set(_rect_border_coords((r0, c0, r1, c1)))
            comp_set = set(comp)
            if comp_set != border:
                continue
            if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
                continue
            rectangles.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "border": tuple(sorted(border)),
                }
            )
    return rectangles


def _marker_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert (on_top + on_bottom + on_left + on_right) == 1, "marker must lie on exactly one border side"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast inward border rays that swap colors with each hollow rectangle they enter.

    Input structure:
    - The background is black.
    - Interior foreground objects are pairwise disjoint hollow monochrome rectangles.
    - Additional non-black border pixels are edge markers; each marker emits a straight inward ray.

    Rule:
    - Keep the original grid as the output base.
    - Each marker paints only background cells along its inward row or column, starting with the marker's color.
    - Whenever a ray enters a hollow rectangle, recolor that rectangle's full border to the ray's current color, then continue the ray using the rectangle's original color; exiting the same rectangle does not trigger another swap.

    Key insight:
    - A hollow rectangle is a color exchanger, not an obstacle: the rectangle receives the incoming ray color, and the ray carries away the rectangle's previous color for the rest of the path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    rectangles = _find_rectangles(g)
    border_lookup: dict[tuple[int, int], int] = {}
    rect_colors: list[int] = []
    rect_borders: list[tuple[tuple[int, int], ...]] = []
    for idx, rect in enumerate(rectangles):
        rect_colors.append(int(rect["color"]))
        rect_borders.append(rect["border"])
        for cell in rect["border"]:
            border_lookup[cell] = idx

    marker_cells: list[tuple[int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if int(g[r, c]) == BG:
                continue
            if (r, c) in border_lookup:
                continue
            marker_cells.append((r, c))

    for r, c in sorted(marker_cells):
        color = int(g[r, c])
        dr, dc = _marker_direction(g.shape, r, c)
        rr, cc = r + dr, c + dc
        inside_rect: int | None = None
        while _in_bounds(g.shape, rr, cc):
            rect_id = border_lookup.get((rr, cc))
            if rect_id is not None:
                if inside_rect is None:
                    for br, bc in rect_borders[rect_id]:
                        out[br, bc] = color
                    inside_rect = rect_id
                    color = rect_colors[rect_id]
                else:
                    assert inside_rect == rect_id, "ray path crossed overlapping rectangles"
                    inside_rect = None
            elif int(g[rr, cc]) == BG:
                out[rr, cc] = color
            rr += dr
            cc += dc

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_spaced_values(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    random.shuffle(values)
    picked: list[int] = []
    for v in values:
        if all(abs(v - u) >= gap for u in picked):
            picked.append(v)
            if len(picked) == count:
                return sorted(picked)
    return None


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    return not separated


def _paint_hollow_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_vertical_layout() -> np.ndarray | None:
    h = random.randint(20, 29)
    w = random.randint(18, 28)
    n_rays = 3 if random.random() < 0.7 else 2
    ray_cols = _sample_spaced_values(3, w - 4, n_rays, gap=7)
    if ray_cols is None:
        return None

    if n_rays == 3:
        rect_counts = [0, 1, 2]
        random.shuffle(rect_counts)
    else:
        rect_counts = random.choice([[0, 1], [0, 2], [1, 2]])
        random.shuffle(rect_counts)

    sides = [random.choice(["top", "bottom"]) for _ in range(n_rays)]
    extra_rects = 1 if random.random() < 0.8 else 0
    total_colors = n_rays + sum(rect_counts) + extra_rects
    colors = random.sample(PALETTE, total_colors)

    grid = np.zeros((h, w), dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    color_index = 0

    for col, side in zip(ray_cols, sides):
        marker_row = 0 if side == "top" else h - 1
        grid[marker_row, col] = colors[color_index]
        color_index += 1

    for col, count in zip(ray_cols, rect_counts):
        if count == 0:
            continue
        widths = [random.choice([3, 5]) for _ in range(count)]
        heights = [random.randint(4, 6) for _ in range(count)]
        margin = 2
        gap = 2
        slack = h - 2 * margin - sum(heights) - gap * (count - 1)
        if slack < 0:
            return None
        extras = _rand_composition(slack, count + 1)
        cursor = margin + extras[0]
        for i in range(count):
            height = heights[i]
            width = widths[i]
            d = width // 2
            bbox = (cursor, col - d, cursor + height - 1, col + d)
            if any(_too_close(bbox, other) for other in placed):
                return None
            placed.append(bbox)
            _paint_hollow_rect(grid, bbox, colors[color_index])
            color_index += 1
            cursor += height + gap + extras[i + 1]

    for _ in range(extra_rects):
        success = False
        for _attempt in range(80):
            width = random.choice([3, 5])
            d = width // 2
            valid_centers = [
                c
                for c in range(3, w - 3)
                if all(not (c - d <= ray_col <= c + d) for ray_col in ray_cols)
            ]
            if not valid_centers:
                break
            center = random.choice(valid_centers)
            height = random.randint(4, 6)
            r0 = random.randint(2, h - height - 2)
            bbox = (r0, center - d, r0 + height - 1, center + d)
            if any(_too_close(bbox, other) for other in placed):
                continue
            placed.append(bbox)
            _paint_hollow_rect(grid, bbox, colors[color_index])
            color_index += 1
            success = True
            break
        if not success:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    grid = _build_vertical_layout()
    if grid is None:
        return None
    if random.random() < 0.5:
        grid = grid.T.copy()

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(np.unique(grid)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
