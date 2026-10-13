from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _rectangle_border_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(points) != 2 * (h + w) - 4:
        return None
    edge = set(points)
    for c in range(c0, c1 + 1):
        if (r0, c) not in edge or (r1, c) not in edge:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in edge or (r, c1) not in edge:
            return None
    return r0, c0, r1, c1


def _find_fences(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = None
    best_rects: list[tuple[int, int, int, int]] = []
    best_score = None

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        rects: list[tuple[int, int, int, int]] = []
        for comp in _components(grid == color, connectivity=4):
            bbox = _rectangle_border_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            seeds = [
                int(v)
                for v in interior.ravel()
                if int(v) != bg and int(v) != color
            ]
            if len(seeds) == 1:
                rects.append(bbox)
        score = (len(rects), sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects))
        if score[0] == 0:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
            best_rects = rects

    assert best_color is not None and best_rects, "expected at least one fenced region"
    best_rects.sort()
    return int(best_color), best_rects


def _extract_templates(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    fence_bottom: int,
) -> dict[int, tuple[tuple[int, int], ...]]:
    template_map: dict[int, tuple[tuple[int, int], ...]] = {}
    mask = (grid != bg) & (grid != wall_color)
    for comp in _components(mask, connectivity=8):
        if min(r for r, _ in comp) <= fence_bottom:
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        cr, cc = r0 + 1, c0 + 1
        if int(grid[cr, cc]) != color:
            continue
        valid = True
        dirs: list[tuple[int, int]] = []
        for rr in range(r0, r1 + 1):
            for cc2 in range(c0, c1 + 1):
                v = int(grid[rr, cc2])
                if v not in (bg, color):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        for r, c in comp:
            dr, dc = r - cr, c - cc
            if dr == 0 and dc == 0:
                continue
            dirs.append((dr, dc))
        if dirs:
            template_map[color] = tuple(sorted(dirs))

    assert template_map, "expected at least one 3x3 growth template"
    return template_map


def _grow_points(
    seed: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
    layers: int,
) -> set[tuple[int, int]]:
    seen = {seed}
    frontier = {seed}
    for _ in range(layers):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if rmin <= nr <= rmax and cmin <= nc <= cmax:
                    nxt.add((nr, nc))
        seen.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed inside its fenced region using the bottom template that matches its color.

    Input structure:
    - A dominant orange background contains several grey one-pixel rectangular fences.
    - Each fence encloses exactly one non-orange, non-grey seed pixel.
    - Below all fences sits a legend made of small 3x3 same-color templates on the orange background.

    Rule:
    - Identify the grey fenced rectangles and the single seed inside each one.
    - Read each 3x3 legend template by color: the template center gives the seed color and every non-orange neighbor of that center gives one allowed growth direction.
    - For a fence with interior size h x w, iteratively expand from its seed for min(h, w) - 1 layers using the matched direction set, clipping any step that would leave the fence interior.
    - Paint all reached interior cells with the seed color, while keeping fences and legend templates unchanged.

    Key insight:
    - Color links each seed to one specific direction stencil, and the enclosing rectangle size supplies the number of expansion rounds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    wall_color, fences = _find_fences(g, bg)
    fence_bottom = max(r1 for r0, c0, r1, c1 in fences)
    templates = _extract_templates(g, bg, wall_color, fence_bottom)

    out = g.copy()
    for r0, c0, r1, c1 in fences:
        interior = [
            (r, c, int(g[r, c]))
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) != bg and int(g[r, c]) != wall_color
        ]
        assert len(interior) == 1, "each fenced region must contain exactly one seed"
        sr, sc, color = interior[0]
        assert color in templates, "every seed color must have a matching template"
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        layers = min(inner_h, inner_w) - 1
        grown = _grow_points(
            (sr, sc),
            templates[color],
            r0 + 1,
            r1 - 1,
            c0 + 1,
            c1 - 1,
            layers,
        )
        for r, c in grown:
            out[r, c] = color

    return out


BG_COLOR = 7
WALL_COLOR = 5
SEED_PALETTE = (1, 2, 3, 4, 6, 8, 9)
PATTERN_LIBRARY = (
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-1, -1), (0, 1), (1, 0)),
    ((-1, 1), (0, -1), (1, 0)),
    ((-1, 0), (0, -1), (1, 1)),
    ((-1, 0), (0, 1), (1, -1)),
    ((-1, -1), (-1, 0), (0, 1), (1, 0)),
    ((-1, 0), (-1, 1), (0, -1), (1, 0)),
)
GEN_RETRIES = 400


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    slots = total + parts - 1
    cuts = sorted(random.sample(range(slots), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [slots]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_template(
    grid: np.ndarray,
    top: int,
    left: int,
    color: int,
    dirs: tuple[tuple[int, int], ...],
) -> None:
    center = (top + 1, left + 1)
    grid[center] = color
    for dr, dc in dirs:
        grid[center[0] + dr, center[1] + dc] = color


def _local_growth_size(
    seed_r: int,
    seed_c: int,
    dirs: tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
) -> int:
    points = _grow_points(
        (seed_r, seed_c),
        dirs,
        0,
        inner_h - 1,
        0,
        inner_w - 1,
        min(inner_h, inner_w) - 1,
    )
    return len(points)


def _generate_attempt() -> np.ndarray | None:
    template_count = random.randint(2, 4)
    fence_count = random.randint(template_count, min(4, template_count + 1))

    min_width = max(16, 4 * template_count + 5, 14 + 2 * fence_count)
    min_height = {2: 16, 3: 20, 4: 23}[fence_count]
    if min_width > 30 or min_height > 30:
        return None

    width = random.randint(min_width, 30)
    height = random.randint(min_height, 30)
    template_top = height - 3

    colors = random.sample(SEED_PALETTE, template_count)
    patterns = random.sample(PATTERN_LIBRARY, template_count)
    template_specs = list(zip(colors, patterns, strict=True))
    random.shuffle(template_specs)
    dirs_by_color = {color: dirs for color, dirs in template_specs}

    base_template_width = 3 * template_count + (template_count - 1)
    slack = width - base_template_width
    parts = _random_composition(slack, template_count + 1)
    template_cols: list[int] = []
    col = parts[0]
    for i in range(template_count):
        template_cols.append(col)
        col += 3
        if i < template_count - 1:
            col += 1 + parts[i + 1]

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for (color, dirs), left in zip(template_specs, template_cols, strict=True):
        _draw_template(grid, template_top, left, color, dirs)

    fence_colors = colors[:]
    while len(fence_colors) < fence_count:
        fence_colors.append(random.choice(colors))
    random.shuffle(fence_colors)

    size_limits = {
        2: ((3, 7), (3, 8)),
        3: ((3, 6), (3, 7)),
        4: ((3, 5), (3, 6)),
    }
    (min_inner_h, max_inner_h), (min_inner_w, max_inner_w) = size_limits[fence_count]

    specs = []
    for color in fence_colors:
        inner_h = random.randint(min_inner_h, max_inner_h)
        inner_w = random.randint(min_inner_w, max_inner_w)
        specs.append(
            {
                "color": color,
                "dirs": dirs_by_color[color],
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)

    placed: list[dict[str, int | tuple[int, int, int, int] | tuple[int, int]]] = []
    boxes: list[tuple[int, int, int, int]] = []

    for spec in specs:
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        max_r0 = template_top - outer_h - 1
        max_c0 = width - outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        placed_here = False
        for _ in range(200):
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(not _boxes_separated(box, other, gap=1) for other in boxes):
                continue

            seed_options = [
                (rr, cc)
                for rr in range(int(spec["inner_h"]))
                for cc in range(int(spec["inner_w"]))
            ]
            random.shuffle(seed_options)
            seed = None
            for rr, cc in seed_options:
                if _local_growth_size(rr, cc, spec["dirs"], int(spec["inner_h"]), int(spec["inner_w"])) > 1:
                    seed = (rr, cc)
                    break
            if seed is None:
                continue

            placed.append(
                {
                    "color": int(spec["color"]),
                    "box": box,
                    "seed": seed,
                    "inner_h": int(spec["inner_h"]),
                    "inner_w": int(spec["inner_w"]),
                }
            )
            boxes.append(box)
            placed_here = True
            break
        if not placed_here:
            return None

    for item in placed:
        r0, c0, r1, c1 = item["box"]
        for c in range(c0, c1 + 1):
            grid[r0, c] = WALL_COLOR
            grid[r1, c] = WALL_COLOR
        for r in range(r0, r1 + 1):
            grid[r, c0] = WALL_COLOR
            grid[r, c1] = WALL_COLOR
        sr = r0 + 1 + item["seed"][0]
        sc = c0 + 1 + item["seed"][1]
        grid[sr, sc] = int(item["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid growth-template puzzle")
