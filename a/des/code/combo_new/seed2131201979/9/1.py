from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
REGION_RETRIES = 80


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_boundary(points: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_checkerboard(block: np.ndarray) -> bool:
    if block.shape != (3, 3):
        return False
    colors = np.unique(block)
    if len(colors) != 2:
        return False
    a = int(block[0, 0])
    b = int(block[0, 1])
    if a == b:
        return False
    for r in range(3):
        for c in range(3):
            want = a if (r + c) % 2 == 0 else b
            if int(block[r, c]) != want:
                return False
    return True


def _split_legend_and_scene(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    seen_non_bg = False
    h = grid.shape[0]
    for r in range(h):
        has_non_bg = bool(np.any(grid[r] != bg))
        if has_non_bg:
            seen_non_bg = True
            continue
        if seen_non_bg and np.any(grid[r + 1 :] != bg):
            scene_start = r + 1
            while scene_start < h and np.all(grid[scene_start] == bg):
                scene_start += 1
            if scene_start < h:
                return grid[:r], scene_start
    raise AssertionError("expected top legend band separated from the lower scene by blank rows")


def _extract_templates(legend: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, w = legend.shape
    mapping: dict[int, np.ndarray] = {}
    for r in range(h - 2):
        for c in range(1, w - 2):
            block = legend[r : r + 3, c : c + 3]
            if not _is_checkerboard(block):
                continue
            left_strip = legend[r : r + 3, c - 1]
            non_bg_rows = [i for i in range(3) if int(left_strip[i]) != bg]
            if len(non_bg_rows) != 1:
                continue
            seed_color = int(left_strip[non_bg_rows[0]])
            if seed_color not in set(map(int, np.unique(block))):
                continue
            if seed_color in mapping:
                assert np.array_equal(mapping[seed_color], block)
            else:
                mapping[seed_color] = block.copy()
    assert mapping, "failed to locate any seed->template legends"
    return mapping


def _score_border_candidate(scene: np.ndarray, bg: int, mapping: dict[int, np.ndarray], color: int) -> tuple[int, int]:
    valid = 0
    invalid = 0
    for comp in _components_mask(scene != color):
        if _touches_boundary(comp, scene.shape):
            continue
        non_bg = [int(scene[r, c]) for r, c in comp if int(scene[r, c]) != bg]
        if len(non_bg) == 1 and non_bg[0] in mapping and len(comp) >= 2:
            valid += 1
        else:
            invalid += 1
    return valid, -invalid


def _infer_border_color(scene: np.ndarray, bg: int, mapping: dict[int, np.ndarray]) -> int:
    colors = [int(v) for v in np.unique(scene) if int(v) != bg]
    border_candidates = [color for color in colors if color not in mapping]
    if len(border_candidates) == 1:
        return border_candidates[0]
    best_color = None
    best_score = (-1, -10**9)
    for color in colors:
        score = _score_border_candidate(scene, bg, mapping, color)
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score[0] > 0, "failed to infer the grey border color"
    return int(best_color)


def _collect_regions(scene: np.ndarray, bg: int, border_color: int, mapping: dict[int, np.ndarray]):
    regions = []
    for comp in _components_mask(scene != border_color):
        if _touches_boundary(comp, scene.shape):
            continue
        seed_points = [(r, c) for r, c in comp if int(scene[r, c]) != bg]
        if len(seed_points) != 1:
            continue
        sr, sc = seed_points[0]
        seed_color = int(scene[sr, sc])
        if seed_color not in mapping:
            continue
        regions.append((comp, seed_color))
    assert regions, "no enclosed seeded regions found"
    return regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend, scene_start = _split_legend_and_scene(g, bg)
    mapping = _extract_templates(legend, bg)

    scene = g[scene_start:]
    border_color = _infer_border_color(scene, bg, mapping)
    regions = _collect_regions(scene, bg, border_color, mapping)

    out = g.copy()
    for comp, seed_color in regions:
        template = mapping[seed_color]
        min_r = min(r for r, _ in comp)
        min_c = min(c for _, c in comp)
        for r, c in comp:
            out[scene_start + r, c] = int(template[(r - min_r) % 3, (c - min_c) % 3])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region by tiling the checkerboard template selected by its interior seed color.

    Input structure:
    - The background is the dominant color.
    - A horizontal legend band sits at the top: each entry has one seed-colored pixel immediately left of a 3×3 two-color checkerboard template.
    - Lower in the grid are irregular regions enclosed by a one-pixel-thick grey border; each enclosed region contains exactly one seed-colored pixel and otherwise background.

    Rule:
    - Keep the legend and the grey borders unchanged.
    - Read the top legend as a mapping from seed color to its 3×3 checkerboard template.
    - For each enclosed lower region, find its seed color, select the matching template, and tile that template across the whole interior from the region bbox's top-left corner; clip the tiling at the region boundary and overwrite the seed pixel too.

    Key insight:
    - The lone interior seed pixel is a lookup key into the legend, and the selected template must be repeated with a bbox-anchored phase rather than copied only near the seed.
    """
    return _solve(grid)


def _make_checkerboard(seed_color: int, fill_color: int, phase: int) -> np.ndarray:
    template = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            parity = (r + c + phase) % 2
            template[r, c] = seed_color if parity == 0 else fill_color
    return template


def _grow_blob(rng: random.Random, h: int, w: int, target_size: int) -> set[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _make_region_object(rng: random.Random):
    for _ in range(REGION_RETRIES):
        inner_h = rng.randint(3, 6)
        inner_w = rng.randint(3, 6)
        area_cap = inner_h * inner_w
        area_min = max(4, area_cap // 2)
        area_max = max(area_min, area_cap - 1)
        blob = _grow_blob(rng, inner_h, inner_w, rng.randint(area_min, area_max))
        blob = {(r + 1, c + 1) for r, c in blob}

        obj_h = inner_h + 2
        obj_w = inner_w + 2
        border = set()
        for r, c in blob:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < obj_h and 0 <= nc < obj_w and (nr, nc) not in blob:
                    border.add((nr, nc))

        border_mask = np.zeros((obj_h, obj_w), dtype=bool)
        for r, c in border:
            border_mask[r, c] = True

        passable_comps = _components_mask(~border_mask)
        enclosed = [comp for comp in passable_comps if not _touches_boundary(comp, border_mask.shape)]
        if len(enclosed) != 1:
            continue

        interior = set(enclosed[0])
        if not blob.issubset(interior):
            continue
        if len(interior) < 4:
            continue

        bbox_h = max(r for r, _ in interior) - min(r for r, _ in interior) + 1
        bbox_w = max(c for _, c in interior) - min(c for _, c in interior) + 1
        if bbox_h < 2 or bbox_w < 2:
            continue

        return {
            "height": obj_h,
            "width": obj_w,
            "border": border,
            "interior": interior,
        }
    raise RuntimeError("failed to generate a valid enclosed region")


def _paint_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    top: int,
    left: int,
    region: dict[str, object],
    border_color: int,
    seed_color: int,
    bg: int,
    rng: random.Random,
):
    border = region["border"]
    interior = region["interior"]
    assert isinstance(border, set)
    assert isinstance(interior, set)

    for r, c in border:
        grid[top + r, left + c] = border_color
    for r, c in interior:
        grid[top + r, left + c] = bg

    sr, sc = rng.choice(tuple(interior))
    grid[top + sr, left + sc] = seed_color

    h = int(region["height"])
    w = int(region["width"])
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c1 = min(grid.shape[1], left + w + 1)
    blocked[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_legends = rng.randint(2, 4)
    n_regions = rng.randint(2, 4)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    border_color = rng.choice(palette)
    palette.remove(border_color)
    rng.shuffle(palette)
    seed_colors = palette[:n_legends]
    fill_colors = palette[n_legends : 2 * n_legends]

    left_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)
    gaps = [rng.randint(1, 3) for _ in range(n_legends - 1)]
    legend_width = left_margin + right_margin + 4 * n_legends + sum(gaps)
    grid_w_min = max(legend_width, 16)
    if grid_w_min > 30:
        return None
    grid_w = rng.randint(grid_w_min, 30)

    separator_rows = rng.randint(1, 2)
    scene_start = 3 + separator_rows
    grid_h = rng.randint(max(scene_start + 8, 14), 30)

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    col = left_margin
    templates: dict[int, np.ndarray] = {}
    for idx, (seed_color, fill_color) in enumerate(zip(seed_colors, fill_colors)):
        template = _make_checkerboard(seed_color, fill_color, rng.randint(0, 1))
        templates[seed_color] = template
        seed_row = rng.randint(0, 2)
        grid[seed_row, col] = seed_color
        grid[0:3, col + 1 : col + 4] = template
        col += 4
        if idx < len(gaps):
            col += gaps[idx]

    used_count = rng.randint(1, min(n_legends, n_regions))
    used_seeds = rng.sample(seed_colors, used_count)
    region_seed_colors = used_seeds[:]
    while len(region_seed_colors) < n_regions:
        region_seed_colors.append(rng.choice(used_seeds))
    rng.shuffle(region_seed_colors)

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[:scene_start, :] = True

    for seed_color in region_seed_colors:
        region = _make_region_object(rng)
        placed = False
        for _ in range(120):
            obj_h = int(region["height"])
            obj_w = int(region["width"])
            max_top = grid_h - obj_h - 1
            max_left = grid_w - obj_w - 1
            if max_top < scene_start or max_left < 1:
                return None
            top = rng.randint(scene_start, max_top)
            left = rng.randint(1, max_left)
            if blocked[top : top + obj_h, left : left + obj_w].any():
                continue
            if blocked[max(0, top - 1) : min(grid_h, top + obj_h + 1), max(0, left - 1) : min(grid_w, left + obj_w + 1)].any():
                continue
            _paint_region(grid, blocked, top, left, region, border_color, seed_color, bg, rng)
            placed = True
            break
        if not placed:
            return None

    if np.all(grid[scene_start:] == bg):
        return None

    legend, parsed_scene_start = _split_legend_and_scene(grid, bg)
    mapping = _extract_templates(legend, bg)
    if set(mapping) != set(seed_colors):
        return None

    scene = grid[parsed_scene_start:]
    parsed_border = _infer_border_color(scene, bg, mapping)
    if parsed_border != border_color:
        return None

    regions = _collect_regions(scene, bg, parsed_border, mapping)
    if len(regions) != n_regions:
        return None

    out = _solve(grid)
    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid puzzle instance")
