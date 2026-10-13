import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return Counter(corners).most_common(1)[0][0]


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _components_mask(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _color_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                    "shape": _normalize_cells(coords),
                }
            )
            comp_id += 1
    return comps


def _component_holes(grid: np.ndarray, comp, bg: int):
    r0, c0, r1, c1 = comp["bbox"]
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes = []
    sub_h, sub_w = sub_bg.shape
    for cells in _components_mask(sub_bg):
        if any(r == 0 or r == sub_h - 1 or c == 0 or c == sub_w - 1 for r, c in cells):
            continue
        holes.append(
            {
                "cells": [(r0 + r, c0 + c) for r, c in cells],
                "shape": _normalize_cells(cells),
                "area": len(cells),
            }
        )
    return holes


def _analyze_instance(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _color_components(g, bg)
    sprites = []
    templates = []
    others = []
    sprite_holes = {}
    for comp in comps:
        area = int(comp["area"])
        if 15 <= area <= 30:
            sprites.append(comp)
        elif 2 <= area <= 6:
            templates.append(comp)
        else:
            others.append(comp)
    for sprite in sprites:
        sprite_holes[sprite["id"]] = _component_holes(g, sprite, bg)
    return {
        "bg": bg,
        "grid": g,
        "sprites": sprites,
        "templates": templates,
        "others": others,
        "sprite_holes": sprite_holes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each sprite hole to an outside template fragment with the same shape,
    fill the hole with the fragment's color, and erase the matched fragment.
    """
    info = _analyze_instance(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = int(info["bg"])

    template_by_shape = {}
    for comp in info["templates"]:
        template_by_shape.setdefault(comp["shape"], []).append(comp)

    out = g.copy()
    used_templates = set()

    for sprite in info["sprites"]:
        holes = info["sprite_holes"][sprite["id"]]
        for hole in holes:
            matches = [comp for comp in template_by_shape.get(hole["shape"], []) if comp["id"] not in used_templates]
            assert len(matches) == 1, "each hole must have exactly one matching template"
            match = matches[0]
            for r, c in hole["cells"]:
                out[r, c] = int(match["color"])
            used_templates.add(match["id"])

    for comp in info["templates"]:
        if comp["id"] in used_templates:
            for r, c in comp["coords"]:
                out[r, c] = bg

    return out


GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
SPRITE_COUNT_OPTIONS = [1, 2, 3]
HOLE_COUNT_OPTIONS = [1, 2, 3]
SPRITE_DIMS = {
    1: [(4, 6), (6, 4), (5, 5), (5, 6), (6, 5)],
    2: [(5, 5), (5, 6), (6, 5), (6, 6), (5, 7), (7, 5)],
    3: [(5, 6), (6, 5), (6, 6), (6, 7), (7, 6)],
}
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PAD = 1
MAX_GEN_ATTEMPTS = 400
MAX_POLY_ATTEMPTS = 200


def _shape_to_patch(shape, color):
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _random_reorient(shape, rng: random.Random):
    patch = _shape_to_patch(shape, 1)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, rng.randrange(4))
    coords = [tuple(map(int, x)) for x in np.argwhere(patch == 1)]
    return _normalize_cells(coords)


def _random_polyomino(rng: random.Random, area: int, max_h: int, max_w: int):
    for _ in range(MAX_POLY_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            frontier = list(frontier)
            rng.shuffle(frontier)
            added = False
            for cand in frontier:
                test = _normalize_cells(list(cells) + [cand])
                sh, sw = _shape_dims(test)
                if sh <= max_h and sw <= max_w:
                    cells = set(test)
                    added = True
                    break
            if not added:
                break
        if len(cells) != area:
            continue
        shape = _random_reorient(tuple(sorted(cells)), rng)
        sh, sw = _shape_dims(shape)
        if sh <= max_h and sw <= max_w:
            return shape
    return None


def _can_place_hole(cells, occupied):
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _build_sprite_patch(rng: random.Random, hole_count: int, used_shapes):
    for _ in range(MAX_GEN_ATTEMPTS):
        h, w = rng.choice(SPRITE_DIMS[hole_count])
        inner_h, inner_w = h - 2, w - 2
        local_used = set()
        occupied = set()
        placed_holes = []
        for _hole_index in range(hole_count):
            placed = False
            for _ in range(MAX_GEN_ATTEMPTS):
                area = rng.randint(2, 6)
                shape = _random_polyomino(rng, area, inner_h, inner_w)
                if shape is None or shape in used_shapes or shape in local_used:
                    continue
                sh, sw = _shape_dims(shape)
                positions = [(rr, cc) for rr in range(inner_h - sh + 1) for cc in range(inner_w - sw + 1)]
                rng.shuffle(positions)
                for rr, cc in positions:
                    cells = {(1 + rr + r, 1 + cc + c) for r, c in shape}
                    if _can_place_hole(cells, occupied):
                        occupied |= cells
                        local_used.add(shape)
                        placed_holes.append(cells)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                break
        if len(placed_holes) != hole_count:
            continue

        corner_cuts = set()
        if rng.random() < 0.5:
            corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            rng.shuffle(corners)
            cut_count = 1 if rng.random() < 0.75 else 2
            for cell in corners[:cut_count]:
                corner_cuts.add(cell)

        patch = np.ones((h, w), dtype=int)
        for r, c in corner_cuts:
            patch[r, c] = 0
        for cells in placed_holes:
            for r, c in cells:
                patch[r, c] = 0

        comps = _color_components(patch, 0)
        if len(comps) != 1:
            continue
        comp = comps[0]
        if not (15 <= comp["area"] <= 30):
            continue
        holes = _component_holes(patch, comp, 0)
        if len(holes) != hole_count:
            continue
        hole_shapes = [hole["shape"] for hole in holes]
        if len(set(hole_shapes)) != hole_count:
            continue
        if any(not (2 <= hole["area"] <= 6) for hole in holes):
            continue
        return patch, holes
    return None, None


def _reserve_region(blocked, r0, c0, h, w):
    rr0 = max(0, r0 - PAD)
    cc0 = max(0, c0 - PAD)
    rr1 = min(blocked.shape[0], r0 + h + PAD)
    cc1 = min(blocked.shape[1], c0 + w + PAD)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patch(grid, blocked, patch, rng: random.Random):
    h, w = patch.shape
    gh, gw = grid.shape
    if gh - h < 2 or gw - w < 2:
        return False
    positions = [(r, c) for r in range(1, gh - h) for c in range(1, gw - w)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        rr0 = max(0, r0 - PAD)
        cc0 = max(0, c0 - PAD)
        rr1 = min(gh, r0 + h + PAD)
        cc1 = min(gw, c0 + w + PAD)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        sub = patch != 0
        grid[r0 : r0 + h, c0 : c0 + w][sub] = patch[sub]
        _reserve_region(blocked, r0, c0, h, w)
        return True
    return False


def _valid_generated_instance(grid: np.ndarray) -> bool:
    info = _analyze_instance(grid)
    if int(info["bg"]) != 0:
        return False
    if info["others"]:
        return False
    if not (1 <= len(info["sprites"]) <= 3):
        return False
    hole_shapes = []
    for sprite in info["sprites"]:
        holes = info["sprite_holes"][sprite["id"]]
        if not (1 <= len(holes) <= 3):
            return False
        for hole in holes:
            if not (2 <= hole["area"] <= 6):
                return False
            hole_shapes.append(hole["shape"])
    if len(set(hole_shapes)) != len(hole_shapes):
        return False
    template_counts = Counter(comp["shape"] for comp in info["templates"])
    if any(template_counts[shape] != 1 for shape in hole_shapes):
        return False
    return True


def _gen_attempt(rng: random.Random):
    sprite_count = rng.choice(SPRITE_COUNT_OPTIONS)
    used_shapes = set()
    sprite_patches = []
    matched_templates = []

    sprite_colors = rng.sample(COLORS, k=sprite_count)
    for sprite_color in sprite_colors:
        hole_count = rng.choice(HOLE_COUNT_OPTIONS)
        patch, holes = _build_sprite_patch(rng, hole_count, used_shapes)
        if patch is None:
            return None
        colored_patch = np.where(patch == 1, sprite_color, 0).astype(int)
        sprite_patches.append(colored_patch)
        for hole in holes:
            used_shapes.add(hole["shape"])
            template_color = rng.choice(COLORS)
            matched_templates.append(_shape_to_patch(hole["shape"], template_color))

    extra_templates = []
    if rng.random() < 0.5:
        distractor_count = rng.randint(1, 2)
        for _ in range(distractor_count):
            shape = None
            for _ in range(MAX_GEN_ATTEMPTS):
                area = rng.randint(2, 6)
                candidate = _random_polyomino(rng, area, 4, 4)
                if candidate is not None and candidate not in used_shapes:
                    shape = candidate
                    break
            if shape is None:
                return None
            used_shapes.add(shape)
            extra_templates.append(_shape_to_patch(shape, rng.choice(COLORS)))

    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for patch in sprite_patches:
        if not _place_patch(grid, blocked, patch, rng):
            return None

    templates = matched_templates + extra_templates
    rng.shuffle(templates)
    for patch in templates:
        if not _place_patch(grid, blocked, patch, rng):
            return None

    if not _valid_generated_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
