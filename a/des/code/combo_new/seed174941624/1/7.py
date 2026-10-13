from collections import deque

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERNS = {
    frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}): ("right", (1, 2)),
    frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}): ("left", (1, 0)),
    frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}): ("up", (0, 1)),
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}): ("down", (2, 1)),
}
ARROW_OFFSETS = {
    "right": [(0, 0), (0, -1), (0, -2), (-1, 0), (1, 0)],
    "left": [(0, 0), (0, 1), (0, 2), (-1, 0), (1, 0)],
    "up": [(0, 0), (1, 0), (2, 0), (0, -1), (0, 1)],
    "down": [(0, 0), (-1, 0), (-2, 0), (0, -1), (0, 1)],
}

GENERATE_RETRIES = 400
PAIR_TRIES = 160
SHAPE_TRIES = 80


def _bg_color(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_arrows(grid):
    arrows = []
    for comp in _components_from_mask(grid == RED):
        if len(comp) != 5:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        norm = frozenset((r - r0, c - c0) for r, c in comp)
        spec = ARROW_PATTERNS.get(norm)
        if spec is None:
            continue
        name, tip_local = spec
        arrows.append(
            {
                "name": name,
                "direction": DIRS[name],
                "tip": (r0 + tip_local[0], c0 + tip_local[1]),
                "cells": sorted(comp),
            }
        )
    arrows.sort(key=lambda arrow: (arrow["tip"][0], arrow["tip"][1], arrow["name"]))
    return arrows


def _extract_objects(grid, bg):
    mask = (grid != bg) & (grid != RED)
    comps = _components_from_mask(mask)
    comps.sort(key=lambda comp: _bbox(comp))
    objects = []
    cell_to_idx = {}
    for idx, comp in enumerate(comps):
        pixels = [(r, c, int(grid[r, c])) for r, c in sorted(comp)]
        bbox = _bbox(comp)
        colors = {v for _, _, v in pixels}
        obj = {"pixels": pixels, "bbox": bbox, "colors": colors}
        objects.append(obj)
        for r, c, _ in pixels:
            cell_to_idx[(r, c)] = idx
    return objects, cell_to_idx


def _scan_ray_target(grid, bg, cell_to_idx, tip, direction):
    h, w = grid.shape
    r, c = tip
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        v = int(grid[r, c])
        if v != bg and v != RED:
            return cell_to_idx[(r, c)]
        r += dr
        c += dc
    return None


def _ray_object_ids(cell_to_idx, shape, tip, direction):
    h, w = shape
    r, c = tip
    dr, dc = direction
    ids = set()
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        idx = cell_to_idx.get((r, c))
        if idx is not None:
            ids.add(idx)
        r += dr
        c += dc
    return ids


def _shift_for_arrow(obj_bbox, tip, direction):
    r0, c0, r1, c1 = obj_bbox
    tr, tc = tip
    if direction == (0, 1):
        return 0, tc + 1 - c0
    if direction == (0, -1):
        return 0, tc - 1 - c1
    if direction == (1, 0):
        return tr + 1 - r0, 0
    return tr - 1 - r1, 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each multi-color object back along a red arrow until it hugs that arrow's tip.

    Input structure:
    - The background is black.
    - Several red 5-pixel arrow markers appear; each arrow is a 3-pixel shaft with two side pixels at the shaft end, pointing up, down, left, or right.
    - Separate 4-connected multi-color foreground objects use colors other than black and red.
    - Along each arrow's forward ray, the first non-black, non-red pixel belongs to that arrow's unique target object.

    Rule:
    - Detect every red arrow, including its pointing direction and the shaft-end tip pixel.
    - From each tip, scan step by step in the arrow direction until the first non-black, non-red pixel is reached; its 4-connected component is the target object.
    - Remove that object from its original location, then translate it straight backward along the arrow axis so the side of its bounding box that faces the arrow becomes immediately adjacent to the tip.
    - Keep all red arrow markers unchanged.

    Key insight:
    - Each arrow supplies both selection and placement: the forward ray chooses which object moves, while the tip itself is the anchor that determines the object's final translated position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    arrows = _find_arrows(g)
    if not arrows:
        return g.copy()

    objects, cell_to_idx = _extract_objects(g, bg)
    moves = {}
    for arrow in arrows:
        idx = _scan_ray_target(g, bg, cell_to_idx, arrow["tip"], arrow["direction"])
        if idx is None or idx in moves:
            continue
        moves[idx] = _shift_for_arrow(objects[idx]["bbox"], arrow["tip"], arrow["direction"])

    out = g.copy()
    for idx in moves:
        for r, c, _ in objects[idx]["pixels"]:
            out[r, c] = bg

    h, w = g.shape
    for idx, (dr, dc) in moves.items():
        for r, c, color in objects[idx]["pixels"]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(rng):
    for _ in range(SHAPE_TRIES):
        target_area = int(rng.integers(4, 9))
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = tuple(frontier)[int(rng.integers(len(frontier)))]
            cells.add(choice)
        norm = _normalize(cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if len(norm) != target_area:
            continue
        if not (2 <= h <= 5 and 2 <= w <= 5):
            continue
        if len({r for r, _ in norm}) < 2 or len({c for _, c in norm}) < 2:
            continue
        return norm, h, w
    return None


def _colorize_shape(cells, rng):
    color_count = 2 if rng.random() < 0.7 else 3
    colors = [int(v) for v in rng.choice(PALETTE, size=color_count, replace=False)]
    phase = int(rng.integers(0, 4))
    colored = {}
    for i, (r, c) in enumerate(sorted(cells)):
        colored[(r, c)] = colors[(r + 2 * c + phase + i) % color_count]
    if len(set(colored.values())) < 2:
        first = next(iter(colored))
        colored[first] = colors[1]
    return colored


def _edge_cells(cells, direction_name):
    if direction_name == "right":
        edge = min(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    if direction_name == "left":
        edge = max(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    if direction_name == "down":
        edge = min(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    edge = max(r for r, _ in cells)
    return [(r, c) for r, c in cells if r == edge]


def _expand(cells, h, w, radius=1):
    expanded = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _ray_cells(shape, tip, direction):
    h, w = shape
    dr, dc = direction
    r, c = tip
    cells = set()
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.add((r, c))
        r += dr
        c += dc
    return cells


def _arrow_cells(tip, direction_name):
    cells = set()
    for rr, cc in ARROW_OFFSETS[direction_name]:
        cells.add((tip[0] + rr, tip[1] + cc))
    return cells


def _place_pair(rng, h, w, direction_name, blocked):
    direction = DIRS[direction_name]
    for _ in range(PAIR_TRIES):
        shape_data = _random_shape(rng)
        if shape_data is None:
            continue
        shape, shape_h, shape_w = shape_data
        anchor = _edge_cells(shape, direction_name)[int(rng.integers(len(_edge_cells(shape, direction_name))))]
        bg_gap = int(rng.integers(1, 5))

        if direction_name == "right":
            row_low = max(0, 1 - anchor[0])
            row_high = min(h - shape_h, h - 2 - anchor[0])
            col_low = 3
            col_high = w - shape_w - bg_gap
        elif direction_name == "left":
            row_low = max(0, 1 - anchor[0])
            row_high = min(h - shape_h, h - 2 - anchor[0])
            col_low = bg_gap
            col_high = w - shape_w - 3
        elif direction_name == "down":
            row_low = 3
            row_high = h - shape_h - bg_gap
            col_low = max(0, 1 - anchor[1])
            col_high = min(w - shape_w, w - 2 - anchor[1])
        else:
            row_low = bg_gap
            row_high = h - shape_h - 3
            col_low = max(0, 1 - anchor[1])
            col_high = min(w - shape_w, w - 2 - anchor[1])

        if row_low > row_high or col_low > col_high:
            continue

        fr0 = int(rng.integers(row_low, row_high + 1))
        fc0 = int(rng.integers(col_low, col_high + 1))

        if direction_name == "right":
            tip = (fr0 + anchor[0], fc0 - 1)
            r0 = fr0
            c0 = fc0 + bg_gap
        elif direction_name == "left":
            tip = (fr0 + anchor[0], fc0 + shape_w)
            r0 = fr0
            c0 = fc0 - bg_gap
        elif direction_name == "down":
            tip = (fr0 - 1, fc0 + anchor[1])
            r0 = fr0 + bg_gap
            c0 = fc0
        else:
            tip = (fr0 + shape_h, fc0 + anchor[1])
            r0 = fr0 - bg_gap
            c0 = fc0

        final_cells = {(fr0 + r, fc0 + c) for r, c in shape}
        object_cells = {(r0 + r, c0 + c) for r, c in shape}
        arrow_cells = _arrow_cells(tip, direction_name)
        ray_cells = _ray_cells((h, w), tip, direction)

        if object_cells & blocked:
            continue
        if arrow_cells & blocked:
            continue
        if final_cells & blocked:
            continue
        if ray_cells & blocked:
            continue
        if final_cells & arrow_cells:
            continue

        colored = _colorize_shape(shape, rng)
        pixels = {(r0 + r, c0 + c): color for (r, c), color in colored.items()}
        block_update = set(ray_cells)
        block_update |= final_cells
        block_update |= _expand(object_cells | arrow_cells, h, w, radius=1)
        return {
            "pixels": pixels,
            "arrow_cells": arrow_cells,
            "blocked": block_update,
        }
    return None


def _validate_generated(grid):
    bg = _bg_color(grid)
    arrows = _find_arrows(grid)
    if not (2 <= len(arrows) <= 4):
        return False
    objects, cell_to_idx = _extract_objects(grid, bg)
    if len(objects) != len(arrows):
        return False
    for obj in objects:
        if len(obj["colors"]) < 2:
            return False

    targeted = []
    for arrow in arrows:
        ids_on_ray = _ray_object_ids(cell_to_idx, grid.shape, arrow["tip"], arrow["direction"])
        if len(ids_on_ray) != 1:
            return False
        idx = _scan_ray_target(grid, bg, cell_to_idx, arrow["tip"], arrow["direction"])
        if idx is None or idx not in ids_on_ray:
            return False
        targeted.append(idx)

    if len(set(targeted)) != len(arrows):
        return False
    if set(targeted) != set(range(len(objects))):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out == RED, grid == RED):
        return False
    return True


def _gen_attempt(rng):
    pair_count = int(rng.integers(2, 5))
    min_size = 14 + 2 * pair_count
    h = int(rng.integers(min_size, 29))
    w = int(rng.integers(min_size, 29))
    grid = np.full((h, w), BG, dtype=int)
    blocked = set()

    dir_names = list(DIRS)
    chosen = [dir_names[i] for i in rng.choice(len(dir_names), size=pair_count, replace=False)]
    rng.shuffle(chosen)

    for direction_name in chosen:
        pair = _place_pair(rng, h, w, direction_name, blocked)
        if pair is None:
            return None
        for (r, c), color in pair["pixels"].items():
            grid[r, c] = color
        for r, c in pair["arrow_cells"]:
            grid[r, c] = RED
        blocked |= pair["blocked"]

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-object input")
