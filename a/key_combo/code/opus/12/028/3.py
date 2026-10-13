import itertools
import random
from collections import Counter, defaultdict

import numpy as np


MIN_RING_SIDE = 4
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
RING_COUNT_CHOICES = (3, 4)
RING_COLORS = tuple(range(1, 10))
CLIP_SIDES = (None, "top", "bottom", "left", "right")
MAX_GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def ring_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def side_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(side)


def render_ring_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def topo_unique(nodes: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indeg = {node: 0 for node in nodes}
    for src, dsts in edges.items():
        for dst in dsts:
            indeg[dst] += 1

    ready = sorted(node for node in nodes if indeg[node] == 0)
    order: list[int] = []
    while ready:
        if len(ready) != 1:
            return None
        node = ready.pop()
        order.append(node)
        for dst in sorted(edges[node]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort()
    if len(order) != len(nodes):
        return None
    return order


def substantial_colors(grid: np.ndarray, bg: int) -> list[int]:
    counts = Counter(map(int, grid.ravel()))
    return sorted(color for color, count in counts.items() if color != bg and count >= 3)


def infer_candidates(
    grid: np.ndarray, color: int, valid_colors: set[int], bg: int
) -> list[dict]:
    points = np.argwhere(grid == color)
    if points.size == 0:
        return []

    h_grid, w_grid = grid.shape
    min_r, min_c = points.min(axis=0)
    max_r, max_c = points.max(axis=0)

    tops = [int(min_r)]
    bottoms = [int(max_r)]
    lefts = [int(min_c)]
    rights = [int(max_c)]
    if int(min_r) == 0:
        tops.append(-1)
    if int(max_r) == h_grid - 1:
        bottoms.append(h_grid)
    if int(min_c) == 0:
        lefts.append(-1)
    if int(max_c) == w_grid - 1:
        rights.append(w_grid)

    candidates: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    observed = {tuple(map(int, rc)) for rc in points}

    for r0, r1, c0, c1 in itertools.product(tops, bottoms, lefts, rights):
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if bbox in seen:
            continue
        seen.add(bbox)

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < MIN_RING_SIDE or width < MIN_RING_SIDE:
            continue

        border = set(ring_cells(bbox))
        if not observed.issubset(border):
            continue

        border_in_bounds = [
            (r, c)
            for r, c in border
            if 0 <= r < h_grid and 0 <= c < w_grid
        ]
        if not border_in_bounds:
            continue
        if any(int(grid[r, c]) == bg for r, c in border_in_bounds):
            continue
        if any(int(grid[r, c]) not in valid_colors for r, c in border_in_bounds):
            continue

        candidates.append(
            {
                "color": int(color),
                "bbox": bbox,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "border_in_bounds": border_in_bounds,
            }
        )

    return candidates


def solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    colors = substantial_colors(g, bg)
    assert 3 <= len(colors) <= 4, "expected 3-4 substantial rectangle colors"
    valid_colors = set(colors)

    candidate_map = {color: infer_candidates(g, color, valid_colors, bg) for color in colors}
    assert all(candidate_map[color] for color in colors), "every substantial color must match a ring bbox"

    valid_solutions: list[tuple[dict[int, dict], list[int]]] = []
    color_lists = [candidate_map[color] for color in colors]
    for combo in itertools.product(*color_lists):
        chosen = {cand["color"]: cand for cand in combo}
        areas = [cand["area"] for cand in combo]
        if len(set(areas)) != len(areas):
            continue

        edges: dict[int, set[int]] = {color: set() for color in colors}
        for color, cand in chosen.items():
            for r, c in cand["border_in_bounds"]:
                occ = int(g[r, c])
                if occ in valid_colors and occ != color:
                    edges[occ].add(color)

        layer_order = topo_unique(colors, edges)
        if layer_order is None:
            continue
        valid_solutions.append((chosen, layer_order))

    assert len(valid_solutions) == 1, "input must determine a unique reconstruction"
    chosen, layer_order = valid_solutions[0]
    by_size = sorted(chosen.values(), key=lambda item: (-item["area"], -item["height"], -item["width"], item["color"]))
    return {
        "bg": int(bg),
        "colors": colors,
        "rings_by_color": chosen,
        "layer_order": layer_order,
        "size_order": [item["color"] for item in by_size],
        "sizes": [(int(item["height"]), int(item["width"])) for item in by_size],
    }


def build_output(details: dict) -> np.ndarray:
    outer_h, outer_w = details["sizes"][0]
    out = np.zeros((outer_h, outer_w), dtype=int)

    prev_r = 0
    prev_c = 0
    prev_h = outer_h
    prev_w = outer_w
    for idx, (height, width) in enumerate(details["sizes"]):
        color = int(details["layer_order"][idx])
        if idx == 0:
            r = 0
            c = 0
        else:
            r = prev_r + prev_h - height - 1
            c = prev_c + 1
        out[r : r + height, c : c + width] = np.maximum(
            out[r : r + height, c : c + width], render_ring_patch(height, width, color)
        )
        prev_r = r
        prev_c = c
        prev_h = height
        prev_w = width
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden rectangle frames, nest them by size, and recolor the nesting by occlusion order.

    Input structure:
    - A black background contains 3-4 differently colored hollow rectangles drawn as 1-pixel rectangular frames.
    - Frames partially overlap, so lower frames lose some border pixels to higher colors, and one frame may be clipped by a grid edge.
    - Tiny isolated components with fewer than 3 pixels are distractor noise and do not belong to any frame.

    Rule:
    - Ignore the tiny distractors and treat each substantial color as one partially visible rectangular frame.
    - Reconstruct each frame's full bounding box from its visible border pixels; whenever another substantial color occupies one of its expected in-bounds border cells, that occupying color is above it.
    - Topologically sort these occlusion relations from top to bottom, independently sort the reconstructed frame sizes by descending area, then draw a nested chain of hollow rectangles from largest to smallest with lower-left interior alignment.
    - The k-th largest nesting position uses the color of the k-th layer in the occlusion order, and the innermost interior remains black.

    Key insight:
    - The output geometry comes from frame sizes, but the output colors come from the separate occlusion order, so both reconstructions are required and neither order alone is sufficient.
    """
    details = solve_details(np.asarray(grid, dtype=int))
    return build_output(details)


def sample_sizes(rng: random.Random, count: int) -> list[tuple[int, int]]:
    for _ in range(300):
        min_outer = 4 + 2 * (count - 1)
        outer_h = rng.randint(min_outer, 13)
        outer_w = rng.randint(min_outer, 13)
        sizes = [(outer_h, outer_w)]
        used_areas = {outer_h * outer_w}
        cur_h, cur_w = outer_h, outer_w
        ok = True

        for idx in range(1, count):
            remaining_children = count - idx - 1
            min_h = 4 + 2 * remaining_children
            min_w = 4 + 2 * remaining_children
            choices: list[tuple[int, int]] = []
            for next_h in range(min_h, cur_h - 1):
                for next_w in range(min_w, cur_w - 1):
                    if next_h <= cur_h - 2 and next_w <= cur_w - 2:
                        area = next_h * next_w
                        if area not in used_areas:
                            choices.append((next_h, next_w))
            if not choices:
                ok = False
                break
            next_h, next_w = rng.choice(choices)
            sizes.append((next_h, next_w))
            used_areas.add(next_h * next_w)
            cur_h, cur_w = next_h, next_w

        if ok:
            return sizes

    raise RuntimeError("failed to sample distinct nested sizes")


def random_bbox(
    rng: random.Random,
    grid_shape: tuple[int, int],
    height: int,
    width: int,
    clip_side: str | None,
) -> tuple[int, int, int, int]:
    h_grid, w_grid = grid_shape
    if clip_side == "top":
        r0 = -1
    elif clip_side == "bottom":
        r0 = h_grid - height + 1
    else:
        r0 = rng.randint(0, h_grid - height)

    if clip_side == "left":
        c0 = -1
    elif clip_side == "right":
        c0 = w_grid - width + 1
    else:
        c0 = rng.randint(0, w_grid - width)
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def side_visibility_ok(
    bbox: tuple[int, int, int, int],
    visible: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> bool:
    h_grid, w_grid = grid_shape
    r0, c0, r1, c1 = bbox
    clipped = {
        "top": r0 < 0,
        "bottom": r1 >= h_grid,
        "left": c0 < 0,
        "right": c1 >= w_grid,
    }
    for side in ("top", "bottom", "left", "right"):
        if clipped[side]:
            continue
        if not any(cell in visible for cell in side_cells(bbox, side)):
            return False
    return True


def place_scene(
    rng: random.Random,
    grid_shape: tuple[int, int],
    objects: list[dict],
    layer_order: list[int],
    clip_target: int | None,
    clip_side: str | None,
) -> list[dict] | None:
    top_scene: dict[tuple[int, int], int] = {}
    placed: list[dict] = []
    color_to_object = {obj["color"]: obj for obj in objects}

    for depth, color in enumerate(layer_order):
        obj = color_to_object[color]
        own_clip_side = clip_side if clip_target == color else None
        for _ in range(400):
            bbox = random_bbox(rng, grid_shape, obj["height"], obj["width"], own_clip_side)
            in_bounds_cells = {
                cell
                for cell in ring_cells(bbox)
                if in_bounds(grid_shape, cell[0], cell[1])
            }
            if len(in_bounds_cells) < 6:
                continue

            overlap_colors = {top_scene[cell] for cell in in_bounds_cells if cell in top_scene}
            visible = {cell for cell in in_bounds_cells if cell not in top_scene}
            if depth > 0 and layer_order[depth - 1] not in overlap_colors:
                continue
            if len(visible) < max(5, (2 * (obj["height"] + obj["width"]) - 4) // 4):
                continue
            if not side_visibility_ok(bbox, visible, grid_shape):
                continue

            placed_obj = {
                **obj,
                "bbox": bbox,
                "visible_preview": visible,
                "in_bounds_cells": in_bounds_cells,
            }
            placed.append(placed_obj)
            for cell in in_bounds_cells:
                top_scene.setdefault(cell, color)
            break
        else:
            return None

    return placed


def render_scene(grid_shape: tuple[int, int], placed: list[dict], noise: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for obj in reversed(placed):
        color = int(obj["color"])
        for r, c in ring_cells(obj["bbox"]):
            if in_bounds(grid_shape, r, c):
                grid[r, c] = color
    for color, cells in noise:
        for r, c in cells:
            grid[r, c] = color
    return grid


def add_noise(
    rng: random.Random,
    grid: np.ndarray,
    forbidden_colors: set[int],
) -> list[tuple[int, list[tuple[int, int]]]]:
    available_colors = [color for color in RING_COLORS if color not in forbidden_colors]
    rng.shuffle(available_colors)
    noise: list[tuple[int, list[tuple[int, int]]]] = []
    component_count = rng.choices([0, 1, 2], weights=[2, 4, 3])[0]
    h_grid, w_grid = grid.shape

    def clear_neighbors(r: int, c: int) -> bool:
        for rr in range(max(0, r - 1), min(h_grid, r + 2)):
            for cc in range(max(0, c - 1), min(w_grid, c + 2)):
                if grid[rr, cc] != 0:
                    return False
        return True

    for idx in range(component_count):
        if idx >= len(available_colors):
            break
        color = available_colors[idx]
        size = rng.choice([1, 1, 2])
        positions = [(r, c) for r in range(h_grid) for c in range(w_grid) if grid[r, c] == 0 and clear_neighbors(r, c)]
        if not positions:
            continue
        if size == 1:
            cell = rng.choice(positions)
            grid[cell] = color
            noise.append((color, [cell]))
            continue

        rng.shuffle(positions)
        placed = False
        for r, c in positions:
            for dr, dc in ((1, 0), (0, 1)):
                rr = r + dr
                cc = c + dc
                if not in_bounds((h_grid, w_grid), rr, cc):
                    continue
                if grid[rr, cc] != 0:
                    continue
                if not clear_neighbors(rr, cc):
                    continue
                grid[r, c] = color
                grid[rr, cc] = color
                noise.append((color, [(r, c), (rr, cc)]))
                placed = True
                break
            if placed:
                break
    return noise


def semantic_checks(grid: np.ndarray) -> bool:
    details = solve_details(grid)
    if details["layer_order"] == details["size_order"]:
        return False

    sizes = details["sizes"]
    for i in range(1, len(sizes)):
        if sizes[i][0] > sizes[i - 1][0] - 2 or sizes[i][1] > sizes[i - 1][1] - 2:
            return False

    clipped = 0
    for color, ring in details["rings_by_color"].items():
        r0, c0, r1, c1 = ring["bbox"]
        if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            clipped += 1
        visible = set(map(tuple, np.argwhere(grid == color)))
        if len(visible) < 4:
            return False
    return clipped <= 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        count = rng.choice(RING_COUNT_CHOICES)
        sizes = sample_sizes(rng, count)
        colors = rng.sample(RING_COLORS, count)
        objects = [
            {"color": int(color), "height": int(height), "width": int(width)}
            for (height, width), color in zip(sizes, colors)
        ]

        size_order = [obj["color"] for obj in sorted(objects, key=lambda item: (-item["height"] * item["width"], -item["height"], -item["width"]))]
        layer_order = size_order[:]
        while layer_order == size_order or layer_order[0] == size_order[0]:
            layer_order = rng.sample(size_order, len(size_order))

        grid_shape = (rng.randint(*GRID_H_RANGE), rng.randint(*GRID_W_RANGE))
        clip_target = None
        clip_side = None
        if rng.random() < 0.45:
            clip_target = rng.choice(layer_order)
            clip_side = rng.choice(CLIP_SIDES[1:])

        placed = place_scene(rng, grid_shape, objects, layer_order, clip_target, clip_side)
        if placed is None:
            continue

        base_grid = render_scene(grid_shape, placed, [])
        noise = add_noise(rng, base_grid, set(colors))
        grid = base_grid

        try:
            details = solve_details(grid)
        except AssertionError:
            continue
        if details["layer_order"] != layer_order:
            continue
        if details["size_order"] != size_order:
            continue
        if not semantic_checks(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
