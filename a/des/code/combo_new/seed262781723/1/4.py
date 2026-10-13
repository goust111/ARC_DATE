from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver helpers ---------------


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _extract_components(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or labels[r, c] != -1:
                continue

            comp_id = len(components)
            labels[r, c] = comp_id
            q = deque([(r, c)])
            coords = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if labels[nr, nc] != -1 or int(g[nr, nc]) != color:
                        continue
                    labels[nr, nc] = comp_id
                    q.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components, labels


def _component_graph(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    components, labels = _extract_components(g)
    adjacency = [set() for _ in components]
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a == -1:
                continue
            if r + 1 < h:
                b = int(labels[r + 1, c])
                if b != -1 and b != a:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            if c + 1 < w:
                b = int(labels[r, c + 1])
                if b != -1 and b != a:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return components, labels, adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every block that touches the unique highest-degree block.

    Input structure:
    - The background is black.
    - Each non-black region is a monochromatic 4-connected block.
    - Different colored blocks may touch along shared edges, creating a block-adjacency graph.

    Rule:
    - Treat each non-black 4-connected block as one graph node.
    - Connect two nodes when their pixels contain at least one 4-adjacent cross-color pair.
    - Find the unique node with the largest degree.
    - Repaint every block directly adjacent to that node using the winner's color, while leaving the winner, all non-neighbors, and the black background unchanged.

    Key insight:
    The transformation depends on counting graph neighbors of color blocks rather than comparing their sizes or positions.
    """
    g = np.asarray(grid, dtype=int)
    components, _labels, adjacency = _component_graph(g)
    assert components, "expected at least one non-background component"

    degrees = [len(neighbors) for neighbors in adjacency]
    max_degree = max(degrees)
    winners = [idx for idx, degree in enumerate(degrees) if degree == max_degree]
    assert len(winners) == 1, "expected a unique maximum-degree component"

    winner = winners[0]
    winner_color = int(components[winner]["color"])
    out = g.copy()
    for neighbor in adjacency[winner]:
        for r, c in components[neighbor]["coords"]:
            out[r, c] = winner_color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
SCENE_MODES = ("star", "leafy", "chain", "mixed")
META_SHAPES = {
    2: ((9, 13), (9, 13)),
    3: ((7, 9), (7, 9)),
}
HUB_H_RANGE = (3, 4)
HUB_W_RANGE = (3, 4)
HUB_DEGREE_RANGE = (3, 6)
NEIGHBOR_AREA_RANGE = (1, 4)
LEAF_AREA_RANGE = (1, 3)
ISOLATED_AREA_RANGE = (1, 3)
DIRECTED_GROW_TRIES = 16
ISOLATED_GROW_TRIES = 16


def _adjacent_existing_ids(id_grid: np.ndarray, cells: set[tuple[int, int]]) -> set[int]:
    h, w = id_grid.shape
    ids = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in cells:
                continue
            comp_id = int(id_grid[nr, nc])
            if comp_id:
                ids.add(comp_id)
    return ids


def _paint_component(id_grid: np.ndarray, cells: set[tuple[int, int]], comp_id: int) -> None:
    for r, c in cells:
        id_grid[r, c] = int(comp_id)


def _direction_box(
    shape: tuple[int, int], anchor: tuple[int, int], normal: tuple[int, int], depth: int, lateral: int
) -> set[tuple[int, int]]:
    h, w = shape
    r, c = anchor
    if normal == (-1, 0):
        r0, r1 = max(0, r - depth), r
        c0, c1 = max(0, c - lateral), min(w - 1, c + lateral)
    elif normal == (1, 0):
        r0, r1 = r, min(h - 1, r + depth)
        c0, c1 = max(0, c - lateral), min(w - 1, c + lateral)
    elif normal == (0, -1):
        r0, r1 = max(0, r - lateral), min(h - 1, r + lateral)
        c0, c1 = max(0, c - depth), c
    else:
        r0, r1 = max(0, r - lateral), min(h - 1, r + lateral)
        c0, c1 = c, min(w - 1, c + depth)

    return {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}


def _center_box(shape: tuple[int, int], seed: tuple[int, int], radius_r: int, radius_c: int) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    r0, r1 = max(0, r - radius_r), min(h - 1, r + radius_r)
    c0, c1 = max(0, c - radius_c), min(w - 1, c + radius_c)
    return {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}


def _grow_component(
    id_grid: np.ndarray,
    start: tuple[int, int],
    allowed: set[tuple[int, int]],
    allowed_neighbor_ids: set[int],
    area_range: tuple[int, int],
) -> set[tuple[int, int]] | None:
    min_area, max_area = area_range
    max_area = min(max_area, len(allowed))
    if max_area < min_area or start not in allowed or int(id_grid[start]) != 0:
        return None

    target = random.randint(min_area, max_area)
    cells = {start}
    h, w = id_grid.shape

    while len(cells) < target:
        frontier = []
        for r, c in tuple(cells):
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in cells or (nr, nc) not in allowed or int(id_grid[nr, nc]) != 0:
                    continue
                ok = True
                for rr, cc in _neighbors4(nr, nc):
                    if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in cells:
                        continue
                    owner = int(id_grid[rr, cc])
                    if owner and owner not in allowed_neighbor_ids:
                        ok = False
                        break
                if ok:
                    frontier.append((nr, nc))

        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    if len(cells) < min_area:
        return None
    if _adjacent_existing_ids(id_grid, cells) != allowed_neighbor_ids:
        return None
    return cells


def _place_hub(id_grid: np.ndarray):
    h, w = id_grid.shape
    hub_h = random.randint(HUB_H_RANGE[0], min(HUB_H_RANGE[1], h - 4))
    hub_w = random.randint(HUB_W_RANGE[0], min(HUB_W_RANGE[1], w - 4))
    if h - hub_h < 4 or w - hub_w < 4:
        return None
    r0 = random.randint(2, h - hub_h - 2)
    c0 = random.randint(2, w - hub_w - 2)
    cells = {(r, c) for r in range(r0, r0 + hub_h) for c in range(c0, c0 + hub_w)}
    _paint_component(id_grid, cells, 1)
    return cells


def _attachment_candidates(id_grid: np.ndarray, parent_cells: set[tuple[int, int]], parent_id: int):
    h, w = id_grid.shape
    seen = {}
    for r, c in parent_cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(id_grid[nr, nc]) != 0:
                continue
            anchor = (nr, nc)
            if _adjacent_existing_ids(id_grid, {anchor}) == {parent_id}:
                seen[anchor] = (dr, dc)
    items = list(seen.items())
    random.shuffle(items)
    return [(anchor, normal) for anchor, normal in items]


def _pick_hub_slots(candidates, degree: int):
    for _ in range(80):
        random.shuffle(candidates)
        chosen = []
        for anchor, normal in candidates:
            if all(abs(anchor[0] - other[0]) + abs(anchor[1] - other[1]) != 1 for other, _ in chosen):
                chosen.append((anchor, normal))
                if len(chosen) == degree:
                    return chosen
    return None


def _directed_component_box(
    shape: tuple[int, int], anchor: tuple[int, int], normal: tuple[int, int]
) -> set[tuple[int, int]]:
    style = random.choice(("slim", "wide", "blob"))
    if style == "slim":
        depth = random.randint(1, 3)
        lateral = random.randint(0, 1)
    elif style == "wide":
        depth = random.randint(1, 2)
        lateral = random.randint(1, 2)
    else:
        depth = random.randint(2, 3)
        lateral = random.randint(1, 2)
    return _direction_box(shape, anchor, normal, depth, lateral)


def _place_attached_component(
    id_grid: np.ndarray,
    components: dict[int, set[tuple[int, int]]],
    parent_id: int,
    area_range: tuple[int, int],
) -> int | None:
    candidates = _attachment_candidates(id_grid, components[parent_id], parent_id)
    for anchor, normal in candidates:
        for _ in range(DIRECTED_GROW_TRIES):
            allowed = _directed_component_box(id_grid.shape, anchor, normal)
            cells = _grow_component(id_grid, anchor, allowed, {parent_id}, area_range)
            if cells is None:
                continue
            new_id = len(components) + 1
            _paint_component(id_grid, cells, new_id)
            components[new_id] = cells
            return new_id
    return None


def _isolated_seeds(id_grid: np.ndarray):
    h, w = id_grid.shape
    seeds = []
    for r in range(h):
        for c in range(w):
            if int(id_grid[r, c]) == 0 and not _adjacent_existing_ids(id_grid, {(r, c)}):
                seeds.append((r, c))
    random.shuffle(seeds)
    return seeds


def _place_isolated_component(
    id_grid: np.ndarray,
    components: dict[int, set[tuple[int, int]]],
    area_range: tuple[int, int],
) -> int | None:
    for seed in _isolated_seeds(id_grid):
        for _ in range(ISOLATED_GROW_TRIES):
            radius_r = random.randint(1, 2)
            radius_c = random.randint(1, 2)
            allowed = _center_box(id_grid.shape, seed, radius_r, radius_c)
            cells = _grow_component(id_grid, seed, allowed, set(), area_range)
            if cells is None:
                continue
            new_id = len(components) + 1
            _paint_component(id_grid, cells, new_id)
            components[new_id] = cells
            return new_id
    return None


def _meta_to_color_grid(id_grid: np.ndarray, n_components: int) -> np.ndarray:
    colors = random.sample(range(1, 10), n_components)
    out = np.zeros_like(id_grid, dtype=int)
    for comp_id, color in enumerate(colors, start=1):
        out[id_grid == comp_id] = int(color)
    return out


def _embed_scaled(meta_grid: np.ndarray, scale: int) -> np.ndarray:
    upscaled = np.kron(meta_grid, np.ones((scale, scale), dtype=int))
    h, w = upscaled.shape
    extra_h = random.randint(0, min(30 - h, 4))
    extra_w = random.randint(0, min(30 - w, 4))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    canvas = np.zeros((h + extra_h, w + extra_w), dtype=int)
    canvas[top : top + h, left : left + w] = upscaled
    return canvas


def _background_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(counts[0]) > int(counts[1:].max(initial=0))


def _build_meta_scene() -> tuple[np.ndarray, int] | None:
    scale = random.choice((2, 3))
    hr, wr = META_SHAPES[scale]
    meta_h = random.randint(*hr)
    meta_w = random.randint(*wr)
    id_grid = np.zeros((meta_h, meta_w), dtype=int)

    hub_cells = _place_hub(id_grid)
    if hub_cells is None:
        return None
    components = {1: hub_cells}

    mode = random.choice(SCENE_MODES)
    hub_degree = random.randint(*HUB_DEGREE_RANGE)
    slots = _pick_hub_slots(_attachment_candidates(id_grid, hub_cells, 1), hub_degree)
    if slots is None:
        return None

    neighbor_ids = []
    for anchor, normal in slots:
        for _ in range(DIRECTED_GROW_TRIES):
            allowed = _directed_component_box(id_grid.shape, anchor, normal)
            cells = _grow_component(id_grid, anchor, allowed, {1}, NEIGHBOR_AREA_RANGE)
            if cells is None:
                continue
            new_id = len(components) + 1
            _paint_component(id_grid, cells, new_id)
            components[new_id] = cells
            neighbor_ids.append(new_id)
            break
        else:
            return None

    if mode == "star":
        if random.random() < 0.5:
            if _place_isolated_component(id_grid, components, ISOLATED_AREA_RANGE) is None:
                return None
    elif mode == "leafy":
        leaf_count = random.randint(1, min(2, len(neighbor_ids)))
        for parent_id in random.sample(neighbor_ids, leaf_count):
            if _place_attached_component(id_grid, components, parent_id, LEAF_AREA_RANGE) is None:
                return None
    elif mode == "chain":
        first = _place_attached_component(id_grid, components, random.choice(neighbor_ids), LEAF_AREA_RANGE)
        if first is None:
            return None
        if _place_attached_component(id_grid, components, first, LEAF_AREA_RANGE) is None:
            return None
    else:
        if _place_attached_component(id_grid, components, random.choice(neighbor_ids), LEAF_AREA_RANGE) is None:
            return None
        if _place_isolated_component(id_grid, components, ISOLATED_AREA_RANGE) is None:
            return None

    if len(components) > 9:
        return None
    return id_grid, scale


def _gen_attempt() -> np.ndarray | None:
    built = _build_meta_scene()
    if built is None:
        return None
    id_grid, scale = built
    n_components = int(id_grid.max())
    meta_grid = _meta_to_color_grid(id_grid, n_components)
    grid = _embed_scaled(meta_grid, scale)

    if max(grid.shape) > 30 or not _background_is_dominant(grid):
        return None

    components, _labels, adjacency = _component_graph(grid)
    degrees = [len(neighbors) for neighbors in adjacency]
    if len(components) != n_components or not degrees:
        return None
    max_degree = max(degrees)
    if max_degree < 3 or sum(degree == max_degree for degree in degrees) != 1:
        return None

    unique_colors = {int(v) for v in np.unique(grid) if int(v) != 0}
    if len(unique_colors) != n_components:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
