from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COARSE_LIMIT = 3
MAX_GEN_ATTEMPTS = 400

Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_signature(coords: list[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "signature": _component_signature(coords),
                }
            )
    return comps


def _extract_scene(grid: np.ndarray) -> tuple[Coord, list[dict], list[set[int]], int]:
    comps = _find_components(grid)
    assert comps, "expected at least one foreground component"

    cell_to_comp: dict[Coord, int] = {}
    for comp_id, comp in enumerate(comps):
        for cell in comp["coords"]:
            cell_to_comp[cell] = comp_id

    marker_comp = None
    marker_cell = None
    start_raw = None
    for comp_id, comp in enumerate(comps):
        if comp["color"] != MAROON or len(comp["coords"]) != 1:
            continue
        cell = comp["coords"][0]
        adjacent = {
            cell_to_comp[nbr]
            for nbr in _neighbors4(*cell)
            if nbr in cell_to_comp and cell_to_comp[nbr] != comp_id
        }
        if len(adjacent) == 1:
            assert marker_comp is None, "expected a unique maroon marker"
            marker_comp = comp_id
            marker_cell = cell
            start_raw = next(iter(adjacent))

    assert marker_comp is not None and marker_cell is not None and start_raw is not None

    target_signature = comps[start_raw]["signature"]
    raw_object_ids = [idx for idx, comp in enumerate(comps) if idx != marker_comp and comp["signature"] == target_signature]
    assert start_raw in raw_object_ids, "marker must touch one of the repeated-shape objects"

    remap = {raw_id: obj_id for obj_id, raw_id in enumerate(raw_object_ids)}
    objects = [comps[raw_id] for raw_id in raw_object_ids]
    cell_to_object = {
        cell: remap[raw_id]
        for raw_id in raw_object_ids
        for cell in comps[raw_id]["coords"]
    }

    adjacency = [set() for _ in objects]
    for obj_id, comp in enumerate(objects):
        for r, c in comp["coords"]:
            for nbr in _neighbors4(r, c):
                other = cell_to_object.get(nbr)
                if other is not None and other != obj_id:
                    adjacency[obj_id].add(other)

    start_id = remap[start_raw]
    return marker_cell, objects, adjacency, start_id


def _chain_order(adjacency: list[set[int]], start_id: int) -> list[int]:
    order = [start_id]
    visited = {start_id}
    prev = None
    cur = start_id
    while True:
        candidates = [nbr for nbr in adjacency[cur] if nbr != prev and nbr not in visited]
        assert len(candidates) <= 1, "chain objects must not branch"
        if not candidates:
            break
        nxt = candidates[0]
        order.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a chain of same-shaped objects by shifting colors forward from the maroon-marked start.

    Input structure:
    - The background is black.
    - The foreground contains multiple monochrome objects with identical shapes but different colors.
    - Exactly one singleton maroon pixel sits 4-neigh adjacent to one object and serves only as a start marker.

    Rule:
    - Find the unique chain of 4-neigh adjacent objects that starts at the object touching the maroon marker.
    - Recolor each object on that chain with the original color of the next object in chain order.
    - Recolor the last chain object maroon, remove the maroon marker itself to black, and leave isolated same-shaped objects unchanged.

    Key insight:
    - The maroon pixel identifies where to start reading object-to-object adjacency; the actual color change is a one-step forward transfer along that chain.
    """
    g = np.asarray(grid, dtype=int)
    marker_cell, objects, adjacency, start_id = _extract_scene(g)
    chain = _chain_order(adjacency, start_id)

    out = g.copy()
    out[marker_cell] = BG
    original_colors = [int(obj["color"]) for obj in objects]

    for idx, obj_id in enumerate(chain):
        new_color = MAROON if idx == len(chain) - 1 else original_colors[chain[idx + 1]]
        for r, c in objects[obj_id]["coords"]:
            out[r, c] = new_color
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _make_shape_mask() -> np.ndarray:
    while True:
        family = random.choice(["rect", "ell", "tee", "plus", "frame"])
        if family == "rect":
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            mask = np.ones((h, w), dtype=bool)
        elif family == "ell":
            h = random.randint(3, 4)
            w = random.randint(3, 4)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[:, 0] = True
        elif family == "tee":
            h = random.randint(3, 4)
            w = random.randint(3, 4)
            mask = np.zeros((h, w), dtype=bool)
            stem_col = random.randrange(w)
            mask[0, :] = True
            mask[:, stem_col] = True
        elif family == "plus":
            mask = np.zeros((3, 3), dtype=bool)
            mask[1, :] = True
            mask[:, 1] = True
        else:
            h = random.randint(3, 4)
            w = random.randint(3, 4)
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True

        mask = np.rot90(mask, random.randint(0, 3))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = _trim_mask(mask)

        coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        h, w = mask.shape
        if all(_touching(coords, {(r + dr, c + dc) for r, c in coords}) for dr, dc in [(-h, 0), (h, 0), (0, -w), (0, w)]):
            return mask


def _touching(a: set[Coord], b: set[Coord]) -> bool:
    if a & b:
        return False
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _marker_ports(shape: set[Coord]) -> dict[str, list[Coord]]:
    ports = {"up": [], "down": [], "left": [], "right": []}
    for r, c in shape:
        if (r - 1, c) not in shape:
            ports["up"].append((r - 1, c))
        if (r + 1, c) not in shape:
            ports["down"].append((r + 1, c))
        if (r, c - 1) not in shape:
            ports["left"].append((r, c - 1))
        if (r, c + 1) not in shape:
            ports["right"].append((r, c + 1))
    return ports


def _build_path(length: int) -> list[Coord]:
    if length == 1:
        return [(0, 0)]
    for _ in range(200):
        path = [(0, 0)]
        used = {path[0]}
        while len(path) < length:
            cr, cc = path[-1]
            candidates: list[Coord] = []
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                nxt = (nr, nc)
                if nr < -COARSE_LIMIT or nr > COARSE_LIMIT or nc < -COARSE_LIMIT or nc > COARSE_LIMIT:
                    continue
                if nxt in used:
                    continue
                if any(abs(nr - ur) + abs(nc - uc) == 1 for ur, uc in used if (ur, uc) != path[-1]):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            path.append(random.choice(candidates))
            used.add(path[-1])
        if len(path) == length:
            return path
    raise RuntimeError("could not build an induced chain path")


def _pick_isolated(count: int, occupied: set[Coord]) -> list[Coord]:
    candidates = [
        (r, c)
        for r in range(-COARSE_LIMIT, COARSE_LIMIT + 1)
        for c in range(-COARSE_LIMIT, COARSE_LIMIT + 1)
    ]
    random.shuffle(candidates)
    chosen: list[Coord] = []
    used = set(occupied)
    for cand in candidates:
        if cand in used:
            continue
        if any(abs(cand[0] - pr) + abs(cand[1] - pc) <= 1 for pr, pc in used):
            continue
        chosen.append(cand)
        used.add(cand)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("could not place isolated objects")


def _translate_shape(shape: set[Coord], anchor: Coord, h: int, w: int) -> set[Coord]:
    ar, ac = anchor
    return {(ar * h + r, ac * w + c) for r, c in shape}


def _expected_output(grid: np.ndarray, object_cells: list[set[Coord]], chain_len: int, colors: list[int], marker: Coord) -> np.ndarray:
    out = grid.copy()
    out[marker] = BG
    for idx in range(chain_len):
        color = MAROON if idx == chain_len - 1 else colors[idx + 1]
        for r, c in object_cells[idx]:
            out[r, c] = color
    return out


def _validate_layout(object_cells: list[set[Coord]], chain_len: int, marker: Coord) -> None:
    for i, cells_i in enumerate(object_cells):
        for j in range(i + 1, len(object_cells)):
            cells_j = object_cells[j]
            assert not (cells_i & cells_j), "objects must not overlap"

    graph = [set() for _ in object_cells]
    for i, cells_i in enumerate(object_cells):
        for j in range(i + 1, len(object_cells)):
            if _touching(cells_i, object_cells[j]):
                graph[i].add(j)
                graph[j].add(i)

    marker_hits = [idx for idx, cells in enumerate(object_cells) if any(abs(marker[0] - r) + abs(marker[1] - c) == 1 for r, c in cells)]
    assert marker_hits == [0], "marker must touch only the start object"

    for idx in range(chain_len):
        if chain_len == 1:
            expected_degree = 0
        elif idx == 0 or idx == chain_len - 1:
            expected_degree = 1
        else:
            expected_degree = 2
        assert len(graph[idx]) == expected_degree, "chain layout must stay a simple path"
        if idx > 0:
            assert idx - 1 in graph[idx]
        if idx + 1 < chain_len:
            assert idx + 1 in graph[idx]

    for idx in range(chain_len, len(object_cells)):
        assert not graph[idx], "non-chain objects must stay isolated"


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    mask = _make_shape_mask()
    shape = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    h, w = mask.shape

    chain_len = random.randint(1, 5)
    min_isolated = 1 if chain_len == 1 else 0
    isolated_count = random.randint(min_isolated, min(3, 8 - chain_len))

    chain_anchors = _build_path(chain_len)
    isolated_anchors = _pick_isolated(isolated_count, set(chain_anchors))
    anchors = chain_anchors + isolated_anchors

    ports = _marker_ports(shape)
    side_steps = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    free_sides = [
        side
        for side, step in side_steps.items()
        if ports[side] and (chain_anchors[0][0] + step[0], chain_anchors[0][1] + step[1]) not in set(anchors)
    ]
    if not free_sides:
        raise RuntimeError("no free side for the marker")

    marker_local = random.choice(ports[random.choice(free_sides)])
    marker_abs = (chain_anchors[0][0] * h + marker_local[0], chain_anchors[0][1] * w + marker_local[1])

    object_cells_raw = [_translate_shape(shape, anchor, h, w) for anchor in anchors]
    _validate_layout(object_cells_raw, chain_len, marker_abs)

    all_cells = {marker_abs}
    for cells in object_cells_raw:
        all_cells |= cells
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    scene_h = max_r - min_r + 1
    scene_w = max_c - min_c + 1
    if scene_h > 30 or scene_w > 30:
        raise RuntimeError("scene does not fit ARC bounds")

    extra_h = random.randint(0, min(8, 30 - scene_h))
    extra_w = random.randint(0, min(8, 30 - scene_w))
    top_pad = random.randint(0, extra_h)
    left_pad = random.randint(0, extra_w)
    out_h = scene_h + extra_h
    out_w = scene_w + extra_w
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c

    object_cells = [
        {(r + shift_r, c + shift_c) for r, c in cells}
        for cells in object_cells_raw
    ]
    marker = (marker_abs[0] + shift_r, marker_abs[1] + shift_c)

    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8], len(object_cells))
    grid = np.zeros((out_h, out_w), dtype=int)
    for color, cells in zip(colors, object_cells, strict=True):
        for r, c in cells:
            grid[r, c] = color
    grid[marker] = MAROON

    expected = _expected_output(grid, object_cells, chain_len, colors, marker)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, expected = _build_case()
            if np.array_equal(main(grid), expected):
                return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
