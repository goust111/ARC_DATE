from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["up", "down", "left", "right"]
SIDE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    colors: list[int] = []
    cells: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1 or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            cid = len(colors)
            stack = [(r, c)]
            labels[r, c] = cid
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if labels[nr, nc] != -1 or int(g[nr, nc]) != color:
                        continue
                    labels[nr, nc] = cid
                    stack.append((nr, nc))
            colors.append(color)
            cells.append(np.array(pts, dtype=int))

    return colors, cells, labels


def _component_graph(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors, cells, labels = _find_components(g, bg)
    h, w = g.shape
    neighbors = [set() for _ in colors]

    for cid, pts in enumerate(cells):
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                nid = int(labels[nr, nc])
                if nid != -1 and nid != cid:
                    neighbors[cid].add(nid)

    return bg, colors, cells, neighbors


def _vote_results(colors: list[int], neighbors: list[set[int]]):
    new_colors = list(colors)
    cases: list[str] = []

    for cid, nbrs in enumerate(neighbors):
        if not nbrs:
            cases.append("isolated")
            continue
        counts = Counter(int(colors[nid]) for nid in nbrs)
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        if len(winners) == 1:
            cases.append("majority")
            new_colors[cid] = int(winners[0])
        else:
            cases.append("tie")
    return cases, new_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by the most frequent color among its
    orthogonally adjacent neighbor objects.
    """
    g = np.asarray(grid, dtype=int)
    _bg, colors, cells, neighbors = _component_graph(g)
    _cases, new_colors = _vote_results(colors, neighbors)

    out = g.copy()
    for cid, pts in enumerate(cells):
        out[pts[:, 0], pts[:, 1]] = int(new_colors[cid])
    return out


# --------------- generator ---------------

ORANGE = 7
PALETTE = [c for c in range(10) if c != ORANGE]

MACRO_H_RANGE = (4, 6)
MACRO_W_RANGE = (4, 6)
TILE_H_RANGE = (5, 7)
TILE_W_RANGE = (5, 7)
EXTRA_ISOLATED_RANGE = (0, 3)
EXTRA_PAIR_PROB = 0.5
PALETTE_SIZE_RANGE = (4, 6)
GENERATION_ATTEMPTS = 400


def _step(cell: tuple[int, int], side: str) -> tuple[int, int]:
    dr, dc = SIDE_DELTAS[side]
    return int(cell[0] + dr), int(cell[1] + dc)


def _in_bounds(cell: tuple[int, int], mh: int, mw: int) -> bool:
    return 0 <= cell[0] < mh and 0 <= cell[1] < mw


def _edge_key(a: tuple[int, int], b: tuple[int, int]):
    return tuple(sorted((a, b)))


def _place_pattern(
    occupied: set[tuple[int, int]],
    mh: int,
    mw: int,
    side_options: list[tuple[str, ...]],
):
    choices: list[tuple[tuple[int, int], tuple[str, ...], dict[str, tuple[int, int]]]] = []
    centers = [(r, c) for r in range(mh) for c in range(mw)]
    random.shuffle(centers)
    random.shuffle(side_options)
    for center in centers:
        if center in occupied:
            continue
        for sides in side_options:
            leaves: dict[str, tuple[int, int]] = {}
            ok = True
            coords = {center}
            for side in sides:
                nbr = _step(center, side)
                if not _in_bounds(nbr, mh, mw) or nbr in occupied or nbr in coords:
                    ok = False
                    break
                leaves[side] = nbr
                coords.add(nbr)
            if ok:
                choices.append((center, sides, leaves))
    if not choices:
        return None
    return random.choice(choices)


def _place_majority_motif(occupied: set[tuple[int, int]], mh: int, mw: int):
    side_options = [tuple(side for side in SIDES if side != missing) for missing in SIDES]
    if mh >= 3 and mw >= 3:
        side_options.append(tuple(SIDES))
    placed = _place_pattern(occupied, mh, mw, side_options)
    if placed is None:
        return None
    center, sides, leaves = placed
    degree = len(sides)
    winner_count = 3 if degree == 4 else 2
    winner_sides = set(random.sample(list(sides), winner_count))
    winner_nodes = [leaves[side] for side in sides if side in winner_sides]
    other_nodes = [leaves[side] for side in sides if side not in winner_sides]
    edges = [_edge_key(center, leaves[side]) for side in sides]
    return {
        "center": center,
        "winners": winner_nodes,
        "others": other_nodes,
        "nodes": [center] + [leaves[side] for side in sides],
        "edges": edges,
    }


def _place_tie_motif(occupied: set[tuple[int, int]], mh: int, mw: int):
    side_options = []
    pairs = []
    for i, side_a in enumerate(SIDES):
        for side_b in SIDES[i + 1 :]:
            pairs.append((side_a, side_b))
    side_options.extend(pairs)
    if mh >= 3 and mw >= 3:
        side_options.append(tuple(SIDES))
    placed = _place_pattern(occupied, mh, mw, side_options)
    if placed is None:
        return None
    center, sides, leaves = placed
    leaf_nodes = [leaves[side] for side in sides]
    edges = [_edge_key(center, leaves[side]) for side in sides]
    return {
        "center": center,
        "leaves": leaf_nodes,
        "nodes": [center] + leaf_nodes,
        "edges": edges,
    }


def _random_empty_cell(occupied: set[tuple[int, int]], mh: int, mw: int):
    options = [(r, c) for r in range(mh) for c in range(mw) if (r, c) not in occupied]
    if not options:
        return None
    return random.choice(options)


def _place_extra_pair(occupied: set[tuple[int, int]], mh: int, mw: int):
    options = []
    for r in range(mh):
        for c in range(mw):
            a = (r, c)
            if a in occupied:
                continue
            for side in SIDES:
                b = _step(a, side)
                if not _in_bounds(b, mh, mw) or b in occupied:
                    continue
                options.append((a, b))
    if not options:
        return None
    a, b = random.choice(options)
    return {"nodes": [a, b], "edges": [_edge_key(a, b)]}


def _choose_colors():
    size = random.randint(*PALETTE_SIZE_RANGE)
    palette = random.sample(PALETTE, size)

    def pick(exclude=()):
        excluded = set(exclude)
        choices = [color for color in palette if color not in excluded]
        return int(random.choice(choices))

    return palette, pick


def _object_shape(tile_h: int, tile_w: int, ports: dict[str, int]) -> set[tuple[int, int]]:
    core_h = random.randint(1, min(2, tile_h - 2))
    core_w = random.randint(1, min(2, tile_w - 2))
    r0 = random.randint(1, tile_h - core_h - 1)
    c0 = random.randint(1, tile_w - core_w - 1)
    r1 = r0 + core_h - 1
    c1 = c0 + core_w - 1

    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    for side, offset in ports.items():
        anchor_r = random.randint(r0, r1)
        anchor_c = random.randint(c0, c1)
        if side in {"left", "right"}:
            for rr in range(min(anchor_r, offset), max(anchor_r, offset) + 1):
                cells.add((rr, anchor_c))
            edge_c = 0 if side == "left" else tile_w - 1
            for cc in range(min(anchor_c, edge_c), max(anchor_c, edge_c) + 1):
                cells.add((offset, cc))
        else:
            for cc in range(min(anchor_c, offset), max(anchor_c, offset) + 1):
                cells.add((anchor_r, cc))
            edge_r = 0 if side == "up" else tile_h - 1
            for rr in range(min(anchor_r, edge_r), max(anchor_r, edge_r) + 1):
                cells.add((rr, offset))

    extra_budget = random.randint(0, max(1, (tile_h - 2) * (tile_w - 2) // 4))
    for _ in range(extra_budget):
        frontier = []
        for r in range(1, tile_h - 1):
            for c in range(1, tile_w - 1):
                if (r, c) in cells:
                    continue
                if any((r + dr, c + dc) in cells for dr, dc in DIR4):
                    frontier.append((r, c))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    for side, offset in ports.items():
        if side == "left":
            assert (offset, 0) in cells
        elif side == "right":
            assert (offset, tile_w - 1) in cells
        elif side == "up":
            assert (0, offset) in cells
        else:
            assert (tile_h - 1, offset) in cells
    return cells


def _render_grid(
    mh: int,
    mw: int,
    tile_h: int,
    tile_w: int,
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((mh * tile_h, mw * tile_w), ORANGE, dtype=int)
    ports: dict[tuple[int, int], dict[str, int]] = {node: {} for node in nodes}

    for a, b in edges:
        if a[0] == b[0]:
            offset = random.randint(1, tile_h - 2)
            left, right = (a, b) if a[1] < b[1] else (b, a)
            ports[left]["right"] = offset
            ports[right]["left"] = offset
        else:
            offset = random.randint(1, tile_w - 2)
            top, bottom = (a, b) if a[0] < b[0] else (b, a)
            ports[top]["down"] = offset
            ports[bottom]["up"] = offset

    for node in nodes:
        shape = _object_shape(tile_h, tile_w, ports[node])
        r_base = node[0] * tile_h
        c_base = node[1] * tile_w
        color = int(colors[node])
        for r, c in shape:
            grid[r_base + r, c_base + c] = color

    return grid


def _branch_types(grid: np.ndarray) -> set[str]:
    _bg, colors, _cells, neighbors = _component_graph(grid)
    cases, _new_colors = _vote_results(colors, neighbors)
    return set(cases)


def _bg_is_dominant(grid: np.ndarray) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    by_color = {int(v): int(c) for v, c in zip(vals, counts)}
    bg_count = by_color.get(ORANGE, 0)
    fg_counts = [count for color, count in by_color.items() if color != ORANGE]
    return bg_count > max(fg_counts, default=0)


def _gen_attempt() -> np.ndarray | None:
    mh = random.randint(*MACRO_H_RANGE)
    mw = random.randint(*MACRO_W_RANGE)
    tile_h = random.randint(*TILE_H_RANGE)
    tile_w = random.randint(*TILE_W_RANGE)

    occupied: set[tuple[int, int]] = set()
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    nodes: list[tuple[int, int]] = []

    majority = _place_majority_motif(occupied, mh, mw)
    if majority is None:
        return None
    occupied.update(majority["nodes"])
    edges.extend(majority["edges"])
    nodes.extend(majority["nodes"])

    tie = _place_tie_motif(occupied, mh, mw)
    if tie is None:
        return None
    occupied.update(tie["nodes"])
    edges.extend(tie["edges"])
    nodes.extend(tie["nodes"])

    isolated = _random_empty_cell(occupied, mh, mw)
    if isolated is None:
        return None
    occupied.add(isolated)
    nodes.append(isolated)

    extra_pair = None
    if random.random() < EXTRA_PAIR_PROB:
        extra_pair = _place_extra_pair(occupied, mh, mw)
        if extra_pair is not None:
            occupied.update(extra_pair["nodes"])
            edges.extend(extra_pair["edges"])
            nodes.extend(extra_pair["nodes"])

    for _ in range(random.randint(*EXTRA_ISOLATED_RANGE)):
        cell = _random_empty_cell(occupied, mh, mw)
        if cell is None:
            break
        occupied.add(cell)
        nodes.append(cell)

    _palette, pick = _choose_colors()
    color_map: dict[tuple[int, int], int] = {}

    majority_center = pick()
    majority_winner = pick({majority_center})
    majority_other = pick({majority_center, majority_winner})
    color_map[majority["center"]] = majority_center
    for node in majority["winners"]:
        color_map[node] = majority_winner
    for node in majority["others"]:
        color_map[node] = majority_other

    tie_center = pick()
    tie_a = pick({tie_center})
    tie_b = pick({tie_center, tie_a})
    color_map[tie["center"]] = tie_center
    tie_leaves = list(tie["leaves"])
    random.shuffle(tie_leaves)
    if len(tie_leaves) == 2:
        color_map[tie_leaves[0]] = tie_a
        color_map[tie_leaves[1]] = tie_b
    else:
        split = len(tie_leaves) // 2
        for node in tie_leaves[:split]:
            color_map[node] = tie_a
        for node in tie_leaves[split:]:
            color_map[node] = tie_b

    color_map[isolated] = pick()

    assigned = set(color_map)
    if extra_pair is not None:
        a, b = extra_pair["nodes"]
        pair_a = pick()
        pair_b = pick({pair_a})
        color_map[a] = pair_a
        color_map[b] = pair_b
        assigned.update(extra_pair["nodes"])

    for node in nodes:
        if node not in color_map:
            color_map[node] = pick()

    grid = _render_grid(mh, mw, tile_h, tile_w, nodes, edges, color_map)
    if not _bg_is_dominant(grid):
        return None

    _bg, _colors, cells, _neighbors = _component_graph(grid)
    if len(cells) != len(nodes):
        return None

    if _branch_types(grid) != {"majority", "tie", "isolated"}:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
