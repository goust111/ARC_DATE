from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEED = 2
GREEN = 3
GREY = 5
OUTLINE = 8

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Tree = tuple["Tree", ...]

LEAF: Tree = ()
CHAIN2: Tree = (LEAF,)
CHAIN3: Tree = (CHAIN2,)
SIBLINGS: Tree = (LEAF, LEAF)
SIBLING_CHAIN: Tree = (LEAF, CHAIN2)

TOPOLOGIES: tuple[tuple[Tree, ...], ...] = (
    (CHAIN2,),
    (CHAIN3,),
    (SIBLINGS,),
    (SIBLING_CHAIN,),
    (CHAIN2, LEAF),
    (SIBLINGS, LEAF),
)

GRID_MIN = 14
GRID_MAX = 22
MAX_ATTEMPTS = 400
MIN_RECT_SIDE = 5
ROOT_GAP = 2
CHILD_GAP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    lab = np.zeros((h, w), dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or lab[r, c] != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            lab[r, c] = next_id
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and lab[nr, nc] == 0:
                        lab[nr, nc] = next_id
                        q.append((nr, nc))
    return lab, next_id


def _dims(slot: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = slot
    return r1 - r0 + 1, c1 - c0 + 1


def _prune_dim_options(options: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pruned: list[tuple[int, int]] = []
    for h, w in sorted(options):
        dominated = False
        for hh, ww in options:
            if (hh, ww) != (h, w) and hh <= h and ww <= w:
                dominated = True
                break
        if not dominated:
            pruned.append((int(h), int(w)))
    return pruned


def _tree_dim_options(tree: Tree) -> list[tuple[int, int]]:
    if not tree:
        return [(MIN_RECT_SIDE, MIN_RECT_SIDE)]
    if len(tree) == 1:
        child_options = _tree_dim_options(tree[0])
        return _prune_dim_options({(h + 4, w + 4) for h, w in child_options})

    left_options = _tree_dim_options(tree[0])
    right_options = _tree_dim_options(tree[1])
    options: set[tuple[int, int]] = set()
    for h1, w1 in left_options:
        for h2, w2 in right_options:
            options.add((max(h1, h2) + 4, w1 + w2 + CHILD_GAP + 4))
            options.add((h1 + h2 + CHILD_GAP + 4, max(w1, w2) + 4))
    return _prune_dim_options(options)


def _random_bbox_in_slot(slot: BBox, min_h: int, min_w: int, rng: random.Random) -> BBox | None:
    slot_h, slot_w = _dims(slot)
    if slot_h < min_h or slot_w < min_w:
        return None

    max_h = min(slot_h, min_h + rng.randint(0, 5))
    max_w = min(slot_w, min_w + rng.randint(0, 5))
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)

    r0, c0, r1, c1 = slot
    top = rng.randint(r0, r1 - h + 1)
    left = rng.randint(c0, c1 - w + 1)
    return (top, left, top + h - 1, left + w - 1)


def _place_tree(tree: Tree, slot: BBox, depth: int, rng: random.Random) -> list[dict] | None:
    slot_h, slot_w = _dims(slot)
    base_options = [(h, w) for h, w in _tree_dim_options(tree) if h <= slot_h and w <= slot_w]
    if not base_options:
        return None

    child_trees = list(tree)
    if len(child_trees) == 2 and rng.random() < 0.5:
        child_trees.reverse()

    for _ in range(40):
        min_h, min_w = rng.choice(base_options)
        bbox = _random_bbox_in_slot(slot, min_h, min_w, rng)
        if bbox is None:
            continue

        node = {"bbox": bbox, "depth": depth}
        nodes = [node]
        if not child_trees:
            return nodes

        r0, c0, r1, c1 = bbox
        inner = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
        inner_h, inner_w = _dims(inner)
        if inner_h <= 0 or inner_w <= 0:
            continue

        if len(child_trees) == 1:
            child_nodes = _place_tree(child_trees[0], inner, depth + 1, rng)
            if child_nodes is not None:
                return nodes + child_nodes
            continue

        left_opts = _tree_dim_options(child_trees[0])
        right_opts = _tree_dim_options(child_trees[1])
        layouts: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        for left_need in left_opts:
            for right_need in right_opts:
                h1, w1 = left_need
                h2, w2 = right_need
                if inner_h >= max(h1, h2) and inner_w >= w1 + CHILD_GAP + w2:
                    layouts.append(("h", left_need, right_need))
                if inner_w >= max(w1, w2) and inner_h >= h1 + CHILD_GAP + h2:
                    layouts.append(("v", left_need, right_need))
        if not layouts:
            continue

        layout, left_need, right_need = rng.choice(layouts)
        ir0, ic0, ir1, ic1 = inner
        if layout == "h":
            left_w = rng.randint(left_need[1], inner_w - CHILD_GAP - right_need[1])
            left_slot = (ir0, ic0, ir1, ic0 + left_w - 1)
            right_slot = (ir0, ic0 + left_w + CHILD_GAP, ir1, ic1)
        else:
            top_h = rng.randint(left_need[0], inner_h - CHILD_GAP - right_need[0])
            left_slot = (ir0, ic0, ir0 + top_h - 1, ic1)
            right_slot = (ir0 + top_h + CHILD_GAP, ic0, ir1, ic1)

        left_nodes = _place_tree(child_trees[0], left_slot, depth + 1, rng)
        if left_nodes is None:
            continue
        right_nodes = _place_tree(child_trees[1], right_slot, depth + 1, rng)
        if right_nodes is None:
            continue
        return nodes + left_nodes + right_nodes

    return None


def _place_forest(forest: tuple[Tree, ...], h: int, w: int, rng: random.Random) -> list[dict] | None:
    root_slot = (1, 1, h - 2, w - 2)
    forest = tuple(forest)
    if len(forest) == 1:
        return _place_tree(forest[0], root_slot, 1, rng)

    roots = list(forest)
    if rng.random() < 0.5:
        roots.reverse()

    slot_h, slot_w = _dims(root_slot)
    left_opts = _tree_dim_options(roots[0])
    right_opts = _tree_dim_options(roots[1])
    layouts: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for left_need in left_opts:
        for right_need in right_opts:
            h1, w1 = left_need
            h2, w2 = right_need
            if slot_h >= max(h1, h2) and slot_w >= w1 + ROOT_GAP + w2:
                layouts.append(("h", left_need, right_need))
            if slot_w >= max(w1, w2) and slot_h >= h1 + ROOT_GAP + h2:
                layouts.append(("v", left_need, right_need))
    if not layouts:
        return None

    for _ in range(40):
        layout, left_need, right_need = rng.choice(layouts)
        r0, c0, r1, c1 = root_slot
        if layout == "h":
            left_w = rng.randint(left_need[1], slot_w - ROOT_GAP - right_need[1])
            left_slot = (r0, c0, r1, c0 + left_w - 1)
            right_slot = (r0, c0 + left_w + ROOT_GAP, r1, c1)
        else:
            top_h = rng.randint(left_need[0], slot_h - ROOT_GAP - right_need[0])
            left_slot = (r0, c0, r0 + top_h - 1, c1)
            right_slot = (r0 + top_h + ROOT_GAP, c0, r1, c1)

        left_nodes = _place_tree(roots[0], left_slot, 1, rng)
        if left_nodes is None:
            continue
        right_nodes = _place_tree(roots[1], right_slot, 1, rng)
        if right_nodes is None:
            continue
        return left_nodes + right_nodes
    return None


def _draw_rectangle_outline(grid: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = OUTLINE
    grid[r1, c0 : c1 + 1] = OUTLINE
    grid[r0 : r1 + 1, c0] = OUTLINE
    grid[r0 : r1 + 1, c1] = OUTLINE


def _build_graph_from_outlines(grid: np.ndarray, region_lab: np.ndarray, n_regions: int) -> dict[int, set[int]]:
    h, w = grid.shape
    outline_lab, n_outlines = _label_mask(grid == OUTLINE)
    assert n_outlines >= 1, "expected at least one closed outline"

    graph = {rid: set() for rid in range(1, n_regions + 1)}
    for outline_id in range(1, n_outlines + 1):
        touching: set[int] = set()
        coords = np.argwhere(outline_lab == outline_id)
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    rid = int(region_lab[nr, nc])
                    if rid > 0:
                        touching.add(rid)
        assert len(touching) == 2, "each closed outline must separate exactly two regions"
        a, b = sorted(touching)
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _analyze_partition(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    region_lab, n_regions = _label_mask(g != OUTLINE)
    assert n_regions >= 2, "outlines must split the canvas into multiple regions"

    border_regions = set(map(int, region_lab[0, :])) | set(map(int, region_lab[-1, :]))
    border_regions |= set(map(int, region_lab[:, 0])) | set(map(int, region_lab[:, -1]))
    border_regions.discard(0)
    assert len(border_regions) == 1, "expected a unique outside region"
    outside = int(next(iter(border_regions)))

    graph = _build_graph_from_outlines(g, region_lab, n_regions)
    parity = {outside: 0}
    depth = {outside: 0}
    q = deque([outside])
    while q:
        rid = q.popleft()
        for nbr in graph[rid]:
            want = 1 - parity[rid]
            if nbr in parity:
                assert parity[nbr] == want, "region adjacency must be bipartite"
                continue
            parity[nbr] = want
            depth[nbr] = depth[rid] + 1
            q.append(nbr)

    assert len(parity) == n_regions, "all regions must be connected through outlines"
    region_cells = {rid: np.argwhere(region_lab == rid) for rid in range(1, n_regions + 1)}
    seed_counts = {
        rid: int(np.count_nonzero(g[region_lab == rid] == SEED))
        for rid in range(1, n_regions + 1)
    }
    return {
        "region_lab": region_lab,
        "n_regions": n_regions,
        "outside": outside,
        "graph": graph,
        "parity": parity,
        "depth": depth,
        "region_cells": region_cells,
        "seed_counts": seed_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed-activated outline regions with alternating parity colors.

    Input structure:
    - A black background is partitioned by one-cell-thick cyan closed outlines.
    - The non-outline cells form 4-connected regions, including one outside region
      touching the border and several enclosed interior regions.
    - Some interior regions contain one red seed pixel; outline pixels remain cyan.

    Rule:
    - Treat cyan pixels as walls and label every 4-connected non-outline region.
    - Build the region adjacency graph by connecting the two regions that lie on
      opposite sides of each closed outline, with the outside region fixed as the
      first bipartite side.
    - Any region containing a red seed is activated: activated regions on the
      outside-side parity become grey, activated regions on the opposite parity
      become green, and the fill replaces both black cells and the seed itself.
    - Regions without seeds stay black, and all cyan outline pixels stay unchanged.

    Key insight:
    - The red seed only decides which region grows; the final grey-versus-green
      color comes entirely from the region's parity in the outline-defined adjacency graph.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _analyze_partition(g)
    region_lab = info["region_lab"]
    parity = info["parity"]

    out = g.copy()
    for rid in range(1, info["n_regions"] + 1):
        mask = region_lab == rid
        if np.any(g[mask] == SEED):
            out[mask] = GREY if parity[rid] == 0 else GREEN
    return out


def _render_outline_forest(nodes: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for node in nodes:
        _draw_rectangle_outline(grid, node["bbox"])
    return grid


def _choose_seed_regions(info: dict, rng: random.Random) -> set[int] | None:
    outside = info["outside"]
    candidates = [rid for rid in range(1, info["n_regions"] + 1) if rid != outside]
    if len(candidates) < 2:
        return None

    odd = [rid for rid in candidates if info["parity"][rid] == 1]
    even = [rid for rid in candidates if info["parity"][rid] == 0]

    for _ in range(80):
        seeded: set[int] = set()
        if odd:
            seeded.add(rng.choice(odd))
        if even and rng.random() < 0.8:
            seeded.add(rng.choice(even))
        for rid in candidates:
            if rid not in seeded and rng.random() < 0.35:
                seeded.add(rid)

        if not seeded or len(seeded) == len(candidates):
            continue
        if any((rid in seeded) != (nbr in seeded) for rid in candidates for nbr in info["graph"][rid]):
            return seeded
    return None


def _place_seeds(grid: np.ndarray, seeded_regions: set[int], info: dict, rng: random.Random) -> np.ndarray | None:
    out = grid.copy()
    region_lab = info["region_lab"]
    for rid in seeded_regions:
        cells = [tuple(map(int, rc)) for rc in np.argwhere(region_lab == rid)]
        if not cells:
            return None
        r, c = rng.choice(cells)
        out[r, c] = SEED
    return out


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        info = _analyze_partition(inp)
    except AssertionError:
        return False

    interior = [rid for rid in range(1, info["n_regions"] + 1) if rid != info["outside"]]
    if len(interior) < 2:
        return False

    seeded = {rid for rid in interior if info["seed_counts"][rid] == 1}
    if not seeded or len(seeded) == len(interior):
        return False
    if info["seed_counts"][info["outside"]] != 0:
        return False
    if any(info["seed_counts"][rid] > 1 for rid in interior):
        return False
    if not any((rid in seeded) != (nbr in seeded) for rid in interior for nbr in info["graph"][rid]):
        return False

    colors = set(map(int, out.flat))
    if OUTLINE not in colors:
        return False
    if not colors.issubset({BG, GREEN, GREY, OUTLINE}):
        return False
    if SEED in colors:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    topology = rng.choice(TOPOLOGIES)

    nodes = _place_forest(topology, h, w, rng)
    if nodes is None or len(nodes) < 2:
        return None

    base = _render_outline_forest(nodes, h, w)
    try:
        info = _analyze_partition(base)
    except AssertionError:
        return None

    if info["n_regions"] != len(nodes) + 1:
        return None

    seeded_regions = _choose_seed_regions(info, rng)
    if seeded_regions is None:
        return None

    inp = _place_seeds(base, seeded_regions, info, rng)
    if inp is None:
        return None

    out = main(inp)
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid outline-partition input")
