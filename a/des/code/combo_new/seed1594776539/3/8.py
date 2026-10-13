from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GRID_MIN = 12
GRID_MAX = 30
GEN_ATTEMPTS = 300
TREE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
DEPTH_CHOICES = [2, 3, 4, 5]
TREE_COUNT_CHOICES = [1, 2, 3]
TREE_COUNT_WEIGHTS = [2, 3, 2]
COLOR_POOL = tuple(range(1, 10))
MAX_TREE_CELLS = 18


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
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


def _degrees(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    coord_set = set(coords)
    return {
        coord: sum((coord[0] + dr, coord[1] + dc) in coord_set for dr, dc in DIRS4)
        for coord in coords
    }


def _identify_root_and_leaves(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    deg = _degrees(coords)
    endpoints = [coord for coord in coords if deg[coord] == 1]
    assert len(endpoints) >= 2, "each tree must have at least one root and one leaf"

    center_r = sum(r for r, _ in coords) / len(coords)
    center_c = sum(c for _, c in coords) / len(coords)
    scored = sorted(
        (((r - center_r) ** 2 + (c - center_c) ** 2), r, c)
        for r, c in endpoints
    )
    if len(scored) > 1:
        assert not math.isclose(scored[0][0], scored[1][0], rel_tol=0.0, abs_tol=1e-9), (
            "root endpoint must be uniquely closest to the tree center"
        )
    root = (scored[0][1], scored[0][2])
    leaves = [coord for coord in endpoints if coord != root]
    assert leaves, "expected at least one leaf endpoint"
    return root, leaves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each pixel tree's root color with its leaf color layer.

    Input structure:
    - The background is black.
    - Each non-black 4-connected component is a multicolor pixel tree.
    - In every tree, endpoints are pixels with exactly one same-tree 4-neighbor;
      one endpoint lies near the component center and the other endpoints are
      outer leaves.

    Rule:
    - For each tree, find all endpoints and choose the unique endpoint closest to
      the component's coordinate mean as the root.
    - Read the root color A and the common leaf color Z.
    - Recolor every leaf endpoint to A and recolor the root pixel to Z.
    - Leave all intermediate tree pixels and the black background unchanged.

    Key insight:
    - Root and leaves look identical locally because they are all endpoints, so
      the tree must be interpreted globally: the centered endpoint is the root
      and the peripheral endpoints are the leaves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for comp in _components(g != 0):
        root, leaves = _identify_root_and_leaves(comp)
        root_color = int(g[root])
        leaf_colors = {int(g[r, c]) for r, c in leaves}
        assert len(leaf_colors) == 1, "all leaves should share one outer-layer color"
        leaf_color = next(iter(leaf_colors))

        out[root] = leaf_color
        for r, c in leaves:
            out[r, c] = root_color

    return out


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _valid_child_dirs(
    coord: tuple[int, int],
    incoming: tuple[int, int],
    depth_map: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    for direction in (incoming, _turn_left(incoming), _turn_right(incoming)):
        child = (coord[0] + direction[0], coord[1] + direction[1])
        if child in depth_map:
            continue
        if any(nei in depth_map and nei != coord for nei in _neighbors4(*child)):
            continue
        valid.append(direction)
    return valid


def _sample_child_count(
    rng: random.Random,
    valid_count: int,
    depth: int,
    max_depth: int,
    is_hub: bool,
) -> int:
    if is_hub:
        options = [k for k in (1, 2, 3) if k <= valid_count]
        weights = {1: 1, 2: 5, 3: 3}
    elif depth == max_depth - 1:
        options = [k for k in (1, 2) if k <= valid_count]
        weights = {1: 4, 2: 2}
    else:
        options = [k for k in (1, 2, 3) if k <= valid_count]
        weights = {1: 5, 2: 3, 3: 1}
    return rng.choices(options, weights=[weights[k] for k in options])[0]


def _generate_tree_depths(rng: random.Random) -> dict[tuple[int, int], int] | None:
    for _ in range(TREE_ATTEMPTS):
        max_depth = rng.choice(DEPTH_CHOICES)
        depth_map: dict[tuple[int, int], int] = {(-1, 0): 0, (0, 0): 1}
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = [((0, 0), (1, 0))]
        ok = True

        for depth in range(1, max_depth):
            rng.shuffle(frontier)
            next_frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for coord, incoming in frontier:
                valid_dirs = _valid_child_dirs(coord, incoming, depth_map)
                if not valid_dirs:
                    ok = False
                    break
                child_count = _sample_child_count(
                    rng=rng,
                    valid_count=len(valid_dirs),
                    depth=depth,
                    max_depth=max_depth,
                    is_hub=(depth == 1 and coord == (0, 0)),
                )
                for direction in rng.sample(valid_dirs, child_count):
                    child = (coord[0] + direction[0], coord[1] + direction[1])
                    depth_map[child] = depth + 1
                    next_frontier.append((child, direction))
                if len(depth_map) > MAX_TREE_CELLS:
                    ok = False
                    break
            if not ok or not next_frontier:
                ok = False
                break
            frontier = next_frontier

        if not ok:
            continue

        coords = list(depth_map)
        try:
            root, leaves = _identify_root_and_leaves(coords)
        except AssertionError:
            continue
        if root != (-1, 0):
            continue
        if any(depth_map[leaf] != max_depth for leaf in leaves):
            continue
        if len(leaves) < 2:
            continue

        return depth_map
    return None


def _transform_coord(coord: tuple[int, int], variant: int) -> tuple[int, int]:
    r, c = coord
    if variant == 0:
        return r, c
    if variant == 1:
        return r, -c
    if variant == 2:
        return -r, c
    if variant == 3:
        return -r, -c
    if variant == 4:
        return c, r
    if variant == 5:
        return c, -r
    if variant == 6:
        return -c, r
    return -c, -r


def _normalize_depth_map(depth_map: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    min_r = min(r for r, _ in depth_map)
    min_c = min(c for _, c in depth_map)
    return {(r - min_r, c - min_c): depth for (r, c), depth in depth_map.items()}


def _build_tree_grid(rng: random.Random) -> np.ndarray | None:
    depth_map = _generate_tree_depths(rng)
    if depth_map is None:
        return None

    variant = rng.randrange(8)
    transformed = {
        _transform_coord(coord, variant): depth
        for coord, depth in depth_map.items()
    }
    normalized = _normalize_depth_map(transformed)
    max_depth = max(normalized.values())
    colors = rng.sample(COLOR_POOL, max_depth + 1)

    h = max(r for r, _ in normalized) + 1
    w = max(c for _, c in normalized) + 1
    tree = np.zeros((h, w), dtype=int)
    for (r, c), depth in normalized.items():
        tree[r, c] = colors[depth]

    return tree


def _reserve_buffer(
    reserved: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    h: int,
    w: int,
) -> None:
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                reserved.add((nr, nc))


def _place_trees(rng: random.Random, trees: list[np.ndarray], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    reserved: set[tuple[int, int]] = set()
    ordered = sorted(trees, key=lambda arr: int(np.count_nonzero(arr)), reverse=True)

    for tree in ordered:
        th, tw = tree.shape
        pixels = [(int(r), int(c), int(tree[r, c])) for r, c in np.argwhere(tree != 0)]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, h - th)
            left = rng.randint(0, w - tw)
            translated = [(top + r, left + c, color) for r, c, color in pixels]
            if any((r, c) in reserved for r, c, _ in translated):
                continue
            for r, c, color in translated:
                grid[r, c] = color
            _reserve_buffer(reserved, [(r, c) for r, c, _ in translated], h, w)
            placed = True
            break
        if not placed:
            return None

    return grid


def _validate_grid(grid: np.ndarray) -> bool:
    comps = _components(grid != 0)
    if not comps:
        return False
    for comp in comps:
        try:
            root, leaves = _identify_root_and_leaves(comp)
        except AssertionError:
            return False
        if len({int(grid[r, c]) for r, c in leaves}) != 1:
            return False
        if int(grid[root]) == 0:
            return False
    out = main(grid)
    return np.any(out != grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    tree_count = rng.choices(TREE_COUNT_CHOICES, weights=TREE_COUNT_WEIGHTS)[0]
    if h * w < 220:
        tree_count = min(tree_count, 2)

    trees: list[np.ndarray] = []
    for _ in range(tree_count):
        tree = _build_tree_grid(rng)
        if tree is None:
            return None
        if tree.shape[0] > h or tree.shape[1] > w:
            return None
        trees.append(tree)

    grid = _place_trees(rng, trees, h, w)
    if grid is None or not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-color-swap input")
