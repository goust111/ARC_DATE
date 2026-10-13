from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _infer_order(side: int) -> int:
    candidates = [n for n in range(2, 6) if n * n + n - 1 == side]
    assert len(candidates) == 1, "grid side must equal n*n+n-1 for exactly one n"
    return candidates[0]


def _separator_indices(n: int) -> list[int]:
    return [k * (n + 1) - 1 for k in range(1, n)]


def _tile_origin(n: int, tile_r: int, tile_c: int) -> tuple[int, int]:
    step = n + 1
    return tile_r * step, tile_c * step


def _parse_tiles(grid: np.ndarray) -> tuple[int, int, np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    side = int(g.shape[0])
    n = _infer_order(side)
    bg = _most_common_color(g)

    for idx in _separator_indices(n):
        assert np.all(g[idx, :] == bg), "separator rows must be background"
        assert np.all(g[:, idx] == bg), "separator columns must be background"

    colors = np.zeros((n, n), dtype=int)
    parents: dict[tuple[int, int], tuple[int, int]] = {}

    for tile_r in range(n):
        for tile_c in range(n):
            r0, c0 = _tile_origin(n, tile_r, tile_c)
            tile = g[r0 : r0 + n, c0 : c0 + n]
            holes = np.argwhere(tile == bg)
            assert holes.shape == (1, 2), "each tile must contain exactly one background hole"
            non_bg = np.unique(tile[tile != bg])
            assert len(non_bg) == 1, "each tile must be monochromatic aside from its hole"

            hole_r, hole_c = map(int, holes[0])
            colors[tile_r, tile_c] = int(non_bg[0])
            parents[(tile_r, tile_c)] = (hole_r, hole_c)

    return n, bg, colors, parents


def _root_map(parents: dict[tuple[int, int], tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    memo: dict[tuple[int, int], tuple[int, int]] = {}
    visiting: set[tuple[int, int]] = set()

    def resolve(node: tuple[int, int]) -> tuple[int, int]:
        if node in memo:
            return memo[node]
        assert node not in visiting, "pointer graph must be acyclic"
        visiting.add(node)
        parent = parents[node]
        if parent == node:
            root = node
        else:
            root = resolve(parent)
        visiting.remove(node)
        memo[node] = root
        return root

    for node in parents:
        resolve(node)
    return memo


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each pointer tile with the color of the root tile reached through its hole chain.

    Input structure:
    - A square grid is partitioned by single background separator rows and columns into an
      n by n array of n by n monochrome tiles.
    - Every tile has exactly one background hole inside it.
    - The hole's local coordinate (r, c) names the next tile in the array: row r, column c.

    Rule:
    - Treat each hole as a coordinate pointer to another tile.
    - From every tile, follow pointer hops until reaching a root tile whose hole points to
      its own array position.
    - Keep the separator rows and columns unchanged, and repaint every tile completely
      (including its former hole) with the color of its root tile.

    Key insight:
    - The hole does not label the current tile; it points elsewhere, so the correct color is
      obtained only by propagating along the whole pointer chain to the self-pointing root.
    """
    n, bg, colors, parents = _parse_tiles(grid)
    roots = _root_map(parents)

    out = np.full_like(np.asarray(grid, dtype=int), bg)
    for tile_r in range(n):
        for tile_c in range(n):
            root_r, root_c = roots[(tile_r, tile_c)]
            color = int(colors[root_r, root_c])
            r0, c0 = _tile_origin(n, tile_r, tile_c)
            out[r0 : r0 + n, c0 : c0 + n] = color
    return out


# --------------- generator ---------------

_BG = 0
_PALETTE = tuple(range(1, 10))
_ORDER_VALUES = (2, 3, 4, 5)
_MAX_ATTEMPTS = 400
_ROOT_COUNT_MAX = 4


def _idx_to_pos(n: int, idx: int) -> tuple[int, int]:
    return divmod(idx, n)


def _follow_parent(idx: int, parent: list[int]) -> int:
    while parent[idx] != idx:
        idx = parent[idx]
    return idx


def _depths(parent: list[int]) -> list[int]:
    out = [0] * len(parent)
    for start in range(len(parent)):
        depth = 0
        cur = start
        seen = set()
        while parent[cur] != cur:
            assert cur not in seen, "generator created a cycle"
            seen.add(cur)
            cur = parent[cur]
            depth += 1
        out[start] = depth
    return out


def _one_step_output(n: int, colors: list[int], parent: list[int]) -> np.ndarray:
    side = n * n + n - 1
    out = np.full((side, side), _BG, dtype=int)
    for idx in range(n * n):
        tile_r, tile_c = _idx_to_pos(n, idx)
        parent_idx = parent[idx]
        color = int(colors[parent_idx])
        r0, c0 = _tile_origin(n, tile_r, tile_c)
        out[r0 : r0 + n, c0 : c0 + n] = color
    return out


def _render_input(n: int, colors: list[int], parent: list[int]) -> np.ndarray:
    side = n * n + n - 1
    grid = np.full((side, side), _BG, dtype=int)
    for idx in range(n * n):
        tile_r, tile_c = _idx_to_pos(n, idx)
        r0, c0 = _tile_origin(n, tile_r, tile_c)
        grid[r0 : r0 + n, c0 : c0 + n] = int(colors[idx])
        hole_r, hole_c = _idx_to_pos(n, parent[idx])
        grid[r0 + hole_r, c0 + hole_c] = _BG
    return grid


def _build_forest(n: int, rng: random.Random) -> tuple[list[int], tuple[int, int, int], list[int]]:
    total = n * n
    max_roots = min(_ROOT_COUNT_MAX, total - 2)
    root_count = rng.randint(2, max_roots)
    roots = rng.sample(range(total), root_count)

    parent = [-1] * total
    for idx in roots:
        parent[idx] = idx

    remaining = [idx for idx in range(total) if idx not in roots]
    rng.shuffle(remaining)
    chain_root = rng.choice(roots)
    mid = remaining.pop()
    leaf = remaining.pop()
    parent[mid] = chain_root
    parent[leaf] = mid

    assigned = set(roots)
    assigned.add(mid)
    assigned.add(leaf)

    extra_roots = [r for r in roots if r != chain_root]
    rng.shuffle(extra_roots)
    for root_idx in extra_roots:
        if not remaining:
            break
        child = remaining.pop()
        parent[child] = root_idx
        assigned.add(child)

    while remaining:
        child = remaining.pop()
        candidates = list(assigned)
        weights = [3 if parent[cand] != cand else 1 for cand in candidates]
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        parent[child] = chosen
        assigned.add(child)

    assert all(p >= 0 for p in parent)
    return parent, (leaf, mid, chain_root), roots


def _assign_colors(parent: list[int], roots: list[int], chain: tuple[int, int, int], rng: random.Random) -> list[int]:
    leaf, mid, chain_root = chain
    root_colors = rng.sample(_PALETTE, len(roots))
    colors = [0] * len(parent)
    for idx, color in zip(roots, root_colors):
        colors[idx] = int(color)

    for idx in range(len(parent)):
        if idx in roots:
            continue
        colors[idx] = int(rng.choice(_PALETTE))

    root_color = colors[chain_root]
    non_root_choices = [c for c in _PALETTE if c != root_color]
    colors[mid] = int(rng.choice(non_root_choices))
    if colors[leaf] == 0:
        colors[leaf] = int(rng.choice(_PALETTE))
    return colors


def _validate_semantics(inp: np.ndarray, out: np.ndarray, n: int, colors: list[int], parent: list[int], chain: tuple[int, int, int]) -> bool:
    leaf, mid, chain_root = chain
    depths = _depths(parent)
    if max(depths) < 2 or depths[leaf] < 2:
        return False

    roots = {_follow_parent(idx, parent) for idx in range(len(parent))}
    if len(roots) < 2:
        return False

    if len({colors[root] for root in roots}) < len(roots):
        return False

    if colors[mid] == colors[chain_root]:
        return False

    expected = main(inp)
    if not np.array_equal(expected, out):
        return False

    if np.array_equal(inp, out):
        return False

    if np.array_equal(_one_step_output(n, colors, parent), out):
        return False

    unique_out_colors = {int(v) for v in np.unique(out) if int(v) != _BG}
    if len(unique_out_colors) < 2:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(_ORDER_VALUES)
    parent, chain, roots = _build_forest(n, rng)
    colors = _assign_colors(parent, roots, chain, rng)
    inp = _render_input(n, colors, parent)
    out = main(inp)
    if not _validate_semantics(inp, out, n, colors, parent, chain):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-pointer-chain input")
