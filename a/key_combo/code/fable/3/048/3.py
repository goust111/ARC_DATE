from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(cells: frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[frozenset[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells = set()
        while q:
            r, c = q.popleft()
            cells.add((r, c))
            for nr, nc in _neighbors4(r, c):
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and bool(mask[nr, nc])
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(frozenset(cells))
    return out


def _split_on_bridge_edges(cells: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    """Split a gray cluster at one-edge block contacts while preserving compact pieces."""
    ordered = sorted(cells)
    index = {cell: i for i, cell in enumerate(ordered)}
    adj: list[list[int]] = [[] for _ in ordered]
    for i, (r, c) in enumerate(ordered):
        for nr, nc in _neighbors4(r, c):
            j = index.get((nr, nc))
            if j is not None:
                adj[i].append(j)

    n = len(ordered)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    bridges: set[tuple[int, int]] = set()
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = low[u] = time
        time += 1
        for v in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for start in range(n):
        if disc[start] == -1:
            dfs(start)

    seen = [False] * n
    pieces: list[frozenset[tuple[int, int]]] = []
    for start in range(n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        piece = set()
        while q:
            u = q.popleft()
            piece.add(ordered[u])
            for v in adj[u]:
                edge = (min(u, v), max(u, v))
                if edge in bridges or seen[v]:
                    continue
                seen[v] = True
                q.append(v)
        pieces.append(frozenset(piece))
    return pieces


def _detect_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY))
    for color in colors:
        for cells in _components(grid == color):
            blocks.append({"cells": cells, "color": color, "role": "seed"})

    for gray_cluster in _components(grid == GRAY):
        for cells in _split_on_bridge_edges(gray_cluster):
            blocks.append({"cells": cells, "color": None, "role": "gray"})

    assert any(b["role"] == "seed" for b in blocks), "expected at least one colored seed block"
    assert any(b["role"] == "gray" for b in blocks), "expected at least one gray block"
    return blocks


def _block_adjacency(blocks: list[dict]) -> dict[int, dict[int, int]]:
    owner: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(blocks):
        for cell in block["cells"]:
            assert cell not in owner, "blocks must not overlap"
            owner[cell] = idx

    adj: dict[int, dict[int, int]] = {i: {} for i in range(len(blocks))}
    for (r, c), i in owner.items():
        for nb in _neighbors4(r, c):
            j = owner.get(nb)
            if j is not None and i < j:
                adj[i][j] = adj[i].get(j, 0) + 1
                adj[j][i] = adj[j].get(i, 0) + 1
    return adj


def _propagate(blocks: list[dict], adj: dict[int, dict[int, int]]):
    colors: dict[int, int] = {
        idx: int(block["color"])
        for idx, block in enumerate(blocks)
        if block["color"] is not None
    }
    generations = {idx: 0 for idx in colors}
    gray_ids = [idx for idx, block in enumerate(blocks) if block["color"] is None]
    decisions = []
    generation = 0

    while any(idx not in colors for idx in gray_ids):
        ready = []
        for idx in gray_ids:
            if idx in colors:
                continue
            contacts = [
                (count, other, colors[other])
                for other, count in adj[idx].items()
                if other in colors
            ]
            if not contacts:
                continue
            best_count = max(count for count, _, _ in contacts)
            winners = [item for item in contacts if item[0] == best_count]
            assert len(winners) == 1, "each gray block must have a unique longest colored boundary"
            count, winner, color = winners[0]
            ready.append((idx, color, winner, count, contacts))

        assert ready, "every gray block must be reachable by generation-wise propagation"
        generation += 1
        for idx, color, winner, count, contacts in ready:
            colors[idx] = int(color)
            generations[idx] = generation
            decisions.append(
                {
                    "block": idx,
                    "color": int(color),
                    "winner": winner,
                    "count": count,
                    "contacts": contacts,
                    "generation": generation,
                }
            )

    return colors, generations, decisions


def _solve(grid: np.ndarray):
    blocks = _detect_blocks(grid)
    adj = _block_adjacency(blocks)
    colors, generations, decisions = _propagate(blocks, adj)
    out = grid.copy()
    for idx, block in enumerate(blocks):
        if block["role"] == "gray":
            for r, c in block["cells"]:
                out[r, c] = colors[idx]
    return out, blocks, colors, generations, decisions, adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each gray block by generation-wise propagation from the colored seed blocks.

    Input structure:
    - A black background contains several 4-connected blocks; non-gray colored blocks are generation-0 seeds and all other block pieces are gray.
    - Adjacent gray pieces touch neighboring blocks by shared sides, and compact gray pieces may meet through narrow one-edge contacts along the propagation graph.

    Rule:
    - At the start of each generation, find every still-gray block with at least one already colored side-neighbor block.
    - For each such gray block, count its shared side edges with each colored neighbor block and recolor it with the uniquely longest neighbor's color.
    - Apply all choices for that generation simultaneously, then repeat until no gray block remains.

    Key insight:
    - The answer depends on the timing of the adjacency chain: later gray blocks can only use colors that had already arrived at the start of their generation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _blocks, _colors, _generations, _decisions, _adj = _solve(g)
    return out


def _rect_cells(r: int, c: int, h: int = 2, w: int = 2) -> frozenset[tuple[int, int]]:
    return frozenset((rr, cc) for rr in range(r, r + h) for cc in range(c, c + w))


def _shared_edges(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> int:
    return sum(1 for r, c in a for nb in _neighbors4(r, c) if nb in b)


def _inside(cells: frozenset[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _has_overlap(cells: frozenset[tuple[int, int]], blocks: list[dict]) -> bool:
    return any(cells & block["cells"] for block in blocks)


def _candidate_cells(
    parent: frozenset[tuple[int, int]], direction: tuple[int, int], offset: int
) -> frozenset[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(parent)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return _rect_cells(r0 + offset, c1 + 1)
    if (dr, dc) == (0, -1):
        return _rect_cells(r0 + offset, c0 - 2)
    if (dr, dc) == (1, 0):
        return _rect_cells(r1 + 1, c0 + offset)
    return _rect_cells(r0 - 2, c0 + offset)


def _place_isolated_seed(
    blocks: list[dict], h: int, w: int, color: int
) -> int | None:
    for _ in range(250):
        r = random.randint(1, h - 3)
        c = random.randint(1, w - 3)
        cells = _rect_cells(r, c)
        if _has_overlap(cells, blocks):
            continue
        if any(_shared_edges(cells, block["cells"]) for block in blocks):
            continue
        blocks.append({"cells": cells, "color": color, "role": "seed"})
        return len(blocks) - 1
    return None


def _place_adjacent(
    blocks: list[dict],
    h: int,
    w: int,
    parent_idx: int,
    color: int | None,
    role: str,
    contact: int,
) -> int | None:
    offsets = [0] if contact == 2 else [-1, 1]
    directions = list(DIRS4)
    random.shuffle(directions)
    candidates = []
    parent = blocks[parent_idx]["cells"]
    for direction in directions:
        trial_offsets = offsets[:]
        random.shuffle(trial_offsets)
        for offset in trial_offsets:
            cells = _candidate_cells(parent, direction, offset)
            if not _inside(cells, h, w) or _has_overlap(cells, blocks):
                continue
            counts = [_shared_edges(cells, block["cells"]) for block in blocks]
            if counts[parent_idx] == contact and sum(counts) == contact:
                candidates.append(cells)
    if not candidates:
        return None
    cells = random.choice(candidates)
    blocks.append({"cells": cells, "color": color, "role": role})
    return len(blocks) - 1


def _grow_chain(
    blocks: list[dict], h: int, w: int, seed_idx: int, length: int
) -> list[int] | None:
    first = _place_adjacent(blocks, h, w, seed_idx, None, "gray", 2)
    if first is None:
        return None
    chain = [first]
    parent = first
    for _ in range(length - 1):
        nxt = _place_adjacent(blocks, h, w, parent, None, "gray", 1)
        if nxt is None:
            return None
        chain.append(nxt)
        parent = nxt
    return chain


def _render_input(blocks: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for block in blocks:
        color = GRAY if block["role"] == "gray" else int(block["color"])
        for r, c in block["cells"]:
            grid[r, c] = color
    return grid


def _render_expected(blocks: list[dict], h: int, w: int) -> tuple[np.ndarray, dict, dict, list]:
    adj = _block_adjacency(blocks)
    colors, generations, decisions = _propagate(blocks, adj)
    out = np.zeros((h, w), dtype=int)
    for idx, block in enumerate(blocks):
        color = colors[idx] if block["role"] == "gray" else int(block["color"])
        for r, c in block["cells"]:
            out[r, c] = color
    return out, colors, generations, decisions


def _gen_validates(grid: np.ndarray, blocks: list[dict]) -> bool:
    known_gray = {block["cells"] for block in blocks if block["role"] == "gray"}
    try:
        expected, known_colors, known_generations, known_decisions = _render_expected(
            blocks, *grid.shape
        )
        actual, detected, _colors, _generations, _decisions, _adj = _solve(grid)
    except AssertionError:
        return False

    detected_gray = {block["cells"] for block in detected if block["role"] == "gray"}
    if detected_gray != known_gray:
        return False
    if not np.array_equal(actual, expected):
        return False

    gray_ids = [idx for idx, block in enumerate(blocks) if block["role"] == "gray"]
    gray_generations = [known_generations[idx] for idx in gray_ids]
    gray_colors = {known_colors[idx] for idx in gray_ids}
    if max(gray_generations) < 3:
        return False
    if len(gray_colors) < 2:
        return False
    if not any(len(d["contacts"]) >= 2 for d in known_decisions):
        return False
    by_generation = defaultdict(int)
    for idx in gray_ids:
        by_generation[known_generations[idx]] += 1
    if max(by_generation.values()) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    colors = random.sample(PALETTE, 3)
    blocks: list[dict] = []

    chain_lengths = [random.randint(3, 5), random.randint(3, 5)]
    chains: list[list[int]] = []
    for chain_id, length in enumerate(chain_lengths):
        seed = _place_isolated_seed(blocks, h, w, colors[chain_id])
        if seed is None:
            return None
        chain = _grow_chain(blocks, h, w, seed, length)
        if chain is None:
            return None
        chains.append(chain)

    target_chain = random.choice(chains)
    distractor = _place_adjacent(blocks, h, w, target_chain[0], colors[2], "seed", 1)
    if distractor is None:
        return None

    branch_sources = [idx for chain in chains for idx in chain[1:-1]]
    random.shuffle(branch_sources)
    for source in branch_sources[:2]:
        if random.random() < 0.7:
            _place_adjacent(blocks, h, w, source, None, "gray", 1)

    grid = _render_input(blocks, h, w)
    if _gen_validates(grid, blocks):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cascade-coloring input")
