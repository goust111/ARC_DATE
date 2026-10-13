from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
SOURCE_PALETTE = (3, 4, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
MIN_PATH_NODES = 3
MAX_PATH_NODES = 5


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = map(int, mask.shape)
    region_ids = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(region_ids[r, c]) != -1:
                continue

            rid = len(regions)
            q = deque([(r, c)])
            region_ids[r, c] = rid
            cells: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not bool(mask[nr, nc]) or int(region_ids[nr, nc]) != -1:
                        continue
                    region_ids[nr, nc] = rid
                    q.append((nr, nc))

            regions.append(cells)

    return region_ids, regions


def _scan_across_blue(
    grid: np.ndarray,
    region_ids: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    h, w = map(int, grid.shape)
    nr = r + dr
    nc = c + dc
    if not (0 <= nr < h and 0 <= nc < w) or int(grid[nr, nc]) != BLUE:
        return None

    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLUE:
        nr += dr
        nc += dc

    if not (0 <= nr < h and 0 <= nc < w):
        return None
    target = int(region_ids[nr, nc])
    return target if target != -1 else None


def _region_adjacency(
    grid: np.ndarray,
    region_ids: np.ndarray,
    regions: list[list[tuple[int, int]]],
) -> list[set[int]]:
    adjacency = [set() for _ in regions]
    for rid, cells in enumerate(regions):
        for r, c in cells:
            for dr, dc in DIR4:
                target = _scan_across_blue(grid, region_ids, r, c, dr, dc)
                if target is not None and target != rid:
                    adjacency[rid].add(target)
    return adjacency


def _cell_targets(
    grid: np.ndarray,
    region_ids: np.ndarray,
    rid: int,
    r: int,
    c: int,
) -> list[int]:
    targets: list[int] = []
    for dr, dc in DIR4:
        target = _scan_across_blue(grid, region_ids, r, c, dr, dc)
        if target is not None and target != rid:
            targets.append(target)
    return targets


def _analyze_regions(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    region_ids, regions = _label_regions(g != BLUE)
    adjacency = _region_adjacency(g, region_ids, regions)
    infos: list[dict[str, object]] = []

    for rid, cells in enumerate(regions):
        vals = [int(g[r, c]) for r, c in cells]
        red_cells = [(r, c) for r, c in cells if int(g[r, c]) == RED]
        source_colors = {v for v in vals if v not in (BLACK, BLUE, RED)}

        info: dict[str, object] = {"cells": cells}
        if source_colors:
            assert len(source_colors) == 1, "source region must use exactly one fill color"
            assert not red_cells, "source region cannot contain a pointer"
            info["type"] = "source"
            info["color"] = int(next(iter(source_colors)))
        elif red_cells:
            assert len(red_cells) == 1, "pointer region must contain exactly one red pixel"
            assert all(v in (BLACK, RED) for v in vals), "pointer region must otherwise be black"
            pr, pc = red_cells[0]
            targets = _cell_targets(g, region_ids, rid, pr, pc)
            assert len(targets) == 1, "pointer pixel must indicate exactly one adjacent region"
            info["type"] = "pointer"
            info["pointer"] = (pr, pc)
            info["target"] = int(targets[0])
        else:
            assert all(v == BLACK for v in vals), "terminal region must be empty black"
            info["type"] = "terminal"

        infos.append(info)

    return region_ids, regions, infos, adjacency


def _paint_region(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color through neighboring pointer regions until a blank terminal region.

    Input structure:
    - The background is black, and one-pixel-thick blue walls partition the grid into closed regions.
    - A region is either a solid non-black non-blue source fill, a black region with one red pointer pixel, or a completely black terminal region.
    - A pointer pixel sits next to exactly one wall side, so the wall side identifies which adjacent region it points across to.

    Rule:
    - Split the non-blue cells into enclosed regions and identify every source region and its color.
    - For each source, inspect its adjacent regions. Any neighboring pointer region whose pointer does not point back into the source becomes a chain start.
    - Once color enters a pointer region, repaint that whole region and keep following the pointer-indicated neighboring region until reaching a blank terminal region, which is also repainted.
    - Source regions stay unchanged, blue walls stay blue, and every visited red pointer pixel is overwritten by the propagated source color.

    Key insight:
    - The red pixel does not name a color directly; its position relative to the enclosing wall encodes a directed edge in the region-adjacency graph, and color flows by following those edges from each source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _region_ids, regions, infos, adjacency = _analyze_regions(g)
    out = g.copy()
    claimed: dict[int, int] = {}

    for src, info in enumerate(infos):
        if info["type"] != "source":
            continue

        color = int(info["color"])
        starts = [
            nbr
            for nbr in sorted(adjacency[src])
            if infos[nbr]["type"] == "pointer" and int(infos[nbr]["target"]) != src
        ]
        assert starts, "every source must have at least one outgoing pointer chain"

        for start in starts:
            cur = start
            seen_chain: set[int] = set()

            while True:
                assert cur not in seen_chain, "pointer chains must terminate instead of looping"
                seen_chain.add(cur)

                existing = claimed.get(cur)
                if existing is not None:
                    assert existing == color, "different sources cannot claim the same region"
                    break
                claimed[cur] = color

                _paint_region(out, regions[cur], color)
                cur_type = str(infos[cur]["type"])
                if cur_type == "terminal":
                    break

                assert cur_type == "pointer", "only pointer regions may continue a chain"
                nxt = int(infos[cur]["target"])
                assert nxt != src, "chain-start pointers must point away from the source"
                assert str(infos[nxt]["type"]) != "source", "chains must terminate in non-source regions"
                cur = nxt

    return out


def _grow_owner_partition(mh: int, mw: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    owner = np.full((mh, mw), -1, dtype=int)
    all_cells = [(r, c) for r in range(mh) for c in range(mw)]
    seeds = rng.sample(all_cells, n_regions)

    frontier = [set() for _ in range(n_regions)]
    sizes = [1] * n_regions
    for rid, (r, c) in enumerate(seeds):
        owner[r, c] = rid
        frontier[rid].add((r, c))

    remaining = mh * mw - n_regions
    while remaining > 0:
        choices: list[tuple[int, list[tuple[int, int]]]] = []
        weights: list[float] = []
        for rid in range(n_regions):
            boundary = set()
            for r, c in frontier[rid]:
                for nr, nc in _neighbors4(r, c, mh, mw):
                    if int(owner[nr, nc]) == -1:
                        boundary.add((nr, nc))
            if boundary:
                choices.append((rid, list(boundary)))
                weights.append(1.0 / sizes[rid])

        if not choices:
            return None

        pick = rng.choices(range(len(choices)), weights=weights, k=1)[0]
        rid, boundary = choices[pick]
        nr, nc = rng.choice(boundary)
        owner[nr, nc] = rid
        frontier[rid].add((nr, nc))
        sizes[rid] += 1
        remaining -= 1

    return owner


def _expand_owner(owner: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    mh, mw = map(int, owner.shape)
    h = mh * (cell_h + 1) + 1
    w = mw * (cell_w + 1) + 1
    grid = np.full((h, w), BLUE, dtype=int)

    for mr in range(mh):
        for mc in range(mw):
            r0 = 1 + mr * (cell_h + 1)
            c0 = 1 + mc * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = BLACK

    for mr in range(mh - 1):
        for mc in range(mw):
            if int(owner[mr, mc]) != int(owner[mr + 1, mc]):
                continue
            wall_r = (mr + 1) * (cell_h + 1)
            c0 = 1 + mc * (cell_w + 1)
            grid[wall_r, c0 : c0 + cell_w] = BLACK

    for mr in range(mh):
        for mc in range(mw - 1):
            if int(owner[mr, mc]) != int(owner[mr, mc + 1]):
                continue
            wall_c = (mc + 1) * (cell_w + 1)
            r0 = 1 + mr * (cell_h + 1)
            grid[r0 : r0 + cell_h, wall_c] = BLACK

    return grid


def _pointer_candidates_for_edge(
    grid: np.ndarray,
    region_ids: np.ndarray,
    cells: list[tuple[int, int]],
    rid: int,
    target: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in cells:
        targets = _cell_targets(grid, region_ids, rid, r, c)
        if len(targets) == 1 and int(targets[0]) == target:
            candidates.append((r, c))
    return candidates


def _all_candidate_paths(
    adjacency: list[set[int]],
    edge_candidates: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    n_regions = len(adjacency)

    for source in range(n_regions):
        def dfs(path: list[int]) -> None:
            cur = path[-1]
            if MIN_PATH_NODES <= len(path) <= MAX_PATH_NODES:
                ok = True
                for i in range(1, len(path) - 1):
                    if not edge_candidates.get((path[i], path[i + 1])):
                        ok = False
                        break
                if ok:
                    paths.append(tuple(path))

            if len(path) == MAX_PATH_NODES:
                return

            for nxt in sorted(adjacency[cur]):
                if nxt in path:
                    continue
                path.append(nxt)
                dfs(path)
                path.pop()

        dfs([source])

    return paths


def _source_pointer_conflict(adjacency: list[set[int]], paths: list[tuple[int, ...]]) -> bool:
    pointer_nodes = {node for path in paths for node in path[1:-1]}
    for path in paths:
        source = path[0]
        allowed = {path[1]}
        if (adjacency[source] & pointer_nodes) - allowed:
            return True
    return False


def _choose_paths(adjacency: list[set[int]], paths: list[tuple[int, ...]], rng: random.Random) -> list[tuple[int, ...]] | None:
    shuffled = paths[:]
    rng.shuffle(shuffled)
    preferred = 2 if rng.random() < 0.5 else 1

    def search(target_count: int) -> list[tuple[int, ...]] | None:
        def backtrack(start_idx: int, chosen: list[tuple[int, ...]], used: set[int]) -> list[tuple[int, ...]] | None:
            if len(chosen) == target_count:
                return chosen[:] if not _source_pointer_conflict(adjacency, chosen) else None

            for idx in range(start_idx, len(shuffled)):
                path = shuffled[idx]
                nodes = set(path)
                if nodes & used:
                    continue
                chosen.append(path)
                result = backtrack(idx + 1, chosen, used | nodes)
                if result is not None:
                    return result
                chosen.pop()
            return None

        return backtrack(0, [], set())

    for target_count in ([preferred, 1] if preferred == 2 else [1]):
        result = search(target_count)
        if result is not None:
            return result
    return None


def _validate_generated(grid: np.ndarray, paths: list[tuple[int, ...]], colors: list[int]) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    _region_ids, regions, infos, adjacency = _analyze_regions(g)
    assert len(paths) == len(colors) >= 1

    expected_fill: dict[int, int] = {}
    for path, color in zip(paths, colors):
        source = path[0]
        assert infos[source]["type"] == "source"
        assert int(infos[source]["color"]) == color
        starts = [
            nbr
            for nbr in sorted(adjacency[source])
            if infos[nbr]["type"] == "pointer" and int(infos[nbr]["target"]) != source
        ]
        assert starts == [path[1]], "each source should activate exactly one chain start"
        for rid in path[1:]:
            expected_fill[rid] = color

    out = main(g)
    assert out.shape == g.shape
    assert np.all(out[g == BLUE] == BLUE)
    assert np.any(out != g)

    for rid, cells in enumerate(regions):
        if rid in expected_fill:
            expected = expected_fill[rid]
            assert all(int(out[r, c]) == expected for r, c in cells)
        elif infos[rid]["type"] == "source":
            expected = int(infos[rid]["color"])
            assert all(int(out[r, c]) == expected for r, c in cells)
        else:
            assert all(int(out[r, c]) == int(g[r, c]) for r, c in cells)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.choice((3, 4))
    cell_w = rng.choice((3, 4))
    max_mh = max(3, 29 // (cell_h + 1))
    max_mw = max(3, 29 // (cell_w + 1))
    mh = rng.randint(3, min(7, max_mh))
    mw = rng.randint(3, min(7, max_mw))

    min_regions = max(4, min(6, (mh * mw) // 3))
    max_regions = min(10, mh * mw)
    if min_regions > max_regions:
        min_regions = max_regions
    n_regions = rng.randint(min_regions, max_regions)

    owner = _grow_owner_partition(mh, mw, n_regions, rng)
    if owner is None:
        return None

    base = _expand_owner(owner, cell_h, cell_w)
    region_ids, regions = _label_regions(base != BLUE)
    adjacency = _region_adjacency(base, region_ids, regions)
    if len(regions) < 3:
        return None

    edge_candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rid, nbrs in enumerate(adjacency):
        for target in nbrs:
            candidates = _pointer_candidates_for_edge(base, region_ids, regions[rid], rid, target)
            if candidates:
                edge_candidates[(rid, target)] = candidates

    paths = _all_candidate_paths(adjacency, edge_candidates)
    if not paths:
        return None

    selected = _choose_paths(adjacency, paths, rng)
    if selected is None:
        return None

    colors = rng.sample(SOURCE_PALETTE, len(selected))
    grid = base.copy()

    for path, color in zip(selected, colors):
        source = path[0]
        for r, c in regions[source]:
            grid[r, c] = color

        for i in range(1, len(path) - 1):
            rid = path[i]
            target = path[i + 1]
            options = edge_candidates.get((rid, target), [])
            if not options:
                return None
            pr, pc = rng.choice(options)
            grid[pr, pc] = RED

    _validate_generated(grid, selected, colors)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain region puzzle")
