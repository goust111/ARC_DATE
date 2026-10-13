from __future__ import annotations

from collections import deque
import random

import numpy as np


WALL = 5
EMPTY = 0
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_cavities(grid: np.ndarray) -> list[dict]:
    cavities: list[dict] = []
    for pts in _components(grid != WALL):
        r0, c0, r1, c1 = _bbox(pts)
        assert len(pts) == (r1 - r0 + 1) * (c1 - c0 + 1), "cavities must be rectangular"
        cavities.append({"coords": pts, "bbox": (r0, c0, r1, c1)})
    cavities.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return cavities


def _seed_cavity_index(grid: np.ndarray, cavities: list[dict]) -> int:
    seed_ids = []
    for idx, cavity in enumerate(cavities):
        if any(int(grid[r, c]) not in (EMPTY, WALL) for r, c in cavity["coords"]):
            seed_ids.append(idx)
    assert len(seed_ids) == 1, "expected exactly one seeded cavity"
    return seed_ids[0]


def _seed_tile(grid: np.ndarray, cavity: dict) -> tuple[np.ndarray, tuple[int, int]]:
    colored = [(r, c) for r, c in cavity["coords"] if int(grid[r, c]) not in (EMPTY, WALL)]
    assert colored, "seed cavity must contain a non-black seed"
    sr0, sc0, sr1, sc1 = _bbox(colored)
    tile = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert tile.shape in {(2, 2), (2, 3)}, "seed tile must be 2x2 or 2x3"
    assert not np.any(tile == WALL), "seed tile may not contain walls"
    return tile, (sr0, sc0)


def _paint_pattern(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    tile: np.ndarray,
    anchor: tuple[int, int],
) -> None:
    tile_h, tile_w = tile.shape
    ar, ac = anchor
    for r, c in coords:
        out[r, c] = int(tile[(r - ar) % tile_h, (c - ac) % tile_w])


def _region_is_empty(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == EMPTY for r, c in coords)


def _region_has_pattern(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    return any(int(grid[r, c]) != EMPTY for r, c in coords)


def _thin_connections(grid: np.ndarray, cavities: list[dict]) -> list[dict]:
    conns: list[dict] = []
    n = len(cavities)
    for i in range(n):
        r0a, c0a, r1a, c1a = cavities[i]["bbox"]
        for j in range(i + 1, n):
            r0b, c0b, r1b, c1b = cavities[j]["bbox"]

            # i left of j
            if c1a < c0b:
                gap = c0b - c1a - 1
                lo = max(r0a, r0b)
                hi = min(r1a, r1b)
                if gap == 1 and lo <= hi:
                    wall = [(r, c1a + 1) for r in range(lo, hi + 1)]
                    if all(int(grid[r, c]) == WALL for r, c in wall):
                        conns.append({"a": i, "b": j, "wall": wall, "key": wall[0]})
                continue

            # j left of i
            if c1b < c0a:
                gap = c0a - c1b - 1
                lo = max(r0a, r0b)
                hi = min(r1a, r1b)
                if gap == 1 and lo <= hi:
                    wall = [(r, c1b + 1) for r in range(lo, hi + 1)]
                    if all(int(grid[r, c]) == WALL for r, c in wall):
                        conns.append({"a": i, "b": j, "wall": wall, "key": wall[0]})
                continue

            # i above j
            if r1a < r0b:
                gap = r0b - r1a - 1
                lo = max(c0a, c0b)
                hi = min(c1a, c1b)
                if gap == 1 and lo <= hi:
                    wall = [(r1a + 1, c) for c in range(lo, hi + 1)]
                    if all(int(grid[r, c]) == WALL for r, c in wall):
                        conns.append({"a": i, "b": j, "wall": wall, "key": wall[0]})
                continue

            # j above i
            if r1b < r0a:
                gap = r0a - r1b - 1
                lo = max(c0a, c0b)
                hi = min(c1a, c1b)
                if gap == 1 and lo <= hi:
                    wall = [(r1b + 1, c) for c in range(lo, hi + 1)]
                    if all(int(grid[r, c]) == WALL for r, c in wall):
                        conns.append({"a": i, "b": j, "wall": wall, "key": wall[0]})

    conns.sort(key=lambda item: item["key"])
    return conns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the seeded cavity, then let that tiling propagate through one-cell walls into empty rectangular cavities.

    Input structure:
    - Grey 5 walls and a 2-cell outer frame partition the interior into several rectangular black cavities.
    - Exactly one cavity contains a small 2x2 or 2x3 multicolor seed block; all other cavities are pure black.
    - Some shared cavity boundaries are 1 cell thick and others are at least 2 cells thick.

    Rule:
    - Find the unique seeded cavity and use the seed block's bbox as a periodic tile.
    - First tile the entire seeded cavity, preserving the phase given by the seed's actual placement.
    - Then repeatedly find 1-cell walls with a tiled cavity on one side and an all-black cavity on the other, erase that wall, and extend the same global tiling through the newly opened space.
    - Process simultaneously eligible thin walls in top-to-bottom, left-to-right order, and never erase thick walls or thin walls whose two sides are both empty.

    Key insight:
    - The seed is not copied cavity-by-cavity; one global periodic phase keeps flowing forward, and only asymmetric thin-wall frontiers allow that flow to continue.
    """
    g = np.asarray(grid, dtype=int)
    cavities = _extract_cavities(g)
    seed_idx = _seed_cavity_index(g, cavities)
    tile, anchor = _seed_tile(g, cavities[seed_idx])

    out = g.copy()
    _paint_pattern(out, cavities[seed_idx]["coords"], tile, anchor)

    conns = _thin_connections(g, cavities)
    while True:
        round_ops: list[tuple[dict, int]] = []
        for conn in conns:
            a, b = conn["a"], conn["b"]
            a_filled = _region_has_pattern(out, cavities[a]["coords"])
            b_filled = _region_has_pattern(out, cavities[b]["coords"])
            a_empty = _region_is_empty(out, cavities[a]["coords"])
            b_empty = _region_is_empty(out, cavities[b]["coords"])
            if a_filled and b_empty:
                round_ops.append((conn, b))
            elif b_filled and a_empty:
                round_ops.append((conn, a))

        if not round_ops:
            break

        round_ops.sort(key=lambda item: item[0]["key"])
        for conn, dst in round_ops:
            for r, c in conn["wall"]:
                out[r, c] = EMPTY
            _paint_pattern(out, conn["wall"] + cavities[dst]["coords"], tile, anchor)

    return out


def _template_choices(nrows: int, ncols: int) -> list[tuple[str, list[bool], list[bool], int]]:
    if (nrows, ncols) == (2, 3):
        return [
            ("line", [False], [True, True], 4),
            ("block_left", [True], [True, False], 3),
            ("block_right", [True], [False, True], 3),
            ("full", [True], [True, True], 1),
        ]
    if (nrows, ncols) == (3, 2):
        return [
            ("line", [True, True], [False], 4),
            ("block_top", [True, False], [True], 3),
            ("block_bottom", [False, True], [True], 3),
            ("full", [True, True], [True], 1),
        ]
    raise ValueError((nrows, ncols))


def _build_rects(
    row_heights: list[int],
    col_widths: list[int],
    row_thins: list[bool],
    col_thins: list[bool],
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    row_starts: list[int] = []
    col_starts: list[int] = []

    r = 2
    for i, h in enumerate(row_heights):
        row_starts.append(r)
        r += h
        if i < len(row_thins):
            r += 1 if row_thins[i] else 2
    total_h = r + 2

    c = 2
    for j, w in enumerate(col_widths):
        col_starts.append(c)
        c += w
        if j < len(col_thins):
            c += 1 if col_thins[j] else 2
    total_w = c + 2

    rects: list[tuple[int, int, int, int]] = []
    for i, h in enumerate(row_heights):
        for j, w in enumerate(col_widths):
            r0 = row_starts[i]
            c0 = col_starts[j]
            rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return rects, (total_h, total_w)


def _thin_graph(
    nrows: int,
    ncols: int,
    row_thins: list[bool],
    col_thins: list[bool],
) -> dict[int, set[int]]:
    def idx(i: int, j: int) -> int:
        return i * ncols + j

    graph = {k: set() for k in range(nrows * ncols)}
    for i in range(nrows):
        for j in range(ncols):
            here = idx(i, j)
            if i + 1 < nrows and row_thins[i]:
                down = idx(i + 1, j)
                graph[here].add(down)
                graph[down].add(here)
            if j + 1 < ncols and col_thins[j]:
                right = idx(i, j + 1)
                graph[here].add(right)
                graph[right].add(here)
    return graph


def _bfs_component(graph: dict[int, set[int]], start: int) -> tuple[set[int], dict[int, int]]:
    q = deque([start])
    dist = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return set(dist), dist


def _choose_sizes(
    nrows: int,
    ncols: int,
    row_thins: list[bool],
    col_thins: list[bool],
) -> tuple[list[int], list[int]]:
    for _ in range(200):
        row_heights = [random.randint(3, 6) for _ in range(nrows)]
        col_widths = [random.randint(3, 8) for _ in range(ncols)]
        _, (h, w) = _build_rects(row_heights, col_widths, row_thins, col_thins)
        if 15 <= h <= 24 and 15 <= w <= 24 and (len(set(row_heights)) > 1 or len(set(col_widths)) > 1):
            return row_heights, col_widths
    raise AssertionError("failed to choose cavity sizes")


def _make_nonperiodic_tile() -> np.ndarray:
    shape = random.choice([(2, 2), (2, 3)])
    colors = random.sample(SEED_COLORS, 2)
    for _ in range(200):
        tile = np.array(
            [[random.choice(colors) for _ in range(shape[1])] for _ in range(shape[0])],
            dtype=int,
        )
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        periodic = False
        for dr in range(shape[0]):
            for dc in range(shape[1]):
                if dr == 0 and dc == 0:
                    continue
                if np.array_equal(tile, np.roll(np.roll(tile, dr, axis=0), dc, axis=1)):
                    periodic = True
                    break
            if periodic:
                break
        if not periodic:
            return tile
    raise AssertionError("failed to choose a non-periodic tile")


def _seed_anchor(rect: tuple[int, int, int, int], tile_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    th, tw = tile_shape
    corners = [
        (r0, c0),
        (r0, c1 - tw + 1),
        (r1 - th + 1, c0),
        (r1 - th + 1, c1 - tw + 1),
    ]
    return random.choice(corners)


def _render_input(
    rects: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    tile: np.ndarray,
    seed_idx: int,
    anchor: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, WALL, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = EMPTY
    ar, ac = anchor
    th, tw = tile.shape
    grid[ar : ar + th, ac : ac + tw] = tile
    return grid


def _adjacency_info(
    rects: list[tuple[int, int, int, int]],
    nrows: int,
    ncols: int,
) -> list[dict]:
    info: list[dict] = []
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            r0, c0, r1, c1 = rects[idx]
            if j + 1 < ncols:
                idx2 = i * ncols + (j + 1)
                r0b, c0b, r1b, c1b = rects[idx2]
                gap = c0b - c1 - 1
                wall = [(r, c) for c in range(c1 + 1, c0b) for r in range(max(r0, r0b), min(r1, r1b) + 1)]
                info.append({"a": idx, "b": idx2, "gap": gap, "wall": wall})
            if i + 1 < nrows:
                idx2 = (i + 1) * ncols + j
                r0b, c0b, r1b, c1b = rects[idx2]
                gap = r0b - r1 - 1
                wall = [(r, c) for r in range(r1 + 1, r0b) for c in range(max(c0, c0b), min(c1, c1b) + 1)]
                info.append({"a": idx, "b": idx2, "gap": gap, "wall": wall})
    return info


def _global_vs_local_diff(
    cavities: list[dict],
    reachable: set[int],
    seed_idx: int,
    tile: np.ndarray,
    anchor: tuple[int, int],
) -> bool:
    th, tw = tile.shape
    ar, ac = anchor
    for idx in sorted(reachable):
        if idx == seed_idx:
            continue
        r0, c0, _r1, _c1 = cavities[idx]["bbox"]
        for r, c in cavities[idx]["coords"]:
            global_value = int(tile[(r - ar) % th, (c - ac) % tw])
            local_value = int(tile[(r - r0) % th, (c - c0) % tw])
            if global_value != local_value:
                return True
    return False


def _all_cavities_filled(grid: np.ndarray, cavities: list[dict], tile: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    for cavity in cavities:
        _paint_pattern(out, cavity["coords"], tile, anchor)
    for conn in _thin_connections(grid, cavities):
        _paint_pattern(out, conn["wall"], tile, anchor)
    return out


def _seed_only_filled(grid: np.ndarray, cavities: list[dict], seed_idx: int, tile: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    _paint_pattern(out, cavities[seed_idx]["coords"], tile, anchor)
    return out


def _generate_one() -> np.ndarray:
    nrows, ncols = random.choice([(2, 3), (3, 2)])
    templates = _template_choices(nrows, ncols)
    names = [item[0] for item in templates]
    weights = [item[3] for item in templates]
    mode = random.choices(names, weights=weights, k=1)[0]
    template = next(item for item in templates if item[0] == mode)
    row_thins, col_thins = list(template[1]), list(template[2])

    row_heights, col_widths = _choose_sizes(nrows, ncols, row_thins, col_thins)
    rects, shape = _build_rects(row_heights, col_widths, row_thins, col_thins)
    graph = _thin_graph(nrows, ncols, row_thins, col_thins)

    seed_candidates = []
    for idx in range(nrows * ncols):
        component, dist = _bfs_component(graph, idx)
        if max(dist.values(), default=0) >= 2:
            if mode == "full" and len(component) == nrows * ncols:
                seed_candidates.append((idx, component))
            if mode != "full" and 3 <= len(component) < nrows * ncols:
                seed_candidates.append((idx, component))
    assert seed_candidates, "template must admit a 2-step propagation seed"

    seed_idx, reachable = random.choice(seed_candidates)
    tile = _make_nonperiodic_tile()
    anchor = _seed_anchor(rects[seed_idx], tile.shape)
    grid = _render_input(rects, shape, tile, seed_idx, anchor)

    cavities = _extract_cavities(grid)
    assert len(cavities) == nrows * ncols
    assert _seed_cavity_index(grid, cavities) == seed_idx

    out = main(grid)
    adjacency = _adjacency_info(rects, nrows, ncols)

    thin_edges_outside = 0
    thick_edges = 0
    for edge in adjacency:
        if edge["gap"] == 1:
            if edge["a"] in reachable and edge["b"] in reachable:
                assert all(int(out[r, c]) != WALL for r, c in edge["wall"]), "reachable thin wall must be removed"
            else:
                thin_edges_outside += 1
                assert all(int(out[r, c]) == WALL for r, c in edge["wall"]), "empty-empty thin wall must remain"
        else:
            thick_edges += 1
            assert all(int(out[r, c]) == WALL for r, c in edge["wall"]), "thick wall must remain"

    assert thick_edges >= 1
    if mode != "full":
        assert thin_edges_outside >= 1

    for idx, cavity in enumerate(cavities):
        cells = [int(out[r, c]) for r, c in cavity["coords"]]
        if idx in reachable:
            assert any(v != EMPTY for v in cells), "reachable cavity must be filled"
        else:
            assert all(v == EMPTY for v in cells), "unreachable cavity must stay empty"

    tile2, anchor2 = _seed_tile(grid, cavities[seed_idx])
    assert np.array_equal(tile, tile2) and anchor == anchor2
    assert _global_vs_local_diff(cavities, reachable, seed_idx, tile, anchor), "phase continuity must matter"
    assert not np.array_equal(out, _seed_only_filled(grid, cavities, seed_idx, tile, anchor))
    if mode != "full":
        assert not np.array_equal(out, _all_cavities_filled(grid, cavities, tile, anchor))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            return _generate_one()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid thin-wall propagation example")
