from __future__ import annotations

from collections import deque
import heapq
import random

import numpy as np


BG = 0
YELLOW = 4
TEAL = 8
CONNECT_COLORS = [1, 2, 3, 5, 6, 7, 9]
MOVE_ORDER = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # horizontal first, then up, then down


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components_for_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        arr = np.array(sorted(coords), dtype=int)
        comps.append(arr)
    comps.sort(key=lambda coords: tuple(map(int, coords.min(axis=0))))
    return comps


def _all_diagonal_rays(grid: np.ndarray) -> set[tuple[int, int]]:
    rays: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid == YELLOW):
        r = int(r)
        c = int(c)
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = r + dr, c + dc
            while _in_bounds(grid.shape, rr, cc):
                if int(grid[rr, cc]) != BG:
                    break
                rays.add((rr, cc))
                rr += dr
                cc += dc
    return rays


def _grid_with_rays(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in _all_diagonal_rays(out):
        out[r, c] = YELLOW
    return out


def _source_colors(grid: np.ndarray) -> list[int]:
    return [
        color
        for color in sorted(set(map(int, grid.ravel())))
        if color not in (BG, YELLOW, TEAL)
    ]


def _order_colors(grid: np.ndarray) -> list[tuple[int, int, list[np.ndarray]]]:
    ordered = []
    totals: list[int] = []
    for color in _source_colors(grid):
        comps = _components_for_color(grid, color)
        assert len(comps) == 2, "each connectable color must have exactly two components"
        total = int(len(comps[0]) + len(comps[1]))
        ordered.append((total, color, comps))
        totals.append(total)
    assert len(totals) == len(set(totals)), "generator must keep total areas distinct"
    ordered.sort(key=lambda item: item[0])
    return ordered


def _allowed_step(canvas: np.ndarray, target: set[tuple[int, int]], r: int, c: int) -> bool:
    return (r, c) in target or int(canvas[r, c]) == BG


def _best_path(
    canvas: np.ndarray,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
) -> tuple[list[tuple[int, int]], int, tuple[int, ...]]:
    target = {tuple(map(int, xy)) for xy in comp_b}
    sources = [tuple(map(int, xy)) for xy in comp_a]
    sources.sort()

    heap: list[tuple[int, tuple[int, ...], int, tuple[int, int]]] = []
    best: dict[tuple[int, int], tuple[int, tuple[int, ...], int]] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}

    for src_rank, (sr, sc) in enumerate(sources):
        for move_rank, (dr, dc) in enumerate(MOVE_ORDER):
            nr, nc = sr + dr, sc + dc
            if not _in_bounds(canvas.shape, nr, nc):
                continue
            if not _allowed_step(canvas, target, nr, nc):
                continue
            key = (move_rank,)
            node = (nr, nc)
            cand = (1, key, src_rank)
            if node not in best or cand < best[node]:
                best[node] = cand
                parent[node] = None
                heapq.heappush(heap, (1, key, src_rank, node))

    assert heap, "no legal first step from source component"

    while heap:
        dist, key, src_rank, node = heapq.heappop(heap)
        if best.get(node) != (dist, key, src_rank):
            continue
        if node in target:
            chain: list[tuple[int, int]] = []
            cur: tuple[int, int] | None = node
            while cur is not None:
                chain.append(cur)
                cur = parent[cur]
            chain.reverse()
            path_cells = [xy for xy in chain if int(canvas[xy]) == BG]
            return path_cells, dist, key

        r, c = node
        for move_rank, (dr, dc) in enumerate(MOVE_ORDER):
            nr, nc = r + dr, c + dc
            if not _in_bounds(canvas.shape, nr, nc):
                continue
            if not _allowed_step(canvas, target, nr, nc):
                continue
            nxt = (nr, nc)
            cand = (dist + 1, key + (move_rank,), src_rank)
            if nxt not in best or cand < best[nxt]:
                best[nxt] = cand
                parent[nxt] = node
                heapq.heappush(heap, (dist + 1, key + (move_rank,), src_rank, nxt))

    raise AssertionError("expected a path between same-colored components")


def _count_shortest_paths(
    canvas: np.ndarray,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    limit: int = 2,
) -> int:
    target = {tuple(map(int, xy)) for xy in comp_b}
    dist: dict[tuple[int, int], int] = {}
    ways: dict[tuple[int, int], int] = {}
    q = deque()

    for sr, sc in [tuple(map(int, xy)) for xy in comp_a]:
        for dr, dc in MOVE_ORDER:
            nr, nc = sr + dr, sc + dc
            node = (nr, nc)
            if not _in_bounds(canvas.shape, nr, nc):
                continue
            if not _allowed_step(canvas, target, nr, nc):
                continue
            if node not in dist:
                dist[node] = 1
                ways[node] = 1
                q.append(node)
            elif dist[node] == 1:
                ways[node] = min(limit, ways[node] + 1)

    best_target_dist: int | None = None
    while q:
        node = q.popleft()
        cur_dist = dist[node]
        if best_target_dist is not None and cur_dist >= best_target_dist:
            continue
        if node in target:
            best_target_dist = cur_dist if best_target_dist is None else min(best_target_dist, cur_dist)
            continue
        r, c = node
        for dr, dc in MOVE_ORDER:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not _in_bounds(canvas.shape, nr, nc):
                continue
            if not _allowed_step(canvas, target, nr, nc):
                continue
            nd = cur_dist + 1
            if nxt not in dist:
                dist[nxt] = nd
                ways[nxt] = ways[node]
                q.append(nxt)
            elif dist[nxt] == nd:
                ways[nxt] = min(limit, ways[nxt] + ways[node])

    total = 0
    for node, node_dist in dist.items():
        if node in target and best_target_dist is not None and node_dist == best_target_dist:
            total = min(limit, total + ways[node])
    return total


def _solve(
    grid: np.ndarray,
    *,
    include_rays: bool = True,
    block_previous_paths: bool = True,
    reverse_order: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    base = _grid_with_rays(grid) if include_rays else np.array(grid, dtype=int, copy=True)
    order = _order_colors(grid)
    if reverse_order:
        order = list(reversed(order))

    out = base.copy()
    traces: list[dict] = []
    for total, color, comps in order:
        canvas = out if block_previous_paths else base
        path_cells, dist, key = _best_path(canvas, comps[0], comps[1])
        shortest_path_count = _count_shortest_paths(canvas, comps[0], comps[1], limit=2)
        for r, c in path_cells:
            assert int(out[r, c]) == BG
            out[r, c] = color
        traces.append(
            {
                "total_area": total,
                "color": color,
                "path_cells": tuple(path_cells),
                "distance": dist,
                "key": key,
                "shortest_path_count": shortest_path_count,
            }
        )
    return out, traces


def _validate_input_structure(grid: np.ndarray) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    yellows = np.argwhere(grid == YELLOW)
    assert 1 <= len(yellows) <= 2
    for r, c in yellows:
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(grid.shape, nr, nc):
                assert int(grid[nr, nc]) == BG
    assert 1 <= int(np.sum(grid == TEAL) > 0)

    colors = _source_colors(grid)
    assert 2 <= len(colors) <= 3
    for color in colors:
        comps = _components_for_color(grid, color)
        assert len(comps) == 2
        for comp in comps:
            assert 2 <= len(comp) <= 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Generate diagonal yellow ray obstacles, then connect each same-colored component pair in area order.

    Input structure:
    - The grid has black background, 2-3 non-black/non-yellow/non-teal colors, and each such color forms exactly two 4-connected components.
    - Yellow pixels are isolated single-cell markers; teal cells form fixed wall segments.
    - The colored components are the only objects to connect, while yellow and teal act only as obstacle sources.

    Rule:
    - First extend every yellow pixel along the four diagonal directions through black cells, stopping before any non-black cell or the grid boundary.
    - Treat all yellow ray cells and teal wall cells as blocked, then sort the connectable colors by the sum of their two component areas from small to large.
    - For each color in that order, draw the shortest 4-neighbor path between its two components using only black cells; previously drawn paths become new obstacles for later colors.
    - If several shortest paths exist, prefer the lexicographically earliest move sequence under horizontal-before-vertical and left-before-right / up-before-down priority.

    Key insight:
    - The yellow singleton markers are not endpoints to connect: they dynamically create diagonal obstacle walls that change the later shortest paths, and the area-based processing order matters because earlier connectors block later ones.
    """
    out, _ = _solve(np.asarray(grid, dtype=int), include_rays=True, block_previous_paths=True)
    return out


def _random_blob(size: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in MOVE_ORDER:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    coords = np.array(sorted(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    coords[:, 0] -= r0
    coords[:, 1] -= c0
    return coords


def _patch_dims(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _can_place_patch(
    grid: np.ndarray,
    coords: np.ndarray,
    top: int,
    left: int,
    *,
    require_clear_8: bool = True,
) -> bool:
    h, w = _patch_dims(coords)
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    occupied = {(top + int(r), left + int(c)) for r, c in coords}
    for r, c in occupied:
        if int(grid[r, c]) != BG:
            return False
        if require_clear_8:
            for nr, nc in _neighbors8(r, c):
                if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in occupied and int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_patch(grid: np.ndarray, coords: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in coords:
        grid[top + int(r), left + int(c)] = color


def _can_place_yellow(grid: np.ndarray, r: int, c: int) -> bool:
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
        return False
    for nr, nc in _neighbors8(r, c):
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
            return False
    return True


def _segment_cells(r: int, c: int, length: int, vertical: bool) -> list[tuple[int, int]]:
    if vertical:
        return [(r + i, c) for i in range(length)]
    return [(r, c + i) for i in range(length)]


def _can_place_segment(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BG, TEAL):
                return False
    return True


def _layout_rows(h: int, n_colors: int, upper_half: bool, center_r: int) -> list[int] | None:
    if upper_half:
        pool = [r for r in range(2, center_r - 1)]
        if len(pool) < n_colors:
            return None
        base = sorted(random.sample(pool, n_colors))
        return base
    pool = [r for r in range(center_r + 2, h - 2)]
    if len(pool) < n_colors:
        return None
    base = sorted(random.sample(pool, n_colors))
    return base


def _place_components_structured(
    grid: np.ndarray,
    colors: list[int],
    totals: list[tuple[int, int]],
    rows: list[int],
) -> bool:
    h, w = grid.shape
    for idx, color in enumerate(colors):
        left_size, right_size = totals[idx]
        placed = False
        for _ in range(80):
            left_blob = _random_blob(left_size)
            right_blob = _random_blob(right_size)
            lh, lw = _patch_dims(left_blob)
            rh, rw = _patch_dims(right_blob)

            left_top = max(1, min(h - lh - 1, rows[idx] - lh // 2 + random.choice([-1, 0, 0, 1])))
            right_top = max(1, min(h - rh - 1, rows[idx] - rh // 2 + random.choice([-1, 0, 0, 1])))
            left_col = random.randint(1, 2)
            right_col = random.randint(max(w - rw - 3, 1), w - rw - 2)

            if not _can_place_patch(grid, left_blob, left_top, left_col):
                continue
            if not _can_place_patch(grid, right_blob, right_top, right_col):
                continue
            _place_patch(grid, left_blob, left_top, left_col, color)
            _place_patch(grid, right_blob, right_top, right_col, color)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_walls(grid: np.ndarray, center_r: int, center_c: int, upper_half: bool) -> bool:
    target = random.choice([1, 2, 2, 3])
    placed = 0
    for _ in range(50):
        if placed >= target:
            break
        vertical = random.random() < 0.6
        length = random.randint(2, 5)
        if upper_half:
            row_lo, row_hi = center_r + 1, grid.shape[0] - length - 2
        else:
            row_lo, row_hi = 1, center_r - length - 1
        if row_lo > row_hi:
            row_lo, row_hi = 1, grid.shape[0] - length - 1
        col_lo = max(1, center_c - 5)
        col_hi = min(grid.shape[1] - 2, center_c + 5)
        if vertical:
            r = random.randint(row_lo, row_hi)
            c = random.randint(col_lo, col_hi)
        else:
            r = random.randint(row_lo, min(grid.shape[0] - 2, row_hi + length - 1))
            c = random.randint(max(1, center_c - 6), min(grid.shape[1] - length - 1, center_c + 2))
        cells = _segment_cells(r, c, length, vertical)
        if _can_place_segment(grid, cells):
            for rr, cc in cells:
                grid[rr, cc] = TEAL
            placed += 1
    return placed >= 1


def _light_semantic_checks(grid: np.ndarray) -> bool:
    try:
        _validate_input_structure(grid)
        normal_out, normal_trace = _solve(grid, include_rays=True, block_previous_paths=True)
        no_rays_out, no_rays_trace = _solve(grid, include_rays=False, block_previous_paths=True)
    except Exception:
        return False

    if np.array_equal(normal_out, grid):
        return False
    if np.array_equal(normal_out, no_rays_out):
        return False
    if not any(item["shortest_path_count"] > 1 for item in normal_trace):
        return False
    if not any(a["path_cells"] != b["path_cells"] for a, b in zip(normal_trace, no_rays_trace)):
        return False
    return True


def _base_template_grid(color_small: int, color_large: int) -> np.ndarray:
    grid = np.zeros((17, 17), dtype=int)
    grid[8, 8] = YELLOW

    for r, c in [(2, 5), (3, 5), (2, 11), (3, 11)]:
        grid[r, c] = color_small
    for r, c in [(3, 1), (3, 2), (3, 3), (3, 12), (4, 11), (4, 12), (5, 12)]:
        grid[r, c] = color_large
    for r, c in [(2, 8), (2, 9)]:
        grid[r, c] = TEAL

    return grid


def _transform_template(grid: np.ndarray, kind: str) -> np.ndarray:
    if kind == "id":
        return np.array(grid, copy=True)
    if kind == "h":
        return np.fliplr(grid)
    if kind == "v":
        return np.flipud(grid)
    if kind == "r":
        return np.rot90(grid, 2)
    raise ValueError(f"unknown transform: {kind}")


def generate_input() -> np.ndarray:
    transforms = ["id", "h", "v", "r"]
    for _ in range(200):
        color_small, color_large = random.sample(CONNECT_COLORS, 2)
        grid = _base_template_grid(color_small, color_large)
        grid = _transform_template(grid, random.choice(transforms))
        if _light_semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
