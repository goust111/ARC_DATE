from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            labels[r, c] = idx
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = idx
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return labels, comps


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _palette_from_bottom_row(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    palette = [int(row[c]) for c in range(grid.shape[1]) if int(row[c]) != bg]
    assert palette, "palette row must contain at least one non-background color"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette


def _cycle_components(area: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = area == int(color)
    _labels, comps = _label_components(mask)
    h, w = area.shape
    cycles: list[list[tuple[int, int]]] = []
    for pts in comps:
        if len(pts) < 4:
            continue
        ok = True
        for r, c in pts:
            degree = 0
            for nr, nc in _neighbors4(r, c, h, w):
                if bool(mask[nr, nc]):
                    degree += 1
            if degree != 2:
                ok = False
                break
        if ok:
            cycles.append(pts)
    return cycles


def _find_contour_color(area: np.ndarray, bg: int) -> tuple[int, list[list[tuple[int, int]]]]:
    best_color = None
    best_cycles: list[list[tuple[int, int]]] = []
    best_score = (-1, -1)
    for color in sorted(set(map(int, area.flatten())) - {int(bg)}):
        cycles = _cycle_components(area, int(color))
        score = (sum(len(pts) for pts in cycles), len(cycles))
        if score > best_score:
            best_score = score
            best_color = int(color)
            best_cycles = cycles
    assert best_color is not None and best_score[0] > 0, "failed to detect contour color"
    return int(best_color), best_cycles


def _region_graph(area: np.ndarray, bg: int, contour: int):
    open_mask = area != int(contour)
    region_labels, regions = _label_components(open_mask)
    _loop_labels, loops = _label_components(area == int(contour))
    graph: list[list[tuple[int, int]]] = [[] for _ in regions]
    loop_neighbors: list[tuple[int, int]] = []

    for loop_id, pts in enumerate(loops):
        adjacent = set()
        for r, c in pts:
            for nr, nc in _neighbors4(r, c, area.shape[0], area.shape[1]):
                if bool(open_mask[nr, nc]):
                    adjacent.add(int(region_labels[nr, nc]))
        assert len(adjacent) == 2, "each contour loop should separate exactly two regions"
        a, b = sorted(adjacent)
        graph[a].append((int(b), int(loop_id)))
        graph[b].append((int(a), int(loop_id)))
        loop_neighbors.append((int(a), int(b)))

    region_markers: list[set[int]] = []
    border_regions: list[int] = []
    h, w = area.shape
    for region_id, pts in enumerate(regions):
        colors = {int(area[r, c]) for r, c in pts if int(area[r, c]) not in (int(bg), int(contour))}
        region_markers.append(colors)
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            border_regions.append(int(region_id))

    return regions, loops, graph, loop_neighbors, region_markers, border_regions


def _depth_tree(
    graph: list[list[tuple[int, int]]],
    border_regions: list[int],
    region_markers: list[set[int]],
) -> tuple[list[int], list[int], list[int], list[set[int]]]:
    n = len(graph)
    depth = [-1] * n
    parent = [-1] * n
    parent_loop = [-1] * n
    q: deque[int] = deque()

    for region_id in border_regions:
        if depth[region_id] == -1:
            depth[region_id] = 0
            q.append(int(region_id))

    while q:
        region_id = q.popleft()
        for nxt, loop_id in graph[region_id]:
            if depth[nxt] != -1:
                continue
            depth[nxt] = int(depth[region_id] + 1)
            parent[nxt] = int(region_id)
            parent_loop[nxt] = int(loop_id)
            q.append(int(nxt))

    children: list[list[int]] = [[] for _ in range(n)]
    for region_id in range(n):
        if parent[region_id] != -1:
            children[parent[region_id]].append(int(region_id))

    subtree_markers: list[set[int]] = [set() for _ in range(n)]
    order = sorted(range(n), key=lambda idx: int(depth[idx]), reverse=True)
    for region_id in order:
        markers = set(region_markers[region_id])
        for child in children[region_id]:
            markers.update(subtree_markers[child])
        subtree_markers[region_id] = markers

    return depth, parent, parent_loop, subtree_markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color nested contour regions by depth and recolor each mountain's contours by its summit marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3

    bg = _get_bg(g)
    palette = _palette_from_bottom_row(g, bg)
    area = g[:-1, :]

    contour, _cycles = _find_contour_color(area, bg)
    regions, loops, graph, _loop_neighbors, region_markers, border_regions = _region_graph(area, bg, contour)
    depth, _parent, parent_loop, subtree_markers = _depth_tree(graph, border_regions, region_markers)

    assert max(depth) <= len(palette), "palette must cover the deepest nesting level"

    out = g.copy()

    for region_id, pts in enumerate(regions):
        if depth[region_id] <= 0:
            continue
        fill = int(palette[int(depth[region_id] - 1)])
        for r, c in pts:
            if int(area[r, c]) == int(bg):
                out[r, c] = fill

    loop_colors: dict[int, int] = {}
    for region_id, loop_id in enumerate(parent_loop):
        if loop_id == -1:
            continue
        markers = subtree_markers[region_id]
        if not markers:
            continue
        loop_colors[int(loop_id)] = int(sorted(markers)[0])

    for loop_id, pts in enumerate(loops):
        if loop_id not in loop_colors:
            continue
        col = int(loop_colors[loop_id])
        for r, c in pts:
            out[r, c] = col

    return out


def _draw_rect_loop(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _mountain_spec(depth: int, rng: random.Random) -> dict[str, int]:
    inner_h = rng.randint(1, 3)
    inner_w = rng.randint(1, 3)
    outer_h = int(inner_h + 2 + 4 * (depth - 1))
    outer_w = int(inner_w + 2 + 4 * (depth - 1))
    return {
        "depth": int(depth),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
    }


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(occupied.shape[0], int(r0 + h + pad))
    cc1 = min(occupied.shape[1], int(c0 + w + pad))
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(occupied.shape[0], int(r0 + h + pad))
    cc1 = min(occupied.shape[1], int(c0 + w + pad))
    occupied[rr0:rr1, cc0:cc1] = True


def _place_mountains(
    area_h: int,
    area_w: int,
    mountains: list[dict[str, int]],
    rng: random.Random,
) -> list[dict[str, int]] | None:
    occupied = np.zeros((area_h, area_w), dtype=bool)
    placed: list[dict[str, int]] = []

    order = sorted(range(len(mountains)), key=lambda idx: mountains[idx]["outer_h"] * mountains[idx]["outer_w"], reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        spec = mountains[idx]
        h = int(spec["outer_h"])
        w = int(spec["outer_w"])
        if h + 2 > area_h or w + 2 > area_w:
            return None
        placed_here = False
        for _ in range(250):
            r0 = rng.randint(1, area_h - h - 1)
            c0 = rng.randint(1, area_w - w - 1)
            if not _bbox_is_free(occupied, r0, c0, h, w, pad=1):
                continue
            _mark_bbox(occupied, r0, c0, h, w, pad=1)
            placements[idx] = (int(r0), int(c0))
            placed_here = True
            break
        if not placed_here:
            return None

    for idx, spec in enumerate(mountains):
        r0, c0 = placements[idx]
        placed.append({**spec, "r0": int(r0), "c0": int(c0)})

    return placed


def _draw_mountain(grid: np.ndarray, spec: dict[str, int], contour: int, marker: int) -> None:
    depth = int(spec["depth"])
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])

    for level in range(depth):
        inset = int(2 * level)
        _draw_rect_loop(
            grid,
            int(r0 + inset),
            int(c0 + inset),
            int(outer_h - 4 * level),
            int(outer_w - 4 * level),
            int(contour),
        )

    core_r0 = int(r0 + 2 * (depth - 1) + 1)
    core_c0 = int(c0 + 2 * (depth - 1) + 1)
    core_h = int(spec["inner_h"])
    core_w = int(spec["inner_w"])
    mr = int(core_r0 + core_h // 2)
    mc = int(core_c0 + core_w // 2)
    grid[mr, mc] = int(marker)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.randrange(10))
    colors = [c for c in range(10) if c != bg]
    contour = int(rng.choice(colors))
    colors.remove(contour)

    n_groups = int(rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0])
    max_depth = int(rng.randint(1, min(4, len(colors) - n_groups)))
    depths = [int(rng.randint(1, max_depth)) for _ in range(n_groups)]
    depths[rng.randrange(n_groups)] = int(max_depth)

    palette = rng.sample(colors, max_depth)
    for color in palette:
        colors.remove(color)
    markers = rng.sample(colors, n_groups)

    mountains = [_mountain_spec(depth, rng) for depth in depths]
    min_area_h = max(spec["outer_h"] for spec in mountains) + 2
    min_area_w = max(spec["outer_w"] for spec in mountains) + 2
    load = int(sum((spec["outer_h"] + 2) * (spec["outer_w"] + 2) for spec in mountains) ** 0.5)
    area_h_lo = max(12, min_area_h, load)
    area_w_lo = max(14, min_area_w, load + 2)
    area_h = int(rng.randint(area_h_lo, area_h_lo + 6))
    area_w = int(rng.randint(area_w_lo, area_w_lo + 8))

    placed = _place_mountains(area_h, area_w, mountains, rng)
    if placed is None:
        return None

    grid = np.full((area_h + 1, area_w), int(bg), dtype=int)
    palette_start = int(rng.randint(0, area_w - len(palette)))
    for offset, color in enumerate(palette):
        grid[area_h, palette_start + offset] = int(color)

    for spec, marker in zip(placed, markers):
        _draw_mountain(grid, spec, contour, int(marker))

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None

    changed = int(np.count_nonzero(grid != out))
    if changed < max_depth + n_groups:
        return None

    return grid


def generate_input() -> np.ndarray:
    """
    Generate layered contour maps with a bottom palette and distinct summit markers.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
