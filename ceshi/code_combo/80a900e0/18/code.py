# Combo variant of 80a900e0 (index 18)
# combo_spec: D1/1+D17/1
# Name: 反射交叉标记

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8

# Design analysis for the fused variant:
# - Concrete behavior: in the original task, an endpoint ray stops as soon as it reaches a non-bg
#   pixel. Here the same ray treats the input objects as static mirrors: a diagonal hit flips both
#   axes, and a side-touch after a move flips the touched axis. The ray keeps going until it exits
#   or repeats a state. If a red path and a green path later visit the same bg cell, that cell is
#   recolored to the dedicated marker color 9 instead of staying red/green.
# - Complexity calibration: the background grammar, 8-pixel diamonds, and short diagonal outer
#   segments stay unchanged so the object vocabulary is still recognizable. The new freedom is
#   limited to one extra interaction color and deterministic reflection rules, while the generator
#   requires 2-3 diamonds and 3-4 occupied edges per diamond so reflections/crossings repeat enough
#   to be visually learnable instead of appearing as a one-off accident.
# - Broken assumptions from the original solver: rays no longer stop at the first obstacle; earlier
#   ray paint also cannot block later rays because crossings must remain possible. The generator
#   therefore keeps all input objects isolated by a 1-cell halo so the diamond-to-segment parsing
#   logic remains valid, and it simulates rays only against the static input objects.
# - Placement legality and cleanliness: only diamonds plus their adjacent segments appear in the
#   input; these cells never overlap, never leave the canvas, and never touch another object's halo.
#   Output-only ray cells may overlap each other, but that overlap is flattened to one marker color,
#   which keeps the final grids readable instead of visually muddy.
# - Observability and non-degeneracy: color 9 is reserved for crossings and is excluded from input
#   objects. The generator rejects any sample with zero reflected rays, zero multi-color crossings,
#   looped paths, or a fused output identical to the straight-ray baseline, so the new behavior is
#   present in every accepted example.

COLLISION_COLOR = 9
OBJECT_COLORS = tuple(range(2, 9))

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (16, 26)
DIAMOND_COUNT_RANGE = (2, 3)
EDGES_PER_DIAMOND = (3, 4)
MIN_SEGMENT_LEN = 2
MAX_GEN_RETRIES = 500
MIN_REFLECTION_EVENTS = 4
MIN_REFLECTED_RAYS = 2
MIN_INTERSECTION_CELLS = 1
MAX_REFLECTIONS_PER_RAY = 8

_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]] | None:
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _expand_halo(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                halo.add((rr, cc))
    return halo


def _extract_sources(
    g: np.ndarray,
) -> tuple[set[int], list[tuple[int, list[tuple[int, int]]]], list[dict[str, object]]]:
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds = sorted([(col, pts) for col, pts in comps if _is_diamond8(pts)], key=lambda item: _bbox(item[1]))

    sources: list[dict[str, object]] = []
    for _, dpts in diamonds:
        dset = set(dpts)
        adj: set[tuple[int, int]] = set()
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.add((rr, cc))
        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True

        for oc in sorted(_cc8(adj_mask), key=_bbox):
            info = _outer_line_info(oc)
            if info is None:
                continue
            ocol = int(g[oc[0][0], oc[0][1]])
            slope_key, endpoints = info
            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])
            for start in (endpoints[0], endpoints[-1]):
                sources.append({"start": start, "dir": outward, "color": ocol})

    return bg_colors, diamonds, sources


def _simulate_reflective_rays(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors, diamonds, sources = _extract_sources(g)
    obstacle_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(g, list(set(int(v) for v in np.unique(g)) - bg_colors)))
    }

    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    ray_infos: list[dict[str, object]] = []
    total_reflections = 0
    loop_count = 0

    for source in sources:
        cur = tuple(source["start"])
        dr, dc = map(int, source["dir"])
        color = int(source["color"])
        seen: set[tuple[int, int, int, int]] = set()
        path: list[tuple[int, int]] = []
        reflections = 0

        while True:
            state = (int(cur[0]), int(cur[1]), int(dr), int(dc))
            if state in seen:
                loop_count += 1
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                break

            if (nr, nc) in obstacle_cells:
                dr, dc = -int(dr), -int(dc)
                reflections += 1
                continue

            cur = (nr, nc)
            path.append(cur)
            visits[cur].add(color)

            touch_r = (nr + dr, nc)
            touch_c = (nr, nc + dc)
            if 0 <= touch_r[0] < h and 0 <= touch_r[1] < w and touch_r in obstacle_cells:
                dr = -int(dr)
                reflections += 1
            if 0 <= touch_c[0] < h and 0 <= touch_c[1] < w and touch_c in obstacle_cells:
                dc = -int(dc)
                reflections += 1

        total_reflections += reflections
        ray_infos.append({"color": color, "path": path, "reflections": reflections})

    intersection_cells = sum(1 for colors in visits.values() if len(colors) >= 2)
    reflected_rays = sum(1 for info in ray_infos if int(info["reflections"]) > 0)
    return {
        "bg_colors": bg_colors,
        "diamonds": diamonds,
        "sources": sources,
        "visits": visits,
        "ray_infos": ray_infos,
        "reflection_count": total_reflections,
        "reflected_rays": reflected_rays,
        "intersection_count": intersection_cells,
        "loop_count": loop_count,
    }


def _render_reflective_output(grid: np.ndarray, visits: dict[tuple[int, int], set[int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for (r, c), colors in visits.items():
        out[r, c] = COLLISION_COLOR if len(colors) >= 2 else next(iter(colors))
    return out


def _straight_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg_colors, _, sources = _extract_sources(g)
    out = g.copy()
    for source in sources:
        color = int(source["color"])
        dr, dc = map(int, source["dir"])
        rr = int(source["start"][0] + dr)
        cc = int(source["start"][1] + dc)
        while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            if int(out[rr, cc]) not in bg_colors:
                break
            out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend endpoint rays from diagonal outer segments, but let them ricochet and mark crossings.

    Input structure:
    - Checkerboard background using the two dominant colors.
    - Hollow 8-pixel diamond outlines on checkerboard phase cells.
    - Short diagonal outer-border segments beside some diamond edges; these are the ray emitters.

    Rule:
    - Each outer segment still emits exactly two outward diagonal rays from its endpoints.
    - The input's non-background pixels now act as fixed mirrors: a direct diagonal hit reverses
      both axes, while a side-touch after a move flips the touched axis, and the ray continues.
    - Rays do not block one another. After tracing every path, a bg cell visited by one color takes
      that color; a bg cell visited by multiple different ray colors becomes the special marker 9.

    Key insight: First recover the original endpoint-and-direction pairs from each diamond-adjacent
    segment, then simulate every ray against only the static input objects and merge the resulting
    path visits at the end so reflections and multi-ray crossings are both preserved.
    """
    sim = _simulate_reflective_rays(np.asarray(grid, dtype=int))
    return _render_reflective_output(np.asarray(grid, dtype=int), sim["visits"])


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        if any(cell in all_cells for cell in segment):
            return None
        all_cells.update(segment)
        edge_parts.append({"edge": edge, "segment": segment})
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "halo": frozenset(_expand_halo(all_cells, h, w, pad=1)),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]
    random.shuffle(even_toplefts)

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        cands = []
        for r0, c0 in even_toplefts:
            placement = _candidate_placement(h, w, r0, c0, edges)
            if placement is not None:
                cands.append(placement)
        random.shuffle(cands)
        candidate_cache[edges] = cands[:140]

    occupied_halo: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            if occupied_halo.isdisjoint(placement["all_cells"]):
                halo = set(placement["halo"])
                if occupied_halo.isdisjoint(halo):
                    occupied_halo.update(halo)
                    placed[spec_i] = placement
                    if backtrack(i + 1):
                        return True
                    occupied_halo.difference_update(halo)
                    placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in OBJECT_COLORS if c not in diamond_set]
    counts = Counter()
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    used_per_diamond: list[set[int]] = [set() for _ in placements]
    refs = [(pi, ei) for pi, pl in enumerate(placements) for ei in range(len(pl["edges"]))]
    random.shuffle(refs)
    for pi, ei in refs:
        allowed = [c for c in palette if c not in used_per_diamond[pi]] or palette
        chosen = min(allowed, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += 1
        used_per_diamond[pi].add(chosen)
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in OBJECT_COLORS), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _sample_specs(diamond_count: int) -> list[tuple[str, ...]]:
    specs = []
    for _ in range(diamond_count):
        edge_count = random.randint(*EDGES_PER_DIAMOND)
        specs.append(tuple(sorted(random.sample(_EDGE_NAMES, edge_count))))
    return specs


def _build_input() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _sample_specs(diamond_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(OBJECT_COLORS), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    grid = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        _paint_cells(grid, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(grid, edge_part["segment"], segment_colors[pi][ei])

    if not _bg_counts_safe(grid):
        return None
    return grid


def _variant_metrics(grid: np.ndarray) -> dict[str, object]:
    sim = _simulate_reflective_rays(grid)
    fused = _render_reflective_output(grid, sim["visits"])
    straight = _straight_output(grid)
    diff_cells = int(np.count_nonzero(fused != straight))
    max_reflections = max((int(info["reflections"]) for info in sim["ray_infos"]), default=0)
    return {
        **sim,
        "fused": fused,
        "straight": straight,
        "diff_cells": diff_cells,
        "marker_cells": int(np.count_nonzero(fused == COLLISION_COLOR)),
        "max_reflections_per_ray": max_reflections,
    }


def _gen_attempt() -> np.ndarray | None:
    grid = _build_input()
    if grid is None:
        return None

    metrics = _variant_metrics(grid)
    if metrics["reflection_count"] < MIN_REFLECTION_EVENTS:
        return None
    if metrics["reflected_rays"] < MIN_REFLECTED_RAYS:
        return None
    if metrics["intersection_count"] < MIN_INTERSECTION_CELLS:
        return None
    if metrics["marker_cells"] < MIN_INTERSECTION_CELLS:
        return None
    if metrics["max_reflections_per_ray"] > MAX_REFLECTIONS_PER_RAY:
        return None
    if metrics["diff_cells"] < 4:
        return None
    if np.array_equal(metrics["fused"], metrics["straight"]):
        return None
    if not np.array_equal(main(grid), metrics["fused"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid reflective-crossing sample after {MAX_GEN_RETRIES} attempts")
