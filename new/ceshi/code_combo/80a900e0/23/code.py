# Combo variant of 80a900e0 (index 23)
# combo_spec: D3/0+D11/0+D17/0
# Name: 全穿透全覆盖

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts, dtype=int))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


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


def _collect_emitters(grid: np.ndarray) -> tuple[set[int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    emitters: list[dict[str, object]] = []

    for _, dpts in sorted(diamonds, key=lambda item: _bbox(item[1])):
        dset = set(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))
        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True

        parsed = []
        for oc in _cc8(adj_mask):
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info
            parsed.append((_bbox(oc), 0 if slope_key == "rc_diff" else 1, endpoints[0], oc, slope_key, endpoints))

        for _, _, _, oc, slope_key, endpoints in sorted(parsed):
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

            emitters.append(
                {
                    "color": int(g[oc[0][0], oc[0][1]]),
                    "segment": tuple(oc),
                    "endpoints": (endpoints[0], endpoints[-1]),
                    "outward": (-best_n[0], -best_n[1]),
                }
            )

    return bg_colors, emitters


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _classic_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg_colors, emitters = _collect_emitters(g)
    for emitter in emitters:
        color = int(emitter["color"])
        outward = emitter["outward"]
        for start in emitter["endpoints"]:
            for r, c in _ray_cells(g.shape, start, outward):
                if int(out[r, c]) in bg_colors:
                    out[r, c] = color
                else:
                    break
    return out


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg_colors, emitters = _collect_emitters(g)

    touch_count: Counter[tuple[int, int]] = Counter()
    object_hits: set[tuple[int, int]] = set()
    for emitter in emitters:
        outward = emitter["outward"]
        for start in emitter["endpoints"]:
            for cell in _ray_cells(g.shape, start, outward):
                touch_count[cell] += 1
                if int(g[cell]) not in bg_colors:
                    object_hits.add(cell)

    fusion = main(g)
    classic = _classic_output(g)
    return {
        "emitters": len(emitters),
        "object_penetration_cells": len(object_hits),
        "ray_overlap_cells": sum(1 for v in touch_count.values() if v >= 2),
        "classic_diff_cells": int(np.count_nonzero(fusion != classic)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal rays from the endpoints of the short outer segments around each diamond, but let the rays slice through everything.

    Input structure:
    - A 0/1 checkerboard fills the canvas, with the two background colors still the dominant colors in the input.
    - Non-background objects are hollow 8-pixel diamonds plus short diagonal outer segments that sit one step beyond chosen diamond edges.
    - Only the diamonds and the short segments appear in the input; the long rays are absent and must be inferred.

    Rule:
    - For every outer segment, infer which diagonal normal points away from the adjacent diamond, then emit one ray from each segment endpoint in that outward direction.
    - Each ray runs all the way to the grid boundary with no blocking at all: it does not stop for diamonds, segments, or previously painted rays.
    - Every traversed cell is recolored to the segment color, including non-background cells, so later rays overwrite earlier ones where paths cross.

    Key insight:
    - The only stable information comes from the input geometry: detect emitters from the untouched diamonds/segments, order them consistently by spatial scan, and then treat the rays as an unrestricted last-write-wins laser pass.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _, emitters = _collect_emitters(g)

    for emitter in emitters:
        color = int(emitter["color"])
        outward = emitter["outward"]
        for start in emitter["endpoints"]:
            for r, c in _ray_cells(g.shape, start, outward):
                out[r, c] = color

    return out


# Design analysis for the combo variant:
# - Concrete behavior trace:
#   In the base task, a ray stops as soon as the next cell is not checkerboard background.
#   Here the same emitter keeps stepping to the boundary and recolors every visited cell.
#   Example: if a BR-directed ray hits another diamond after 3 cells, the classic output stops
#   there; this combo instead recolors the diamond pixels it crosses and continues past it.
#   If a later ray crosses that same corridor, the later color replaces the earlier one.
# - Complexity calibration:
#   The generator fixes two diamonds and gives each 2-3 active edges, so every sample has
#   4-6 emitters with unique segment colors. That is enough repeated structure to reveal
#   "full laser coverage" and overlap priority, but avoids the unreadable density of 3+ diamonds.
# - Implicit-assumption audit:
#   The original solver relied on `out` as a blocking state and on rays never touching objects.
#   Both assumptions are removed. Detection now reads emitters only from the untouched input,
#   while painting uses a separate unrestricted pass whose order is geometric rather than
#   color-based. The generator enforces a one-cell moat between input objects so adjacent
#   segments from different diamonds cannot merge during emitter detection.
# - Placement legality:
#   Diamonds and visible outer segments are disjoint and buffered from one another in the input.
#   Rays are allowed to overlap anything in the output, but every segment must still be a
#   clipped 2-3 pixel diagonal and every diamond must remain a valid 5x5 hollow outline.
# - Observability / cleanliness:
#   Samples are accepted only if the unrestricted output differs from the classic blocked output,
#   if at least one ray crosses another ray, and if at least one ray penetrates an existing
#   non-background object. Unique segment colors make the overwrite order visible without using
#   hidden numeric color preferences.


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (16, 28)
DIAMOND_COUNT = 2
EDGES_PER_DIAMOND = (2, 3)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
PAIR_SCAN_LIMIT = 140

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _expand8(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    frontier = set(cells)
    for _ in range(pad):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in out:
                        nxt.add((rr, cc))
        out.update(nxt)
        frontier = nxt
    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    object_cells = set(diamond)
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
        if any(cell in object_cells for cell in segment):
            return None
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        object_cells.update(segment)
        edge_parts.append(
            {
                "edge_name": edge,
                "segment": tuple(segment),
                "rays": tuple(rays),
                "ray_set": frozenset(rays),
                "weight": len(segment) + len(rays),
            }
        )

    ray_cells = set()
    for edge_part in edge_parts:
        ray_cells.update(edge_part["ray_set"])

    return {
        "diamond": tuple(diamond),
        "diamond_bbox": (r0, c0, r0 + 4, c0 + 4),
        "edges": tuple(edge_parts),
        "object_cells": frozenset(object_cells),
        "buffered_cells": frozenset(_expand8(object_cells, h, w, pad=1)),
        "ray_cells": frozenset(ray_cells),
    }


def _pair_score(left: dict[str, object], right: dict[str, object]) -> tuple[int, int, int]:
    object_hits = len(left["ray_cells"] & right["object_cells"]) + len(right["ray_cells"] & left["object_cells"])
    ray_overlap = len(left["ray_cells"] & right["ray_cells"])
    strong = int(object_hits >= 2 and ray_overlap >= 1)
    return strong, object_hits, ray_overlap


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]
    random.shuffle(even_toplefts)

    candidate_lists: list[list[dict[str, object]]] = []
    for edges in specs:
        cands = []
        for r0, c0 in even_toplefts:
            cand = _candidate_placement(h, w, r0, c0, edges)
            if cand is not None:
                cands.append(cand)
        if not cands:
            return None
        random.shuffle(cands)
        candidate_lists.append(cands[:PAIR_SCAN_LIMIT])

    best_pair = None
    best_score = (-1, -1, -1)
    for left in candidate_lists[0]:
        for right in candidate_lists[1]:
            if not left["buffered_cells"].isdisjoint(right["buffered_cells"]):
                continue
            score = _pair_score(left, right)
            if score > best_score:
                best_score = score
                best_pair = [left, right]
                if score[0] == 1 and score[1] >= 3 and score[2] >= 2:
                    return best_pair
    return best_pair if best_score > (0, 0, 0) else None


def _build_input(
    h: int,
    w: int,
    placements: list[dict[str, object]],
    diamond_colors: list[int],
    segment_colors: list[list[int]],
) -> np.ndarray:
    inp = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(inp, edge_part["segment"], segment_colors[pi][ei])
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    specs = []
    for _ in range(DIAMOND_COUNT):
        edge_count = random.randint(*EDGES_PER_DIAMOND)
        specs.append(tuple(sorted(random.sample(_EDGE_NAMES, edge_count))))

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    all_colors = random.sample(list(COLOR_RANGE), 8)
    diamond_colors = all_colors[:DIAMOND_COUNT]
    segment_palette = all_colors[DIAMOND_COUNT:]

    total_edges = sum(len(pl["edges"]) for pl in placements)
    if total_edges > len(segment_palette):
        return None

    random.shuffle(segment_palette)
    cursor = 0
    segment_colors: list[list[int]] = []
    for placement in placements:
        take = len(placement["edges"])
        segment_colors.append(segment_palette[cursor : cursor + take])
        cursor += take

    inp = _build_input(h, w, placements, diamond_colors, segment_colors)
    metrics = _fusion_metrics(inp)
    if metrics["emitters"] < 4:
        return None
    if metrics["object_penetration_cells"] < 2:
        return None
    if metrics["ray_overlap_cells"] < 1:
        return None
    if metrics["classic_diff_cells"] < 6:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
