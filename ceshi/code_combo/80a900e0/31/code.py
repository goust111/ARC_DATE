# Combo variant of 80a900e0 (index 31)
# combo_spec: D3/2+D17/0
# Name: 同色停异色穿

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


# Design analysis for combo D3/2 + D17/0
# a) Concrete behavior trace:
#    Base task: from each outer-segment endpoint, step outward and stop at the first non-background
#    cell in the evolving output, so both static objects and earlier rays block later rays.
#    Fusion task: step outward while checking only the original input grid. A ray stops only when
#    the next input cell already has the same color as that ray. Different-color diamonds/segments
#    are painted through, and previously painted rays never block because they are absent from the
#    stop test. Example: if a red ray meets a green diamond after 3 cells and a red segment after
#    6 cells, the original task stops at the green diamond; this fusion paints through the green
#    diamond and stops only before the red segment.
# b) Complexity calibration:
#    The generator uses exactly two diamonds with 2-3 active edges each. That gives 4-6 emitters,
#    enough for repeated evidence of color-matched stopping and foreign-object penetration, while
#    staying visually readable. Segment colors are deliberately reused and may match the other
#    diamond's color, so "same color" is a visible, repeated category rather than a one-off.
# c) Implicit-assumption audit:
#    The original main() relied on two hidden assumptions: (1) any non-background cell blocks,
#    and (2) out-state is a correct proxy for future blocking, so earlier rays stop later rays.
#    Both fail here. The new solver therefore separates emitter detection (from untouched input)
#    from ray painting, and the stop predicate reads only the static input grid. Stable emitter
#    ordering is still needed because crossing rays must resolve to a single final color; we keep
#    a geometric scan order for painting, but that order no longer affects stopping.
# d) Placement legality:
#    Input objects (diamonds plus visible outer segments) must remain disjoint, in-bounds, and
#    separated by a one-cell 8-neighborhood moat so segment detection cannot merge components
#    across diamonds. Rays are allowed to cross objects and each other in the output.
# e) Visual observability:
#    Samples are rejected unless they contain all three signature effects: at least one ray stops
#    because of a same-color input object, at least one ray paints through a different-color input
#    object, and at least one cell is visited by multiple rays. This makes the fusion behavior
#    inferable from a small number of examples rather than collapsing back to the base task.
# f) Visual cleanliness:
#    Only two diamonds are used, and input objects are buffered apart. Overlap is confined to the
#    inferred rays in the output, which keeps the visible causes (diamonds and short segments)
#    readable.
# g) Color-rule inferability:
#    The palette is small and intentionally reused. Segment colors are drawn from "other diamond
#    colors + 1-2 extras", so the solver can visually test a simple rule: same-color contact ends
#    a ray; different-color contact does not.


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


def _trace_fusion_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    visited: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if int(g[r, c]) == color:
            return visited, (r, c)
        visited.append((r, c))
        r += dr
        c += dc
    return visited, None


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
    diff_object_cells: set[tuple[int, int]] = set()
    same_color_stop_cells: set[tuple[int, int]] = set()
    same_color_stop_rays = 0

    for emitter in emitters:
        color = int(emitter["color"])
        outward = emitter["outward"]
        for start in emitter["endpoints"]:
            visited, stop_cell = _trace_fusion_ray(g, start, outward, color)
            for cell in visited:
                touch_count[cell] += 1
                if int(g[cell]) not in bg_colors and int(g[cell]) != color:
                    diff_object_cells.add(cell)
            if stop_cell is not None:
                same_color_stop_rays += 1
                same_color_stop_cells.add(stop_cell)

    fusion = main(g)
    classic = _classic_output(g)
    return {
        "emitters": len(emitters),
        "same_color_stop_rays": same_color_stop_rays,
        "same_color_stop_cells": len(same_color_stop_cells),
        "diff_object_penetration_cells": len(diff_object_cells),
        "ray_overlap_cells": sum(1 for v in touch_count.values() if v >= 2),
        "classic_diff_cells": int(np.count_nonzero(fusion != classic)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from the endpoint pair of each outer segment around the diamonds, but stop only on matching colors.

    Input structure:
    - The canvas is still a checkerboard whose two dominant colors are the background.
    - Non-background input objects are hollow 8-pixel diamonds plus short diagonal outer segments placed one step beyond chosen diamond edges.
    - No long rays are present in the input; only the diamonds and their short emitters are visible.

    Rule:
    - For each outer segment, infer the outward diagonal direction by comparing the segment with its adjacent diamond, then emit one ray from each segment endpoint.
    - A ray advances cell by cell until it would enter an input pixel of the same color; different-color diamonds, different-color segments, and previously painted rays do not stop it.
    - Every traversed cell is recolored to the segment color, so crossing rays are resolved only by the stable painting order, not by blocking.

    Key insight:
    - Detection and stopping must be separated: infer all emitters from the untouched input geometry first, then trace rays while checking color matches against the original grid rather than the evolving output.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _, emitters = _collect_emitters(g)

    for emitter in emitters:
        color = int(emitter["color"])
        outward = emitter["outward"]
        for start in emitter["endpoints"]:
            visited, _ = _trace_fusion_ray(g, start, outward, color)
            for r, c in visited:
                out[r, c] = color

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (16, 28)
DIAMOND_COUNT = 2
EDGES_PER_DIAMOND = (2, 3)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = tuple(range(2, 10))
MAX_GEN_RETRIES = 450
PAIR_SCAN_LIMIT = 120
COLOR_ASSIGN_RETRIES = 80

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
    h: int,
    w: int,
    r0: int,
    c0: int,
    edges: tuple[str, ...],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    diamond_set = set(diamond)
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
                "endpoints": (segment[0], segment[-1]),
                "outward": info["outward"],
                "rays": tuple(rays),
                "ray_set": frozenset(rays),
            }
        )

    ray_cells = set()
    for edge_part in edge_parts:
        ray_cells.update(edge_part["ray_set"])

    return {
        "diamond": tuple(diamond),
        "diamond_cells": frozenset(diamond_set),
        "diamond_bbox": (r0, c0, r0 + 4, c0 + 4),
        "edges": tuple(edge_parts),
        "object_cells": frozenset(object_cells),
        "buffered_cells": frozenset(_expand8(object_cells, h, w, pad=1)),
        "ray_cells": frozenset(ray_cells),
    }


def _pair_score(left: dict[str, object], right: dict[str, object]) -> tuple[int, int, int]:
    diamond_hits = len(left["ray_cells"] & right["diamond_cells"]) + len(right["ray_cells"] & left["diamond_cells"])
    object_hits = len(left["ray_cells"] & right["object_cells"]) + len(right["ray_cells"] & left["object_cells"])
    ray_overlap = len(left["ray_cells"] & right["ray_cells"])
    return diamond_hits, object_hits, ray_overlap


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
                if score[0] >= 2 and score[1] >= 4 and score[2] >= 1:
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


def _choose_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]] | None:
    extra_count = random.randint(1, 2)
    extras = random.sample([c for c in COLOR_RANGE if c not in diamond_colors], extra_count)
    shared_palette = list(dict.fromkeys(diamond_colors + extras))

    segment_colors: list[list[int]] = []
    for pi, placement in enumerate(placements):
        own_diamond = diamond_colors[pi]
        options = [c for c in shared_palette if c != own_diamond]
        if not options:
            return None
        colors = [random.choice(options) for _ in placement["edges"]]
        segment_colors.append(colors)

    other0 = diamond_colors[1]
    other1 = diamond_colors[0]
    if placements[0]["edges"]:
        segment_colors[0][random.randrange(len(segment_colors[0]))] = other0
    if placements[1]["edges"]:
        segment_colors[1][random.randrange(len(segment_colors[1]))] = other1
    return segment_colors


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

    for _ in range(COLOR_ASSIGN_RETRIES):
        diamond_colors = random.sample(list(COLOR_RANGE), DIAMOND_COUNT)
        segment_colors = _choose_segment_colors(placements, diamond_colors)
        if segment_colors is None:
            continue
        inp = _build_input(h, w, placements, diamond_colors, segment_colors)
        metrics = _fusion_metrics(inp)
        if metrics["emitters"] < 4:
            continue
        if metrics["same_color_stop_rays"] < 1:
            continue
        if metrics["diff_object_penetration_cells"] < 2:
            continue
        if metrics["ray_overlap_cells"] < 1:
            continue
        if metrics["classic_diff_cells"] < 6:
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
