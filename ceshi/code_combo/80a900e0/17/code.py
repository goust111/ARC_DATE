# Combo variant of 80a900e0 (index 17)
# combo_spec: D1/0+D3/1
# Name: 正交长度编码射线

# Design analysis for the fused variant:
# - Original 80a900e0 infers an outward diagonal normal from each short diagonal
#   outer segment, then extends two diagonal rays to the grid boundary.
# - D1/0 changes the path family: the same outward normal is now decomposed into
#   its row sign and column sign, producing one vertical ray and one horizontal
#   ray instead of two diagonal rays.
# - D3/1 changes the stop rule: each ray has a fixed length N rather than
#   running until the boundary. We use N = 2 * visible_segment_length so the
#   task has only two observable lengths (4 and 6), which keeps the hypothesis
#   space small while still forcing the solver to read segment length.
# - A direct reuse of the original "3 pixels, clipped to 2 by the boundary"
#   generator would conflict with fixed orthogonal rays: clipping against the
#   same side that determines a ray direction can force a zero-length ray and
#   erase the distance cue. To preserve the fused behavior, the generator keeps
#   the same diagonal segment family but explicitly chooses visible segment
#   lengths 2 or 3 in-bounds.
# - The solver's main hidden assumption remains: any non-background pixels in
#   the 8-neighborhood of a diamond must be that diamond's own outer segments.
#   The generator therefore reserves a one-cell moat around every diamond so no
#   foreign diamond/segment/ray can be mistaken for an adjacent segment.
# - Visual cleanliness check: each segment always yields exactly two straight
#   rays, one vertical and one horizontal, and the generator forbids overlaps.
#   This keeps the output readable enough that humans can see both the direction
#   shift and the 2-cell vs 3-cell length encoding from a few examples.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve()
for parent in ROOT.parents:
    if (parent / "arc_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from arc_utils import bbox_coords, components_mask, neighbors8


RAY_MULTIPLIER = 2


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


def _orthogonal_emitters(
    endpoints: list[tuple[int, int]], outward: tuple[int, int]
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    dr, dc = outward
    vertical_origin = min(endpoints, key=lambda p: p[0]) if dr < 0 else max(endpoints, key=lambda p: p[0])
    horizontal_origin = min(endpoints, key=lambda p: p[1]) if dc < 0 else max(endpoints, key=lambda p: p[1])
    if vertical_origin == horizontal_origin:
        return None
    return [(vertical_origin, (dr, 0)), (horizontal_origin, (0, dc))]


def _fixed_ray_cells(
    shape: tuple[int, int], origin: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    dr, dc = direction
    r, c = origin
    cells: list[tuple[int, int]] = []
    for _ in range(int(steps)):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((int(r), int(c)))
    return cells


def _orthogonal_ray_cells(
    shape: tuple[int, int], endpoints: list[tuple[int, int]], outward: tuple[int, int], steps: int
) -> list[tuple[int, int]] | None:
    emitters = _orthogonal_emitters(endpoints, outward)
    if emitters is None:
        return None
    cells: list[tuple[int, int]] = []
    for origin, direction in emitters:
        ray = _fixed_ray_cells(shape, origin, direction, steps)
        if ray is None:
            return None
        cells.extend(ray)
    return cells if len(set(cells)) == len(cells) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend short diagonal outer segments into fixed-length orthogonal rays around checkerboard diamonds.

    Input structure:
    - The background is a two-color checkerboard made from the two dominant colors.
    - Foreground objects are hollow 8-pixel diamonds plus short diagonal outer segments sitting just outside some diamond edges.
    - Each outer segment is 2 or 3 pixels long and uses its own non-background color.

    Rule:
    - For every outer segment, infer the outward diagonal normal exactly as in the base task by checking which side faces the diamond.
    - Split that outward diagonal into its row and column signs: the endpoint farthest in the row direction emits a vertical ray, and the endpoint farthest in the column direction emits a horizontal ray.
    - Each of those two rays runs for exactly N cells, where N = 2 * visible segment length, painting background cells with the segment color.

    Key insight:
    - The diagonal outer segment still tells you which way is "out", but the actual extension is no longer diagonal or boundary-driven.
    - You must combine two inferences: convert the outward diagonal into one vertical plus one horizontal ray, and read the ray length from the segment's own pixel count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj.append((int(rr), int(cc)))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, ordered = info

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
            ray_steps = len(ordered) * RAY_MULTIPLIER
            endpoints = [ordered[0], ordered[-1]]
            emitters = _orthogonal_emitters(endpoints, outward)
            if emitters is None:
                continue

            rays: list[list[tuple[int, int]]] = []
            claimed: set[tuple[int, int]] = set()
            valid = True
            for origin, direction in emitters:
                ray = _fixed_ray_cells((h, w), origin, direction, ray_steps)
                if ray is None:
                    valid = False
                    break
                if any(int(g[r, c]) not in bg_colors for r, c in ray):
                    valid = False
                    break
                if not claimed.isdisjoint(ray):
                    valid = False
                    break
                claimed.update(ray)
                rays.append(ray)
            if not valid:
                continue
            for ray in rays:
                for r, c in ray:
                    out[r, c] = ocol

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
SEGMENT_LENGTHS = (2, 3)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500


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


def _segment_cells(
    h: int, w: int, r0: int, c0: int, edge_name: str, seg_len: int
) -> list[tuple[int, int]] | None:
    info = _EDGE_INFO[edge_name]
    cells = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    ordered = _sort_segment(cells, info["slope"])
    return ordered[:seg_len]


def _diamond_forbidden_zone(h: int, w: int, diamond: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in diamond:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    zone.add((int(rr), int(cc)))
    return frozenset(zone)


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[tuple[str, int], ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []

    for edge_name, seg_len in edges:
        info = _EDGE_INFO[edge_name]
        segment = _segment_cells(h, w, r0, c0, edge_name, seg_len)
        if segment is None or len(segment) < MIN_SEGMENT_LEN:
            return None
        steps = len(segment) * RAY_MULTIPLIER
        rays = _orthogonal_ray_cells((h, w), [segment[0], segment[-1]], info["outward"], steps)
        if rays is None:
            return None
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": rays,
                "weight": len(part_cells),
                "seg_len": len(segment),
                "edge": edge_name,
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "diamond_forbidden": _diamond_forbidden_zone(h, w, diamond),
    }


def _search_placements(h: int, w: int, specs: list[tuple[tuple[str, int], ...]]) -> list[dict[str, object]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: sum(seg_len for _, seg_len in specs[i]),
        reverse=True,
    )
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[tuple[str, int], ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        cands = []
        for r0, c0 in positions:
            cand = _candidate_placement(h, w, r0, c0, edges)
            if cand is not None:
                cands.append(cand)
        random.shuffle(cands)
        candidate_cache[edges] = cands

    occupied_cells: set[tuple[int, int]] = set()
    reserved_near_diamond: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            forbidden = placement["diamond_forbidden"]
            if not reserved_near_diamond.isdisjoint(forbidden):
                continue
            if not occupied_cells.isdisjoint(forbidden):
                continue
            if not reserved_near_diamond.isdisjoint(cells):
                continue
            occupied_cells.update(cells)
            reserved_near_diamond.update(forbidden)
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_cells.difference_update(cells)
            reserved_near_diamond.difference_update(forbidden)
            placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _choose_specs(diamond_count: int) -> list[tuple[tuple[str, int], ...]]:
    specs: list[list[tuple[str, int]]] = []
    edge_slots: list[tuple[int, int]] = []
    for di in range(diamond_count):
        edge_names = random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))
        spec = [(edge_name, random.choice(SEGMENT_LENGTHS)) for edge_name in sorted(edge_names)]
        specs.append(spec)
        for ei in range(len(spec)):
            edge_slots.append((di, ei))

    if len(edge_slots) >= 2:
        for idx, forced_len in zip(edge_slots[:2], SEGMENT_LENGTHS):
            di, ei = idx
            edge_name, _ = specs[di][ei]
            specs[di][ei] = (edge_name, forced_len)

    return [tuple(spec) for spec in specs]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _choose_specs(diamond_count)

    flat_lengths = [seg_len for spec in specs for _, seg_len in spec]
    if set(flat_lengths) != set(SEGMENT_LENGTHS):
        return None

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs_arr = np.asarray(rs, dtype=int)
        cs_arr = np.asarray(cs, dtype=int)
        inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
