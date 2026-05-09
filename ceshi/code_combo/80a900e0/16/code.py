# Combo variant of 80a900e0 (index 16)
# combo_spec: D14/0+D15/1
# Name: 颜色位置编码加数值激活
#
# Design analysis
# ===============
# spec_variant:
#   background:
#     same_as_source: 0/1 checkerboard with top-left=1 and all foreground on even parity cells
#   objects:
#     diamond:
#       same_as_source: 8-pixel hollow diamond in a 5x5 box
#       new_color_pool: exactly three distinct diamond colors {3, 5, 7}
#     outer_segment:
#       same_geometry: 3-pixel diagonal segment (or 2 if clipped) one step outside a diamond edge
#       new_color_rule: edge position determines color globally
#         TR -> 2
#         TL -> 8
#         BL -> 4
#         BR -> 6
#       activation_rule: a segment emits rays iff its position-color value is greater than its own diamond color
#   output_rule:
#     same_ray_geometry: active segments emit two outward diagonal rays from their endpoints
#     inert_segments: non-active segments remain visible in input/output but never grow rays
#
# a) Concrete behavior trace:
#   Source task: every adjacent outer diagonal component emits rays using its own color.
#   Fusion task: first classify that component as TR/TL/BL/BR from geometry, then read its color as a global
#   side label, then compare that side color with the local diamond color. Example with fixed side colors
#   {TR:2, BL:4, BR:6, TL:8}: around a diamond of color 5, BR(6) and TL(8) are active but TR(2) and BL(4)
#   stay inert. The same BR color 6 is active next to diamond 5 but inactive next to diamond 7. That is the
#   essential difference from the source and from pure D15.
#
# b) Complexity calibration:
#   A raw numeric `segment_color > diamond_color` rule is hard to see unless the palette is constrained.
#   To make the threshold human-observable, generator uses odd diamond colors {3,5,7} and even side colors
#   {2,4,6,8}, producing nested activation sets:
#     diamond 3 -> active side colors {4,6,8}
#     diamond 5 -> active side colors {6,8}
#     diamond 7 -> active side colors {8}
#   Each sample contains three diamonds with repeated side positions, so both the global side-color code and
#   the activation threshold have repeated support.
#
# c) Implicit assumptions audit for main():
#   - `bg_colors = two most common colors` still assumes 0/1 dominate; generator checks this.
#   - Diamond detection by exact 8-pixel shape still works because diamond colors are distinct and disjoint
#     from the fixed side-color palette.
#   - The source solver assumed every adjacent outer component emits. Fusion breaks that, so solver now stores
#     the parent diamond color and gates emission with `edge_to_color[edge_name] > diamond_color`.
#   - D15 also breaks the source assumption that "component color is merely local decoration". Solver therefore
#     infers a global edge->color map from repeated observations, with fixed defaults as a fallback.
#   - Geometry-based outward normal inference is unchanged, so the original segment classification logic remains valid.
#
# d) Placement legality:
#   Generation stays output-first. Candidate placements reserve diamond pixels, all segments, and only the rays
#   of active segments. This guarantees no overlaps, no out-of-bounds placements, and no fake collisions caused
#   by rays that should not exist under the fusion rule.
#
# e) Visual observability:
#   Every generated grid contains all four side positions, at least two positions that appear both active and
#   inactive across different diamonds, and one diamond for each threshold level 3/5/7. This lets a viewer see:
#   same side -> same segment color, but same side-color does not always emit; emission depends on the diamond color.
#
# f) Visual cleanliness:
#   Active rays are the only long structures. Inert segments remain short, local, and disjoint. Output-first
#   placement forbids ray/segment/diamond overlap, so the grid remains readable despite repeated side colors.
#
# g) Change inferability:
#   The palette is intentionally small and structured. Four fixed side colors encode TR/TL/BL/BR, while the three
#   diamond colors create a visible activation ladder. The generator rejects samples unless they contain both active
#   and inert segments and at least two side positions whose activation status changes across diamonds.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts))


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


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


_EDGE_TO_COLOR = {
    "TR": 2,
    "TL": 8,
    "BL": 4,
    "BR": 6,
}

_EDGE_FROM_ORIENTATION = {
    ("rc_diff", (-1, 1)): "TR",
    ("rc_sum", (-1, -1)): "TL",
    ("rc_diff", (1, -1)): "BL",
    ("rc_sum", (1, 1)): "BR",
}


def _classify_outer_component(
    g: np.ndarray,
    comp: list[tuple[int, int]],
    dset: set[tuple[int, int]],
) -> dict[str, object] | None:
    info = _outer_line_info(comp)
    if info is None:
        return None
    slope_key, endpoints = info

    if slope_key == "rc_diff":
        normals = [(-1, 1), (1, -1)]
    else:
        normals = [(-1, -1), (1, 1)]

    best_n = None
    best_score = -1
    for n in normals:
        score = sum(((r + n[0], c + n[1]) in dset) for r, c in comp)
        if score > best_score:
            best_score = score
            best_n = n
    if best_n is None or best_score <= 0:
        return None

    outward = (-best_n[0], -best_n[1])
    edge_name = _EDGE_FROM_ORIENTATION.get((slope_key, outward))
    if edge_name is None:
        return None

    return {
        "edge_name": edge_name,
        "outward": outward,
        "endpoints": (endpoints[0], endpoints[-1]),
        "observed_color": int(g[comp[0][0], comp[0][1]]),
    }


def _extract_structure(
    g: np.ndarray,
) -> tuple[set[int], list[dict[str, object]], list[dict[str, object]]]:
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds = [{"color": int(col), "pts": pts} for col, pts in comps if _is_diamond8(pts)]
    diamonds.sort(key=lambda item: _bbox(item["pts"]))

    outer_parts: list[dict[str, object]] = []
    for diamond_index, diamond in enumerate(diamonds):
        dpts = diamond["pts"]
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
        for comp in _cc8(adj_mask):
            part = _classify_outer_component(g, comp, dset)
            if part is None:
                continue
            part["diamond_index"] = diamond_index
            part["diamond_color"] = int(diamond["color"])
            outer_parts.append(part)

    return bg_colors, diamonds, outer_parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend only selected diagonal rays around checkerboard diamonds.

    Input structure:
    - A 0/1 checkerboard background.
    - One or more 8-pixel hollow diamonds in non-background colors.
    - Short outer diagonal segments next to diamond edges, with segment color determined by edge position
      (TR/TL/BL/BR share the same color code everywhere in the grid).

    Rule:
    - Classify each outer segment by geometry to recover which diamond side it belongs to and which outward
      diagonal direction its rays would follow.
    - Infer the global side-position color code from all visible segments, then emit rays from a segment's two
      endpoints only when that side color is numerically greater than the parent diamond's color.

    Key insight:
    - The task combines two interacting cues: segment color encodes side position globally, and that recovered
      side color acts as a threshold against the local diamond color to decide whether the segment is active.
    """
    g = np.asarray(grid, dtype=int)
    bg_colors, _diamonds, outer_parts = _extract_structure(g)
    out = g.copy()

    edge_color_votes: dict[str, Counter[int]] = {edge: Counter() for edge in _EDGE_TO_COLOR}
    for part in outer_parts:
        edge_color_votes[part["edge_name"]].update([part["observed_color"]])
    edge_to_color = {
        edge: int(votes.most_common(1)[0][0]) if votes else _EDGE_TO_COLOR[edge]
        for edge, votes in edge_color_votes.items()
    }

    h, w = out.shape
    for part in outer_parts:
        segment_color = edge_to_color[part["edge_name"]]
        if segment_color <= part["diamond_color"]:
            continue
        dr, dc = part["outward"]
        for pr, pc in part["endpoints"]:
            rr, cc = pr + dr, pc + dc
            while 0 <= rr < h and 0 <= cc < w:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = segment_color
                else:
                    break
                rr += dr
                cc += dc

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (18, 30)
MIN_SEGMENT_LEN = 2
MAX_GEN_RETRIES = 600
COLOR_RANGE = range(2, 10)

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3),
    (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)
_DIAMOND_COLORS = (3, 5, 7)
_OBJECT_SPEC_TEMPLATES = (
    (
        (3, ("BL", "BR", "TL")),
        (5, ("BL", "BR", "TL", "TR")),
        (7, ("BR", "TL", "TR")),
    ),
    (
        (3, ("BL", "BR", "TL", "TR")),
        (5, ("BL", "BR", "TL")),
        (7, ("BL", "TL", "TR")),
    ),
    (
        (3, ("BL", "BR", "TL")),
        (5, ("BR", "TL", "TR")),
        (7, ("BL", "BR", "TL")),
    ),
)


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


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    diamond_color: int,
    edges: tuple[str, ...],
) -> dict[str, object] | None:
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
        active = _EDGE_TO_COLOR[edge] > diamond_color
        rays = []
        if active:
            rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "edge_name": edge,
                "segment": segment,
                "rays": rays,
                "active": active,
                "weight": len(part_cells),
            }
        )
    return {
        "diamond_color": diamond_color,
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
    }


def _search_placements(
    h: int,
    w: int,
    object_specs: list[tuple[int, tuple[str, ...]]],
) -> list[dict[str, object]] | None:
    order = sorted(range(len(object_specs)), key=lambda i: (len(object_specs[i][1]), -object_specs[i][0]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[int, tuple[str, ...]], list[dict[str, object]]] = {}
    for spec in {object_specs[i] for i in order}:
        diamond_color, edges = spec
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[spec] = [
            p
            for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, diamond_color, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(object_specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        spec = object_specs[spec_i]
        for placement in candidate_cache[spec]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells):
                occupied.update(cells)
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _choose_object_specs() -> list[tuple[int, tuple[str, ...]]]:
    template = random.choice(_OBJECT_SPEC_TEMPLATES)
    specs = list(template)
    random.shuffle(specs)
    return specs


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _active_status(part: dict[str, object]) -> bool:
    return _EDGE_TO_COLOR[part["edge_name"]] > part["diamond_color"]


def _variant_observability_ok(inp: np.ndarray) -> bool:
    _bg_colors, diamonds, outer_parts = _extract_structure(inp)
    if len(diamonds) != 3:
        return False
    if sorted(int(d["color"]) for d in diamonds) != list(_DIAMOND_COLORS):
        return False
    if not outer_parts:
        return False
    if {part["edge_name"] for part in outer_parts} != set(_EDGE_NAMES):
        return False
    if any(part["observed_color"] != _EDGE_TO_COLOR[part["edge_name"]] for part in outer_parts):
        return False

    active_parts = [part for part in outer_parts if _active_status(part)]
    inactive_parts = [part for part in outer_parts if not _active_status(part)]
    if not active_parts or not inactive_parts:
        return False

    mixed_edges = 0
    for edge in _EDGE_NAMES:
        edge_parts = [part for part in outer_parts if part["edge_name"] == edge]
        if not edge_parts:
            return False
        statuses = {_active_status(part) for part in edge_parts}
        if statuses == {False, True}:
            mixed_edges += 1
    if mixed_edges < 2:
        return False

    color_to_active_count: dict[int, int] = {}
    for diamond_index, diamond in enumerate(diamonds):
        color = int(diamond["color"])
        color_to_active_count[color] = sum(
            _active_status(part) for part in outer_parts if part["diamond_index"] == diamond_index
        )
    if color_to_active_count.get(7) != 1:
        return False
    if not (color_to_active_count.get(3, 0) > color_to_active_count.get(5, 0) > color_to_active_count.get(7, 0)):
        return False

    return True


def _fusion_signature(inp: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
    _bg_colors, diamonds, outer_parts = _extract_structure(inp)
    color_to_active_count: dict[int, int] = {}
    mixed_edges: list[str] = []
    for diamond_index, diamond in enumerate(diamonds):
        color = int(diamond["color"])
        color_to_active_count[color] = sum(
            _active_status(part) for part in outer_parts if part["diamond_index"] == diamond_index
        )
    for edge in _EDGE_NAMES:
        edge_parts = [part for part in outer_parts if part["edge_name"] == edge]
        if edge_parts and {_active_status(part) for part in edge_parts} == {False, True}:
            mixed_edges.append(edge)
    return tuple(sorted(color_to_active_count.items())), tuple(sorted(mixed_edges))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    object_specs = _choose_object_specs()

    placements = _search_placements(h, w, object_specs)
    if placements is None:
        return None

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for placement in placements:
        _paint_cells(out, placement["diamond"], placement["diamond_color"])
        for edge_part in placement["edges"]:
            color = _EDGE_TO_COLOR[edge_part["edge_name"]]
            _paint_cells(out, edge_part["segment"], color)
            if edge_part["rays"]:
                _paint_cells(out, edge_part["rays"], color)
                ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.asarray(rs)
        cs = np.asarray(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not _variant_observability_ok(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
