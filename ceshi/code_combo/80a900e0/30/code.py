# Combo variant of 80a900e0 (index 30)
# combo_spec: D2/2+D8/0
# Name: 顶点菱形色射线
#
# Design analysis
# ===============
# fused_spec:
#   background:
#     unchanged: 0/1 checkerboard, top-left = 1, and all non-bg structure stays on one phase
#   diamond:
#     unchanged geometry: 8-pixel hollow diamond in a 5x5 bbox
#   outer_segment:
#     unchanged appearance: a clipped 2-3 pixel diagonal guide just outside one diamond edge
#     changed role:
#       - it no longer provides the emission endpoints
#       - it no longer provides the ray color
#       - it only identifies which diamond vertex is activated and which outward diagonal is used
#   fused_rule:
#     clockwise edge-to-vertex activation is fixed and low-entropy:
#       TR guide -> top vertex -> NE ray
#       BR guide -> right vertex -> SE ray
#       BL guide -> bottom vertex -> SW ray
#       TL guide -> left vertex -> NW ray
#     every such ray is painted with the owning diamond color, not the guide color
#
# behavior_trace:
#   original_local_example:
#     a red diamond with a green TR guide produced two green NE rays from the guide's endpoints
#   fused_local_example:
#     the same green TR guide now only marks "activate the top vertex"
#     the ray starts from the red diamond's top tip, steps NE, skips any cells of that same guide,
#     then paints the remaining diagonal background cells in red until blocked or until the boundary
#   visible_difference:
#     geometry changes from two endpoint rays to one tip-anchored ray, and color changes from guide-colored
#     to diamond-colored; both changes are simultaneously visible on the same object
#
# complexity_calibration:
#   - Only four edge cases exist, so the cue->vertex mapping is learnable from 2-4 ARC examples.
#   - Guide colors are chosen from a palette excluding all diamond colors, so "ray color comes from the
#     diamond" is visually explicit.
#   - The generator requires visible ray cells and rejects samples where the fused output accidentally
#     matches the original task on the same input.
#
# implicit_assumption_audit:
#   - The original solver assumed guide endpoints are the ray sources; that is removed.
#   - The original solver wrote the guide color onto rays; here the correct source color is the diamond color.
#   - Vertex-fired rays would immediately collide with their own guide if we kept the original "stop on first
#     non-bg cell" behavior, so the solver must skip cells belonging to the same guide before painting.
#   - Mapping a guide to a source tip now depends on canonical diamond vertices from the 5x5 bbox; the
#     generator preserves that exact template.
#
# placement_legality:
#   - Generation remains output-first: place diamonds and guides, derive the fused rays, then erase those
#     rays back to checkerboard to obtain the input.
#   - Candidate placements are rejected if any diamond, guide, or ray overlaps another object or ray.
#   - A halo spacing constraint keeps separate objects from becoming visually tangled.
#
# visual_observability:
#   - Around one diamond, multiple differently colored guides can all lead to rays of the same diamond color.
#   - Each guide yields exactly one ray from a tip rather than two from guide endpoints, so the topological
#     change is visible even without relying on color.
#
# cleanliness:
#   - No auxiliary markers are introduced.
#   - The grid only contains checkerboard background, diamond outlines, guide segments, and emitted rays.
#
# color_inference:
#   - Non-background colors stay within ARC's standard 2..9 palette.
#   - The color rule is structural rather than numeric: rays inherit the color of the diamond they belong to.
#
# new_main_assumptions:
#   - Each adjacent non-bg component around a detected diamond is a single clipped diagonal guide.
#   - Inward-shifting a guide by one diagonal step touches the vertex set consistent with its edge.
#   - Valid generated samples always contain at least one painted fused ray and differ from the original
#     endpoint-and-guide-color semantics on the same input.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
MIN_RAY_LEN = 1
MIN_TOTAL_RAY_CELLS = 4
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

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

_VERTEX_OFFSETS = {
    "T": (0, 2),
    "R": (2, 4),
    "B": (4, 2),
    "L": (2, 0),
}

_OUTWARD_TO_ACTIVE_VERTEX = {
    (-1, 1): "T",
    (1, 1): "R",
    (1, -1): "B",
    (-1, -1): "L",
}

_OUTWARD_TO_TOUCHED_VERTICES = {
    (-1, 1): {"T", "R"},
    (1, 1): {"R", "B"},
    (1, -1): {"B", "L"},
    (-1, -1): {"L", "T"},
}

_EDGE_INFO = {
    "TR": {
        "segment": ((-1, 3), (0, 4), (1, 5)),
        "outward": (-1, 1),
        "slope": "diff",
        "active_vertex": (0, 2),
    },
    "TL": {
        "segment": ((-1, 1), (0, 0), (1, -1)),
        "outward": (-1, -1),
        "slope": "sum",
        "active_vertex": (2, 0),
    },
    "BL": {
        "segment": ((3, -1), (4, 0), (5, 1)),
        "outward": (1, -1),
        "slope": "diff",
        "active_vertex": (4, 2),
    },
    "BR": {
        "segment": ((3, 5), (4, 4), (5, 3)),
        "outward": (1, 1),
        "slope": "sum",
        "active_vertex": (2, 4),
    },
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
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


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], p[1]))
        return ("rc_diff", endpoints)
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], -p[1]))
        return ("rc_sum", endpoints)
    return None


def _diamond_vertices(pts: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    r0, c0, _, _ = _bbox(pts)
    return {name: (r0 + dr, c0 + dc) for name, (dr, dc) in _VERTEX_OFFSETS.items()}


def _guide_outward(
    oc: list[tuple[int, int]],
    dset: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    info = _outer_line_info(oc)
    if info is None:
        return None
    slope_key, _ = info
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
        return None
    outward = (-best_n[0], -best_n[1])
    return best_n, outward


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


def _ray_cells_from_vertex(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: tuple[int, int],
    guide_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = source[0] + dr, source[1] + dc
    while 0 <= r < h and 0 <= c < w and (r, c) in guide_cells:
        r += dr
        c += dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _halo8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _render(
    grid: np.ndarray,
    *,
    vertex_source: bool,
    diamond_color_rays: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for dcol, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        vertices = _diamond_vertices(dpts)

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
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            info = _outer_line_info(oc)
            inward_outward = _guide_outward(oc, dset)
            if info is None or inward_outward is None:
                continue
            best_n, outward = inward_outward
            _, endpoints = info
            ray_color = dcol if diamond_color_rays else int(g[oc[0][0], oc[0][1]])

            if vertex_source:
                active_name = _OUTWARD_TO_ACTIVE_VERTEX.get(outward)
                if active_name is None:
                    continue
                shifted = {(r + best_n[0], c + best_n[1]) for r, c in oc}
                touched_vertices = {name for name, coord in vertices.items() if coord in shifted}
                if active_name not in touched_vertices:
                    continue
                if not touched_vertices.issubset(_OUTWARD_TO_TOUCHED_VERTICES[outward]):
                    continue
                starts = [vertices[active_name]]
                guide_cells = set(oc)
            else:
                starts = [endpoints[0], endpoints[-1]]
                guide_cells = set()

            for sr, sc in starts:
                rr, cc = sr + outward[0], sc + outward[1]
                while 0 <= rr < h and 0 <= cc < w and (rr, cc) in guide_cells:
                    rr += outward[0]
                    cc += outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = ray_color
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


def _original_main(grid: np.ndarray) -> np.ndarray:
    return _render(grid, vertex_source=False, diamond_color_rays=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward diagonal rays on a checkerboard, but treat the short outer diagonals as guides rather than as emitters.

    Input structure:
    - The background is a 0/1 checkerboard with top-left cell 1.
    - Each object contains an 8-pixel hollow diamond outline in one non-background color.
    - Adjacent to 2-4 diamond edges there are short diagonal guide segments in other non-background colors.

    Rule:
    - Each guide segment identifies one activated diamond vertex and one outward diagonal direction:
      TR->top->NE, BR->right->SE, BL->bottom->SW, TL->left->NW.
    - Start from that vertex, skip any cells of the same guide that lie immediately in front of it,
      then paint the remaining outward diagonal background cells with the diamond's color until a non-background
      cell or the grid boundary is reached.

    Key insight:
    - The outer guide still determines direction, but neither the start point nor the ray color comes from it;
      the ray is anchored at the diamond tip and inherits the diamond color.
    """
    return _render(grid, vertex_source=True, diamond_color_rays=True)


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
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
        guide_cells = set(segment)
        source = (r0 + info["active_vertex"][0], c0 + info["active_vertex"][1])
        rays = _ray_cells_from_vertex((h, w), source, info["outward"], guide_cells)
        if len(rays) < MIN_RAY_LEN:
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
                "source": source,
                "weight": len(part_cells),
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "halo_cells": frozenset(_halo8(all_cells, (h, w))),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p for r0, c0 in positions if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    halo_counts: Counter[tuple[int, int]] = Counter()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells) and all(halo_counts[cell] == 0 for cell in cells):
                occupied.update(cells)
                for cell in placement["halo_cells"]:
                    halo_counts[cell] += 1
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                for cell in placement["halo_cells"]:
                    halo_counts[cell] -= 1
                    if halo_counts[cell] == 0:
                        del halo_counts[cell]
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

    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for pi, placement in enumerate(placements):
        used_local: set[int] = set()
        edge_order = sorted(
            range(len(placement["edges"])),
            key=lambda ei: (placement["edges"][ei]["weight"], random.random()),
            reverse=True,
        )
        for ei in edge_order:
            options = [c for c in palette if c not in used_local] or palette
            chosen = min(options, key=lambda c: (counts[c], random.random()))
            segment_colors[pi][ei] = chosen
            counts[chosen] += placement["edges"][ei]["weight"]
            used_local.add(chosen)
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        dcolor = diamond_colors[pi]
        _paint_cells(out, placement["diamond"], dcolor)
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(out, edge_part["segment"], segment_colors[pi][ei])
            _paint_cells(out, edge_part["rays"], dcolor)
            ray_coords.extend(edge_part["rays"])

    if len(ray_coords) < MIN_TOTAL_RAY_CELLS:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.array(rs)
        cs = np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    combo_out = main(inp)
    if not np.array_equal(combo_out, out):
        return None
    if np.array_equal(_original_main(inp), combo_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
