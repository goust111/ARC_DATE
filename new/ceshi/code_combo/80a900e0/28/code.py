# Combo variant of 80a900e0 (index 28)
# combo_spec: D5/2+D7/1
# Name: 大菱形全辐射
#
# Design analysis
# a) concretization + behavior trace:
#    - Base task: a 5x5 hollow diamond may have only some outer diagonal segments; each segment emits 2 outward rays.
#    - Fused task: the source diamond size is variable, but only at the two visually distinct endpoint sizes from D5/2:
#      3x3 and 7x7. A 3x3 outline has 4 pixels and 2-cell outer segments; a 7x7 outline has 12 pixels and 4-cell
#      outer segments. D7/1 then forces all 4 edges to carry outer segments, so every diamond emits 8 rays.
#    - Concrete trace for a size-7 top-right edge: the edge-side pixels are [(0,3),(1,4),(2,5),(3,6)] and the
#      outer segment is [(-1,4),(0,5),(1,6),(2,7)] relative to the diamond anchor. Rays start from both segment
#      endpoints and travel in direction (-1,+1). The same construction happens on TL/BL/BR, producing the symmetric
#      cross-like radiation pattern described in the combo.
#    - Difference from the base task: the solver must infer the diamond size before it can infer the expected segment
#      lengths, and it must handle the denser "4 segments -> 8 rays" layout rather than a sparse subset of edges.
# b) complexity calibration:
#    - I keep the size domain small: only 3 and 7. This matches the provided D5/2 endpoint examples and keeps the
#      parameter inference learnable from 2-4 ARC examples. Adding 5 would increase ambiguity without adding much
#      visual signal.
#    - Every diamond uses all 4 edges, so the only varying structural degree of freedom is size. This makes the fused
#      change visually obvious instead of hiding it behind many independent per-edge choices.
# c) hidden assumption audit:
#    - Original `_is_diamond8` assumed len==8 and bbox span==4; that breaks immediately for 3x3 and 7x7 diamonds.
#    - Original generator assumed the same 3-pixel outer segment everywhere; here segment geometry must be rebuilt
#      from size because 3x3 uses length 2 and 7x7 uses length 4.
#    - Original solver grouped adjacent non-bg cells without caring about color reuse. For the fusion variant, the
#      generator gives each diamond a unique outer color so 4 full-coverage segments stay easy to separate.
#    - Full radiation creates much larger occupied footprints; placement must now guard not only against overlap, but
#      also against foreign non-bg cells entering the 8-neighborhood halo around another diamond, otherwise `main()`
#      could misread them as that diamond's adjacent outer markers.
# d) placement legality:
#    - Generation is output-first. For each candidate placement, I build the diamond, its 4 full outer segments, and
#      all 8 rays to the border, rejecting any candidate with self-overlap, out-of-bounds segment cells, or short
#      clipped segments.
#    - Multi-object placement uses both occupied cells and diamond halos, so different objects do not overlap and do
#      not place foreign rays/segments next to another diamond.
# e) visual observability:
#    - Size is directly visible in the input: a 3x3 diamond has 4 pixels while a 7x7 diamond has 12 pixels.
#    - Full coverage is also directly visible in the input because all 4 sides carry outer segments, and the output
#      amplifies that into a symmetric 8-ray burst.
# f) visual cleanliness:
#    - Each diamond uses exactly 2 colors: one unique outline color and one unique outer/ray color. This avoids
#      clutter from per-edge recoloring and makes each radiating structure read as a coherent object.
#    - Every sample contains one large diamond and one small diamond. This makes the size-variation rule visible
#      inside each individual ARC example while keeping the board readable despite the long rays of the large source.
# g) change inferability:
#    - No hidden color mapping is introduced. The outer-segment color is simply propagated along the rays.
#    - The fusion signal is measured after implementation by checking that generated boards always contain full
#      4-segment diamonds and that every sample visibly mixes the two allowed non-5 sizes by construction.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


def _checkerboard_value(r: int, c: int) -> int:
    return (r + c + 1) % 2


def _diamond_outline_offsets(size: int) -> tuple[tuple[int, int], ...]:
    radius = (size - 1) // 2
    center = radius
    pts = []
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) == radius:
                pts.append((r, c))
    return tuple(pts)


DIAMOND_SIZES = (3, 7)
_DIAMOND_OFFSETS = {size: _diamond_outline_offsets(size) for size in DIAMOND_SIZES}
_EDGE_NAMES = ("TR", "TL", "BL", "BR")


def _build_edge_info(size: int) -> dict[str, dict[str, object]]:
    radius = (size - 1) // 2
    return {
        "TR": {
            "segment": tuple((i - 1, radius + i + 1) for i in range(radius + 1)),
            "outward": (-1, 1),
            "slope": "diff",
        },
        "TL": {
            "segment": tuple((i - 1, radius - i - 1) for i in range(radius + 1)),
            "outward": (-1, -1),
            "slope": "sum",
        },
        "BL": {
            "segment": tuple((radius + i + 1, i - 1) for i in range(radius + 1)),
            "outward": (1, -1),
            "slope": "diff",
        },
        "BR": {
            "segment": tuple((radius + i + 1, 2 * radius - i + 1) for i in range(radius + 1)),
            "outward": (1, 1),
            "slope": "sum",
        },
    }


_EDGE_INFO = {size: _build_edge_info(size) for size in DIAMOND_SIZES}


def _diamond_size(comp: list[tuple[int, int]]) -> int | None:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r not in _DIAMOND_OFFSETS:
        return None
    norm = {(r - r0, c - c0) for r, c in comp}
    if len(comp) != len(_DIAMOND_OFFSETS[size_r]):
        return None
    return size_r if norm == set(_DIAMOND_OFFSETS[size_r]) else None


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend outward diagonal rays from full sets of outer-border segments around variable-size diamonds on a checkerboard.

    Input structure:
    - Checkerboard background in the two most frequent colors.
    - Hollow diamond outlines in non-background colors, where each diamond is either a small 3x3 outline or a large 7x7 outline.
    - Every one of the diamond's 4 sides has an adjacent outer diagonal segment in another non-background color.

    Rule:
    - For each outer segment, infer which diagonal normal points back into the neighboring diamond.
    - Reverse that normal to get the outward direction, then emit a diagonal ray from each segment endpoint, coloring only
      background cells with the segment color until the ray hits the border or another non-background pixel.

    Key insight:
    - The size of the diamond changes the expected segment length, but the ray logic is unchanged: each of the 4 outer
      border segments contributes exactly 2 endpoint-launched rays, yielding 8 rays per diamond.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]], int]] = []
    for col, pts in comps:
        size = _diamond_size(pts)
        if size is not None:
            diamonds.append((col, pts, size))

    out = g.copy()

    def bbox_key(item):
        _, pts, _ = item
        return _bbox(pts)

    for _, dpts, _ in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj_by_color: dict[int, set[tuple[int, int]]] = {}
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj_by_color.setdefault(v, set()).add((rr, cc))

        for ocol, adj_cells in adj_by_color.items():
            adj_mask = np.zeros((h, w), dtype=bool)
            for r, c in adj_cells:
                adj_mask[r, c] = True
            outer_comps = _cc8(adj_mask)

            for oc in outer_comps:
                info = _outer_line_info(oc)
                if info is None:
                    continue
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

                for pr, pc in (endpoints[0], endpoints[-1]):
                    rr, cc = pr + outward[0], pc + outward[1]
                    while 0 <= rr < h and 0 <= cc < w:
                        if int(out[rr, cc]) in bg_colors:
                            out[rr, cc] = ocol
                        else:
                            break
                        rr += outward[0]
                        cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (20, 30)
DIAMOND_COUNT_CHOICES = (2,)
DIAMOND_COUNT_WEIGHTS = (1,)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 600


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


def _diamond_halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                halo.add((rr, cc))
    halo.difference_update(cells)
    return frozenset(halo)


def _candidate_placement(h: int, w: int, r0: int, c0: int, size: int) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_OFFSETS[size]]
    all_cells = set(diamond)
    edge_parts = []
    ray_coords: list[tuple[int, int]] = []

    for edge in _EDGE_NAMES:
        info = _EDGE_INFO[size][edge]
        segment = []
        for dr, dc in info["segment"]:
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            segment.append((rr, cc))
        segment = _sort_segment(segment, info["slope"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"edge": edge, "segment": segment, "rays": rays})
        ray_coords.extend(rays)

    return {
        "size": size,
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "diamond_halo": _diamond_halo(diamond, (h, w)),
        "ray_coords": tuple(ray_coords),
        "weight": len(all_cells),
    }


def _sample_sizes(diamond_count: int) -> list[int]:
    if diamond_count == 2:
        sizes = [7, 3]
        random.shuffle(sizes)
        return sizes
    sizes = [7, 3, 3]
    random.shuffle(sizes)
    return sizes


def _search_placements(h: int, w: int, sizes: list[int]) -> list[dict[str, object]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    candidate_cache: dict[int, list[dict[str, object]]] = {}

    for size in set(sizes):
        top_lefts = [
            (r, c)
            for r in range(h - size + 1)
            for c in range(w - size + 1)
        ]
        random.shuffle(top_lefts)
        candidates = [
            placement
            for r0, c0 in top_lefts
            if (placement := _candidate_placement(h, w, r0, c0, size)) is not None
        ]
        candidates.sort(key=lambda p: (p["weight"], random.random()), reverse=True)
        candidate_cache[size] = candidates

    occupied_all: set[tuple[int, int]] = set()
    occupied_halo_counts: Counter[tuple[int, int]] = Counter()
    placed: list[dict[str, object] | None] = [None] * len(sizes)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        for placement in candidate_cache[sizes[idx]]:
            all_cells = placement["all_cells"]
            diamond_halo = placement["diamond_halo"]
            if not all_cells.isdisjoint(occupied_all):
                continue
            if any(cell in occupied_halo_counts for cell in all_cells):
                continue
            if not diamond_halo.isdisjoint(occupied_all):
                continue

            occupied_all.update(all_cells)
            occupied_halo_counts.update(diamond_halo)
            placed[idx] = placement
            if backtrack(i + 1):
                return True
            occupied_all.difference_update(all_cells)
            for cell in diamond_halo:
                occupied_halo_counts[cell] -= 1
                if occupied_halo_counts[cell] == 0:
                    del occupied_halo_counts[cell]
            placed[idx] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_color_pairs(count: int) -> tuple[list[int], list[int]]:
    palette = random.sample(list(COLOR_RANGE), 2 * count)
    diamond_colors = palette[:count]
    outer_colors = palette[count:]
    return diamond_colors, outer_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.choices(DIAMOND_COUNT_CHOICES, weights=DIAMOND_COUNT_WEIGHTS, k=1)[0]
    sizes = _sample_sizes(diamond_count)

    placements = _search_placements(h, w, sizes)
    if placements is None:
        return None

    diamond_colors, outer_colors = _assign_color_pairs(len(placements))

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for i, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[i])
        for edge_part in placement["edges"]:
            _paint_cells(out, edge_part["segment"], outer_colors[i])
            _paint_cells(out, edge_part["rays"], outer_colors[i])
        ray_coords.extend(placement["ray_coords"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    for r, c in ray_coords:
        inp[r, c] = _checkerboard_value(r, c)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
