# Combo variant of 80a900e0 (index 7)
# combo_spec: D4/0+D8/1
# Name: 双色虚线射线
#
# Design analysis
# ===============
# a) Concrete behavior and trace:
#    - The geometric scaffold is unchanged: each 5x5 hollow diamond may have 2-4 short
#      diagonal outer segments, and each segment still emits two outward diagonal rays from
#      its endpoints.
#    - The original task painted every traversed ray cell with the segment color.
#    - This combo changes two linked layers on the same hidden full path:
#      1. step rhythm: walk one cell at a time but only paint step 1,3,5,... away from
#         the endpoint, leaving step 2,4,6,... as checkerboard background;
#      2. paint-color rhythm: among those painted cells, alternate
#         segment-color, diamond-color, segment-color, diamond-color, ...
#    - Example for one endpoint with outward=(1,1) and full path
#      [(5,5),(6,6),(7,7),(8,8),(9,9)]:
#      original output: all five cells get outer color A;
#      combo output: (5,5)=A, (6,6)=skip, (7,7)=diamond color B, (8,8)=skip, (9,9)=A.
#      If a non-background obstacle sits at (7,7), the walk stops there before painting B;
#      the solver must therefore check occupancy on every step, not only on paint steps.
# b) Complexity calibration:
#    - If all rays were length <= 2, the color alternation would be invisible because only
#      the first painted pixel A would appear; if all visible two-color rays stopped at
#      length 3 or 4, examples would show only A-blank-B without ever returning to A. The
#      generator therefore requires enough skipped cells, enough painted diamond-color ray
#      pixels, and at least one ray long enough to display A-blank-B-blank-A explicitly.
#    - Colors are still limited to the original 2-9 non-background palette and segment
#      colors exclude all chosen diamond colors. That keeps the color vocabulary small while
#      making the A/B alternation visually crisp.
# c) Implicit-assumption audit for main():
#    - `bg_colors = top-2 colors` remains valid because generation still keeps checkerboard
#      colors 0 and 1 more frequent than any single non-background color.
#    - `_is_diamond8`, `_outer_line_info`, and the inward/outward normal inference depend
#      only on geometry, so they survive unchanged.
#    - The original assignment `out[rr, cc] = ocol` assumed every painted ray cell uses the
#      outer segment color. That assumption is explicitly broken here; the new code derives
#      paint-vs-skip from the full path index and derives A/B color from the painted-index.
#    - Because skipped cells remain background, two rays cannot be allowed to overlap even on
#      skipped coordinates; otherwise a later ray would incorrectly pass through space that
#      conceptually belongs to another ray. The generator therefore reserves the entire
#      continuous path, not just the actually painted cells.
# d) Placement legality:
#    - Generation is still output-first. For each candidate diamond placement we compute the
#      full endpoint-to-boundary path for every chosen edge, reject any overlap among diamond,
#      segment, and full ray paths, then paint only the dashed alternating subset into output.
#    - Input is obtained by restoring checkerboard colors on the painted ray cells only.
#      Unpainted skipped cells already remain background in output.
# e) Visual observability:
#    - A solver can infer the rule from examples because the short outer segment shows which
#      color must start the ray, the diamond supplies the second alternating color, and the
#      regularly spaced gaps reveal the dash cadence.
#    - The generator demands multiple skipped cells and multiple diamond-colored ray pixels,
#      so examples do not collapse into "almost original" solid rays or single-color dashes.
# f) Visual cleanliness:
#    - No new marker objects are introduced. The scene still contains only checkerboard,
#      hollow diamonds, short segments, and rays, with non-overlap enforced on full paths so
#      the alternating stripes remain readable instead of stacking in one region.
# g) Change inferability:
#    - The color rule is not an arbitrary palette remapping. The two colors come from local
#      structure that is already present in the input: the segment itself and its owning
#      diamond. Repetition across 2-4 edges per diamond gives the viewer several aligned
#      instances of the same A/B rule inside one example.
# 9) New assumptions introduced by main():
#    - Each ray restarts both rhythms independently at each endpoint: the first eligible cell
#      is painted and uses the segment color.
#    - Foreign non-background pixels must not intrude onto a ray's continuous path, because
#      the while-loop treats the first non-background cell as a hard stop.
#    - The generator enforces these assumptions by reserving full paths and rejecting any case
#      where `main(input)` fails to reproduce the constructed output.

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


def _painted_color_for_step(step_idx: int, outer_color: int, diamond_color: int) -> int:
    painted_idx = step_idx // 2
    return outer_color if painted_idx % 2 == 0 else diamond_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the same outward diagonal rays as the source task, but render each ray as a
    two-color dashed stripe instead of a solid single-color line.

    Input structure:
    - A checkerboard background using the two most common colors.
    - Hollow 8-pixel diamonds in non-background colors.
    - Short diagonal outer segments, each adjacent to exactly one diamond edge and colored
      differently from that diamond.

    Rule:
    - For each outer segment, infer the outward diagonal direction by comparing the segment to
      its neighboring diamond exactly as in the source task.
    - From each endpoint, walk cell by cell along that outward diagonal until the grid edge or
      the first non-background obstacle.
    - Only paint the 1st, 3rd, 5th, ... cells along that walk; leave the 2nd, 4th, 6th, ...
      cells as background. The painted cells alternate segment color, diamond color, segment
      color, diamond color, restarting from the segment color at each endpoint.

    Key insight:
    - Geometry still identifies which way each ray travels, but the visible output comes from
      a hidden full path with two synchronized rhythms: paint/skip on every other step, and
      A/B color alternation on the painted steps only.
    """
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
            ocol = int(g[oc[0][0], oc[0][1]])
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
                step_idx = 0
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) not in bg_colors:
                        break
                    if step_idx % 2 == 0:
                        out[rr, cc] = _painted_color_for_step(step_idx, ocol, dcol)
                    step_idx += 1
                    rr += outward[0]
                    cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
MIN_TOTAL_SKIPPED_CELLS = 4
MIN_TOTAL_DIAMOND_RAY_PIXELS = 4
MIN_LONG_RAY_COUNT = 2
MIN_ABA_RAY_COUNT = 1

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


def _split_dashed_ray(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    return path[0::4], path[2::4], path[::2]


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    skipped_count = 0
    diamond_ray_count = 0
    long_ray_count = 0
    aba_ray_count = 0
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
        ray_paths = [
            _ray_cells((h, w), segment[0], info["outward"]),
            _ray_cells((h, w), segment[-1], info["outward"]),
        ]
        full_rays = [cell for path in ray_paths for cell in path]
        part_cells = segment + full_rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)

        ray_outer_cells: list[tuple[int, int]] = []
        ray_diamond_cells: list[tuple[int, int]] = []
        painted_rays: list[tuple[int, int]] = []
        for path in ray_paths:
            outer_cells, diamond_cells, painted_cells = _split_dashed_ray(path)
            ray_outer_cells.extend(outer_cells)
            ray_diamond_cells.extend(diamond_cells)
            painted_rays.extend(painted_cells)
            skipped_count += len(path) // 2
            diamond_ray_count += len(diamond_cells)
            long_ray_count += int(len(path) >= 3)
            aba_ray_count += int(len(path) >= 5)

        edge_parts.append(
            {
                "segment": segment,
                "ray_paths": ray_paths,
                "ray_outer_cells": ray_outer_cells,
                "ray_diamond_cells": ray_diamond_cells,
                "painted_rays": painted_rays,
                "weight": len(segment) + len(ray_outer_cells),
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "skipped_count": skipped_count,
        "diamond_ray_count": diamond_ray_count,
        "long_ray_count": long_ray_count,
        "aba_ray_count": aba_ray_count,
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
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


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})

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
            counts[chosen] += int(placement["edges"][ei]["weight"])
            used_local.add(chosen)
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _variant_visibility_ok(placements: list[dict[str, object]]) -> bool:
    total_skipped = sum(int(pl["skipped_count"]) for pl in placements)
    total_diamond_ray = sum(int(pl["diamond_ray_count"]) for pl in placements)
    total_long_rays = sum(int(pl["long_ray_count"]) for pl in placements)
    total_aba_rays = sum(int(pl["aba_ray_count"]) for pl in placements)
    return (
        total_skipped >= MIN_TOTAL_SKIPPED_CELLS
        and total_diamond_ray >= MIN_TOTAL_DIAMOND_RAY_PIXELS
        and total_long_rays >= MIN_LONG_RAY_COUNT
        and total_aba_rays >= MIN_ABA_RAY_COUNT
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None or not _variant_visibility_ok(placements):
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        diamond_color = diamond_colors[pi]
        _paint_cells(out, placement["diamond"], diamond_color)
        for ei, edge_part in enumerate(placement["edges"]):
            segment_color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], segment_color)
            _paint_cells(out, edge_part["ray_outer_cells"], segment_color)
            _paint_cells(out, edge_part["ray_diamond_cells"], diamond_color)
            ray_coords.extend(edge_part["painted_rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs, cs = np.array(rs), np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
