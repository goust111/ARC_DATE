# Variant of 80a900e0
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 连续涂色：射线路径上每个背景像素都涂上外边框颜色（每个都涂，无跳过）
#         → 周期性多色模式：每条射线按 [外边框色, 对应菱形色] 周期 2 交替着色。

"""
Design analysis for the D4 variant.

spec_variant:
  unchanged_geometry:
    grid: 14-30 cols x 20-24 rows
    background: checkerboard of 0/1 with bg[r,c]=(r+c+1)%2
    phase: all non-bg pixels still lie on (r+c)%2==0 cells
    diamond:
      shape: same 8-pixel hollow diamond in 5x5 bbox
      count: 1-3
      color: each diamond keeps one distinct non-bg color
    outer_segment:
      shape: same short diagonal segment adjacent to one diamond edge
      count_per_diamond: 2-4
      color: non-bg and different from its diamond color
  changed_trajectory:
    old_rule: each ray cell gets the outer-segment color
    new_rule:
      period: 2
      cycle: [outer_segment_color, owning_diamond_color]
      start_phase: first emitted ray cell uses outer_segment_color
      scope: each endpoint ray restarts the same cycle independently
      stopping: still stop at first non-bg obstacle or grid boundary

behavior_trace:
  concrete_picture: |
    Input still shows only the checkerboard, diamonds, and short outer diagonal segments.
    Output still adds two diagonal rays per segment, but the rays now visibly alternate
    between the segment color and the color of the diamond that emitted that segment.
    Example: a teal segment around a red diamond produces ray colors teal, red, teal, red...
    from each endpoint outward.
  local_step_trace: |
    Original logic for one endpoint:
      step1 bg -> paint outer color
      step2 bg -> paint outer color
      step3 bg -> paint outer color
    Variant logic:
      step1 bg -> paint outer color
      step2 bg -> paint diamond color
      step3 bg -> paint outer color
      step4 bg -> paint diamond color
    The geometry, direction inference, and collision behavior stay the same; only the color
    chosen at each step depends on the step index parity.

complexity_calibration:
  rationale: |
    The period is fixed at 2 rather than letting each sample choose a different length, which
    keeps the rule inferable from 2-4 examples. Using the two already-visible local colors
    (segment color and adjacent diamond color) avoids hidden metadata while still making the
    output clearly different from the original single-color rays.
  anti_simplification: |
    A single short ray could look identical to the original task, so the generator rejects
    low-evidence boards and keeps only instances with multiple rays long enough to show at
    least one color flip.

implicit_assumption_review:
  main_related_lines:
    - Background detection via top-2 colors still holds because the board remains checkerboard.
    - Diamond detection by 8-connected same-color component still holds because diamonds are
      unchanged and ray cells are absent from the input.
    - Adjacent non-bg extraction around a diamond still isolates only the short outer segments
      because generated inputs never include ray pixels.
    - Outward normal inference from shifted overlap with the diamond still holds because segment
      geometry is unchanged.
    - The original write rule "always assign outer color" depended on color invariance along
      the ray; this is the exact assumption replaced here with parity-based color selection.
    - Using `out` during painting remains safe only because generator placement forbids ray-ray,
      ray-segment, and ray-diamond overlap; otherwise newly painted non-bg cells could become
      accidental blockers for later rays.

placement_legality:
  guarantee: |
    Generator still constructs full rays output-first and uses their complete occupied footprint
    during backtracking. This ensures no overlap among diamonds, segments, or any future ray cell.
  added_constraints:
    - Keep segment and diamond colors distinct so the alternating pattern is visible.
    - Reject boards with too little periodic evidence (too few long rays / too few diamond-colored
      ray pixels), because those would be visually close to the original task.

visual_observability:
  why_humans_can_infer: |
    The two colors used on a ray are already paired in the input: every segment sits next to its
    owning diamond, and the output reuses exactly those two colors in an alternating sequence.
    Multiple segments per board and multiple boards repeat the same local relation, which gives
    enough support to infer "the ray alternates between the segment and diamond colors".

visual_cleanliness:
  note: |
    No extra marks are introduced; only the color of existing ray trajectories changes. Since
    placements remain non-overlapping, the checkerboard and object boundaries stay legible.

color_change_inferability:
  rule: |
    The color rule depends only on a visible adjacency relation: ray belongs to a specific outer
    segment, and that segment belongs to a specific diamond. No ordering by numeric color id or
    hidden global palette property is used.
"""

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
    # Returns (slope_key, endpoints_sorted_by_r) or None if not diagonal collinear.
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


def _ray_cycle_colors(outer_color: int, diamond_color: int) -> tuple[int, int]:
    return int(outer_color), int(diamond_color)


def main(grid: np.ndarray) -> np.ndarray:
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
        adj = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
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
                sc = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])
            cycle = _ray_cycle_colors(ocol, dcol)

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                step = 0
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) not in bg_colors:
                        break
                    out[rr, cc] = cycle[step % len(cycle)]
                    rr += outward[0]
                    cc += outward[1]
                    step += 1

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300
MIN_LONG_RAYS = 4
MIN_ALT_PIXELS = 6

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
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
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
        ray0 = _ray_cells((h, w), segment[0], info["outward"])
        ray1 = _ray_cells((h, w), segment[-1], info["outward"])
        rays = ray0 + ray1
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "ray_starts": (segment[0], segment[-1]),
                "rays": (ray0, ray1),
                "weight": len(part_cells),
                "long_ray_count": sum(len(ray) >= 3 for ray in (ray0, ray1)),
                "alt_pixels": sum(len(ray) // 2 for ray in (ray0, ray1)),
            }
        )
    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


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
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        (
            (ep["weight"], random.random(), pi, ei)
            for pi, pl in enumerate(placements)
            for ei, ep in enumerate(pl["edges"])
        ),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        dcol = diamond_colors[pi]
        candidates = [c for c in palette if c != dcol]
        chosen = min(candidates, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _paint_periodic_ray(grid: np.ndarray, ray: list[tuple[int, int]], outer_color: int, diamond_color: int) -> None:
    cycle = _ray_cycle_colors(outer_color, diamond_color)
    for step, (r, c) in enumerate(ray):
        grid[r, c] = cycle[step % len(cycle)]


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _periodic_evidence_ok(placements: list[dict[str, object]]) -> bool:
    long_rays = 0
    alt_pixels = 0
    for placement in placements:
        for edge_part in placement["edges"]:
            long_rays += int(edge_part["long_ray_count"])
            alt_pixels += int(edge_part["alt_pixels"])
    return long_rays >= MIN_LONG_RAYS and alt_pixels >= MIN_ALT_PIXELS


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None or not _periodic_evidence_ok(placements):
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        dcol = diamond_colors[pi]
        _paint_cells(out, placement["diamond"], dcol)
        for ei, edge_part in enumerate(placement["edges"]):
            ocol = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], ocol)
            for ray in edge_part["rays"]:
                _paint_periodic_ray(out, ray, ocol, dcol)
                ray_coords.extend(ray)

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs_arr = np.array(rs)
        cs_arr = np.array(cs)
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
