# Combo variant of 80a900e0 (index 6)
# combo_spec: D3/1+D4/1
# Name: 彩虹短射线

"""
Design analysis for the D3/1 + D4/1 combo variant.

spec_variant:
  unchanged_geometry:
    grid: 14-30 cols x 20-24 rows
    background: checkerboard of 0/1 with bg[r,c]=(r+c+1)%2
    diamond:
      shape: same 8-pixel hollow diamond in a 5x5 bbox
      count: 1-3
      color: each diamond keeps one distinct non-bg color
    outer_segment:
      shape: same diagonal segment adjacent to one diamond edge
      count_per_diamond: 2-4
      color: non-bg and different from the owning diamond
  changed_dst:
    old_rule: each endpoint ray keeps going until obstacle or boundary
    new_rule:
      encoded_budget: ray length = visible outer-segment length * 2
      effect: for the common 3-pixel segment, each endpoint gets a 6-step budget
      boundary_behavior: the ray still cannot leave the grid, so visibility is min(budget, boundary room)
  changed_trajectory:
    old_rule: every ray cell uses the outer-segment color
    new_rule:
      cycle: [outer_segment_color, owning_diamond_color]
      period: 2
      reset: each endpoint ray restarts at phase 0

behavior_trace:
  concrete_picture: |
    Suppose a 3-pixel outer segment has color A and its diamond has color B. In the original task,
    one endpoint would paint A, A, A, A, ... until the border. In this combo, that same endpoint
    paints exactly six diagonal cells (if there is room): A, B, A, B, A, B, then stops even if the
    checkerboard continues. The other endpoint does the same independently.
  changed_steps: |
    Original endpoint:
      step1 -> A
      step2 -> A
      step3 -> A
      ...
      stop only at obstacle / edge
    Combo endpoint:
      step1 -> A
      step2 -> B
      step3 -> A
      step4 -> B
      step5 -> A
      step6 -> B
      stop because the 3-pixel segment encodes budget 6
    The two dimensions interact directly: the fixed distance determines how much of the A/B cycle
    becomes visible, so short rays can end mid-pattern rather than reaching the boundary.

complexity_calibration:
  rationale: |
    The length multiplier is fixed at 2 and the color cycle is fixed at length 2. That keeps the
    rule learnable: the only numeric inference is "measure the segment, then double it", and the
    only color inference is "alternate between the two nearby object colors".
  anti_simplification: |
    The generator rejects boards that would look too close to the original task. It requires several
    rays whose boundary distance exceeds the encoded budget, ensuring the fixed-length stop is visible,
    and it also requires enough diamond-colored ray cells to make the A/B alternation obvious.
  caveat_handling: |
    Boundary-clipped 2-pixel segments in the original geometry usually have almost no outward room,
    so they are weak evidence for the distance rule. The generator therefore prefers placements with
    strong interior evidence for "stop after N" rather than relying on degenerate clipped cases.

implicit_assumption_review:
  main_related_lines:
    - Background detection via the two most common colors still holds because the checkerboard stays unchanged.
    - Diamond detection by same-color 8-connected component still holds because rays are absent from the input.
    - Adjacent non-bg extraction around each diamond still isolates only its short outer segments.
    - Outward-direction inference from segment-vs-diamond overlap still holds because segment geometry is unchanged.
    - The original loop assumed color invariance along a ray; that assumption is explicitly replaced by parity-based color choice.
    - The original loop also assumed "stop condition comes from collision / edge". Here, the operative stop is the encoded budget.
      A conservative non-bg guard is kept only to avoid overwriting malformed inputs; valid generated boards never trigger it.

placement_legality:
  guarantee: |
    The generator works output-first with the finite rays already materialized. Backtracking treats diamonds,
    segments, and every future ray cell as occupied, so no two structures overlap and no ray crosses another object.
  added_constraints:
    - Candidate edges need enough visible ray cells to show the alternating pattern.
    - Boards need several rays where the path to the boundary is longer than the encoded budget.
    - Segment and diamond colors always differ, otherwise the cycle would collapse visually.

visual_observability:
  why_humans_can_infer: |
    The two ray colors are already present locally in the input: each short segment sits beside its owning diamond.
    The output reuses exactly those two colors in an alternating diagonal trace. Multiple rays on the same board repeat
    the same relation, while the consistent short stopping distance shows that the new rays are intentionally truncated.

visual_cleanliness:
  note: |
    The combo adds no extra markers beyond the rays themselves. Because placements are non-overlapping and rays are
    deliberately short, the checkerboard remains readable and the association "this segment belongs to this diamond"
    stays visually clean.

color_change_inferability:
  rule: |
    The color sequence depends only on a visible structural relation: each ray belongs to one outer segment, and that
    segment belongs to one diamond. No numeric ordering on color ids is used.

post_change_assumption_audit:
  solver_side:
    - Every processed outer component must still be a straight diagonal segment of length at least 2.
    - Each such segment must be adjacent to exactly one detected diamond strongly enough for normal inference to score > 0.
    - The encoded budget uses the visible segment size in the input, so the generator must preserve segment geometry exactly.
    - Using `out` as the collision guard is safe only because the generator forbids ray-ray overlap; that invariant is enforced.
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


RAY_LENGTH_FACTOR = 2


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
        endpoints = sorted(pts, key=lambda p: (p[0], p[1]))
        return ("rc_diff", endpoints)
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], -p[1]))
        return ("rc_sum", endpoints)
    return None


def _ray_cycle_colors(outer_color: int, diamond_color: int) -> tuple[int, int]:
    return int(outer_color), int(diamond_color)


def _ray_budget(segment_len: int) -> int:
    return int(segment_len) * RAY_LENGTH_FACTOR


# Grow finite diagonal rays from segment endpoints, alternating between segment and diamond colors.
# Input structure:
# - A checkerboard background uses the two dominant colors.
# - One or more 8-pixel hollow diamonds appear in distinct non-background colors.
# - Short diagonal outer segments sit just outside selected diamond edges in their own colors.
#
# Rule:
# - For each outer segment, infer which diagonal direction points away from the owning diamond.
# - Each endpoint emits one ray outward, but only for a fixed number of steps:
#   `visible_segment_length * 2`.
# - Along that finite ray, colors alternate `[outer_segment_color, diamond_color]`, restarting
#   from the segment color at each endpoint.
#
# Key insight:
# - The segment still determines both direction and ownership, but now it also encodes distance.
#   The output is not "fill to the border"; it is "emit a short two-color burst whose length comes
#   from the segment size".
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
            cycle = _ray_cycle_colors(ocol, dcol)
            budget = _ray_budget(len(oc))

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                for step in range(budget):
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    if int(out[rr, cc]) not in bg_colors:
                        break
                    out[rr, cc] = cycle[step % len(cycle)]
                    rr += outward[0]
                    cc += outward[1]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

MIN_EDGE_VISIBLE_RAY_PIXELS = 4
MIN_EDGE_LONG_RAY = 3
MIN_TRUNCATED_RAYS = 4
MIN_ALT_PIXELS = 6
MIN_LONG_RAYS = 4

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


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], max_steps: int
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    for _ in range(max_steps):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _boundary_distance(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    return len(_ray_cells(shape, start, direction, 10 ** 9))


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
        budget = _ray_budget(len(segment))
        ray0 = _ray_cells((h, w), segment[0], info["outward"], budget)
        ray1 = _ray_cells((h, w), segment[-1], info["outward"], budget)
        total_visible = len(ray0) + len(ray1)
        if total_visible < MIN_EDGE_VISIBLE_RAY_PIXELS or max(len(ray0), len(ray1)) < MIN_EDGE_LONG_RAY:
            return None
        part_cells = segment + ray0 + ray1
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        dist0 = _boundary_distance((h, w), segment[0], info["outward"])
        dist1 = _boundary_distance((h, w), segment[-1], info["outward"])
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": (ray0, ray1),
                "weight": len(part_cells),
                "segment_len": len(segment),
                "ray_budget": budget,
                "visible_lengths": (len(ray0), len(ray1)),
                "truncated_by_budget_count": int(dist0 > budget) + int(dist1 > budget),
                "alt_pixels": (len(ray0) // 2) + (len(ray1) // 2),
                "long_rays": int(len(ray0) >= 3) + int(len(ray1) >= 3),
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
        cands = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]
        random.shuffle(cands)
        candidate_cache[edges] = cands

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
            (edge_part["weight"], random.random(), pi, ei)
            for pi, placement in enumerate(placements)
            for ei, edge_part in enumerate(placement["edges"])
        ),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
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


def _fusion_evidence_ok(placements: list[dict[str, object]]) -> bool:
    truncated_rays = 0
    alt_pixels = 0
    long_rays = 0
    for placement in placements:
        for edge_part in placement["edges"]:
            truncated_rays += int(edge_part["truncated_by_budget_count"])
            alt_pixels += int(edge_part["alt_pixels"])
            long_rays += int(edge_part["long_rays"])
    return truncated_rays >= MIN_TRUNCATED_RAYS and alt_pixels >= MIN_ALT_PIXELS and long_rays >= MIN_LONG_RAYS


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None or not _fusion_evidence_ok(placements):
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
