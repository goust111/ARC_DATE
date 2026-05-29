# Combo variant of 80a900e0 (index 20)
# combo_spec: D3/1+D4/1+D6/2
# Name: 三参数编码射线

"""
Design analysis for the fused variant.

spec_variant:
  unchanged_geometry:
    grid: 14-30 cols x 20-24 rows
    background: checkerboard of 0/1 with bg[r,c]=(r+c+1)%2
    phase: all non-bg pixels still occupy the even-parity checkerboard cells
    diamond:
      shape: same 8-pixel hollow diamond in a 5x5 box
      count: 1-3
      color: each diamond still has its own non-bg color
  fused_change:
    outer_segment:
      visible_length: each outer diagonal segment now has length L in [2,5]
      placement: still parallel to one diamond edge and offset one diagonal step outward
    ray_extent:
      old_rule: rays ran until an obstacle or the grid boundary
      new_rule: each endpoint emits exactly 2*L cells, so segment length directly encodes distance
    ray_coloring:
      cycle: [outer_segment_color, owning_diamond_color]
      restart: each endpoint ray restarts at phase 0
      visible_periods: because each ray has 2*L cells and the cycle length is 2, every ray shows
        exactly L full color periods

behavior_trace:
  concrete_picture: |
    Suppose a top-right outer segment has visible length L=4. In the input it appears as a 4-cell
    diagonal slash beside the diamond. In the output each endpoint emits an 8-cell ray rather than
    reaching the border. Along that ray the colors are outer, diamond, outer, diamond, outer,
    diamond, outer, diamond. A shorter segment with L=2 emits only 4 cells and therefore only two
    visible alternations. The same measured input feature controls both the geometric reach and the
    number of repeated color cycles.
  local_step_trace: |
    Original endpoint logic:
      keep walking until blocked or off-grid
      paint every visited cell with the outer color
    Fused endpoint logic for length L:
      step 1 -> outer color
      step 2 -> diamond color
      step 3 -> outer color
      step 4 -> diamond color
      ...
      stop after exactly 2*L painted cells
    The changed behavior is therefore not "periodic color only" or "fixed distance only" in
    isolation; the same measured segment length drives both.

complexity_calibration:
  rationale: |
    The length alphabet is small (2-5), so the solver only has to infer one compact encoding table.
    Using ray_length = 2*L keeps the mapping simple and visually obvious: longer segments produce
    longer rays and more alternations in lockstep. The cycle itself uses only colors already visible
    in the local diamond-segment pair, avoiding hidden metadata.
  anti_simplification: |
    The generator rejects boards that use only one segment length or only very similar lengths.
    Every accepted sample therefore contains at least two distinct length classes, so the encoding
    is visible within a single board instead of only across the dataset.

implicit_assumption_review:
  original_main_breakpoints:
    - Background detection from the two most frequent colors still holds because the checkerboard
      remains unchanged and the generator keeps both bg colors dominant.
    - Diamond detection by same-color 8-connected component still holds because diamonds are
      unchanged and retain unique colors.
    - The original "collect only pixels adjacent to the diamond" assumption is no longer valid:
      a 4- or 5-cell segment extends beyond the 8-neighborhood of the diamond. The solver must
      scan all non-diamond diagonal components and then assign each one back to its diamond.
    - The original "ray extends until obstacle" assumption is intentionally removed. The solver now
      computes a fixed number of steps from the measured segment length.
    - Using the evolving output as an obstacle source would create order dependence once rays gain
      diamond-colored cells. The new solver checks occupancy against the original input grid instead,
      so later rays are not blocked by earlier painted rays.

placement_legality:
  guarantee: |
    The generator stays output-first. For every candidate edge it constructs the exact segment and
    both fixed-length rays before placement is accepted. Backtracking reserves the full occupied
    footprint, so diamonds, segments, and future ray cells never overlap.
  added_constraints:
    - Full segment length must be visible; placements that would clip a 2-5 segment are rejected.
    - Full ray length 2*L must fit inside the grid for both endpoints.
    - Same-color segment components must stay separate in the input so the solver can recover each
      length unambiguously.

visual_observability:
  why_humans_can_infer: |
    A person can measure the segment directly in the input, then compare outputs: a longer segment
    produces a proportionally longer ray and also more outer/diamond alternations. Because the ray
    colors come from the visibly adjacent pair, the cyclic pattern is locally explainable without
    referring to color ids or hidden ordering.

visual_cleanliness:
  note: |
    The generator forbids ray-ray and ray-object collisions, so the checkerboard remains legible.
    The output only adds two clean diagonals per segment, and all color changes occur on those
    diagonals rather than in crowded mixed regions.

color_change_inferability:
  rule: |
    The color rule depends only on visible structure: which segment owns the ray and which diamond
    that segment belongs to. No branch depends on numeric color values, global ranking, or any
    property not present in the pixels.

post_change_solver_assumptions:
  assumptions_and_generator_support:
    - Every non-diamond non-bg component in the input is a single straight diagonal segment of
      length 2-5. The generator validates this after coloring.
    - Each segment has a unique best diamond/normal match with positive shifted-overlap score.
      Backtracking keeps enough spacing that this ownership inference is stable.
    - The fixed ray cells remain background in the input. Output-first construction reserves all ray
      footprints, so the solver's "paint only original background cells" check is satisfied.
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask


RAY_MULTIPLIER = 2
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
SEGMENT_LEN_RANGE = (2, 5)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
MIN_DISTINCT_LENGTHS = 2
MIN_LENGTH_SPAN = 2

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

_EDGE_INFO = {
    "TR": {"anchor": (0, 4), "segment_dir": (1, 1), "outward": (-1, 1), "slope": "diff"},
    "TL": {"anchor": (0, 0), "segment_dir": (1, -1), "outward": (-1, -1), "slope": "sum"},
    "BL": {"anchor": (4, 0), "segment_dir": (1, 1), "outward": (1, -1), "slope": "diff"},
    "BR": {"anchor": (4, 4), "segment_dir": (1, -1), "outward": (1, 1), "slope": "sum"},
}
_EDGE_ORDER = ("TR", "TL", "BL", "BR")


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    return int(r0), int(c0), int(r1), int(c1)


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
        ordered = sorted(pts, key=lambda p: (p[0], p[1]))
        if all(
            ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == 1
            for i in range(len(ordered) - 1)
        ):
            return ("rc_diff", ordered)
        return None
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        ordered = sorted(pts, key=lambda p: (p[0], -p[1]))
        if all(
            ordered[i + 1][0] - ordered[i][0] == 1 and ordered[i + 1][1] - ordered[i][1] == -1
            for i in range(len(ordered) - 1)
        ):
            return ("rc_sum", ordered)
        return None
    return None


def _ray_cycle_colors(outer_color: int, diamond_color: int) -> tuple[int, int]:
    return int(outer_color), int(diamond_color)


def _segment_t_values(length: int) -> list[int]:
    start = -(length // 2) + (1 if length % 2 == 0 else 0)
    return list(range(start, start + length))


def _segment_cells(r0: int, c0: int, edge: str, length: int) -> list[tuple[int, int]]:
    info = _EDGE_INFO[edge]
    ar, ac = info["anchor"]
    dr, dc = info["segment_dir"]
    return [(int(r0 + ar + t * dr), int(c0 + ac + t * dc)) for t in _segment_t_values(length)]


def _fixed_ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], steps: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, steps + 1):
        r = int(start[0] + dr * step)
        c = int(start[1] + dc * step)
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((r, c))
    return cells


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _paint_periodic_ray(grid: np.ndarray, ray: list[tuple[int, int]], outer_color: int, diamond_color: int) -> None:
    cycle = _ray_cycle_colors(outer_color, diamond_color)
    for step, (r, c) in enumerate(ray):
        grid[r, c] = cycle[step % len(cycle)]


def _match_segment_to_diamond(
    pts: list[tuple[int, int]], slope_key: str, diamonds: list[dict[str, object]]
) -> tuple[dict[str, object], tuple[int, int]] | None:
    normals = [(-1, 1), (1, -1)] if slope_key == "rc_diff" else [(-1, -1), (1, 1)]
    seg_box = _bbox(pts)

    best_key = None
    best_match = None
    for diamond in diamonds:
        dset = diamond["set"]
        dbox = diamond["bbox"]
        box_dist = abs(seg_box[0] - dbox[0]) + abs(seg_box[1] - dbox[1])
        for normal in normals:
            score = sum(((r + normal[0], c + normal[1]) in dset) for r, c in pts)
            key = (int(score), -int(box_dist))
            if best_key is None or key > best_key:
                best_key = key
                best_match = (diamond, normal)

    if best_key is None or best_key[0] <= 0 or best_match is None:
        return None
    return best_match


def _segment_specs_valid(specs: list[tuple[tuple[str, int], ...]]) -> bool:
    lengths = [int(length) for spec in specs for _, length in spec]
    if len(lengths) < 2:
        return False
    return len(set(lengths)) >= MIN_DISTINCT_LENGTHS and (max(lengths) - min(lengths)) >= MIN_LENGTH_SPAN


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, spec: tuple[tuple[str, int], ...]
) -> dict[str, object] | None:
    diamond = [(int(r0 + dr), int(c0 + dc)) for dr, dc in _DIAMOND_COORDS]
    if any(not (0 <= r < h and 0 <= c < w) for r, c in diamond):
        return None

    all_cells = set(diamond)
    edge_parts = []
    for edge, seg_len in spec:
        info = _EDGE_INFO[edge]
        segment = _segment_cells(r0, c0, edge, seg_len)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
            return None
        line_info = _outer_line_info(segment)
        if line_info is None:
            return None
        _, ordered_segment = line_info

        ray_len = int(seg_len * RAY_MULTIPLIER)
        ray0 = _fixed_ray_cells((h, w), ordered_segment[0], info["outward"], ray_len)
        ray1 = _fixed_ray_cells((h, w), ordered_segment[-1], info["outward"], ray_len)
        if ray0 is None or ray1 is None:
            return None

        part_cells = ordered_segment + ray0 + ray1
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        edge_parts.append(
            {
                "edge": edge,
                "segment_len": int(seg_len),
                "ray_len": int(ray_len),
                "segment": ordered_segment,
                "rays": (ray0, ray1),
                "weight": len(part_cells),
            }
        )

    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(h: int, w: int, specs: list[tuple[tuple[str, int], ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: (len(specs[i]), sum(length for _, length in specs[i])), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[tuple[str, int], ...], list[dict[str, object]]] = {}
    for spec in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidates = []
        for r0, c0 in positions:
            placement = _candidate_placement(h, w, r0, c0, spec)
            if placement is not None:
                candidates.append(placement)
        if not candidates:
            return None
        candidate_cache[spec] = candidates

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
            (int(edge_part["weight"]), random.random(), pi, ei)
            for pi, placement in enumerate(placements)
            for ei, edge_part in enumerate(placement["edges"])
        ),
        reverse=True,
    )

    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = int(chosen)
        counts[chosen] += int(placements[pi]["edges"][ei]["weight"])
    return segment_colors


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _input_components_clean(inp: np.ndarray) -> bool:
    bg_colors = set(_top_k_colors(inp, 2))
    non_bg = set(int(v) for v in np.unique(inp)) - bg_colors
    comps = _components8_by_color(inp, non_bg)
    diamonds = 0
    segments = 0
    for _color, pts in comps:
        if _is_diamond8(pts):
            diamonds += 1
            continue
        line_info = _outer_line_info(pts)
        if line_info is None:
            return False
        seg_len = len(pts)
        if not (SEGMENT_LEN_RANGE[0] <= seg_len <= SEGMENT_LEN_RANGE[1]):
            return False
        segments += 1
    return diamonds >= 1 and segments >= 2


def _sample_specs(diamond_count: int) -> list[tuple[tuple[str, int], ...]]:
    specs: list[tuple[tuple[str, int], ...]] = []
    for _ in range(diamond_count):
        edge_count = random.randint(*EDGES_PER_DIAMOND)
        edges = random.sample(_EDGE_ORDER, edge_count)
        spec = tuple(sorted(((edge, random.randint(*SEGMENT_LEN_RANGE)) for edge in edges), key=lambda x: _EDGE_ORDER.index(x[0])))
        specs.append(spec)
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _sample_specs(diamond_count)
    if not _segment_specs_valid(specs):
        return None

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        dcol = int(diamond_colors[pi])
        _paint_cells(out, placement["diamond"], dcol)
        for ei, edge_part in enumerate(placement["edges"]):
            ocol = int(segment_colors[pi][ei])
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

    if not _input_components_clean(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit length-coded diagonal rays from variable outer border segments around checkerboard diamonds.

    Input structure:
    - The background is still a two-color checkerboard.
    - Each non-background 8-pixel diamond outline keeps its own color.
    - Beside some diamond edges there are diagonal outer segments of visible length 2-5 in another color.

    Rule:
    - For each outer segment, infer which diamond it belongs to and measure its visible length L.
    - From each segment endpoint, fire an outward diagonal ray of exactly 2*L cells.
    - Color each ray with the repeating cycle [outer-segment color, owning-diamond color], restarting
      the cycle independently at each endpoint.

    Key insight:
    - Segment length is the shared code. The same measured L controls both how far the ray travels
      and how many alternating outer/diamond color periods appear on that ray.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)

    diamonds: list[dict[str, object]] = []
    segments: list[tuple[int, list[tuple[int, int]]]] = []
    for color, pts in comps:
        if _is_diamond8(pts):
            diamonds.append({"color": int(color), "pts": pts, "set": set(pts), "bbox": _bbox(pts)})
        else:
            segments.append((int(color), pts))

    diamonds.sort(key=lambda item: item["bbox"])
    out = g.copy()

    for outer_color, pts in segments:
        info = _outer_line_info(pts)
        if info is None or not diamonds:
            continue
        slope_key, ordered = info
        match = _match_segment_to_diamond(ordered, slope_key, diamonds)
        if match is None:
            continue

        diamond, inward_normal = match
        outward = (-int(inward_normal[0]), -int(inward_normal[1]))
        diamond_color = int(diamond["color"])
        ray_steps = len(ordered) * RAY_MULTIPLIER
        cycle = _ray_cycle_colors(outer_color, diamond_color)

        for start in (ordered[0], ordered[-1]):
            for step in range(ray_steps):
                rr = int(start[0] + outward[0] * (step + 1))
                cc = int(start[1] + outward[1] * (step + 1))
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                # D3 uses fixed-length rays: occupancy does not change the step budget.
                if int(g[rr, cc]) in bg_colors:
                    out[rr, cc] = cycle[step % len(cycle)]

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
