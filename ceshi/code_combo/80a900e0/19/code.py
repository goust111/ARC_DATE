# Combo variant of 80a900e0 (index 19)
# combo_spec: D2/0+D3/1+D4/0
# Name: 密集短虚线簇

"""
Design analysis
===============

a) Concrete behavior and trace
   - Preserved base structure:
     * background is still a 0/1 checkerboard with top-left = 1;
     * diamonds are still 8-pixel hollow outlines in 5x5 boxes;
     * each occupied diamond edge still has a same-colored outer diagonal segment.
   - Fused behavior:
     * D2/0: every segment pixel is now an emitter, so a full 3-pixel segment emits
       3 parallel rays instead of only 2 endpoint rays;
     * D3/1: ray extent is no longer "until boundary", but exactly
       `visible_segment_length * RAY_LENGTH_FACTOR` steps;
     * D4/0: along each fixed-length path, only every other step is painted.
   - Concrete trace for a TR edge:
     * segment = [(r-1,c+3), (r,c+4), (r+1,c+5)], outward = (-1,+1);
     * original task: emit only from the first and third pixels, and keep painting
       all the way to the edge;
     * fused task here: emit from all three segment pixels, walk exactly 6 steps from
       each emitter, and paint steps 1,3,5 while leaving steps 2,4,6 as checkerboard.
     * Visually this becomes a compact 3-ray dashed bundle rather than two long solids.

b) Complexity calibration
   - Allowing clipped 2-pixel segments would make D2 partially collapse back to the
     original "two rays" behavior, so the generator only places full 3-pixel segments.
   - Fixing segment length to 3 does simplify the distance code, but it makes the fused
     motif stable and readable: each edge consistently becomes a dense short dashed cluster.
   - To keep D3 visible, the generator requires at least one spare background step beyond
     every fixed-length path, so the bundle stops in open space instead of coinciding with
     the boundary.
   - To keep examples feasible, sampling is biased toward 1-2 diamonds and 2-3 occupied
     edges, while still allowing the full 1-3 / 2-4 spec ranges.

c) Implicit-assumption audit for main()
   - `_top_k_colors`, `_components8_by_color`, `_is_diamond8`, adjacency harvesting, and
     diagonal-line detection still hold because the input objects themselves are unchanged.
   - The original endpoint loop is invalid under D2: emitters must be the full sorted
     segment, not only `endpoints[0]` and `endpoints[-1]`.
   - The original boundary-driven while-loop is invalid under D3: the solver must measure
     the segment length first and then cap each ray to a fixed number of steps.
   - Under D4, every traversed step must still check for unexpected non-background pixels
     before deciding whether to paint or skip; otherwise a hidden obstacle on a skipped
     step would be crossed incorrectly.
   - Because the solver now uses segment length as a control value, it implicitly assumes
     each outer component is a single straight diagonal segment whose visible length is the
     intended code. The generator guarantees this by construction.

d) Placement legality
   - The generator remains output-first.
   - For each chosen edge, it builds the 3-pixel segment, then all three full 6-step ray
     corridors, and reserves every corridor cell, including skipped cells.
   - Candidate placements are rejected if any corridor cell overlaps another segment,
     diamond, or ray corridor, or if any emitter lacks the required post-path boundary
     margin.
   - This guarantees that `main()` never needs to resolve ambiguous collisions and that the
     fixed-length interpretation is geometrically realizable.

e) Visual observability
   - A solver should be able to see three coupled regularities:
     * per occupied edge there are 3 rays, not 2;
     * the rays are short and stop well before the boundary;
     * each ray is dashed with a paint/skip cadence.
   - These three signals reinforce each other on the same local motif, which is easier to
     infer from 2-4 ARC examples than three unrelated global changes.

f) Visual cleanliness
   - Dense bundles can clutter the board, so the generator keeps the original no-overlap
     discipline and additionally enforces empty slack beyond each bundle.
   - The result is a compact, separated cluster near each occupied diamond edge rather than
     long diagonals tangling across the whole grid.

g) Color inference
   - No new color mapping is introduced. Each bundle inherits the outer segment's color.
   - Because geometry, not palette identity, changes, the rule remains inferable without
     relying on color-value semantics.

h) New solver assumptions introduced by the fusion
   - Every usable outer component must be a diagonal segment of full visible length 3.
   - Every emitter on a segment uses the same outward direction and the same step budget.
   - Fixed-length paths are assumed obstacle-free in valid tasks; the solver still breaks
     defensively on unexpected non-background pixels, and the generator guarantees those
     breaks are never needed in produced examples.
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
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _steps_to_boundary(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    steps = 0
    while 0 <= r < h and 0 <= c < w:
        steps += 1
        r += dr
        c += dc
    return steps


def _paint_fixed_dash_ray(
    out: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    bg_colors: set[int],
    max_steps: int,
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start
    for step_idx in range(max_steps):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(out[r, c]) not in bg_colors:
            break
        if step_idx % 2 == 0:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build compact dashed ray bundles from the outer diagonal segments around each diamond.

    Input structure:
    - A 0/1 checkerboard background where the top-left cell is 1.
    - One to three non-background hollow diamonds, each drawn as the same 8-pixel outline as
      in the source task.
    - Full 3-pixel diagonal outer segments placed one step outside selected diamond edges.

    Rule:
    - For each outer segment, infer its outward diagonal direction from the adjacent diamond.
    - Let m be the visible segment length. Every segment pixel emits one ray in that outward
      direction, each ray runs for exactly `2 * m` steps, and only the 1st, 3rd, 5th, ...
      traversed cells are painted with the segment color.

    Key insight:
    - The segment is both the emitter set and the length code: a 3-pixel segment creates
      three parallel short dashed rays, and the rays stop after a fixed local distance rather
      than continuing to the boundary.
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

    for _, dpts in sorted(diamonds, key=bbox_key):
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
            slope_key, segment_points = info

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
            max_steps = len(segment_points) * RAY_LENGTH_FACTOR

            for pixel in segment_points:
                _paint_fixed_dash_ray(out, pixel, outward, ocol, bg_colors, max_steps)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_CHOICES = (1, 2, 3)
DIAMOND_COUNT_WEIGHTS = (4, 3, 1)
EDGES_PER_DIAMOND_CHOICES = (2, 3, 4)
EDGES_PER_DIAMOND_WEIGHTS = (4, 3, 1)
SEGMENT_LEN = 3
RAY_LENGTH_FACTOR = 2
RAY_STEPS = SEGMENT_LEN * RAY_LENGTH_FACTOR
MIN_POST_PATH_MARGIN = 1
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500
MIN_TOTAL_PAINTED_RAY_CELLS = 9
MIN_TOTAL_BOUNDARY_SLACK = 3

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


def _fixed_ray_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    for _ in range(steps):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((r, c))
    return cells


def _painted_path_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path[::2]


def _segment_bundle(shape: tuple[int, int], segment: list[tuple[int, int]], direction: tuple[int, int]):
    ray_paths: list[list[tuple[int, int]]] = []
    painted_rays: list[tuple[int, int]] = []
    full_rays: list[tuple[int, int]] = []
    total_boundary_slack = 0
    for pixel in segment:
        available = _steps_to_boundary(shape, pixel, direction)
        if available < RAY_STEPS + MIN_POST_PATH_MARGIN:
            return None
        path = _fixed_ray_path(shape, pixel, direction, RAY_STEPS)
        if path is None:
            return None
        ray_paths.append(path)
        painted_rays.extend(_painted_path_cells(path))
        full_rays.extend(path)
        total_boundary_slack += available - RAY_STEPS
    return {
        "ray_paths": ray_paths,
        "painted_rays": painted_rays,
        "full_rays": full_rays,
        "boundary_slack": total_boundary_slack,
    }


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    total_boundary_slack = 0
    total_painted_rays = 0
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
            return None
        segment = _sort_segment(segment, info["slope"])
        if len(segment) != SEGMENT_LEN:
            return None

        bundle = _segment_bundle((h, w), segment, info["outward"])
        if bundle is None:
            return None

        part_cells = segment + bundle["full_rays"]
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        total_boundary_slack += int(bundle["boundary_slack"])
        total_painted_rays += len(bundle["painted_rays"])
        edge_parts.append(
            {
                "segment": segment,
                "ray_paths": bundle["ray_paths"],
                "painted_rays": bundle["painted_rays"],
                "full_rays": bundle["full_rays"],
                "weight": len(segment) + len(bundle["painted_rays"]),
                "boundary_slack": int(bundle["boundary_slack"]),
            }
        )
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "painted_ray_count": total_painted_rays,
        "boundary_slack": total_boundary_slack,
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
        random.shuffle(candidate_cache[edges])

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
        ((ep["weight"], random.random(), pi, ei)
         for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
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


def _variant_visibility_ok(placements: list[dict[str, object]]) -> bool:
    total_painted = sum(int(pl["painted_ray_count"]) for pl in placements)
    total_slack = sum(int(pl["boundary_slack"]) for pl in placements)
    return total_painted >= MIN_TOTAL_PAINTED_RAY_CELLS and total_slack >= MIN_TOTAL_BOUNDARY_SLACK


def _sample_specs() -> list[tuple[str, ...]]:
    diamond_count = random.choices(DIAMOND_COUNT_CHOICES, weights=DIAMOND_COUNT_WEIGHTS, k=1)[0]
    return [
        tuple(sorted(random.sample(
            _EDGE_NAMES,
            random.choices(EDGES_PER_DIAMOND_CHOICES, weights=EDGES_PER_DIAMOND_WEIGHTS, k=1)[0],
        )))
        for _ in range(diamond_count)
    ]


def _gen_attempt(return_meta: bool = False):
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    specs = _sample_specs()

    placements = _search_placements(h, w, specs)
    if placements is None or not _variant_visibility_ok(placements):
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    segment_count = 0
    total_emitters = 0
    total_boundary_slack = 0
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["painted_rays"], color)
            ray_coords.extend(edge_part["painted_rays"])
            segment_count += 1
            total_emitters += len(edge_part["segment"])
            total_boundary_slack += int(edge_part["boundary_slack"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs_arr = np.array(rs, dtype=int)
        cs_arr = np.array(cs, dtype=int)
        inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

    if not np.array_equal(main(inp), out):
        return None

    if not return_meta:
        return inp

    return inp, {
        "shape": (h, w),
        "specs": specs,
        "segment_count": segment_count,
        "emitters_per_segment": SEGMENT_LEN,
        "total_emitters": total_emitters,
        "ray_steps": RAY_STEPS,
        "painted_cells_per_ray": len(_painted_path_cells([(0, 0)] * RAY_STEPS)),
        "painted_ray_cells": len(ray_coords),
        "boundary_slack": total_boundary_slack,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
