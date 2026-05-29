# Combo variant of 80a900e0 (index 1)
# combo_spec: D2/0+D4/0
# Name: 密集栅格射线

"""
Design analysis for the D2/0 + D4/0 combo.

a) Concrete behavior and trace
   - Base structure from the source task is preserved:
     * background is still the 0/1 checkerboard with top-left fixed to 1;
     * diamonds are still 8-pixel hollow outlines in a 5x5 box;
     * outer objects are still short diagonal segments adjacent to chosen diamond edges.
   - The combo changes two parts of the ray stage together:
     * D2: a 3-pixel outer segment emits from all 3 segment pixels, not only its two ends;
     * D4: each emitted ray walks the same outward diagonal path as before, but only paints
       path positions 1, 3, 5, ... from its own emitter.
   - Concrete trace for a TR segment
       segment = [(r-1,c+3), (r,c+4), (r+1,c+5)], outward = (-1,+1)
       original continuous endpoint-only output:
         rays from first and last segment pixels, painting every visited cell.
       combo output:
         three parallel paths are considered, one per segment pixel;
         each path paints visited cells at local indices 0, 2, 4, ... and leaves
         local indices 1, 3, 5, ... as checkerboard.
   - So the visible difference is not just "more rays" or just "dashed rays":
     the middle emitter adds an extra dashed lane, producing a denser lattice texture.

b) Complexity calibration
   - If clipped 2-pixel segments were allowed, D2 could become visually weak because
     "every pixel emits" would nearly collapse to the source task's 2 emitters.
   - To keep the combo legible, the generator only allows full 3-pixel segments.
   - If rays were too short, D4 could also become hard to infer, so the generator requires
     enough path length to create skipped cells and visible middle-lane paint.
   - Freedoms remain close to the source task: 1-3 diamonds, 2-4 occupied edges per diamond,
     same palette, same output-first construction.

c) Implicit-assumption audit for main()
   - `_top_k_colors`, `_components8_by_color`, `_is_diamond8` still work because the
     checkerboard and diamond geometry are unchanged.
   - Collecting non-background pixels adjacent to a diamond still isolates each outer segment:
     the generator keeps segments as single straight diagonal 8-components.
   - `_outer_line_info` still assumes one collinear diagonal component per outer segment;
     the generator enforces this.
   - Inward/outward normal inference by overlap with the diamond still holds because segment
     placement relative to the diamond is unchanged.
   - The source solver assumed only the two extreme segment pixels emit. D2 breaks that, so
     the emission loop must iterate over the full sorted segment.
   - The source solver also assumed "check background, then paint immediately". D4 changes
     this to "check background on every step, but paint only on alternating steps". The
     background check must still happen on skipped steps so the ray cannot jump over blockers.
   - Using `out` during painting introduces a new assumption that different valid ray paths do
     not collide before the boundary. The generator reserves full paths, including skipped
     cells, so later rays never see false blockers created by other objects.

d) Placement legality
   - Generation is output-first.
   - For each chosen edge, the generator builds the 3-pixel segment, then computes three full
     outward paths, one from each segment pixel.
   - It reserves every path cell, not only painted cells. This is necessary because skipped
     cells still belong to the ray trajectory and must stay empty for the solver.
   - Any candidate is rejected if diamond pixels, segment pixels, or any reserved path cells
     overlap another object or go out of bounds.

e) Visual observability
   - Human-visible evidence for D2: each occupied edge can show a painted middle lane that
     cannot come from endpoint-only emission.
   - Human-visible evidence for D4: each lane contains a regular paint/skip rhythm.
   - The generator therefore requires both a minimum number of skipped cells and a minimum
     number of painted cells contributed by middle emitters across the whole instance.

f) Visual cleanliness
   - Dense dashed bundles can become messy if different objects touch or intersect.
   - The generator avoids that by forbidding overlap of complete ray trajectories, not just
     colored output cells. This preserves clear parallel bundles around each diamond.

g) Color inference
   - No new color mapping is added. Each dashed bundle keeps the color of its own outer
     segment, and diamonds keep their own separate colors.
   - Because the change is geometric rather than palette-based, the rule remains inferable
     from repeated spatial patterns rather than from hidden numeric color identities.

h) New solver assumptions introduced by the combo
   - Every outer component is a straight diagonal segment of length 3.
   - The full path of each emitted ray remains unobstructed until the first intended blocker
     or the boundary.
   - There is enough path length in the sample set for both the middle lane and the dash
     rhythm to be visible.
   - `generate_input()` enforces all three assumptions explicitly.
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


def _paint_dashed_ray(
    out: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    bg_colors: set[int],
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    step_idx = 0
    while 0 <= r < h and 0 <= c < w:
        if int(out[r, c]) not in bg_colors:
            break
        if step_idx % 2 == 0:
            out[r, c] = color
        step_idx += 1
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct dense dashed ray bundles emitted from diagonal outer segments around checkerboard diamonds.

    Input structure:
    - The background is a two-color checkerboard; its two colors are the most frequent colors.
    - Non-background objects include 8-pixel hollow diamonds and short diagonal outer segments placed one step farther out from chosen diamond edges.
    - Each outer segment is a straight 3-pixel diagonal component in its own color, separate from the diamond color.

    Rule:
    - For each outer segment, infer which diagonal direction points away from the adjacent diamond.
    - Treat every pixel of the segment as an emitter: from each one, walk an outward diagonal ray until the grid edge or the first non-background obstacle.
    - Along each such ray, paint only alternating path cells, starting with the first step away from the segment pixel.

    Key insight:
    - The decisive cue is the combination of two changes: emission comes from the full segment rather than only the endpoints, and each emitted path is dashed rather than continuous.
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

            for pixel in segment_points:
                _paint_dashed_ray(out, pixel, outward, ocol, bg_colors)

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
SEGMENT_LEN = 3
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500
MIN_TOTAL_SKIPPED_CELLS = 6
MIN_TOTAL_MIDDLE_PAINTED = 3

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


def _painted_path_cells(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return path[::2]


def _segment_ray_paths(
    shape: tuple[int, int],
    segment: list[tuple[int, int]],
    direction: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    return [_ray_cells(shape, pixel, direction) for pixel in segment]


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    total_skipped = 0
    total_middle_painted = 0

    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
            return None
        if len(segment) != SEGMENT_LEN:
            return None

        segment = _sort_segment(segment, info["slope"])
        ray_paths = _segment_ray_paths((h, w), segment, info["outward"])
        if any(len(path) < 2 for path in ray_paths):
            return None

        full_rays = [cell for path in ray_paths for cell in path]
        painted_rays = [cell for path in ray_paths for cell in _painted_path_cells(path)]
        part_cells = segment + full_rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        skipped_here = sum(len(path) // 2 for path in ray_paths)
        middle_painted_here = len(_painted_path_cells(ray_paths[1]))
        total_skipped += skipped_here
        total_middle_painted += middle_painted_here
        edge_parts.append(
            {
                "segment": segment,
                "ray_paths": ray_paths,
                "painted_rays": painted_rays,
                "weight": len(segment) + len(painted_rays),
                "skipped_count": skipped_here,
                "middle_painted_count": middle_painted_here,
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "skipped_count": total_skipped,
        "middle_painted_count": total_middle_painted,
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
        (
            (ep["weight"], random.random(), pi, ei)
            for pi, pl in enumerate(placements)
            for ei, ep in enumerate(pl["edges"])
        ),
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
    total_skipped = sum(int(pl["skipped_count"]) for pl in placements)
    total_middle_painted = sum(int(pl["middle_painted_count"]) for pl in placements)
    return total_skipped >= MIN_TOTAL_SKIPPED_CELLS and total_middle_painted >= MIN_TOTAL_MIDDLE_PAINTED


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
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["painted_rays"], color)
            ray_coords.extend(edge_part["painted_rays"])

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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
