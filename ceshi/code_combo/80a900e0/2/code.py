# Combo variant of 80a900e0 (index 2)
# combo_spec: D2/0+D3/1
# Name: 密集短射线

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Base scene stays recognizable: a 0/1 checkerboard, 1-3 hollow 8-pixel diamonds,
#      and short diagonal outer segments adjacent to selected diamond edges.
#    - The fused change has two coupled parts. First, every visible segment pixel emits an
#      outward ray instead of only the two segment endpoints. Second, those rays no longer
#      run until the boundary; each ray lasts exactly `visible_segment_length * 2` steps.
#    - Small trace: suppose a TR outer segment is the 3-pixel diagonal
#      [(r-1,c+3), (r,c+4), (r+1,c+5)] with outward direction (-1,+1).
#      Original task: only the first and third pixels emit, and both rays continue until
#      blocked or until they leave the grid. Combo task: all three segment pixels emit, and
#      each ray paints exactly 6 cells. This creates a dense but clipped 3-lane bundle.
#    - For a 2-pixel visible segment, the same rule emits 2 rays of length 4. The segment
#      length therefore controls both how many parallel rays exist and how far they travel.
#
# b) Complexity calibration:
#    - If every segment stayed length 3, the distance code would collapse to one fixed ray
#      length and D3 would become invisible. If length-2 segments appeared only by boundary
#      clipping, their outward rays would immediately hit the edge and the code would again
#      be hard to read.
#    - The generator therefore uses only two visible lengths, 2 and 3, away from the grid
#      boundary. A 2-pixel segment is a contiguous subsegment of the canonical 3-pixel outer
#      slot, so it still looks like the same object family rather than a new symbol system.
#    - Each generated grid contains both lengths. That guarantees one visible D3 contrast
#      (4-step vs 6-step bundles) and at least one 3-pixel segment where D2 is visibly
#      different from the original endpoint-only behavior.
#
# c) Implicit-assumption audit for main():
#    - Background inference from the two most frequent colors still holds because the
#      generator keeps ray bundles short enough that colors 0 and 1 remain dominant.
#    - Diamond detection via 8-connected size-8 components is unchanged.
#    - `_outer_line_info` only assumes a straight diagonal component of length >= 2, so it
#      still works for both visible lengths 2 and 3.
#    - The original endpoint loop encoded a now-invalid assumption: only the extreme pixels
#      matter. The combo variant must iterate over every detected segment pixel.
#    - The original stopping rule also encoded an invalid assumption: hitting a non-bg pixel
#      or the boundary determines ray length. Here the segment length determines ray length,
#      so the solver must step exactly `len(segment) * 2` times, truncating only if the
#      grid boundary is reached early. The generator forbids such early truncation.
#
# d) Placement legality:
#    - Generation remains output-first. For each candidate edge we first choose its visible
#      segment length, then derive the full dense short ray bundle implied by the combo rule.
#    - A candidate is rejected if any diamond cell, segment cell, or ray cell overlaps any
#      previously reserved output cell. Cross-diamond input objects also get an 8-neighbor
#      buffer so one diamond cannot mistake another object's pixels for its own outer segment.
#    - The generator additionally rejects placements whose encoded rays would be boundary-
#      truncated, ensuring that the intended 4-step / 6-step code is actually visible.
#
# e) Visual observability:
#    - A human sees two repeated regularities: 3-pixel segments produce denser 3-lane bundles
#      than the original task, and longer segments also produce longer bundles.
#    - Because each puzzle contains both visible lengths, the length code is supported by
#      repeated examples rather than one isolated coincidence.
#
# f) Visual cleanliness:
#    - Rays are deliberately short and all output-side overlaps are forbidden, so the result
#      stays as separated diagonal bundles instead of becoming a mesh of long crossing lines.
#    - Restricting the code to only two segment lengths keeps the pattern language compact.
#
# g) Color inferability:
#    - No hidden color ranking is introduced. Every emitted ray bundle simply uses the color
#      of its source outer segment, exactly as in the source task.
#    - Since colors do not control either emitter selection or ray length, the combo remains
#      inferable from geometry alone.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, neighbors8


RAY_LENGTH_FACTOR = 2


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
    # Returns (slope_key, points_sorted_along_line) or None if not diagonal collinear.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _emit_segment_bundle(
    out: np.ndarray,
    segment_points: list[tuple[int, int]],
    direction: tuple[int, int],
    color: int,
    *,
    emit_all_pixels: bool,
    fixed_length: bool,
    bg_colors: set[int],
) -> None:
    h, w = out.shape
    emitters = segment_points if emit_all_pixels else [segment_points[0], segment_points[-1]]
    steps = len(segment_points) * RAY_LENGTH_FACTOR if fixed_length else None

    for sr, sc in emitters:
        rr, cc = sr, sc
        if steps is None:
            while True:
                rr += direction[0]
                cc += direction[1]
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if int(out[rr, cc]) not in bg_colors:
                    break
                out[rr, cc] = color
        else:
            for _ in range(steps):
                rr += direction[0]
                cc += direction[1]
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                out[rr, cc] = color


def _simulate(
    grid: np.ndarray,
    *,
    emit_all_pixels: bool,
    fixed_length: bool,
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
        outer_comps = sorted(_cc8(adj_mask), key=_bbox)

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

            _emit_segment_bundle(
                out,
                segment_points,
                outward,
                ocol,
                emit_all_pixels=emit_all_pixels,
                fixed_length=fixed_length,
                bg_colors=bg_colors,
            )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit dense short diagonal bundles from outer segments next to checkerboard diamonds.

    Input structure:
    - The background is a 0/1 checkerboard with top-left cell = 1.
    - Non-background objects are 8-pixel hollow diamonds plus diagonal outer segments placed
      one step farther out from selected diamond edges.
    - Outer segments have visible length 2 or 3, and each segment is a straight diagonal
      component in its own color.

    Rule:
    - For each outer segment, infer the outward diagonal direction by checking which normal
      points away from the adjacent diamond.
    - Every pixel of that segment emits one outward ray in parallel.
    - Each emitted ray paints exactly `visible_segment_length * 2` cells, unless the grid
      boundary is reached first.

    Key insight:
    - The segment length carries two pieces of information at once: it sets both the number
      of parallel rays and their travel distance, so longer visible segments create denser
      and longer bundles instead of boundary-reaching endpoint rays.
    """
    return _simulate(grid, emit_all_pixels=True, fixed_length=True)


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (16, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
VISIBLE_SEGMENT_LENGTHS = (2, 3)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500

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


def _ray_cells_fixed(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    for _ in range(steps):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
    return cells


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _visible_segment(full_segment: list[tuple[int, int]], visible_len: int, anchor: int) -> list[tuple[int, int]]:
    if visible_len == 3:
        return full_segment[:]
    if visible_len == 2 and anchor in (0, 1):
        return full_segment[anchor : anchor + 2]
    raise ValueError("unsupported visible segment spec")


def _segment_bundle_rays(
    shape: tuple[int, int],
    segment: list[tuple[int, int]],
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    steps = len(segment) * RAY_LENGTH_FACTOR
    rays: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pixel in segment:
        ray = _ray_cells_fixed(shape, pixel, direction, steps)
        if len(ray) != steps:
            return None
        for cell in ray:
            if cell in seen:
                return None
            seen.add(cell)
            rays.append(cell)
    return rays


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    edge_specs: tuple[tuple[str, int, int], ...],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    output_cells = set(diamond)
    input_cells = set(diamond)
    edge_parts = []

    for edge, visible_len, anchor in edge_specs:
        info = _EDGE_INFO[edge]
        full_segment = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in full_segment):
            return None
        full_segment = _sort_segment(full_segment, info["slope"])
        segment = _visible_segment(full_segment, visible_len, anchor)
        rays = _segment_bundle_rays((h, w), segment, info["outward"])
        if rays is None:
            return None
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in output_cells for cell in part_cells):
            return None
        output_cells.update(part_cells)
        input_cells.update(segment)
        edge_parts.append(
            {
                "segment": segment,
                "rays": rays,
                "weight": len(part_cells),
                "visible_len": visible_len,
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(output_cells),
        "input_cells": frozenset(input_cells),
    }


def _search_placements(h: int, w: int, specs: list[tuple[tuple[str, int, int], ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[tuple[str, int, int], ...], list[dict[str, object]]] = {}
    for edge_specs in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        placements = [
            p for r0, c0 in positions if (p := _candidate_placement(h, w, r0, c0, edge_specs)) is not None
        ]
        random.shuffle(placements)
        candidate_cache[edge_specs] = placements

    occupied_output: set[tuple[int, int]] = set()
    occupied_input_buffer: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            all_cells = placement["all_cells"]
            input_cells = set(placement["input_cells"])
            if not occupied_output.isdisjoint(all_cells):
                continue
            if not occupied_input_buffer.isdisjoint(input_cells):
                continue

            input_buffer = _expand8(input_cells, (h, w))
            if not occupied_input_buffer.isdisjoint(input_buffer):
                continue

            occupied_output.update(all_cells)
            occupied_input_buffer.update(input_buffer)
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_output.difference_update(all_cells)
            occupied_input_buffer.difference_update(input_buffer)
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


def _make_specs(diamond_count: int) -> list[tuple[tuple[str, int, int], ...]]:
    edge_sets = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]
    total_edges = sum(len(edges) for edges in edge_sets)
    if total_edges < 2:
        raise ValueError("expected at least two edges overall")

    if total_edges == 2:
        short_count = 1
    else:
        short_count = random.randint(1, total_edges - 1)
    if total_edges >= 4:
        short_count = random.randint(2, total_edges - 2)
    lengths = [2] * short_count + [3] * (total_edges - short_count)
    random.shuffle(lengths)

    specs: list[tuple[tuple[str, int, int], ...]] = []
    idx = 0
    for edges in edge_sets:
        cur = []
        for edge in edges:
            visible_len = lengths[idx]
            idx += 1
            anchor = random.randint(0, 3 - visible_len)
            cur.append((edge, visible_len, anchor))
        specs.append(tuple(sorted(cur)))
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _make_specs(diamond_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    visible_lengths: list[int] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])
            visible_lengths.append(int(edge_part["visible_len"]))

    if not _bg_counts_safe(out):
        return None
    if set(visible_lengths) != {2, 3}:
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
