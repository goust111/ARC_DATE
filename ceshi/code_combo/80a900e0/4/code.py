# Combo variant of 80a900e0 (index 4)
# combo_spec: D3/1+D6/2
# Name: 双重长度编码

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis:
# a) Concrete behavior: in the base task a 3-pixel outer segment emits two rays that keep going
#    until the boundary. Here the segment itself may be 2-5 pixels long, and each endpoint emits a
#    ray of exactly 2 * segment_length pixels. Example: a 4-pixel TR segment now paints two rays of
#    length 8; the old solver would have painted all the way to the corner.
# b) Complexity calibration: the multiplier is global and fixed at 2, while each grid uses only
#    2-3 segment lengths with repetition. That keeps the numerical rule visible instead of hiding it
#    behind too many free parameters.
# c) Broken assumptions from the original solver:
#    - Collecting only pixels adjacent to the diamond no longer recovers the whole segment when
#      length is 4 or 5, because the extra endpoint pixels sit beyond the diamond edge span.
#    - `while ... until boundary` is no longer valid; ray length must be decoded from the segment.
#    - Boundary-clipped segments would make the visible length ambiguous, so the generator forbids
#      clipping and keeps one extra empty step beyond every ray tip to separate this combo from the
#      original boundary-reaching behavior.
# d) Placement legality: the generator is output-first, rejects any overlap, requires all fixed rays
#    to fit fully inside the canvas, and also requires one extra in-bounds slack cell beyond each
#    ray so the combo never degenerates to the original stopping rule.
# e) Visual observability: each generated grid has at least three segments, at least two distinct
#    lengths, and at least one non-3 length. A human can compare short vs long segments and see that
#    longer segments create proportionally longer rays without reaching the edge.
# f) Visual cleanliness: only diamonds, diagonal segments, and their two rays occupy the scene, with
#    non-overlap enforced both within a diamond placement and across placements.
# g) New solver assumptions audited against the generator:
#    - Non-background components other than diamonds are contiguous diagonal segments of length 2-5.
#    - Each segment has a unique best one-step inward overlap with exactly one diamond.
#    - The encoded ray cells are background in the input.
#    The generator constructs inputs to satisfy all three constraints and validates `main(inp)==out`.

RAY_MULTIPLIER = 2

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
COLOR_RANGE = range(2, 10)
SEGMENT_LENGTHS = (2, 3, 4, 5)
MIN_TOTAL_SEGMENTS = 3
MAX_GEN_RETRIES = 500

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

_DIAMOND8_OFFSETS = set(_DIAMOND_COORDS)

_EDGE_INFO = {
    "TR": {"anchor": (0, 4), "line_dir": (1, 1), "outward": (-1, 1), "slope": "diff"},
    "TL": {"anchor": (0, 0), "line_dir": (1, -1), "outward": (-1, -1), "slope": "sum"},
    "BL": {"anchor": (4, 0), "line_dir": (1, 1), "outward": (1, -1), "slope": "diff"},
    "BR": {"anchor": (4, 4), "line_dir": (1, -1), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
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


def _ordered_diagonal_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None

    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        ordered = sorted(pts, key=lambda p: (p[0], p[1]))
        step = (1, 1)
        slope_key = "rc_diff"
    elif all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        ordered = sorted(pts, key=lambda p: (p[0], -p[1]))
        step = (1, -1)
        slope_key = "rc_sum"
    else:
        return None

    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        if (r1 - r0, c1 - c0) != step:
            return None
    return slope_key, ordered


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


def _segment_cells(r0: int, c0: int, edge: str, seg_len: int) -> list[tuple[int, int]]:
    info = _EDGE_INFO[edge]
    ar, ac = info["anchor"]
    dr, dc = info["line_dir"]
    start_t = -((seg_len - 1) // 2)
    return [(r0 + ar + (start_t + i) * dr, c0 + ac + (start_t + i) * dc) for i in range(seg_len)]


def _ray_cells_fixed(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], ray_len: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(1, ray_len + 1):
        r = start[0] + dr * step
        c = start[1] + dc * step
        if not (0 <= r < h and 0 <= c < w):
            return None
        cells.append((r, c))

    # Require extra room so this combo never collapses to "extend until boundary".
    tail_r = start[0] + dr * (ray_len + 1)
    tail_c = start[1] + dc * (ray_len + 1)
    if not (0 <= tail_r < h and 0 <= tail_c < w):
        return None
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each outer diagonal segment's length and emit two rays whose lengths are a fixed multiple of it.

    Input structure:
    - Checkerboard background made from the two dominant colors.
    - Hollow 8-pixel diamonds in non-background colors.
    - One diagonal outer segment per selected diamond edge, still offset one step outward, but now the
      visible segment length can be 2, 3, 4, or 5 pixels.

    Rule:
    - Identify each diagonal outer segment and pair it with the diamond it faces by checking which
      one-step inward shift best overlaps a diamond edge.
    - Let L be the segment's visible length. From both segment endpoints, move outward along the
      perpendicular diagonal and paint exactly `2 * L` cells in the segment's color.

    Key insight: the output length is no longer read from available empty space. The segment length is
    itself the code, so solving requires two linked numeric steps: measure the segment, then double it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)

    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]
    diamond_sets = [set(pts) for _, pts in diamonds]
    diamond_union = set().union(*diamond_sets) if diamond_sets else set()

    segments = []
    for col, pts in comps:
        if any(cell in diamond_union for cell in pts):
            continue
        info = _ordered_diagonal_info(pts)
        if info is None or not (2 <= len(pts) <= 5):
            continue
        segments.append((int(col), pts, info))

    out = g.copy()

    def seg_key(item):
        _col, pts, _info = item
        return _bbox(pts)

    for col, pts, info in sorted(segments, key=seg_key):
        slope_key, ordered = info
        if slope_key == "rc_diff":
            normals = [(-1, 1), (1, -1)]
        else:
            normals = [(-1, -1), (1, 1)]

        best_score = -1
        best_normal = None
        for dset in diamond_sets:
            for normal in normals:
                score = sum(((r + normal[0], c + normal[1]) in dset) for r, c in pts)
                if score > best_score:
                    best_score = score
                    best_normal = normal

        if best_score <= 0 or best_normal is None:
            continue

        outward = (-best_normal[0], -best_normal[1])
        ray_len = len(ordered) * RAY_MULTIPLIER
        for start in (ordered[0], ordered[-1]):
            for step in range(1, ray_len + 1):
                rr = start[0] + outward[0] * step
                cc = start[1] + outward[1] * step
                if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = col

    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, spec: tuple[tuple[str, int], ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge, seg_len in spec:
        info = _EDGE_INFO[edge]
        segment = _segment_cells(r0, c0, edge, seg_len)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
            return None

        segment = _sort_segment(segment, info["slope"])
        rays_a = _ray_cells_fixed((h, w), segment[0], info["outward"], seg_len * RAY_MULTIPLIER)
        rays_b = _ray_cells_fixed((h, w), segment[-1], info["outward"], seg_len * RAY_MULTIPLIER)
        if rays_a is None or rays_b is None:
            return None

        part_cells = segment + rays_a + rays_b
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        edge_parts.append(
            {
                "edge": edge,
                "segment_len": seg_len,
                "segment": segment,
                "rays": rays_a + rays_b,
                "weight": len(part_cells),
            }
        )

    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(h: int, w: int, specs: list[tuple[tuple[str, int], ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: (-sum(seg_len for _, seg_len in specs[i]), -len(specs[i])))
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[tuple[str, int], ...], list[dict[str, object]]] = {}
    for spec in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[spec] = [
            p for r0, c0 in positions if (p := _candidate_placement(h, w, r0, c0, spec)) is not None
        ]
        if not candidate_cache[spec]:
            return None

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
    for color in diamond_colors:
        counts[color] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        (
            (edge_part["weight"], random.random(), pi, ei)
            for pi, placement in enumerate(placements)
            for ei, edge_part in enumerate(placement["edges"])
        ),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _weight, _noise, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _build_specs(diamond_count: int) -> list[tuple[tuple[str, int], ...]] | None:
    edge_sets = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]
    total_segments = sum(len(edges) for edges in edge_sets)
    if total_segments < MIN_TOTAL_SEGMENTS:
        return None

    short_len = random.choice((2, 3))
    long_len = random.choice((4, 5))
    if random.random() < 0.5:
        a, b = short_len, long_len
    else:
        a, b = long_len, short_len
    length_pool = [a, b]
    if total_segments >= 5 and random.random() < 0.35:
        extras = [x for x in SEGMENT_LENGTHS if x not in length_pool]
        length_pool.append(random.choice(extras))

    lengths = [a, b, a]
    while len(lengths) < total_segments:
        lengths.append(random.choice(length_pool))
    random.shuffle(lengths)
    if set(lengths) == {3}:
        return None

    specs: list[tuple[tuple[str, int], ...]] = []
    idx = 0
    for edges in edge_sets:
        pairs = tuple(sorted((edge, lengths[idx + j]) for j, edge in enumerate(edges)))
        idx += len(edges)
        specs.append(pairs)

    used_lengths = {seg_len for spec in specs for _, seg_len in spec}
    if len(used_lengths) < 2 or used_lengths == {3}:
        return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _build_specs(diamond_count)
    if specs is None:
        return None

    placements = _search_placements(h, w, specs)
    if placements is None:
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
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.asarray(rs, dtype=int)
        cs = np.asarray(cs, dtype=int)
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
