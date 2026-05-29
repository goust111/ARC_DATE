# Variant of 80a900e0
# Dimension: D18 — 输入输出关系方向 (meta)
# Change: 输出先构建（output-first）：先放菱形和射线，再删除射线生成输入。解题方向为从输入推断并添加射线... → 输入含部分射线，输出补全射线
#
# Design analysis
# a) Concrete behavior:
#    - The terminal output geometry stays the same as the original task: checkerboard background,
#      5x5 hollow diamonds, 2-4 outer diagonal segments, and two outward diagonal rays per segment.
#    - The input now keeps a contiguous prefix of each ray starting from the segment endpoint.
#      The distal suffix is removed back to checkerboard. So an input can show:
#      segment + short visible ray stub -> output extends the same color further to the boundary.
#    - Example trace for one edge:
#      original generator: build segment + full ray, then delete the whole ray from input.
#      variant generator: build segment + full ray, then keep the first k cells (k may be 0/full,
#      but at least some rays in every sample satisfy 0 < k < full_len).
#      original solver: detect a clean 3-cell outer segment component and cast both rays.
#      variant solver: the segment is no longer isolated because partial rays are 8-connected to it,
#      so component-level diagonal-collinearity breaks. The solver must instead detect the segment
#      from the diamond's fixed edge template, then continue through any same-color prefix and fill
#      the remaining suffix until the boundary.
# b) Complexity calibration:
#    - The variant is visibly different because some ray cells already exist in input.
#    - The trimming freedom is controlled: each ray keeps one contiguous prefix only; no holes, no
#      arbitrary subsets. This is enough to show "extend the clipped ray" without introducing many
#      unrelated cases.
#    - Generator enforces at least two genuinely partial rays and at least one longer ray, so the
#      new behavior is repeatedly visible instead of collapsing back to the original all-rays-missing case.
# c) Original implicit assumptions that no longer hold:
#    - In the original main(), adjacent non-bg cells around a diamond were split into components,
#      and each outer component was assumed to be a single diagonal segment. That relied on rays
#      being absent in the input. With partial rays present, segment + ray(s) form a bent component,
#      so `_outer_line_info(component)` fails.
#    - The painting loop originally assumed the first cell after an endpoint was background. Now it
#      may already be the same ray color, so the extension loop must step through same-color prefix
#      cells instead of treating non-bg as a hard stop.
# d) Placement legality:
#    - Output-first placement still uses the original non-overlap condition on full rays, so the
#      completed output remains collision-free and every ray has a unique unobstructed path.
#    - The input is derived only by restoring some ray suffix cells to checkerboard, so it cannot
#      introduce overlaps or destroy the segment templates that the solver needs.
# e) Visual observability:
#    - A human now sees short diagonal stubs already attached to some outer segments and can infer
#      that output extends those stubs outward to the edge.
#    - Because the kept part is always contiguous from the segment outward, the direction is visible
#      in the input itself; there is no hidden bookkeeping such as color-order or index-based sorting.
# f) Visual cleanliness:
#    - No new object types are added. The only change is ray length in the input, so the scene stays
#      readable. Full-ray non-overlap from the original generator also prevents dense crossing clutter.
# g) Color inference:
#    - Colors are unchanged: each ray uses its segment color exactly as before. The task is purely
#      geometric completion, not a color remapping puzzle.
# h) New solver assumptions and generator guarantees:
#    - Solver assumes every real outer segment still occupies its fixed template cells adjacent to a
#      diamond and those visible template cells are monochrome non-background. Generator preserves that.
#    - Solver assumes trimmed rays remain contiguous prefixes from the segment endpoints. Generator
#      only deletes suffixes, never interior cells.
#    - Solver assumes full output rays do not collide with other non-bg objects. Generator still
#      validates placements against the completed output before deriving the input.

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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


# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 600

# ── Geometry constants ──────────────────────────────────────
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


def _diamond_anchor(dpts: list[tuple[int, int]]) -> tuple[int, int]:
    return min(r for r, _ in dpts), min(c for _, c in dpts)


def _visible_segment(anchor: tuple[int, int], shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = anchor
    info = _EDGE_INFO[edge]
    cells = [
        (r0 + dr, c0 + dc)
        for dr, dc in info["segment"]
        if 0 <= r0 + dr < h and 0 <= c0 + dc < w
    ]
    if len(cells) < MIN_SEGMENT_LEN:
        return []
    return _sort_segment(cells, info["slope"])


def _iter_diamonds(g: np.ndarray, bg_colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    diamonds = [(col, pts) for col, pts in _components8_by_color(g, non_bg) if _is_diamond8(pts)]
    return sorted(diamonds, key=lambda item: _diamond_anchor(item[1]))


def _detect_edge_segment(
    g: np.ndarray, bg_colors: set[int], anchor: tuple[int, int], edge: str
) -> tuple[int, list[tuple[int, int]], tuple[int, int]] | None:
    segment = _visible_segment(anchor, g.shape, edge)
    if not segment:
        return None
    vals = [int(g[r, c]) for r, c in segment]
    if any(v in bg_colors for v in vals):
        return None
    if len(set(vals)) != 1:
        return None
    return vals[0], segment, _EDGE_INFO[edge]["outward"]


def _extend_ray(
    out: np.ndarray,
    bg_colors: set[int],
    color: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        v = int(out[r, c])
        if v == color:
            r += dr
            c += dc
            continue
        if v in bg_colors:
            out[r, c] = color
            r += dr
            c += dc
            continue
        break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete partially clipped diagonal rays around fixed 5x5 checkerboard diamonds.

    Compared with the original task, the input may already contain a ray prefix attached to a
    segment endpoint. The solver therefore reads each potential outer segment from the diamond's
    fixed edge templates and extends through any existing same-color prefix until the boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg_colors = set(_top_k_colors(g, 2))
    out = g.copy()

    for _, dpts in _iter_diamonds(g, bg_colors):
        anchor = _diamond_anchor(dpts)
        for edge in _EDGE_NAMES:
            found = _detect_edge_segment(g, bg_colors, anchor, edge)
            if found is None:
                continue
            color, segment, outward = found
            for endpoint in (segment[0], segment[-1]):
                _extend_ray(out, bg_colors, color, endpoint, outward)

    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = _visible_segment((r0, c0), (h, w), edge)
        if not segment:
            return None
        ray_a = _ray_cells((h, w), segment[0], info["outward"])
        ray_b = _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + ray_a + ray_b
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": (ray_a, ray_b),
                "weight": len(part_cells),
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


def _choose_keep_len(full_len: int) -> int:
    if full_len == 0:
        return 0
    if full_len == 1:
        return random.randint(0, 1)
    roll = random.random()
    if roll < 0.7:
        return random.randint(1, full_len - 1)
    if roll < 0.85:
        return 0
    return full_len


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
    ray_groups: list[list[tuple[int, int]]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            for ray in edge_part["rays"]:
                _paint_cells(out, ray, color)
                ray_groups.append(ray)

    if not _bg_counts_safe(out):
        return None

    long_rays = [ray for ray in ray_groups if len(ray) >= 2]
    if len(long_rays) < 2 or max((len(ray) for ray in ray_groups), default=0) < 4:
        return None

    inp = out.copy()
    partial_ray_count = 0
    shown_ray_cells = 0
    hidden_ray_cells = 0
    for ray in ray_groups:
        keep = _choose_keep_len(len(ray))
        if 0 < keep < len(ray):
            partial_ray_count += 1
        shown_ray_cells += keep
        hidden_ray_cells += len(ray) - keep
        for r, c in ray[keep:]:
            inp[r, c] = (r + c + 1) % 2

    if partial_ray_count < 2 or shown_ray_cells == 0 or hidden_ray_cells == 0:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
