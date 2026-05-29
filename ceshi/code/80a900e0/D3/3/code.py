# Variant of 80a900e0
# Dimension: D3 — 射线终止条件 (dst)
# Change: 射线遇到非背景像素或网格边界时停止... → 堆叠/累积（射线颜色覆盖被击中对象）
#
# Design analysis
# a) Concrete behavior and trace
# - Input stays visually close to the original task: checkerboard background, 8-pixel hollow
#   diamonds, and 2-3 pixel outer diagonal segments adjacent to some diamond edges.
# - The change is entirely in the output dynamics. Rays still launch from the two endpoints of
#   each outer segment and move diagonally outward. The original task stops a ray as soon as it
#   reaches any non-background pixel. This variant does not stop there: the hit pixel is repainted
#   to the ray color and the ray continues until the grid boundary.
# - Small trace: suppose a red segment emits a ray toward a teal diamond. In the original task the
#   first teal cell blocks the ray, so only the background cells before the hit become red. In this
#   variant that teal cell becomes red, later teal cells on the same diagonal also become red, and
#   the ray keeps going onto the background beyond the diamond.
#
# b) Complexity calibration
# - If the generator still allowed only one diamond, most instances would collapse back to the
#   original behavior because there would be nothing for rays to hit. To make D3 observable, the
#   generator uses 2-3 diamonds and actively prefers placements where some ray path intersects
#   another object or another ray path.
# - The overlap budget is kept small. We require a few visible recolors and continuation pixels,
#   but reject overly chaotic cases with too many overwritten object pixels.
#
# c) Hidden-assumption audit for the original main()
# - `bg_colors = top-2 colors` still works because the input still contains only checkerboard plus
#   a modest number of object pixels; the generator preserves that frequency gap.
# - Diamond detection via 8-connected size-8 components still works because input objects never
#   overlap. Only output rays overlap.
# - Collecting adjacent outer-segment pixels from `g` rather than `out` is now essential. Later
#   rays may recolor earlier objects in `out`, so all object discovery must remain anchored to the
#   untouched input grid.
# - The original line `if out[rr, cc] in bg_colors: paint else: break` relied on the old stopping
#   rule. That assumption is invalid here. The fix is to break only in the old simulation helper,
#   and otherwise always repaint the current cell and continue.
# - Once overwriting is allowed, processing order becomes visible. The original code implicitly
#   relied on component-label order. This variant makes the priority explicit by sorting diamonds
#   and adjacent outer segments by bounding box before painting.
#
# d) Placement legality
# - Input-side objects are still strictly non-overlapping and in-bounds.
# - Self-overlap inside one diamond placement is also forbidden: a diamond's own rays cannot cross
#   its own diamond, segments, or other rays in the candidate template.
# - Cross-object overlap is only allowed for predicted output rays, never for input objects. This
#   keeps the solver's perception task identical to the original task while enabling stacked output.
#
# e) Visual observability
# - The generator enforces that stacked output differs from stop-on-hit output by several pixels,
#   and that at least a couple of original object pixels are recolored in the stacked result.
# - This makes the new rule visible from examples: rays clearly pass through a colored object and
#   continue beyond it, instead of terminating at first contact.
#
# f) Visual cleanliness
# - Overlaps are intentionally sparse. The generator rejects outputs where too many input object
#   pixels are overwritten, preventing the scene from turning into unreadable diagonal clutter.
# - Segment colors are chosen from a small repeated subset so the visual language stays compact.
#
# g) Color-change inferability
# - No numeric ordering of colors is used. A hit cell simply takes the visible color of the ray
#   that reaches it later in the deterministic scan order.
# - Because only 2-3 segment colors are reused across several rays, the overwrite behavior has
#   repeated support within a few examples instead of looking like arbitrary one-off recoloring.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


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
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _simulate(grid: np.ndarray, *, stop_on_non_bg: bool) -> np.ndarray:
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
        outer_comps = sorted(_cc8(adj_mask), key=_bbox)

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
                while 0 <= rr < h and 0 <= cc < w:
                    if stop_on_non_bg and int(out[rr, cc]) not in bg_colors:
                        break
                    out[rr, cc] = ocol
                    rr += outward[0]
                    cc += outward[1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: endpoint rays never stop at previously colored pixels.
    They repaint any hit object/ray cell and continue to the grid boundary.
    """
    return _simulate(grid, stop_on_non_bg=False)


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (2, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
SEGMENT_PALETTE_SIZE_RANGE = (2, 3)
MIN_STACK_DIFF = 5
MIN_CONTINUATION_PIXELS = 3
MIN_OBJECT_OVERWRITES = 2
MAX_OBJECT_OVERWRITES = 10
MAX_GEN_RETRIES = 400

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
    object_cells = set(diamond)
    all_self_cells = set(diamond)
    edge_parts = []
    ray_cells_all: list[tuple[int, int]] = []

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
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_self_cells for cell in part_cells):
            return None
        all_self_cells.update(part_cells)
        object_cells.update(segment)
        ray_cells_all.extend(rays)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "object_cells": frozenset(object_cells),
        "ray_cells": frozenset(ray_cells_all),
    }


def _interaction_score(
    placement: dict[str, object],
    occupied_objects: set[tuple[int, int]],
    occupied_rays: set[tuple[int, int]],
) -> int:
    object_hits = len(placement["object_cells"] & occupied_rays)
    ray_hits = len(placement["ray_cells"] & occupied_objects)
    ray_crosses = len(placement["ray_cells"] & occupied_rays)
    return 4 * object_hits + 3 * ray_hits + ray_crosses


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            placement
            for r0, c0 in positions
            if (placement := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]
        if not candidate_cache[edges]:
            return None

    occupied_objects: set[tuple[int, int]] = set()
    occupied_ray_counts: Counter[tuple[int, int]] = Counter()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int, interaction_total: int) -> bool:
        if i == len(order):
            return interaction_total > 0

        spec_i = order[i]
        occupied_rays = set(occupied_ray_counts)
        candidates = []
        for placement in candidate_cache[specs[spec_i]]:
            obj_cells = placement["object_cells"]
            if not occupied_objects.isdisjoint(obj_cells):
                continue
            score = _interaction_score(placement, occupied_objects, occupied_rays)
            candidates.append((score, random.random(), placement))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for score, _, placement in candidates:
            obj_cells = placement["object_cells"]
            ray_cells = placement["ray_cells"]
            occupied_objects.update(obj_cells)
            for cell in ray_cells:
                occupied_ray_counts[cell] += 1
            placed[spec_i] = placement
            if backtrack(i + 1, interaction_total + score):
                return True
            occupied_objects.difference_update(obj_cells)
            for cell in ray_cells:
                occupied_ray_counts[cell] -= 1
                if occupied_ray_counts[cell] == 0:
                    del occupied_ray_counts[cell]
            placed[spec_i] = None
        return False

    if not backtrack(0, 0):
        return None
    return [placement for placement in placed if placement is not None]


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    active_size = min(len(palette), random.randint(*SEGMENT_PALETTE_SIZE_RANGE))
    active_palette = random.sample(palette, active_size)
    counts = Counter({c: 0 for c in active_palette})

    weighted_edges = sorted(
        (
            (placement["edges"][ei]["weight"], random.random(), pi, ei)
            for pi, placement in enumerate(placements)
            for ei in range(len(placement["edges"]))
        ),
        reverse=True,
    )

    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(active_palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _stack_metrics(inp: np.ndarray, stacked: np.ndarray, blocked: np.ndarray) -> dict[str, int]:
    bg = _checkerboard(*inp.shape)
    object_mask = inp != bg
    return {
        "stack_diff": int(np.count_nonzero(stacked != blocked)),
        "continuation_pixels": int(np.count_nonzero((inp == bg) & (stacked != blocked))),
        "object_overwrites": int(np.count_nonzero(object_mask & (stacked != inp))),
    }


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

    inp = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(inp, edge_part["segment"], segment_colors[pi][ei])

    if not _bg_counts_safe(inp):
        return None

    stacked = main(inp)
    blocked = _simulate(inp, stop_on_non_bg=True)
    metrics = _stack_metrics(inp, stacked, blocked)
    if metrics["stack_diff"] < MIN_STACK_DIFF:
        return None
    if metrics["continuation_pixels"] < MIN_CONTINUATION_PIXELS:
        return None
    if not (MIN_OBJECT_OVERWRITES <= metrics["object_overwrites"] <= MAX_OBJECT_OVERWRITES):
        return None
    if np.array_equal(stacked, blocked):
        return None
    if not np.array_equal(main(inp), stacked):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
