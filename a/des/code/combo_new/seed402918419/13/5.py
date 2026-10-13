from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 0
GRAY_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
MIN_GRID_SIZE = 20
MAX_GRID_SIZE = 30
LEGEND_LENGTH_CHOICES = (3, 4, 4, 5)
GROUP_COUNT_CHOICES = (1, 2, 2, 3)
LAYER_BOUNDS_BY_GROUP_COUNT = {
    1: (2, 5),
    2: (2, 4),
    3: (2, 3),
}
EXTRA_SPAN_MAX = 6
PLACEMENT_TRIES = 180
GENERATION_RETRIES = 500
WRAP_PROB = 0.5
CORNERS = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(g: np.ndarray, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c] or seen[r, c]:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                        if int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_ring_bbox(g: np.ndarray, coords: list[tuple[int, int]], color: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_area = 2 * (h + w) - 4
    if len(coords) != expected_area:
        return None

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None
    if np.any(patch[1:-1, 1:-1] == color):
        return None
    return r0, c0, r1, c1


def _find_legend(g: np.ndarray) -> tuple[list[int], np.ndarray]:
    candidates: list[tuple[list[int], list[tuple[int, int]]]] = []
    for comp in _mask_components(g != BG_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != 1 or len(comp) != w or w < 3:
            continue
        colors = [int(g[r0, c]) for c in range(c0, c1 + 1)]
        if len(set(colors)) != w:
            continue
        candidates.append((colors, comp))

    assert len(candidates) == 1, "expected exactly one horizontal multicolor legend"
    cycle, coords = candidates[0]
    legend_mask = np.zeros(g.shape, dtype=bool)
    for r, c in coords:
        legend_mask[r, c] = True
    return cycle, legend_mask


def _find_rings(g: np.ndarray, legend_mask: np.ndarray) -> list[dict[str, object]]:
    blocked = legend_mask | (g == BG_COLOR)
    rings: list[dict[str, object]] = []
    for comp in _monochrome_components(g, blocked):
        color = int(g[comp[0]])
        bbox = _rect_ring_bbox(g, comp, color)
        assert bbox is not None, "all non-legend components must be rectangular rings"
        rings.append({"coords": comp, "color": color, "bbox": bbox})
    assert rings, "expected at least one ring"
    return rings


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor inner concentric rectangle rings by advancing along the corner legend's color cycle.

    Input structure:
    - The background is black.
    - One corner contains a horizontal multicolor legend strip whose left-to-right
      order defines a cyclic color sequence.
    - Elsewhere, the non-background objects are separated one-cell-thick
      rectangular rings arranged into concentric groups with black gaps between rings.
    - In each group, the outermost ring already has a legend color and the inner
      rings are placeholder gray.

    Rule:
    - Read the legend strip from left to right to obtain the global color cycle.
    - Parse every other non-background component as a rectangular ring and recover
      nesting by strict bounding-box containment.
    - For each concentric group, use the outermost ring color as the cycle start,
      then assign the next legend color to the next inner ring, continuing inward
      and wrapping around the cycle when needed.
    - Leave the legend strip unchanged.

    Key insight:
    - The horizontal strip is not another object to transform; it is the shared
      cyclic color key, and each ring group's geometry determines how many cycle
      steps to propagate inward from its already-colored outer shell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cycle, legend_mask = _find_legend(g)
    cycle_index = {color: i for i, color in enumerate(cycle)}
    assert len(cycle_index) == len(cycle), "legend colors must be distinct"

    rings = _find_rings(g, legend_mask)
    parents: list[int | None] = [None] * len(rings)
    for i, ring in enumerate(rings):
        best_parent = None
        best_area = None
        for j, candidate in enumerate(rings):
            if i == j:
                continue
            if not _contains(candidate["bbox"], ring["bbox"]):
                continue
            area = _bbox_area(candidate["bbox"])
            if best_parent is None or area < best_area:
                best_parent = j
                best_area = area
        parents[i] = best_parent

    memo: dict[int, tuple[int, int]] = {}

    def _root_and_depth(idx: int) -> tuple[int, int]:
        if idx in memo:
            return memo[idx]
        parent = parents[idx]
        if parent is None:
            memo[idx] = (idx, 0)
            return memo[idx]
        root, depth = _root_and_depth(parent)
        memo[idx] = (root, depth + 1)
        return memo[idx]

    out = g.copy()
    for i, ring in enumerate(rings):
        root, depth = _root_and_depth(i)
        start_color = int(rings[root]["color"])
        assert start_color in cycle_index, "outer ring color must appear in the legend"
        target = cycle[(cycle_index[start_color] + depth) % len(cycle)]
        for r, c in ring["coords"]:
            out[r, c] = target

    return out


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _layer_bbox(outer_bbox: tuple[int, int, int, int], depth: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer_bbox
    offset = 2 * depth
    return r0 + offset, c0 + offset, r1 - offset, c1 - offset


def _corner_legend_bbox(shape: tuple[int, int], legend_len: int, corner: str) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0, 0, legend_len - 1
    if corner == "tr":
        return 0, w - legend_len, 0, w - 1
    if corner == "bl":
        return h - 1, 0, h - 1, legend_len - 1
    assert corner == "br"
    return h - 1, w - legend_len, h - 1, w - 1


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _place_groups(
    shape: tuple[int, int], legend_bbox: tuple[int, int, int, int], specs: list[dict[str, int]]
) -> list[dict[str, int]] | None:
    h, w = shape
    reserved = np.zeros(shape, dtype=bool)
    _reserve(reserved, legend_bbox, pad=1)

    placed: list[dict[str, int]] = []
    ordered_specs = sorted(specs, key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)
    for spec in ordered_specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        if outer_h > h - 2 or outer_w > w - 2:
            return None

        positions = [
            (r0, c0)
            for r0 in range(1, h - outer_h)
            for c0 in range(1, w - outer_w)
        ]
        random.shuffle(positions)
        bbox = None
        for r0, c0 in positions:
            candidate = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _can_place(reserved, candidate, pad=1):
                bbox = candidate
                break
        if bbox is None:
            return None

        placed_spec = dict(spec)
        placed_spec["r0"] = bbox[0]
        placed_spec["c0"] = bbox[1]
        placed_spec["r1"] = bbox[2]
        placed_spec["c1"] = bbox[3]
        placed.append(placed_spec)
        _reserve(reserved, bbox, pad=1)

    return placed


def _build_input(shape: tuple[int, int], cycle: list[int], corner: str, groups: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)

    lr0, lc0, _lr1, lc1 = _corner_legend_bbox(shape, len(cycle), corner)
    for offset, color in enumerate(cycle):
        grid[lr0, lc0 + offset] = color

    for spec in groups:
        outer_bbox = (spec["r0"], spec["c0"], spec["r1"], spec["c1"])
        for depth in range(spec["layers"]):
            color = spec["start_color"] if depth == 0 else GRAY_COLOR
            _paint_ring(grid, _layer_bbox(outer_bbox, depth), color)

    return grid


def _build_expected(grid: np.ndarray, cycle: list[int], groups: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    cycle_index = {color: i for i, color in enumerate(cycle)}
    for spec in groups:
        outer_bbox = (spec["r0"], spec["c0"], spec["r1"], spec["c1"])
        start = cycle_index[spec["start_color"]]
        for depth in range(spec["layers"]):
            color = cycle[(start + depth) % len(cycle)]
            _paint_ring(out, _layer_bbox(outer_bbox, depth), color)
    return out


def _sample_group_specs(shape: tuple[int, int], cycle_len: int) -> list[dict[str, int]] | None:
    h, w = shape
    n_groups = random.choice(GROUP_COUNT_CHOICES)
    layer_lo, layer_hi = LAYER_BOUNDS_BY_GROUP_COUNT[n_groups]
    if min(h, w) < 4 * layer_lo + 1:
        return None

    specs: list[dict[str, int]] = []
    layers_list = [random.randint(layer_lo, layer_hi) for _ in range(n_groups)]
    wrap_candidates = [i for i in range(n_groups) if cycle_len + 1 <= layer_hi]
    if wrap_candidates and random.random() < WRAP_PROB:
        idx = random.choice(wrap_candidates)
        layers_list[idx] = random.randint(cycle_len + 1, layer_hi)

    for layers in layers_list:
        min_span = 4 * layers - 1
        if min_span > h - 2 or min_span > w - 2:
            return None
        max_h = min(EXTRA_SPAN_MAX, h - 2 - min_span)
        max_w = min(EXTRA_SPAN_MAX, w - 2 - min_span)
        outer_h = min_span + random.randint(0, max_h)
        outer_w = min_span + random.randint(0, max_w)
        specs.append(
            {
                "layers": layers,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "start_color": random.choice(COLOR_POOL),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    legend_len = random.choice(LEGEND_LENGTH_CHOICES)
    if w < legend_len + 2:
        return None

    cycle = random.sample(COLOR_POOL, legend_len)
    corner = random.choice(CORNERS)
    legend_bbox = _corner_legend_bbox((h, w), legend_len, corner)

    specs = _sample_group_specs((h, w), len(cycle))
    if specs is None:
        return None

    for spec in specs:
        spec["start_color"] = random.choice(cycle)

    placed = _place_groups((h, w), legend_bbox, specs)
    if placed is None:
        return None

    grid = _build_input((h, w), cycle, corner, placed)
    expected = _build_expected(grid, cycle, placed)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric color-cycle puzzle")
