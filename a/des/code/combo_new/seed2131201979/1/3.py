from __future__ import annotations

import random

import numpy as np


BG = 0
RING = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _ring_radii(grid: np.ndarray, center: tuple[int, int], ring_color: int) -> list[int]:
    cr, cc = center
    radii = {
        max(abs(int(r) - cr), abs(int(c) - cc))
        for r, c in np.argwhere(grid == int(ring_color))
        if (int(r), int(c)) != (cr, cc)
    }
    out = sorted(int(radius) for radius in radii if int(radius) > 0)
    assert out
    return out


def _layer_for_distance(dist: int, radii: list[int]) -> int:
    return max(1, sum(int(radius) <= int(dist) for radius in radii))


def _growth_kind(dr: int, dc: int) -> str:
    if dc > 0 and dr >= 0:
        return "SE"
    if dr > 0 and dc <= 0:
        return "SW"
    if dc < 0 and dr <= 0:
        return "NW"
    return "NE"


def _zone8(dr: int, dc: int) -> str:
    if dr < 0 and dc == 0:
        return "N"
    if dr < 0 and dc > 0:
        return "NE"
    if dr == 0 and dc > 0:
        return "E"
    if dr > 0 and dc > 0:
        return "SE"
    if dr > 0 and dc == 0:
        return "S"
    if dr > 0 and dc < 0:
        return "SW"
    if dr == 0 and dc < 0:
        return "W"
    return "NW"


def _square_bbox(r: int, c: int, size: int, kind: str) -> tuple[int, int, int, int]:
    if kind == "SE":
        return int(r), int(r + size), int(c), int(c + size)
    if kind == "SW":
        return int(r), int(r + size), int(c - size + 1), int(c + 1)
    if kind == "NE":
        return int(r - size + 1), int(r + 1), int(c), int(c + size)
    return int(r - size + 1), int(r + 1), int(c - size + 1), int(c + 1)


def _bbox_in_bounds(bbox: tuple[int, int, int, int], n: int) -> bool:
    r0, r1, c0, c1 = bbox
    return 0 <= int(r0) < int(r1) <= int(n) and 0 <= int(c0) < int(c1) <= int(n)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    cr = h // 2
    cc = w // 2
    ring_color = int(g[cr, cc])
    radii = _ring_radii(g, (cr, cc), ring_color)

    out = np.full_like(g, BG)
    seed_mask = (g != BG) & (g != int(ring_color))
    for r, c in np.argwhere(seed_mask):
        rr = int(r)
        cc2 = int(c)
        dist = max(abs(rr - cr), abs(cc2 - cc))
        size = _layer_for_distance(dist, radii)
        kind = _growth_kind(rr - cr, cc2 - cc)
        r0, r1, c0, c1 = _square_bbox(rr, cc2, size, kind)
        assert _bbox_in_bounds((r0, r1, c0, c1), h)
        out[r0:r1, c0:c1] = int(g[rr, cc2])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a solid square whose size comes from the concentric-ring layer and whose direction comes from the seed's quadrant.

    Input structure:
    - The background is black.
    - The grid is an odd-sized square with a single gray center marker.
    - Gray concentric square rings surround the center, with one-pixel gaps between neighboring rings.
    - Colored non-gray single-pixel seeds are placed on ring borders or in the regions assigned to those rings.

    Rule:
    - Determine the ring layer of each colored seed from its Chebyshev distance to the center: crossing one more gray ring increases the layer index by one.
    - A seed in layer k grows into a solid k x k square of the same color.
    - The growth anchor is chosen by the seed's quadrant relative to the center so the square expands away from the center.
    - If a seed lies on a horizontal or vertical axis, treat it as belonging to the clockwise-adjacent quadrant.
    - Remove the gray rings and center marker in the output; keep only the grown colored squares on a black background.

    Key insight: The gray ring geometry independently encodes square size, while the seed's quadrant independently encodes which corner of the square stays fixed.
    """
    return _transform(grid)


def _paint_rings(n: int, radii: list[int]) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    cr = cc = n // 2
    grid[cr, cc] = RING
    for radius in radii:
        for r in range(cr - radius, cr + radius + 1):
            grid[r, cc - radius] = RING
            grid[r, cc + radius] = RING
        for c in range(cc - radius, cc + radius + 1):
            grid[cr - radius, c] = RING
            grid[cr + radius, c] = RING
    return grid


def _candidate_cells(n: int, radii: list[int]) -> list[dict]:
    cr = cc = n // 2
    outer_radius = int(radii[-1])
    candidates: list[dict] = []
    for r in range(n):
        for c in range(n):
            if (r, c) == (cr, cc):
                continue
            dist = max(abs(r - cr), abs(c - cc))
            if dist == 0 or dist > outer_radius:
                continue
            layer = _layer_for_distance(dist, radii)
            kind = _growth_kind(r - cr, c - cc)
            bbox = _square_bbox(r, c, layer, kind)
            if not _bbox_in_bounds(bbox, n):
                continue
            candidates.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "dist": int(dist),
                    "layer": int(layer),
                    "kind": kind,
                    "zone": _zone8(r - cr, c - cc),
                    "axis": bool(r == cr or c == cc),
                    "on_ring": bool(dist in radii),
                    "bbox": bbox,
                }
            )
    return candidates


def _pick_nonconflicting(pool: list[dict], selected: list[dict]) -> dict | None:
    order = pool[:]
    random.shuffle(order)
    for cand in order:
        if any(_boxes_overlap(cand["bbox"], item["bbox"]) for item in selected):
            continue
        if any((cand["r"], cand["c"]) == (item["r"], item["c"]) for item in selected):
            continue
        return cand
    return None


def _build_expected(n: int, seeds: list[dict]) -> np.ndarray:
    out = np.full((n, n), BG, dtype=int)
    for seed in seeds:
        r0, r1, c0, c1 = seed["bbox"]
        out[r0:r1, c0:c1] = int(seed["color"])
    return out


def _sample_seed_set(candidates: list[dict], ring_count: int) -> list[dict] | None:
    if not candidates:
        return None

    max_seed_count = min(len(PALETTE), max(3, min(7, len(candidates))))
    seed_count = random.randint(3, max_seed_count)

    want_axis = any(cand["axis"] for cand in candidates) and random.random() < 0.6
    want_ring = any(cand["on_ring"] for cand in candidates) and random.random() < 0.7
    want_gap = any(not cand["on_ring"] for cand in candidates) and random.random() < 0.7

    axis_pool = [cand for cand in candidates if cand["axis"]]
    ring_pool = [cand for cand in candidates if cand["on_ring"]]
    gap_pool = [cand for cand in candidates if not cand["on_ring"]]

    for _ in range(300):
        selected: list[dict] = []

        for pool, needed in ((axis_pool, want_axis), (ring_pool, want_ring), (gap_pool, want_gap)):
            if not needed:
                continue
            cand = _pick_nonconflicting(pool, selected)
            if cand is not None:
                selected.append(cand)

        while len(selected) < seed_count:
            layer_counts = {layer: 0 for layer in range(1, ring_count + 1)}
            for item in selected:
                layer_counts[int(item["layer"])] += 1

            min_count = min(layer_counts.values())
            preferred_layers = {
                layer for layer, count in layer_counts.items() if int(count) == int(min_count)
            }
            pool = [cand for cand in candidates if int(cand["layer"]) in preferred_layers]
            cand = _pick_nonconflicting(pool or candidates, selected)
            if cand is None:
                break
            selected.append(cand)

        if len(selected) != seed_count:
            continue

        if len({int(item["layer"]) for item in selected}) < min(2, ring_count):
            continue
        if len({item["kind"] for item in selected}) < min(2, seed_count):
            continue
        if want_axis and not any(item["axis"] for item in selected):
            continue
        if want_ring and not any(item["on_ring"] for item in selected):
            continue
        if want_gap and not any(not item["on_ring"] for item in selected):
            continue
        if seed_count >= 4 and len({item["zone"] for item in selected}) < 3:
            continue

        chosen = [dict(item) for item in selected]
        colors = random.sample(PALETTE, len(chosen))
        for item, color in zip(chosen, colors):
            item["color"] = int(color)
        return chosen

    return None


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.randint(2, 6)
    start_radius = random.choice([1, 2])
    radii = [int(start_radius + 2 * idx) for idx in range(ring_count)]
    outer_radius = int(radii[-1])

    extra_margin = random.choice([0, 0, 1, 1, 2])
    n = 2 * (outer_radius + extra_margin) + 1
    if n > 29:
        return None

    base = _paint_rings(n, radii)
    candidates = _candidate_cells(n, radii)
    seeds = _sample_seed_set(candidates, ring_count)
    if seeds is None:
        return None

    inp = base.copy()
    for seed in seeds:
        inp[int(seed["r"]), int(seed["c"])] = int(seed["color"])

    expected = _build_expected(n, seeds)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid concentric-ring seed puzzle")
