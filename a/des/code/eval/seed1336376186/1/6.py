import random
from collections import deque

import numpy as np


BG = 0
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (20, 35)
GRID_W_RANGE = (20, 35)
RING_COUNT_OPTIONS = [1, 2, 3, 4]
RING_COUNT_WEIGHTS = [2, 3, 3, 1]
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 8)
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _closed_ring_mask(height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    t = int(thickness)
    mask[:t, :] = True
    mask[-t:, :] = True
    mask[:, :t] = True
    mask[:, -t:] = True
    return mask


def _gap_mask(
    height: int,
    width: int,
    thickness: int,
    side: str,
    start: int,
    gap_len: int,
) -> np.ndarray:
    gap = np.zeros((height, width), dtype=bool)
    t = int(thickness)
    if side == "top":
        gap[:t, start : start + gap_len] = True
    elif side == "bottom":
        gap[-t:, start : start + gap_len] = True
    elif side == "left":
        gap[start : start + gap_len, :t] = True
    else:
        gap[start : start + gap_len, -t:] = True
    return gap


def _parse_ring(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    color = int(comp["color"])

    mask = np.zeros((height, width), dtype=bool)
    rel = comp["coords"] - np.array([[r0, c0]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True

    for thickness in (1, 2):
        if height <= 2 * thickness or width <= 2 * thickness:
            continue
        closed = _closed_ring_mask(height, width, thickness)
        if np.any(mask & (~closed)):
            continue

        missing = closed & (~mask)
        if not np.any(missing):
            continue

        for side in ("top", "bottom", "left", "right"):
            span = width if side in ("top", "bottom") else height
            for gap_len in (1, 2, 3):
                lo = thickness
                hi = span - thickness - gap_len
                if hi < lo:
                    continue
                for start in range(lo, hi + 1):
                    gap = _gap_mask(height, width, thickness, side, start, gap_len)
                    if not np.array_equal(missing, gap):
                        continue
                    direction = {
                        "top": (-1, 0),
                        "bottom": (1, 0),
                        "left": (0, -1),
                        "right": (0, 1),
                    }[side]
                    distance = (height if side in ("top", "bottom") else width) * thickness
                    gap_cells = [
                        (r0 + int(rr), c0 + int(cc))
                        for rr, cc in np.argwhere(gap)
                    ]
                    return {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "height": height,
                        "width": width,
                        "thickness": thickness,
                        "side": side,
                        "gap_cells": gap_cells,
                        "direction": direction,
                        "distance": int(distance),
                    }
    return None


def _component_inside_ring(comp: dict, ring: dict) -> bool:
    r0, c0, r1, c1 = ring["bbox"]
    t = int(ring["thickness"])
    for r, c in comp["coords"]:
        if not (r0 + t <= int(r) <= r1 - t and c0 + t <= int(c) <= c1 - t):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each gapped rectangular ring, eject the interior key through the gap,
    copy it outward by (outer size in that direction) * thickness, seal the gap
    with the key color, and erase the original key.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ring_infos: list[dict] = []
    key_comps: list[dict] = []
    for comp in _find_components(g, bg=BG):
        ring = _parse_ring(comp)
        if ring is None:
            key_comps.append(comp)
        else:
            ring_infos.append(ring)

    used_keys: set[int] = set()
    assignments: list[tuple[dict, dict]] = []
    for ring in ring_infos:
        matches = [
            idx
            for idx, comp in enumerate(key_comps)
            if idx not in used_keys and _component_inside_ring(comp, ring)
        ]
        assert len(matches) == 1
        key = key_comps[matches[0]]
        used_keys.add(matches[0])
        assignments.append((ring, key))

    assert len(assignments) == len(ring_infos)

    for ring, key in assignments:
        key_color = int(key["color"])
        for r, c in key["coords"]:
            out[int(r), int(c)] = BG
        for r, c in ring["gap_cells"]:
            out[int(r), int(c)] = key_color

        dr, dc = ring["direction"]
        distance = int(ring["distance"])
        for r, c in key["coords"]:
            rr = int(r + dr * distance)
            cc = int(c + dc * distance)
            out[rr, cc] = key_color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    return norm, max_r + 1, max_c + 1


def _random_key_shape(rng: random.Random, max_h: int, max_w: int) -> tuple[list[tuple[int, int]], int, int]:
    max_area = min(4, max_h * max_w)
    for _ in range(80):
        target = rng.randint(1, max_area)
        cells = {(0, 0)}
        frontier = {(-1, 0), (1, 0), (0, -1), (0, 1)}
        while len(cells) < target:
            choices = []
            for cand in frontier:
                trial = set(cells)
                trial.add(cand)
                _, h, w = _normalize_shape(trial)
                if h <= max_h and w <= max_w:
                    choices.append(cand)
            if not choices:
                break
            picked = rng.choice(choices)
            cells.add(picked)
            frontier.discard(picked)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (picked[0] + dr, picked[1] + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) == target:
            return _normalize_shape(cells)
    return [(0, 0)], 1, 1


def _cells_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expand_bbox(
    bbox: tuple[int, int, int, int], pad: int, height: int, width: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(height - 1, r1 + pad),
        min(width - 1, c1 + pad),
    )


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_ring_spec(rng: random.Random, wall_color: int, key_color: int) -> dict:
    thickness = rng.choice([1, 2])
    inner_h = rng.randint(*INNER_H_RANGE)
    inner_w = rng.randint(*INNER_W_RANGE)
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness

    side = rng.choice(["top", "bottom", "left", "right"])
    span = inner_w if side in ("top", "bottom") else inner_h
    gap_len = rng.randint(1, min(3, span))
    gap_start = rng.randint(thickness, (outer_w if side in ("top", "bottom") else outer_h) - thickness - gap_len)

    key_shape, key_h, key_w = _random_key_shape(rng, inner_h, inner_w)
    key_top = thickness + rng.randint(0, inner_h - key_h)
    key_left = thickness + rng.randint(0, inner_w - key_w)
    key_cells = [(key_top + r, key_left + c) for r, c in key_shape]

    closed = _closed_ring_mask(outer_h, outer_w, thickness)
    gap = _gap_mask(outer_h, outer_w, thickness, side, gap_start, gap_len)
    wall_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(closed & (~gap))
    ]
    gap_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(gap)
    ]

    direction = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    distance = (outer_h if side in ("top", "bottom") else outer_w) * thickness
    dr, dc = direction
    dest_cells = [(r + dr * distance, c + dc * distance) for r, c in key_cells]

    rel_rows = [0, outer_h - 1] + [r for r, _ in dest_cells]
    rel_cols = [0, outer_w - 1] + [c for _, c in dest_cells]

    return {
        "wall_color": int(wall_color),
        "key_color": int(key_color),
        "thickness": int(thickness),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "wall_cells": wall_cells,
        "gap_cells": gap_cells,
        "key_cells": key_cells,
        "dest_cells": dest_cells,
        "rel_row_min": int(min(rel_rows)),
        "rel_row_max": int(max(rel_rows)),
        "rel_col_min": int(min(rel_cols)),
        "rel_col_max": int(max(rel_cols)),
        "priority": int(outer_h * outer_w + len(dest_cells) * distance),
    }


def _place_spec(
    spec: dict, blocked: np.ndarray, height: int, width: int, rng: random.Random
) -> tuple[int, int] | None:
    top_min = -int(spec["rel_row_min"])
    top_max = height - 1 - int(spec["rel_row_max"])
    left_min = -int(spec["rel_col_min"])
    left_max = width - 1 - int(spec["rel_col_max"])
    if top_min > top_max or left_min > left_max:
        return None

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)

        ring_bbox = (top, left, top + spec["outer_h"] - 1, left + spec["outer_w"] - 1)
        dest_global = [(top + r, left + c) for r, c in spec["dest_cells"]]
        dest_bbox = _cells_bbox(dest_global)

        ring_pad = _expand_bbox(ring_bbox, 1, height, width)
        dest_pad = _expand_bbox(dest_bbox, 1, height, width)
        if not _bbox_is_free(blocked, ring_pad):
            continue
        if not _bbox_is_free(blocked, dest_pad):
            continue
        return top, left

    return None


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    ring_count = rng.choices(RING_COUNT_OPTIONS, weights=RING_COUNT_WEIGHTS, k=1)[0]

    if 2 * ring_count > len(COLORS):
        return None

    palette = COLORS[:]
    rng.shuffle(palette)
    specs = []
    for i in range(ring_count):
        specs.append(_sample_ring_spec(rng, palette[2 * i], palette[2 * i + 1]))
    specs.sort(key=lambda spec: -spec["priority"])

    blocked = np.zeros((height, width), dtype=bool)
    placements: list[tuple[dict, int, int]] = []
    for spec in specs:
        placed = _place_spec(spec, blocked, height, width, rng)
        if placed is None:
            return None
        top, left = placed
        placements.append((spec, top, left))
        ring_bbox = (top, left, top + spec["outer_h"] - 1, left + spec["outer_w"] - 1)
        dest_global = [(top + r, left + c) for r, c in spec["dest_cells"]]
        dest_bbox = _cells_bbox(dest_global)
        _mark_bbox(blocked, _expand_bbox(ring_bbox, 1, height, width))
        _mark_bbox(blocked, _expand_bbox(dest_bbox, 1, height, width))

    grid = np.full((height, width), BG, dtype=int)
    for spec, top, left in placements:
        for r, c in spec["wall_cells"]:
            grid[top + r, left + c] = spec["wall_color"]
        for r, c in spec["key_cells"]:
            grid[top + r, left + c] = spec["key_color"]

    out = grid.copy()
    for spec, top, left in placements:
        for r, c in spec["key_cells"]:
            out[top + r, left + c] = BG
        for r, c in spec["gap_cells"]:
            out[top + r, left + c] = spec["key_color"]
        for r, c in spec["dest_cells"]:
            out[top + r, left + c] = spec["key_color"]

    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_case(rng)
        if built is None:
            continue
        grid, expected = built
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
