import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
CORNERS = ("tl", "tr", "bl", "br")
SIZE_CHOICES = [22, 26, 30]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(points)
    r0, c0, _, _ = _bbox(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _norm_with_dims(points: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm = _normalize(points)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, h, w


def _rot90(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    norm, h, _ = _norm_with_dims(points)
    return _normalize((c, h - 1 - r) for r, c in norm)


def _flip_h(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    norm, _, w = _norm_with_dims(points)
    return _normalize((r, w - 1 - c) for r, c in norm)


def _orientations(points: Iterable[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = _normalize(points)
    for _ in range(4):
        for cand in (cur, _flip_h(cur)):
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        cur = _rot90(cur)
    return out


def _canonical(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_orientations(points))


def _corner_zone(corner: str, bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    zone_h = max(4, h // 3)
    zone_w = max(4, w // 3)
    if corner == "tl":
        return r1 < zone_h and c1 < zone_w
    if corner == "tr":
        return r1 < zone_h and c0 >= w - zone_w
    if corner == "bl":
        return r0 >= h - zone_h and c1 < zone_w
    if corner == "br":
        return r0 >= h - zone_h and c0 >= w - zone_w
    raise ValueError(corner)


def _corner_distance(corner: str, bbox: tuple[int, int, int, int], h: int, w: int) -> int:
    r0, c0, r1, c1 = bbox
    if corner == "tl":
        return r0 + c0
    if corner == "tr":
        return r0 + (w - 1 - c1)
    if corner == "bl":
        return (h - 1 - r1) + c0
    if corner == "br":
        return (h - 1 - r1) + (w - 1 - c1)
    raise ValueError(corner)


def _quadrant_bounds(h: int, w: int, corner: str) -> tuple[int, int, int, int]:
    mid_r = h // 2
    mid_c = w // 2
    if corner == "tl":
        return 0, mid_r - 1, 0, mid_c - 1
    if corner == "tr":
        return 0, mid_r - 1, mid_c, w - 1
    if corner == "bl":
        return mid_r, h - 1, 0, mid_c - 1
    if corner == "br":
        return mid_r, h - 1, mid_c, w - 1
    raise ValueError(corner)


def _target_top_left(h: int, w: int, corner: str, obj_h: int, obj_w: int) -> tuple[int, int]:
    qr0, qr1, qc0, qc1 = _quadrant_bounds(h, w, corner)
    center2_r = qr0 + qr1
    center2_c = qc0 + qc1
    assert (center2_r - (obj_h - 1)) % 2 == 0
    assert (center2_c - (obj_w - 1)) % 2 == 0
    r0 = (center2_r - (obj_h - 1)) // 2
    c0 = (center2_c - (obj_w - 1)) // 2
    return r0, c0


def _extract_legends(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    grey_comps = _components(grid == GREY)
    legends: list[dict] = []
    used_ids: set[int] = set()

    for corner in CORNERS:
        candidates = []
        for idx, comp in enumerate(grey_comps):
            bb = _bbox(comp)
            if _corner_zone(corner, bb, h, w):
                candidates.append((_corner_distance(corner, bb, h, w), idx, comp))
        assert candidates, f"missing grey legend in {corner}"
        candidates.sort(key=lambda item: item[0])
        _, idx, comp = candidates[0]
        assert idx not in used_ids, "corner legends must be distinct"
        used_ids.add(idx)

        comp_set = set(comp)
        color_pixels = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp_set:
                    val = int(grid[nr, nc])
                    if val != BG and val != GREY:
                        color_pixels.add((nr, nc))
        assert len(color_pixels) == 1, "legend must have a unique adjacent replacement pixel"
        color_pos = next(iter(color_pixels))
        legends.append(
            {
                "corner": corner,
                "shape": _normalize(comp),
                "shape_orientations": set(_orientations(comp)),
                "color": int(grid[color_pos]),
                "cells": comp_set | {color_pos},
            }
        )
    return legends


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legends = _extract_legends(g)
    reserved = set().union(*(legend["cells"] for legend in legends))

    obj_mask = g != BG
    for r, c in reserved:
        obj_mask[r, c] = False
    objects = _components(obj_mask)
    assert objects, "grid must contain at least one foreground object"

    out = np.full_like(g, BG)
    written = np.zeros_like(g, dtype=bool)
    legend_by_corner = {legend["corner"]: legend for legend in legends}

    for comp in objects:
        shape = _normalize(comp)
        matches = [corner for corner, legend in legend_by_corner.items() if shape in legend["shape_orientations"]]
        assert len(matches) == 1, "each object must match exactly one legend shape"
        corner = matches[0]
        color = legend_by_corner[corner]["color"]

        r0, c0, r1, c1 = _bbox(comp)
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        new_r0, new_c0 = _target_top_left(h, w, corner, obj_h, obj_w)

        for r, c in comp:
            rr = new_r0 + (r - r0)
            cc = new_c0 + (c - c0)
            assert 0 <= rr < h and 0 <= cc < w
            assert not written[rr, cc], "translated objects must not overlap"
            written[rr, cc] = True
            out[rr, cc] = color

    return out


def _enumerate_shapes() -> list[tuple[tuple[int, int], ...]]:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for mask in range(1, 1 << 9):
        pts = [(i // 3, i % 3) for i in range(9) if (mask >> i) & 1]
        if not (4 <= len(pts) <= 6):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if (r0, c0, r1, c1) != (0, 0, 2, 2):
            continue
        if len(_components(np.array([[((r, c) in set(pts)) for c in range(3)] for r in range(3)], dtype=bool))) != 1:
            continue
        canon = _canonical(pts)
        if canon not in seen:
            seen.add(canon)
            shapes.append(canon)
    assert len(shapes) >= 4
    return shapes


SHAPE_LIBRARY = _enumerate_shapes()


def _random_orientation(shape: tuple[tuple[int, int], ...], rng: random.Random) -> tuple[tuple[int, int], ...]:
    return rng.choice(_orientations(shape))


def _paint(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _legend_anchor(corner: str, h: int, w: int, rng: random.Random) -> tuple[int, int]:
    jr = rng.randint(0, 1)
    jc = rng.randint(0, 1)
    if corner == "tl":
        return 1 + jr, 1 + jc
    if corner == "tr":
        return 1 + jr, w - 4 - jc
    if corner == "bl":
        return h - 4 - jr, 1 + jc
    if corner == "br":
        return h - 4 - jr, w - 4 - jc
    raise ValueError(corner)


def _legend_color_candidates(
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    corner: str,
    h: int,
    w: int,
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    dirs = {
        "tl": [(0, 1), (1, 0)],
        "tr": [(0, -1), (1, 0)],
        "bl": [(0, 1), (-1, 0)],
        "br": [(0, -1), (-1, 0)],
    }[corner]
    grey_cells = {(top + dr, left + dc) for dr, dc in shape}
    cand = set()
    for r, c in grey_cells:
        for dr, dc in dirs:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in grey_cells and grid[rr, cc] == BG:
                cand.add((rr, cc))
    return sorted(cand)


def _can_place_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, pad: int = 1) -> bool:
    h, w = grid.shape
    cells = {(top + dr, left + dc) for dr, dc in shape}
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells and grid[rr, cc] != BG:
                    return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(SIZE_CHOICES)
    w = rng.choice(SIZE_CHOICES)
    grid = np.full((h, w), BG, dtype=int)

    shapes = rng.sample(SHAPE_LIBRARY, 4)
    rng.shuffle(shapes)
    legend_shapes = dict(zip(CORNERS, shapes))
    replacement_colors = rng.sample(PALETTE, 4)
    legend_colors = dict(zip(CORNERS, replacement_colors))

    for corner in CORNERS:
        shape = _random_orientation(legend_shapes[corner], rng)
        top, left = _legend_anchor(corner, h, w, rng)
        _paint(grid, shape, top, left, GREY)
        candidates = _legend_color_candidates(shape, top, left, corner, h, w, grid)
        if not candidates:
            return None
        rr, cc = rng.choice(candidates)
        grid[rr, cc] = legend_colors[corner]

    object_count = rng.choice([2, 3, 4])
    used_corners = rng.sample(list(CORNERS), object_count)
    input_colors = rng.sample([c for c in PALETTE if c not in replacement_colors], object_count)

    row_lo = h // 4
    row_hi = h - h // 4 - 3
    col_lo = w // 4
    col_hi = w - w // 4 - 3
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for corner, color in zip(used_corners, input_colors):
        shape = _random_orientation(legend_shapes[corner], rng)
        placed = False
        for _ in range(80):
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)
            if _can_place_shape(grid, shape, top, left, pad=1):
                _paint(grid, shape, top, left, color)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
