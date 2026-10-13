from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _ring_path(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    path.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _is_ring_component(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = set(_ring_path(r0, c0, r1, c1))
    return len(points) == len(border) and set(points) == border


def _is_bar_component(grid: np.ndarray, points: list[tuple[int, int]]) -> bool:
    if not (2 <= len(points) <= 4):
        return False
    r0, c0, r1, c1 = _bbox(points)
    if r0 != r1 and c0 != c1:
        return False
    length = max(r1 - r0 + 1, c1 - c0 + 1)
    if len(points) != length:
        return False
    colors = [int(grid[r, c]) for r, c in points]
    if any(color in (BLACK, GREY) for color in colors):
        return False
    return len(set(colors)) == len(colors)


def _read_bar_template(grid: np.ndarray, points: list[tuple[int, int]]) -> list[int]:
    r0, c0, r1, c1 = _bbox(points)
    if r0 == r1:
        ordered = sorted(points, key=lambda p: p[1])
    else:
        ordered = sorted(points, key=lambda p: p[0])
    return [int(grid[r, c]) for r, c in ordered]


def _component_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = None
    for ar, ac in a:
        for br, bc in b:
            dist = abs(ar - br) + abs(ac - bc)
            if best is None or dist < best:
                best = dist
    assert best is not None
    return int(best)


def _extract_objects(grid: np.ndarray):
    rings = []
    bars = []
    for points in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox(points)
        if _is_ring_component(points):
            rings.append(
                {
                    "cells": points,
                    "bbox": (r0, c0, r1, c1),
                    "path": _ring_path(r0, c0, r1, c1),
                }
            )
        elif _is_bar_component(grid, points):
            bars.append({"cells": points, "bbox": (r0, c0, r1, c1)})
    return rings, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular bead track by cycling the nearby color bar template.

    The bar nearest to a track defines its repeating color order. Starting from the
    track's top-left corner and walking clockwise, the first already-colored bead
    fixes the template phase; the rest of the grey track cells inherit colors by
    continuing that cycle around the whole loop.
    """
    g = np.asarray(grid, dtype=int)
    rings, bars = _extract_objects(g)
    assert rings and bars

    ring_to_bars: dict[int, list[dict]] = {i: [] for i in range(len(rings))}
    for bar in bars:
        best_idx = None
        best_dist = None
        for idx, ring in enumerate(rings):
            dist = _component_distance(bar["cells"], ring["cells"])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        assert best_idx is not None
        ring_to_bars[best_idx].append(bar)

    out = g.copy()
    for idx, ring in enumerate(rings):
        assigned = ring_to_bars[idx]
        assert len(assigned) == 1
        template = _read_bar_template(g, assigned[0]["cells"])
        path = ring["path"]

        start_idx = None
        for pos_idx, (r, c) in enumerate(path):
            if int(g[r, c]) != GREY:
                start_idx = pos_idx
                break
        assert start_idx is not None

        bead_color = int(g[path[start_idx]])
        assert bead_color in template
        phase = template.index(bead_color)

        for pos_idx, (r, c) in enumerate(path):
            out[r, c] = template[(phase + pos_idx - start_idx) % len(template)]

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RING_H_RANGE = (4, 9)
RING_W_RANGE = (4, 11)
RING_COUNT_CHOICES = (1, 2, 3, 4)
TEMPLATE_LENGTHS = (2, 3, 4)
PAD = 2
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _padded_cells(cells: list[tuple[int, int]], h: int, w: int, pad: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _place_object(
    rng: random.Random,
    h: int,
    w: int,
    forbidden: np.ndarray,
) -> dict | None:
    side = rng.choice(SIDES)
    template_len = rng.choice(TEMPLATE_LENGTHS)

    if side in ("top", "bottom"):
        ring_h = rng.randint(RING_H_RANGE[0], RING_H_RANGE[1])
        ring_w = rng.randint(max(RING_W_RANGE[0], template_len), RING_W_RANGE[1])
    else:
        ring_h = rng.randint(max(RING_H_RANGE[0], template_len), RING_H_RANGE[1])
        ring_w = rng.randint(RING_W_RANGE[0], RING_W_RANGE[1])

    if side == "top":
        r0_min, r0_max = 3, h - ring_h - 1
        c0_min, c0_max = 1, w - ring_w - 1
    elif side == "bottom":
        r0_min, r0_max = 1, h - ring_h - 3
        c0_min, c0_max = 1, w - ring_w - 1
    elif side == "left":
        r0_min, r0_max = 1, h - ring_h - 1
        c0_min, c0_max = 3, w - ring_w - 1
    else:
        r0_min, r0_max = 1, h - ring_h - 1
        c0_min, c0_max = 1, w - ring_w - 3

    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = rng.randint(r0_min, r0_max)
    c0 = rng.randint(c0_min, c0_max)
    r1 = r0 + ring_h - 1
    c1 = c0 + ring_w - 1
    path = _ring_path(r0, c0, r1, c1)

    if side == "top":
        bar_row = r0 - 2
        bar_col = rng.randint(c0, c1 - template_len + 1)
        bar_cells = [(bar_row, bar_col + i) for i in range(template_len)]
    elif side == "bottom":
        bar_row = r1 + 2
        bar_col = rng.randint(c0, c1 - template_len + 1)
        bar_cells = [(bar_row, bar_col + i) for i in range(template_len)]
    elif side == "left":
        bar_col = c0 - 2
        bar_row = rng.randint(r0, r1 - template_len + 1)
        bar_cells = [(bar_row + i, bar_col) for i in range(template_len)]
    else:
        bar_col = c1 + 2
        bar_row = rng.randint(r0, r1 - template_len + 1)
        bar_cells = [(bar_row + i, bar_col) for i in range(template_len)]

    actual_cells = path + bar_cells
    if any(forbidden[r, c] for r, c in actual_cells):
        return None

    template = rng.sample(PALETTE_COLORS, template_len)
    start_idx = rng.randint(0, len(path) - 2)
    remaining = len(path) - start_idx
    bead_count = rng.randint(2, min(8, remaining))
    bead_indices = {start_idx}
    if bead_count > 1:
        bead_indices.update(rng.sample(range(start_idx + 1, len(path)), bead_count - 1))
    phase = rng.randrange(template_len)

    return {
        "side": side,
        "template": template,
        "path": path,
        "bar_cells": bar_cells,
        "actual_cells": actual_cells,
        "start_idx": start_idx,
        "bead_indices": bead_indices,
        "phase": phase,
    }


def _expected_output(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for obj in objects:
        template = obj["template"]
        for idx, (r, c) in enumerate(obj["path"]):
            expected[r, c] = template[(obj["phase"] + idx - obj["start_idx"]) % len(template)]
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    area = h * w
    if area >= 600:
        max_rings = 4
    elif area >= 450:
        max_rings = 3
    elif area >= 280:
        max_rings = 2
    else:
        max_rings = 1
    ring_target = rng.choice([n for n in RING_COUNT_CHOICES if n <= max_rings])

    objects: list[dict] = []
    for _ in range(ring_target):
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            candidate = _place_object(rng, h, w, forbidden)
            if candidate is None:
                continue
            placed = candidate
            objects.append(candidate)
            for r, c in _padded_cells(candidate["actual_cells"], h, w, PAD):
                forbidden[r, c] = True
            break
        if placed is None:
            return None

    for obj in objects:
        for r, c in obj["path"]:
            grid[r, c] = GREY
        for idx in obj["bead_indices"]:
            r, c = obj["path"][idx]
            grid[r, c] = obj["template"][(obj["phase"] + idx - obj["start_idx"]) % len(obj["template"])]
        for (r, c), color in zip(obj["bar_cells"], obj["template"]):
            grid[r, c] = color

    expected = _expected_output(grid, objects)
    if not np.array_equal(main(grid), expected):
        return None

    rings, bars = _extract_objects(grid)
    if len(rings) != ring_target or len(bars) != ring_target:
        return None

    for obj in objects:
        if sum(int(grid[r, c] != GREY) for r, c in obj["path"]) < 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
