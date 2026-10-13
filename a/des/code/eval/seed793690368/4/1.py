from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _ray_cells(
    seed: tuple[int, int],
    gap: tuple[int, int],
    dr: int,
    dc: int,
    length: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    r, c = seed
    while len(cells) < length:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return []
        if (r, c) == gap:
            continue
        cells.append((r, c))
    return cells


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    objects: list[dict] = []

    for comp in _components(g != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perimeter = _perimeter_coords(r0, c0, r1, c1)
        if any(int(g[r, c]) == bg for r, c in perimeter):
            continue

        border_counts = Counter(int(g[r, c]) for r, c in perimeter)
        if len(border_counts) != 2:
            continue
        border_color, border_count = border_counts.most_common(1)[0]
        if border_count != len(perimeter) - 1:
            continue
        gap_color = next(color for color, count in border_counts.items() if count == 1)
        gap = next((r, c) for r, c in perimeter if int(g[r, c]) == gap_color)

        interior: list[tuple[int, int]] = []
        seeds: list[tuple[int, int]] = []
        valid = True
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                v = int(g[r, c])
                interior.append((r, c))
                if v == gap_color:
                    seeds.append((r, c))
                elif v != bg:
                    valid = False
                    break
            if not valid:
                break
        if not valid or len(seeds) != 1:
            continue

        seed = seeds[0]
        sr, sc = seed
        gr, gc = gap
        if sr != gr and sc != gc:
            continue
        dr = 0 if sr == gr else (-1 if gr < sr else 1)
        dc = 0 if sc == gc else (-1 if gc < sc else 1)
        if abs(dr) + abs(dc) != 1:
            continue

        rr, cc = sr + dr, sc + dc
        clear = True
        while (rr, cc) != gap:
            if not (0 <= rr < h and 0 <= cc < w) or int(g[rr, cc]) != bg:
                clear = False
                break
            rr += dr
            cc += dc
        if not clear:
            continue

        n_bg = sum(1 for r, c in interior if int(g[r, c]) == bg)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border_color": border_color,
                "gap_color": gap_color,
                "gap": gap,
                "seed": seed,
                "dir": (dr, dc),
                "path_len": n_bg,
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"], obj["gap"]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each hollow rectangle, pair the unique border gap color with the same-colored
    seed inside, then shoot a straight path from the seed through the gap.

    The path length equals the number of background cells inside that rectangle.
    The gap cell itself is restored to background in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for obj in _find_rectangles(g):
        gap = obj["gap"]
        seed = obj["seed"]
        dr, dc = obj["dir"]
        path_len = int(obj["path_len"])
        color = int(obj["gap_color"])

        out[gap] = bg
        for r, c in _ray_cells(seed, gap, dr, dc, path_len, g.shape):
            out[r, c] = color

    return out


# --------------- generator ---------------

GRID_MIN = 22
GRID_MAX = 30
INNER_H_RANGE = (2, 4)
INNER_W_RANGE = (2, 4)
OBJECT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
MAX_ATTEMPTS = 400
SPEC_TRIES = 120


def _pad_cells(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for nc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                out.add((nr, nc))
    return out


def _side_seed_ranges(side: str, inner_h: int, inner_w: int) -> tuple[list[int], list[int]]:
    if side == "top":
        rows = list(range(0, max(1, (inner_h + 1) // 2)))
        cols = list(range(inner_w))
    elif side == "bottom":
        rows = list(range(inner_h // 2, inner_h))
        cols = list(range(inner_w))
    elif side == "left":
        rows = list(range(inner_h))
        cols = list(range(0, max(1, (inner_w + 1) // 2)))
    else:
        rows = list(range(inner_h))
        cols = list(range(inner_w // 2, inner_w))
    return rows, cols


def _make_candidate(
    h: int,
    w: int,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    seed_row: int,
    seed_col: int,
    border_color: int,
    gap_color: int,
) -> dict | None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    perimeter = set(_perimeter_coords(r0, c0, r1, c1))

    if side == "top":
        gap = (r0, c0 + 1 + seed_col)
        inside_used = seed_row
        dr, dc = -1, 0
    elif side == "bottom":
        gap = (r1, c0 + 1 + seed_col)
        inside_used = inner_h - 1 - seed_row
        dr, dc = 1, 0
    elif side == "left":
        gap = (r0 + 1 + seed_row, c0)
        inside_used = seed_col
        dr, dc = 0, -1
    else:
        gap = (r0 + 1 + seed_row, c1)
        inside_used = inner_w - 1 - seed_col
        dr, dc = 0, 1

    seed = (r0 + 1 + seed_row, c0 + 1 + seed_col)
    path_len = inner_h * inner_w - 1
    path_cells = _ray_cells(seed, gap, dr, dc, path_len, (h, w))
    if len(path_cells) != path_len:
        return None

    nonbg = set(perimeter)
    nonbg.add(seed)
    if seed in perimeter and seed != gap:
        return None

    path_set = set(path_cells)
    if seed in path_set or gap in path_set:
        return None
    if nonbg & path_set:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "bbox_cells": bbox_cells,
        "bbox_pad": _pad_cells(bbox_cells, (h, w), pad=1),
        "perimeter": perimeter,
        "nonbg": nonbg,
        "path": path_set,
        "path_pad": _pad_cells(path_set, (h, w), pad=1),
        "border_color": border_color,
        "gap_color": gap_color,
        "gap": gap,
        "seed": seed,
        "dir": (dr, dc),
        "path_len": path_len,
        "inside_used": inside_used,
    }


def _placement_ranges(
    h: int,
    w: int,
    inner_h: int,
    inner_w: int,
    side: str,
    seed_row: int,
    seed_col: int,
) -> tuple[range, range]:
    path_len = inner_h * inner_w - 1
    if side == "top":
        outside_needed = path_len - seed_row
        return range(outside_needed, h - (inner_h + 2) + 1), range(0, w - (inner_w + 2) + 1)
    if side == "bottom":
        outside_needed = path_len - (inner_h - 1 - seed_row)
        return range(0, h - (inner_h + 2) - outside_needed + 1), range(0, w - (inner_w + 2) + 1)
    if side == "left":
        outside_needed = path_len - seed_col
        return range(0, h - (inner_h + 2) + 1), range(outside_needed, w - (inner_w + 2) + 1)

    outside_needed = path_len - (inner_w - 1 - seed_col)
    return range(0, h - (inner_h + 2) + 1), range(0, w - (inner_w + 2) - outside_needed + 1)


def _try_place_object(
    rng: random.Random,
    shape: tuple[int, int],
    side: str,
    border_color: int,
    gap_color: int,
    forbidden: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(SPEC_TRIES):
        inner_h = rng.randint(INNER_H_RANGE[0], INNER_H_RANGE[1])
        inner_w = rng.randint(INNER_W_RANGE[0], INNER_W_RANGE[1])
        rows, cols = _side_seed_ranges(side, inner_h, inner_w)
        if not rows or not cols:
            continue

        center = (inner_h // 2, inner_w // 2)
        choices = [(sr, sc) for sr in rows for sc in cols if (sr, sc) != center]
        if not choices:
            continue
        seed_row, seed_col = rng.choice(choices)

        row_range, col_range = _placement_ranges(h, w, inner_h, inner_w, side, seed_row, seed_col)
        positions = [(r0, c0) for r0 in row_range for c0 in col_range]
        rng.shuffle(positions)
        for r0, c0 in positions:
            cand = _make_candidate(
                h,
                w,
                r0,
                c0,
                inner_h,
                inner_w,
                side,
                seed_row,
                seed_col,
                border_color,
                gap_color,
            )
            if cand is None:
                continue
            if cand["bbox_cells"] & forbidden:
                continue
            if cand["path"] & forbidden:
                continue
            return cand
    return None


def _render_grid(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for obj in placed:
        for r, c in obj["perimeter"]:
            grid[r, c] = int(obj["border_color"])
        gr, gc = obj["gap"]
        sr, sc = obj["seed"]
        grid[gr, gc] = int(obj["gap_color"])
        grid[sr, sc] = int(obj["gap_color"])
    return grid


def _validate_generated(grid: np.ndarray, placed: list[dict]) -> bool:
    found = _find_rectangles(grid)
    if len(found) != len(placed):
        return False

    expected = {
        (obj["bbox"], obj["gap"], obj["seed"], obj["gap_color"], obj["path_len"])
        for obj in placed
    }
    actual = {
        (obj["bbox"], obj["gap"], obj["seed"], obj["gap_color"], obj["path_len"])
        for obj in found
    }
    if actual != expected:
        return False

    bg = _get_bg(grid)
    out = main(grid)
    if _get_bg(out) != bg:
        return False

    for obj in placed:
        if int(out[obj["gap"]]) != bg:
            return False
        if int(out[obj["seed"]]) != int(obj["gap_color"]):
            return False
        for r, c in obj["path"]:
            if int(out[r, c]) != int(obj["gap_color"]):
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        n_obj = rng.choice(OBJECT_COUNT_CHOICES)
        base = GRID_MIN + 2 * max(0, n_obj - 2)
        h = rng.randint(base, GRID_MAX)
        w = rng.randint(base, GRID_MAX)

        n_border_colors = rng.randint(1, min(2, n_obj))
        colors = list(rng.sample(PALETTE, n_obj + n_border_colors))
        gap_colors = colors[:n_obj]
        border_palette = colors[n_obj:]
        border_colors = [rng.choice(border_palette) for _ in range(n_obj)]

        sides = list(rng.sample(SIDES, k=min(len(SIDES), n_obj)))
        while len(sides) < n_obj:
            sides.append(rng.choice(SIDES))
        rng.shuffle(sides)

        placed: list[dict] = []
        forbidden: set[tuple[int, int]] = set()
        success = True

        for idx in range(n_obj):
            cand = _try_place_object(
                rng,
                (h, w),
                sides[idx],
                border_colors[idx],
                gap_colors[idx],
                forbidden,
            )
            if cand is None:
                success = False
                break
            placed.append(cand)
            forbidden |= cand["bbox_pad"]
            forbidden |= cand["path_pad"]

        if not success:
            continue

        grid = _render_grid((h, w), placed)
        if _validate_generated(grid, placed):
            return grid

    raise RuntimeError("generate_input: exceeded retry budget")
