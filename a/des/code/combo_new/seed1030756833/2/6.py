from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rs = [rr for rr, _cc in coords]
            cs = [cc for _rr, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "cells": set(coords),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _arrow_infos(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> list[dict]:
    """Return every valid interpretation of a line-plus-tip arrow component."""
    pts = set((int(r), int(c)) for r, c in coords)
    n = len(pts)
    if not (3 <= n <= 6):
        return []

    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    infos: list[dict] = []

    if height == 2 and width == n - 1:
        row_cols = {r: sorted(c for rr, c in pts if rr == r) for r in (r0, r1)}
        shaft_rows = [
            r
            for r, cols in row_cols.items()
            if len(cols) == width and cols == list(range(c0, c1 + 1))
        ]
        tip_rows = [r for r, cols in row_cols.items() if len(cols) == 1]
        if len(shaft_rows) == 1 and len(tip_rows) == 1:
            shaft_r = shaft_rows[0]
            tip_r = tip_rows[0]
            tip_c = row_cols[tip_r][0]
            if tip_c in (c0, c1) and (shaft_r, tip_c) in pts:
                infos.append(
                    {
                        "cells": pts,
                        "length": width,
                        "tip": (tip_r, tip_c),
                        "direction": (tip_r - shaft_r, 0),
                    }
                )

    if width == 2 and height == n - 1:
        col_rows = {c: sorted(r for r, cc in pts if cc == c) for c in (c0, c1)}
        shaft_cols = [
            c
            for c, rows in col_rows.items()
            if len(rows) == height and rows == list(range(r0, r1 + 1))
        ]
        tip_cols = [c for c, rows in col_rows.items() if len(rows) == 1]
        if len(shaft_cols) == 1 and len(tip_cols) == 1:
            shaft_c = shaft_cols[0]
            tip_c = tip_cols[0]
            tip_r = col_rows[tip_c][0]
            if tip_r in (r0, r1) and (tip_r, shaft_c) in pts:
                infos.append(
                    {
                        "cells": pts,
                        "length": height,
                        "tip": (tip_r, tip_c),
                        "direction": (0, tip_c - shaft_c),
                    }
                )

    unique: list[dict] = []
    seen = set()
    for info in infos:
        key = (info["length"], info["tip"], info["direction"])
        if key not in seen:
            seen.add(key)
            unique.append(info)
    return unique


def _direction_score(info: dict, obj: dict) -> tuple[int, int, int, int] | None:
    dr, dc = info["direction"]
    tr, tc = info["tip"]
    coords = obj["coords"]

    projections = [(r - tr) * dr + (c - tc) * dc for r, c in coords]
    if min(projections) <= 0:
        return None

    r0, c0, r1, c1 = obj["bbox"]
    if dr:
        if c0 <= tc <= c1:
            lateral = 0
        else:
            lateral = min(abs(tc - c0), abs(tc - c1))
    else:
        if r0 <= tr <= r1:
            lateral = 0
        else:
            lateral = min(abs(tr - r0), abs(tr - r1))

    return (int(lateral), int(min(projections)), int(obj["area"]), int(r0 + c0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored target object away from the matching arrow by the arrow's encoded length.

    Input structure:
    - The background is the dominant color.
    - Each non-background color forms one small connected target object and one nearby same-color arrow.
    - An arrow is a horizontal or vertical line segment of length 2-5 plus a single orthogonal tip pixel attached to one segment endpoint.
    - The tip side of the arrow points toward the same-color target object.

    Rule:
    - For each color, identify the arrow component and the same-color target object component.
    - Decode the arrow direction from the segment endpoint toward the single-pixel tip.
    - Decode the movement distance as the straight segment length, excluding the tip pixel.
    - Clear the arrow and the target object's original cells, then translate the object by that distance in the opposite direction.

    Key insight: The arrow encodes both direction and distance, but the object moves opposite the direction in which the arrow points.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)

    by_color: dict[int, list[dict]] = {}
    for comp in _components(g, bg):
        by_color.setdefault(int(comp["color"]), []).append(comp)

    moves: list[dict] = []
    for color, comps in by_color.items():
        used_arrows: set[int] = set()
        used_objects: set[int] = set()
        while True:
            best = None
            for ai, arrow_comp in enumerate(comps):
                if ai in used_arrows:
                    continue
                for info in _arrow_infos(arrow_comp["coords"]):
                    for oi, obj in enumerate(comps):
                        if oi == ai or oi in used_objects:
                            continue
                        score = _direction_score(info, obj)
                        if score is None:
                            continue
                        key = (score, ai, oi, info)
                        if best is None or key[:3] < best[:3]:
                            best = key
            if best is None:
                break

            _score, ai, oi, info = best
            used_arrows.add(int(ai))
            used_objects.add(int(oi))
            moves.append(
                {
                    "color": int(color),
                    "arrow": comps[ai]["coords"],
                    "object": comps[oi]["coords"],
                    "direction": info["direction"],
                    "length": int(info["length"]),
                }
            )

    out = g.copy()
    for move in moves:
        for r, c in move["arrow"]:
            out[int(r), int(c)] = bg
        for r, c in move["object"]:
            out[int(r), int(c)] = bg

    h, w = out.shape
    for move in moves:
        dr, dc = move["direction"]
        dist = int(move["length"])
        color = int(move["color"])
        for r, c in move["object"]:
            nr = int(r) - int(dr) * dist
            nc = int(c) - int(dc) * dist
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color
    return out


# -------------------------- generator --------------------------

_PALETTE = tuple(range(10))
_GEN_ATTEMPTS = 500
_PAIR_ATTEMPTS = 500


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    return {
        (int(top + r), int(left + c))
        for r, c in np.argwhere(mask)
    }


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    rel = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _c in rel) + 1
    w = max(c for _r, c in rel) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in rel:
        mask[r, c] = True
    return mask


def _random_object_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        if h * w < 2:
            continue
        target = random.randint(2, min(7, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = _normalize_cells(cells)
        if mask.shape[0] > 3 or mask.shape[1] > 3:
            continue
        if _arrow_infos(_mask_cells(mask, 0, 0)):
            continue
        return mask
    return np.array([[1, 1], [1, 1]], dtype=bool)


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int], length: int, side: int) -> set[tuple[int, int]]:
    tr, tc = tip
    dr, dc = direction
    endpoint = (int(tr - dr), int(tc - dc))
    if dr:
        step = (0, int(side))
    else:
        step = (int(side), 0)
    return {
        (int(endpoint[0] + i * step[0]), int(endpoint[1] + i * step[1]))
        for i in range(int(length))
    } | {(int(tr), int(tc))}


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _dilated(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        if (r, c) in bset:
            return True
        for dr, dc in DIRS8:
            if (r + dr, c + dc) in bset:
                return True
    return False


def _tip_for_object(
    top: int,
    left: int,
    mask: np.ndarray,
    direction: tuple[int, int],
    gap: int,
) -> tuple[int, int]:
    h, w = mask.shape
    dr, dc = direction
    if dr == 1:
        return int(top - gap), int(left + random.randrange(w))
    if dr == -1:
        return int(top + h - 1 + gap), int(left + random.randrange(w))
    if dc == 1:
        return int(top + random.randrange(h)), int(left - gap)
    return int(top + random.randrange(h)), int(left + w - 1 + gap)


def _try_place_pair(
    inp: np.ndarray,
    expected: np.ndarray,
    input_reserved: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
) -> bool:
    h, w = inp.shape
    for _ in range(_PAIR_ATTEMPTS):
        mask = _random_object_mask()
        mh, mw = mask.shape
        length = random.randint(2, 5)
        top = random.randint(0, h - mh)
        left = random.randint(0, w - mw)
        dr, dc = direction
        out_top = top - dr * length
        out_left = left - dc * length
        if not (0 <= out_top <= h - mh and 0 <= out_left <= w - mw):
            continue

        obj_cells = _mask_cells(mask, top, left)
        moved_cells = _mask_cells(mask, out_top, out_left)
        if obj_cells & input_reserved or moved_cells & output_reserved:
            continue

        gap = random.randint(2, 4)
        tip = _tip_for_object(top, left, mask, direction, gap)
        sides = [-1, 1]
        random.shuffle(sides)
        for side in sides:
            arr_cells = _arrow_cells(tip, direction, length, side)
            if not _in_bounds(arr_cells, h, w):
                continue
            if arr_cells & input_reserved:
                continue
            if arr_cells & obj_cells or _touches(arr_cells, obj_cells):
                continue

            pair_cells = obj_cells | arr_cells
            if pair_cells & input_reserved or moved_cells & output_reserved:
                continue

            for r, c in obj_cells | arr_cells:
                inp[r, c] = int(color)
            for r, c in moved_cells:
                expected[r, c] = int(color)

            input_reserved |= _dilated(pair_cells, h, w)
            output_reserved |= _dilated(moved_cells, h, w)
            return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(14, 30)
    w = random.randint(14, 30)
    bg = random.choice(_PALETTE)
    inp = np.full((h, w), int(bg), dtype=int)
    expected = np.full((h, w), int(bg), dtype=int)

    max_pairs = min(5, len(_PALETTE) - 1)
    pair_count = random.randint(2, max_pairs)
    colors = random.sample([c for c in _PALETTE if c != bg], pair_count)

    direction_pool = list(DIRS4)
    random.shuffle(direction_pool)
    while len(direction_pool) < pair_count:
        extra = list(DIRS4)
        random.shuffle(extra)
        direction_pool.extend(extra)
    directions = direction_pool[:pair_count]

    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()
    order = list(range(pair_count))
    random.shuffle(order)
    for i in order:
        if not _try_place_pair(inp, expected, input_reserved, output_reserved, colors[i], directions[i]):
            return None

    if int(Counter(map(int, inp.ravel())).most_common(1)[0][0]) != bg:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"generate_input failed after {_GEN_ATTEMPTS} attempts")
