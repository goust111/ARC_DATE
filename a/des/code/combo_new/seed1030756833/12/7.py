from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 800


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _color_components(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "shape": _shape_key(coords),
                }
            )
    return comps


def _strip_candidates(g: np.ndarray) -> list[dict[str, object]]:
    h, w = g.shape
    out: list[dict[str, object]] = []
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))

    for r, c in corners:
        dc = 1 if c == 0 else -1
        cells: list[tuple[int, int]] = []
        cc = c
        while 0 <= cc < w and int(g[r, cc]) != BG:
            cells.append((r, cc))
            cc += dc
        if len(cells) >= 2:
            raw = [int(g[rr, cc]) for rr, cc in cells]
            if len(set(raw)) == len(raw):
                seq = [int(g[rr, cc]) for rr, cc in sorted(cells, key=lambda p: p[1])]
                out.append({"cells": tuple(cells), "sequence": tuple(seq), "orientation": "h"})

        dr = 1 if r == 0 else -1
        cells = []
        rr = r
        while 0 <= rr < h and int(g[rr, c]) != BG:
            cells.append((rr, c))
            rr += dr
        if len(cells) >= 2:
            raw = [int(g[rr, cc]) for rr, cc in cells]
            if len(set(raw)) == len(raw):
                seq = [int(g[rr, cc]) for rr, cc in sorted(cells, key=lambda p: p[0])]
                out.append({"cells": tuple(cells), "sequence": tuple(seq), "orientation": "v"})
    return out


def _parse_grid(g: np.ndarray) -> tuple[tuple[int, ...], list[tuple[dict[str, object], dict[str, object]]]]:
    valid: list[tuple[tuple[int, ...], list[tuple[dict[str, object], dict[str, object]]]]] = []

    for cand in _strip_candidates(g):
        work = g.copy()
        for cell in cand["cells"]:
            work[cell] = BG

        comps = _color_components(work)
        if not comps or len(comps) % 2:
            continue
        seq = tuple(int(v) for v in cand["sequence"])
        if set(comp["color"] for comp in comps) != set(seq):
            continue
        if any(not (2 <= len(comp["coords"]) <= 6) for comp in comps):
            continue

        groups: dict[object, list[dict[str, object]]] = defaultdict(list)
        for comp in comps:
            groups[comp["shape"]].append(comp)

        pairs: list[tuple[dict[str, object], dict[str, object]]] = []
        ok = True
        for key in sorted(groups):
            group = groups[key]
            if len(group) != 2 or int(group[0]["color"]) == int(group[1]["color"]):
                ok = False
                break
            pairs.append((group[0], group[1]))
        if ok:
            valid.append((seq, pairs))

    assert valid, "expected one corner color-order strip and paired objects"
    valid.sort(key=lambda item: (len(item[0]), len(item[1])), reverse=True)
    return valid[0]


def _center(comp: dict[str, object]) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return (int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2


def _paint_segment(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        out[ar, c0 : c1 + 1] = int(color)
    else:
        assert ac == bc, "segment must be axis-aligned"
        r0, r1 = sorted((ar, br))
        out[r0 : r1 + 1, ac] = int(color)


def _paint_marker(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = out.shape
    cr, cc = center
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of translated equal-shape objects using the middle color from a corner order strip.

    Input structure:
    - The background is black.
    - A corner row or column is a color-order strip of distinct object colors, read left-to-right or top-to-bottom.
    - Away from the strip, every colored 4-connected monochrome object has 2 to 6 pixels.
    - Each object shape, using bbox-relative coordinates with no rotation or reflection, appears exactly twice.

    Rule:
    - Remove the corner strip for object detection, then pair the two objects with each equal shape.
    - For each pair, look up both object colors in the strip and choose the strip color at floor((left_position + right_position) / 2).
    - Starting at the object whose color appears earlier in the strip, draw a one-pixel horizontal-then-vertical L path to the other object's bbox center.
    - Fill a 3x3 square of the same connection color centered on the L path corner.

    Key insight:
    Equal translated shapes define the pairs, the corner strip supplies an order over colors, and that order selects both the connection color and the fixed L-path direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    sequence, pairs = _parse_grid(g)
    order = {color: i for i, color in enumerate(sequence)}
    out = g.copy()

    for a, b in pairs:
        ai = order[int(a["color"])]
        bi = order[int(b["color"])]
        lo, hi = sorted((ai, bi))
        connect_color = int(sequence[(lo + hi) // 2])

        start, end = (a, b) if ai < bi else (b, a)
        sr, sc = _center(start)
        er, ec = _center(end)
        corner = (sr, ec)

        _paint_segment(out, (sr, sc), corner, connect_color)
        _paint_segment(out, corner, (er, ec), connect_color)
        _paint_marker(out, corner, connect_color)

    return out


def _mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    m = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        m[r, c] = True
    return m


SHAPES = (
    _mask(((0, 0), (0, 1), (0, 2))),
    _mask(((0, 0), (1, 0), (2, 0))),
    _mask(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
    _mask(((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))),
    _mask(((0, 1), (1, 1), (2, 0), (2, 1), (2, 2))),
    _mask(((0, 0), (1, 0), (1, 1), (1, 2), (2, 0))),
    _mask(((0, 2), (1, 0), (1, 1), (1, 2), (2, 2))),
    _mask(((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1))),
    _mask(((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2))),
)

PAIR_TEMPLATES = {
    2: (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ),
    3: (
        ((0, 1), (2, 4), (3, 5)),
        ((0, 2), (1, 4), (3, 5)),
        ((0, 3), (1, 2), (4, 5)),
    ),
    4: (
        ((0, 1), (2, 4), (3, 6), (5, 7)),
        ((0, 3), (1, 2), (4, 6), (5, 7)),
        ((0, 5), (1, 3), (2, 4), (6, 7)),
    ),
}


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _mask_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _center_for(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int]:
    return int(r0 + (mask.shape[0] - 1) // 2), int(c0 + (mask.shape[1] - 1) // 2)


def _valid_positions(mask: np.ndarray, forbidden: np.ndarray) -> list[tuple[int, int]]:
    h, w = forbidden.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            cr, cc = _center_for(mask, r0, c0)
            if cr < 2 or cc < 2 or cr > h - 3 or cc > w - 3:
                continue
            region = forbidden[r0 : r0 + mh, c0 : c0 + mw]
            if not bool(region[mask].any()):
                positions.append((r0, c0))
    return positions


def _to_bool(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r, c in cells:
        m[r, c] = True
    return m


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    sr, sc = start
    er, ec = end
    corner = (sr, ec)
    path = {(sr, c) for c in range(min(sc, ec), max(sc, ec) + 1)}
    path.update((r, ec) for r in range(min(sr, er), max(sr, er) + 1))
    marker = {(corner[0] + dr, corner[1] + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    return path, marker, corner


def _paint_strip(grid: np.ndarray, sequence: tuple[int, ...], spec: tuple[str, str]) -> set[tuple[int, int]]:
    h, w = grid.shape
    corner, orient = spec
    top = "top" in corner
    left = "left" in corner
    cells: set[tuple[int, int]] = set()

    if orient == "h":
        r = 0 if top else h - 1
        c0 = 0 if left else w - len(sequence)
        for i, color in enumerate(sequence):
            grid[r, c0 + i] = int(color)
            cells.add((r, c0 + i))
    else:
        c = 0 if left else w - 1
        r0 = 0 if top else h - len(sequence)
        for i, color in enumerate(sequence):
            grid[r0 + i, c] = int(color)
            cells.add((r0 + i, c))
    return cells


def _choose_pairs(n_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs = [tuple(pair) for pair in rng.choice(PAIR_TEMPLATES[n_pairs])]
    rng.shuffle(pairs)
    return [(min(a, b), max(a, b)) for a, b in pairs]


def _place_pair(
    grid: np.ndarray,
    mask: np.ndarray,
    idx_pair: tuple[int, int],
    sequence: tuple[int, ...],
    object_cells: np.ndarray,
    bar_cells: np.ndarray,
    path_reserved: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    forbidden = _dilate(object_cells | bar_cells, 1) | path_reserved
    positions = _valid_positions(mask, forbidden)
    if len(positions) < 2:
        return False

    for _ in range(500):
        p1, p2 = rng.sample(positions, 2)
        cells1 = _mask_cells(mask, p1[0], p1[1])
        cells2 = _mask_cells(mask, p2[0], p2[1])
        if cells1 & cells2:
            continue

        temp = _to_bool(grid.shape, cells1)
        if bool((_dilate(temp, 1) & _to_bool(grid.shape, cells2)).any()):
            continue

        i, j = idx_pair
        color_i, color_j = int(sequence[i]), int(sequence[j])
        if rng.random() < 0.5:
            loc_i, loc_j = p1, p2
            cells_i, cells_j = cells1, cells2
        else:
            loc_i, loc_j = p2, p1
            cells_i, cells_j = cells2, cells1

        center_i = _center_for(mask, loc_i[0], loc_i[1])
        center_j = _center_for(mask, loc_j[0], loc_j[1])
        if center_i[0] == center_j[0] or center_i[1] == center_j[1]:
            continue

        start, end = (center_i, center_j) if i < j else (center_j, center_i)
        path, marker, _corner = _path_cells(start, end)
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in marker):
            continue

        path_mask = _to_bool(grid.shape, path)
        marker_mask = _to_bool(grid.shape, marker)
        own_mask = _to_bool(grid.shape, cells_i | cells_j)
        if bool(marker_mask.any() and (marker_mask & (bar_cells | object_cells | own_mask | path_reserved)).any()):
            continue
        if bool((path_mask & (bar_cells | object_cells | path_reserved)).any()):
            continue

        for r, c in cells_i:
            grid[r, c] = color_i
            object_cells[r, c] = True
        for r, c in cells_j:
            grid[r, c] = color_j
            object_cells[r, c] = True
        path_reserved[path_mask | marker_mask] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(2, 4)
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)

    sequence = tuple(rng.sample(COLORS, 2 * n_pairs))
    strip_specs = [
        ("top_left", "h"),
        ("top_right", "h"),
        ("bottom_left", "h"),
        ("bottom_right", "h"),
        ("top_left", "v"),
        ("top_right", "v"),
        ("bottom_left", "v"),
        ("bottom_right", "v"),
    ]
    bar_set = _paint_strip(grid, sequence, rng.choice(strip_specs))
    bar_cells = _to_bool(grid.shape, bar_set)

    object_cells = np.zeros(grid.shape, dtype=bool)
    path_reserved = np.zeros(grid.shape, dtype=bool)
    shape_indexes = rng.sample(range(len(SHAPES)), n_pairs)
    pair_indexes = _choose_pairs(n_pairs, rng)

    for shape_idx, idx_pair in zip(shape_indexes, pair_indexes):
        if not _place_pair(
            grid,
            SHAPES[shape_idx],
            idx_pair,
            sequence,
            object_cells,
            bar_cells,
            path_reserved,
            rng,
        ):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30 or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-shape color-strip input")
