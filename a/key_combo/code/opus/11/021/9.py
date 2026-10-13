from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
FRAME = 1
MARKER = 2
RAY = 3

_FAMILIES = ("top", "split", "bottom")
_SIDES = ("top", "right", "bottom", "left")
_CORNERS = ("tl", "tr", "bl", "br")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_of_points(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_hollow_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _nearest_corner_direction(r: int, c: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    top = r
    bottom = h - 1 - r
    left = c
    right = w - 1 - c
    assert top != bottom and left != right, "nearest grid corner must be unique"
    dr = -1 if top < bottom else 1
    dc = -1 if left < right else 1
    return dr, dc


def _ray_endpoint_kind(r: int, c: int, dr: int, dc: int, shape: tuple[int, int]) -> str:
    h, w = shape
    row_dist = r if dr < 0 else h - 1 - r
    col_dist = c if dc < 0 else w - 1 - c
    if row_dist == col_dist:
        return "corner"
    return "row" if row_dist < col_dist else "col"


def _cast_ray_with_cap(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> None:
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    while True:
        grid[r, c] = RAY
        if r in (0, h - 1) or c in (0, w - 1):
            break
        r += dr
        c += dc

    for rr, cc in ((r, c - dc), (r - dr, c)):
        if _in_bounds(grid.shape, rr, cc):
            grid[rr, cc] = RAY


def _analyze_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> dict[str, object]:
    r0, c0, r1, c1 = box
    corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }
    dirs = {name: _nearest_corner_direction(r, c, shape) for name, (r, c) in corners.items()}
    kinds = {
        name: _ray_endpoint_kind(r, c, dirs[name][0], dirs[name][1], shape)
        for name, (r, c) in corners.items()
    }
    return {"corners": corners, "dirs": dirs, "kinds": kinds}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a rectangle from three marker points, then emit capped diagonal rays from its corners.

    Input structure:
    - The grid is a black background with exactly three isolated red marker pixels.
    - The three markers lie on the boundary of one implied axis-aligned rectangle, and their bbox is that rectangle.
    - Blue and green do not appear in the input.

    Rule:
    - Compute the bbox of the three red markers and draw its 1-pixel hollow rectangle in blue.
    - For each rectangle corner, choose the unique nearest grid corner; this fixes one diagonal step direction with row and column both changing by 1.
    - Starting at the rectangle corner, draw a green diagonal ray until it first reaches any grid boundary cell, then add a 3-pixel L-cap by extending one pixel inward along the horizontal and vertical bounce directions.
    - Finally erase the three original marker coordinates back to black unless a green corner ray occupies that cell.

    Key insight:
    - The markers matter only to define the rectangle; the later ray directions are determined by each rectangle corner's position relative to the whole grid, not by a fixed corner-to-corner rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    markers = np.argwhere(g == MARKER)
    assert markers.shape[0] == 3, "expected exactly three red markers"
    box = _bbox_of_points(markers)
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "rectangle must be at least 4x4"

    out = np.full_like(g, BG)
    _draw_hollow_rect(out, box, FRAME)

    info = _analyze_box(box, g.shape)
    for name in ("tl", "tr", "bl", "br"):
        _cast_ray_with_cap(out, info["corners"][name], info["dirs"][name])

    for r, c in markers:
        out[int(r), int(c)] = BG
    for corner in info["corners"].values():
        out[corner] = RAY
    return out


def _choose_axis_pair(length: int, family: str) -> tuple[int, int] | None:
    mid = (length - 1) / 2.0
    top_vals = [v for v in range(2, length - 2) if v < mid]
    bot_vals = [v for v in range(2, length - 2) if v > mid]
    if family == "split":
        pairs = [(a, b) for a in top_vals for b in bot_vals if b - a >= 3]
    else:
        pool = top_vals if family == "top" else bot_vals
        pairs = [(a, b) for a, b in combinations(pool, 2) if b - a >= 3]
    return random.choice(pairs) if pairs else None


def _marker_candidates(box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = box
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
        "tl": [(r0, c0)],
        "tr": [(r0, c1)],
        "bl": [(r1, c0)],
        "br": [(r1, c1)],
    }


def _touch_sides(point: tuple[int, int], box: tuple[int, int, int, int]) -> set[str]:
    r, c = point
    r0, c0, r1, c1 = box
    touched = set()
    if r == r0:
        touched.add("top")
    if r == r1:
        touched.add("bottom")
    if c == c0:
        touched.add("left")
    if c == c1:
        touched.add("right")
    return touched


def _well_separated(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _choose_markers(box: tuple[int, int, int, int], mode: str) -> list[tuple[int, int]] | None:
    cand = _marker_candidates(box)
    corner_points = {cand[name][0] for name in _CORNERS}
    for _ in range(200):
        if mode == "one_corner":
            corner = random.choice(list(_CORNERS))
            if corner == "tl":
                sides = ("bottom", "right")
            elif corner == "tr":
                sides = ("bottom", "left")
            elif corner == "bl":
                sides = ("top", "right")
            else:
                sides = ("top", "left")
            pts = [cand[corner][0], random.choice(cand[sides[0]]), random.choice(cand[sides[1]])]
        else:
            diagonal = random.choice((("tl", "br"), ("tr", "bl")))
            third_side = random.choice(list(_SIDES))
            pts = [cand[diagonal[0]][0], cand[diagonal[1]][0], random.choice(cand[third_side])]

        if len(set(pts)) != 3 or not _well_separated(pts):
            continue
        if len(corner_points & set(pts)) < 1:
            continue
        touched = set()
        for pt in pts:
            touched |= _touch_sides(pt, box)
        if len(touched) < 3:
            continue
        if not any(pt not in corner_points for pt in pts):
            continue
        return pts
    return None


def _sample_box(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], dict[str, object]] | None:
    h, w = shape
    for _ in range(300):
        row_family = random.choice(_FAMILIES)
        col_family = random.choice(_FAMILIES)
        pair_r = _choose_axis_pair(h, row_family)
        pair_c = _choose_axis_pair(w, col_family)
        if pair_r is None or pair_c is None:
            continue
        box = (pair_r[0], pair_c[0], pair_r[1], pair_c[1])
        info = _analyze_box(box, shape)
        kinds = set(info["kinds"].values())
        if len(kinds) < 2:
            continue
        return box, {"row_family": row_family, "col_family": col_family, **info}
    return None


def _validate_input(grid: np.ndarray) -> None:
    pts = np.argwhere(grid == MARKER)
    assert pts.shape[0] == 3
    assert not np.any(grid == FRAME)
    assert not np.any(grid == RAY)
    box = _bbox_of_points(pts)
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4
    assert r0 >= 2 and c0 >= 2
    assert grid.shape[0] - 1 - r1 >= 2 and grid.shape[1] - 1 - c1 >= 2
    points = [tuple(map(int, p)) for p in pts]
    assert _well_separated(points)
    touched = set()
    for pt in points:
        touched |= _touch_sides(pt, box)
    assert len(touched) >= 3
    info = _analyze_box(box, grid.shape)
    assert len(set(info["kinds"].values())) >= 2
    corners = set(info["corners"].values())
    assert any(pt in corners for pt in points)
    assert any(pt not in corners for pt in points)


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    assert out.shape == inp.shape
    assert not np.any(out == MARKER)
    assert np.any(out == FRAME)
    assert np.any(out == RAY)
    pts = np.argwhere(inp == MARKER)
    box = _bbox_of_points(pts)
    info = _analyze_box(box, inp.shape)
    for name, corner in info["corners"].items():
        assert out[corner] == RAY
        kind = info["kinds"][name]
        r, c = corner
        dr, dc = info["dirs"][name]
        row_dist = r if dr < 0 else inp.shape[0] - 1 - r
        col_dist = c if dc < 0 else inp.shape[1] - 1 - c
        assert min(row_dist, col_dist) >= 2
        assert kind in {"row", "col", "corner"}


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = random.randint(12, 19)
        w = random.randint(12, 19)
        sampled = _sample_box((h, w))
        if sampled is None:
            continue
        box, info = sampled
        mode = random.choices(
            ["one_corner", "two_corners"],
            weights=[4, 1],
            k=1,
        )[0]
        markers = _choose_markers(box, mode)
        if markers is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r, c in markers:
            grid[r, c] = MARKER

        try:
            _validate_input(grid)
            out = main(grid)
            _validate_output(grid, out)
        except AssertionError:
            continue

        nonlocal_dirs = sum(
            1
            for name, (_dr, dc) in info["dirs"].items()
            if (name.endswith("l") and dc > 0) or (name.endswith("r") and dc < 0)
        )
        if info["row_family"] == "split" and info["col_family"] == "split" and nonlocal_dirs == 0:
            if mode != "one_corner":
                continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
