from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), max(rows), min(cols), max(cols)


def _reflect(points: Iterable[tuple[int, int]], axis: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    mirrored: list[tuple[int, int]] = []
    for r, c in points:
        if axis == "top":
            mirrored.append((2 * r0 - 1 - r, c))
        elif axis == "bottom":
            mirrored.append((2 * r1 + 1 - r, c))
        elif axis == "left":
            mirrored.append((r, 2 * c0 - 1 - c))
        elif axis == "right":
            mirrored.append((r, 2 * c1 + 1 - c))
        else:
            raise ValueError(axis)
    return mirrored


def _corner_axes(corner: str) -> tuple[str, str]:
    return {
        "tl": ("top", "left"),
        "tr": ("top", "right"),
        "bl": ("bottom", "left"),
        "br": ("bottom", "right"),
    }[corner]


def _identify_colors(sub: np.ndarray) -> tuple[int, int]:
    colors = [int(v) for v in np.unique(sub)]
    assert len(colors) == 2, "each envelope must contain exactly two colors"

    corners = [int(sub[0, 0]), int(sub[0, -1]), int(sub[-1, 0]), int(sub[-1, -1])]
    corner_hits = {color: sum(v == color for v in corners) for color in colors}
    wing = [color for color in colors if corner_hits[color] == 1]
    base = [color for color in colors if corner_hits[color] == 3]
    if len(base) == 1 and len(wing) == 1:
        return base[0], wing[0]

    counts = {color: int(np.sum(sub == color)) for color in colors}
    base_color = max(colors, key=lambda color: counts[color])
    wing_color = min(colors, key=lambda color: counts[color])
    assert base_color != wing_color
    return base_color, wing_color


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _components(g != BG):
        r0, r1, c0, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub != BG), "each object must occupy a solid rectangle"

        base_color, wing_color = _identify_colors(sub)
        wing_points = [(r, c) for r, c in comp if int(g[r, c]) == wing_color]
        assert wing_points, "missing wing"

        if int(sub[0, 0]) == wing_color:
            corner = "tl"
        elif int(sub[0, -1]) == wing_color:
            corner = "tr"
        elif int(sub[-1, 0]) == wing_color:
            corner = "bl"
        elif int(sub[-1, -1]) == wing_color:
            corner = "br"
        else:
            raise AssertionError("wing must occupy exactly one rectangle corner")

        valid: list[list[tuple[int, int]]] = []
        for axis in _corner_axes(corner):
            mirrored = _reflect(wing_points, axis, (r0, r1, c0, c1))
            if all(0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BG for rr, cc in mirrored):
                valid.append(mirrored)
        assert len(valid) == 1, "fold axis must be uniquely determined by the surrounding black space"

        for r, c in wing_points:
            out[r, c] = base_color
        for r, c in valid[0]:
            out[r, c] = wing_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each folded triangular envelope flap from a colored rectangle back to the outside.

    Input structure:
    - The background is black.
    - Each non-black object is a solid axis-aligned rectangle made of one base color.
    - One corner of each rectangle contains a differently colored right-triangle wing folded into the rectangle interior.
    - The wing touches exactly one rectangle corner and lies along the two rectangle edges meeting at that corner.

    Rule:
    - Treat each connected non-black object as one rectangle with one folded wing.
    - Identify the wing color and the corner it occupies, then test the two boundary edges adjacent to that corner.
    - The true fold axis is the unique adjacent edge whose mirror image places the wing entirely onto black cells outside the rectangle.
    - Restore the interior wing cells to the rectangle's base color and draw the mirrored wing outside the rectangle across that edge.

    Key insight:
    - The corner triangle is not a decoration inside the rectangle; it is an exterior flap shown in its folded-in state, so the transformation reverses that fold by mirroring the triangle out across the correct edge.
    """
    return _solve(np.asarray(grid, dtype=int))


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 26
RECT_SIDE_MIN = 4
RECT_SIDE_MAX = 8
TRI_SIDE_MIN = 2
TRI_SIDE_MAX = 5
OBJECT_COUNT_CHOICES = (1, 2, 3)
BRANCHES = (
    ("tl", "top"),
    ("tl", "left"),
    ("tr", "top"),
    ("tr", "right"),
    ("bl", "bottom"),
    ("bl", "left"),
    ("br", "bottom"),
    ("br", "right"),
)
GEN_RETRIES = 400
PLACE_RETRIES = 120


def _triangle_mask(height: int, width: int, mode: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if mode == "rows":
        for r in range(height):
            span = max(1, ((height - r) * width + height - 1) // height)
            mask[r, :span] = True
    else:
        for c in range(width):
            span = max(1, ((width - c) * height + width - 1) // width)
            mask[:span, c] = True
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _orient_mask(mask: np.ndarray, corner: str) -> np.ndarray:
    out = mask
    if "b" in corner:
        out = np.flipud(out)
    if "r" in corner:
        out = np.fliplr(out)
    return out


def _wing_cells(
    r0: int,
    c0: int,
    rh: int,
    rw: int,
    corner: str,
    tri_mask: np.ndarray,
) -> list[tuple[int, int]]:
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    oriented = _orient_mask(tri_mask, corner)
    th, tw = oriented.shape
    sr0 = r0 if corner in ("tl", "tr") else r1 - th + 1
    sc0 = c0 if corner in ("tl", "bl") else c1 - tw + 1
    pts = np.argwhere(oriented)
    return [(sr0 + int(rr), sc0 + int(cc)) for rr, cc in pts]


def _position_bounds(
    h: int,
    w: int,
    rh: int,
    rw: int,
    th: int,
    tw: int,
    corner: str,
    axis: str,
) -> tuple[int, int, int, int] | None:
    r_lo, r_hi = 0, h - rh
    c_lo, c_hi = 0, w - rw

    if axis == "top":
        r_lo = max(r_lo, th)
    elif axis == "bottom":
        r_hi = min(r_hi, h - rh - th)
    elif axis == "left":
        c_lo = max(c_lo, tw)
    elif axis == "right":
        c_hi = min(c_hi, w - rw - tw)

    if corner == "tl":
        if axis == "top":
            c_hi = min(c_hi, tw - 1)
        else:
            r_hi = min(r_hi, th - 1)
    elif corner == "tr":
        if axis == "top":
            c_lo = max(c_lo, w - rw - (tw - 1))
        else:
            r_hi = min(r_hi, th - 1)
    elif corner == "bl":
        if axis == "bottom":
            c_hi = min(c_hi, tw - 1)
        else:
            r_lo = max(r_lo, h - rh - (th - 1))
    else:
        if axis == "bottom":
            c_lo = max(c_lo, w - rw - (tw - 1))
        else:
            r_lo = max(r_lo, h - rh - (th - 1))

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return r_lo, r_hi, c_lo, c_hi


def _mark_blocked(blocked: np.ndarray, cells: Iterable[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_object(
    inp: np.ndarray,
    out: np.ndarray,
    blocked: np.ndarray,
    base_color: int,
    wing_color: int,
    corner: str,
    axis: str,
) -> bool:
    h, w = inp.shape
    for _ in range(PLACE_RETRIES):
        rh = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, h - 2))
        rw = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, w - 2))
        th = random.randint(TRI_SIDE_MIN, min(TRI_SIDE_MAX, rh - 1))
        tw = random.randint(TRI_SIDE_MIN, min(TRI_SIDE_MAX, rw - 1))
        tri_mask = _triangle_mask(th, tw, random.choice(("rows", "cols")))
        wing_area = int(tri_mask.sum())
        if wing_area < 3 or wing_area * 2 >= rh * rw:
            continue

        bounds = _position_bounds(h, w, rh, rw, th, tw, corner, axis)
        if bounds is None:
            continue
        r_lo, r_hi, c_lo, c_hi = bounds
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1

        rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        wing_cells = _wing_cells(r0, c0, rh, rw, corner, tri_mask)
        mirrored = _reflect(wing_cells, axis, (r0, r1, c0, c1))
        footprint = rect_cells | set(mirrored)

        if not all(0 <= r < h and 0 <= c < w for r, c in mirrored):
            continue
        if any(blocked[r, c] for r, c in footprint):
            continue

        inp[r0 : r1 + 1, c0 : c1 + 1] = base_color
        out[r0 : r1 + 1, c0 : c1 + 1] = base_color
        for r, c in wing_cells:
            inp[r, c] = wing_color
        for r, c in mirrored:
            out[r, c] = wing_color

        _mark_blocked(blocked, footprint)
        return True
    return False


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    obj_count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(range(1, 10), 2 * obj_count)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for i in range(obj_count):
        base_color = colors[2 * i]
        wing_color = colors[2 * i + 1]
        corner, axis = random.choice(BRANCHES)
        ok = _place_object(inp, out, blocked, base_color, wing_color, corner, axis)
        if not ok:
            return None

    if not np.array_equal(_solve(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid folded-envelope input")
