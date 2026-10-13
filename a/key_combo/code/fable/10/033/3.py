from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _angle_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coords = np.asarray(points, dtype=float)
    cr, cc = coords.mean(axis=0)
    enriched = []
    for r, c in points:
        angle = math.atan2(r - cr, c - cc)
        dist = (r - cr) ** 2 + (c - cc) ** 2
        enriched.append((angle, dist, int(r), int(c)))
    enriched.sort()
    return [(r, c) for _ang, _dist, r, c in enriched]


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0:
        step_r, step_c = 0, 1 if dc > 0 else -1
        length = abs(dc)
    elif dc == 0:
        step_r, step_c = 1 if dr > 0 else -1, 0
        length = abs(dr)
    elif abs(dr) == abs(dc):
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1
        length = abs(dr)
    else:
        raise ValueError("segment must be horizontal, vertical, or 45-degree diagonal")
    return [(r0 + step_r * k, c0 + step_c * k) for k in range(length + 1)]


def _point_on_segment(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    pr, pc = p
    ar, ac = a
    br, bc = b
    if (br - ar) * (pc - ac) != (bc - ac) * (pr - ar):
        return False
    return min(ar, br) <= pr <= max(ar, br) and min(ac, bc) <= pc <= max(ac, bc)


def _point_in_polygon(p: tuple[int, int], cycle: list[tuple[int, int]]) -> bool:
    if any(_point_on_segment(p, cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))):
        return False
    y = float(p[0])
    x = float(p[1])
    inside = False
    for i in range(len(cycle)):
        y1, x1 = cycle[i]
        y2, x2 = cycle[(i + 1) % len(cycle)]
        if (y1 > y) == (y2 > y):
            continue
        cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if x < cross_x:
            inside = not inside
    return inside


def _polygon_masks(
    shape: tuple[int, int], cycle: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    boundary = np.zeros((h, w), dtype=bool)
    for i in range(len(cycle)):
        for r, c in _line_points(cycle[i], cycle[(i + 1) % len(cycle)]):
            boundary[r, c] = True
    interior = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not boundary[r, c] and _point_in_polygon((r, c), cycle):
                interior[r, c] = True
    return boundary, interior


def _cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def _segments_intersect(
    a1: tuple[int, int],
    a2: tuple[int, int],
    b1: tuple[int, int],
    b2: tuple[int, int],
) -> bool:
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if v == 0 else (1 if v > 0 else -1)

    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_seg(a1, b1, a2):
        return True
    if o2 == 0 and on_seg(a1, b2, a2):
        return True
    if o3 == 0 and on_seg(b1, a1, b2):
        return True
    if o4 == 0 and on_seg(b1, a2, b2):
        return True
    return False


def _is_simple_polygon(cycle: list[tuple[int, int]]) -> bool:
    n = len(cycle)
    for i in range(n):
        a1 = cycle[i]
        a2 = cycle[(i + 1) % n]
        for j in range(i + 1, n):
            if j in {i, (i + 1) % n}:
                continue
            if (j + 1) % n == i:
                continue
            b1 = cycle[j]
            b2 = cycle[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def _is_concave(cycle: list[tuple[int, int]]) -> bool:
    signs = []
    n = len(cycle)
    for i in range(n):
        val = _cross(cycle[i - 1], cycle[i], cycle[(i + 1) % n])
        if val != 0:
            signs.append(1 if val > 0 else -1)
    return len(set(signs)) > 1


def _normalize(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in points)
    min_c = min(c for _r, c in points)
    return [(r - min_r, c - min_c) for r, c in points]


def _transform_bundle(
    points: list[tuple[int, int]], shear: int, rot: int, flip: bool
) -> list[tuple[int, int]]:
    pts = points[:]
    if shear:
        pts = [(r, c + shear * r) for r, c in pts]
        pts = _normalize(pts)
    else:
        pts = _normalize(pts)
    h = max(r for r, _c in pts) + 1
    w = max(c for _r, c in pts) + 1
    rotated = []
    for r, c in pts:
        if rot == 0:
            rr, cc = r, c
            h2, w2 = h, w
        elif rot == 1:
            rr, cc = c, h - 1 - r
            h2, w2 = w, h
        elif rot == 2:
            rr, cc = h - 1 - r, w - 1 - c
            h2, w2 = h, w
        else:
            rr, cc = w - 1 - c, r
            h2, w2 = w, h
        if flip:
            cc = w2 - 1 - cc
        rotated.append((rr, cc))
    return _normalize(rotated)


def _translate_mask(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct an invisible concave polygon from its vertices, then fill it with the color of the unique strict interior signal pixel.

    Input structure:
    - The background contains one repeated outline color whose pixels are exactly the vertices of a single closed concave polygon.
    - Sorting those vertices by polar angle around their centroid makes every consecutive pair horizontal, vertical, or 45-degree diagonal.
    - Additional non-outline colors appear only as singleton candidate pixels placed on the boundary, strictly inside, or outside the rebuilt polygon.

    Rule:
    - Order the outline-color vertices by polar angle and connect consecutive vertices, including the last back to the first, using the outline color.
    - Classify every singleton candidate against that rebuilt polygon: boundary candidates are overwritten by the outline, outside candidates are deleted, and exactly one candidate lies strictly inside.
    - Fill every strict interior cell of the polygon with the color of that unique interior candidate while keeping the outline color on the boundary.

    Key insight:
    - The decisive structure is the polygon that is absent from the input image: only after rebuilding its concave outline can you distinguish the true interior signal from the misleading pixel trapped inside the notch.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    counts = Counter(map(int, g.ravel()))
    outline_color = max(colors, key=lambda c: (counts[c], c))

    vertices = [tuple(map(int, p)) for p in np.argwhere(g == outline_color)]
    cycle = _angle_cycle(vertices)
    boundary, interior = _polygon_masks(g.shape, cycle)

    winners = []
    for r, c in np.argwhere((g != bg) & (g != outline_color)):
        rr = int(r)
        cc = int(c)
        if boundary[rr, cc]:
            continue
        if interior[rr, cc]:
            winners.append((rr, cc, int(g[rr, cc])))
    if len(winners) != 1:
        raise ValueError("expected exactly one strict interior candidate")

    fill_color = winners[0][2]
    out = np.full_like(g, bg)
    out[interior] = fill_color
    out[boundary] = outline_color
    return out


def _make_base_shape() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    bottom = random.randint(7, 12)
    right = random.randint(8, 14)
    notch_row = random.randint(2, bottom - 3)
    notch_col = random.randint(2, right - 3)
    vertices = [
        (0, 0),
        (0, right),
        (notch_row, right),
        (notch_row, notch_col),
        (bottom, notch_col),
        (bottom, 0),
    ]
    pocket_r = random.randint(notch_row + 1, bottom - 1)
    pocket_c = random.randint(notch_col + 1, right - 1)
    return vertices, (pocket_r, pocket_c)


def _pick_inside_point(interior: np.ndarray) -> tuple[int, int]:
    h, w = interior.shape
    best = []
    fallback = []
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        fallback.append((rr, cc))
        safe8 = True
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if not (0 <= nr < h and 0 <= nc < w and interior[nr, nc]):
                    safe8 = False
                    break
            if not safe8:
                break
        if safe8:
            best.append((rr, cc))
    pool = best if best else fallback
    return random.choice(pool)


def _pick_line_point(
    cycle: list[tuple[int, int]], boundary: np.ndarray
) -> tuple[int, int] | None:
    vertices = set(cycle)
    candidates = []
    for r, c in np.argwhere(boundary):
        cell = (int(r), int(c))
        if cell not in vertices:
            candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _choose_canvas(local_h: int, local_w: int) -> tuple[int, int, int, int]:
    h_min = max(local_h + 2, 14)
    w_min = max(local_w + 2, 14)
    h_max = min(30, local_h + 10)
    w_max = min(30, local_w + 10)
    h = random.randint(h_min, h_max)
    w = random.randint(w_min, w_max)
    top = random.randint(1, h - local_h - 1)
    left = random.randint(1, w - local_w - 1)
    return h, w, top, left


def _pick_far_outside(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], used: set[tuple[int, int]]
) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in used:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            candidates.append((r, c))
    if not candidates:
        raise ValueError("no far outside cell available")
    return random.choice(candidates)


def _expected_output(
    shape: tuple[int, int],
    boundary: np.ndarray,
    interior: np.ndarray,
    outline_color: int,
    fill_color: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[interior] = fill_color
    out[boundary] = outline_color
    return out


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    outline_color: int,
    candidates: list[tuple[int, int, int]],
    boundary: np.ndarray,
    interior: np.ndarray,
    bbox: tuple[int, int, int, int],
    inside_pos: tuple[int, int],
    pocket_pos: tuple[int, int],
    line_pos: tuple[int, int] | None,
) -> bool:
    if np.array_equal(inp, expected):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if expected.shape != inp.shape or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if int(np.count_nonzero(inp == outline_color)) < 6:
        return False

    strict_inside = []
    strict_outside = []
    on_boundary = []
    for r, c, color in candidates:
        if color == outline_color:
            return False
        if boundary[r, c]:
            on_boundary.append((r, c))
        elif interior[r, c]:
            strict_inside.append((r, c))
        else:
            strict_outside.append((r, c))
    if strict_inside != [inside_pos]:
        return False
    if pocket_pos not in strict_outside:
        return False
    if line_pos is not None and line_pos not in on_boundary:
        return False
    if expected[pocket_pos] != BG:
        return False
    if line_pos is not None and expected[line_pos] != outline_color:
        return False
    if np.count_nonzero(expected == expected[inside_pos]) <= 1:
        return False
    r0, c0, r1, c1 = bbox
    notch_gap = (~boundary) & (~interior)
    notch_gap[:r0, :] = False
    notch_gap[r1 + 1 :, :] = False
    notch_gap[:, :c0] = False
    notch_gap[:, c1 + 1 :] = False
    if not np.any(notch_gap):
        return False
    fill_color = int(expected[inside_pos])
    if np.any(expected[notch_gap] == fill_color):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base_vertices, base_pocket = _make_base_shape()
    shear = random.choice([0, 1, -1])
    rot = random.randrange(4)
    flip = random.random() < 0.5

    bundle = _transform_bundle(base_vertices + [base_pocket], shear=shear, rot=rot, flip=flip)
    vertices = bundle[:-1]
    pocket = bundle[-1]

    cycle = _angle_cycle(vertices)
    if len({math.atan2(r - np.mean([p[0] for p in vertices]), c - np.mean([p[1] for p in vertices])) for r, c in vertices}) != len(vertices):
        return None
    if not _is_simple_polygon(cycle) or not _is_concave(cycle):
        return None
    try:
        for i in range(len(cycle)):
            _line_points(cycle[i], cycle[(i + 1) % len(cycle)])
    except ValueError:
        return None

    local_h = max(r for r, _c in cycle) + 1
    local_w = max(c for _r, c in cycle) + 1
    local_boundary, local_interior = _polygon_masks((local_h, local_w), cycle)
    if not local_interior.any():
        return None
    if pocket[0] >= local_h or pocket[1] >= local_w:
        return None
    if local_boundary[pocket] or local_interior[pocket]:
        return None

    inside_local = _pick_inside_point(local_interior)
    line_local = _pick_line_point(cycle, local_boundary) if random.random() < 0.7 else None
    h, w, top, left = _choose_canvas(local_h, local_w)
    boundary = _translate_mask(local_boundary, top, left, (h, w))
    interior = _translate_mask(local_interior, top, left, (h, w))

    vertices_global = [(r + top, c + left) for r, c in vertices]
    inside_pos = (inside_local[0] + top, inside_local[1] + left)
    pocket_pos = (pocket[0] + top, pocket[1] + left)
    line_pos = None if line_local is None else (line_local[0] + top, line_local[1] + left)

    bbox = (top, left, top + local_h - 1, left + local_w - 1)
    used = set(vertices_global) | {inside_pos, pocket_pos}
    if line_pos is not None:
        used.add(line_pos)
    far_outside = _pick_far_outside((h, w), bbox, used)
    used.add(far_outside)
    extra_outside = None
    if random.random() < 0.45:
        extra_outside = _pick_far_outside((h, w), bbox, used)
        used.add(extra_outside)

    candidate_positions = [inside_pos, pocket_pos, far_outside]
    if line_pos is not None:
        candidate_positions.append(line_pos)
    if extra_outside is not None:
        candidate_positions.append(extra_outside)

    colors = random.sample(PALETTE, 1 + len(candidate_positions))
    outline_color = colors[0]
    candidate_colors = colors[1:]

    inp = np.full((h, w), BG, dtype=int)
    for r, c in vertices_global:
        inp[r, c] = outline_color
    candidates = []
    for (r, c), color in zip(candidate_positions, candidate_colors):
        inp[r, c] = color
        candidates.append((r, c, color))

    fill_color = int(inp[inside_pos])
    expected = _expected_output((h, w), boundary, interior, outline_color, fill_color)

    if not _validate_example(
        inp=inp,
        expected=expected,
        outline_color=outline_color,
        candidates=candidates,
        boundary=boundary,
        interior=interior,
        bbox=bbox,
        inside_pos=inside_pos,
        pocket_pos=pocket_pos,
        line_pos=line_pos,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
