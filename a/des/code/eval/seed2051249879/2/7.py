from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MAGENTA = 6
MAROON = 9
RECT_PALETTE = (1, 2, 3, 4, 7, 8)
ORIENTATIONS = ("TL", "TR", "BL", "BR")

GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
ZONE_SIZE_RANGE = (2, 3)
GEN_RETRIES = 400
BOX_SAMPLE_RETRIES = 120


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _run_length(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = mask.shape
    length = 0
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
        length += 1
        rr += dr
        cc += dc
    return length


def _bbox_from_corner(r: int, c: int, width: int, height: int, orient: str) -> tuple[int, int, int, int]:
    if orient == "TL":
        return (r, c, r + height - 1, c + width - 1)
    if orient == "TR":
        return (r, c - width + 1, r + height - 1, c)
    if orient == "BL":
        return (r - height + 1, c, r, c + width - 1)
    if orient == "BR":
        return (r - height + 1, c - width + 1, r, c)
    raise ValueError(orient)


def _protected_cells(bbox: tuple[int, int, int, int], orient: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if orient in ("TL", "TR"):
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
    else:
        for c in range(c0, c1 + 1):
            cells.add((r1, c))
    if orient in ("TL", "BL"):
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
    else:
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    return cells


def _bbox_hits_cells(bbox: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _parse_layer_order(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    strip = g[-1]
    values, counts = np.unique(strip, return_counts=True)
    legend_fill = int(values[np.argmax(counts)])
    order = [int(v) for v in strip if int(v) != legend_fill]
    assert order and len(order) == len(set(order)), "legend colors must be unique"
    canvas = g[:-1].copy()
    return canvas, order


def _infer_rectangle(canvas: np.ndarray, color: int, higher_colors: set[int]) -> tuple[int, int, int, int]:
    mask = canvas == color
    pts = np.argwhere(mask)
    assert pts.size, f"color {color} must be visible"
    h, w = mask.shape
    allowed = [color, *sorted(higher_colors)]
    candidates: set[tuple[int, int, int, int]] = set()

    corner_specs = {
        "TL": {"blocked": [(-1, 0), (0, -1)], "dirs": ((0, 1), (1, 0))},
        "TR": {"blocked": [(-1, 0), (0, 1)], "dirs": ((0, -1), (1, 0))},
        "BL": {"blocked": [(1, 0), (0, -1)], "dirs": ((0, 1), (-1, 0))},
        "BR": {"blocked": [(1, 0), (0, 1)], "dirs": ((0, -1), (-1, 0))},
    }

    for rr, cc in pts:
        r = int(rr)
        c = int(cc)
        for orient, spec in corner_specs.items():
            blocked_ok = True
            for dr, dc in spec["blocked"]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                    blocked_ok = False
                    break
            if not blocked_ok:
                continue

            (dr1, dc1), (dr2, dc2) = spec["dirs"]
            width = _run_length(mask, r, c, dr1, dc1)
            height = _run_length(mask, r, c, dr2, dc2)
            if width < 2 or height < 2:
                continue

            bbox = _bbox_from_corner(r, c, width, height, orient)
            r0, c0, r1, c1 = bbox
            if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
                continue

            if np.any((pts[:, 0] < r0) | (pts[:, 0] > r1) | (pts[:, 1] < c0) | (pts[:, 1] > c1)):
                continue

            protected = _protected_cells(bbox, orient)
            if any(not bool(mask[pr, pc]) for pr, pc in protected):
                continue

            window = canvas[r0 : r1 + 1, c0 : c1 + 1]
            if not np.isin(window, allowed).all():
                continue

            candidates.add(bbox)

    assert candidates, f"failed to reconstruct rectangle for color {color}"
    ranked = sorted(candidates, key=lambda b: (_bbox_area(b), b))
    best_area = _bbox_area(ranked[0])
    best = [bbox for bbox in ranked if _bbox_area(bbox) == best_area]
    assert len(best) == 1, f"ambiguous reconstruction for color {color}"
    return best[0]


def _reconstruct_rectangles(canvas: np.ndarray, order: list[int]) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for i, color in enumerate(order):
        higher_colors = set(order[i + 1 :])
        rects.append(_infer_rectangle(canvas, color, higher_colors))
    return rects


def _paint_overlap(rects: list[tuple[int, int, int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    counts = np.zeros(shape, dtype=int)
    for bbox, color in zip(rects, colors):
        r0, c0, r1, c1 = bbox
        region = np.s_[r0 : r1 + 1, c0 : c1 + 1]
        prev = counts[region]
        patch = out[region]
        patch[prev == 0] = color
        patch[prev == 1] = MAROON
        patch[prev >= 2] = MAGENTA
        out[region] = patch
        counts[region] = prev + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore partially occluded rectangles using the bottom layer-order strip, then recolor overlaps by depth.
    """
    canvas, order = _parse_layer_order(grid)
    rects = _reconstruct_rectangles(canvas, order)
    return _paint_overlap(rects, order, canvas.shape)


def _sample_strip_positions(width: int, n: int) -> list[int]:
    slots = list(range(1, width - 1, 2))
    assert len(slots) >= n
    return sorted(random.sample(slots, n))


def _sample_zone(h: int, w: int, kind: str) -> tuple[int, int, int, int]:
    zh = random.randint(*ZONE_SIZE_RANGE)
    zw = random.randint(*ZONE_SIZE_RANGE)

    if kind == "center":
        r0 = random.randint(4, h - zh - 5)
        c0 = random.randint(4, w - zw - 5)
        return (r0, c0, r0 + zh - 1, c0 + zw - 1)

    if kind == "upper":
        r0 = random.randint(3, max(3, h // 2 - zh - 3))
        c0 = random.randint(4, w - zw - 5)
        return (r0, c0, r0 + zh - 1, c0 + zw - 1)

    if kind == "lower":
        r0 = random.randint(h // 2 + 1, h - zh - 4)
        c0 = random.randint(4, w - zw - 5)
        return (r0, c0, r0 + zh - 1, c0 + zw - 1)

    if kind == "left":
        r0 = random.randint(4, h - zh - 5)
        c0 = random.randint(3, max(3, w // 2 - zw - 3))
        return (r0, c0, r0 + zh - 1, c0 + zw - 1)

    if kind == "right":
        r0 = random.randint(4, h - zh - 5)
        c0 = random.randint(w // 2 + 1, w - zw - 4)
        return (r0, c0, r0 + zh - 1, c0 + zw - 1)

    raise ValueError(kind)


def _sample_bbox_for_orientation(
    h: int,
    w: int,
    orient: str,
    zones: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
    avoid_rects: list[tuple[int, int, int, int]] | None = None,
) -> tuple[int, int, int, int] | None:
    avoid_rects = avoid_rects or []
    zr0 = min(z[0] for z in zones)
    zc0 = min(z[1] for z in zones)
    zr1 = max(z[2] for z in zones)
    zc1 = max(z[3] for z in zones)

    top_room = zr0 - 1
    left_room = zc0 - 1
    bottom_room = (h - 2) - zr1
    right_room = (w - 2) - zc1

    for _ in range(BOX_SAMPLE_RETRIES):
        if orient == "TL":
            top_min, left_min, bottom_min, right_min = 2, 2, 1, 1
        elif orient == "TR":
            top_min, left_min, bottom_min, right_min = 2, 1, 1, 2
        elif orient == "BL":
            top_min, left_min, bottom_min, right_min = 1, 2, 2, 1
        else:
            top_min, left_min, bottom_min, right_min = 1, 1, 2, 2

        if min(top_room - top_min, left_room - left_min, bottom_room - bottom_min, right_room - right_min) < 0:
            return None

        top_ext = random.randint(top_min, min(top_room, top_min + 3))
        left_ext = random.randint(left_min, min(left_room, left_min + 3))
        bottom_ext = random.randint(bottom_min, min(bottom_room, bottom_min + 3))
        right_ext = random.randint(right_min, min(right_room, right_min + 3))

        bbox = (zr0 - top_ext, zc0 - left_ext, zr1 + bottom_ext, zc1 + right_ext)
        if _bbox_hits_cells(bbox, forbidden):
            continue
        if any(_boxes_intersect(bbox, other) for other in avoid_rects):
            continue
        if _bbox_area(bbox) < 20:
            continue
        return bbox
    return None


def _render_input(rects: list[tuple[int, int, int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    canvas = np.full((h, w), BG, dtype=int)
    for bbox, color in zip(rects, colors):
        r0, c0, r1, c1 = bbox
        canvas[r0 : r1 + 1, c0 : c1 + 1] = color
    strip = np.full((1, w), GREY, dtype=int)
    for pos, color in zip(_sample_strip_positions(w, len(colors)), colors):
        strip[0, pos] = color
    return np.vstack([canvas, strip])


def _coverage_counts(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def _scene_is_valid(inp: np.ndarray, rects: list[tuple[int, int, int, int]], colors: list[int], mode: str) -> bool:
    canvas = inp[:-1]
    counts = _coverage_counts(rects, canvas.shape)
    if counts.max() < 2:
        return False
    if mode == "double_only" and counts.max() != 2:
        return False
    if mode == "triple" and counts.max() < 3:
        return False
    if not np.any(counts == 2):
        return False

    for color in colors:
        if int(np.sum(canvas == color)) < 6:
            return False

    expected = _paint_overlap(rects, colors, canvas.shape)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _generate_triple_scene(h: int, w: int) -> tuple[list[tuple[int, int, int, int]], str] | None:
    common_zone = _sample_zone(h, w, "center")
    n_rects = random.choice((3, 4))
    orientations = random.sample(list(ORIENTATIONS), n_rects)
    rects: list[tuple[int, int, int, int]] = []
    forbidden: set[tuple[int, int]] = set()

    for orient in orientations:
        bbox = _sample_bbox_for_orientation(h, w, orient, [common_zone], forbidden)
        if bbox is None:
            return None
        rects.append(bbox)
        forbidden |= _protected_cells(bbox, orient)

    return rects, "triple"


def _generate_double_scene(h: int, w: int) -> tuple[list[tuple[int, int, int, int]], str] | None:
    layout = random.choice(("v_left", "v_right", "h_top", "h_bottom"))
    rects: list[tuple[int, int, int, int]] = []
    forbidden: set[tuple[int, int]] = set()

    if layout.startswith("v"):
        for _ in range(20):
            zone_a = _sample_zone(h, w, "upper")
            zone_b = _sample_zone(h, w, "lower")
            if zone_a[2] + 2 < zone_b[0]:
                break
        else:
            return None

        if layout == "v_left":
            base_orient = random.choice(("TL", "BL"))
            branches = [("TR", zone_a), ("BR", zone_b)]
        else:
            base_orient = random.choice(("TR", "BR"))
            branches = [("TL", zone_a), ("BL", zone_b)]
    else:
        for _ in range(20):
            zone_a = _sample_zone(h, w, "left")
            zone_b = _sample_zone(h, w, "right")
            if zone_a[3] + 2 < zone_b[1]:
                break
        else:
            return None

        if layout == "h_top":
            base_orient = random.choice(("TL", "TR"))
            branches = [("BL", zone_a), ("BR", zone_b)]
        else:
            base_orient = random.choice(("BL", "BR"))
            branches = [("TL", zone_a), ("TR", zone_b)]

    base_bbox = _sample_bbox_for_orientation(h, w, base_orient, [zone_a, zone_b], forbidden)
    if base_bbox is None:
        return None
    rects.append(base_bbox)
    forbidden |= _protected_cells(base_bbox, base_orient)

    random.shuffle(branches)
    branch_rects: list[tuple[int, int, int, int]] = []
    for orient, zone in branches:
        bbox = _sample_bbox_for_orientation(h, w, orient, [zone], forbidden, avoid_rects=branch_rects)
        if bbox is None:
            return None
        branch_rects.append(bbox)
        forbidden |= _protected_cells(bbox, orient)

    rects.extend(branch_rects)
    return rects, "double_only"


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    mode = random.choice(("double_only", "triple"))

    scene = _generate_triple_scene(h, w) if mode == "triple" else _generate_double_scene(h, w)
    if scene is None:
        return None
    rects, mode_name = scene
    colors = random.sample(list(RECT_PALETTE), len(rects))
    inp = _render_input(rects, colors, (h, w))
    if not _scene_is_valid(inp, rects, colors, mode_name):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
