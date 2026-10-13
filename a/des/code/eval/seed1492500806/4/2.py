from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
RECT_H_RANGE = (5, 11)
RECT_W_RANGE = (5, 11)
RECT_COUNT_OPTIONS = (2, 3, 4, 5)
RECT_COUNT_WEIGHTS = (3, 4, 3, 1)
MAX_GENERATION_ATTEMPTS = 600


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_corners(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _corners_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return any(_point_in_bbox(point, b) for point in _bbox_corners(a)) or any(
        _point_in_bbox(point, a) for point in _bbox_corners(b)
    )


def _pattern_vertical(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((ih, iw), dtype=bool)
    phase = rng.randrange(2)
    mask[:, phase::2] = True
    return mask


def _pattern_horizontal(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((ih, iw), dtype=bool)
    phase = rng.randrange(2)
    mask[phase::2, :] = True
    return mask


def _pattern_checker(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    phase = rng.randrange(2)
    rr, cc = np.indices((ih, iw))
    return ((rr + cc + phase) % 2) == 0


def _pattern_plus(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((ih, iw), dtype=bool)
    mask[ih // 2, :] = True
    mask[:, iw // 2] = True
    return mask


def _pattern_diagonals(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    del rng
    mask = np.zeros((ih, iw), dtype=bool)
    row_scale = max(ih - 1, 1)
    col_scale = max(iw - 1, 1)
    for r in range(ih):
        c1 = (r * col_scale + row_scale // 2) // row_scale
        c2 = iw - 1 - c1
        mask[r, c1] = True
        mask[r, c2] = True
    return mask


def _pattern_dots(ih: int, iw: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((ih, iw), dtype=bool)
    cells = [(r, c) for r in range(ih) for c in range(iw)]
    count = rng.randint(2, max(2, min(len(cells) - 1, (len(cells) + 2) // 4)))
    for r, c in rng.sample(cells, count):
        mask[r, c] = True
    return mask


def _make_inner_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    ih, iw = h - 2, w - 2
    builders = [
        _pattern_vertical,
        _pattern_horizontal,
        _pattern_checker,
        _pattern_plus,
        _pattern_diagonals,
        _pattern_dots,
    ]
    while True:
        mask = rng.choice(builders)(ih, iw, rng)
        if 0 < int(mask.sum()) < mask.size:
            return mask


def _make_patch(h: int, w: int, color: int, marked_corners: set[int], inner_mask: np.ndarray) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][inner_mask] = color

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for index in marked_corners:
        rr, cc = corners[index]
        patch[rr, cc] = MAROON
    return patch


def _render(shape: tuple[int, int], rects: list[dict], order: list[int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    for idx in order:
        rect = rects[idx]
        r0, c0, r1, c1 = rect["bbox"]
        patch = rect["patch"]
        sub = canvas[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BLACK
        sub[mask] = patch[mask]
    return canvas


def _recover_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects = []
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, MAROON)]
    for color in colors:
        points = np.argwhere(g == color)
        assert len(points) > 0, "each rectangle color must remain visible"
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "visible rectangle bbox must be at least 3x3"

        assert np.any(g[r0, c0 + 1 : c1] == color), "top side must expose a border pixel"
        assert np.any(g[r1, c0 + 1 : c1] == color), "bottom side must expose a border pixel"
        assert np.any(g[r0 + 1 : r1, c0] == color), "left side must expose a border pixel"
        assert np.any(g[r0 + 1 : r1, c1] == color), "right side must expose a border pixel"

        layer = np.zeros_like(g)
        patch = layer[r0 : r1 + 1, c0 : c1 + 1]
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        patch[1:-1, 1:-1][g[r0 + 1 : r1, c0 + 1 : c1] == color] = color

        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        local_corners = [(0, 0), (0, c1 - c0), (r1 - r0, 0), (r1 - r0, c1 - c0)]
        depth = 0
        for (gr, gc), (lr, lc) in zip(corners, local_corners):
            value = int(g[gr, gc])
            assert value in (color, MAROON), "rectangle corners must remain visible"
            if value == MAROON:
                patch[lr, lc] = MAROON
                depth += 1

        rects.append({"color": color, "depth": depth, "layer": layer})

    assert rects, "input must contain at least one rectangle"
    return rects


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rects = _recover_rectangles(g)
    rects.sort(key=lambda item: (int(item["depth"]), int(item["color"])))
    out = np.zeros_like(g)
    for rect in rects:
        layer = rect["layer"]
        mask = layer != BLACK
        out[mask] = layer[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _place_bbox(
    rng: random.Random,
    H: int,
    W: int,
    h: int,
    w: int,
    existing: list[dict],
    force_overlap: bool,
) -> tuple[int, int, int, int] | None:
    max_r0 = H - h
    max_c0 = W - w
    if max_r0 < 0 or max_c0 < 0:
        return None

    for _ in range(120):
        if force_overlap and existing and rng.random() < 0.7:
            anchor = rng.choice(existing)["bbox"]
            r_lo = max(0, anchor[0] - h + 2)
            r_hi = min(max_r0, anchor[2] - 1)
            c_lo = max(0, anchor[1] - w + 2)
            c_hi = min(max_c0, anchor[3] - 1)
            if r_lo <= r_hi and c_lo <= c_hi:
                r0 = rng.randint(r_lo, r_hi)
                c0 = rng.randint(c_lo, c_hi)
            else:
                r0 = rng.randint(0, max_r0)
                c0 = rng.randint(0, max_c0)
        else:
            r0 = rng.randint(0, max_r0)
            c0 = rng.randint(0, max_c0)

        cand = (r0, c0, r0 + h - 1, c0 + w - 1)
        overlaps = 0
        ok = True
        for other in existing:
            other_bbox = other["bbox"]
            inter = _intersection_area(cand, other_bbox)
            if inter == 0:
                continue
            overlaps += 1
            if inter > min(_bbox_area(cand), _bbox_area(other_bbox)) // 3:
                ok = False
                break
        if not ok:
            continue
        if force_overlap and overlaps == 0:
            continue
        return cand
    return None


def _has_any_overlap(rects: list[dict]) -> bool:
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection_area(rects[i]["bbox"], rects[j]["bbox"]) > 0:
                return True
    return False


def _validate_instance(rects: list[dict], inp: np.ndarray, expected: np.ndarray) -> bool:
    if not _has_any_overlap(rects):
        return False
    if np.array_equal(inp, expected):
        return False

    for rect in rects:
        color = int(rect["color"])
        bbox = rect["bbox"]
        patch = rect["patch"]
        r0, c0, r1, c1 = bbox

        points = np.argwhere(inp == color)
        if len(points) == 0:
            return False
        seen_bbox = tuple(map(int, (*points.min(axis=0), *points.max(axis=0))))
        if seen_bbox != bbox:
            return False

        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        local_corners = [(0, 0), (0, c1 - c0), (r1 - r0, 0), (r1 - r0, c1 - c0)]
        for (gr, gc), (lr, lc) in zip(corners, local_corners):
            if int(inp[gr, gc]) != int(patch[lr, lc]):
                return False

        if not np.any(inp[r0, c0 + 1 : c1] == color):
            return False
        if not np.any(inp[r1, c0 + 1 : c1] == color):
            return False
        if not np.any(inp[r0 + 1 : r1, c0] == color):
            return False
        if not np.any(inp[r0 + 1 : r1, c1] == color):
            return False

        inner_mask = patch[1:-1, 1:-1] == color
        if np.any(inp[r0 + 1 : r1, c0 + 1 : c1][inner_mask] != color):
            return False

    return np.array_equal(_solve(inp), expected)


def _generate_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    H = rng.randint(*GRID_H_RANGE)
    W = rng.randint(*GRID_W_RANGE)
    rect_count = rng.choices(RECT_COUNT_OPTIONS, weights=RECT_COUNT_WEIGHTS, k=1)[0]

    depths = rng.sample([0, 1, 2, 3, 4], rect_count)
    colors = rng.sample(COLOR_POOL, rect_count)

    rects: list[dict] = []
    for i in range(rect_count):
        h_hi = min(RECT_H_RANGE[1], H)
        w_hi = min(RECT_W_RANGE[1], W)
        if h_hi < RECT_H_RANGE[0] or w_hi < RECT_W_RANGE[0]:
            return None

        h = rng.randint(RECT_H_RANGE[0], h_hi)
        w = rng.randint(RECT_W_RANGE[0], w_hi)
        inner_mask = _make_inner_mask(h, w, rng)
        depth = depths[i]
        marked_corners = set(rng.sample(range(4), depth))
        bbox = _place_bbox(
            rng,
            H,
            W,
            h,
            w,
            rects,
            force_overlap=(i > 0 and (i == 1 or rng.random() < 0.75)),
        )
        if bbox is None:
            return None

        rects.append(
            {
                "color": colors[i],
                "depth": depth,
                "bbox": bbox,
                "patch": _make_patch(h, w, colors[i], marked_corners, inner_mask),
            }
        )

    sorted_order = sorted(range(rect_count), key=lambda idx: rects[idx]["depth"])
    for _ in range(20):
        input_order = list(range(rect_count))
        rng.shuffle(input_order)
        if input_order == sorted_order:
            continue
        inp = _render((H, W), rects, input_order)
        expected = _render((H, W), rects, sorted_order)
        if _validate_instance(rects, inp, expected):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _generate_instance(rng)
        if instance is not None:
            inp, _ = instance
            return inp
    raise RuntimeError("generate_input: exhausted attempts")
