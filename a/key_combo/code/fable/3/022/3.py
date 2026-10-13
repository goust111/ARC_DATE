from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 500
RECT_GAP = 1


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _size_key(h: int, w: int) -> tuple[int, int]:
    return tuple(sorted((int(h), int(w))))


def _rect_from_color(g: np.ndarray, color: int) -> dict:
    coords = np.argwhere(g == int(color))
    assert coords.shape[0] == 2, "each non-black color must appear exactly twice"
    (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in coords]
    assert r0 != r1 and c0 != c1, "paired corners must be opposite diagonal corners"
    top, bottom = sorted((r0, r1))
    left, right = sorted((c0, c1))
    box = (top, left, bottom, right)
    h = _bbox_h(box)
    w = _bbox_w(box)
    return {
        "color": int(color),
        "bbox": box,
        "height": h,
        "width": w,
        "size": _size_key(h, w),
    }


def _extract_rectangles(g: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(g) if int(c) != BLACK]
    assert len(colors) >= 3 and len(colors) % 2 == 1, "expected paired rectangles plus one singleton"

    rects = [_rect_from_color(g, color) for color in colors]

    occupied = np.zeros(g.shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool(region.any()), "implicit rectangles must not overlap"
        region[:, :] = True

    return rects


def _group_by_size(rects: list[dict]) -> dict[tuple[int, int], list[dict]]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rect in rects:
        groups[rect["size"]].append(rect)

    singleton_count = 0
    for group in groups.values():
        assert len(group) in (1, 2), "each size class must be a pair or the lone singleton"
        singleton_count += len(group) == 1
    assert singleton_count == 1, "expected exactly one unpaired rectangle"
    return groups


def _paint_solid(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_outline(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore hidden rectangles from same-color corner pairs, swapping partner colors and outlining the lone rectangle.

    Input structure:
    - The grid has a black background and several colored singleton pixels.
    - Each non-black color appears exactly twice; those two pixels are opposite corners of one axis-aligned hidden rectangle.
    - Hidden rectangles do not overlap, and size classes are unique except for two-rectangle partner classes.

    Rule:
    - Reconstruct every rectangle from its two same-color corners.
    - Rectangles with equal height/width after allowing transposition are partners.
    - Each partner rectangle is filled solid in the other partner's color.
    - The one rectangle without a same-size partner is drawn only as a border in its own color, leaving its interior black.

    Key insight:
    - The visible pixels are only corner labels; the color swap is determined by matching reconstructed rectangle sizes, not by position or color order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert BLACK in set(map(int, np.unique(g))), "expected black background"

    rects = _extract_rectangles(g)
    groups = _group_by_size(rects)

    out = np.zeros_like(g, dtype=int)
    for group in groups.values():
        if len(group) == 2:
            first, second = group
            _paint_solid(out, first["bbox"], second["color"])
            _paint_solid(out, second["bbox"], first["color"])
        else:
            (single,) = group
            _paint_outline(out, single["bbox"], single["color"])
    return out


# ----------------------------- generator -----------------------------


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = RECT_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_rectangles(shape: tuple[int, int], dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    rows, cols = map(int, shape)
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    candidates: list[list[tuple[int, int, int, int]]] = []
    for h, w in dims:
        opts = [
            (r, c, r + int(h) - 1, c + int(w) - 1)
            for r in range(rows - int(h) + 1)
            for c in range(cols - int(w) + 1)
        ]
        random.shuffle(opts)
        candidates.append(opts)

    placed: list[tuple[int, int, int, int] | None] = [None] * len(dims)

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        for box in candidates[idx]:
            if all(prev is None or _separated(box, prev) for prev in placed):
                placed[idx] = box
                if rec(k + 1):
                    return True
                placed[idx] = None
        return False

    if not rec(0):
        return None
    return [box for box in placed if box is not None]


def _all_size_keys() -> list[tuple[int, int]]:
    return [(h, w) for h in range(3, 11) for w in range(h, 11)]


def _choose_specs() -> list[dict]:
    pair_count = random.choice((2, 2, 3))
    equal_area_cores = (
        ((3, 8), (4, 6)),
        ((3, 10), (5, 6)),
        ((4, 10), (5, 8)),
    )
    transposed_key, same_key = random.choice(equal_area_cores)

    pair_keys = [transposed_key, same_key]
    used_keys = set(pair_keys)

    pool = [key for key in _all_size_keys() if key not in used_keys]
    random.shuffle(pool)
    while len(pair_keys) < pair_count:
        key = pool.pop()
        pair_keys.append(key)
        used_keys.add(key)

    singleton_key = pool.pop()
    colors = random.sample(COLORS, 2 * pair_count + 1)
    random.shuffle(colors)

    specs: list[dict] = []
    color_i = 0

    # Mandatory non-square transposed pair: h x w matched to w x h.
    h, w = transposed_key
    specs.append({"color": colors[color_i], "dims": (h, w), "pair": 0})
    color_i += 1
    specs.append({"color": colors[color_i], "dims": (w, h), "pair": 0})
    color_i += 1

    # Mandatory same-orientation pair with the same area as the first pair.
    h, w = same_key
    if random.random() < 0.5:
        h, w = w, h
    specs.append({"color": colors[color_i], "dims": (h, w), "pair": 1})
    color_i += 1
    specs.append({"color": colors[color_i], "dims": (h, w), "pair": 1})
    color_i += 1

    for pair_idx, key in enumerate(pair_keys[2:], start=2):
        h, w = key
        if h != w and random.random() < 0.55:
            dims = [(h, w), (w, h)]
        else:
            dims0 = (w, h) if h != w and random.random() < 0.5 else (h, w)
            dims = [dims0, dims0]
        for dims_i in dims:
            specs.append({"color": colors[color_i], "dims": dims_i, "pair": pair_idx})
            color_i += 1

    sh, sw = singleton_key
    if sh != sw and random.random() < 0.5:
        sh, sw = sw, sh
    specs.append({"color": colors[color_i], "dims": (sh, sw), "pair": None})
    random.shuffle(specs)
    return specs


def _render_input(shape: tuple[int, int], specs: list[dict], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec, box in zip(specs, boxes):
        r0, c0, r1, c1 = box
        color = int(spec["color"])
        if random.random() < 0.5:
            corners = ((r0, c0), (r1, c1))
        else:
            corners = ((r0, c1), (r1, c0))
        for r, c in corners:
            assert grid[r, c] == BLACK
            grid[r, c] = color
    return grid


def _expected_output(shape: tuple[int, int], specs: list[dict], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    rects = []
    for spec, box in zip(specs, boxes):
        h = _bbox_h(box)
        w = _bbox_w(box)
        rects.append({"color": int(spec["color"]), "bbox": box, "height": h, "width": w, "size": _size_key(h, w)})

    out = np.zeros(shape, dtype=int)
    groups = _group_by_size(rects)
    for group in groups.values():
        if len(group) == 2:
            a, b = group
            _paint_solid(out, a["bbox"], b["color"])
            _paint_solid(out, b["bbox"], a["color"])
        else:
            (single,) = group
            _paint_outline(out, single["bbox"], single["color"])
    return out


def _semantic_checks(specs: list[dict], boxes: list[tuple[int, int, int, int]], inp: np.ndarray, out: np.ndarray) -> bool:
    colors = [int(spec["color"]) for spec in specs]
    if len(colors) != len(set(colors)):
        return False
    if any(int(np.count_nonzero(inp == color)) != 2 for color in colors):
        return False

    rects = []
    for spec, box in zip(specs, boxes):
        h = _bbox_h(box)
        w = _bbox_w(box)
        rects.append({**spec, "bbox": box, "height": h, "width": w, "size": _size_key(h, w)})

    groups = defaultdict(list)
    for rect in rects:
        groups[rect["size"]].append(rect)
    if sorted(len(group) for group in groups.values()).count(1) != 1:
        return False
    if any(len(group) not in (1, 2) for group in groups.values()):
        return False

    pair_groups = [group for group in groups.values() if len(group) == 2]
    has_transposed = any(
        group[0]["height"] == group[1]["width"]
        and group[0]["width"] == group[1]["height"]
        and group[0]["height"] != group[0]["width"]
        for group in pair_groups
    )
    if not has_transposed:
        return False

    pair_areas = [group[0]["height"] * group[0]["width"] for group in pair_groups]
    if len(pair_areas) == len(set(pair_areas)):
        return False

    singleton = next(group[0] for group in groups.values() if len(group) == 1)
    if singleton["height"] < 3 or singleton["width"] < 3:
        return False
    r0, c0, r1, c1 = singleton["bbox"]
    if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_specs()
    dims = [tuple(map(int, spec["dims"])) for spec in specs]
    pair_count = (len(specs) - 1) // 2
    min_side = 24 if pair_count == 2 else 27
    shape = (random.randint(min_side, 30), random.randint(min_side, 30))

    boxes = _place_rectangles(shape, dims)
    if boxes is None:
        return None

    inp = _render_input(shape, specs, boxes)
    expected = _expected_output(shape, specs, boxes)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(specs, boxes, inp, actual):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-pair hidden-rectangle input")
