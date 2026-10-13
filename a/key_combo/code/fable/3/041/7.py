from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _corners(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _draw_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def _assert_input_pixels_are_isolated(g: np.ndarray) -> None:
    h, w = g.shape
    for r, c in np.argwhere(g != BLACK):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            assert not (0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != BLACK), (
                "input corner pixels must be isolated"
            )


def _rect_from_three_corners(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == int(color))
    assert coords.shape == (3, 2), "each color must appear exactly three times"

    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    assert len(rows) == 2 and len(cols) == 2, "three pixels must span two rows and two columns"

    r0, r1 = rows
    c0, c1 = cols
    assert r0 < r1 and c0 < c1, "rectangle must have positive height and width"

    possible = set(_corners((r0, c0, r1, c1)))
    actual = {(int(r), int(c)) for r, c in coords}
    assert actual.issubset(possible) and len(possible - actual) == 1, (
        "the three pixels must be exactly three corners of one rectangle"
    )
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the hidden hollow rectangles implied by three same-colored corner pixels.

    Input structure:
    - A black background contains several isolated colored pixels.
    - Every non-black color appears exactly three times.
    - The three pixels of each color are three corners of one axis-aligned rectangle.

    Rule:
    - For each color, infer the two occupied rows and two occupied columns, which uniquely identify the missing fourth corner.
    - Draw the complete one-pixel-wide hollow rectangle border for that color on a black canvas of the same shape.
    - When rectangle borders overlap, draw larger-area rectangles first and smaller-area rectangles later, so smaller borders cover larger borders.

    Key insight:
    - The input shows only corner generators; after completing each invisible rectangle, the color at crossings is determined by hidden area-based layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors 0..9"

    _assert_input_pixels_are_isolated(g)

    colors = [int(c) for c in np.unique(g) if int(c) != BLACK]
    assert colors, "expected at least one colored rectangle"

    rects: list[dict] = []
    for color in colors:
        bbox = _rect_from_three_corners(g, color)
        rects.append({"color": int(color), "bbox": bbox, "area": _area(bbox)})

    areas = [int(rect["area"]) for rect in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct for unique layering"

    out = np.full_like(g, BLACK)
    for rect in sorted(rects, key=lambda item: int(item["area"]), reverse=True):
        _draw_border(out, rect["bbox"], int(rect["color"]))
    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 500
GRID_SIDE_RANGE = (16, 30)
RECT_COUNT_CHOICES = (2, 3, 3, 4, 5)
SMALL_SIDE_RANGE = (4, 8)
BIG_SIDE_RANGE = (8, 15)
EXTRA_SIDE_RANGE = (4, 10)


def _sample_crossing_pair(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(300):
        bh = random.randint(BIG_SIDE_RANGE[0], min(BIG_SIDE_RANGE[1], h - 2))
        bw = random.randint(BIG_SIDE_RANGE[0], min(BIG_SIDE_RANGE[1], w - 2))
        br0 = random.randint(1, h - bh - 1)
        bc0 = random.randint(1, w - bw - 1)
        big = (br0, bc0, br0 + bh - 1, bc0 + bw - 1)

        sh = random.randint(SMALL_SIDE_RANGE[0], min(SMALL_SIDE_RANGE[1], h - 2))
        sw = random.randint(SMALL_SIDE_RANGE[0], min(SMALL_SIDE_RANGE[1], w - 2))
        if sh * sw >= bh * bw:
            continue

        if random.choice((True, False)):
            small = _cross_small_vertical_side(h, w, big, sh, sw)
        else:
            small = _cross_small_horizontal_side(h, w, big, sh, sw)
        if small is not None and _area(small) < _area(big):
            return big, small
    return None


def _cross_small_vertical_side(
    h: int, w: int, big: tuple[int, int, int, int], sh: int, sw: int
) -> tuple[int, int, int, int] | None:
    br0, bc0, br1, bc1 = map(int, big)
    cross_row = random.choice((br0, br1))
    r_lo = max(0, cross_row - sh + 2)
    r_hi = min(cross_row - 1, h - sh)
    if r_lo > r_hi:
        return None

    x_choices = list(range(bc0 + 1, bc1))
    random.shuffle(x_choices)
    side_choices = ["left", "right"]
    random.shuffle(side_choices)
    for x in x_choices:
        for side in side_choices:
            if side == "left":
                sc0 = x
            else:
                sc0 = x - sw + 1
            sc1 = sc0 + sw - 1
            if not (0 <= sc0 and sc1 < w):
                continue
            sr0 = random.randint(r_lo, r_hi)
            return int(sr0), int(sc0), int(sr0 + sh - 1), int(sc1)
    return None


def _cross_small_horizontal_side(
    h: int, w: int, big: tuple[int, int, int, int], sh: int, sw: int
) -> tuple[int, int, int, int] | None:
    br0, bc0, br1, bc1 = map(int, big)
    cross_col = random.choice((bc0, bc1))
    c_lo = max(0, cross_col - sw + 2)
    c_hi = min(cross_col - 1, w - sw)
    if c_lo > c_hi:
        return None

    y_choices = list(range(br0 + 1, br1))
    random.shuffle(y_choices)
    side_choices = ["top", "bottom"]
    random.shuffle(side_choices)
    for y in y_choices:
        for side in side_choices:
            if side == "top":
                sr0 = y
            else:
                sr0 = y - sh + 1
            sr1 = sr0 + sh - 1
            if not (0 <= sr0 and sr1 < h):
                continue
            sc0 = random.randint(c_lo, c_hi)
            return int(sr0), int(sc0), int(sr1), int(sc0 + sw - 1)
    return None


def _sample_extra_bbox(h: int, w: int, used_areas: set[int]) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        rh = random.randint(EXTRA_SIDE_RANGE[0], min(EXTRA_SIDE_RANGE[1], h - 2))
        rw = random.randint(EXTRA_SIDE_RANGE[0], min(EXTRA_SIDE_RANGE[1], w - 2))
        area = rh * rw
        if area in used_areas:
            continue
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        return int(r0), int(c0), int(r0 + rh - 1), int(c0 + rw - 1)
    return None


def _render_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    expected = np.full(shape, BLACK, dtype=int)
    for spec in sorted(specs, key=lambda item: _area(item["bbox"]), reverse=True):
        _draw_border(expected, spec["bbox"], int(spec["color"]))
    return expected


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    inp = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        corners = _corners(spec["bbox"])
        missing = int(spec["missing"])
        for idx, (r, c) in enumerate(corners):
            if idx != missing:
                if int(inp[r, c]) != BLACK:
                    raise ValueError("two input corner pixels collided")
                inp[r, c] = color
    return inp


def _is_isolated_input(inp: np.ndarray) -> bool:
    h, w = inp.shape
    for r, c in np.argwhere(inp != BLACK):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) != BLACK:
                return False
    return True


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if not _is_isolated_input(inp):
        return False

    seen_missing = {int(spec["missing"]) for spec in specs}
    if len(seen_missing) < min(2, len(specs)):
        return False

    areas = [_area(spec["bbox"]) for spec in specs]
    if len(set(areas)) != len(areas):
        return False

    masks = [_border_cells(spec["bbox"]) for spec in specs]
    clear_two_layer_overlap = False
    for i in range(len(specs)):
        if int(np.count_nonzero(expected == int(specs[i]["color"]))) < 4:
            return False
        for j in range(i + 1, len(specs)):
            common = masks[i] & masks[j]
            for r, c in common:
                covering = [k for k, mask in enumerate(masks) if (r, c) in mask]
                if len(covering) != 2:
                    continue
                top = min(covering, key=lambda k: _area(specs[k]["bbox"]))
                bottom = max(covering, key=lambda k: _area(specs[k]["bbox"]))
                if _area(specs[top]["bbox"]) < _area(specs[bottom]["bbox"]) and int(expected[r, c]) == int(specs[top]["color"]):
                    clear_two_layer_overlap = True

    if not clear_two_layer_overlap:
        return False

    for color in {int(spec["color"]) for spec in specs}:
        if int(np.count_nonzero(inp == color)) != 3:
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    count = random.choice([n for n in RECT_COUNT_CHOICES if n <= len(COLORS)])

    pair = _sample_crossing_pair(h, w)
    if pair is None:
        return None

    colors = random.sample(COLORS, count)
    specs = [
        {"color": int(colors[0]), "bbox": pair[0]},
        {"color": int(colors[1]), "bbox": pair[1]},
    ]
    used_areas = {_area(pair[0]), _area(pair[1])}

    for color in colors[2:]:
        bbox = _sample_extra_bbox(h, w, used_areas)
        if bbox is None:
            return None
        specs.append({"color": int(color), "bbox": bbox})
        used_areas.add(_area(bbox))

    missing_order = list(range(4))
    random.shuffle(missing_order)
    while len(missing_order) < len(specs):
        extra = list(range(4))
        random.shuffle(extra)
        missing_order.extend(extra)
    random.shuffle(specs)
    for spec, missing in zip(specs, missing_order):
        spec["missing"] = int(missing)

    try:
        inp = _render_input((h, w), specs)
    except ValueError:
        return None

    expected = _render_expected((h, w), specs)
    if not _semantic_checks(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rectangle input")
