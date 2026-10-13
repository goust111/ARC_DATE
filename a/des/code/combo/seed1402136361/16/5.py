from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h

    @property
    def c1(self) -> int:
        return self.c0 + self.w

    @property
    def area(self) -> int:
        return self.h * self.w

    def shifted(self, dr: int, dc: int) -> "Rect":
        return Rect(self.color, self.r0 + dr, self.c0 + dc, self.h, self.w)


# Canonical translucent color system used by this task family:
# blue + red -> magenta, blue + green -> cyan, red + green -> yellow.
PRIMARY_COLORS = (1, 2, 3)
PRIMARY_BITS = {1: 1, 2: 2, 3: 4}
VISIBLE_FROM_BITS = {
    1: 1,
    2: 2,
    4: 3,
    3: 6,
    5: 8,
    6: 4,
}
RELATED_VISIBLE = {
    1: (1, 6, 8),
    2: (2, 6, 4),
    3: (3, 8, 4),
}
BACKGROUND_OPTIONS = (0, 5, 7, 9)


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_from_bbox(color: int, bbox: tuple[int, int, int, int]) -> Rect:
    r0, c0, r1, c1 = bbox
    return Rect(color, r0, c0, r1 - r0 + 1, c1 - c0 + 1)


def _render_visible(rects: list[Rect], bg: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    bits = np.zeros((height, width), dtype=int)
    for rect in rects:
        bits[rect.r0 : rect.r1, rect.c0 : rect.c1] |= PRIMARY_BITS[rect.color]
    assert not np.any(bits == 7), "triple overlap is not allowed"

    out = np.full((height, width), int(bg), dtype=int)
    for bitmask, color in VISIBLE_FROM_BITS.items():
        out[bits == bitmask] = int(color)
    return out


def _expected_output(rects: list[Rect], bg: int) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: (-rect.area, rect.r0, rect.c0))
    out_h = max(rect.h for rect in ordered)
    out_w = sum(rect.w for rect in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), int(bg), dtype=int)

    col = 0
    for rect in ordered:
        out[: rect.h, col : col + rect.w] = int(rect.color)
        col += rect.w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each translucent base rectangle and lay the solid rectangles out by area.

    The task family uses a fixed visible mixing rule:
    - blue(1) + red(2) -> magenta(6)
    - blue(1) + green(3) -> cyan(8)
    - red(2) + green(3) -> yellow(4)

    For each primary color, the full hidden rectangle is exactly the bounding box
    of all cells that either show that primary color directly or show a mixed color
    that contains it. The recovered solid rectangles are then packed left-to-right
    in descending area order with one background column between neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    rects: list[Rect] = []
    for color in PRIMARY_COLORS:
        mask = np.isin(g, np.array(RELATED_VISIBLE[color], dtype=int))
        if not mask.any():
            continue
        rects.append(_rect_from_bbox(color, _bbox_from_mask(mask)))

    assert len(rects) >= 2, "expected at least two primary rectangles"
    return _expected_output(rects, bg)


# --------------- generator ---------------

GEN_RETRIES = 400
PAIR_DIM_RANGE = (4, 8)
ANCHOR_DIM_RANGE = (7, 10)
OTHER_DIM_RANGE = (4, 7)
PAD_RANGE = (1, 5)
GRID_LIMIT = 30
MIN_OVERLAP = 2
MODES = ("pair", "line3", "corner3")


def _rand_size(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _sample_dims_pair() -> list[tuple[int, int]]:
    for _ in range(100):
        dims = [(_rand_size(PAIR_DIM_RANGE), _rand_size(PAIR_DIM_RANGE)) for _ in range(2)]
        areas = {h * w for h, w in dims}
        if len(areas) == 2:
            return dims
    raise RuntimeError("failed to sample unique pair dimensions")


def _sample_dims_three() -> list[tuple[int, int]]:
    for _ in range(200):
        anchor = (_rand_size(ANCHOR_DIM_RANGE), _rand_size(ANCHOR_DIM_RANGE))
        others = [(_rand_size(OTHER_DIM_RANGE), _rand_size(OTHER_DIM_RANGE)) for _ in range(2)]
        dims = [anchor, others[0], others[1]]
        areas = [h * w for h, w in dims]
        if len(set(areas)) != 3:
            continue
        if areas[0] <= max(areas[1:]):
            continue
        return dims
    raise RuntimeError("failed to sample unique triple dimensions")


def _place_corner(anchor: Rect, color: int, h: int, w: int, corner: str, oh: int, ow: int) -> Rect:
    if corner == "tl":
        return Rect(color, anchor.r0 - (h - oh), anchor.c0 - (w - ow), h, w)
    if corner == "tr":
        return Rect(color, anchor.r0 - (h - oh), anchor.c0 + anchor.w - ow, h, w)
    if corner == "bl":
        return Rect(color, anchor.r0 + anchor.h - oh, anchor.c0 - (w - ow), h, w)
    if corner == "br":
        return Rect(color, anchor.r0 + anchor.h - oh, anchor.c0 + anchor.w - ow, h, w)
    raise ValueError(f"unknown corner: {corner}")


def _overlap_area(a: Rect, b: Rect) -> int:
    dh = max(0, min(a.r1, b.r1) - max(a.r0, b.r0))
    dw = max(0, min(a.c1, b.c1) - max(a.c0, b.c0))
    return dh * dw


def _pure_area(rect: Rect, others: list[Rect]) -> int:
    cover = np.zeros((rect.h, rect.w), dtype=bool)
    for other in others:
        rr0 = max(rect.r0, other.r0)
        rr1 = min(rect.r1, other.r1)
        cc0 = max(rect.c0, other.c0)
        cc1 = min(rect.c1, other.c1)
        if rr0 < rr1 and cc0 < cc1:
            cover[rr0 - rect.r0 : rr1 - rect.r0, cc0 - rect.c0 : cc1 - rect.c0] = True
    return rect.area - int(cover.sum())


def _normalize_with_padding(rects: list[Rect]) -> tuple[list[Rect], tuple[int, int]] | None:
    min_r = min(rect.r0 for rect in rects)
    min_c = min(rect.c0 for rect in rects)
    max_r = max(rect.r1 for rect in rects)
    max_c = max(rect.c1 for rect in rects)
    draw_h = max_r - min_r
    draw_w = max_c - min_c

    top = _rand_size(PAD_RANGE)
    bottom = _rand_size(PAD_RANGE)
    left = _rand_size(PAD_RANGE)
    right = _rand_size(PAD_RANGE)
    height = draw_h + top + bottom
    width = draw_w + left + right
    if height > GRID_LIMIT or width > GRID_LIMIT:
        return None

    shifted = [rect.shifted(top - min_r, left - min_c) for rect in rects]
    return shifted, (height, width)


def _valid_scene(rects: list[Rect], shape: tuple[int, int]) -> bool:
    height, width = shape
    if not (1 <= height <= GRID_LIMIT and 1 <= width <= GRID_LIMIT):
        return False

    for rect in rects:
        if rect.r0 < 0 or rect.c0 < 0 or rect.r1 > height or rect.c1 > width:
            return False

    areas = [rect.area for rect in rects]
    if len(set(areas)) != len(areas):
        return False

    overlap_pairs = 0
    for i, rect in enumerate(rects):
        if _pure_area(rect, rects[:i] + rects[i + 1 :]) < MIN_OVERLAP * MIN_OVERLAP:
            return False
        for other in rects[i + 1 :]:
            area = _overlap_area(rect, other)
            if area > 0:
                overlap_pairs += 1
                if area < MIN_OVERLAP * MIN_OVERLAP:
                    return False

    visible = _render_visible(rects, bg=0, shape=shape) != 0
    if visible.sum() * 2 >= height * width:
        return False

    return overlap_pairs >= 1


def _layout_pair() -> list[Rect] | None:
    dims = _sample_dims_pair()
    colors = random.sample(PRIMARY_COLORS, 2)
    base = Rect(colors[0], 0, 0, dims[0][0], dims[0][1])
    h1, w1 = dims[1]

    max_oh = min(base.h - 2, h1 - 2)
    max_ow = min(base.w - 2, w1 - 2)
    if max_oh < MIN_OVERLAP or max_ow < MIN_OVERLAP:
        return None

    oh = random.randint(MIN_OVERLAP, max_oh)
    ow = random.randint(MIN_OVERLAP, max_ow)
    corner = random.choice(("tl", "tr", "bl", "br"))
    other = _place_corner(base, colors[1], h1, w1, corner, oh, ow)
    return [base, other]


def _layout_line3() -> list[Rect] | None:
    dims = _sample_dims_three()
    colors = random.sample(PRIMARY_COLORS, 3)
    center = Rect(colors[0], 0, 0, dims[0][0], dims[0][1])
    a_h, a_w = dims[1]
    b_h, b_w = dims[2]

    if random.random() < 0.5:
        max_ow_left = min(a_w - 2, center.w - 3)
        max_ow_right = min(b_w - 2, center.w - 3)
        if max_ow_left < MIN_OVERLAP or max_ow_right < MIN_OVERLAP:
            return None
        ow_left = random.randint(MIN_OVERLAP, max_ow_left)
        max_right_after_left = min(max_ow_right, center.w - ow_left - 1)
        if max_right_after_left < MIN_OVERLAP:
            return None
        ow_right = random.randint(MIN_OVERLAP, max_right_after_left)

        max_oh_left = min(a_h - 2, center.h - 2)
        max_oh_right = min(b_h - 2, center.h - 2)
        if max_oh_left < MIN_OVERLAP or max_oh_right < MIN_OVERLAP:
            return None
        oh_left = random.randint(MIN_OVERLAP, max_oh_left)
        oh_right = random.randint(MIN_OVERLAP, max_oh_right)

        top_overlap_left = random.randint(0, center.h - oh_left)
        top_in_left = random.randint(0, a_h - oh_left)
        left = Rect(
            colors[1],
            center.r0 + top_overlap_left - top_in_left,
            center.c0 - (a_w - ow_left),
            a_h,
            a_w,
        )

        top_overlap_right = random.randint(0, center.h - oh_right)
        top_in_right = random.randint(0, b_h - oh_right)
        right = Rect(
            colors[2],
            center.r0 + top_overlap_right - top_in_right,
            center.c0 + center.w - ow_right,
            b_h,
            b_w,
        )
        if _overlap_area(left, right) > 0:
            return None
        return [center, left, right]

    max_oh_top = min(a_h - 2, center.h - 3)
    max_oh_bottom = min(b_h - 2, center.h - 3)
    if max_oh_top < MIN_OVERLAP or max_oh_bottom < MIN_OVERLAP:
        return None
    oh_top = random.randint(MIN_OVERLAP, max_oh_top)
    max_bottom_after_top = min(max_oh_bottom, center.h - oh_top - 1)
    if max_bottom_after_top < MIN_OVERLAP:
        return None
    oh_bottom = random.randint(MIN_OVERLAP, max_bottom_after_top)

    max_ow_top = min(a_w - 2, center.w - 2)
    max_ow_bottom = min(b_w - 2, center.w - 2)
    if max_ow_top < MIN_OVERLAP or max_ow_bottom < MIN_OVERLAP:
        return None
    ow_top = random.randint(MIN_OVERLAP, max_ow_top)
    ow_bottom = random.randint(MIN_OVERLAP, max_ow_bottom)

    left_overlap_top = random.randint(0, center.w - ow_top)
    left_in_top = random.randint(0, a_w - ow_top)
    top = Rect(
        colors[1],
        center.r0 - (a_h - oh_top),
        center.c0 + left_overlap_top - left_in_top,
        a_h,
        a_w,
    )

    left_overlap_bottom = random.randint(0, center.w - ow_bottom)
    left_in_bottom = random.randint(0, b_w - ow_bottom)
    bottom = Rect(
        colors[2],
        center.r0 + center.h - oh_bottom,
        center.c0 + left_overlap_bottom - left_in_bottom,
        b_h,
        b_w,
    )
    if _overlap_area(top, bottom) > 0:
        return None
    return [center, top, bottom]


def _layout_corner3() -> list[Rect] | None:
    dims = _sample_dims_three()
    colors = random.sample(PRIMARY_COLORS, 3)
    anchor = Rect(colors[0], 0, 0, dims[0][0], dims[0][1])
    a_h, a_w = dims[1]
    b_h, b_w = dims[2]

    max_oh_a = min(a_h - 2, anchor.h - 3)
    max_oh_b = min(b_h - 2, anchor.h - 3)
    max_ow_a = min(a_w - 2, anchor.w - 3)
    max_ow_b = min(b_w - 2, anchor.w - 3)
    if min(max_oh_a, max_oh_b, max_ow_a, max_ow_b) < MIN_OVERLAP:
        return None

    for _ in range(40):
        oh_a = random.randint(MIN_OVERLAP, max_oh_a)
        oh_b = random.randint(MIN_OVERLAP, max_oh_b)
        ow_a = random.randint(MIN_OVERLAP, max_ow_a)
        ow_b = random.randint(MIN_OVERLAP, max_ow_b)
        if oh_a + oh_b >= anchor.h or ow_a + ow_b >= anchor.w:
            continue

        if random.random() < 0.5:
            rect_a = _place_corner(anchor, colors[1], a_h, a_w, "tl", oh_a, ow_a)
            rect_b = _place_corner(anchor, colors[2], b_h, b_w, "br", oh_b, ow_b)
        else:
            rect_a = _place_corner(anchor, colors[1], a_h, a_w, "tr", oh_a, ow_a)
            rect_b = _place_corner(anchor, colors[2], b_h, b_w, "bl", oh_b, ow_b)

        if _overlap_area(rect_a, rect_b) == 0:
            return [anchor, rect_a, rect_b]
    return None


def _sample_rects(mode: str) -> list[Rect] | None:
    if mode == "pair":
        return _layout_pair()
    if mode == "line3":
        return _layout_line3()
    if mode == "corner3":
        return _layout_corner3()
    raise ValueError(f"unknown mode: {mode}")


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    bg = random.choice(BACKGROUND_OPTIONS)
    rects = _sample_rects(mode)
    if rects is None:
        return None

    normalized = _normalize_with_padding(rects)
    if normalized is None:
        return None
    rects, shape = normalized
    if not _valid_scene(rects, shape):
        return None

    inp = _render_visible(rects, bg, shape)
    if _get_bg(inp) != bg:
        return None

    expected = _expected_output(rects, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangles input")
