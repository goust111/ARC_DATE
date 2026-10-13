from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


BG = 0
TEAL = 8
DEPTH2_COLOR = 5
DEPTH3_COLOR = 6
INPUT_COLORS = (1, 2, 3, 4, 7, 9)
MODE_SEQUENCE = ("pair", "triple", "clip")
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    g = np.asarray(mask, dtype=bool)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    if int(connectivity) == 4:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        neighbors = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not g[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


@dataclass(frozen=True)
class RectSpec:
    r0: int
    c0: int
    h: int
    w: int
    color: int
    dr: int
    dc: int

    @property
    def input_bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1)

    @property
    def output_bbox(self) -> tuple[int, int, int, int]:
        return (
            self.r0 + self.dr,
            self.c0 + self.dc,
            self.r0 + self.dr + self.h - 1,
            self.c0 + self.dc + self.w - 1,
        )

    @property
    def control_local(self) -> tuple[int, int]:
        return (
            0 if self.dr == -1 else self.h - 1,
            0 if self.dc == -1 else self.w - 1,
        )


def _clip_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h - 1, r1)
    c1 = min(w - 1, c1)
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _parse_rectangles(grid: np.ndarray) -> tuple[int, list[RectSpec]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=8)
    assert comps, "expected at least one rectangle component"

    teal_pixels = {tuple(map(int, p)) for p in np.argwhere(g == TEAL)}
    used_teal: set[tuple[int, int]] = set()
    rects: list[RectSpec] = []

    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = region.shape
        assert h >= 3 and w >= 3, "rectangles must be at least 3x3"
        assert np.all(region != bg), "rectangle bbox must be fully filled"

        marker_locs = np.argwhere(region == TEAL)
        assert marker_locs.shape == (1, 2), "each rectangle must contain one teal control pixel"
        mr, mc = map(int, marker_locs[0])
        assert mr in (0, h - 1) and mc in (0, w - 1), "teal pixel must be at a rectangle corner"

        colors = set(map(int, np.unique(region))) - {TEAL}
        assert len(colors) == 1, "each rectangle must have one main color"
        color = int(next(iter(colors)))

        expected = np.full((h, w), color, dtype=int)
        expected[mr, mc] = TEAL
        assert np.array_equal(region, expected), "component must be a filled rectangle with one replaced corner"

        dr = -1 if mr == 0 else 1
        dc = -1 if mc == 0 else 1
        used_teal.add((r0 + mr, c0 + mc))
        rects.append(RectSpec(r0=r0, c0=c0, h=h, w=w, color=color, dr=dr, dc=dc))

    assert used_teal == teal_pixels, "every teal pixel must belong to exactly one rectangle"
    return bg, rects


def _render_output(shape: tuple[int, int], rects: Iterable[RectSpec], bg: int = BG) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    h, w = shape
    depth = np.zeros((h, w), dtype=int)
    first_color = np.full((h, w), int(bg), dtype=int)
    clipped: list[bool] = []

    for rect in rects:
        visible = _clip_bbox(rect.output_bbox, shape)
        assert visible is not None, "a one-cell shift should leave some visible area"
        vr0, vc0, vr1, vc1 = visible
        or0, oc0, or1, oc1 = rect.output_bbox
        clipped.append((vr0, vc0, vr1, vc1) != (or0, oc0, or1, oc1))

        depth_view = depth[vr0 : vr1 + 1, vc0 : vc1 + 1]
        first_view = first_color[vr0 : vr1 + 1, vc0 : vc1 + 1]
        new_mask = depth_view == 0
        first_view[new_mask] = rect.color
        depth_view += 1

    out = np.full((h, w), int(bg), dtype=int)
    out[depth == 1] = first_color[depth == 1]
    out[depth == 2] = DEPTH2_COLOR
    out[depth >= 3] = DEPTH3_COLOR
    return out, depth, clipped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each teal-cornered filled rectangle by one diagonal step and recolor overlap depth.

    Input structure:
    - The grid has a dominant black background and 3 to 5 separated filled rectangles.
    - Each rectangle has one corner cell replaced by a teal control pixel.
    - Aside from that control pixel, each rectangle is monochrome and axis-aligned.

    Rule:
    - Treat the teal pixel as the rectangle's missing corner, restoring the full solid rectangle.
    - Move that completed rectangle one cell diagonally toward the teal corner and erase every original rectangle position.
    - Clip translated cells at the grid boundary, then count how many translated rectangles cover each output cell.
    - Keep depth-1 cells in the covering rectangle's original color, paint depth-2 cells grey, and paint depth-3-or-more cells magenta.

    Key insight:
    - The teal corner encodes both the diagonal translation vector and the corner that must be repaired before overlap depth is computed.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects = _parse_rectangles(g)
    out, _depth, _clipped = _render_output(g.shape, rects, bg=bg)
    return out


QUADRANT_TO_SHIFT = {
    "ul": (1, 1),
    "ur": (1, -1),
    "ll": (-1, 1),
    "lr": (-1, -1),
}

PAIR_CHOICES = (
    ("ul", "ur"),
    ("ul", "ll"),
    ("ur", "lr"),
    ("ll", "lr"),
    ("ul", "lr"),
    ("ur", "ll"),
)
TRIPLE_CHOICES = (
    ("ul", "ur", "ll"),
    ("ul", "ur", "lr"),
    ("ul", "ll", "lr"),
    ("ur", "ll", "lr"),
)

_gen_call_count = 0


def _rect_from_anchor(anchor_r: int, anchor_c: int, quadrant: str, h: int, w: int, color: int) -> RectSpec:
    dr, dc = QUADRANT_TO_SHIFT[quadrant]
    if quadrant == "ul":
        r0 = anchor_r - h
        c0 = anchor_c - w
    elif quadrant == "ur":
        r0 = anchor_r - h
        c0 = anchor_c + 1
    elif quadrant == "ll":
        r0 = anchor_r + 1
        c0 = anchor_c - w
    else:
        r0 = anchor_r + 1
        c0 = anchor_c + 1
    return RectSpec(r0=r0, c0=c0, h=h, w=w, color=color, dr=dr, dc=dc)


def _anchor_ranges(shape: tuple[int, int], placements: list[tuple[str, int, int]]) -> tuple[int, int, int, int] | None:
    h, w = shape
    min_r, max_r = 0, h - 1
    min_c, max_c = 0, w - 1
    for quadrant, rh, rw in placements:
        if quadrant in ("ul", "ur"):
            min_r = max(min_r, rh)
        else:
            max_r = min(max_r, h - rh - 1)
        if quadrant in ("ul", "ll"):
            min_c = max(min_c, rw)
        else:
            max_c = min(max_c, w - rw - 1)
    if min_r > max_r or min_c > max_c:
        return None
    return int(min_r), int(max_r), int(min_c), int(max_c)


def _random_group_specs(shape: tuple[int, int], quadrants: tuple[str, ...], colors: list[int]) -> list[RectSpec] | None:
    placements = []
    for quadrant in quadrants:
        rect_h = random.randint(3, 5)
        rect_w = random.randint(3, 5)
        placements.append((quadrant, rect_h, rect_w))

    ranges = _anchor_ranges(shape, placements)
    if ranges is None:
        return None
    min_r, max_r, min_c, max_c = ranges
    if max_r - min_r >= 4:
        min_r += 1
        max_r -= 1
    if max_c - min_c >= 4:
        min_c += 1
        max_c -= 1
    if min_r > max_r or min_c > max_c:
        return None

    anchor_r = random.randint(min_r, max_r)
    anchor_c = random.randint(min_c, max_c)
    specs = [
        _rect_from_anchor(anchor_r, anchor_c, quadrant, rect_h, rect_w, color)
        for (quadrant, rect_h, rect_w), color in zip(placements, colors, strict=True)
    ]

    for i, rect in enumerate(specs):
        ibox = rect.input_bbox
        if ibox[0] < 0 or ibox[1] < 0 or ibox[2] >= shape[0] or ibox[3] >= shape[1]:
            return None
        for other in specs[:i]:
            if _bboxes_too_close(ibox, other.input_bbox, gap=1):
                return None
    return specs


def _random_isolated_spec(
    shape: tuple[int, int],
    used_specs: list[RectSpec],
    color: int,
    *,
    clipped: bool,
) -> RectSpec | None:
    h, w = shape
    visible_boxes = [box for spec in used_specs if (box := _clip_bbox(spec.output_bbox, shape)) is not None]

    for _ in range(300):
        rect_h = random.randint(3, 5)
        rect_w = random.randint(3, 5)
        dr, dc = random.choice(tuple(QUADRANT_TO_SHIFT.values()))

        if clipped:
            axis = random.choice(("row", "col"))
            if axis == "row":
                r0 = 0 if dr == -1 else h - rect_h
                c_min = 1 if dc == -1 else 0
                c_max = w - rect_w - (1 if dc == 1 else 0)
                if c_min > c_max:
                    continue
                c0 = random.randint(c_min, c_max)
            else:
                c0 = 0 if dc == -1 else w - rect_w
                r_min = 1 if dr == -1 else 0
                r_max = h - rect_h - (1 if dr == 1 else 0)
                if r_min > r_max:
                    continue
                r0 = random.randint(r_min, r_max)
        else:
            r_min = 1 if dr == -1 else 0
            r_max = h - rect_h - (1 if dr == 1 else 0)
            c_min = 1 if dc == -1 else 0
            c_max = w - rect_w - (1 if dc == 1 else 0)
            if r_min > r_max or c_min > c_max:
                continue
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)

        candidate = RectSpec(r0=r0, c0=c0, h=rect_h, w=rect_w, color=color, dr=dr, dc=dc)

        if any(_bboxes_too_close(candidate.input_bbox, other.input_bbox, gap=1) for other in used_specs):
            continue
        visible = _clip_bbox(candidate.output_bbox, shape)
        if visible is None:
            continue
        if clipped == ((visible[0], visible[1], visible[2], visible[3]) == candidate.output_bbox):
            continue
        if any(_bboxes_too_close(visible, other_box, gap=1) for other_box in visible_boxes):
            continue
        return candidate
    return None


def _render_input(shape: tuple[int, int], rects: Iterable[RectSpec]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for rect in rects:
        out[rect.r0 : rect.r0 + rect.h, rect.c0 : rect.c0 + rect.w] = rect.color
        mr, mc = rect.control_local
        out[rect.r0 + mr, rect.c0 + mc] = TEAL
    return out


def _depth_stats(depth: np.ndarray) -> tuple[int, int, int]:
    depth2 = int(np.count_nonzero(depth == 2))
    depth3 = int(np.count_nonzero(depth >= 3))
    max_depth = int(depth.max()) if depth.size else 0
    return depth2, depth3, max_depth


def _input_is_well_formed(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    if bg != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if not (3 <= len(comps) <= 5):
        return False
    try:
        _parse_rectangles(grid)
    except AssertionError:
        return False
    return True


def _mode_target() -> str:
    global _gen_call_count
    mode = MODE_SEQUENCE[_gen_call_count % len(MODE_SEQUENCE)]
    _gen_call_count += 1
    return mode


def _choose_pair() -> tuple[str, ...]:
    if random.random() < 0.75:
        adjacent = PAIR_CHOICES[:4]
        return random.choice(adjacent)
    return random.choice(PAIR_CHOICES)


def _generate_case(mode: str) -> np.ndarray | None:
    if mode == "pair":
        total_rects = random.randint(3, 5)
        group_quadrants = _choose_pair()
    elif mode == "triple":
        total_rects = random.randint(3, 5)
        group_quadrants = random.choice(TRIPLE_CHOICES)
    else:
        total_rects = random.randint(4, 5)
        group_quadrants = _choose_pair()

    h = random.randint(14 if total_rects <= 4 else 15, 18)
    w = random.randint(14 if total_rects <= 4 else 15, 18)
    shape = (h, w)
    colors = random.sample(INPUT_COLORS, total_rects)

    group_specs = _random_group_specs(shape, group_quadrants, colors[: len(group_quadrants)])
    if group_specs is None:
        return None
    specs = list(group_specs)

    next_color_idx = len(group_specs)
    if mode == "clip":
        clipped_spec = _random_isolated_spec(shape, specs, colors[next_color_idx], clipped=True)
        if clipped_spec is None:
            return None
        specs.append(clipped_spec)
        next_color_idx += 1

    while next_color_idx < total_rects:
        extra = _random_isolated_spec(shape, specs, colors[next_color_idx], clipped=False)
        if extra is None:
            return None
        specs.append(extra)
        next_color_idx += 1

    grid = _render_input(shape, specs)
    expected, depth, clipped_flags = _render_output(shape, specs, bg=BG)
    depth2, depth3, max_depth = _depth_stats(depth)

    if not _input_is_well_formed(grid):
        return None
    if np.count_nonzero(grid == BG) <= (grid.size // 2):
        return None
    if depth2 == 0:
        return None
    if max_depth > 4:
        return None
    if np.count_nonzero(depth >= 2) * 5 > np.count_nonzero(depth >= 1) * 2:
        return None
    if mode == "pair" and max_depth != 2:
        return None
    if mode == "triple" and not (depth3 > 0 and max_depth >= 3):
        return None
    if mode == "clip" and not any(clipped_flags):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _mode_target()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample for mode={mode}")
