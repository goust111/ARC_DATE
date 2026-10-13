from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
MAX_ATTEMPTS = 400
GRID_HEIGHTS = tuple(range(12, 19))
GRID_WIDTHS = tuple(range(13, 21))
DEPTH_OPTIONS = (2, 3)
PAIR_SPAN_OPTIONS = (3, 4)
EXTRA_SPAN_OPTIONS = (2, 3)
SINGLETON_SPAN_OPTIONS = (2, 3, 4)
PALETTE = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _is_solid_rect(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(
        comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)
        and np.all(sub == int(comp["color"]))
    )


def _touch_side(comp: dict, h: int, w: int) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, f"each rectangle must touch exactly one border, got {sides}"
    return sides[0]


def _row_index(comps: list[dict], h: int) -> list[dict | None]:
    rows: list[dict | None] = [None] * h
    for comp in comps:
        r0, _c0, r1, _c1 = comp["bbox"]
        for r in range(r0, r1 + 1):
            assert rows[r] is None, "same-side repeated rectangles must have disjoint row bands"
            rows[r] = comp
    return rows


def _col_index(comps: list[dict], w: int) -> list[dict | None]:
    cols: list[dict | None] = [None] * w
    for comp in comps:
        _r0, c0, _r1, c1 = comp["bbox"]
        for c in range(c0, c1 + 1):
            assert cols[c] is None, "same-side repeated rectangles must have disjoint column bands"
            cols[c] = comp
    return cols


def _paint_horizontal(out: np.ndarray, occupied: np.ndarray, lefts: list[dict], rights: list[dict], color: int) -> None:
    h, w = map(int, out.shape)
    left_by_row = _row_index(lefts, h)
    right_by_row = _row_index(rights, h)

    for r in range(h):
        left = left_by_row[r]
        right = right_by_row[r]
        if left is not None and right is not None:
            start = int(left["bbox"][3]) + 1
            end = int(right["bbox"][1]) - 1
            assert start <= end, "opposing repeated rectangles must leave an interior corridor"
            occupied_between = np.flatnonzero(occupied[r, start : end + 1]) + start
            if occupied_between.size:
                first = int(occupied_between[0])
                last = int(occupied_between[-1])
                if start < first:
                    out[r, start:first] = color
                if last < end:
                    out[r, last + 1 : end + 1] = color
            else:
                gap = end - start + 1
                span = gap // 2
                if span > 0:
                    out[r, start : start + span] = color
                    out[r, end - span + 1 : end + 1] = color
        elif left is not None:
            start = int(left["bbox"][3]) + 1
            blockers = np.flatnonzero(occupied[r, start:w]) + start
            limit = int(blockers[0]) if blockers.size else w - 1
            if start < limit:
                out[r, start:limit] = color
        elif right is not None:
            end = int(right["bbox"][1]) - 1
            blockers = np.flatnonzero(occupied[r, : end + 1])
            limit = int(blockers[-1]) if blockers.size else 0
            if limit < end:
                out[r, limit + 1 : end + 1] = color


def _paint_vertical(out: np.ndarray, occupied: np.ndarray, tops: list[dict], bottoms: list[dict], color: int) -> None:
    h, w = map(int, out.shape)
    top_by_col = _col_index(tops, w)
    bottom_by_col = _col_index(bottoms, w)

    for c in range(w):
        top = top_by_col[c]
        bottom = bottom_by_col[c]
        if top is not None and bottom is not None:
            start = int(top["bbox"][2]) + 1
            end = int(bottom["bbox"][0]) - 1
            assert start <= end, "opposing repeated rectangles must leave an interior corridor"
            occupied_between = np.flatnonzero(occupied[start : end + 1, c]) + start
            if occupied_between.size:
                first = int(occupied_between[0])
                last = int(occupied_between[-1])
                if start < first:
                    out[start:first, c] = color
                if last < end:
                    out[last + 1 : end + 1, c] = color
            else:
                gap = end - start + 1
                span = gap // 2
                if span > 0:
                    out[start : start + span, c] = color
                    out[end - span + 1 : end + 1, c] = color
        elif top is not None:
            start = int(top["bbox"][2]) + 1
            blockers = np.flatnonzero(occupied[start:h, c]) + start
            limit = int(blockers[0]) if blockers.size else h - 1
            if start < limit:
                out[start:limit, c] = color
        elif bottom is not None:
            end = int(bottom["bbox"][0]) - 1
            blockers = np.flatnonzero(occupied[: end + 1, c])
            limit = int(blockers[-1]) if blockers.size else 0
            if limit < end:
                out[limit + 1 : end + 1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Emit inward rays from the repeated border rectangles, splitting unobstructed meetings and stopping at singleton blockers.

    Input structure:
    - A black background contains 4 to 5 solid monochromatic rectangles, each flush against exactly one border.
    - Exactly one foreground color appears in multiple separate 4-connected rectangles; every other foreground color appears once.
    - The repeated-color rectangles act as emitters, while singleton rectangles remain fixed and can sit inside the emitters' future corridors.

    Rule:
    - Identify the unique repeated foreground color by counting 4-connected monochromatic components.
    - Each repeated-color rectangle emits an inward ray from its non-border face, with width equal to the rectangle's span perpendicular to the ray.
    - A one-sided ray extends through background until just before the first non-background cell, or just before the opposite border if nothing blocks it.
    - When opposite-side repeated rectangles face each other on the same row band or column band with only background between them, split that background gap in half; if the gap length is odd, leave the center row or column black.
    - Singleton rectangles are preserved unchanged and only serve as blockers.

    Key insight:
    - The transformation is controlled by multiplicity first, then by border-facing ray direction, and finally by whether an opposing repeated ray reaches the same corridor without a singleton obstacle interrupting it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    comps = _find_components(g, bg)
    assert 4 <= len(comps) <= 5, "expected 4 to 5 foreground rectangles"
    assert all(_is_solid_rect(g, comp) for comp in comps), "all components must be solid rectangles"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    repeated_colors = [color for color, items in by_color.items() if len(items) > 1]
    assert len(repeated_colors) == 1, "expected exactly one repeated foreground color"
    repeated_color = int(repeated_colors[0])
    repeated = by_color[repeated_color]
    assert 2 <= len(repeated) <= 3, "expected 2 or 3 repeated-color rectangles"

    emitters = {"left": [], "right": [], "top": [], "bottom": []}
    for comp in repeated:
        emitters[_touch_side(comp, h, w)].append(comp)

    out = g.copy()
    occupied = g != bg
    _paint_horizontal(out, occupied, emitters["left"], emitters["right"], repeated_color)
    _paint_vertical(out, occupied, emitters["top"], emitters["bottom"], repeated_color)
    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _fill_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    assert np.all(grid[r0 : r0 + h, c0 : c0 + w] == BG)
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _choose_gap_width(left_depth: int, right_depth: int, want_odd: bool) -> int:
    candidates = [
        width
        for width in GRID_WIDTHS
        if width - left_depth - right_depth >= 7 and (width - left_depth - right_depth) % 2 == int(want_odd)
    ]
    return random.choice(candidates)


def _pick_extra_interval(height: int, span: int, blocked: tuple[int, int]) -> tuple[int, int]:
    candidates = []
    for r0 in range(1, height - span):
        r1 = r0 + span - 1
        if r1 >= height - 1:
            continue
        if r1 + 1 < blocked[0] or blocked[1] + 1 < r0:
            candidates.append((r0, r1))
    if not candidates:
        raise AssertionError("no room for extra repeated rectangle")
    return random.choice(candidates)


def _singleton_col(width: int, left_depth: int, right_depth: int, span: int) -> int:
    low = left_depth + 1
    high = width - right_depth - span - 1
    assert low <= high, "not enough horizontal room for singleton"
    return random.randint(low, high)


def _gen_horizontal_attempt() -> np.ndarray:
    repeated_color, singleton_a, singleton_b = random.sample(PALETTE, 3)
    left_depth = random.choice(DEPTH_OPTIONS)
    right_depth = random.choice(DEPTH_OPTIONS)
    want_odd = bool(random.getrandbits(1))
    width = _choose_gap_width(left_depth, right_depth, want_odd)
    height = random.choice(GRID_HEIGHTS)

    pair_left_span = random.choice(PAIR_SPAN_OPTIONS)
    pair_right_span = random.choice(PAIR_SPAN_OPTIONS)
    block_edge = random.choice(("top", "bottom"))
    block_depth = random.choice((2, 3))

    if block_edge == "top":
        pair_r0 = 1
        pair_r1 = max(pair_r0 + pair_left_span - 1, pair_r0 + pair_right_span - 1)
        if pair_r1 >= height - 2:
            raise AssertionError("grid too short for top-blocked pair")
        assert pair_r1 >= block_depth
        block_rows = (0, block_depth - 1)
    else:
        pair_r1 = height - 2
        pair_r0 = pair_r1 - max(pair_left_span, pair_right_span) + 1
        if pair_r0 <= 0:
            raise AssertionError("grid too short for bottom-blocked pair")
        assert pair_r0 <= height - block_depth - 1
        block_rows = (height - block_depth, height - 1)

    rep_count = random.choice((2, 3))
    if rep_count == 2:
        left_count, right_count = 1, 1
    else:
        left_count, right_count = random.choice(((2, 1), (1, 2)))

    grid = np.zeros((height, width), dtype=int)

    left_pair_h = pair_left_span
    right_pair_h = pair_right_span
    left_pair_r0 = pair_r0
    right_pair_r0 = pair_r0
    _fill_rect(grid, left_pair_r0, 0, left_pair_h, left_depth, repeated_color)
    _fill_rect(grid, right_pair_r0, width - right_depth, right_pair_h, right_depth, repeated_color)

    if left_count == 2:
        extra_h = random.choice(EXTRA_SPAN_OPTIONS)
        extra_r0, _extra_r1 = _pick_extra_interval(height, extra_h, (left_pair_r0, left_pair_r0 + left_pair_h - 1))
        _fill_rect(grid, extra_r0, 0, extra_h, left_depth, repeated_color)
    if right_count == 2:
        extra_h = random.choice(EXTRA_SPAN_OPTIONS)
        extra_r0, _extra_r1 = _pick_extra_interval(height, extra_h, (right_pair_r0, right_pair_r0 + right_pair_h - 1))
        _fill_rect(grid, extra_r0, width - right_depth, extra_h, right_depth, repeated_color)

    block_span = random.choice(SINGLETON_SPAN_OPTIONS)
    block_c0 = _singleton_col(width, left_depth, right_depth, block_span)
    if block_edge == "top":
        _fill_rect(grid, 0, block_c0, block_depth, block_span, singleton_a)
    else:
        _fill_rect(grid, height - block_depth, block_c0, block_depth, block_span, singleton_a)

    distract_depth = random.choice((2, 3))
    distract_span = random.choice(SINGLETON_SPAN_OPTIONS)
    distract_c0 = _singleton_col(width, left_depth, right_depth, distract_span)
    distract_edge = "bottom" if block_edge == "top" else "top"
    if distract_edge == "top":
        _fill_rect(grid, 0, distract_c0, distract_depth, distract_span, singleton_b)
    else:
        _fill_rect(grid, height - distract_depth, distract_c0, distract_depth, distract_span, singleton_b)

    overlap_rows = range(max(left_pair_r0, right_pair_r0), min(left_pair_r0 + left_pair_h - 1, right_pair_r0 + right_pair_h - 1) + 1)
    assert overlap_rows
    blocked_overlap = [
        r for r in overlap_rows if _intervals_overlap(r, r, block_rows[0], block_rows[1])
    ]
    clear_overlap = [
        r for r in overlap_rows if not _intervals_overlap(r, r, block_rows[0], block_rows[1])
    ]
    assert blocked_overlap, "blocker must interrupt at least one opposing row"
    assert clear_overlap, "at least one opposing row must remain unobstructed"

    out = main(grid)
    assert out.shape == grid.shape
    assert max(out.shape) <= 30
    assert np.any(out != grid), "expected a nontrivial transformation"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid = _gen_horizontal_attempt()
            if random.random() < 0.5:
                grid = grid.T.copy()
            out = main(grid)
            assert out.shape == grid.shape
            assert max(grid.shape) <= 30 and max(out.shape) <= 30
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid repeated-ray gap-splitting input")
