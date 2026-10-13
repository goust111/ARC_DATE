from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int, int]  # (color, r0, c0, r1, c1), inclusive bbox


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    _color, r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _overlap(a: Rect, b: Rect) -> tuple[int, int, int, int] | None:
    _ca, ar0, ac0, ar1, ac1 = a
    _cb, br0, bc0, br1, bc1 = b
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar1, br1)
    c1 = min(ac1, bc1)
    if r0 <= r1 and c0 <= c1:
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _rectangles_from_visible_edges(g: np.ndarray) -> list[Rect]:
    h, w = g.shape
    grid_area = h * w
    rects: list[Rect] = []

    for raw_color in np.unique(g):
        color = int(raw_color)
        coords = np.argwhere(g == color)
        if coords.shape[0] < 4:
            continue

        r0, c0, r1, c1 = _bbox_coords(coords)
        rh = r1 - r0 + 1
        rw = c1 - c0 + 1
        area = rh * rw
        if rh < 3 or rw < 3:
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if area > int(0.55 * grid_area):
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]
        touches_top = bool(np.any(rows == r0))
        touches_bottom = bool(np.any(rows == r1))
        touches_left = bool(np.any(cols == c0))
        touches_right = bool(np.any(cols == c1))
        if not (touches_top and touches_bottom and touches_left and touches_right):
            continue

        rects.append((color, r0, c0, r1, c1))

    assert len(rects) >= 3, "expected several visible-edged rectangle colors"
    return rects


def _coverage_counts(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for _color, r0, c0, r1, c1 in rects:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _find_bottom_rect(g: np.ndarray, rects: list[Rect], cover: np.ndarray) -> Rect:
    candidates: list[Rect] = []
    for rect in rects:
        color, r0, c0, r1, c1 = rect
        local_cover = cover[r0 : r1 + 1, c0 : c1 + 1]
        local_grid = g[r0 : r1 + 1, c0 : c1 + 1]
        overlap_mask = local_cover >= 2
        assert np.any(overlap_mask), "each rectangle must overlap another rectangle"
        own_visible_in_overlap = bool(np.any((local_grid == color) & overlap_mask))
        if not own_visible_in_overlap:
            candidates.append(rect)

    assert len(candidates) == 1, "expected a unique rectangle that loses every overlap"
    return candidates[0]


def _period_values(
    g: np.ndarray, blocked: np.ndarray, row_period: int, col_period: int
) -> dict[tuple[int, int], int] | None:
    values: dict[tuple[int, int], int] = {}
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            key = (r % row_period, c % col_period)
            value = int(g[r, c])
            if key in values and values[key] != value:
                return None
            values[key] = value
    return values


def _period_fill_for_coords(
    g: np.ndarray, blocked: np.ndarray, coords: np.ndarray
) -> tuple[int, ...]:
    h, w = g.shape
    candidates: list[tuple[int, int, dict[tuple[int, int], int], tuple[int, ...]]] = []
    target_coords = [(int(r), int(c)) for r, c in coords]

    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            values = _period_values(g, blocked, row_period, col_period)
            if values is None:
                continue
            keys = [(r % row_period, c % col_period) for r, c in target_coords]
            if not all(key in values for key in keys):
                continue
            fill = tuple(int(values[key]) for key in keys)
            candidates.append((row_period, col_period, values, fill))

    assert candidates, "visible background must determine the hidden periodic cells"
    min_area = min(rp * cp for rp, cp, _values, _fill in candidates)
    fills = [fill for rp, cp, _values, fill in candidates if rp * cp == min_area]
    first = fills[0]
    assert all(fill == first for fill in fills), "minimal periodic completions must agree"
    return first


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the unique bottom rectangle from an overlaid rectangle stack and restore the periodic background under its exposed area.

    Input structure:
    - A small multicolor row/column periodic tile fills the whole grid underneath.
    - Several differently colored solid rectangles are overlaid on top; overlaps show only the higher rectangle color.
    - Every rectangle still has visible pixels on all four sides, so its full bbox is recoverable from its visible color.

    Rule:
    - Reconstruct every rectangle bbox from the visible pixels of its color.
    - In rectangle-overlap cells, the visible rectangle color is above the other covering rectangles; the bottom rectangle is the unique one whose own color is never visible in any overlap.
    - Mask all reconstructed rectangles, infer the smallest periodic tile from the remaining visible background, and repaint only the bottom rectangle's non-overlapped cells with that tile while preserving cells covered by higher rectangles.

    Key insight:
    - The target is not selected by size or missing area: it is the only rectangle that loses every visible overlap, and its erased cells must be recovered by extending the hidden periodic background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _rectangles_from_visible_edges(g)
    cover = _coverage_counts(g.shape, rects)
    bottom = _find_bottom_rect(g, rects, cover)

    _color, r0, c0, r1, c1 = bottom
    bottom_mask = np.zeros(g.shape, dtype=bool)
    bottom_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    exclusive = bottom_mask & (cover == 1)
    coords = np.argwhere(exclusive)
    assert coords.shape[0] > 0, "bottom rectangle must have an exclusive visible area"

    blocked = cover > 0
    fill = _period_fill_for_coords(g, blocked, coords)

    out = g.copy()
    for (r, c), value in zip(coords, fill):
        out[int(r), int(c)] = int(value)
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 800
LAYOUT_ATTEMPTS = 300
GRID_SIZE_RANGE = (22, 30)
RECT_COUNT_CHOICES = (4, 5)
PERIOD_CHOICES = (3, 4)


def _tile_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].astype(int, copy=True)


def _gen_tile(period: int, colors: list[int]) -> np.ndarray:
    row_step = random.choice([x for x in range(1, period) if np.gcd(x, period) == 1])
    col_step = random.choice([x for x in range(1, period) if np.gcd(x, period) == 1])
    offset = random.randrange(period)
    tile = np.empty((period, period), dtype=int)
    for r in range(period):
        for c in range(period):
            tile[r, c] = colors[(offset + row_step * r + col_step * c) % period]
    return tile


def _make_rect(color: int, r0: int, c0: int, height: int, width: int) -> Rect:
    return (int(color), int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))


def _point_in_rect(r: int, c: int, rect: Rect) -> bool:
    _color, r0, c0, r1, c1 = rect
    return bool(r0 <= r <= r1 and c0 <= c <= c1)


def _side_has_uncovered_cell(cells: list[tuple[int, int]], higher: list[Rect]) -> bool:
    for r, c in cells:
        if not any(_point_in_rect(r, c, rect) for rect in higher):
            return True
    return False


def _all_edges_uncovered_by_higher(rects_bottom_to_top: list[Rect]) -> bool:
    for idx, rect in enumerate(rects_bottom_to_top):
        _color, r0, c0, r1, c1 = rect
        higher = rects_bottom_to_top[idx + 1 :]
        top = [(r0, c) for c in range(c0, c1 + 1)]
        bottom = [(r1, c) for c in range(c0, c1 + 1)]
        left = [(r, c0) for r in range(r0, r1 + 1)]
        right = [(r, c1) for r in range(r0, r1 + 1)]
        if not (
            _side_has_uncovered_cell(top, higher)
            and _side_has_uncovered_cell(bottom, higher)
            and _side_has_uncovered_cell(left, higher)
            and _side_has_uncovered_cell(right, higher)
        ):
            return False
    return True


def _sample_bottom_rect(height: int, width: int, color: int) -> Rect | None:
    for _ in range(80):
        rh = random.randint(6, min(9, height - 10))
        rw = random.randint(6, min(9, width - 10))
        r0 = random.randint(3, height - rh - 4)
        c0 = random.randint(3, width - rw - 4)
        return _make_rect(color, r0, c0, rh, rw)
    return None


def _sample_overlapping_rect(
    shape: tuple[int, int],
    color: int,
    target: Rect,
    height: int,
    width: int,
    *,
    prefer_large_overlap: bool = False,
) -> Rect | None:
    rows, cols = shape
    _tc, tr0, tc0, tr1, tc1 = target
    th = tr1 - tr0 + 1
    tw = tc1 - tc0 + 1

    for _ in range(120):
        max_oh = min(height, th, 6 if not prefer_large_overlap else min(height, th))
        max_ow = min(width, tw, 6 if not prefer_large_overlap else min(width, tw))
        if max_oh < 2 or max_ow < 2:
            continue
        if prefer_large_overlap:
            oh = random.randint(max(2, max_oh - 2), max_oh)
            ow = random.randint(max(2, max_ow - 2), max_ow)
        else:
            oh = random.randint(2, max_oh)
            ow = random.randint(2, max_ow)

        if tr1 - tr0 + 1 >= oh + 2:
            overlap_r0 = random.randint(tr0 + 1, tr1 - oh)
        else:
            overlap_r0 = random.randint(tr0, tr1 - oh + 1)
        if tc1 - tc0 + 1 >= ow + 2:
            overlap_c0 = random.randint(tc0 + 1, tc1 - ow)
        else:
            overlap_c0 = random.randint(tc0, tc1 - ow + 1)

        if height >= oh + 2:
            inside_r = random.randint(1, height - oh - 1)
        else:
            inside_r = random.randint(0, height - oh)
        if width >= ow + 2:
            inside_c = random.randint(1, width - ow - 1)
        else:
            inside_c = random.randint(0, width - ow)
        r0 = overlap_r0 - inside_r
        c0 = overlap_c0 - inside_c
        rect = _make_rect(color, r0, c0, height, width)
        _color, rr0, cc0, rr1, cc1 = rect
        if rr0 < 1 or cc0 < 1 or rr1 >= rows - 1 or cc1 >= cols - 1:
            continue
        inter = _overlap(rect, target)
        if inter is None:
            continue
        ir0, ic0, ir1, ic1 = inter
        inter_area = (ir1 - ir0 + 1) * (ic1 - ic0 + 1)
        area = _rect_area(rect)
        target_area = _rect_area(target)
        if inter_area >= area or inter_area >= target_area:
            continue
        if inter_area > int(0.80 * area):
            continue
        return rect
    return None


def _gen_rectangles(shape: tuple[int, int], colors: list[int]) -> list[Rect] | None:
    rows, cols = shape
    n = len(colors)

    for _ in range(LAYOUT_ATTEMPTS):
        bh = random.randint(7, 8)
        bw = random.randint(7, 8)
        ah = random.randint(7, 8)
        aw = random.randint(7, 8)
        ch = random.randint(max(8, bh + 1), 9)
        cw = random.randint(max(8, bw + 1), 9)
        dh = random.randint(5, 6)
        dw = random.randint(5, 6)

        rect_specs: list[tuple[int, int, int, int]] = []
        rect_specs.append((0, 0, bh, bw))

        a_r = random.randint(1, 2)
        a_c = bw - random.randint(2, 3)
        rect_specs.append((a_r, a_c, ah, aw))

        c_r = a_r + random.randint(1, 2)
        c_c = a_c + random.randint(1, 2)
        rect_specs.append((c_r, c_c, ch, cw))

        d_r = c_r + ch - random.randint(3, 4)
        d_c = c_c + cw - random.randint(3, 4)
        rect_specs.append((d_r, d_c, dh, dw))

        if n == 5:
            eh = random.randint(4, 5)
            ew = random.randint(4, 5)
            e_r = d_r + random.randint(1, 2)
            e_c = d_c + dw - random.randint(2, 3)
            rect_specs.append((e_r, e_c, eh, ew))

        max_r = max(r + h - 1 for r, _c, h, _w in rect_specs)
        max_c = max(c + w - 1 for _r, c, _h, w in rect_specs)
        if max_r + 4 >= rows or max_c + 4 >= cols:
            continue

        shift_r = random.randint(2, rows - max_r - 3)
        shift_c = random.randint(2, cols - max_c - 3)
        rects = [
            _make_rect(color, shift_r + r, shift_c + c, height, width)
            for color, (r, c, height, width) in zip(colors, rect_specs)
        ]
        if len(rects) != n or not _all_edges_uncovered_by_higher(rects):
            continue

        cover = _coverage_counts(shape, rects)
        coverage = float(np.count_nonzero(cover > 0)) / float(rows * cols)
        overlap = int(np.count_nonzero(cover >= 2))
        if not (0.12 <= coverage <= 0.42 and overlap >= 8):
            continue
        return rects
    return None


def _render(base: np.ndarray, rects_bottom_to_top: list[Rect]) -> np.ndarray:
    out = base.copy()
    for color, r0, c0, r1, c1 in rects_bottom_to_top:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _edge_visible(g: np.ndarray, rect: Rect) -> bool:
    color, r0, c0, r1, c1 = rect
    return bool(
        np.any(g[r0, c0 : c1 + 1] == color)
        and np.any(g[r1, c0 : c1 + 1] == color)
        and np.any(g[r0 : r1 + 1, c0] == color)
        and np.any(g[r0 : r1 + 1, c1] == color)
    )


def _all_period_residues_visible(blocked: np.ndarray, period: int) -> bool:
    seen = np.zeros((period, period), dtype=bool)
    for r in range(blocked.shape[0]):
        for c in range(blocked.shape[1]):
            if not blocked[r, c]:
                seen[r % period, c % period] = True
    return bool(seen.all())


def _expected_output(base: np.ndarray, rendered: np.ndarray, rects: list[Rect]) -> np.ndarray:
    cover = _coverage_counts(rendered.shape, rects)
    _color, r0, c0, r1, c1 = rects[0]
    bottom_mask = np.zeros(rendered.shape, dtype=bool)
    bottom_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    exclusive = bottom_mask & (cover == 1)
    out = rendered.copy()
    out[exclusive] = base[exclusive]
    return out


def _semantic_checks(base: np.ndarray, rendered: np.ndarray, rects: list[Rect], period: int) -> bool:
    shape = rendered.shape
    cover = _coverage_counts(shape, rects)
    blocked = cover > 0

    try:
        parsed = set(_rectangles_from_visible_edges(rendered))
    except AssertionError:
        return False
    if parsed != set(rects):
        return False
    if not _all_period_residues_visible(blocked, period):
        return False
    if not all(_edge_visible(rendered, rect) for rect in rects):
        return False

    top_visible_in_overlap: list[bool] = []
    for rect in rects:
        color, r0, c0, r1, c1 = rect
        local_cover = cover[r0 : r1 + 1, c0 : c1 + 1]
        local_grid = rendered[r0 : r1 + 1, c0 : c1 + 1]
        overlap_mask = local_cover >= 2
        if not np.any(overlap_mask):
            return False
        top_visible_in_overlap.append(bool(np.any((local_grid == color) & overlap_mask)))

    bottom_candidates = [idx for idx, wins in enumerate(top_visible_in_overlap) if not wins]
    if bottom_candidates != [0]:
        return False
    if not all(top_visible_in_overlap[1:]):
        return False

    hidden_areas = []
    visible_areas = []
    full_areas = []
    for rect in rects:
        color, r0, c0, r1, c1 = rect
        area = _rect_area(rect)
        visible = int(np.count_nonzero(rendered[r0 : r1 + 1, c0 : c1 + 1] == color))
        full_areas.append(area)
        visible_areas.append(visible)
        hidden_areas.append(area - visible)

    if not (min(hidden_areas) < hidden_areas[0] < max(hidden_areas)):
        return False
    if not (min(full_areas) < full_areas[0] < max(full_areas)):
        return False
    if visible_areas[0] == max(visible_areas):
        return False

    bottom = rects[0]
    _color, r0, c0, r1, c1 = bottom
    bottom_mask = np.zeros(shape, dtype=bool)
    bottom_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    exclusive = bottom_mask & (cover == 1)
    if int(np.count_nonzero(exclusive)) < 8:
        return False
    if len(set(map(int, base[exclusive].ravel()))) < 2:
        return False

    expected = _expected_output(base, rendered, rects)
    if np.array_equal(expected, rendered):
        return False
    try:
        actual = main(rendered)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    period = random.choice([p for p in PERIOD_CHOICES if p + rect_count <= 10])
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)

    palette = random.sample(range(10), period + rect_count)
    tile_colors = palette[:period]
    rect_colors = palette[period:]
    random.shuffle(rect_colors)

    tile = _gen_tile(period, tile_colors)
    base = _tile_grid(tile, height, width)
    rects = _gen_rectangles((height, width), rect_colors)
    if rects is None:
        return None

    rendered = _render(base, rects)
    if not _semantic_checks(base, rendered, rects, period):
        return None
    return rendered


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bottom-rectangle periodic restoration input")
