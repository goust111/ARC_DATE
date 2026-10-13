from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, find_components, neighbors4
import random


def _orientation(
    coords: np.ndarray,
    contacts: list[tuple[int, int, int, int]],
) -> str:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == 1 or w == 1, "expected each gray source to be a one-cell-thick bar"

    if h == 1 and w > 1:
        return "horizontal"
    if w == 1 and h > 1:
        return "vertical"

    implied: set[str] = set()
    for tr, tc, sr, sc in contacts:
        if tr == sr:
            implied.add("horizontal")
        elif tc == sc:
            implied.add("vertical")
        else:
            raise AssertionError("gray source must touch orange by a 4-neighbor side")
    assert len(implied) == 1, "singleton gray source has ambiguous orientation"
    return implied.pop()


def _fill_rows(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    count: int,
    contact_rows: set[int],
) -> None:
    width = int(c1 - c0 + 1)
    height = int(r1 - r0 + 1)
    assert count % width == 0, "gray count must make complete orange rows"
    stripe_count = int(count // width)
    assert 0 < stripe_count <= height, "row fill must fit inside the orange rectangle"

    assert contact_rows in ({r0}, {r1}), "horizontal bars must touch exactly one extreme row"
    if contact_rows == {r0}:
        rows = list(range(r0, r0 + stripe_count))
    else:
        rows = list(range(r1 - stripe_count + 1, r1 + 1))

    out[rows, c0 : c1 + 1] = Color.GRAY


def _fill_cols(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    count: int,
    contact_cols: set[int],
) -> None:
    width = int(c1 - c0 + 1)
    height = int(r1 - r0 + 1)
    assert count % height == 0, "gray count must make complete orange columns"
    stripe_count = int(count // height)
    assert 0 < stripe_count <= width, "column fill must fit inside the orange rectangle"

    assert contact_cols in ({c0}, {c1}), "vertical bars must touch exactly one extreme column"
    if contact_cols == {c0}:
        cols = list(range(c0, c0 + stripe_count))
    else:
        cols = list(range(c1 - stripe_count + 1, c1 + 1))

    out[r0 : r1 + 1, cols] = Color.GRAY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray bar into its adjacent orange rectangle as full stripes from the contacted edge.

    Input structure
    - The scene uses black outside space, yellow fill, gray one-cell-thick bars, and solid orange rectangles.
    - Each gray bar 4-touches exactly one orange rectangle; some orange rectangles have no touching gray bar.
    - Gray bars touching the same orange rectangle all have the same orientation.

    Rule
    - Recolor every original gray cell to yellow.
    - For each orange rectangle touched by gray bars, count the gray cells assigned to that rectangle.
    - Horizontal gray bars repaint that many cells as complete orange rows starting from the contacted top or bottom row.
    - Vertical gray bars repaint that many cells as complete orange columns starting from the contacted left or right column.
    - Orange rectangles with no touching gray bar remain unchanged.

    Key insight
    - The gray pixels are not copied by shape; their count is transferred into the neighboring orange rectangle as edge-aligned row or column stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.YELLOW, Color.GRAY, Color.ORANGE}

    orange = find_components(g, bg=Color.YELLOW, colors=[Color.ORANGE], connectivity=4)
    gray = find_components(g, bg=Color.YELLOW, colors=[Color.GRAY], connectivity=4)
    assert orange and gray, "expected orange rectangles and gray source bars"

    orange_index: dict[tuple[int, int], int] = {}
    orange_bboxes: list[tuple[int, int, int, int]] = []
    for i, obj in enumerate(orange):
        r0, c0, r1, c1 = obj["bbox"]
        assert obj["area"] == bbox_h(obj["bbox"]) * bbox_w(obj["bbox"])
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == Color.ORANGE)
        orange_bboxes.append((int(r0), int(c0), int(r1), int(c1)))
        for r, c in obj["coords"]:
            orange_index[(int(r), int(c))] = i

    source_by_target: dict[int, list[tuple[dict, list[tuple[int, int, int, int]]]]] = {}
    for src in gray:
        touched: dict[int, list[tuple[int, int, int, int]]] = {}
        for sr_raw, sc_raw in src["coords"]:
            sr = int(sr_raw)
            sc = int(sc_raw)
            for tr, tc in neighbors4(sr, sc):
                target_i = orange_index.get((int(tr), int(tc)))
                if target_i is not None:
                    touched.setdefault(target_i, []).append((int(tr), int(tc), sr, sc))

        assert len(touched) == 1, "each gray bar must touch exactly one orange rectangle"
        target_i = next(iter(touched))
        source_by_target.setdefault(target_i, []).append((src, touched[target_i]))

    out = g.copy()
    out[g == Color.GRAY] = Color.YELLOW

    for target_i, sources in source_by_target.items():
        r0, c0, r1, c1 = orange_bboxes[target_i]
        orientations = {_orientation(src["coords"], contacts) for src, contacts in sources}
        assert len(orientations) == 1, "all gray bars for one orange rectangle must share orientation"
        count = sum(int(src["area"]) for src, _contacts in sources)
        contacts = [contact for _src, src_contacts in sources for contact in src_contacts]

        if orientations == {"horizontal"}:
            contact_rows = {tr for tr, _tc, _sr, _sc in contacts}
            assert contact_rows <= {r0, r1}
            _fill_rows(out, r0, c0, r1, c1, count, contact_rows)
        else:
            assert orientations == {"vertical"}
            contact_cols = {tc for _tr, tc, _sr, _sc in contacts}
            assert contact_cols <= {c0, c1}
            _fill_cols(out, r0, c0, r1, c1, count, contact_cols)

    return out

# --------------- generator (att1) ---------------

GRID_H_RANGE, GRID_W_RANGE = (10, 18), (14, 18)
CONTENT_H_RANGE, CONTENT_W_RANGE = (6, 13), (12, 15)
BORDER_RANGE = (0, 4)

EDGES, HORIZONTAL_EDGES = ("top", "bottom", "left", "right"), ("top", "bottom")
ORANGE_COUNT_CHOICES, ORANGE_COUNT_WEIGHTS = [1, 2, 3, 4], [1, 2, 3, 4]
ORANGE_H_RANGE, ORANGE_W_RANGE = (1, 5), (2, 6)
ORANGE_AREA_RANGE = (4, 12)
ORANGE_PLACE_RETRIES = 80

MAX_UNTOUCHED_ORANGES = 1
BARS_PER_ORANGE_RANGE, TOTAL_BARS_RANGE = (1, 2), (2, 6)
BAR_LENGTH_RANGE = (1, 7)
BAR_COUNT_RETRIES = 50

GENERATE_RETRIES = 300


def _place_orange(edge, cr0, cc0, cr1, cc1, occupied):
    """Try to place an orange rect flush against the given edge. Returns (or0,oc0,or1,oc1) or None."""
    for _ in range(ORANGE_PLACE_RETRIES):
        oh = random.randint(*ORANGE_H_RANGE)
        ow = random.randint(*ORANGE_W_RANGE)
        if not (ORANGE_AREA_RANGE[0] <= oh * ow <= ORANGE_AREA_RANGE[1]):
            continue
        if edge in HORIZONTAL_EDGES:
            or0 = cr0 if edge == 'top' else cr1 - oh + 1
            or1 = or0 + oh - 1
            lo, hi = cc0 + 1, cc1 - ow  # avoid touching perpendicular edges
        else:
            oc0 = cc0 if edge == 'left' else cc1 - ow + 1
            oc1 = oc0 + ow - 1
            lo, hi = cr0 + 1, cr1 - oh  # avoid touching perpendicular edges
        if hi < lo:
            continue
        v = random.randint(lo, hi)
        if edge in HORIZONTAL_EDGES:
            oc0, oc1 = v, v + ow - 1
        else:
            or0, or1 = v, v + oh - 1
        if or0 < cr0 or or1 > cr1 or oc0 < cc0 or oc1 > cc1:
            continue
        if occupied[or0:or1+1, oc0:oc1+1].any():
            continue
        return (or0, oc0, or1, oc1)
    return None


def _make_canvas() -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    content_h = random.randint(*CONTENT_H_RANGE)
    content_w = random.randint(*CONTENT_W_RANGE)
    border_top = random.randint(*BORDER_RANGE)
    border_bottom = random.randint(*BORDER_RANGE)
    border_left = random.randint(*BORDER_RANGE)
    border_right = random.randint(*BORDER_RANGE)
    grid_h = content_h + border_top + border_bottom
    grid_w = content_w + border_left + border_right
    if not (GRID_H_RANGE[0] <= grid_h <= GRID_H_RANGE[1] and GRID_W_RANGE[0] <= grid_w <= GRID_W_RANGE[1]):
        return None

    grid = np.full((grid_h, grid_w), Color.BLACK, dtype=int)
    cr0, cc0 = border_top, border_left
    cr1, cc1 = cr0 + content_h - 1, cc0 + content_w - 1
    grid[cr0:cr1+1, cc0:cc1+1] = Color.YELLOW
    return grid, (cr0, cc0, cr1, cc1)


def _place_oranges(content_box, occupied) -> list[tuple[str, int, int, int, int]] | None:
    cr0, cc0, cr1, cc1 = content_box
    num_orange = random.choices(ORANGE_COUNT_CHOICES, weights=ORANGE_COUNT_WEIGHTS)[0]
    edges = random.sample(EDGES, num_orange)
    orange_rects = []
    for edge in edges:
        result = _place_orange(edge, cr0, cc0, cr1, cc1, occupied)
        if result is None:
            return None
        or0, oc0, or1, oc1 = result
        occupied[or0:or1+1, oc0:oc1+1] = True
        orange_rects.append((edge, or0, oc0, or1, oc1))
    return orange_rects


def _choose_touched_oranges(num_orange):
    max_untouched = min(MAX_UNTOUCHED_ORANGES, num_orange - 1)
    num_untouched = random.randint(0, max_untouched) if num_orange > 1 else 0
    indices = list(range(num_orange))
    random.shuffle(indices)
    untouched_set = set(indices[:num_untouched])
    return [i for i in range(num_orange) if i not in untouched_set]


def _choose_bars_per_orange(touched_list):
    for _ in range(BAR_COUNT_RETRIES):
        bpo = [random.randint(*BARS_PER_ORANGE_RANGE) for _ in touched_list]
        if TOTAL_BARS_RANGE[0] <= sum(bpo) <= TOTAL_BARS_RANGE[1]:
            return bpo
    return None


def _split_bar_lengths(total_gray, n_bars):
    min_len, max_len = BAR_LENGTH_RANGE
    if n_bars == 2 and total_gray >= 2:
        lo_b1 = max(min_len, total_gray - max_len)
        hi_b1 = min(total_gray - min_len, max_len)
        if lo_b1 > hi_b1:
            return None
        b1 = random.randint(lo_b1, hi_b1)
        return [b1, total_gray - b1]
    if total_gray > max_len:
        return None
    return [total_gray]


def _place_bars(axis, fixed, span0, span1, lo_bound, hi_bound, bar_lengths, occupied):
    placed_bars = []
    for bl in bar_lengths:
        sides = []
        if span0 - lo_bound >= bl:
            sides.append(0)
        if hi_bound - span1 >= bl:
            sides.append(1)
        random.shuffle(sides)
        placed = False
        for s in sides:
            b0 = (span0 - bl) if s == 0 else (span1 + 1)
            b1 = b0 + bl - 1
            cells = occupied[fixed, b0:b1+1] if axis == 'h' else occupied[b0:b1+1, fixed]
            if b0 < lo_bound or b1 > hi_bound or cells.any():
                continue
            if axis == 'h':
                occupied[fixed, b0:b1+1] = True
                placed_bars.append(('h', fixed, b0, b1))
            else:
                occupied[b0:b1+1, fixed] = True
                placed_bars.append(('v', b0, b1, fixed))
            placed = True
            break
        if not placed:
            return None
    return placed_bars


def _plan_bars(orange_rects, touched_list, bpo, content_box, occupied):
    """Sample gray bars and return oracle metadata for expected output."""
    cr0, cc0, cr1, cc1 = content_box
    bars_info = []
    for idx, oi in enumerate(touched_list):
        edge, or0, oc0, or1, oc1 = orange_rects[oi]
        oh, ow = or1 - or0 + 1, oc1 - oc0 + 1
        n_bars = bpo[idx]

        if edge in HORIZONTAL_EDGES:
            max_stripes, divisor = oh, ow
        else:
            max_stripes, divisor = ow, oh

        stripe_count = random.randint(1, max_stripes)
        total_gray = stripe_count * divisor
        bar_lengths = _split_bar_lengths(total_gray, n_bars)
        if bar_lengths is None:
            return None

        if edge == 'top':
            bar_rc = or0
        elif edge == 'bottom':
            bar_rc = or1
        elif edge == 'left':
            bar_rc = oc0
        else:
            bar_rc = oc1
        if edge in HORIZONTAL_EDGES:
            placed_bars = _place_bars('h', bar_rc, oc0, oc1, cc0, cc1, bar_lengths, occupied)
        else:
            placed_bars = _place_bars('v', bar_rc, or0, or1, cr0, cr1, bar_lengths, occupied)
        if placed_bars is None:
            return None

        bars_info.append((oi, placed_bars, edge, stripe_count))
    return bars_info


def _render_input_grid(grid, orange_rects, bars_info):
    for _, or0, oc0, or1, oc1 in orange_rects:
        grid[or0:or1+1, oc0:oc1+1] = Color.ORANGE
    for _, bars, _, _ in bars_info:
        for bar in bars:
            if bar[0] == 'h':
                grid[bar[1], bar[2]:bar[3]+1] = Color.GRAY
            else:
                grid[bar[1]:bar[2]+1, bar[3]] = Color.GRAY


def _expected_output(grid, orange_rects, bars_info):
    expected = grid.copy()
    expected[grid == Color.GRAY] = Color.YELLOW
    for oi, _, contact_side, stripe_count in bars_info:
        _, or0, oc0, or1, oc1 = orange_rects[oi]
        if contact_side == 'top':
            expected[or0:or0+stripe_count, oc0:oc1+1] = Color.GRAY
        elif contact_side == 'bottom':
            expected[or1-stripe_count+1:or1+1, oc0:oc1+1] = Color.GRAY
        elif contact_side == 'left':
            expected[or0:or1+1, oc0:oc0+stripe_count] = Color.GRAY
        else:
            expected[or0:or1+1, oc1-stripe_count+1:oc1+1] = Color.GRAY
    return expected


def _gen_attempt() -> np.ndarray | None:
    canvas = _make_canvas()
    if canvas is None:
        return None
    grid, content_box = canvas
    occupied = np.zeros(grid.shape, dtype=bool)

    orange_rects = _place_oranges(content_box, occupied)
    if orange_rects is None:
        return None

    touched_list = _choose_touched_oranges(len(orange_rects))
    bpo = _choose_bars_per_orange(touched_list)
    if bpo is None:
        return None

    bars_info = _plan_bars(orange_rects, touched_list, bpo, content_box, occupied)
    if bars_info is None:
        return None

    _render_input_grid(grid, orange_rects, bars_info)
    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, _expected_output(grid, orange_rects, bars_info)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATE_RETRIES} attempts")
