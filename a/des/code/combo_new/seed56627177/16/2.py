import random

import numpy as np


BG = 0
INDICATOR_COLOR = 3
LINE_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (14, 24)
GRID_WIDTH_RANGE = (14, 24)
SEG_LEN_RANGE = (2, 4)
GAP_RANGE = (2, 7)
PAIR_PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 400


def _find_indicator(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marked = [(r, c, int(grid[r, c])) for r, c in corners if int(grid[r, c]) != BG]
    assert len(marked) == 1, "expected exactly one non-background corner indicator"
    r, c, color = marked[0]
    assert int(np.count_nonzero(grid == color)) == 1, "indicator color must be unique"
    return r, c, color


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        components.append(comp)
    return components


def _segment_from_component(component: list[tuple[int, int]]) -> dict[str, int | str]:
    rows = sorted({r for r, _c in component})
    cols = sorted({c for _r, c in component})
    if len(rows) == 1:
        assert cols == list(range(cols[0], cols[-1] + 1))
        assert 2 <= len(cols) <= 4
        return {"orientation": "h", "row": rows[0], "c0": cols[0], "c1": cols[-1]}
    if len(cols) == 1:
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert 2 <= len(rows) <= 4
        return {"orientation": "v", "col": cols[0], "r0": rows[0], "r1": rows[-1]}
    raise AssertionError("each component must be a straight segment")


def _pair_segments(grid: np.ndarray, color: int) -> tuple[dict[str, int | str], dict[str, int | str]]:
    components = _components_for_color(grid, color)
    assert len(components) == 2, "each line color must form exactly two components"
    segments = [_segment_from_component(component) for component in components]
    horizontals = [segment for segment in segments if segment["orientation"] == "h"]
    verticals = [segment for segment in segments if segment["orientation"] == "v"]
    assert len(horizontals) == 1 and len(verticals) == 1
    return horizontals[0], verticals[0]


def _candidate_bends(
    horizontal: dict[str, int | str], vertical: dict[str, int | str]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    rh = int(horizontal["row"])
    c0 = int(horizontal["c0"])
    c1 = int(horizontal["c1"])
    cv = int(vertical["col"])
    r0 = int(vertical["r0"])
    r1 = int(vertical["r1"])

    assert not (c0 <= cv <= c1), "vertical line must miss the horizontal segment"
    assert not (r0 <= rh <= r1), "horizontal line must miss the vertical segment"

    ch = c0 if cv < c0 else c1
    rv = r0 if rh < r0 else r1
    endpoint_h = (rh, ch)
    endpoint_v = (rv, cv)
    bend_intersection = (rh, cv)
    bend_opposite = (rv, ch)
    return endpoint_h, endpoint_v, bend_intersection, bend_opposite


def _paint_line(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        strip = out[r0, lo : hi + 1]
        assert not np.any((strip != BG) & (strip != color)), "paths must stay disjoint"
        out[r0, lo : hi + 1] = color
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        strip = out[lo : hi + 1, c0]
        assert not np.any((strip != BG) & (strip != color)), "paths must stay disjoint"
        out[lo : hi + 1, c0] = color
        return
    raise AssertionError("connection legs must be axis-aligned")


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each colored horizontal/vertical segment pair with the elbow route selected by a corner marker.

    Input structure:
    - The background is black.
    - Exactly one green singleton sits in one corner as a global routing indicator.
    - Every other non-background color appears in exactly two straight segments: one horizontal and one vertical, each 2-4 cells long.
    - For each color, the horizontal row and vertical column cross at an empty cell, and the two facing endpoints define an empty rectangle with two possible bend positions.

    Rule:
    - Keep the original segments and the corner indicator unchanged.
    - For each color, find the horizontal endpoint facing the vertical segment and the vertical endpoint facing the horizontal segment.
    - Those two endpoints can be connected by two one-bend Manhattan paths: one bends at the row/column intersection of the extended lines, and the other bends at the opposite corner of the endpoint rectangle.
    - If the green marker is in the top-left or bottom-right corner, choose the path whose bend row is smaller; if it is in the top-right or bottom-left corner, choose the path whose bend column is smaller.
    - Paint the chosen path in that same color.

    Key insight:
    - The corner green pixel is a global elbow preference: it does not pair with any segment, but tells every color whether to take the upper bend or the leftward bend when both are valid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    indicator_r, indicator_c, _indicator_color = _find_indicator(g)
    prefer_smaller_row = (indicator_r, indicator_c) in {(0, 0), (h - 1, w - 1)}

    out = g.copy()
    line_colors: list[int] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        coords = np.argwhere(g == color)
        if coords.shape[0] == 1 and int(coords[0, 0]) == indicator_r and int(coords[0, 1]) == indicator_c:
            continue
        line_colors.append(color)

    assert line_colors, "expected at least one line-segment pair"
    for color in line_colors:
        horizontal, vertical = _pair_segments(g, color)
        endpoint_h, endpoint_v, bend_a, bend_b = _candidate_bends(horizontal, vertical)
        assert int(g[bend_a]) == BG and int(g[bend_b]) == BG

        if prefer_smaller_row:
            bend = bend_a if bend_a[0] < bend_b[0] else bend_b
        else:
            bend = bend_a if bend_a[1] < bend_b[1] else bend_b

        _paint_line(out, endpoint_h, bend, color)
        _paint_line(out, bend, endpoint_v, color)

    return out


def _pair_count_for_shape(height: int, width: int) -> int:
    max_pairs = min(4, max(2, height * width // 140 + 1))
    return random.randint(2, max_pairs)


def _build_pair_geometry(
    top: int,
    left: int,
    len_h: int,
    len_v: int,
    gap_x: int,
    gap_y: int,
    vertical_right: bool,
    vertical_below: bool,
) -> dict[str, object]:
    width = len_h + gap_x
    height = len_v + gap_y
    bottom = top + height - 1
    right = left + width - 1

    if vertical_right:
        cv = right
        h_c0 = left
        h_c1 = left + len_h - 1
    else:
        cv = left
        h_c0 = right - len_h + 1
        h_c1 = right

    if vertical_below:
        rh = top
        v_r0 = top + gap_y
        v_r1 = top + gap_y + len_v - 1
    else:
        rh = bottom
        v_r0 = top
        v_r1 = top + len_v - 1

    horizontal = [(rh, c) for c in range(h_c0, h_c1 + 1)]
    vertical = [(r, cv) for r in range(v_r0, v_r1 + 1)]
    return {
        "bbox": (top, left, bottom, right),
        "horizontal": horizontal,
        "vertical": vertical,
        "branch_differs": vertical_right == vertical_below,
    }


def _place_pair(
    height: int,
    width: int,
    reserved: np.ndarray,
    *,
    must_flip_branch: bool,
) -> dict[str, object] | None:
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        len_h = random.randint(*SEG_LEN_RANGE)
        len_v = random.randint(*SEG_LEN_RANGE)
        max_gap_x = min(GAP_RANGE[1], width - len_h - 2)
        max_gap_y = min(GAP_RANGE[1], height - len_v - 2)
        if max_gap_x < GAP_RANGE[0] or max_gap_y < GAP_RANGE[0]:
            continue

        gap_x = random.randint(GAP_RANGE[0], max_gap_x)
        gap_y = random.randint(GAP_RANGE[0], max_gap_y)
        vertical_right = random.choice([False, True])
        vertical_below = vertical_right if must_flip_branch else random.choice([False, True])

        box_w = len_h + gap_x
        box_h = len_v + gap_y
        max_top = height - box_h - 1
        max_left = width - box_w - 1
        if max_top < 1 or max_left < 1:
            continue

        top = random.randint(1, max_top)
        left = random.randint(1, max_left)
        if np.any(reserved[top - 1 : top + box_h + 1, left - 1 : left + box_w + 1]):
            continue

        return _build_pair_geometry(
            top, left, len_h, len_v, gap_x, gap_y, vertical_right, vertical_below
        )
    return None


def _opposite_branch_corner(height: int, width: int, prefer_smaller_row: bool) -> tuple[int, int]:
    return (0, width - 1) if prefer_smaller_row else (0, 0)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = _pair_count_for_shape(height, width)
    if pair_count > len(LINE_COLORS):
        return None

    indicator_pos = random.choice([(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)])
    prefer_smaller_row = indicator_pos in {(0, 0), (height - 1, width - 1)}
    colors = random.sample(LINE_COLORS, pair_count)

    grid = np.full((height, width), BG, dtype=int)
    grid[indicator_pos] = INDICATOR_COLOR
    reserved = np.zeros((height, width), dtype=bool)
    reserved[indicator_pos] = True

    pairs: list[dict[str, object]] = []
    for pair_idx in range(pair_count):
        pair = _place_pair(height, width, reserved, must_flip_branch=pair_idx == 0)
        if pair is None:
            return None
        top, left, bottom, right = pair["bbox"]
        reserved[top - 1 : bottom + 2, left - 1 : right + 2] = True
        pairs.append(pair)

    for color, pair in zip(colors, pairs):
        for r, c in pair["horizontal"]:
            grid[r, c] = color
        for r, c in pair["vertical"]:
            grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    alt_grid = grid.copy()
    alt_grid[indicator_pos] = BG
    alt_corner = _opposite_branch_corner(height, width, prefer_smaller_row)
    alt_grid[alt_corner] = INDICATOR_COLOR
    if np.array_equal(main(alt_grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
