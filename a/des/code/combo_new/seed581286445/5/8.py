import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
BRUSH_COLORS = (1, 2, 3, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 30
INTERIOR_MIN = 6
INTERIOR_MAX = 12
BRUSH_COUNT_MIN = 2
BRUSH_COUNT_MAX = 5
OUTER_MARGIN_MIN = 0
OUTER_MARGIN_MAX = 1
FRAME_TOP_LEFT_MIN = 4
BRUSH_LEN_MIN = 2
BRUSH_LEN_MAX = 6

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _line_segment_info(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    if len(coords) < 2:
        return None
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(c for _r, c in coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            return ("h", (row, ordered[0]), (row, ordered[-1]))
    if len(cols) == 1:
        col = cols[0]
        ordered = sorted(r for r, _c in coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            return ("v", (ordered[0], col), (ordered[-1], col))
    return None


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey canvas frame"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    assert int(np.count_nonzero(grid == GREY)) == expected, "grey pixels must form one rectangle border"
    assert np.all(grid[r0, c0 : c1 + 1] == GREY)
    assert np.all(grid[r1, c0 : c1 + 1] == GREY)
    assert np.all(grid[r0 : r1 + 1, c0] == GREY)
    assert np.all(grid[r0 : r1 + 1, c1] == GREY)
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY)
    return int(r0), int(c0), int(r1), int(c1)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    counts = Counter(map(int, g.flat))
    assert counts[BLACK] == max(counts.values()), "black must be the dominant background"

    fr0, fc0, fr1, fc1 = _frame_bbox(g)
    ir0, ic0 = fr0 + 1, fc0 + 1
    ir1, ic1 = fr1 - 1, fc1 - 1

    interior_coords = [
        (int(r), int(c))
        for r in range(ir0, ir1 + 1)
        for c in range(ic0, ic1 + 1)
        if int(g[r, c]) not in (BLACK, GREY)
    ]
    assert interior_coords, "expected colored anchors inside the canvas"
    assert not any(int(g[r, c]) == YELLOW for r, c in interior_coords), "yellow markers must stay outside the canvas"

    anchor_rows = {r for r, _c in interior_coords}
    anchor_cols = {c for _r, c in interior_coords}
    if len(anchor_rows) == 1:
        assert next(iter(anchor_rows)) == ir0, "top-edge anchors must sit on the first interior row"
    elif len(anchor_cols) == 1:
        assert next(iter(anchor_cols)) == ic0, "left-edge anchors must sit on the first interior column"
    else:
        raise AssertionError("anchors must lie on one canvas edge")

    anchor_by_color: dict[int, tuple[int, int]] = {}
    for r, c in interior_coords:
        color = int(g[r, c])
        assert color not in (BLACK, YELLOW, GREY)
        assert color not in anchor_by_color
        anchor_by_color[color] = (r, c)

    yellow_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}
    assert yellow_cells, "expected yellow brush-tip markers"
    used_yellow: set[tuple[int, int]] = set()

    out = g.copy()
    claimed: dict[tuple[int, int], int] = {}

    for color, anchor in anchor_by_color.items():
        all_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        brush_cells = [cell for cell in all_cells if cell != anchor]
        assert BRUSH_LEN_MIN <= len(brush_cells) <= BRUSH_LEN_MAX
        assert all(not (fr0 <= r <= fr1 and fc0 <= c <= fc1) for r, c in brush_cells), "brush must stay outside the canvas"

        info = _line_segment_info(brush_cells)
        assert info is not None, "each brush must be one straight 4-connected segment"
        _orient, end_a, end_b = info

        marker_hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for end_r, end_c in (end_a, end_b):
            for nr, nc in _neighbors4(end_r, end_c):
                if (nr, nc) in yellow_cells:
                    marker_hits.append(((nr, nc), (nr - end_r, nc - end_c)))

        assert len(marker_hits) == 1, "expected one yellow marker adjacent to one brush endpoint"
        marker_cell, direction = marker_hits[0]
        assert marker_cell not in used_yellow, "yellow markers must pair unambiguously with brushes"
        used_yellow.add(marker_cell)

        ar, ac = anchor
        dr, dc = direction
        for step in range(len(brush_cells)):
            rr = ar + dr * step
            cc = ac + dc * step
            assert ir0 <= rr <= ir1 and ic0 <= cc <= ic1, "stroke must remain inside the canvas"
            if step > 0:
                assert int(g[rr, cc]) == BLACK, "canvas interior must start empty except for anchors"
            if (rr, cc) in claimed:
                assert claimed[(rr, cc)] == color, "strokes must not overlap across colors"
            claimed[(rr, cc)] = color
            out[rr, cc] = color

    assert used_yellow == yellow_cells, "every yellow marker must belong to exactly one brush"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each outside brush with its same-color canvas anchor, then draw the brush-sized stroke inside the canvas.

    Input structure:
    - The background is Black.
    - A one-pixel-thick Grey rectangular frame encloses a blank canvas near the lower-right.
    - Inside the frame, single-pixel colored anchors lie on exactly one interior edge
      (either the top edge or the left edge), with one unique anchor per brush color.
    - Outside the frame, each brush is a monochrome horizontal or vertical line segment,
      and one adjacent Yellow pixel sits next to one brush endpoint.

    Rule:
    - Match each brush to the anchor of the same color.
    - Read the stroke direction from the Yellow marker's offset relative to the marked
      brush endpoint.
    - Starting at the anchor, draw a straight line in that direction inside the canvas.
    - The stroke length equals the brush's pixel count, and all original brushes,
      anchors, markers, and the Grey frame remain unchanged.

    Key insight:
    - Color chooses the anchor, the Yellow offset chooses the drawing direction, and
      the brush's own size chooses the stroke length; all three cues must be combined.
    """
    return _solve(np.asarray(grid, dtype=int))


def _stroke_variants(length: int, direction: tuple[int, int]) -> list[tuple[list[tuple[int, int]], tuple[int, int], int, int]]:
    variants: list[tuple[list[tuple[int, int]], tuple[int, int], int, int]] = []

    def add_variant(brush: list[tuple[int, int]], marker: tuple[int, int]) -> None:
        all_cells = brush + [marker]
        min_r = min(r for r, _c in all_cells)
        min_c = min(c for _r, c in all_cells)
        shifted_brush = [(r - min_r, c - min_c) for r, c in brush]
        shifted_marker = (marker[0] - min_r, marker[1] - min_c)
        height = max(r for r, _c in shifted_brush + [shifted_marker]) + 1
        width = max(c for _r, c in shifted_brush + [shifted_marker]) + 1
        key = (tuple(sorted(shifted_brush)), shifted_marker)
        if key not in seen:
            seen.add(key)
            variants.append((shifted_brush, shifted_marker, height, width))

    seen: set[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = set()

    horizontal = [(0, c) for c in range(length)]
    if direction == DIR_LEFT:
        add_variant(horizontal, (0, -1))
    elif direction == DIR_RIGHT:
        add_variant(horizontal, (0, length))
    elif direction == DIR_UP:
        add_variant(horizontal, (-1, 0))
        add_variant(horizontal, (-1, length - 1))
    elif direction == DIR_DOWN:
        add_variant(horizontal, (1, 0))
        add_variant(horizontal, (1, length - 1))

    vertical = [(r, 0) for r in range(length)]
    if direction == DIR_UP:
        add_variant(vertical, (-1, 0))
    elif direction == DIR_DOWN:
        add_variant(vertical, (length, 0))
    elif direction == DIR_LEFT:
        add_variant(vertical, (0, -1))
        add_variant(vertical, (length - 1, -1))
    elif direction == DIR_RIGHT:
        add_variant(vertical, (0, 1))
        add_variant(vertical, (length - 1, 1))

    return variants


def _can_place(
    layout: np.ndarray,
    forbidden: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    H, W = layout.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if forbidden[r, c]:
            return False
        if int(layout[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr in range(max(0, r - 1), min(H, r + 2)):
            for nc in range(max(0, c - 1), min(W, c + 2)):
                if (nr, nc) in cell_set:
                    continue
                if int(layout[nr, nc]) != BLACK:
                    return False
    return True


def _paint_brush(
    grid: np.ndarray,
    brush_cells: list[tuple[int, int]],
    marker_cell: tuple[int, int],
    color: int,
) -> None:
    for r, c in brush_cells:
        grid[r, c] = color
    grid[marker_cell] = YELLOW


def _place_brushes(
    inp: np.ndarray,
    out: np.ndarray,
    forbidden: np.ndarray,
    brush_specs: list[dict[str, object]],
) -> bool:
    layout = inp.copy()
    order = sorted(brush_specs, key=lambda spec: int(spec["length"]), reverse=True)
    random.shuffle(order)
    H, W = inp.shape

    for spec in order:
        color = int(spec["color"])
        length = int(spec["length"])
        direction = spec["direction"]
        variants = _stroke_variants(length, direction)
        random.shuffle(variants)
        placed = False

        for brush_rel, marker_rel, ph, pw in variants:
            candidates: list[tuple[int, int]] = []
            for r0 in range(H - ph + 1):
                for c0 in range(W - pw + 1):
                    brush_abs = [(r0 + r, c0 + c) for r, c in brush_rel]
                    marker_abs = (r0 + marker_rel[0], c0 + marker_rel[1])
                    all_abs = brush_abs + [marker_abs]
                    if _can_place(layout, forbidden, all_abs):
                        candidates.append((r0, c0))
            if not candidates:
                continue

            r0, c0 = random.choice(candidates)
            brush_abs = [(r0 + r, c0 + c) for r, c in brush_rel]
            marker_abs = (r0 + marker_rel[0], c0 + marker_rel[1])
            _paint_brush(layout, brush_abs, marker_abs, color)
            _paint_brush(inp, brush_abs, marker_abs, color)
            _paint_brush(out, brush_abs, marker_abs, color)
            placed = True
            break

        if not placed:
            return False
    return True


def _anchor_positions(span: int, n: int) -> list[int] | None:
    if n == 1:
        if span < 3:
            return None
        return [span // 2]

    max_step = (span - 3) // (n - 1)
    if max_step < 2:
        return None
    step = random.randint(2, max_step)
    max_start = (span - 2) - step * (n - 1)
    if max_start < 1:
        return None
    start = random.randint(1, max_start)
    return [start + step * i for i in range(n)]


def _stroke_length(edge: str, direction: tuple[int, int], pos: int, ih: int, iw: int) -> int:
    if edge == "top":
        if direction == DIR_LEFT:
            max_len = pos + 1
        elif direction == DIR_RIGHT:
            max_len = iw - pos
        else:
            max_len = ih
    else:
        if direction == DIR_UP:
            max_len = pos + 1
        elif direction == DIR_DOWN:
            max_len = ih - pos
        else:
            max_len = iw
    assert max_len >= BRUSH_LEN_MIN
    return random.randint(BRUSH_LEN_MIN, min(BRUSH_LEN_MAX, max_len))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    edge = random.choice(["top", "left"])
    n_brushes = random.randint(BRUSH_COUNT_MIN, BRUSH_COUNT_MAX)

    bottom_margin = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)
    right_margin = random.randint(OUTER_MARGIN_MIN, OUTER_MARGIN_MAX)

    min_ih = max(INTERIOR_MIN, 2 * n_brushes + 1) if edge == "left" else INTERIOR_MIN
    min_iw = max(INTERIOR_MIN, 2 * n_brushes + 1) if edge == "top" else INTERIOR_MIN

    max_ih = min(INTERIOR_MAX, H - FRAME_TOP_LEFT_MIN - bottom_margin - 2)
    max_iw = min(INTERIOR_MAX, W - FRAME_TOP_LEFT_MIN - right_margin - 2)
    if max_ih < min_ih or max_iw < min_iw:
        return None

    ih = random.randint(min_ih, max_ih)
    iw = random.randint(min_iw, max_iw)
    fh, fw = ih + 2, iw + 2
    fr1 = H - 1 - bottom_margin
    fc1 = W - 1 - right_margin
    fr0 = fr1 - fh + 1
    fc0 = fc1 - fw + 1
    if fr0 < FRAME_TOP_LEFT_MIN or fc0 < FRAME_TOP_LEFT_MIN:
        return None

    out = np.full((H, W), BLACK, dtype=int)
    out[fr0, fc0 : fc1 + 1] = GREY
    out[fr1, fc0 : fc1 + 1] = GREY
    out[fr0 : fr1 + 1, fc0] = GREY
    out[fr0 : fr1 + 1, fc1] = GREY

    if edge == "top":
        positions = _anchor_positions(iw, n_brushes)
        if positions is None:
            return None
        side_flags = random.choice([{"L"}, {"R"}, {"L", "R"}])
        directions = [DIR_DOWN] * n_brushes
        if "L" in side_flags:
            directions[0] = DIR_LEFT
        if "R" in side_flags:
            directions[-1] = DIR_RIGHT
        anchors = [(fr0 + 1, fc0 + 1 + pos) for pos in positions]
    else:
        positions = _anchor_positions(ih, n_brushes)
        if positions is None:
            return None
        side_flags = random.choice([{"U"}, {"D"}, {"U", "D"}])
        directions = [DIR_RIGHT] * n_brushes
        if "U" in side_flags:
            directions[0] = DIR_UP
        if "D" in side_flags:
            directions[-1] = DIR_DOWN
        anchors = [(fr0 + 1 + pos, fc0 + 1) for pos in positions]

    colors = random.sample(BRUSH_COLORS, n_brushes)
    brush_specs: list[dict[str, object]] = []
    occupied_strokes: set[tuple[int, int]] = set()

    for color, anchor, direction, pos in zip(colors, anchors, directions, positions):
        out[anchor] = color
        length = _stroke_length(edge, direction, pos, ih, iw)
        cells = []
        for step in range(length):
            rr = anchor[0] + direction[0] * step
            cc = anchor[1] + direction[1] * step
            if not (fr0 + 1 <= rr <= fr1 - 1 and fc0 + 1 <= cc <= fc1 - 1):
                return None
            cells.append((rr, cc))
        if any(cell in occupied_strokes for cell in cells):
            return None
        occupied_strokes.update(cells)
        for rr, cc in cells:
            out[rr, cc] = color
        brush_specs.append(
            {
                "color": color,
                "anchor": anchor,
                "direction": direction,
                "length": length,
                "stroke_cells": cells,
            }
        )

    inp = out.copy()
    for spec in brush_specs:
        for rr, cc in spec["stroke_cells"][1:]:
            inp[rr, cc] = BLACK

    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[max(0, fr0 - 1) : min(H, fr1 + 2), max(0, fc0 - 1) : min(W, fc1 + 2)] = True

    if not _place_brushes(inp, out, forbidden, brush_specs):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
