from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


FRAME_COLOR = Color.GREY
SLIDER_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRY_LIMIT = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _parse_frame(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    frame_mask = g == FRAME_COLOR
    frame_box = bbox(frame_mask)
    assert frame_box is not None, "expected a gray frame"
    r0, c0, r1, c1 = frame_box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a cavity"

    border_cells = set(_frame_border_cells(r0, c0, r1, c1))
    frame_cells = {tuple(map(int, rc)) for rc in np.argwhere(frame_mask)}
    assert frame_cells <= border_cells, "gray pixels must stay on one rectangle border"

    missing = [cell for cell in border_cells if cell not in frame_cells]
    assert len(missing) == 1, "expected exactly one border gap"
    gap_r, gap_c = missing[0]
    assert gap_r in (r0, r1) or gap_c in (c0, c1)
    assert not ((gap_r in (r0, r1)) and (gap_c in (c0, c1))), "gap cannot be a corner"

    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    movable = (g != bg) & (g != FRAME_COLOR)
    allowed = np.zeros_like(movable, dtype=bool)
    allowed[r0 : r1 + 1, :c0] = True
    allowed[r0 : r1 + 1, c1 + 1 :] = True
    allowed[:r0, c0 : c1 + 1] = True
    allowed[r1 + 1 :, c0 : c1 + 1] = True
    if gap_side in ("left", "right"):
        allowed[gap_r, c0 + 1 : c1] = True
    else:
        allowed[r0 + 1 : r1, gap_c] = True
    assert not np.any(movable & ~allowed), "movable pixels must be outside the frame or on the gate line inside"

    return {
        "bg": bg,
        "shape": g.shape,
        "frame": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "gap_pos": (gap_r, gap_c),
        "movable": movable,
    }


def _place_horizontal(out: np.ndarray, row: int, start_col: int, colors: list[int]) -> None:
    for offset, color in enumerate(colors):
        out[row, start_col + offset] = int(color)


def _place_vertical(out: np.ndarray, col: int, start_row: int, colors: list[int]) -> None:
    for offset, color in enumerate(colors):
        out[start_row + offset, col] = int(color)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _parse_frame(g)
    bg = int(info["bg"])
    r0, c0, r1, c1 = info["frame"]
    gap_side = info["gap_side"]
    gap_r, gap_c = info["gap_pos"]
    movable = info["movable"]

    out = g.copy()
    out[movable] = bg

    if gap_side in ("left", "right"):
        gate_inside_positions = [(gap_r, c) for c in range(c0 + 1, c1) if movable[gap_r, c]]
        gate_inside_colors = [int(g[r, c]) for r, c in gate_inside_positions]
    else:
        gate_inside_positions = [(r, gap_c) for r in range(r0 + 1, r1) if movable[r, gap_c]]
        gate_inside_colors = [int(g[r, c]) for r, c in gate_inside_positions]

    gate_triggered = False

    for row in range(r0, r1 + 1):
        left_ext = [int(g[row, c]) for c in range(c0) if movable[row, c]]
        if gap_side == "left" and row == gap_r:
            gate_triggered = bool(left_ext)
            if gate_triggered:
                seq = left_ext + gate_inside_colors
                assert len(seq) <= c1 - c0 - 1, "gate queue must fit inside the frame"
                _place_horizontal(out, row, c1 - len(seq), seq)
        elif left_ext:
            assert len(left_ext) <= c0, "left stack exceeds available margin"
            _place_horizontal(out, row, c0 - len(left_ext), left_ext)

        right_ext = [int(g[row, c]) for c in range(c1 + 1, g.shape[1]) if movable[row, c]]
        if gap_side == "right" and row == gap_r:
            gate_triggered = bool(right_ext)
            if gate_triggered:
                seq = gate_inside_colors + right_ext
                assert len(seq) <= c1 - c0 - 1, "gate queue must fit inside the frame"
                _place_horizontal(out, row, c0 + 1, seq)
        elif right_ext:
            assert len(right_ext) <= g.shape[1] - c1 - 1, "right stack exceeds available margin"
            _place_horizontal(out, row, c1 + 1, right_ext)

    for col in range(c0, c1 + 1):
        top_ext = [int(g[r, col]) for r in range(r0) if movable[r, col]]
        if gap_side == "top" and col == gap_c:
            gate_triggered = bool(top_ext)
            if gate_triggered:
                seq = top_ext + gate_inside_colors
                assert len(seq) <= r1 - r0 - 1, "gate queue must fit inside the frame"
                _place_vertical(out, col, r1 - len(seq), seq)
        elif top_ext:
            assert len(top_ext) <= r0, "top stack exceeds available margin"
            _place_vertical(out, col, r0 - len(top_ext), top_ext)

        bottom_ext = [int(g[r, col]) for r in range(r1 + 1, g.shape[0]) if movable[r, col]]
        if gap_side == "bottom" and col == gap_c:
            gate_triggered = bool(bottom_ext)
            if gate_triggered:
                seq = gate_inside_colors + bottom_ext
                assert len(seq) <= r1 - r0 - 1, "gate queue must fit inside the frame"
                _place_vertical(out, col, r0 + 1, seq)
        elif bottom_ext:
            assert len(bottom_ext) <= g.shape[0] - r1 - 1, "bottom stack exceeds available margin"
            _place_vertical(out, col, r1 + 1, bottom_ext)

    if not gate_triggered:
        for r, c in gate_inside_positions:
            out[r, c] = int(g[r, c])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide outside pixels toward a gapped hollow frame, letting gate-aligned ones enter and compact an interior queue.

    Input structure:
    - A black background contains one gray one-cell-thick hollow rectangular frame.
    - Exactly one non-corner border cell of the frame is missing, creating a one-cell gate.
    - Colored single-pixel sliders lie outside the frame on rows or columns perpendicular to some frame side.
    - The frame interior may contain loose colored pixels only on the same row or column as the gate.

    Rule:
    - Every outside slider moves straight toward the frame along its row or column.
    - Sliders that meet a solid wall stop in the nearest outside cell, and additional sliders on that same line stack behind them while preserving line order.
    - Sliders whose path goes through the gate enter the frame; if at least one such slider exists, it pushes every loose interior pixel on that gate line until the whole ordered queue is compacted against the opposite inner wall.
    - The gray frame and the open gate stay unchanged, and all other pixels remain background.

    Key insight:
    - The puzzle decomposes into independent side-aligned lines: solid lines compact outside the frame, while the unique gate line redirects its queue into the cavity and compresses it against the far inner wall.
    """
    return _transform(grid)


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int], gap_side: str, gap_pos: tuple[int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    gr, gc = gap_pos
    grid[gr, gc] = Color.BLACK


def _sample_weighted_count(max_count: int, weights: dict[int, int], *, allow_zero: bool) -> int:
    choices = [k for k in range(max_count + 1) if allow_zero or k > 0]
    if not choices:
        return 0
    picked = random.choices(choices, weights=[weights.get(k, 1) for k in choices], k=1)[0]
    return int(picked)


def _sample_positions(pool: list[int], count: int, forbidden_sorted: list[int] | None = None) -> list[int]:
    if count == 0:
        return []
    assert len(pool) >= count
    for _ in range(40):
        picked = sorted(random.sample(pool, count))
        if forbidden_sorted is None or picked != forbidden_sorted:
            return picked
    return sorted(random.sample(pool, count))


def _axis_candidates(side: str, frame: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame
    if side in ("left", "right"):
        return list(range(r0, r1 + 1))
    return list(range(c0, c1 + 1))


def _margin_length(side: str, shape: tuple[int, int], frame: tuple[int, int, int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = frame
    if side == "left":
        return c0
    if side == "right":
        return w - c1 - 1
    if side == "top":
        return r0
    return h - r1 - 1


def _solid_compact_positions(side: str, shape: tuple[int, int], frame: tuple[int, int, int, int], count: int) -> list[int]:
    h, w = shape
    r0, c0, r1, c1 = frame
    if side == "left":
        return list(range(c0 - count, c0))
    if side == "right":
        return list(range(c1 + 1, c1 + 1 + count))
    if side == "top":
        return list(range(r0 - count, r0))
    return list(range(r1 + 1, r1 + 1 + count))


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full(spec["shape"], spec["bg"], dtype=int)
    _draw_frame(grid, spec["frame"], spec["gap_side"], spec["gap_pos"])

    for row, entries in spec["outside"]["left"].items():
        for col, color in entries:
            grid[row, col] = color
    for row, entries in spec["outside"]["right"].items():
        for col, color in entries:
            grid[row, col] = color
    for col, entries in spec["outside"]["top"].items():
        for row, color in entries:
            grid[row, col] = color
    for col, entries in spec["outside"]["bottom"].items():
        for row, color in entries:
            grid[row, col] = color
    for r, c, color in spec["inside"]:
        grid[r, c] = color
    return grid


def _expected_from_spec(spec: dict) -> np.ndarray:
    out = _render_input(spec)
    bg = spec["bg"]
    r0, c0, r1, c1 = spec["frame"]
    gap_side = spec["gap_side"]
    gap_r, gap_c = spec["gap_pos"]

    for row, entries in spec["outside"]["left"].items():
        for col, _color in entries:
            out[row, col] = bg
    for row, entries in spec["outside"]["right"].items():
        for col, _color in entries:
            out[row, col] = bg
    for col, entries in spec["outside"]["top"].items():
        for row, _color in entries:
            out[row, col] = bg
    for col, entries in spec["outside"]["bottom"].items():
        for row, _color in entries:
            out[row, col] = bg
    for r, c, _color in spec["inside"]:
        out[r, c] = bg

    if gap_side in ("left", "right"):
        inside_axis = sorted((c, color) for r, c, color in spec["inside"])
        inside_colors = [color for _c, color in inside_axis]
    else:
        inside_axis = sorted((r, color) for r, c, color in spec["inside"])
        inside_colors = [color for _r, color in inside_axis]

    gate_triggered = False

    for row, entries in spec["outside"]["left"].items():
        colors = [color for _col, color in sorted(entries)]
        if gap_side == "left" and row == gap_r:
            gate_triggered = bool(colors)
            if gate_triggered:
                seq = colors + inside_colors
                _place_horizontal(out, row, c1 - len(seq), seq)
        elif colors:
            _place_horizontal(out, row, c0 - len(colors), colors)

    for row, entries in spec["outside"]["right"].items():
        colors = [color for _col, color in sorted(entries)]
        if gap_side == "right" and row == gap_r:
            gate_triggered = bool(colors)
            if gate_triggered:
                seq = inside_colors + colors
                _place_horizontal(out, row, c0 + 1, seq)
        elif colors:
            _place_horizontal(out, row, c1 + 1, colors)

    for col, entries in spec["outside"]["top"].items():
        colors = [color for _row, color in sorted(entries)]
        if gap_side == "top" and col == gap_c:
            gate_triggered = bool(colors)
            if gate_triggered:
                seq = colors + inside_colors
                _place_vertical(out, col, r1 - len(seq), seq)
        elif colors:
            _place_vertical(out, col, r0 - len(colors), colors)

    for col, entries in spec["outside"]["bottom"].items():
        colors = [color for _row, color in sorted(entries)]
        if gap_side == "bottom" and col == gap_c:
            gate_triggered = bool(colors)
            if gate_triggered:
                seq = inside_colors + colors
                _place_vertical(out, col, r0 + 1, seq)
        elif colors:
            _place_vertical(out, col, r1 + 1, colors)

    if not gate_triggered:
        for r, c, color in spec["inside"]:
            out[r, c] = color

    return out


def _build_spec() -> dict:
    top_margin = random.randint(3, 6)
    bottom_margin = random.randint(3, 6)
    left_margin = random.randint(3, 6)
    right_margin = random.randint(3, 6)

    outer_h = random.randint(6, 12)
    outer_w = random.randint(6, 12)
    shape = (top_margin + outer_h + bottom_margin, left_margin + outer_w + right_margin)
    assert max(shape) <= 30

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    frame = (r0, c0, r1, c1)

    gap_side = random.choice(SIDES)
    if gap_side == "left":
        gap_pos = (random.randint(r0 + 1, r1 - 1), c0)
    elif gap_side == "right":
        gap_pos = (random.randint(r0 + 1, r1 - 1), c1)
    elif gap_side == "top":
        gap_pos = (r0, random.randint(c0 + 1, c1 - 1))
    else:
        gap_pos = (r1, random.randint(c0 + 1, c1 - 1))

    spec = {
        "bg": Color.BLACK,
        "shape": shape,
        "frame": frame,
        "gap_side": gap_side,
        "gap_pos": gap_pos,
        "outside": {side: {} for side in SIDES},
        "inside": [],
    }

    if gap_side in ("left", "right"):
        gate_span = c1 - c0 - 1
        gate_line = gap_pos[0]
    else:
        gate_span = r1 - r0 - 1
        gate_line = gap_pos[1]

    gate_margin = _margin_length(gap_side, shape, frame)
    max_gate_ext = min(3, gate_margin, gate_span)
    gate_ext_count = _sample_weighted_count(max_gate_ext, {0: 1, 1: 3, 2: 3, 3: 2}, allow_zero=True)
    max_inside = min(3, gate_span - gate_ext_count if gate_ext_count > 0 else gate_span)
    inside_count = _sample_weighted_count(max_inside, {0: 2, 1: 3, 2: 2, 3: 1}, allow_zero=True)

    gate_palette = random.sample(SLIDER_COLORS, gate_ext_count + inside_count)
    random.shuffle(gate_palette)
    gate_ext_colors = gate_palette[:gate_ext_count]
    inside_colors = gate_palette[gate_ext_count:]

    if gap_side == "left":
        ext_positions = _sample_positions(
            list(range(c0)),
            gate_ext_count,
            forbidden_sorted=_solid_compact_positions("left", shape, frame, gate_ext_count) if gate_ext_count else None,
        )
        spec["outside"]["left"][gate_line] = list(zip(ext_positions, gate_ext_colors))
        inside_positions = _sample_positions(list(range(c0 + 1, c1)), inside_count)
        spec["inside"] = [(gate_line, c, color) for c, color in zip(inside_positions, inside_colors)]
    elif gap_side == "right":
        ext_positions = _sample_positions(
            list(range(c1 + 1, shape[1])),
            gate_ext_count,
            forbidden_sorted=_solid_compact_positions("right", shape, frame, gate_ext_count) if gate_ext_count else None,
        )
        spec["outside"]["right"][gate_line] = list(zip(ext_positions, gate_ext_colors))
        inside_positions = _sample_positions(list(range(c0 + 1, c1)), inside_count)
        spec["inside"] = [(gate_line, c, color) for c, color in zip(inside_positions, inside_colors)]
    elif gap_side == "top":
        ext_positions = _sample_positions(
            list(range(r0)),
            gate_ext_count,
            forbidden_sorted=_solid_compact_positions("top", shape, frame, gate_ext_count) if gate_ext_count else None,
        )
        spec["outside"]["top"][gate_line] = list(zip(ext_positions, gate_ext_colors))
        inside_positions = _sample_positions(list(range(r0 + 1, r1)), inside_count)
        spec["inside"] = [(r, gate_line, color) for r, color in zip(inside_positions, inside_colors)]
    else:
        ext_positions = _sample_positions(
            list(range(r1 + 1, shape[0])),
            gate_ext_count,
            forbidden_sorted=_solid_compact_positions("bottom", shape, frame, gate_ext_count) if gate_ext_count else None,
        )
        spec["outside"]["bottom"][gate_line] = list(zip(ext_positions, gate_ext_colors))
        inside_positions = _sample_positions(list(range(r0 + 1, r1)), inside_count)
        spec["inside"] = [(r, gate_line, color) for r, color in zip(inside_positions, inside_colors)]

    active_solid_lines = 0
    for side in SIDES:
        candidates = _axis_candidates(side, frame)
        if side == gap_side:
            candidates = [line for line in candidates if line != gate_line]
        random.shuffle(candidates)
        take = min(len(candidates), random.randint(0, 2))
        for line in candidates[:take]:
            margin = _margin_length(side, shape, frame)
            max_count = min(3, margin)
            count = _sample_weighted_count(max_count, {1: 3, 2: 3, 3: 2}, allow_zero=False)
            colors = random.sample(SLIDER_COLORS, count)
            if side == "left":
                positions = _sample_positions(
                    list(range(c0)),
                    count,
                    forbidden_sorted=_solid_compact_positions(side, shape, frame, count),
                )
                spec["outside"]["left"][line] = list(zip(positions, colors))
            elif side == "right":
                positions = _sample_positions(
                    list(range(c1 + 1, shape[1])),
                    count,
                    forbidden_sorted=_solid_compact_positions(side, shape, frame, count),
                )
                spec["outside"]["right"][line] = list(zip(positions, colors))
            elif side == "top":
                positions = _sample_positions(
                    list(range(r0)),
                    count,
                    forbidden_sorted=_solid_compact_positions(side, shape, frame, count),
                )
                spec["outside"]["top"][line] = list(zip(positions, colors))
            else:
                positions = _sample_positions(
                    list(range(r1 + 1, shape[0])),
                    count,
                    forbidden_sorted=_solid_compact_positions(side, shape, frame, count),
                )
                spec["outside"]["bottom"][line] = list(zip(positions, colors))
            active_solid_lines += 1

    assert active_solid_lines + int(gate_ext_count > 0) >= 2, "need multiple active lines for clarity"
    return spec


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            spec = _build_spec()
            inp = _render_input(spec)
            expected = _expected_from_spec(spec)
        except AssertionError:
            continue
        if np.array_equal(inp, expected):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        return inp
    raise RuntimeError("failed to generate a valid gap-gate compaction puzzle")
