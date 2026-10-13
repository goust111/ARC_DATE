import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each one-gap rectangular frame in red and let the fill spill straight out through the gap.

    Input structure
    - The grid has a uniform background color and one foreground color.
    - Each 4-connected foreground object is a 4x4 rectangular frame border with exactly one non-corner border pixel missing.
    Rule
    - Preserve the foreground frame pixels.
    - Recolor each frame's 2x2 interior, its missing border pixel, and the background ray continuing outward from that gap to red.
    Key insight
    - The missing border pixel identifies both the leak point and the outward direction for the red fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg != Color.RED, "red is reserved for the output fill"
    frame_colors = colors_present(g, exclude={bg})
    assert len(frame_colors) == 1, "expected exactly one frame color"
    frame_color = frame_colors[0]
    assert frame_color != Color.RED, "input frame color must differ from red"

    frames = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[frame_color],
        return_format="dict",
    )
    assert frames, "expected at least one frame"

    out = g.copy()
    height, width = map(int, g.shape)

    for frame in frames:
        assert frame["color"] == frame_color
        assert frame["area"] == 11, "expected a 4x4 border with one missing pixel"
        r0, c0, r1, c1 = frame["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == (4, 4), "expected a 4x4 frame"

        coords = {tuple(map(int, rc)) for rc in frame["coords"]}
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        gaps = list(border - coords)
        assert len(gaps) == 1, "expected exactly one missing frame pixel"
        gap_r, gap_c = gaps[0]

        if gap_r == r0:
            assert c0 < gap_c < c1, "gap must be a non-corner border pixel"
            dr, dc = -1, 0
        elif gap_r == r1:
            assert c0 < gap_c < c1, "gap must be a non-corner border pixel"
            dr, dc = 1, 0
        elif gap_c == c0:
            assert r0 < gap_r < r1, "gap must be a non-corner border pixel"
            dr, dc = 0, -1
        elif gap_c == c1:
            assert r0 < gap_r < r1, "gap must be a non-corner border pixel"
            dr, dc = 0, 1
        else:
            raise AssertionError("gap must lie on the frame border")

        paint = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
        paint.add((gap_r, gap_c))
        r, c = gap_r + dr, gap_c + dc
        while 0 <= r < height and 0 <= c < width:
            paint.add((r, c))
            r += dr
            c += dc

        for r, c in paint:
            assert g[r, c] == bg, "red fill must only replace background pixels"
            out[r, c] = Color.RED

    return out

# --------------- generator (o5x) ---------------

_GEN_BG_COLORS = (0, 3, 4, 5, 6, 7, 8, 9)

# Canvas and frame counts.
_GEN_MIN_HEIGHT = 7
_GEN_MAX_HEIGHT_EXCL = 12
_GEN_MIN_WIDTH = 9
_GEN_MAX_WIDTH_EXCL = 16
_GEN_THREE_FRAME_MIN_HEIGHT = 9
_GEN_MIN_FRAME_COUNT = 1
_GEN_MAX_FRAME_COUNT_EXCL = 4

# Frame geometry and colors.
_GEN_FRAME_SIZE = 4
_GEN_FRAME_EDGE = _GEN_FRAME_SIZE - 1
_GEN_INTERIOR_MARGIN = 1
_GEN_FRAME_COLOR = Color.BLUE
_GEN_FILL_COLOR = Color.RED
_GEN_GAP_OFFSETS = (1, 2)
_GEN_SIDES = ("top", "bottom", "left", "right")
_GEN_SIDE_DELTAS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Placement and retry limits.
_GEN_MAX_RAY_LEN = 3
_GEN_MIN_BBOX_GAP = 1
_GEN_MAX_ATTEMPTS = 300


def _gen_ray_cells(height, width, gap_r, gap_c, side):
    dr, dc = _GEN_SIDE_DELTAS[side]
    cells = [(gap_r, gap_c)]
    r, c = gap_r + dr, gap_c + dc
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return tuple(cells)


def _gen_frame_patch(bg, side, offset):
    patch = np.full((_GEN_FRAME_SIZE, _GEN_FRAME_SIZE), bg, dtype=int)
    patch[0, :] = _GEN_FRAME_COLOR
    patch[_GEN_FRAME_EDGE, :] = _GEN_FRAME_COLOR
    patch[:, 0] = _GEN_FRAME_COLOR
    patch[:, _GEN_FRAME_EDGE] = _GEN_FRAME_COLOR
    gap_r, gap_c = _gen_gap_cell(0, 0, side, offset)
    patch[gap_r, gap_c] = bg
    return patch


def _gen_bbox_cells(box):
    r0, c0, r1, c1 = box
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }


def _gen_bbox_separated(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= _GEN_MIN_BBOX_GAP or col_gap >= _GEN_MIN_BBOX_GAP


def _gen_gap_cell(r0, c0, side, offset):
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + _GEN_FRAME_EDGE, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + _GEN_FRAME_EDGE


def _gen_origin_ranges(height, width, side):
    """Return candidate top-left ranges whose outward rays are short enough."""
    last_row = height - _GEN_FRAME_SIZE
    last_col = width - _GEN_FRAME_SIZE

    if side == "top":
        row_start, row_stop = 0, min(_GEN_MAX_RAY_LEN, last_row) + 1
        col_start, col_stop = 0, last_col + 1
    elif side == "bottom":
        row_start = max(0, last_row - _GEN_MAX_RAY_LEN)
        row_stop = last_row + 1
        col_start, col_stop = 0, last_col + 1
    elif side == "left":
        row_start, row_stop = 0, last_row + 1
        col_start, col_stop = 0, min(_GEN_MAX_RAY_LEN, last_col) + 1
    else:
        row_start, row_stop = 0, last_row + 1
        col_start = max(0, last_col - _GEN_MAX_RAY_LEN)
        col_stop = last_col + 1

    return range(row_start, row_stop), range(col_start, col_stop)


def _gen_candidate_specs(height, width):
    """Enumerate all frame placements that satisfy the local ray-length bound."""
    specs = []
    for side in _GEN_SIDES:
        row_choices, col_choices = _gen_origin_ranges(height, width, side)
        for r0 in row_choices:
            for c0 in col_choices:
                for offset in _GEN_GAP_OFFSETS:
                    gap_r, gap_c = _gen_gap_cell(r0, c0, side, offset)
                    box = (r0, c0, r0 + _GEN_FRAME_EDGE, c0 + _GEN_FRAME_EDGE)
                    ray = _gen_ray_cells(height, width, gap_r, gap_c, side)
                    if len(ray) - 1 <= _GEN_MAX_RAY_LEN:
                        specs.append(
                            {
                                "bbox": box,
                                "bbox_cells": _gen_bbox_cells(box),
                                "gap": (gap_r, gap_c),
                                "offset": offset,
                                "ray": frozenset(ray),
                                "side": side,
                            }
                        )
    return specs


def _gen_compatible(candidate, selected):
    for other in selected:
        if not _gen_bbox_separated(candidate["bbox"], other["bbox"]):
            return False
        if candidate["ray"] & other["bbox_cells"]:
            return False
        if other["ray"] & candidate["bbox_cells"]:
            return False
    return True


def _gen_choose(seq):
    return seq[np.random.randint(0, len(seq))]


def _gen_select_frames(candidates, count):
    """Sample compatible frame specs while preserving the side/offset preference order."""
    selected = []
    for _ in range(count):
        compatible = [c for c in candidates if _gen_compatible(c, selected)]
        if not compatible:
            return None

        side = _gen_choose(_GEN_SIDES)
        offset = int(np.random.randint(min(_GEN_GAP_OFFSETS), max(_GEN_GAP_OFFSETS) + 1))
        preferred = [
            c
            for c in compatible
            if c["side"] == side and c["offset"] == offset
        ]
        if not preferred:
            preferred = [c for c in compatible if c["side"] == side]
        if not preferred:
            preferred = compatible

        selected.append(_gen_choose(preferred))
    return selected


def _gen_attempt():
    frame_count = int(np.random.randint(_GEN_MIN_FRAME_COUNT, _GEN_MAX_FRAME_COUNT_EXCL))
    if frame_count == 3:
        height = int(np.random.randint(_GEN_THREE_FRAME_MIN_HEIGHT, _GEN_MAX_HEIGHT_EXCL))
        width = int(np.random.randint(_GEN_MIN_WIDTH, _GEN_MAX_WIDTH_EXCL))
    else:
        height = int(np.random.randint(_GEN_MIN_HEIGHT, _GEN_MAX_HEIGHT_EXCL))
        width = int(np.random.randint(_GEN_MIN_WIDTH, _GEN_MAX_WIDTH_EXCL))

    bg = int(_gen_choose(_GEN_BG_COLORS))
    candidates = _gen_candidate_specs(height, width)
    frames = _gen_select_frames(candidates, frame_count)
    if frames is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = _gen_frame_patch(bg, frame["side"], frame["offset"])
        paste(inp, patch, r0, c0)
        paste(expected, patch, r0, c0)
        expected[
            r0 + _GEN_INTERIOR_MARGIN : r1,
            c0 + _GEN_INTERIOR_MARGIN : c1,
        ] = _GEN_FILL_COLOR
        for r, c in frame["ray"]:
            expected[r, c] = _GEN_FILL_COLOR

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 292dd178 input")
