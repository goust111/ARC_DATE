import numpy as np

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey open-top frame with red and extend a red spill line above it toward its doubled top cap.

    Input structure
    - The background is the dominant empty color.
    - The only foreground color is grey.
    - The grey pixels form one or more 4-connected rectangular frames with solid left, right, and bottom sides.
    - Each frame's top side has exactly one background gap and a two-pixel grey cap on one side of that gap.

    Rule
    - Preserve all grey pixels.
    - For each frame, color the background pixels inside the frame and the missing top pixel red.
    - On the row just above the frame, draw red from the top gap to the grid edge in the direction of the two-pixel cap.

    Key insight
    - The asymmetric two-pixel cap on the top edge marks the horizontal direction of the red spill ray.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    W = g.shape[1]
    assert bg != Color.GREY
    assert colors_present(g, exclude={bg}) == [Color.GREY]

    out = g.copy()
    components = components_mask(g == Color.GREY, connectivity=4)
    assert components

    for coords in components:
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r0 > 0
        assert r1 - r0 + 1 >= 3
        assert c1 - c0 + 1 == 4

        assert np.all(g[r0 + 1 : r1, c0] == Color.GREY)
        assert np.all(g[r0 + 1 : r1, c1] == Color.GREY)
        assert np.all(g[r1, c0 : c1 + 1] == Color.GREY)
        assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg)

        top = g[r0, c0 : c1 + 1]
        assert np.count_nonzero(top == Color.GREY) == 3
        gap_offsets = np.flatnonzero(top == bg)
        assert len(gap_offsets) == 1
        gap = c0 + int(gap_offsets[0])

        left_cap = gap - 2 >= c0 and np.all(g[r0, gap - 2 : gap] == Color.GREY)
        right_cap = gap + 2 <= c1 and np.all(g[r0, gap + 1 : gap + 3] == Color.GREY)
        assert left_cap != right_cap

        fill_region = out[r0:r1, c0 + 1 : c1]
        fill_region[g[r0:r1, c0 + 1 : c1] == bg] = Color.RED

        if left_cap:
            ray_cols = slice(0, gap + 1)
        else:
            ray_cols = slice(gap, W)
        assert np.all(g[r0 - 1, ray_cols] == bg)
        out[r0 - 1, ray_cols] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Grid and colors.
_GEN_SHAPE = (10, 10)
_GEN_H, _GEN_W = _GEN_SHAPE
_GEN_BG, _GEN_GREY, _GEN_RED = Color.BLACK, Color.GREY, Color.RED

# Frame geometry and orientation.
_FRAME_WIDTH = 4
_FRAME_RIGHT_OFFSET = _FRAME_WIDTH - 1
_FRAME_TOP_ROW_MIN, _FRAME_COL_MIN = 1, 0
_FRAME_COUNT_RANGE, _FRAME_HEIGHT_RANGE = (1, 2), (3, 5)
_SIDE_OPTIONS = ("left", "right")
_GAP_OFFSET_BY_SIDE = {"right": 1, "left": 2}

# Ray and placement constraints.
_RAY_LENGTH_RANGE, _TOUCH_DISTANCE = (4, 8), 1

# Retry budget for rejection sampling.
_GENERATE_RETRIES = 300


def _gen_gap_offset(side):
    return _GAP_OFFSET_BY_SIDE[side]


def _gen_frame_bbox(frame):
    r0, c0, h, _side = frame
    return r0, c0, r0 + h - 1, c0 + _FRAME_RIGHT_OFFSET


def _gen_frame_cells(frame):
    r0, c0, h, side = frame
    r1 = r0 + h - 1
    c1 = c0 + _FRAME_RIGHT_OFFSET
    gap_off = _gen_gap_offset(side)
    cells = [(r0, c0 + dc) for dc in range(_FRAME_WIDTH) if dc != gap_off]

    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    cells.extend((r1, c0 + dc) for dc in range(_FRAME_WIDTH))

    return cells


def _gen_ray_info(frame):
    r0, c0, _h, side = frame
    gap = c0 + _gen_gap_offset(side)
    if side == "left":
        return r0 - 1, slice(0, gap + 1)
    return r0 - 1, slice(gap, _GEN_W)


def _gen_ray_length(frame):
    _ray_r, ray_cols = _gen_ray_info(frame)
    return ray_cols.stop - ray_cols.start


def _gen_draw_frame(grid, frame):
    r0, c0, h, side = frame
    r1 = r0 + h - 1
    c1 = c0 + _FRAME_RIGHT_OFFSET
    gap_off = _gen_gap_offset(side)

    grid[r0, c0 : c0 + gap_off] = _GEN_GREY
    grid[r0, c0 + gap_off + 1 : c0 + _FRAME_WIDTH] = _GEN_GREY
    grid[r0 + 1 : r1, c0] = _GEN_GREY
    grid[r0 + 1 : r1, c1] = _GEN_GREY
    grid[r1, c0 : c0 + _FRAME_WIDTH] = _GEN_GREY


def _gen_render_frames(frames):
    grid = np.zeros(_GEN_SHAPE, dtype=int)
    for frame in frames:
        _gen_draw_frame(grid, frame)
    return grid


def _gen_bbox_overlaps(a, b):
    ar0, ac0, ar1, ac1 = _gen_frame_bbox(a)
    br0, bc0, br1, bc1 = _gen_frame_bbox(b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_grey_cells_touch(a, b):
    for ar, ac in _gen_frame_cells(a):
        for br, bc in _gen_frame_cells(b):
            if max(abs(ar - br), abs(ac - bc)) <= _TOUCH_DISTANCE:
                return True
    return False


def _gen_candidate_ok(candidate, placed):
    if not (_RAY_LENGTH_RANGE[0] <= _gen_ray_length(candidate) <= _RAY_LENGTH_RANGE[1]):
        return False

    candidate_ray_r, _ = _gen_ray_info(candidate)
    for frame in placed:
        ray_r, _ = _gen_ray_info(frame)
        if ray_r == candidate_ray_r:
            return False
        if _gen_bbox_overlaps(candidate, frame):
            return False
        if _gen_grey_cells_touch(candidate, frame):
            return False

    frames = placed + [candidate]
    grid = _gen_render_frames(frames)
    for frame in frames:
        ray_r, ray_cols = _gen_ray_info(frame)
        if np.any(grid[ray_r, ray_cols] != _GEN_BG):
            return False
    return True


def _gen_candidates(placed, height, side):
    candidates = []
    for r0 in range(_FRAME_TOP_ROW_MIN, _GEN_H - height + 1):
        for c0 in range(_FRAME_COL_MIN, _GEN_W - _FRAME_RIGHT_OFFSET):
            frame = (r0, c0, height, side)
            if _gen_candidate_ok(frame, placed):
                candidates.append(frame)
    return candidates


def _gen_expected_output(inp, frames):
    out = inp.copy()
    for frame in frames:
        r0, c0, h, _side = frame
        r1 = r0 + h - 1
        c1 = c0 + _FRAME_RIGHT_OFFSET
        fill = out[r0:r1, c0 + 1 : c1]
        fill[inp[r0:r1, c0 + 1 : c1] == _GEN_BG] = _GEN_RED
        ray_r, ray_cols = _gen_ray_info(frame)
        out[ray_r, ray_cols] = _GEN_RED
    return out


def _gen_choose_frame(placed):
    height = random.randint(*_FRAME_HEIGHT_RANGE)
    side = random.choice(_SIDE_OPTIONS)
    candidates = _gen_candidates(placed, height, side)
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    frames = []

    for _ in range(frame_count):
        frame = _gen_choose_frame(frames)
        if frame is None:
            return None
        frames.append(frame)

    inp = _gen_render_frames(frames)
    expected = _gen_expected_output(inp, frames)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GENERATE_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 4e469f39 input")
