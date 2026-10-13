from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg
import random


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1))
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the enclosed table slots by the single object color inside each slot, using the slot background color for empty slots.

    Input structure
    - The most common color forms the outer background and all enclosed slot
      backgrounds.
    - The enclosed background components form a complete rectangular array of
      slots, ordered by their positions.
    - Each slot contains either no other color or exactly one object color.

    Rule
    - Find all 4-connected components of the most common color that do not touch
      the input edge.
    - Sort their bounding boxes into row and column order to define the output
      table.
    - For each slot, output its unique non-background color if present; otherwise
      output the background color.

    Key insight
    - The large grid is an expanded table: the enclosed background components are
      the cells, and any colored object drawn inside a cell is that cell's summary
      value.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    slot_bboxes: list[tuple[int, int, int, int]] = []
    for coords in components_mask(g == bg, connectivity=4):
        if _touches_edge(coords, height, width):
            continue
        slot_bboxes.append(bbox_coords(coords))

    assert slot_bboxes, "expected enclosed background slots"
    row_starts = sorted({b[0] for b in slot_bboxes})
    col_starts = sorted({b[1] for b in slot_bboxes})
    assert len(slot_bboxes) == len(row_starts) * len(col_starts), (
        "expected a complete rectangular array of slots"
    )

    slot_by_anchor = {(b[0], b[1]): b for b in slot_bboxes}
    assert len(slot_by_anchor) == len(slot_bboxes), "slot anchors must be unique"

    out = np.full((len(row_starts), len(col_starts)), bg, dtype=int)
    for out_r, r0 in enumerate(row_starts):
        for out_c, c0 in enumerate(col_starts):
            assert (r0, c0) in slot_by_anchor, "missing slot in rectangular array"
            sr0, sc0, sr1, sc1 = slot_by_anchor[(r0, c0)]
            window = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            object_colors = np.unique(window[window != bg])
            assert object_colors.size <= 1, "each slot must contain at most one object color"
            if object_colors.size == 1:
                out[out_r, out_c] = int(object_colors[0])

    return out

# --------------- generator (att2) ---------------

# Canvas, table, frame, and slot dimensions.
CANVAS_MIN, CANVAS_MAX = 13, 19
TABLE_ROWS_RANGE, TABLE_COLS_RANGE = (2, 3), (2, 3)
ROW_SLOT_SIZE_RANGE, COL_SLOT_SIZE_RANGE = (3, 6), (3, 4)
AXIS_OUTER_OPTIONS = AXIS_PADDING_OPTIONS = AXIS_SEPARATOR_OPTIONS = (1, 2)

# Object and palette sampling limits.
N_COLORS, MAX_OBJECT_COLORS = 10, 6
MAX_OBJECT_HEIGHT, MAX_OBJECT_WIDTH = 3, 2
OBJECT_MARGIN, MAX_OBJECT_AREA = 1, 6

# Rejection sampling cap.
GEN_RETRY_LIMIT = 300


def _gen_axis_options(n_slots, slot_min, slot_max):
    options = []

    def rec_slots(i, cur):
        if i == n_slots:
            rec_seps(0, [], cur)
            return
        for size in range(slot_min, slot_max + 1):
            rec_slots(i + 1, cur + [size])

    def rec_seps(i, cur, slots):
        if i == n_slots - 1:
            for before in AXIS_OUTER_OPTIONS:
                for after in AXIS_OUTER_OPTIONS:
                    for pad_before in AXIS_PADDING_OPTIONS:
                        for pad_after in AXIS_PADDING_OPTIONS:
                            total = (
                                before
                                + after
                                + pad_before
                                + pad_after
                                + sum(slots)
                                + sum(cur)
                            )
                            if 13 <= total <= 19:
                                options.append(
                                    (
                                        tuple(slots),
                                        tuple(cur),
                                        before,
                                        after,
                                        pad_before,
                                        pad_after,
                                        total,
                                )
                            )
            return
        for sep in AXIS_SEPARATOR_OPTIONS:
            rec_seps(i + 1, cur + [sep], slots)

    rec_slots(0, [])
    return options


def _gen_positions(axis):
    sizes, seps, outer_before, _outer_after, pad_before, _pad_after, _total = axis
    starts = []
    pos = outer_before + pad_before
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i < len(seps):
            pos += seps[i]
    return starts


def _gen_choose_axis(n_slots, slot_min, slot_max, other_axis=None, max_object_loss=0):
    options = _gen_axis_options(n_slots, slot_min, slot_max)
    if other_axis is None:
        return random.choice(options)

    good = []
    for axis in options:
        row_axis = axis if slot_max == ROW_SLOT_SIZE_RANGE[1] else other_axis
        col_axis = axis if slot_max == COL_SLOT_SIZE_RANGE[1] else other_axis
        height = row_axis[-1]
        width = col_axis[-1]
        frame_height = height - row_axis[2] - row_axis[3]
        frame_width = width - col_axis[2] - col_axis[3]
        slot_area = sum(row_axis[0]) * sum(col_axis[0])
        outer_bg = height * width - frame_height * frame_width
        bg_room = outer_bg + slot_area - max_object_loss
        frame_area = frame_height * frame_width - slot_area
        if bg_room > frame_area:
            good.append(axis)
    return random.choice(good or options)


def _gen_assign_colors(shape, bg, frame):
    n_slots = shape[0] * shape[1]
    filled_count = random.randint(1, n_slots - 1)
    filled = random.sample(range(n_slots), filled_count)
    colors = [c for c in range(N_COLORS) if c not in (bg, frame)]
    n_colors = random.randint(1, min(MAX_OBJECT_COLORS, filled_count, len(colors)))
    palette = random.sample(colors, n_colors)
    assignments = palette[:]
    assignments.extend(random.choice(palette) for _ in range(filled_count - n_colors))
    random.shuffle(assignments)

    out = np.full(shape, bg, dtype=int)
    for idx, color in zip(filled, assignments):
        out[idx // shape[1], idx % shape[1]] = color
    return out, filled_count


def _gen_object_shape(slot_h, slot_w):
    by_area = {}
    for obj_h in range(1, min(MAX_OBJECT_HEIGHT, slot_h - 2 * OBJECT_MARGIN) + 1):
        for obj_w in range(1, min(MAX_OBJECT_WIDTH, slot_w - 2 * OBJECT_MARGIN) + 1):
            by_area.setdefault(obj_h * obj_w, []).append((obj_h, obj_w))
    area = random.choice(list(by_area))
    return random.choice(by_area[area])


def _gen_choose_layout(rows, cols, filled_count):
    row_axis = _gen_choose_axis(rows, *ROW_SLOT_SIZE_RANGE)
    loss = MAX_OBJECT_AREA * filled_count
    col_axis = _gen_choose_axis(cols, *COL_SLOT_SIZE_RANGE, row_axis, max_object_loss=loss)
    return row_axis, col_axis, _gen_positions(row_axis), _gen_positions(col_axis)


def _gen_render_frame(bg, frame, row_axis, col_axis, row_starts, col_starts):
    height = row_axis[-1]
    width = col_axis[-1]
    grid = np.full((height, width), bg, dtype=int)
    rt, rb = row_axis[2], row_axis[3]
    cl, cr = col_axis[2], col_axis[3]
    grid[rt : height - rb, cl : width - cr] = frame

    for r, slot_h in zip(row_starts, row_axis[0]):
        for c, slot_w in zip(col_starts, col_axis[0]):
            grid[r : r + slot_h, c : c + slot_w] = bg
    return grid


def _gen_place_objects(grid, out, bg, row_axis, col_axis, row_starts, col_starts):
    for out_r, slot_r in enumerate(row_starts):
        slot_h = row_axis[0][out_r]
        for out_c, slot_c in enumerate(col_starts):
            color = int(out[out_r, out_c])
            if color == bg:
                continue
            slot_w = col_axis[0][out_c]
            obj_h, obj_w = _gen_object_shape(slot_h, slot_w)
            obj_r = random.randint(slot_r + OBJECT_MARGIN, slot_r + slot_h - obj_h - OBJECT_MARGIN)
            obj_c = random.randint(slot_c + OBJECT_MARGIN, slot_c + slot_w - obj_w - OBJECT_MARGIN)
            grid[obj_r : obj_r + obj_h, obj_c : obj_c + obj_w] = color


def _gen_is_valid(grid, out, bg, frame):
    counts = np.bincount(grid.ravel(), minlength=N_COLORS)
    if int(np.argmax(counts)) != bg:
        return False
    if counts[frame] <= max(counts[c] for c in range(N_COLORS) if c not in (bg, frame)):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, out))


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*TABLE_ROWS_RANGE)
    cols = random.randint(*TABLE_COLS_RANGE)
    bg, frame = random.sample(range(N_COLORS), 2)

    out, filled_count = _gen_assign_colors((rows, cols), bg, frame)
    row_axis, col_axis, row_starts, col_starts = _gen_choose_layout(rows, cols, filled_count)
    grid = _gen_render_frame(bg, frame, row_axis, col_axis, row_starts, col_starts)
    _gen_place_objects(grid, out, bg, row_axis, col_axis, row_starts, col_starts)

    if _gen_is_valid(grid, out, bg, frame):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ac6f9922 input")
