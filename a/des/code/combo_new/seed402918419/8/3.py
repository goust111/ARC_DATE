from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 22
GRID_MAX = 30
SLOT_COUNT_RANGE = (2, 4)
BASE_SIDE_RANGE = (3, 5)
OUTER_MARGIN_RANGE = (1, 3)
INTERNAL_GAP_RANGE = (2, 3)
GEN_RETRIES = 300
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _rotation_count(mask: np.ndarray) -> int:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)})


def _solve_assignment(candidates: list[list[tuple[int, int]]], slot_count: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(candidates)), key=lambda idx: len(candidates[idx]))
    assignment: list[tuple[int, int] | None] = [None] * len(candidates)
    used_slots: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        piece_idx = order[pos]
        for slot_idx, rot in candidates[piece_idx]:
            if slot_idx in used_slots:
                continue
            used_slots.add(slot_idx)
            assignment[piece_idx] = (slot_idx, rot)
            if dfs(pos + 1):
                return True
            used_slots.remove(slot_idx)
            assignment[piece_idx] = None
        return False

    if slot_count != len(candidates):
        return None
    if not dfs(0):
        return None
    return [(int(slot_idx), int(rot)) for slot_idx, rot in assignment if slot_idx is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered puzzle piece back into the matching toothed cavity of the gray frame.

    Input structure:
    - The most common color is the plain background.
    - One large gray connected structure contains several enclosed background cavities whose
      edges have one-cell teeth and notches.
    - Outside the gray structure, several monochrome connected pieces have the same irregular
      cavity shapes up to rotation.

    Rule:
    - Identify the gray frame and collect every enclosed background cavity as a slot shape.
    - Collect every external colored piece, try all four rotations, and match it to the unique
      slot with the same occupied mask.
    - Remove the scattered pieces from the outside background and repaint each matched slot with
      the piece's color, leaving the gray frame unchanged.

    Key insight:
    - The tooth/notch complement is equivalent to exact shape fitting: a rotated piece occupies
      exactly the background cells of its matching cavity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _most_common_color(g)

    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected a frame color and at least one piece color"
    frame = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    slots = []
    for coords in _components(g == bg, connectivity=4):
        if _touches_edge(coords, h, w):
            continue
        bbox, mask = _crop_mask(coords)
        slots.append({"bbox": bbox, "mask": mask})
    assert slots, "expected enclosed cavities inside the frame"

    pieces = []
    for color in np.unique(g):
        color = int(color)
        if color in {bg, frame}:
            continue
        for coords in _components(g == color, connectivity=4):
            _bbox0, mask = _crop_mask(coords)
            pieces.append({"color": color, "mask": mask})
    assert pieces, "expected scattered puzzle pieces"
    assert len(slots) == len(pieces), "slot and piece counts must match"

    candidates: list[list[tuple[int, int]]] = []
    for piece in pieces:
        piece_candidates: list[tuple[int, int]] = []
        for slot_idx, slot in enumerate(slots):
            for rot in range(4):
                rotated = np.rot90(piece["mask"], rot)
                if rotated.shape == slot["mask"].shape and np.array_equal(rotated, slot["mask"]):
                    piece_candidates.append((slot_idx, rot))
        assert piece_candidates, "every piece should fit at least one slot"
        candidates.append(piece_candidates)

    assignment = _solve_assignment(candidates, len(slots))
    assert assignment is not None, "piece-to-slot assignment must be solvable"

    out = np.full_like(g, bg)
    out[g == frame] = frame
    for piece, (slot_idx, _rot) in zip(pieces, assignment):
        r0, c0, r1, c1 = slots[slot_idx]["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[slots[slot_idx]["mask"]] = int(piece["color"])
    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_from_mods(height: int, width: int, mods: list[tuple[str, int, int]]) -> np.ndarray:
    canvas = np.zeros((height + 2, width + 2), dtype=bool)
    canvas[1 : height + 1, 1 : width + 1] = True
    for side, idx, delta in mods:
        if side == "top":
            if delta > 0:
                canvas[0, idx + 1] = True
            else:
                canvas[1, idx + 1] = False
        elif side == "bottom":
            if delta > 0:
                canvas[height + 1, idx + 1] = True
            else:
                canvas[height, idx + 1] = False
        elif side == "left":
            if delta > 0:
                canvas[idx + 1, 0] = True
            else:
                canvas[idx + 1, 1] = False
        elif side == "right":
            if delta > 0:
                canvas[idx + 1, width + 1] = True
            else:
                canvas[idx + 1, width] = False
    return _trim(canvas)


def _make_slot_shape() -> np.ndarray | None:
    base_h = random.randint(*BASE_SIDE_RANGE)
    base_w = random.randint(*BASE_SIDE_RANGE)
    options: list[tuple[str, int]] = []
    for c in range(1, base_w - 1):
        options.append(("top", c))
        options.append(("bottom", c))
    for r in range(1, base_h - 1):
        options.append(("left", r))
        options.append(("right", r))
    if len(options) < 4:
        return None

    mod_count = random.randint(4, min(7, len(options)))
    picks = random.sample(options, mod_count)
    deltas = [random.choice((-1, 1)) for _ in picks]
    if all(delta == 1 for delta in deltas):
        deltas[0] = -1
    if all(delta == -1 for delta in deltas):
        deltas[0] = 1
    mods = [(side, idx, delta) for (side, idx), delta in zip(picks, deltas)]
    mask = _shape_from_mods(base_h, base_w, mods)

    if _rotation_count(mask) != 4:
        return None
    if len(_components(mask, connectivity=4)) != 1:
        return None
    if np.all(mask):
        return None
    if mask.shape[0] > 7 or mask.shape[1] > 7:
        return None
    return mask


def _row_partition(items: list[np.ndarray], row_count: int) -> list[list[np.ndarray]]:
    counts = [len(items) // row_count] * row_count
    for idx in range(len(items) % row_count):
        counts[idx] += 1
    rows: list[list[np.ndarray]] = []
    start = 0
    for count in counts:
        rows.append(items[start : start + count])
        start += count
    return rows


def _pack_slots(shapes: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]]:
    shuffled = shapes[:]
    random.shuffle(shuffled)
    row_count = random.randint(1, min(2, len(shuffled)))
    rows = _row_partition(shuffled, row_count)

    row_heights = [max(shape.shape[0] for shape in row) for row in rows]
    row_gaps = [random.randint(*INTERNAL_GAP_RANGE) for _ in range(max(0, row_count - 1))]
    top_margin = random.randint(*OUTER_MARGIN_RANGE)
    bottom_margin = random.randint(*OUTER_MARGIN_RANGE)
    left_margin = random.randint(*OUTER_MARGIN_RANGE)
    right_margin = random.randint(*OUTER_MARGIN_RANGE)

    board_h = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
    board_w = 0
    row_widths = []
    for row in rows:
        col_gaps = [random.randint(*INTERNAL_GAP_RANGE) for _ in range(max(0, len(row) - 1))]
        row_width = left_margin + right_margin + sum(shape.shape[1] for shape in row) + sum(col_gaps)
        row_widths.append((row_width, col_gaps))
        board_w = max(board_w, row_width)

    positions_by_id: dict[int, tuple[int, int]] = {}
    cur_r = top_margin
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        row_width, col_gaps = row_widths[row_idx]
        extra_w = board_w - row_width
        cur_c = left_margin + random.randint(0, extra_w)
        for shape_idx, shape in enumerate(row):
            jitter_r = random.randint(0, row_h - shape.shape[0])
            positions_by_id[id(shape)] = (cur_r + jitter_r, cur_c)
            cur_c += shape.shape[1]
            if shape_idx < len(col_gaps):
                cur_c += col_gaps[shape_idx]
        cur_r += row_h
        if row_idx < len(row_gaps):
            cur_r += row_gaps[row_idx]

    return board_h, board_w, [positions_by_id[id(shape)] for shape in shapes]


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, value: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(value)


def _mark_reserved(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _place_piece(
    grid: np.ndarray,
    reserved: np.ndarray,
    piece_mask: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    ph, pw = piece_mask.shape
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        if reserved[top : top + ph, left : left + pw].any():
            continue
        pad_r0 = max(0, top - 1)
        pad_c0 = max(0, left - 1)
        pad_r1 = min(h, top + ph + 1)
        pad_c1 = min(w, left + pw + 1)
        if reserved[pad_r0:pad_r1, pad_c0:pad_c1].any():
            continue
        _paint_mask(grid, top, left, piece_mask, color)
        _mark_reserved(reserved, top, left, ph, pw, pad=1)
        return True
    return False


def _expected_output(
    height: int,
    width: int,
    board_top: int,
    board_left: int,
    board_h: int,
    board_w: int,
    shapes: list[np.ndarray],
    slot_positions: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    out[board_top : board_top + board_h, board_left : board_left + board_w] = FRAME
    for shape, (slot_r, slot_c), color in zip(shapes, slot_positions, colors):
        _paint_mask(out, board_top + slot_r, board_left + slot_c, shape, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    slot_count = random.randint(*SLOT_COUNT_RANGE)

    shapes: list[np.ndarray] = []
    seen_keys = set()
    for _ in range(slot_count):
        shape = None
        for _ in range(SHAPE_RETRIES):
            candidate = _make_slot_shape()
            if candidate is None:
                continue
            key = _canonical_shape_key(candidate)
            if key in seen_keys:
                continue
            shape = candidate
            seen_keys.add(key)
            break
        if shape is None:
            return None
        shapes.append(shape)

    board_h, board_w, slot_positions = _pack_slots(shapes)
    min_h = max(GRID_MIN, board_h + 6)
    min_w = max(GRID_MIN, board_w + 6)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = random.randint(min_h, GRID_MAX)
    width = random.randint(min_w, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)

    board_top = random.randint(1, height - board_h - 1)
    board_left = random.randint(1, width - board_w - 1)
    grid[board_top : board_top + board_h, board_left : board_left + board_w] = FRAME
    for shape, (slot_r, slot_c) in zip(shapes, slot_positions):
        _paint_mask(grid, board_top + slot_r, board_left + slot_c, shape, BG)

    colors = random.sample(PIECE_COLORS, slot_count)
    expected = _expected_output(
        height,
        width,
        board_top,
        board_left,
        board_h,
        board_w,
        shapes,
        slot_positions,
        colors,
    )

    reserved = np.zeros((height, width), dtype=bool)
    _mark_reserved(reserved, board_top, board_left, board_h, board_w, pad=1)

    for shape, color in zip(shapes, colors):
        rotation = random.randint(0, 3)
        piece_mask = np.rot90(shape, rotation)
        if not _place_piece(grid, reserved, piece_mask, color):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
