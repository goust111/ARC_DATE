import random

import numpy as np


BG = 0
YELLOW = 4
MAROON = 9
ROW_COLORS = (1, 2, 3, 5, 6, 7, 8)
DIRECTIONS = ("right", "down", "left", "up")

# The flag is a 2x2 L triomino. We use a fixed cyclic decoding:
# above-left nub -> right, above-right nub -> down,
# below-right nub -> left, below-left nub -> up.
FLAG_SHAPES = {
    "right": frozenset({(0, 0), (1, 0), (1, 1)}),
    "down": frozenset({(0, 1), (1, 0), (1, 1)}),
    "left": frozenset({(0, 0), (0, 1), (1, 1)}),
    "up": frozenset({(0, 0), (0, 1), (1, 0)}),
}
FLAG_DECODE = {shape: direction for direction, shape in FLAG_SHAPES.items()}

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
FRAME_H_RANGE = (5, 7)
FRAME_W_RANGE = (5, 7)
OBJECT_GAP = 1
PLACEMENT_TRIES = 250
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)

    return comps


def _frame_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == MAROON):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if not (4 <= height <= 7 and 4 <= width <= 7):
            continue
        if set(pts) != _frame_border(r0, c0, r1, c1):
            continue
        frames.append((r0, c0, r1, c1))
    return sorted(frames)


def _decode_flag(interior: np.ndarray) -> str:
    yellow = np.argwhere(interior == YELLOW)
    assert yellow.shape[0] == 3
    r0, c0 = yellow.min(axis=0)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in yellow)
    assert norm in FLAG_DECODE
    return FLAG_DECODE[norm]


def _frame_row_color_and_count(interior: np.ndarray) -> tuple[int, int]:
    colors = sorted(set(int(v) for v in interior.flatten()) - {BG, YELLOW, MAROON})
    assert len(colors) == 1
    row_color = colors[0]
    row_count = sum(bool(np.any(interior[r] == row_color)) for r in range(interior.shape[0]))
    assert row_count > 0
    return row_color, row_count


def _draw_stair(
    out: np.ndarray,
    frame: tuple[int, int, int, int],
    direction: str,
    color: int,
    step_size: int,
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    if direction in ("left", "right"):
        size = min(step_size, inner_h)
        start_row = r0 + 1 + max(0, (inner_h - size) // 2)
        for offset in range(size):
            col = c1 + 1 + offset if direction == "right" else c0 - 1 - offset
            if not (0 <= col < w):
                continue
            top = start_row + offset if direction == "right" else start_row
            bottom = start_row + size - 1 if direction == "right" else start_row + size - 1 - offset
            for row in range(top, bottom + 1):
                if 0 <= row < h and out[row, col] == BG:
                    out[row, col] = color
        return

    size = min(step_size, inner_w)
    start_col = c0 + 1 + max(0, (inner_w - size) // 2)
    for offset in range(size):
        row = r1 + 1 + offset if direction == "down" else r0 - 1 - offset
        if not (0 <= row < h):
            continue
        left = start_col + offset if direction == "down" else start_col
        right = start_col + size - 1 if direction == "down" else start_col + size - 1 - offset
        for col in range(left, right + 1):
            if 0 <= col < w and out[row, col] == BG:
                out[row, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each maroon rectangular frame, count its colored interior rows, decode the
    yellow L-flag direction, and draw a centered outward staircase from the matching side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        row_color, step_size = _frame_row_color_and_count(interior)
        direction = _decode_flag(interior)
        _draw_stair(out, frame, direction, row_color, step_size)

    return out


def _choose_object_count(height: int, width: int) -> int:
    area = height * width
    if area < 420:
        return random.choice((1, 2, 2, 3))
    if area < 560:
        return random.choice((2, 2, 3, 3, 4))
    return random.choice((2, 3, 3, 4, 4))


def _footprint_extents(spec: dict) -> tuple[int, int, int, int]:
    frame_h = int(spec["frame_h"])
    frame_w = int(spec["frame_w"])
    step_size = int(spec["step_size"])
    direction = str(spec["direction"])

    dr0 = 0
    dc0 = 0
    dr1 = frame_h - 1
    dc1 = frame_w - 1

    if direction == "right":
        dc1 += step_size
    elif direction == "left":
        dc0 -= step_size
    elif direction == "down":
        dr1 += step_size
    else:
        dr0 -= step_size

    return dr0, dc0, dr1, dc1


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + OBJECT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + OBJECT_GAP)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + OBJECT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + OBJECT_GAP)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_frame(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    frame_h = int(spec["frame_h"])
    frame_w = int(spec["frame_w"])
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    color = int(spec["row_color"])
    row_indices = spec["row_indices"]
    flag_r = int(spec["flag_r"])
    flag_c = int(spec["flag_c"])
    direction = str(spec["direction"])

    bottom = top + frame_h - 1
    right = left + frame_w - 1

    grid[top : bottom + 1, left] = MAROON
    grid[top : bottom + 1, right] = MAROON
    grid[top, left : right + 1] = MAROON
    grid[bottom, left : right + 1] = MAROON

    for rr in row_indices:
        grid[top + 1 + int(rr), left + 1 : left + 1 + inner_w] = color

    for dr, dc in FLAG_SHAPES[direction]:
        grid[top + 1 + flag_r + dr, left + 1 + flag_c + dc] = YELLOW


def _make_spec(direction: str, color: int) -> dict:
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    step_size = random.randint(1, min(inner_h, inner_w))
    row_indices = sorted(random.sample(range(inner_h), step_size))
    flag_r = random.randint(0, inner_h - 2)
    flag_c = random.randint(0, inner_w - 2)
    return {
        "direction": direction,
        "row_color": color,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "step_size": step_size,
        "row_indices": row_indices,
        "flag_r": flag_r,
        "flag_c": flag_c,
    }


def _place_specs(height: int, width: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)

    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            (_footprint_extents(specs[idx])[2] - _footprint_extents(specs[idx])[0] + 1)
            * (_footprint_extents(specs[idx])[3] - _footprint_extents(specs[idx])[1] + 1)
        ),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        dr0, dc0, dr1, dc1 = _footprint_extents(spec)
        min_top = max(0, -dr0)
        max_top = height - 1 - dr1
        min_left = max(0, -dc0)
        max_left = width - 1 - dc1
        if min_top > max_top or min_left > max_left:
            return None

        for _ in range(PLACEMENT_TRIES):
            top = random.randint(min_top, max_top)
            left = random.randint(min_left, max_left)
            rr0 = top + dr0
            cc0 = left + dc0
            rr1 = top + dr1
            cc1 = left + dc1
            if _can_place(occupied, rr0, cc0, rr1, cc1):
                _reserve(occupied, rr0, cc0, rr1, cc1)
                placements[idx] = (top, left)
                break
        else:
            return None

    assert all(p is not None for p in placements)
    return [(int(r), int(c)) for r, c in placements]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    object_count = _choose_object_count(height, width)

    directions = random.sample(DIRECTIONS, k=object_count)
    colors = random.sample(ROW_COLORS, k=object_count)
    specs = [_make_spec(direction, color) for direction, color in zip(directions, colors)]

    placements = _place_specs(height, width, specs)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for spec, (top, left) in zip(specs, placements):
        _paint_frame(grid, spec, top, left)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
