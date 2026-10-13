import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GRAY = 5
DIRS = ("up", "right", "down", "left")
DIR_TO_INDEX = {name: i for i, name in enumerate(DIRS)}


@dataclass
class Frame:
    color: int
    h: int
    w: int
    gap: str
    area: int


def _border_cells(top, left, h, w):
    cells = set()
    bottom = top + h - 1
    right = left + w - 1
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _gap_position(top, left, h, w, gap):
    if gap == "up":
        return top, left + w // 2
    if gap == "down":
        return top + h - 1, left + w // 2
    if gap == "left":
        return top + h // 2, left
    return top + h // 2, left + w - 1


def _draw_frame(grid, top, left, h, w, color, gap):
    bottom = top + h - 1
    right = left + w - 1
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color
    gr, gc = _gap_position(top, left, h, w, gap)
    grid[gr, gc] = BLACK


def _rotated_dims(h, w, source_gap, target_gap):
    turns = (DIR_TO_INDEX[target_gap] - DIR_TO_INDEX[source_gap]) % 4
    if turns % 2:
        return w, h
    return h, w


def _extract_canvas(grid):
    gray = np.argwhere(grid == GRAY)
    if len(gray) < 4:
        raise ValueError("target canvas needs four gray corner pixels")
    r0, c0 = gray.min(axis=0)
    r1, c1 = gray.max(axis=0)
    if r1 - r0 != c1 - c0:
        raise ValueError("gray target markers must form a square")
    return int(r0), int(c0), int(r1 - r0 + 1)


def _extract_frames(grid):
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GRAY):
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        expected = _border_cells(r0, c0, h, w)
        actual = {(int(r), int(c)) for r, c in coords}
        missing = sorted(expected - actual)
        if not missing:
            raise ValueError("frame has no detectable gap")
        mr, mc = missing[0]
        if mr == r0:
            gap = "up"
        elif mr == r1:
            gap = "down"
        elif mc == c0:
            gap = "left"
        elif mc == c1:
            gap = "right"
        else:
            raise ValueError("frame gap is not on the border")
        frames.append(Frame(color=color, h=h, w=w, gap=gap, area=h * w))
    frames.sort(key=lambda f: f.area, reverse=True)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Nest all broken rectangular frames in the gray-marked target canvas.

    Input structure:
    The input is a black grid containing several one-pixel-wide hollow rectangular
    frames, each in a unique non-black, non-gray color. Each frame has exactly one
    missing border cell centered on one of its four sides. A square target canvas
    is marked near the lower-right by four gray corner pixels, with black interior.

    Rule:
    Detect every colored frame, infer its outer rectangle and the side containing
    its gap, then sort frames by outer rectangle area from largest to smallest. The
    largest frame keeps its observed gap direction, which becomes the standard
    direction. Each smaller frame is rotated by a multiple of 90 degrees so its gap
    points in that same direction. Return only the target canvas contents: gray
    corner marks plus the rotated frames drawn from outside to inside, with the
    outer frame centered in the canvas and every inner frame centered in the direct
    outer frame's interior.

    Key insight:
    The reconstruction is determined jointly by size order and by aligning all gap
    directions to the largest frame before drawing the centered nesting.
    """
    arr = np.array(grid, copy=False)
    r0, c0, side = _extract_canvas(arr)
    frames = _extract_frames(arr)
    out = np.zeros((side, side), dtype=arr.dtype)
    out[0, 0] = GRAY
    out[0, side - 1] = GRAY
    out[side - 1, 0] = GRAY
    out[side - 1, side - 1] = GRAY
    if not frames:
        return out

    standard = frames[0].gap
    placed = []
    for frame in frames:
        h, w = _rotated_dims(frame.h, frame.w, frame.gap, standard)
        placed.append((frame.color, h, w))

    prev = None
    for color, h, w in placed:
        if prev is None:
            top = (side - h) // 2
            left = (side - w) // 2
        else:
            pt, pl, ph, pw = prev
            top = pt + 1 + (ph - 2 - h) // 2
            left = pl + 1 + (pw - 2 - w) // 2
        _draw_frame(out, top, left, h, w, color, standard)
        prev = (top, left, h, w)

    out[0, 0] = GRAY
    out[0, side - 1] = GRAY
    out[side - 1, 0] = GRAY
    out[side - 1, side - 1] = GRAY
    return out


def _make_nested_dims(count):
    if count == 3:
        inner = random.choice([(5, 7), (7, 5)])
        return [(inner[0] + 8, inner[1] + 8),
                (inner[0] + 4, inner[1] + 4),
                inner]
    inner = random.choice([(5, 7), (7, 5), (7, 9), (9, 7)])
    grow = random.choice([4, 6])
    return [(inner[0] + grow, inner[1] + grow), inner]


def _intersects(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_frame_randomly(grid, occupied, target_box, h, w, color, gap):
    rows, cols = grid.shape
    for _ in range(500):
        top = random.randrange(0, rows - h + 1)
        left = random.randrange(0, cols - w + 1)
        box = (top, left, top + h - 1, left + w - 1)
        if _intersects(box, target_box):
            continue
        padded = (max(0, top - 1), max(0, left - 1),
                  min(rows - 1, top + h), min(cols - 1, left + w))
        if any(_intersects(padded, old) for old in occupied):
            continue
        _draw_frame(grid, top, left, h, w, color, gap)
        occupied.append(padded)
        return True
    return False


def _try_generate():
    count = 3 if random.random() < 0.7 else 2
    final_dims = _make_nested_dims(count)
    outer_h, outer_w = final_dims[0]
    side = max(outer_h, outer_w) + 2
    rows = random.randint(max(side + min(outer_h, outer_w), 24), 30)
    cols = random.randint(max(side + min(outer_h, outer_w), 24), 30)
    target_top = rows - side
    target_left = cols - side
    target_box = (target_top, target_left, rows - 1, cols - 1)

    grid = np.zeros((rows, cols), dtype=np.int64)
    grid[target_top, target_left] = GRAY
    grid[target_top, cols - 1] = GRAY
    grid[rows - 1, target_left] = GRAY
    grid[rows - 1, cols - 1] = GRAY

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], count)
    standard = random.choice(DIRS)
    original = []
    for i, (h, w) in enumerate(final_dims):
        gap = standard if i == 0 else random.choice(DIRS)
        turns = (DIR_TO_INDEX[standard] - DIR_TO_INDEX[gap]) % 4
        oh, ow = (w, h) if turns % 2 else (h, w)
        original.append((colors[i], oh, ow, gap))

    occupied = []
    for color, h, w, gap in sorted(original, key=lambda item: item[1] * item[2], reverse=True):
        if not _place_frame_randomly(grid, occupied, target_box, h, w, color, gap):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(100):
        grid = _try_generate()
        if grid is not None:
            return grid
    # Deterministic fallback that still exercises rotation and nesting.
    grid = np.zeros((26, 26), dtype=np.int64)
    side = 17
    top = 26 - side
    left = 26 - side
    grid[top, left] = GRAY
    grid[top, 25] = GRAY
    grid[25, left] = GRAY
    grid[25, 25] = GRAY
    _draw_frame(grid, 1, 1, 13, 15, 2, "right")
    _draw_frame(grid, 16, 1, 11, 9, 4, "up")
    _draw_frame(grid, 1, 18, 5, 7, 7, "left")
    return grid
