import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

TOP_WALL_COLOR = RED
BOTTOM_WALL_COLOR = BLUE
LEFT_WALL_COLOR = GREEN
RIGHT_WALL_COLOR = YELLOW
WALL_COLORS = (TOP_WALL_COLOR, BOTTOM_WALL_COLOR, LEFT_WALL_COLOR, RIGHT_WALL_COLOR)

GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (14, 20)
EXTRA_OBJECTS_CHOICES = (0, 1, 2)
EXTRA_OBJECTS_WEIGHTS = (1, 2, 2)
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300
PROTECTED_MARGIN = 2
PADDING = 1

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.flat)
    return max(counts.items(), key=lambda item: item[1])[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior shape onto the wall with the matching color.

    Red shapes project upward to the top wall, blue downward to the bottom wall,
    green leftward to the left wall, and yellow rightward to the right wall.
    Each occupied row/column of a shape paints an orange cell on its wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    out = g.copy()
    top_color = _majority_color(g[0, 1:-1])
    bottom_color = _majority_color(g[-1, 1:-1])
    left_color = _majority_color(g[1:-1, 0])
    right_color = _majority_color(g[1:-1, -1])

    inner = g[1:-1, 1:-1]

    top_cols = np.where(np.any(inner == top_color, axis=0))[0] + 1
    bottom_cols = np.where(np.any(inner == bottom_color, axis=0))[0] + 1
    left_rows = np.where(np.any(inner == left_color, axis=1))[0] + 1
    right_rows = np.where(np.any(inner == right_color, axis=1))[0] + 1

    out[0, top_cols] = ORANGE
    out[-1, bottom_cols] = ORANGE
    out[left_rows, 0] = ORANGE
    out[right_rows, -1] = ORANGE
    return out


def _rectangle_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(2, 5)
    return np.ones((h, w), dtype=bool)


def _l_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    thickness = 1 if min(h, w) <= 3 else rng.randint(1, 2)
    shape = np.zeros((h, w), dtype=bool)
    shape[:, :thickness] = True
    shape[-thickness:, :] = True
    return shape


def _t_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    shape = np.zeros((h, w), dtype=bool)
    shape[0, :] = True
    stem_left = (w - 1) // 2
    shape[:, stem_left : stem_left + 1] = True
    return shape


def _step_shape() -> np.ndarray:
    return np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)


def _plus_shape() -> np.ndarray:
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _polyomino_shape(rng: random.Random) -> np.ndarray:
    target_area = rng.randint(4, 8)
    cells = {(0, 0)}
    while len(cells) < target_area:
        candidates = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                trial = cells | {(nr, nc)}
                rows = [pt[0] for pt in trial]
                cols = [pt[1] for pt in trial]
                if max(rows) - min(rows) + 1 > 5 or max(cols) - min(cols) + 1 > 5:
                    continue
                candidates.append((nr, nc))
        if not candidates:
            break
        cells.add(rng.choice(candidates))

    rows = [pt[0] for pt in cells]
    cols = [pt[1] for pt in cells]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    shape = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        shape[r - r0, c - c0] = True
    return shape


def _transform_shape(shape: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(shape, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, rng.randrange(4))
    return out.astype(bool)


def _random_shape(rng: random.Random) -> np.ndarray:
    family = rng.choices(
        ("rectangle", "l", "t", "step", "plus", "polyomino"),
        weights=(2, 2, 2, 1, 1, 3),
        k=1,
    )[0]
    if family == "rectangle":
        shape = _rectangle_shape(rng)
    elif family == "l":
        shape = _l_shape(rng)
    elif family == "t":
        shape = _t_shape(rng)
    elif family == "step":
        shape = _step_shape()
    elif family == "plus":
        shape = _plus_shape()
    else:
        shape = _polyomino_shape(rng)
    shape = _transform_shape(shape, rng)
    if shape.shape[0] == 1 or shape.shape[1] == 1:
        return _rectangle_shape(rng)
    return shape


def _projection_interval(color: int, top: int, left: int, shape: np.ndarray) -> Tuple[int, int]:
    if color in (TOP_WALL_COLOR, BOTTOM_WALL_COLOR):
        return left, left + shape.shape[1] - 1
    return top, top + shape.shape[0] - 1


def _intervals_are_separated(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[1] + 1 < b[0] or b[1] + 1 < a[0]


def _can_place(
    blocked: np.ndarray,
    placed: List[Dict[str, int]],
    color: int,
    top: int,
    left: int,
    shape: np.ndarray,
) -> bool:
    h, w = blocked.shape
    sh, sw = shape.shape
    pad_r0 = max(0, top - PADDING)
    pad_c0 = max(0, left - PADDING)
    pad_r1 = min(h, top + sh + PADDING)
    pad_c1 = min(w, left + sw + PADDING)
    if blocked[pad_r0:pad_r1, pad_c0:pad_c1].any():
        return False

    current_interval = _projection_interval(color, top, left, shape)
    same_color = [item for item in placed if item["color"] == color]
    if len(same_color) >= 1:
        separated = all(
            _intervals_are_separated(current_interval, (item["start"], item["end"]))
            for item in same_color
        )
        if not separated:
            return False
    return True


def _paint_border(grid: np.ndarray) -> None:
    grid[0, :] = TOP_WALL_COLOR
    grid[-1, :] = BOTTOM_WALL_COLOR
    grid[1:-1, 0] = LEFT_WALL_COLOR
    grid[1:-1, -1] = RIGHT_WALL_COLOR


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    _paint_border(grid)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    colors = list(WALL_COLORS)
    extra_count = rng.choices(EXTRA_OBJECTS_CHOICES, weights=EXTRA_OBJECTS_WEIGHTS, k=1)[0]
    colors.extend(rng.choices(WALL_COLORS, k=extra_count))
    rng.shuffle(colors)

    placed: List[Dict[str, int]] = []
    for color in colors:
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _random_shape(rng)
            sh, sw = shape.shape
            min_top = PROTECTED_MARGIN
            max_top = h - PROTECTED_MARGIN - sh
            min_left = PROTECTED_MARGIN
            max_left = w - PROTECTED_MARGIN - sw
            if min_top > max_top or min_left > max_left:
                return None

            top = rng.randint(min_top, max_top)
            left = rng.randint(min_left, max_left)
            if not _can_place(blocked, placed, color, top, left, shape):
                continue

            rows, cols = np.where(shape)
            grid[top + rows, left + cols] = color

            pad_r0 = max(0, top - PADDING)
            pad_c0 = max(0, left - PADDING)
            pad_r1 = min(h, top + sh + PADDING)
            pad_c1 = min(w, left + sw + PADDING)
            blocked[pad_r0:pad_r1, pad_c0:pad_c1] = True

            start, end = _projection_interval(color, top, left, shape)
            placed.append({"color": color, "start": start, "end": end})
            success = True
            break
        if not success:
            return None

    out = main(grid)
    if not np.any(out == ORANGE):
        return None

    inner = grid[1:-1, 1:-1]
    if not all(np.any(inner == color) for color in WALL_COLORS):
        return None

    if not np.any(out[0, 1:-1] == ORANGE):
        return None
    if not np.any(out[-1, 1:-1] == ORANGE):
        return None
    if not np.any(out[1:-1, 0] == ORANGE):
        return None
    if not np.any(out[1:-1, -1] == ORANGE):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
