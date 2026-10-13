import random

import numpy as np


BLACK = 0
COLOR_COUNT = 10


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lo, hi in edges:
        assert lo != hi
        if hi not in above[lo]:
            above[lo].add(hi)
            indegree[hi] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "layer constraints must imply one order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for hi in above[color]:
            indegree[hi] -= 1
    return order


def _concentric(order: list[int]) -> np.ndarray:
    size = 2 * len(order) - 1
    out = np.full((size, size), BLACK, dtype=int)
    for i, color in enumerate(order):
        r0 = c0 = i
        r1 = c1 = size - 1 - i
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert overlaid rectangular color frames into concentric rings ordered by depth.

    Input structure:
    - The background is Black.
    - Each non-Black color belongs to one one-cell-thick axis-aligned rectangular frame.
    - Frames overlap, and at shared frame cells only the top frame color remains visible.

    Rule:
    - Reconstruct each color's complete frame as the border of the bounding box of
      that color's visible pixels.
    - At every cell covered by two or more reconstructed frames, the visible color
      is above all other reconstructed frames covering that cell.
    - Infer the unique bottom-to-top frame order from these overlap constraints.
    - Draw a new smallest square canvas of touching one-pixel concentric rings:
      the outer ring uses the bottom color, and each next inner ring uses the
      next higher color, ending with the top color in the center.

    Key insight:
    The input geometry only reveals stacking order; the output discards the
    original positions and turns the hidden layer order into nested rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert colors, "input must contain at least one foreground frame"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None
        r0, c0, r1, c1 = box
        assert r1 > r0 and c1 > c0, "frames must have nonzero height and width"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show a covering frame color"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    if len(colors) > 1:
        assert edges, "overlapping frames are needed to determine layer order"
    order = _unique_layer_order(colors, edges)
    return _concentric(order)


# --------------- generator ---------------

FRAME_COUNT_RANGE = (2, 7)
GRID_SIZE_RANGE = (12, 30)
FRAME_SIDE_RANGE = (5, 15)
STEP_RANGE = (2, 4)
GENERATION_ATTEMPTS = 500


def _shifted_boxes(
    n: int,
    rows: int,
    cols: int,
    frame_h: int,
    frame_w: int,
    step_r: int,
    step_c: int,
    sign_r: int,
    sign_c: int,
) -> list[tuple[int, int, int, int]]:
    span_h = frame_h + step_r * (n - 1)
    span_w = frame_w + step_c * (n - 1)
    top = random.randint(0, rows - span_h)
    left = random.randint(0, cols - span_w)

    if sign_r < 0:
        first_r = top + step_r * (n - 1)
    else:
        first_r = top
    if sign_c < 0:
        first_c = left + step_c * (n - 1)
    else:
        first_c = left

    boxes = []
    for i in range(n):
        r0 = first_r + sign_r * step_r * i
        c0 = first_c + sign_c * step_c * i
        boxes.append((r0, c0, r0 + frame_h - 1, c0 + frame_w - 1))
    return boxes


def _render_frames(
    shape: tuple[int, int], layer_order: list[int], boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for color, box in zip(layer_order, boxes):
        _paint_frame(grid, box, color)
    return grid


def _visible_bboxes_match(
    grid: np.ndarray, layer_order: list[int], boxes: list[tuple[int, int, int, int]]
) -> bool:
    for color, box in zip(layer_order, boxes):
        if _bbox(grid == color) != box:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*FRAME_COUNT_RANGE)
    colors = random.sample(range(1, COLOR_COUNT), n)
    random.shuffle(colors)

    step_r = random.randint(*STEP_RANGE)
    step_c = random.randint(*STEP_RANGE)
    min_h = max(FRAME_SIDE_RANGE[0], step_r + 2)
    min_w = max(FRAME_SIDE_RANGE[0], step_c + 2)
    max_h = min(FRAME_SIDE_RANGE[1], 30 - step_r * (n - 1))
    max_w = min(FRAME_SIDE_RANGE[1], 30 - step_c * (n - 1))
    if min_h > max_h or min_w > max_w:
        return None

    frame_h = random.randint(min_h, max_h)
    frame_w = random.randint(min_w, max_w)
    span_h = frame_h + step_r * (n - 1)
    span_w = frame_w + step_c * (n - 1)
    min_rows = max(GRID_SIZE_RANGE[0], span_h)
    min_cols = max(GRID_SIZE_RANGE[0], span_w)
    if min_rows > GRID_SIZE_RANGE[1] or min_cols > GRID_SIZE_RANGE[1]:
        return None

    rows = random.randint(min_rows, GRID_SIZE_RANGE[1])
    cols = random.randint(min_cols, GRID_SIZE_RANGE[1])
    sign_r = random.choice((-1, 1))
    sign_c = random.choice((-1, 1))
    boxes = _shifted_boxes(n, rows, cols, frame_h, frame_w, step_r, step_c, sign_r, sign_c)
    grid = _render_frames((rows, cols), colors, boxes)

    if not _visible_bboxes_match(grid, colors, boxes):
        return None

    expected = _concentric(colors)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
