import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GEN_ATTEMPTS = 500
HEIGHT_RANGE = (8, 10)
WIDTH_RANGE = (12, 16)
SPAN_RANGE = (3, 4)
SHAPE_HEIGHT_RANGE = (5, 7)

# These templates all satisfy:
# - input has no enclosed black region
# - after row-r right shift, exactly one intended chamber becomes enclosed
CLOSED_TEMPLATES = [
    (-1, 0, -2, -2),
    (-2, 1, -1, -2),
    (0, 0, -3, -2),
    (-1, 2, 0, -2),
]

# Near-miss distractors that stay open even after the shear.
OPEN_TEMPLATES = [
    (-2, -2, -2, -2),
    (-1, -1, 2, -2),
    (0, -2, 0, 0),
    (-2, 2, 2, 2),
    (-2, 0, 2, 1),
]


def _shear_down_right(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        out[r, r : r + w] = grid[r]
    return out


def _exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if grid[r, c] == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return exterior


def _enclosed_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == BLACK) & ~_exterior_background(grid)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        seen[rr, cc] = True
        queue: deque[tuple[int, int]] = deque([(rr, cc)])
        while queue:
            r0, c0 = queue.popleft()
            for nr, nc in ((r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear the red wall layout down-right, then fill the newly enclosed background chambers with blue.

    Input structure:
    - The grid is a black background with red single-pixel wall segments.
    - Each row keeps its internal pattern; the puzzle information is in how red segments
      on different rows almost line up but do not yet seal regions in the original grid.
    - Blue is absent from the input and appears only in the output fill.

    Rule:
    - Create a black output canvas of the same height and width W + H - 1.
    - Copy input row r unchanged into output row r, starting at column r.
    - In the sheared grid, flood-fill from the outer border through black cells.
    - Recolor every black cell not reachable from that border flood-fill to blue.
      Leave the red walls and the exterior black background unchanged.

    Key insight:
    - The original grid has no enclosed background region; the enclosure appears only
      after the row-dependent right shift changes how wall fragments align across rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sheared = _shear_down_right(g)
    enclosed = _enclosed_mask(sheared)

    out = sheared.copy()
    out[enclosed] = BLUE
    return out


def _shape_bounds(width: int, span: int, offsets: tuple[int, int, int, int]) -> tuple[int, int] | None:
    top_left, top_right, bottom_left, bottom_right = offsets
    left_min = max(0, -top_left, -bottom_left)
    left_max = min(
        width - 1 - span,
        width - 1 - (span + top_right),
        width - 1 - (span + bottom_right),
    )
    if left_min > left_max:
        return None
    return int(left_min), int(left_max)


def _draw_shape(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    span: int,
    offsets: tuple[int, int, int, int],
) -> dict:
    top_left, top_right, bottom_left, bottom_right = offsets
    right = left + span

    grid[top + 1 : bottom, left] = RED
    grid[top + 1 : bottom, right] = RED
    grid[top, left + top_left : right + top_right + 1] = RED
    grid[bottom, left + bottom_left : right + bottom_right + 1] = RED

    mid_row = (top + bottom) // 2
    seed_col = left + mid_row + max(1, span // 2)
    min_col = min(left, left + top_left, left + bottom_left)
    max_col = max(right, right + top_right, right + bottom_right)
    return {
        "bbox": (top, min_col, bottom, max_col),
        "seed": (mid_row, seed_col),
    }


def _bbox_free(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    return not blocked[r0 : r1 + 1, c0 : c1 + 1].any()


def _block_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(h - 1, r1 + pad)
    c1 = min(w - 1, c1 + pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    templates: list[tuple[int, int, int, int]],
    kind: str,
    attempts: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(attempts):
        offsets = random.choice(templates)
        span = random.randint(*SPAN_RANGE)
        shape_height = random.randint(SHAPE_HEIGHT_RANGE[0], min(SHAPE_HEIGHT_RANGE[1], h))
        top = random.randint(0, h - shape_height)
        bottom = top + shape_height - 1

        bounds = _shape_bounds(w, span, offsets)
        if bounds is None:
            continue
        left_min, left_max = bounds
        left = random.randint(left_min, left_max)

        probe = np.array(grid, copy=True)
        shape = _draw_shape(probe, top, bottom, left, span, offsets)
        if not _bbox_free(blocked, shape["bbox"]):
            continue

        _draw_shape(grid, top, bottom, left, span, offsets)
        shape["kind"] = kind
        shape["offsets"] = offsets
        shape["span"] = span
        _block_bbox(blocked, shape["bbox"])
        return shape
    return None


def _place_noise_segment(grid: np.ndarray, blocked: np.ndarray, attempts: int = 60) -> bool:
    h, w = grid.shape
    for _ in range(attempts):
        length = random.randint(1, 3)
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - length)
        bbox = (r, c0, r, c0 + length - 1)
        if not _bbox_free(blocked, bbox):
            continue
        grid[r, c0 : c0 + length] = RED
        _block_bbox(blocked, bbox, pad=0)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    closed_shape = _place_shape(grid, blocked, CLOSED_TEMPLATES, "closed", attempts=250)
    if closed_shape is None:
        return None

    open_shape = _place_shape(grid, blocked, OPEN_TEMPLATES, "open", attempts=250)
    if open_shape is None:
        return None

    for _ in range(random.randint(0, 2)):
        _place_noise_segment(grid, blocked)

    if np.any(_enclosed_mask(grid)):
        return None

    sheared = _shear_down_right(grid)
    enclosed = _enclosed_mask(sheared)
    if enclosed.sum() < 6:
        return None
    if _component_count(enclosed) != 1:
        return None

    closed_r, closed_c = closed_shape["seed"]
    open_r, open_c = open_shape["seed"]
    if not enclosed[closed_r, closed_c]:
        return None
    if enclosed[open_r, open_c]:
        return None

    out = main(grid)
    if not np.array_equal(out == BLUE, enclosed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shear-then-fill input")
