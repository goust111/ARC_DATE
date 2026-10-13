import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- shared helpers ---------------


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_cell_count(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height <= 0 or width <= 0:
        return 0
    if height == 1:
        return width
    if width == 1:
        return height
    return 2 * height + 2 * width - 4


def _is_rectangular_frame(coords: np.ndarray) -> tuple[int, int, int, int]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 2 and width >= 2, "frame must have at least 2 cells per side"
    assert int(coords.shape[0]) == _frame_cell_count(bbox), "component must be a full rectangle border"

    coord_set = {tuple(map(int, rc)) for rc in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r == r0 or r == r1 or c == c0 or c == c1
            if on_border:
                assert (r, c) in coord_set, "missing border cell in frame"
            else:
                assert (r, c) not in coord_set, "frame interior must stay empty"
    return bbox


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _nested_bboxes(frame_components: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    boxes = [_is_rectangular_frame(coords) for coords in frame_components]
    boxes.sort(key=lambda b: (b[0], b[1], -b[2], -b[3]))
    for outer, inner in zip(boxes, boxes[1:]):
        assert outer[0] < inner[0] and outer[1] < inner[1], "frames must be strictly nested"
        assert outer[2] > inner[2] and outer[3] > inner[3], "frames must be strictly nested"
    return boxes


def _palette_info(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    palette_mask = (grid != BLACK) & (grid != GREY)
    components = _connected_components(palette_mask)
    assert len(components) == 1, "expected one palette strip"
    coords = components[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    assert c0 == c1, "palette must be one cell wide"
    assert int(coords.shape[0]) == r1 - r0 + 1, "palette cells must be vertically contiguous"
    for r in range(r0, r1 + 1):
        assert palette_mask[r, c0], "palette rows must be consecutive"
    colors = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
    return colors, coords


def _expected_output(
    grid: np.ndarray,
    frame_boxes: list[tuple[int, int, int, int]],
    palette_colors: list[int],
    palette_coords: np.ndarray,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[palette_coords[:, 0], palette_coords[:, 1]] = BLACK
    for bbox, color in zip(frame_boxes, palette_colors):
        _paint_frame(out, bbox, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested grey frame by its depth using the right-side vertical palette.

    Input structure:
    - The background is black.
    - One group of one-cell-thick grey rectangular frames is nested from outermost
      to innermost, with at least one background cell between consecutive frames.
    - A separate one-column palette strip sits in the upper-right area; its pixels
      list one color per row from top to bottom.

    Rule:
    - Read the palette strip from top to bottom.
    - Sort the grey frame components by containment depth from outside to inside.
    - Recolor the first, second, ..., K-th frame with the first, second, ..., K-th
      palette colors respectively, then erase the palette strip to black.

    Key insight:
    - The frames are all visually identical in the input, so the only way to map
      colors onto them is to recover the one-dimensional depth order from the
      two-dimensional nesting relation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette_colors, palette_coords = _palette_info(g)
    frame_components = _connected_components(g == GREY)
    assert frame_components, "expected at least one grey frame"
    frame_boxes = _nested_bboxes(frame_components)
    assert len(frame_boxes) == len(palette_colors), "palette length must match frame count"

    return _expected_output(g, frame_boxes, palette_colors, palette_coords)


# --------------- generator ---------------

MIN_LAYERS = 2
MAX_LAYERS = 6
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400

MAX_TOTAL_INSET = 12
MAX_CANVAS_EXTRA = 10
MAX_TOP_MARGIN = 2
MAX_RIGHT_MARGIN = 2


def _step_sequence(layer_count: int) -> list[int] | None:
    """Return insets between consecutive frames; each step is frame-thickness + gap."""
    if layer_count == 1:
        return []

    step_count = layer_count - 1
    min_total = 2 * step_count
    max_total = min(MAX_TOTAL_INSET, 4 * step_count)
    if min_total > max_total:
        return None

    total = random.randint(min_total, max_total)
    steps = [2] * step_count
    remaining = total - min_total
    while remaining > 0:
        idx = random.randrange(step_count)
        if steps[idx] < 4:
            steps[idx] += 1
            remaining -= 1
    random.shuffle(steps)
    return steps


def _frame_boxes_from_spec(
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    steps: list[int],
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    inset = 0
    for i in range(len(steps) + 1):
        r0 = top + inset
        c0 = left + inset
        r1 = top + outer_h - 1 - inset
        c1 = left + outer_w - 1 - inset
        boxes.append((r0, c0, r1, c1))
        if i < len(steps):
            inset += steps[i]
    return boxes


def _sample_spec() -> tuple[np.ndarray, np.ndarray] | None:
    layer_count = random.randint(MIN_LAYERS, MAX_LAYERS)
    steps = _step_sequence(layer_count)
    if steps is None:
        return None

    total_inset = sum(steps)
    min_outer_h = 2 + 2 * total_inset
    min_outer_w = 2 + 2 * total_inset
    min_h = max(min_outer_h, layer_count)
    min_w = min_outer_w + 2
    if min_h > MAX_GRID_SIZE or min_w > MAX_GRID_SIZE:
        return None

    canvas_h_max = min(MAX_GRID_SIZE, min_h + MAX_CANVAS_EXTRA)
    canvas_w_max = min(MAX_GRID_SIZE, min_w + MAX_CANVAS_EXTRA)
    height = random.randint(min_h, canvas_h_max)
    width = random.randint(min_w, canvas_w_max)

    outer_h = random.randint(min_outer_h, height)
    outer_w = random.randint(min_outer_w, width - 2)

    max_top = height - layer_count
    if max_top < 0:
        return None
    palette_top = random.randint(0, min(MAX_TOP_MARGIN, max_top))

    max_right = min(MAX_RIGHT_MARGIN, width - outer_w - 2)
    if max_right < 0:
        return None
    right_margin = random.randint(0, max_right)
    palette_col = width - 1 - right_margin

    max_left = palette_col - outer_w - 1
    if max_left < 0:
        return None
    frame_left = random.randint(0, max_left)
    frame_top = random.randint(0, height - outer_h)

    frame_boxes = _frame_boxes_from_spec(frame_top, frame_left, outer_h, outer_w, steps)
    palette_colors = random.sample(COLOR_POOL, layer_count)

    inp = np.full((height, width), BLACK, dtype=int)
    for bbox in frame_boxes:
        _paint_frame(inp, bbox, GREY)
    for idx, color in enumerate(palette_colors):
        inp[palette_top + idx, palette_col] = int(color)

    expected = np.full((height, width), BLACK, dtype=int)
    for bbox, color in zip(frame_boxes, palette_colors):
        _paint_frame(expected, bbox, color)

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sampled = _sample_spec()
        if sampled is None:
            continue
        inp, expected = sampled
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid nested-frame input")
