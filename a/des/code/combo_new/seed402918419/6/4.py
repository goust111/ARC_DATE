from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
MIN_LAYERS = 2
MAX_LAYERS = 5
MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _same_color_component(
    grid: np.ndarray,
    start: tuple[int, int],
    seen: np.ndarray,
) -> list[tuple[int, int]]:
    color = int(grid[start])
    h, w = grid.shape
    stack = [start]
    seen[start] = True
    cells: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return cells


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_pixels(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return border


def _is_hollow_frame(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    return set(cells) == _border_pixels((r0, c0, r1, c1))


def _frame_height(frame: dict) -> int:
    r0, _c0, r1, _c1 = frame["bbox"]
    return int(r1 - r0 + 1)


def _strictly_contains(outer: dict, inner: dict) -> bool:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    return or0 + 2 <= ir0 and oc0 + 2 <= ic0 and ir1 <= or1 - 2 and ic1 <= oc1 - 2


def _extract_frames_and_marker(grid: np.ndarray) -> tuple[list[dict], tuple[int, int] | None]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []
    extras: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BG or seen[r, c]:
                continue
            cells = _same_color_component(grid, (r, c), seen)
            if _is_hollow_frame(cells):
                frames.append(
                    {
                        "color": int(grid[r, c]),
                        "bbox": _bbox(cells),
                        "pixels": set(cells),
                    }
                )
            else:
                extras.extend(cells)

    assert MIN_LAYERS <= len(frames) <= MAX_LAYERS, "expected 2-5 nested rectangular frames"
    assert len({frame["color"] for frame in frames}) == len(frames), "frame colors must be distinct"
    assert len(extras) in (0, 1), "expected at most one extra marker pixel"
    return frames, (extras[0] if extras else None)


def _ordered_frames(frames: list[dict]) -> list[dict]:
    depths = []
    for i, frame in enumerate(frames):
        depth = 0
        for j, other in enumerate(frames):
            if i != j and _strictly_contains(other, frame):
                depth += 1
        depths.append(depth)

    assert sorted(depths) == list(range(len(frames))), "frames must form one strict nesting chain"
    ordered = [frames[i] for i in sorted(range(len(frames)), key=lambda idx: depths[idx])]

    for i, outer in enumerate(ordered):
        for inner in ordered[i + 1 :]:
            assert _strictly_contains(outer, inner), "every deeper frame must lie inside each outer frame"

    return ordered


def _render_output(ordered: list[dict], marker_color: int | None) -> np.ndarray:
    height = _frame_height(ordered[0])
    out = np.empty((height, len(ordered)), dtype=int)
    for col, frame in enumerate(ordered):
        out[:, col] = int(frame["color"])
    if marker_color is not None:
        out[height // 2, -1] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the color order of nested rectangular frames and redraw it as vertical bars.

    Input structure:
    - The background is black.
    - A single structure of 2-5 one-pixel-thick hollow rectangular frames is
      present, with a different color on each frame.
    - Consecutive frames are strictly nested with at least one background cell
      between their borders, and the innermost interior may contain one marker pixel.

    Rule:
    - Detect every monochromatic hollow rectangle and order them from outermost
      to innermost by containment.
    - Create an output whose height equals the outermost frame height and whose
      width equals the number of frames.
    - Fill column k with the color of the k-th frame in nesting order, and if
      the innermost interior contains a marker pixel, place that marker color at
      the vertical center of the last column.

    Key insight:
    - The 2D geometry only serves to encode a 1D containment sequence, so the
      solver must recover the nesting tree of the rectangular borders rather
      than rely on centering or color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, marker = _extract_frames_and_marker(g)
    ordered = _ordered_frames(frames)

    marker_color = None
    if marker is not None:
        mr, mc = marker
        ir0, ic0, ir1, ic1 = ordered[-1]["bbox"]
        assert ir0 < mr < ir1 and ic0 < mc < ic1, "marker must lie inside the innermost frame"
        marker_color = int(g[mr, mc])

    return _render_output(ordered, marker_color)


def _odd_values(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _min_box_size(layer_count: int) -> int:
    return 3 + 4 * (layer_count - 1)


def _choose_position(lo: int, hi: int, centered: int) -> tuple[int, bool]:
    options = list(range(lo, hi + 1))
    off_center = [value for value in options if value != centered]
    if off_center and random.random() < 0.8:
        return random.choice(off_center), True
    return random.choice(options), bool(off_center)


def _sample_child_bbox(
    parent_bbox: tuple[int, int, int, int],
    remaining_layers: int,
) -> tuple[tuple[int, int, int, int] | None, bool, bool]:
    pr0, pc0, pr1, pc1 = parent_bbox
    parent_h = pr1 - pr0 + 1
    parent_w = pc1 - pc0 + 1
    child_min = _min_box_size(remaining_layers)

    child_heights = _odd_values(child_min, parent_h - 4)
    child_widths = _odd_values(child_min, parent_w - 4)
    if not child_heights or not child_widths:
        return None, False, False

    child_h = random.choice(child_heights)
    child_w = random.choice(child_widths)
    row_lo = pr0 + 2
    row_hi = pr0 + parent_h - child_h - 2
    col_lo = pc0 + 2
    col_hi = pc0 + parent_w - child_w - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None, False, False

    centered_row = pr0 + (parent_h - child_h) // 2
    centered_col = pc0 + (parent_w - child_w) // 2
    child_r0, row_can_shift = _choose_position(row_lo, row_hi, centered_row)
    child_c0, col_can_shift = _choose_position(col_lo, col_hi, centered_col)
    child_bbox = (child_r0, child_c0, child_r0 + child_h - 1, child_c0 + child_w - 1)
    shifted = child_r0 != centered_row or child_c0 != centered_col
    return child_bbox, shifted, row_can_shift or col_can_shift


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _expected_output(frame_colors: list[int], outer_height: int, marker_present: bool) -> np.ndarray:
    out = np.empty((outer_height, len(frame_colors)), dtype=int)
    for col, color in enumerate(frame_colors):
        out[:, col] = int(color)
    if marker_present:
        out[outer_height // 2, -1] = YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(MIN_LAYERS, MAX_LAYERS)
    marker_present = bool(random.getrandbits(1))
    frame_colors = random.sample(FRAME_COLORS, layer_count)

    outer_min = _min_box_size(layer_count)
    outer_h = random.choice(_odd_values(outer_min, MAX_GRID_SIZE - 1))
    outer_w = random.choice(_odd_values(outer_min, MAX_GRID_SIZE - 1))

    extra_h = random.randint(0, min(6, MAX_GRID_SIZE - outer_h))
    extra_w = random.randint(0, min(6, MAX_GRID_SIZE - outer_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    grid_h = outer_h + extra_h
    grid_w = outer_w + extra_w

    bboxes = [(top, left, top + outer_h - 1, left + outer_w - 1)]
    shifted_any = False
    shift_possible = False
    for depth in range(1, layer_count):
        child_bbox, shifted, can_shift = _sample_child_bbox(bboxes[-1], layer_count - depth)
        if child_bbox is None:
            return None
        bboxes.append(child_bbox)
        shifted_any = shifted_any or shifted
        shift_possible = shift_possible or can_shift

    if shift_possible and not shifted_any:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, bbox in zip(frame_colors, bboxes):
        _draw_frame(grid, bbox, color)

    if marker_present:
        r0, c0, r1, c1 = bboxes[-1]
        marker_r = random.randint(r0 + 1, r1 - 1)
        marker_c = random.randint(c0 + 1, c1 - 1)
        grid[marker_r, marker_c] = YELLOW

    try:
        out = main(grid)
    except Exception:
        return None

    expected = _expected_output(frame_colors, outer_h, marker_present)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
