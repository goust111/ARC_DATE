from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, paste
import random


def _contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0, "expected a nonempty side run"
    lo = int(indices[0])
    hi = int(indices[-1])
    assert np.array_equal(indices, np.arange(lo, hi + 1)), "side run must be contiguous"
    return lo, hi


def _centered_span(source_lo: int, source_hi: int, length: int) -> tuple[int, int]:
    assert length > 0, "run length must be positive"
    center2 = int(source_lo) + int(source_hi)
    assert (center2 - (length - 1)) % 2 == 0, "shrunk run must stay centered on original run"
    lo = (center2 - (length - 1)) // 2
    hi = (center2 + (length - 1)) // 2
    assert hi - lo + 1 == length
    return int(lo), int(hi)


def _paint_row(out: np.ndarray, row: int, c0: int, c1: int, bg: int) -> None:
    if not 0 <= row < out.shape[0]:
        return
    for col in range(int(c0), int(c1) + 1):
        if 0 <= col < out.shape[1] and int(out[row, col]) == bg:
            out[row, col] = Color.YELLOW


def _paint_col(out: np.ndarray, col: int, r0: int, r1: int, bg: int) -> None:
    if not 0 <= col < out.shape[1]:
        return
    for row in range(int(r0), int(r1) + 1):
        if 0 <= row < out.shape[0] and int(out[row, col]) == bg:
            out[row, col] = Color.YELLOW


def _paint_shrinking_rows(out: np.ndarray, row: int, step: int, c0: int, c1: int, bg: int) -> None:
    source_length = int(c1) - int(c0) + 1
    assert source_length >= 3, "source cap row must be wide enough to shrink"
    distance = 1
    length = source_length - 2
    while length > 0:
        lo, hi = _centered_span(c0, c1, length)
        _paint_row(out, int(row) + int(step) * distance, lo, hi, bg)
        distance += 1
        length -= 2


def _paint_shrinking_cols(out: np.ndarray, col: int, step: int, r0: int, r1: int, bg: int) -> None:
    source_length = int(r1) - int(r0) + 1
    assert source_length >= 3, "source cap column must be tall enough to shrink"
    distance = 1
    length = source_length - 2
    while length > 0:
        lo, hi = _centered_span(r0, r1, length)
        _paint_col(out, int(col) + int(step) * distance, lo, hi, bg)
        distance += 1
        length -= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each teal rounded object outward with yellow shrinking cap runs.

    Input structure:
    - The background is black.
    - All foreground pixels share one non-yellow color and form 4-connected rounded objects.
    - Each object's top, bottom, left, and right side runs are contiguous and at least three pixels long.

    Rule:
    - Preserve the input pixels.
    - For each object side, continue the side run outside the object, shrinking its length by two pixels per step.
    - Paint every in-bounds background pixel in those shrinking continuations yellow.

    Key insight:
    - The yellow pixels are not an outline; they are the clipped extrapolated caps of each object's horizontal and vertical run profiles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected one foreground color"
    foreground = foreground_colors[0]
    assert foreground != Color.YELLOW, "yellow is reserved for the added pixels"

    objects = find_components(
        g,
        bg=bg,
        colors=[foreground],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert objects, "expected at least one foreground object"

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        mask = np.zeros(g.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        r0, c0, r1, c1 = obj["bbox"]

        top_c0, top_c1 = _contiguous_span(np.flatnonzero(mask[r0, :]))
        bottom_c0, bottom_c1 = _contiguous_span(np.flatnonzero(mask[r1, :]))
        left_r0, left_r1 = _contiguous_span(np.flatnonzero(mask[:, c0]))
        right_r0, right_r1 = _contiguous_span(np.flatnonzero(mask[:, c1]))

        _paint_shrinking_rows(out, r0, -1, top_c0, top_c1, bg)
        _paint_shrinking_rows(out, r1, 1, bottom_c0, bottom_c1, bg)
        _paint_shrinking_cols(out, c0, -1, left_r0, left_r1, bg)
        _paint_shrinking_cols(out, c1, 1, right_r0, right_r1, bg)

    return out

# --------------- generator (o5x) ---------------

# Canvas bounds.
_GEN_ROW_RANGE = (20, 29)
_GEN_COL_RANGE = (20, 29)

# Rounded-object geometry.
_GEN_RADIUS_RANGE = (1, 4)
_GEN_HEIGHT_RANGE = (6, 15)
_GEN_WIDTH_RANGE = (5, 17)
_GEN_MIN_SIDE_RUN = 3
_GEN_FILL_RATIO_RANGE = (0.76, 0.93)
_GEN_CAP_SHRINK = 2

# Placement and retry limits.
_GEN_BBOX_GAP = 2
_GEN_FIRST_PLACEMENT_TRIES = 30
_GEN_RETRY_LIMIT = 300

# Palette.
_GEN_FOREGROUND_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]


def _gen_centered_span(source_lo: int, source_hi: int, length: int) -> tuple[int, int]:
    center2 = int(source_lo) + int(source_hi)
    lo = (center2 - (int(length) - 1)) // 2
    hi = (center2 + (int(length) - 1)) // 2
    return lo, hi


def _gen_mark_row(mask: np.ndarray, row: int, c0: int, c1: int) -> bool:
    rows, cols = mask.shape
    clipped = not 0 <= row < rows or c0 < 0 or c1 >= cols
    if 0 <= row < rows:
        lo = max(0, int(c0))
        hi = min(cols - 1, int(c1))
        if lo <= hi:
            mask[int(row), lo : hi + 1] = True
    return clipped


def _gen_mark_col(mask: np.ndarray, col: int, r0: int, r1: int) -> bool:
    rows, cols = mask.shape
    clipped = not 0 <= col < cols or r0 < 0 or r1 >= rows
    if 0 <= col < cols:
        lo = max(0, int(r0))
        hi = min(rows - 1, int(r1))
        if lo <= hi:
            mask[lo : hi + 1, int(col)] = True
    return clipped


def _gen_paint_shrinking_rows(mask: np.ndarray, row: int, step: int, c0: int, c1: int) -> bool:
    source_length = int(c1) - int(c0) + 1
    clipped = False
    distance = 1
    length = source_length - _GEN_CAP_SHRINK
    while length > 0:
        lo, hi = _gen_centered_span(c0, c1, length)
        clipped |= _gen_mark_row(mask, int(row) + int(step) * distance, lo, hi)
        distance += 1
        length -= _GEN_CAP_SHRINK
    return clipped


def _gen_paint_shrinking_cols(mask: np.ndarray, col: int, step: int, r0: int, r1: int) -> bool:
    source_length = int(r1) - int(r0) + 1
    clipped = False
    distance = 1
    length = source_length - _GEN_CAP_SHRINK
    while length > 0:
        lo, hi = _gen_centered_span(r0, r1, length)
        clipped |= _gen_mark_col(mask, int(col) + int(step) * distance, lo, hi)
        distance += 1
        length -= _GEN_CAP_SHRINK
    return clipped


def _gen_rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    for r in range(int(height)):
        edge_distance = min(r, int(height) - 1 - r, int(radius))
        inset = int(radius) - edge_distance
        mask[r, inset : int(width) - inset] = True
    return mask


def _gen_shape_params_by_radius() -> dict[int, list[tuple[int, int]]]:
    by_radius: dict[int, list[tuple[int, int]]] = {}
    for radius in range(_GEN_RADIUS_RANGE[0], _GEN_RADIUS_RANGE[1] + 1):
        pairs: list[tuple[int, int]] = []
        min_side = 2 * radius + _GEN_MIN_SIDE_RUN
        for height in range(max(_GEN_HEIGHT_RANGE[0], min_side), _GEN_HEIGHT_RANGE[1] + 1):
            for width in range(max(_GEN_WIDTH_RANGE[0], min_side), _GEN_WIDTH_RANGE[1] + 1):
                if height - 2 * radius < _GEN_MIN_SIDE_RUN or width - 2 * radius < _GEN_MIN_SIDE_RUN:
                    continue
                area = int(_gen_rounded_mask(height, width, radius).sum())
                fill_ratio = area / float(height * width)
                if _GEN_FILL_RATIO_RANGE[0] <= fill_ratio <= _GEN_FILL_RATIO_RANGE[1]:
                    pairs.append((height, width))
        by_radius[radius] = pairs
    return by_radius


_GEN_SHAPE_PARAMS_BY_RADIUS = _gen_shape_params_by_radius()
_GEN_VALID_RADII = [radius for radius, pairs in _GEN_SHAPE_PARAMS_BY_RADIUS.items() if pairs]


def _gen_shape_params() -> tuple[int, int, int]:
    radius = random.choice(_GEN_VALID_RADII)
    height, width = random.choice(_GEN_SHAPE_PARAMS_BY_RADIUS[radius])
    return radius, height, width


def _gen_side_spans(obj_mask: np.ndarray, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    height, width = obj_mask.shape
    r1 = int(r0) + height - 1
    c1 = int(c0) + width - 1

    top_cols = np.flatnonzero(obj_mask[0])
    bottom_cols = np.flatnonzero(obj_mask[-1])
    left_rows = np.flatnonzero(obj_mask[:, 0])
    right_rows = np.flatnonzero(obj_mask[:, -1])

    top_c0, top_c1 = int(c0) + int(top_cols[0]), int(c0) + int(top_cols[-1])
    bottom_c0, bottom_c1 = int(c0) + int(bottom_cols[0]), int(c0) + int(bottom_cols[-1])
    left_r0, left_r1 = int(r0) + int(left_rows[0]), int(r0) + int(left_rows[-1])
    right_r0, right_r1 = int(r0) + int(right_rows[0]), int(r0) + int(right_rows[-1])
    return (top_c0, top_c1), (bottom_c0, bottom_c1), (left_r0, left_r1), (right_r0, right_r1)


def _gen_cap_mask(grid_shape: tuple[int, int], obj_mask: np.ndarray, r0: int, c0: int) -> tuple[np.ndarray, bool]:
    caps = np.zeros(grid_shape, dtype=bool)
    height, width = obj_mask.shape
    r1 = int(r0) + height - 1
    c1 = int(c0) + width - 1
    (top_c0, top_c1), (bottom_c0, bottom_c1), (left_r0, left_r1), (right_r0, right_r1) = _gen_side_spans(obj_mask, r0, c0)
    clipped = False
    clipped |= _gen_paint_shrinking_rows(caps, int(r0), -1, top_c0, top_c1)
    clipped |= _gen_paint_shrinking_rows(caps, r1, 1, bottom_c0, bottom_c1)
    clipped |= _gen_paint_shrinking_cols(caps, int(c0), -1, left_r0, left_r1)
    clipped |= _gen_paint_shrinking_cols(caps, c1, 1, right_r0, right_r1)
    return caps, clipped


def _gen_bbox(pos: tuple[int, int], mask: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pos
    h, w = mask.shape
    return int(r0), int(c0), int(r0) + h - 1, int(c0) + w - 1


def _gen_bboxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + _GEN_BBOX_GAP < br0
        or br1 + _GEN_BBOX_GAP < ar0
        or ac1 + _GEN_BBOX_GAP < bc0
        or bc1 + _GEN_BBOX_GAP < ac0
    )


def _gen_object_cells(grid_shape: tuple[int, int], obj_mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    cells = np.zeros(grid_shape, dtype=bool)
    h, w = obj_mask.shape
    cells[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = obj_mask
    return cells


def _gen_all_positions(grid_shape: tuple[int, int], obj_mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = grid_shape
    height, width = obj_mask.shape
    return [
        (r, c)
        for r in range(rows - height + 1)
        for c in range(cols - width + 1)
    ]


def _gen_clipped_first_positions(grid_shape: tuple[int, int], obj_mask: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    clipped_positions = []
    for pos in _gen_all_positions(grid_shape, obj_mask):
        cap, clipped = _gen_cap_mask(grid_shape, obj_mask, *pos)
        if clipped:
            clipped_positions.append((pos, cap))
    return clipped_positions


def _gen_valid_second_positions(
    grid_shape: tuple[int, int],
    mask1: np.ndarray,
    pos1: tuple[int, int],
    cap1: np.ndarray,
    mask2: np.ndarray,
) -> list[tuple[int, int]]:
    bbox1 = _gen_bbox(pos1, mask1)
    cells1 = _gen_object_cells(grid_shape, mask1, *pos1)
    valid: list[tuple[int, int]] = []
    for pos2 in _gen_all_positions(grid_shape, mask2):
        if not _gen_bboxes_have_gap(bbox1, _gen_bbox(pos2, mask2)):
            continue
        cells2 = _gen_object_cells(grid_shape, mask2, *pos2)
        cap2, _ = _gen_cap_mask(grid_shape, mask2, *pos2)
        if np.any(cap1 & cap2):
            continue
        if np.any(cap1 & cells2) or np.any(cap2 & cells1):
            continue
        valid.append(pos2)
    return valid


def _gen_object_masks() -> tuple[np.ndarray, np.ndarray]:
    radius1, height1, width1 = _gen_shape_params()
    radius2, height2, width2 = _gen_shape_params()
    return _gen_rounded_mask(height1, width1, radius1), _gen_rounded_mask(height2, width2, radius2)


def _gen_place_objects(grid_shape: tuple[int, int], mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray | None:
    clipped_positions1 = _gen_clipped_first_positions(grid_shape, mask1)
    if not clipped_positions1:
        return None

    random.shuffle(clipped_positions1)
    for pos1, cap1 in clipped_positions1[:_GEN_FIRST_PLACEMENT_TRIES]:
        positions2 = _gen_valid_second_positions(grid_shape, mask1, pos1, cap1, mask2)
        if not positions2:
            continue
        pos2 = random.choice(positions2)
        cap2, _ = _gen_cap_mask(grid_shape, mask2, *pos2)

        color = random.choice(_GEN_FOREGROUND_COLORS)
        grid = np.zeros(grid_shape, dtype=int)
        paste(grid, np.where(mask1, color, Color.BLACK), pos1[0], pos1[1], transparent=Color.BLACK)
        paste(grid, np.where(mask2, color, Color.BLACK), pos2[0], pos2[1], transparent=Color.BLACK)

        expected = grid.copy()
        object_cells = grid != Color.BLACK
        expected[(cap1 | cap2) & ~object_cells] = Color.YELLOW
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            return None

    return None


def _gen_attempt() -> np.ndarray | None:
    grid_shape = (random.randint(*_GEN_ROW_RANGE), random.randint(*_GEN_COL_RANGE))
    mask1, mask2 = _gen_object_masks()
    return _gen_place_objects(grid_shape, mask1, mask2)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 9772c176 input")
