from __future__ import annotations

import random

import numpy as np


ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
MIN_DEPTH = 2
MAX_DEPTH = 4
INNER_SIZE_RANGE = (2, 5)
INTER_FRAME_PAD_RANGE = (0, 2)
MAX_GRID_SIDE = 26
MIN_TOTAL_MARGIN = 4
MAX_TOTAL_MARGIN = 12
PATTERN_COLOR_COUNT_CHOICES = (2, 3)


def _get_background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if int(coords.shape[0]) != 2 * (height + width) - 4:
        return None

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    if not np.all(region[border_mask] == int(color)):
        return None
    if np.any(region[~border_mask] == int(color)):
        return None
    return (r0, c0, r1, c1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _find_nested_frames(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    bg = _get_background(grid)
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        bbox = _frame_bbox(grid, color)
        if bbox is not None:
            frames.append((color, bbox))

    frames.sort(key=lambda item: _bbox_area(item[1]), reverse=True)
    assert len(frames) >= 2, "expected at least two nested border colors"
    for idx in range(len(frames) - 1):
        assert _strictly_contains(frames[idx][1], frames[idx + 1][1]), "frames must form one strict nesting chain"
    return frames


def _apply_transform(region: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flip":
        return np.fliplr(region)
    if mode == "rot180":
        return np.rot90(region, 2)
    raise AssertionError(f"unknown transform mode: {mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect nested single-pixel rectangular borders, infer the nesting depth,
    and transform only the innermost interior region in place.

    Odd depth -> horizontal flip.
    Even depth -> 180-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_nested_frames(g)
    depth = len(frames)
    _color, innermost_bbox = frames[-1]
    r0, c0, r1, c1 = innermost_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0, "innermost frame must enclose a non-empty region"

    mode = "flip" if depth % 2 == 1 else "rot180"
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = _apply_transform(interior, mode)
    return out


def _paint_random_shape(pattern: np.ndarray, color: int) -> None:
    height, width = map(int, pattern.shape)
    op = random.choice(("point", "hline", "vline", "rect", "lshape"))

    if op == "point":
        pattern[random.randrange(height), random.randrange(width)] = color
        return

    if op == "hline":
        row = random.randrange(height)
        c0 = random.randrange(width)
        c1 = random.randrange(c0, width)
        pattern[row, c0 : c1 + 1] = color
        return

    if op == "vline":
        col = random.randrange(width)
        r0 = random.randrange(height)
        r1 = random.randrange(r0, height)
        pattern[r0 : r1 + 1, col] = color
        return

    if op == "rect":
        r0 = random.randrange(height)
        r1 = random.randrange(r0, height)
        c0 = random.randrange(width)
        c1 = random.randrange(c0, width)
        pattern[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    corner_r = random.randrange(height)
    corner_c = random.randrange(width)
    arm_h = random.randint(1, height)
    arm_w = random.randint(1, width)
    vertical_dir = random.choice((-1, 1))
    horizontal_dir = random.choice((-1, 1))
    for step in range(arm_h):
        rr = corner_r + vertical_dir * step
        if 0 <= rr < height:
            pattern[rr, corner_c] = color
    for step in range(arm_w):
        cc = corner_c + horizontal_dir * step
        if 0 <= cc < width:
            pattern[corner_r, cc] = color


def _make_pattern(height: int, width: int, bg: int, palette: list[int], mode: str) -> np.ndarray | None:
    other_mode = "rot180" if mode == "flip" else "flip"
    max_ops = max(4, height + width)

    for _ in range(120):
        pattern = np.full((height, width), bg, dtype=int)
        for _ in range(random.randint(3, max_ops)):
            _paint_random_shape(pattern, random.choice(palette))

        used_colors = {int(v) for v in np.unique(pattern)}
        non_bg_coords = np.argwhere(pattern != bg)
        if len(used_colors) < 2 or non_bg_coords.shape[0] < 3:
            continue
        if len(set(map(int, non_bg_coords[:, 0]))) < 2:
            continue
        if len(set(map(int, non_bg_coords[:, 1]))) < 2:
            continue

        transformed = _apply_transform(pattern, mode)
        alternative = _apply_transform(pattern, other_mode)
        if np.array_equal(pattern, transformed):
            continue
        if np.array_equal(transformed, alternative):
            continue
        return pattern

    return None


def _place_inner_object(pattern: np.ndarray, frame_color: int, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = map(int, pattern.shape)
    obj = np.full((height + 2, width + 2), bg, dtype=int)
    obj[1:-1, 1:-1] = pattern
    obj[0, :] = frame_color
    obj[-1, :] = frame_color
    obj[:, 0] = frame_color
    obj[:, -1] = frame_color
    return obj, (0, 0, height + 1, width + 1)


def _wrap_with_frame(
    inner: np.ndarray,
    frame_color: int,
    bg: int,
    top_pad: int,
    bottom_pad: int,
    left_pad: int,
    right_pad: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    inner_h, inner_w = map(int, inner.shape)
    out = np.full(
        (inner_h + top_pad + bottom_pad + 2, inner_w + left_pad + right_pad + 2),
        bg,
        dtype=int,
    )
    row0 = 1 + top_pad
    col0 = 1 + left_pad
    out[row0 : row0 + inner_h, col0 : col0 + inner_w] = inner
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    return out, (row0, col0)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _choose_margin_pair(object_side: int) -> tuple[int, int] | None:
    max_extra = MAX_GRID_SIDE - object_side
    if max_extra < MIN_TOTAL_MARGIN:
        return None
    total = random.randint(MIN_TOTAL_MARGIN, min(MAX_TOTAL_MARGIN, max_extra))
    before = random.randint(1, total - 1)
    after = total - before
    return before, after


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(colors == int(bg))[0][0]])
    return all(bg_count > int(count) for color, count in zip(colors, counts) if int(color) != int(bg))


def _gen_attempt() -> np.ndarray | None:
    if random.random() < 0.5:
        depth = 3
    else:
        depth = random.choice((2, 4))
    mode = "flip" if depth % 2 == 1 else "rot180"

    bg = random.choice(ALL_COLORS)
    usable_colors = [color for color in ALL_COLORS if color != bg]
    random.shuffle(usable_colors)
    frame_colors = usable_colors[:depth]
    pattern_palette_size = random.choice(PATTERN_COLOR_COUNT_CHOICES)
    pattern_palette = usable_colors[depth : depth + pattern_palette_size]

    pattern_h = random.randint(*INNER_SIZE_RANGE)
    pattern_w = random.randint(*INNER_SIZE_RANGE)
    pattern = _make_pattern(pattern_h, pattern_w, bg, pattern_palette, mode)
    if pattern is None:
        return None

    nested, deepest_bbox = _place_inner_object(pattern, frame_colors[-1], bg)
    for color in reversed(frame_colors[:-1]):
        top_pad = random.randint(*INTER_FRAME_PAD_RANGE)
        bottom_pad = random.randint(*INTER_FRAME_PAD_RANGE)
        left_pad = random.randint(*INTER_FRAME_PAD_RANGE)
        right_pad = random.randint(*INTER_FRAME_PAD_RANGE)
        nested, (dr, dc) = _wrap_with_frame(
            nested,
            color,
            bg,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
        )
        deepest_bbox = _shift_bbox(deepest_bbox, dr, dc)

    row_margins = _choose_margin_pair(int(nested.shape[0]))
    col_margins = _choose_margin_pair(int(nested.shape[1]))
    if row_margins is None or col_margins is None:
        return None
    top_margin, bottom_margin = row_margins
    left_margin, right_margin = col_margins

    grid = np.full(
        (int(nested.shape[0]) + top_margin + bottom_margin, int(nested.shape[1]) + left_margin + right_margin),
        bg,
        dtype=int,
    )
    grid[top_margin : top_margin + nested.shape[0], left_margin : left_margin + nested.shape[1]] = nested
    deepest_bbox = _shift_bbox(deepest_bbox, top_margin, left_margin)

    if not _background_is_dominant(grid, bg):
        return None

    try:
        detected = _find_nested_frames(grid)
    except AssertionError:
        return None
    detected_colors = [color for color, _bbox in detected]
    if detected_colors != frame_colors:
        return None

    expected = grid.copy()
    r0, c0, r1, c1 = deepest_bbox
    interior = expected[r0 + 1 : r1, c0 + 1 : c1].copy()
    expected[r0 + 1 : r1, c0 + 1 : c1] = _apply_transform(interior, mode)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame ARC input")
