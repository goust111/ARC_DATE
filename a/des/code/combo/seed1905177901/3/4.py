import random

import numpy as np


BG = 0
SEPARATOR = 7
FG_POOL = tuple(color for color in range(1, 10) if color != SEPARATOR)
QUADRANT_COUNT = 4
NEXT_QUADRANT = (1, 2, 3, 0)
GEN_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _find_cross(grid: np.ndarray) -> tuple[int, int, int, int]:
    bg = _background_color(grid)
    row_candidates = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg:
            row_candidates.append((r, int(grid[r, 0])))

    col_candidates = []
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg:
            col_candidates.append((c, int(grid[0, c])))

    matches = [
        (r, c, color_r)
        for r, color_r in row_candidates
        for c, color_c in col_candidates
        if color_r == color_c
    ]
    assert len(matches) == 1, "expected one full cross separator"
    cross_r, cross_c, sep_color = matches[0]
    assert 0 < cross_r < grid.shape[0] - 1
    assert 0 < cross_c < grid.shape[1] - 1
    return int(cross_r), int(cross_c), int(sep_color), int(bg)


def _quadrant_shapes(height: int, width: int, cross_r: int, cross_c: int) -> list[tuple[int, int]]:
    top_h = cross_r
    bottom_h = height - cross_r - 1
    left_w = cross_c
    right_w = width - cross_c - 1
    return [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, right_w),
        (bottom_h, left_w),
    ]


def _offset_from_cell(row: int, col: int, cross_r: int, cross_c: int) -> tuple[int, int]:
    if row < cross_r:
        dr = cross_r - 1 - row
    else:
        dr = row - cross_r - 1
    if col < cross_c:
        dc = cross_c - 1 - col
    else:
        dc = col - cross_c - 1
    return int(dr), int(dc)


def _cell_from_offset(quadrant: int, dr: int, dc: int, cross_r: int, cross_c: int) -> tuple[int, int]:
    if quadrant == 0:
        return cross_r - 1 - dr, cross_c - 1 - dc
    if quadrant == 1:
        return cross_r - 1 - dr, cross_c + 1 + dc
    if quadrant == 2:
        return cross_r + 1 + dr, cross_c + 1 + dc
    if quadrant == 3:
        return cross_r + 1 + dr, cross_c - 1 - dc
    raise ValueError(f"unknown quadrant: {quadrant}")


def _apply_rule(grid: np.ndarray, bg: int, cross_r: int, cross_c: int, sep_color: int) -> np.ndarray:
    height, width = grid.shape
    shapes = _quadrant_shapes(height, width, cross_r, cross_c)
    out = np.full_like(grid, bg)
    out[cross_r, :] = sep_color
    out[:, cross_c] = sep_color

    for quadrant, (quad_h, quad_w) in enumerate(shapes):
        next_quadrant = NEXT_QUADRANT[quadrant]
        next_h, next_w = shapes[next_quadrant]
        for dr in range(quad_h):
            for dc in range(quad_w):
                row, col = _cell_from_offset(quadrant, dr, dc, cross_r, cross_c)
                if int(grid[row, col]) == bg:
                    continue
                if dr >= next_h or dc >= next_w:
                    continue
                src_row, src_col = _cell_from_offset(next_quadrant, dr, dc, cross_r, cross_c)
                if int(grid[src_row, src_col]) != bg:
                    out[row, col] = int(grid[src_row, src_col])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate colors between the four separator-defined quadrants by matching each
    foreground pixel with the clockwise-next quadrant at the same center-relative
    offset. Pixels whose counterparts are missing become background.
    """
    g = np.asarray(grid, dtype=int)
    cross_r, cross_c, sep_color, bg = _find_cross(g)
    return _apply_rule(g, bg, cross_r, cross_c, sep_color)


def _valid_offsets(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(dr, dc) for dr in range(h) for dc in range(w)]


def _build_expected(
    occupied: list[dict[tuple[int, int], int]],
    shapes: list[tuple[int, int]],
    cross_r: int,
    cross_c: int,
) -> np.ndarray:
    height = shapes[0][0] + 1 + shapes[2][0]
    width = shapes[0][1] + 1 + shapes[1][1]
    out = np.full((height, width), BG, dtype=int)
    out[cross_r, :] = SEPARATOR
    out[:, cross_c] = SEPARATOR

    for quadrant in range(QUADRANT_COUNT):
        next_quadrant = NEXT_QUADRANT[quadrant]
        for offset in occupied[quadrant]:
            color = occupied[next_quadrant].get(offset, BG)
            if color == BG:
                continue
            row, col = _cell_from_offset(quadrant, offset[0], offset[1], cross_r, cross_c)
            out[row, col] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(3, 8)
    bottom_h = random.randint(3, 8)
    left_w = random.randint(3, 8)
    right_w = random.randint(3, 8)
    shapes = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, right_w),
        (bottom_h, left_w),
    ]

    min_area = min(h * w for h, w in shapes)
    palette_size = random.randint(2, min(4, min_area))
    palette = random.sample(FG_POOL, palette_size)

    common_offsets = [
        (dr, dc)
        for dr in range(min(top_h, bottom_h))
        for dc in range(min(left_w, right_w))
    ]
    if not common_offsets:
        return None

    match_offset = random.choice(common_offsets)
    match_colors = [random.choice(palette) for _ in range(QUADRANT_COUNT)]
    if len(set(match_colors)) == 1 and len(palette) > 1:
        idx = random.randrange(QUADRANT_COUNT)
        alternatives = [color for color in palette if color != match_colors[idx]]
        match_colors[idx] = random.choice(alternatives)

    q_clear = random.randrange(QUADRANT_COUNT)
    q_next = NEXT_QUADRANT[q_clear]
    clear_candidates = [offset for offset in _valid_offsets(shapes[q_clear]) if offset != match_offset]
    if not clear_candidates:
        return None

    out_of_bounds_clear = [
        offset
        for offset in clear_candidates
        if offset[0] >= shapes[q_next][0] or offset[1] >= shapes[q_next][1]
    ]
    if out_of_bounds_clear and random.random() < 0.5:
        clear_offset = random.choice(out_of_bounds_clear)
    else:
        clear_offset = random.choice(clear_candidates)

    occupied = [dict() for _ in range(QUADRANT_COUNT)]
    banned = [set() for _ in range(QUADRANT_COUNT)]
    for quadrant in range(QUADRANT_COUNT):
        occupied[quadrant][match_offset] = match_colors[quadrant]

    occupied[q_clear][clear_offset] = random.choice(palette)
    if clear_offset[0] < shapes[q_next][0] and clear_offset[1] < shapes[q_next][1]:
        banned[q_next].add(clear_offset)

    for quadrant in range(QUADRANT_COUNT):
        missing_colors = [color for color in palette if color not in set(occupied[quadrant].values())]
        candidates = [
            offset
            for offset in _valid_offsets(shapes[quadrant])
            if offset not in occupied[quadrant] and offset not in banned[quadrant]
        ]
        if len(candidates) < len(missing_colors):
            return None
        random.shuffle(candidates)
        for color, offset in zip(missing_colors, candidates[: len(missing_colors)]):
            occupied[quadrant][offset] = color

    for quadrant in range(QUADRANT_COUNT):
        candidates = [
            offset
            for offset in _valid_offsets(shapes[quadrant])
            if offset not in occupied[quadrant] and offset not in banned[quadrant]
        ]
        if not candidates:
            continue
        extra_count = random.randint(0, min(4, len(candidates)))
        for offset in random.sample(candidates, extra_count):
            occupied[quadrant][offset] = random.choice(palette)

    if any(set(quad.values()) != set(palette) for quad in occupied):
        return None
    if not any(offset in occupied[q] and offset not in occupied[NEXT_QUADRANT[q]] for q in range(QUADRANT_COUNT) for offset in occupied[q]):
        return None

    cross_r = top_h
    cross_c = left_w
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w
    grid = np.full((height, width), BG, dtype=int)
    grid[cross_r, :] = SEPARATOR
    grid[:, cross_c] = SEPARATOR

    for quadrant in range(QUADRANT_COUNT):
        for (dr, dc), color in occupied[quadrant].items():
            row, col = _cell_from_offset(quadrant, dr, dc, cross_r, cross_c)
            grid[row, col] = int(color)

    bg_count = int(np.count_nonzero(grid == BG))
    non_bg_colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if not non_bg_colors:
        return None
    if bg_count <= max(int(np.count_nonzero(grid == color)) for color in non_bg_colors):
        return None

    expected = _build_expected(occupied, shapes, cross_r, cross_c)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
