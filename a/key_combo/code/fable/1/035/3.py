from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BG_COLOR = BLACK
BLOCK_COLORS = [GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
RNG = np.random.default_rng()

GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
START_FRACTION_LOW = 0.25
START_FRACTION_HIGH = 0.75
MIN_PREFIX_STEPS = 7
MAX_HIT_INDEX = 60
MIN_BLOCK_SIZE = 1
MAX_BLOCK_SIZE = 6
MIN_BLOCK_AREA = 2
TOTAL_BLOCK_RANGE = (2, 5)
GEN_RETRY_LIMIT = 400
PLACEMENT_TRIES = 200


def get_bg(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel()).argmax())


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(int(lo), int(hi) + 1))


def _find_marker(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    marker_mask = ((grid == BLUE) | (grid == RED)) & (grid != bg)
    coords = np.argwhere(marker_mask)
    assert coords.shape[0] == 1, "expected exactly one red/blue marker"
    r, c = map(int, coords[0])
    color = int(grid[r, c])
    for nr, nc in _neighbors8(r, c):
        if in_bounds(grid.shape, nr, nc):
            assert int(grid[nr, nc]) == bg, "marker must be isolated"
    return (r, c), color


def _spiral_directions(marker_color: int) -> list[tuple[int, int]]:
    if marker_color == RED:
        return [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if marker_color == BLUE:
        return [(-1, 0), (0, -1), (1, 0), (0, 1)]
    raise AssertionError("marker color must be red or blue")


def _spiral_trace(shape: tuple[int, int], start: tuple[int, int], marker_color: int) -> list[tuple[int, int]]:
    pos = tuple(map(int, start))
    directions = _spiral_directions(marker_color)
    direction_index = 0
    segment_length = 1
    trace: list[tuple[int, int]] = []

    while True:
        for _ in range(2):
            dr, dc = directions[direction_index % 4]
            for _ in range(segment_length):
                nr = pos[0] + dr
                nc = pos[1] + dc
                if not in_bounds(shape, nr, nc):
                    return trace
                pos = (nr, nc)
                trace.append(pos)
            direction_index += 1
        segment_length += 1


def _spiral_hit_info(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int], int, list[tuple[int, int]], tuple[int, int], int]:
    bg = get_bg(grid)
    marker, marker_color = _find_marker(grid, bg)
    pos = marker
    directions = _spiral_directions(marker_color)
    direction_index = 0
    segment_length = 1
    trail: list[tuple[int, int]] = []

    while True:
        for _ in range(2):
            dr, dc = directions[direction_index % 4]
            for _ in range(segment_length):
                nr = pos[0] + dr
                nc = pos[1] + dc
                if not in_bounds(grid.shape, nr, nc):
                    raise AssertionError("spiral left the grid before hitting a block")
                value = int(grid[nr, nc])
                if value == bg:
                    trail.append((nr, nc))
                    pos = (nr, nc)
                    continue
                assert (nr, nc) != marker, "spiral should not revisit the marker"
                assert value not in (BLUE, RED, GREY), "blocks must avoid red, blue, and grey"
                return bg, marker, marker_color, trail, (nr, nc), value
            direction_index += 1
        segment_length += 1


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    target = int(grid[sr, sc])
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    cells: list[tuple[int, int]] = []

    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != target:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return cells


def _all_block_components(
    grid: np.ndarray, bg: int, marker: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    visited[marker] = True
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            assert color not in (BLUE, RED, GREY), "only the marker may use red/blue and no input block may be grey"
            comp = _same_color_component(grid, (r, c))
            for rr, cc in comp:
                visited[rr, cc] = True
            components.append(comp)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the hidden square spiral in grey until it hits a block, then swap colors between the marker and that block.

    Input structure:
    - The background is the dominant color.
    - Exactly one isolated single-pixel marker is red or blue.
    - Every other foreground object is a solid monochromatic rectangle whose color is not red, blue, or grey.

    Rule:
    - Start from the marker and follow a square spiral with segment lengths 1, 1, 2, 2, 3, 3, ... and first direction upward.
    - A red marker means the spiral turns clockwise; a blue marker means it turns counterclockwise.
    - Recolor every traversed background cell to grey, stopping immediately before entering the first block encountered.
    - Recolor that entire hit block to the marker color, and recolor the marker cell to the block's original color.

    Key insight:
    - The spiral path is implicit: marker color encodes chirality, while the first collision selects the only block that exchanges color with the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, marker, marker_color, trail, hit, hit_color = _spiral_hit_info(g)
    out = g.copy()

    for r, c in trail:
        out[r, c] = GREY
    for r, c in _same_color_component(g, hit):
        out[r, c] = marker_color
    out[marker] = hit_color

    assert get_bg(out) == bg or bg not in (BLUE, RED, GREY)
    return out


def _contains(rect: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _mark_reserved(
    reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int = 1
) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(reserved.shape[0] - 1, r1 + margin)
    cc1 = min(reserved.shape[1] - 1, c1 + margin)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rect_intersects_forbidden(
    rect: tuple[int, int, int, int], forbidden: set[tuple[int, int]]
) -> bool:
    r0, c0, r1, c1 = rect
    for r, c in forbidden:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _candidate_hit_rectangles(
    shape: tuple[int, int], start: tuple[int, int], trace: list[tuple[int, int]], hit_index: int
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, shape)
    hit = trace[hit_index]
    forbidden = {start, *trace[:hit_index]}
    candidates: list[tuple[int, int, int, int]] = []

    for height in range(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, h) + 1):
        for width in range(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, w) + 1):
            if height * width < MIN_BLOCK_AREA:
                continue
            r_lo = max(0, hit[0] - height + 1)
            r_hi = min(hit[0], h - height)
            c_lo = max(0, hit[1] - width + 1)
            c_hi = min(hit[1], w - width)
            for r0 in range(r_lo, r_hi + 1):
                for c0 in range(c_lo, c_hi + 1):
                    rect = (r0, c0, r0 + height - 1, c0 + width - 1)
                    if _rect_intersects_forbidden(rect, forbidden):
                        continue
                    candidates.append(rect)
    return candidates


def _try_place_distractor(
    grid: np.ndarray,
    reserved: np.ndarray,
    forbidden: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = map(int, grid.shape)
    for _ in range(PLACEMENT_TRIES):
        height = _randint(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, h))
        width = _randint(MIN_BLOCK_SIZE, min(MAX_BLOCK_SIZE, w))
        if height * width < MIN_BLOCK_AREA:
            continue
        r0 = _randint(0, h - height)
        c0 = _randint(0, w - width)
        rect = (r0, c0, r0 + height - 1, c0 + width - 1)
        if np.any(reserved[r0 : rect[2] + 1, c0 : rect[3] + 1]):
            continue
        if _rect_intersects_forbidden(rect, forbidden):
            continue
        _paint_rect(grid, rect, color)
        _mark_reserved(reserved, *rect, margin=1)
        return True
    return False


def _pick_start(size: int) -> tuple[int, int]:
    lo = max(2, int(round(size * START_FRACTION_LOW)))
    hi = min(size - 3, int(round(size * START_FRACTION_HIGH)))
    if lo > hi:
        lo, hi = 2, size - 3
    return lo, hi


def _validate_generated_grid(grid: np.ndarray) -> None:
    bg = get_bg(grid)
    marker, marker_color = _find_marker(grid, bg)
    assert marker_color in (BLUE, RED)
    blocks = _all_block_components(grid, bg, marker)
    assert len(blocks) >= 2, "generator should include at least one distractor block"
    for comp in blocks:
        assert _is_solid_rectangle(comp), "every block must be a solid rectangle"
    out = main(grid)
    assert out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)
    row_lo, row_hi = _pick_start(h)
    col_lo, col_hi = _pick_start(w)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    marker = (_randint(row_lo, row_hi), _randint(col_lo, col_hi))
    marker_color = RED if RNG.random() < 0.5 else BLUE
    trace = _spiral_trace((h, w), marker, marker_color)
    if len(trace) <= MIN_PREFIX_STEPS:
        return None

    max_index = min(MAX_HIT_INDEX, len(trace) - 1)
    hit_options: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for hit_index in range(MIN_PREFIX_STEPS, max_index + 1):
        rects = _candidate_hit_rectangles((h, w), marker, trace, hit_index)
        if rects:
            hit_options.append((hit_index, rects))
    if not hit_options:
        return None

    hit_index, rects = hit_options[_randint(0, len(hit_options) - 1)]
    hit_rect = rects[_randint(0, len(rects) - 1)]

    colors = BLOCK_COLORS.copy()
    RNG.shuffle(colors)
    hit_color = int(colors[0])
    distractor_colors = [int(color) for color in colors[1:]]

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[marker] = marker_color
    reserved = np.zeros((h, w), dtype=bool)
    _mark_reserved(reserved, marker[0], marker[1], marker[0], marker[1], margin=1)

    _paint_rect(grid, hit_rect, hit_color)
    _mark_reserved(reserved, *hit_rect, margin=1)

    forbidden = {marker, *trace[: hit_index + 1]}
    target_blocks = _randint(*TOTAL_BLOCK_RANGE)
    placed_blocks = 1
    for color in distractor_colors:
        if placed_blocks >= target_blocks:
            break
        if _try_place_distractor(grid, reserved, forbidden, color):
            placed_blocks += 1
    if placed_blocks < 2:
        return None

    try:
        _bg, actual_marker, actual_marker_color, trail, hit_cell, actual_hit_color = _spiral_hit_info(grid)
    except AssertionError:
        return None
    if actual_marker != marker or actual_marker_color != marker_color:
        return None
    if len(trail) != hit_index:
        return None
    if not _contains(hit_rect, hit_cell):
        return None
    if actual_hit_color != hit_color:
        return None

    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-spiral input")
