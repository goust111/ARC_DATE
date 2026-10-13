from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MAGENTA = 6
DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
NON_SIGNAL_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_anchor_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == MAGENTA):
                candidates.append((r0, c0, r0 + 2, c0 + 2))
    assert len(candidates) == 1, "expected exactly one solid 3x3 magenta anchor"
    return candidates[0]


def _outside_anchor(
    r: int, c: int, anchor: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = anchor
    return not (r0 <= r <= r1 and c0 <= c <= c1)


def _infer_signal_ray(
    grid: np.ndarray, anchor: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    scatter = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == MAGENTA)
        if _outside_anchor(int(r), int(c), anchor)
    }
    assert scatter, "expected at least one magenta scatter pixel"

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for br in range(r0, r1 + 1):
        for bc in range(c0, c1 + 1):
            for dr, dc in DIRS:
                rr = br + dr
                cc = bc + dc
                if not _in_bounds(grid.shape, rr, cc):
                    continue
                if not _outside_anchor(rr, cc, anchor):
                    continue

                line_cells: set[tuple[int, int]] = set()
                while _in_bounds(grid.shape, rr, cc):
                    line_cells.add((rr, cc))
                    rr += dr
                    cc += dc

                if scatter <= line_cells:
                    candidates.append(((br, bc), (dr, dc)))

    uniq = list(dict.fromkeys(candidates))
    assert len(uniq) == 1, "signal ray must be uniquely determined"
    return uniq[0]


def _component_from(
    grid: np.ndarray, start: tuple[int, int]
) -> tuple[int, set[tuple[int, int]]]:
    color = int(grid[start])
    assert color not in (BG, MAGENTA)

    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return color, seen


def _square_bbox(
    grid: np.ndarray, start: tuple[int, int]
) -> tuple[int, int, int, int, int]:
    color, cells = _component_from(grid, start)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    side = r1 - r0 + 1
    assert side >= 2 and side == c1 - c0 + 1, "hit object must be a square"
    assert len(cells) == side * side, "hit object must be a solid square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    return r0, c0, r1, c1, color


def _first_hit_square(
    grid: np.ndarray,
    anchor_cell: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int, int, int, int]:
    dr, dc = direction
    r = anchor_cell[0] + dr
    c = anchor_cell[1] + dc
    while _in_bounds(grid.shape, r, c):
        val = int(grid[r, c])
        if val not in (BG, MAGENTA):
            return _square_bbox(grid, (r, c))
        r += dr
        c += dc
    raise AssertionError("signal ray must hit a colored square")


def _paint_periodic_copies(
    out: np.ndarray,
    square: tuple[int, int, int, int, int],
    direction: tuple[int, int],
    *,
    bg: int,
) -> None:
    r0, c0, r1, c1, color = square
    dr, dc = direction
    side = r1 - r0 + 1
    step = side
    k = 1
    while True:
        nr0 = r0 + dr * step * k
        nc0 = c0 + dc * step * k
        nr1 = nr0 + side - 1
        nc1 = nc0 + side - 1
        if nr0 < 0 or nc0 < 0 or nr1 >= out.shape[0] or nc1 >= out.shape[1]:
            break
        patch = out[nr0 : nr1 + 1, nc0 : nc1 + 1]
        patch[patch == bg] = color
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = patch
        k += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the magenta signal ray, find the first square hit by that ray, then
    repeat that square along the same direction with period equal to its side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    anchor = _find_anchor_bbox(g)
    anchor_cell, direction = _infer_signal_ray(g, anchor)
    square = _first_hit_square(g, anchor_cell, direction)

    out = g.copy()
    _paint_periodic_copies(out, square, direction, bg=bg)
    return out


def _anchor_boundary_offsets(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    out: list[tuple[int, int]] = []
    for br in range(3):
        for bc in range(3):
            nr = br + dr
            nc = bc + dc
            if not (0 <= nr < 3 and 0 <= nc < 3):
                out.append((br, bc))
    return out


def _frontier_offsets(side: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    frontier: list[tuple[int, int]] = []
    for rr in range(side):
        for cc in range(side):
            pr = rr - dr
            pc = cc - dc
            if not (0 <= pr < side and 0 <= pc < side):
                frontier.append((rr, cc))
    return frontier


def _copy_positions(
    top: int, left: int, side: int, direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    out: list[tuple[int, int]] = []
    k = 1
    while True:
        nr0 = top + k * side * dr
        nc0 = left + k * side * dc
        nr1 = nr0 + side - 1
        nc1 = nc0 + side - 1
        if nr0 < 0 or nc0 < 0 or nr1 >= shape[0] or nc1 >= shape[1]:
            break
        out.append((nr0, nc0))
        k += 1
    return out


def _inflate_box(
    r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int], margin: int = 1
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(shape[0] - 1, r1 + margin),
        min(shape[1] - 1, c1 + margin),
    )


def _mark_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _can_place(
    blocked: np.ndarray, r0: int, c0: int, side: int
) -> bool:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    if r0 < 0 or c0 < 0 or r1 >= blocked.shape[0] or c1 >= blocked.shape[1]:
        return False
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _choose_signal_layout(
    h: int, w: int, direction: tuple[int, int]
) -> tuple[int, int, int, int, int, int, tuple[int, int]] | None:
    boundary_offsets = _anchor_boundary_offsets(direction)
    for _ in range(300):
        side = random.randint(2, min(5, h - 4, w - 4))
        frontier = _frontier_offsets(side, direction)
        hit_offset = random.choice(frontier)
        top = random.randint(0, h - side)
        left = random.randint(0, w - side)
        hit_r = top + hit_offset[0]
        hit_c = left + hit_offset[1]

        copy_starts = _copy_positions(top, left, side, direction, (h, w))
        if not copy_starts:
            continue

        boundary = random.choice(boundary_offsets)
        gap_choices = list(range(3, max(h, w) + 1))
        random.shuffle(gap_choices)
        for gap in gap_choices:
            anchor_top = hit_r - gap * direction[0] - boundary[0]
            anchor_left = hit_c - gap * direction[1] - boundary[1]
            if anchor_top < 0 or anchor_left < 0 or anchor_top + 2 >= h or anchor_left + 2 >= w:
                continue

            anchor_r1 = anchor_top + 2
            anchor_c1 = anchor_left + 2
            if not (
                anchor_r1 < top
                or top + side - 1 < anchor_top
                or anchor_c1 < left
                or left + side - 1 < anchor_left
            ):
                continue

            return anchor_top, anchor_left, top, left, side, gap, boundary
    return None


def _scatter_positions(
    anchor_top: int,
    anchor_left: int,
    boundary: tuple[int, int],
    direction: tuple[int, int],
    gap: int,
) -> list[tuple[int, int]]:
    start_r = anchor_top + boundary[0]
    start_c = anchor_left + boundary[1]
    options = [
        (start_r + t * direction[0], start_c + t * direction[1])
        for t in range(1, gap)
    ]
    count = random.randint(2, min(len(options), 4))
    chosen = sorted(random.sample(options, count))
    return chosen


def _place_square(
    grid: np.ndarray,
    moat: np.ndarray,
    top: int,
    left: int,
    side: int,
    color: int,
) -> None:
    grid[top : top + side, left : left + side] = color
    r0, c0, r1, c1 = _inflate_box(top, left, top + side - 1, left + side - 1, grid.shape)
    _mark_box(moat, r0, c0, r1, c1)


def _reserve_signal_geometry(
    h: int,
    w: int,
    anchor_top: int,
    anchor_left: int,
    boundary: tuple[int, int],
    direction: tuple[int, int],
    gap: int,
    square_top: int,
    square_left: int,
    side: int,
) -> np.ndarray:
    blocked = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = _inflate_box(anchor_top, anchor_left, anchor_top + 2, anchor_left + 2, (h, w))
    _mark_box(blocked, r0, c0, r1, c1)
    r0, c0, r1, c1 = _inflate_box(square_top, square_left, square_top + side - 1, square_left + side - 1, (h, w))
    _mark_box(blocked, r0, c0, r1, c1)

    start_r = anchor_top + boundary[0]
    start_c = anchor_left + boundary[1]
    for t in range(1, gap):
        rr = start_r + t * direction[0]
        cc = start_c + t * direction[1]
        r0, c0, r1, c1 = _inflate_box(rr, cc, rr, cc, (h, w))
        _mark_box(blocked, r0, c0, r1, c1)

    for copy_top, copy_left in _copy_positions(square_top, square_left, side, direction, (h, w)):
        r0, c0, r1, c1 = _inflate_box(copy_top, copy_left, copy_top + side - 1, copy_left + side - 1, (h, w))
        _mark_box(blocked, r0, c0, r1, c1)

    dr, dc = direction
    rr = start_r + dr
    cc = start_c + dc
    while _in_bounds((h, w), rr, cc):
        r0, c0, r1, c1 = _inflate_box(rr, cc, rr, cc, (h, w), margin=0)
        _mark_box(blocked, r0, c0, r1, c1)
        rr += dr
        cc += dc
    return blocked


def _place_distractors(
    grid: np.ndarray,
    moat: np.ndarray,
    blocked: np.ndarray,
    colors: list[int],
) -> bool:
    h, w = grid.shape
    count = random.randint(2, min(5, len(colors)))
    placed = 0
    for color in colors[:count]:
        success = False
        for _ in range(120):
            side = random.randint(2, min(4, h - 2, w - 2))
            top = random.randint(0, h - side)
            left = random.randint(0, w - side)
            if not _can_place(moat, top, left, side):
                continue
            if bool(blocked[top : top + side, left : left + side].any()):
                continue
            _place_square(grid, moat, top, left, side, color)
            success = True
            placed += 1
            break
        if not success:
            continue
    return placed >= 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    direction = random.choice(DIRS)

    layout = _choose_signal_layout(h, w, direction)
    if layout is None:
        return None
    anchor_top, anchor_left, square_top, square_left, side, gap, boundary = layout

    grid = np.full((h, w), BG, dtype=int)
    moat = np.zeros((h, w), dtype=bool)

    total_square_colors = random.randint(3, 6)
    colors = random.sample(NON_SIGNAL_COLORS, total_square_colors)
    hit_color = colors[0]
    distractor_colors = colors[1:]

    grid[anchor_top : anchor_top + 3, anchor_left : anchor_left + 3] = MAGENTA
    r0, c0, r1, c1 = _inflate_box(anchor_top, anchor_left, anchor_top + 2, anchor_left + 2, (h, w))
    _mark_box(moat, r0, c0, r1, c1)

    for rr, cc in _scatter_positions(anchor_top, anchor_left, boundary, direction, gap):
        grid[rr, cc] = MAGENTA
        r0, c0, r1, c1 = _inflate_box(rr, cc, rr, cc, (h, w), margin=0)
        _mark_box(moat, r0, c0, r1, c1)

    _place_square(grid, moat, square_top, square_left, side, hit_color)

    blocked = _reserve_signal_geometry(
        h,
        w,
        anchor_top,
        anchor_left,
        boundary,
        direction,
        gap,
        square_top,
        square_left,
        side,
    )
    if not _place_distractors(grid, moat, blocked, distractor_colors):
        return None

    try:
        _infer_signal_ray(grid, _find_anchor_bbox(grid))
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None

    # The unique hit color should become more common after extension.
    before = Counter(map(int, grid.flat))
    after = Counter(map(int, out.flat))
    if after[hit_color] <= before[hit_color]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
