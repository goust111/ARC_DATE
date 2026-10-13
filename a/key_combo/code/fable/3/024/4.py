from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
Route = tuple[str, int]

BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_mask(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(g: np.ndarray, bg: int) -> list[int]:
    bottom = g[-1]
    end = 0
    while end < bottom.size and int(bottom[end]) != bg:
        end += 1

    assert end >= 2, "bottom-left legend must contain at least two colors"
    assert np.all(bottom[end:] == bg), "legend row must be blank after the left legend string"

    seq = [int(v) for v in bottom[:end]]
    assert len(seq) == len(set(seq)), "legend colors must be unique"
    assert bg not in seq and GREY not in seq, "legend cannot use background or grey"
    return seq


def _bar_mask_and_bbox(board: np.ndarray) -> tuple[np.ndarray, BBox]:
    mask = board == GREY
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub), "grey attractor must be a solid bar"
    assert int(np.count_nonzero(mask)) == int(sub.size), "expected exactly one grey bar"
    assert (r0 == r1) ^ (c0 == c1), "grey attractor must be one-cell-thick"
    assert max(r1 - r0 + 1, c1 - c0 + 1) >= 4, "grey bar is too short"
    return mask, (r0, c0, r1, c1)


def _parse_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, list[int], np.ndarray, BBox, dict[int, Coord]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    h, _w = g.shape
    assert h >= 8, "expected board, spacer row, and legend row"

    bg = _most_common_color(g)
    assert np.all(g[-2] == bg), "row above the legend must be empty background"
    legend = _parse_legend(g, bg)

    board = g[:-2]
    bar_mask, bar_bbox = _bar_mask_and_bbox(board)

    foreground = set(map(int, np.unique(board))) - {bg, GREY}
    assert foreground == set(legend), "board colors must match the legend exactly"

    pieces: dict[int, Coord] = {}
    for color in legend:
        coords = np.argwhere(board == color)
        assert len(coords) == 1, "each legend color must mark exactly one single-pixel piece"
        r, c = coords[0]
        pieces[int(color)] = (int(r), int(c))

    return g, bg, legend, bar_mask, bar_bbox, pieces


def _slide_destination(start: Coord, occupied: np.ndarray) -> Coord:
    h, w = occupied.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w, "piece starts outside the board"
    assert not occupied[sr, sc], "piece starts inside the accumulated mass"

    candidates: list[tuple[int, Coord]] = []
    for dr, dc in DIRS4:
        r, c = sr, sc
        steps_to_hit = 0
        while True:
            r += dr
            c += dc
            steps_to_hit += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if occupied[r, c]:
                travel = steps_to_hit - 1
                candidates.append((travel, (sr + dr * travel, sc + dc * travel)))
                break

    assert candidates, "piece must see the accumulated mass in a cardinal direction"
    candidates.sort(key=lambda item: item[0])
    assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], (
        "nearest cardinal attraction direction must be unique"
    )
    return candidates[0][1]


def _solve_with_order(grid: np.ndarray, order: list[int]) -> np.ndarray:
    g, bg, legend, bar_mask, _bar_bbox, pieces = _parse_grid(grid)
    assert set(order) == set(legend) and len(order) == len(legend), "order must list each piece color once"

    board_h = g.shape[0] - 2
    out = np.full_like(g, bg)
    out[:board_h][bar_mask] = GREY
    out[-1, :] = g[-1, :]

    occupied = bar_mask.copy()
    for color in order:
        dst = _slide_destination(pieces[int(color)], occupied)
        assert not occupied[dst], "piece destination is already occupied"
        occupied[dst] = True
        out[dst] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored single-pixel pieces to the central grey bar in the order named by the bottom legend.

    Input structure:
    - A central one-cell-thick grey bar sits in the board area on a background field.
    - The bottom row begins with a contiguous color legend string, and the row above it is blank background.
    - Each legend color appears exactly once more in the board as a single-pixel colored piece.
    - Some pieces share the same cardinal docking route toward the grey bar.

    Rule:
    - Read the bottom legend from left to right and process those colored pieces in that exact order.
    - For the current piece, choose the unique cardinal direction that reaches the grey bar plus already docked pieces in the fewest steps.
    - Slide the piece along that direction until the next step would collide, leaving it in its original color.
    - The bottom legend row is preserved unchanged in the output.

    Key insight:
    - The edge legend is a timetable, not a palette: earlier colors become obstacles, so the legend order determines which color lands inside or outside on shared routes.
    """
    _g, _bg, legend, _bar_mask, _bar_bbox, _pieces = _parse_grid(grid)
    return _solve_with_order(grid, legend)


def _route_capacity(bar: BBox, board_shape: tuple[int, int], route: Route) -> int:
    side, _anchor = route
    h, w = board_shape
    r0, c0, r1, c1 = bar
    if side == "top":
        return r0
    if side == "bottom":
        return h - 1 - r1
    if side == "left":
        return c0
    if side == "right":
        return w - 1 - c1
    raise ValueError(f"unknown route side {side!r}")


def _anchors_for_side(bar: BBox, side: str) -> list[int]:
    r0, c0, r1, c1 = bar
    if side in ("top", "bottom"):
        return list(range(c0, c1 + 1))
    if side in ("left", "right"):
        return list(range(r0, r1 + 1))
    raise ValueError(f"unknown side {side!r}")


def _route_start(bar: BBox, route: Route, dist: int) -> Coord:
    side, anchor = route
    r0, c0, r1, c1 = bar
    if side == "top":
        return r0 - dist, anchor
    if side == "bottom":
        return r1 + dist, anchor
    if side == "left":
        return anchor, c0 - dist
    if side == "right":
        return anchor, c1 + dist
    raise ValueError(f"unknown route side {side!r}")


def _pick_bar(board_h: int, board_w: int) -> BBox | None:
    margin = 4
    orientation = random.choice(("horizontal", "vertical"))
    if orientation == "horizontal":
        max_len = min(10, board_w - 2 * margin)
        if max_len < 5 or board_h < 2 * margin + 1:
            return None
        length = random.randint(5, max_len)
        r = random.randint(margin, board_h - margin - 1)
        c0 = random.randint(margin, board_w - margin - length)
        return (r, c0, r, c0 + length - 1)

    max_len = min(10, board_h - 2 * margin)
    if max_len < 5 or board_w < 2 * margin + 1:
        return None
    length = random.randint(5, max_len)
    r0 = random.randint(margin, board_h - margin - length)
    c = random.randint(margin, board_w - margin - 1)
    return (r0, c, r0 + length - 1, c)


def _spaced_distances(capacity: int, count: int) -> list[int] | None:
    choices = list(range(2, capacity + 1))
    if len(choices) < count:
        return None
    for _ in range(80):
        vals = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _initial_bar_distance(start: Coord, bar_mask: np.ndarray) -> int:
    candidates: list[int] = []
    h, w = bar_mask.shape
    sr, sc = start
    for dr, dc in DIRS4:
        r, c = sr, sc
        steps_to_hit = 0
        while True:
            r += dr
            c += dc
            steps_to_hit += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if bar_mask[r, c]:
                candidates.append(steps_to_hit - 1)
                break
    assert candidates, "piece must have an initial route to the grey bar"
    return min(candidates)


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        right = main(grid)
        _g, _bg, legend, bar_mask, _bar_bbox, pieces = _parse_grid(grid)

        distance_order = sorted(
            legend,
            key=lambda color: (_initial_bar_distance(pieces[int(color)], bar_mask), pieces[int(color)]),
        )
        row_major_order = sorted(legend, key=lambda color: pieces[int(color)])
        if distance_order == legend or row_major_order == legend:
            return False

        by_distance = _solve_with_order(grid, distance_order)
        by_position = _solve_with_order(grid, row_major_order)
        if np.array_equal(right, by_distance) or np.array_equal(right, by_position):
            return False

        for color in legend:
            old = pieces[int(color)]
            new_coords = np.argwhere(right[:-2] == int(color))
            if len(new_coords) != 1:
                return False
            if tuple(map(int, new_coords[0])) == old:
                return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    board_h = random.randint(16, 25)
    board_w = random.randint(17, 28)
    bar = _pick_bar(board_h, board_w)
    if bar is None:
        return None

    conflict_side = random.choice(("bottom", "right"))
    conflict_routes = [
        (conflict_side, anchor)
        for anchor in _anchors_for_side(bar, conflict_side)
        if _route_capacity(bar, (board_h, board_w), (conflict_side, anchor)) >= 4
    ]
    if not conflict_routes:
        return None
    conflict_route = random.choice(conflict_routes)
    conflict_capacity = _route_capacity(bar, (board_h, board_w), conflict_route)
    conflict_count = 3 if conflict_capacity >= 6 and random.random() < 0.45 else 2
    conflict_distances = _spaced_distances(conflict_capacity, conflict_count)
    if conflict_distances is None:
        return None

    extra_routes: list[Route] = []
    for side in ("top", "left"):
        for anchor in _anchors_for_side(bar, side):
            route = (side, anchor)
            if route != conflict_route and _route_capacity(bar, (board_h, board_w), route) >= 2:
                extra_routes.append(route)
    random.shuffle(extra_routes)
    max_extra = min(len(extra_routes), len(COLORS) - conflict_count)
    if max_extra < 2:
        return None
    extra_count = random.randint(2, max_extra)

    piece_count = conflict_count + extra_count
    colors = random.sample(COLORS, piece_count)
    conflict_colors = colors[:conflict_count]
    extra_colors = colors[conflict_count:]

    pieces: list[tuple[int, Coord, Route, int]] = []
    for color, dist in zip(conflict_colors, conflict_distances):
        pieces.append((int(color), _route_start(bar, conflict_route, dist), conflict_route, int(dist)))

    for color, route in zip(extra_colors, extra_routes[:extra_count]):
        cap = _route_capacity(bar, (board_h, board_w), route)
        dist = random.randint(2, cap)
        pieces.append((int(color), _route_start(bar, route, dist), route, int(dist)))

    starts = [start for _color, start, _route, _dist in pieces]
    if len(starts) != len(set(starts)):
        return None

    grid = np.full((board_h + 2, board_w), BG, dtype=int)
    r0, c0, r1, c1 = bar
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for color, (r, c), _route, _dist in pieces:
        if not (0 <= r < board_h and 0 <= c < board_w) or int(grid[r, c]) != BG:
            return None
        grid[r, c] = int(color)

    conflict_by_far_first = [
        color
        for color, _start, _route, _dist in sorted(
            (p for p in pieces if p[2] == conflict_route),
            key=lambda item: item[3],
            reverse=True,
        )
    ]
    extras = [color for color, _start, route, _dist in pieces if route != conflict_route]
    random.shuffle(extras)

    legend: list[int] = []
    for color in conflict_by_far_first:
        while extras and random.random() < 0.4:
            legend.append(extras.pop())
        legend.append(color)
    legend.extend(extras)
    if len(legend) != piece_count or len(set(legend)) != piece_count:
        return None

    grid[-1, : len(legend)] = np.asarray(legend, dtype=int)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid timetable-legend attraction input")
