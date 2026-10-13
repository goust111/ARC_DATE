from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
LEGEND_SIZE = 2
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _sign(x: int) -> int:
    return 1 if x > 0 else -1


def _line_exclusive_start(start: Coord, end: Coord) -> list[Coord]:
    sr, sc = start
    er, ec = end
    assert sr == er or sc == ec
    dr = 0 if er == sr else _sign(er - sr)
    dc = 0 if ec == sc else _sign(ec - sc)
    out: list[Coord] = []
    r, c = sr + dr, sc + dc
    while True:
        out.append((r, c))
        if (r, c) == (er, ec):
            return out
        r += dr
        c += dc


def _route_for_order(start: Coord, anchor: Coord, order: str) -> list[Coord] | None:
    sr, sc = start
    ar, ac = anchor
    if sr == ar or sc == ac:
        return None
    if abs(sr - ar) < 2 or abs(sc - ac) < 2:
        return None

    dr = _sign(ar - sr)
    dc = _sign(ac - sc)

    if order == "hv":
        bend = (sr, ac)
        end = (ar - dr, ac)
    elif order == "vh":
        bend = (ar, sc)
        end = (ar, ac - dc)
    else:
        raise ValueError(f"unknown order: {order}")

    if bend == start or end == start or end == bend or end == anchor:
        return None

    route = _line_exclusive_start(start, bend)
    route.extend(_line_exclusive_start(bend, end))
    return route


def _order_bend(start: Coord, anchor: Coord, order: str) -> Coord:
    if order == "hv":
        return (start[0], anchor[1])
    if order == "vh":
        return (anchor[0], start[1])
    raise ValueError(f"unknown order: {order}")


def _other_order(order: str) -> str:
    return "vh" if order == "hv" else "hv"


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:LEGEND_SIZE, :LEGEND_SIZE] = True
    return mask


def _outside_singleton(grid: np.ndarray, color: int, legend_mask: np.ndarray) -> Coord:
    coords = np.argwhere((grid == int(color)) & ~legend_mask)
    assert coords.shape == (1, 2), f"expected one non-legend pixel for color {color}"
    return tuple(map(int, coords[0]))


def _route_is_clear(grid: np.ndarray, route: list[Coord], bg: int) -> bool:
    for r, c in route:
        if int(grid[r, c]) != int(bg):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Move each legend-row left color to the side of its row-mate anchor along the only clear one-bend path.

    Input structure:
    - The background is black, and the upper-left corner is a solid 2x2 legend
      with four distinct non-background colors.
    - Outside the legend, each legend color appears exactly once as an isolated
      marker pixel, and there may be extra isolated obstacle pixels in other colors.
    - In each legend row, the left-color marker is the mover and the right-color
      marker is the anchor; these two scene pixels are neither row-aligned nor
      column-aligned.

    Rule:
    - Keep the legend, anchor pixels, and obstacle pixels unchanged.
    - Process the two legend rows from top to bottom. For each row, move the
      left-color marker along a one-bend orthogonal path to the cell immediately
      beside the right-color anchor on the side facing the mover.
    - The only allowed bend orders are horizontal-then-vertical or
      vertical-then-horizontal; exactly one route is fully empty once earlier
      painted paths are treated as new obstacles. Clear the mover's origin and
      paint every traversed cell, including the endpoint, in the mover's color.

    Key insight:
    - The 2x2 legend encodes ordered mover-to-anchor instructions by row, and the
      obstacle layout uniquely selects which of the two L-shaped route orders is valid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend = g[:LEGEND_SIZE, :LEGEND_SIZE]
    legend_colors = [int(v) for v in legend.ravel()]
    assert len(set(legend_colors)) == 4
    assert all(color != bg for color in legend_colors)

    legend_mask = _legend_mask(g.shape)
    out = g.copy()

    for row in range(LEGEND_SIZE):
        mover_color = int(legend[row, 0])
        anchor_color = int(legend[row, 1])

        start = _outside_singleton(out, mover_color, legend_mask)
        anchor = _outside_singleton(out, anchor_color, legend_mask)

        routes = {}
        for order in ("hv", "vh"):
            route = _route_for_order(start, anchor, order)
            if route is not None and _route_is_clear(out, route, bg):
                routes[order] = route

        assert len(routes) == 1, "expected exactly one clear one-bend route"
        route = next(iter(routes.values()))

        out[start] = bg
        for r, c in route:
            out[r, c] = mover_color

    return out


def _is_isolated_input_cell(cell: Coord, occupied: set[Coord]) -> bool:
    return cell not in occupied and all(nei not in occupied for nei in _neighbors4(cell))


def _sample_isolated_cell(
    shape: tuple[int, int],
    occupied_input: set[Coord],
    forbidden: set[Coord],
    margin: int = 3,
    tries: int = 250,
) -> Coord | None:
    h, w = shape
    r_lo = margin
    r_hi = h - margin
    c_lo = margin
    c_hi = w - margin
    if r_lo >= r_hi or c_lo >= c_hi:
        return None

    for _ in range(tries):
        cell = (random.randrange(r_lo, r_hi), random.randrange(c_lo, c_hi))
        if cell in forbidden:
            continue
        if _is_isolated_input_cell(cell, occupied_input):
            return cell
    return None


def _route_avoids(route: list[Coord], blocked: set[Coord]) -> bool:
    return all(cell not in blocked for cell in route)


def _sample_static_pair(
    shape: tuple[int, int],
    occupied_input: set[Coord],
    blocked_exec: set[Coord],
    desired_order: str,
) -> tuple[Coord, Coord, list[Coord], list[Coord], Coord] | None:
    forbidden = set(blocked_exec)
    for _ in range(600):
        start = _sample_isolated_cell(shape, occupied_input, forbidden)
        if start is None:
            return None
        anchor = _sample_isolated_cell(shape, occupied_input | {start}, forbidden | {start})
        if anchor is None:
            continue

        desired = _route_for_order(start, anchor, desired_order)
        wrong = _route_for_order(start, anchor, _other_order(desired_order))
        if desired is None or wrong is None:
            continue
        if not _route_avoids(desired, blocked_exec):
            continue

        blocker = _order_bend(start, anchor, _other_order(desired_order))
        if blocker in blocked_exec or blocker in desired:
            continue
        if not _is_isolated_input_cell(blocker, occupied_input | {start, anchor}):
            continue
        if any(cell in blocked_exec for cell in wrong if cell != blocker):
            continue
        return start, anchor, desired, wrong, blocker
    return None


def _sample_dependent_pair(
    shape: tuple[int, int],
    occupied_input: set[Coord],
    path_blockers: list[Coord],
    desired_order: str,
) -> tuple[Coord, Coord, list[Coord], list[Coord]] | None:
    h, w = shape
    path_blocker_set = set(path_blockers)
    blocked_exec = occupied_input | path_blocker_set
    candidates = path_blockers[:]
    random.shuffle(candidates)

    for blocker in candidates:
        br, bc = blocker
        for _ in range(80):
            d1 = random.choice((-1, 1)) * random.randint(2, 5)
            d2 = random.choice((-1, 1)) * random.randint(2, 5)

            if desired_order == "hv":
                start = (br + d1, bc)
                anchor = (br, bc + d2)
            else:
                start = (br, bc + d1)
                anchor = (br + d2, bc)

            sr, sc = start
            ar, ac = anchor
            if not (3 <= sr < h - 3 and 3 <= sc < w - 3 and 3 <= ar < h - 3 and 3 <= ac < w - 3):
                continue
            if start in blocked_exec or anchor in blocked_exec:
                continue
            if not _is_isolated_input_cell(start, occupied_input):
                continue
            if not _is_isolated_input_cell(anchor, occupied_input | {start}):
                continue

            desired = _route_for_order(start, anchor, desired_order)
            wrong = _route_for_order(start, anchor, _other_order(desired_order))
            if desired is None or wrong is None:
                continue
            if not _route_avoids(desired, blocked_exec):
                continue

            wrong_hits = [cell for cell in wrong if cell in path_blocker_set]
            if not wrong_hits:
                continue
            if any(cell in occupied_input for cell in wrong if cell not in path_blocker_set):
                continue
            return start, anchor, desired, wrong
    return None


def _place_cell(grid: np.ndarray, cell: Coord, color: int) -> None:
    grid[cell] = int(color)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    grid = np.zeros((h, w), dtype=int)

    legend_colors = random.sample(PALETTE, 4)
    legend = np.array(legend_colors, dtype=int).reshape(LEGEND_SIZE, LEGEND_SIZE)
    grid[:LEGEND_SIZE, :LEGEND_SIZE] = legend

    occupied_input: set[Coord] = {(0, 0), (0, 1), (1, 0), (1, 1)}

    order1 = random.choice(("hv", "vh"))
    pair1 = _sample_static_pair((h, w), occupied_input, occupied_input.copy(), order1)
    if pair1 is None:
        return None
    start1, anchor1, path1, wrong1, blocker1 = pair1

    occupied_input.update({start1, anchor1, blocker1})

    use_dependent_second = random.random() < 0.5
    order2 = random.choice(("hv", "vh"))
    path1_set = set(path1)

    if use_dependent_second:
        pair2 = _sample_dependent_pair((h, w), occupied_input, path1, order2)
        if pair2 is None:
            return None
        start2, anchor2, path2, wrong2 = pair2
        blocker2 = None
    else:
        blocked_exec2 = occupied_input | path1_set
        pair2 = _sample_static_pair((h, w), occupied_input, blocked_exec2, order2)
        if pair2 is None:
            return None
        start2, anchor2, path2, wrong2, blocker2 = pair2
        occupied_input.add(blocker2)

    occupied_input.update({start2, anchor2})

    remaining_colors = [c for c in PALETTE if c not in legend_colors]
    random.shuffle(remaining_colors)
    obstacle_colors = iter(remaining_colors)

    _place_cell(grid, start1, legend_colors[0])
    _place_cell(grid, anchor1, legend_colors[1])
    _place_cell(grid, start2, legend_colors[2])
    _place_cell(grid, anchor2, legend_colors[3])
    _place_cell(grid, blocker1, next(obstacle_colors))
    if blocker2 is not None:
        _place_cell(grid, blocker2, next(obstacle_colors))

    forbidden_extras = occupied_input | path1_set | set(path2) | set(wrong1) | set(wrong2)
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        extra = _sample_isolated_cell((h, w), occupied_input, forbidden_extras)
        if extra is None:
            break
        color = next(obstacle_colors, None)
        if color is None:
            break
        _place_cell(grid, extra, color)
        occupied_input.add(extra)
        forbidden_extras.add(extra)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
