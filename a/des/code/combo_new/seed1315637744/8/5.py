import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
BODY_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _scale(vec, k):
    return (vec[0] * k, vec[1] * k)


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r, c):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape, cell):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _components8(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _body_variants(direction):
    dr, dc = direction
    back = (-dr, -dc)
    perp = (-dc, dr)
    return [
        {
            back,
            _scale(back, 2),
            _scale(back, 3),
            _add(back, perp),
            _add(back, _scale(perp, -1)),
        },
        {
            back,
            _scale(back, 2),
            _scale(back, 3),
            perp,
            _scale(perp, -1),
        },
        {
            back,
            _scale(back, 2),
            _add(back, perp),
            _add(back, _scale(perp, -1)),
        },
        {
            back,
            _scale(back, 2),
            perp,
            _scale(perp, -1),
        },
    ]


def _extract_arrows(grid):
    g = np.asarray(grid, dtype=int)
    mask = (g != BLACK) & (g != GREY)
    arrows = []
    for comp in _components8(mask):
        values = [int(g[r, c]) for r, c in comp]
        azure_cells = [(r, c) for r, c in comp if int(g[r, c]) == AZURE]
        assert len(azure_cells) == 1, "each arrow component must contain exactly one azure tip"
        tip = azure_cells[0]

        body_colors = {v for v in values if v != AZURE}
        assert len(body_colors) == 1, "each arrow component must contain exactly one body color"
        body_color = next(iter(body_colors))
        assert body_color not in (BLACK, GREY, AZURE), "invalid arrow body color"

        rel_body = {
            (r - tip[0], c - tip[1])
            for r, c in comp
            if (r, c) != tip
        }

        matches = []
        for name, direction in DIRS.items():
            for variant in _body_variants(direction):
                if rel_body == variant:
                    matches.append((name, direction))
                    break
        assert len(matches) == 1, "arrow geometry must determine a unique direction"
        name, direction = matches[0]
        arrows.append(
            {
                "tip": tip,
                "color": body_color,
                "direction": direction,
                "direction_name": name,
                "cells": set(comp),
            }
        )
    return arrows


def _trace_projection(tip, direction, obstacle_mask, shape):
    r, c = tip
    dr, dc = direction
    draw_here = True
    drawn = []
    visited = []
    while 0 <= r < shape[0] and 0 <= c < shape[1] and not obstacle_mask[r, c]:
        visited.append((r, c))
        if draw_here:
            drawn.append((r, c))
        draw_here = not draw_here
        r += dr
        c += dc
    reason = "boundary"
    if 0 <= r < shape[0] and 0 <= c < shape[1]:
        reason = "obstacle"
    return drawn, visited, reason


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project striped color rays from arrow glyphs until they hit grey obstacles or the grid edge.

    Input structure:
    - The background is black.
    - Grey pixels form fixed obstacle blocks of arbitrary connected shape.
    - Each arrow marker contains one azure tip plus one non-black non-grey body color; the body color is the projected stripe color and the tip/body geometry encodes one of the four cardinal directions.

    Rule:
    - Remove every arrow marker, turning all of its pixels black.
    - For each arrow, start at the former tip cell and move one step at a time in the arrow direction.
    - Paint every other visited cell with the arrow's body color, leave the alternating cells black, and stop before any grey obstacle cell or the grid boundary.
    - When multiple stripes paint the same cell, later arrows in top-to-bottom then left-to-right tip order overwrite earlier ones.

    Key insight:
    - The arrow simultaneously specifies the start cell, the travel direction, and the color of an alternating projection ray.
    """
    g = np.asarray(grid, dtype=int)
    obstacle_mask = g == GREY
    arrows = sorted(_extract_arrows(g), key=lambda arrow: arrow["tip"])

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    for arrow in arrows:
        drawn, _, _ = _trace_projection(arrow["tip"], arrow["direction"], obstacle_mask, g.shape)
        for r, c in drawn:
            out[r, c] = arrow["color"]
    return out


def _canonical_arrow_cells(tip, direction_name, color):
    direction = DIRS[direction_name]
    body = _body_variants(direction)[0]
    cells = {tip: AZURE}
    for dr, dc in body:
        cells[(tip[0] + dr, tip[1] + dc)] = color
    return cells


def _halo(cells, shape):
    out = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    out.add((rr, cc))
    return out


def _steps_to_boundary(tip, direction, shape):
    r, c = tip
    dr, dc = direction
    steps = 0
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        steps += 1
        r += dr
        c += dc
    return steps


def _can_place_arrow(cells, arrow_cells, shape, direction, min_runway):
    coords = set(cells)
    if not all(_in_bounds(shape, cell) for cell in coords):
        return False
    if coords & arrow_cells:
        return False
    if _halo(coords, shape) & arrow_cells:
        return False
    tip = next(cell for cell, value in cells.items() if value == AZURE)
    if _steps_to_boundary(tip, direction, shape) < min_runway:
        return False
    return True


def _random_arrow_tip(rng, shape, direction_name):
    h, w = shape
    if direction_name == "up":
        return (rng.randint(0, h - 5), rng.randint(1, w - 2))
    if direction_name == "down":
        return (rng.randint(4, h - 1), rng.randint(1, w - 2))
    if direction_name == "left":
        return (rng.randint(1, h - 2), rng.randint(0, w - 5))
    return (rng.randint(1, h - 2), rng.randint(4, w - 1))


def _place_random_arrow(rng, shape, arrow_cells, color, direction_name, min_runway=5):
    direction = DIRS[direction_name]
    for _ in range(160):
        tip = _random_arrow_tip(rng, shape, direction_name)
        cells = _canonical_arrow_cells(tip, direction_name, color)
        if _can_place_arrow(cells, arrow_cells, shape, direction, min_runway):
            return {
                "tip": tip,
                "color": color,
                "direction": direction,
                "direction_name": direction_name,
                "cells": set(cells),
                "paint": cells,
            }
    return None


def _tip_from_intersection(intersection, direction, distance):
    return (intersection[0] - direction[0] * distance, intersection[1] - direction[1] * distance)


def _place_cross_pair(rng, shape, arrow_cells, colors):
    h, w = shape
    h_name = rng.choice(["left", "right"])
    v_name = rng.choice(["up", "down"])
    h_dir = DIRS[h_name]
    v_dir = DIRS[v_name]
    for _ in range(200):
        intersection = (rng.randint(4, h - 5), rng.randint(4, w - 5))
        h_dist = rng.choice([2, 4, 6])
        v_dist = rng.choice([2, 4, 6])
        h_tip = _tip_from_intersection(intersection, h_dir, h_dist)
        v_tip = _tip_from_intersection(intersection, v_dir, v_dist)
        if h_tip == v_tip:
            continue
        first = _canonical_arrow_cells(h_tip, h_name, colors[0])
        second = _canonical_arrow_cells(v_tip, v_name, colors[1])
        if not _can_place_arrow(first, arrow_cells, shape, h_dir, h_dist + 3):
            continue
        if not _can_place_arrow(second, arrow_cells | set(first), shape, v_dir, v_dist + 3):
            continue
        return [
            {
                "tip": h_tip,
                "color": colors[0],
                "direction": h_dir,
                "direction_name": h_name,
                "cells": set(first),
                "paint": first,
            },
            {
                "tip": v_tip,
                "color": colors[1],
                "direction": v_dir,
                "direction_name": v_name,
                "cells": set(second),
                "paint": second,
            },
        ]
    return None


def _grow_obstacle(rng, shape, forbidden, anchor=None, size_range=(1, 6)):
    h, w = shape
    for _ in range(120):
        if anchor is None:
            seed = (rng.randint(0, h - 1), rng.randint(0, w - 1))
            if seed in forbidden:
                continue
        else:
            seed = anchor
            if seed in forbidden:
                return None
        target = rng.randint(size_range[0], size_range[1])
        comp = {seed}
        frontier = [cell for cell in _neighbors4(*seed) if _in_bounds(shape, cell) and cell not in forbidden]
        while frontier and len(comp) < target:
            cell = frontier.pop(rng.randrange(len(frontier)))
            if cell in comp or cell in forbidden:
                continue
            comp.add(cell)
            for nbr in _neighbors4(*cell):
                if _in_bounds(shape, nbr) and nbr not in comp and nbr not in forbidden and nbr not in frontier:
                    frontier.append(nbr)
        if anchor is not None and anchor not in comp:
            continue
        return comp
    return None


def _details_for_arrows(arrows, obstacle_cells, shape):
    obstacle_mask = np.zeros(shape, dtype=bool)
    for r, c in obstacle_cells:
        obstacle_mask[r, c] = True
    details = []
    for arrow in sorted(arrows, key=lambda item: item["tip"]):
        drawn, visited, reason = _trace_projection(arrow["tip"], arrow["direction"], obstacle_mask, shape)
        details.append(
            {
                "tip": arrow["tip"],
                "drawn": drawn,
                "visited": visited,
                "reason": reason,
            }
        )
    return details


def _place_deliberate_blocker(rng, arrows, obstacle_cells, shape, protected_cells):
    arrow_cells = set().union(*(arrow["cells"] for arrow in arrows))
    details = _details_for_arrows(arrows, obstacle_cells, shape)
    candidates = []
    for detail in details:
        for distance, cell in enumerate(detail["visited"]):
            if distance < 3:
                continue
            if cell in arrow_cells or cell in protected_cells or cell in obstacle_cells:
                continue
            candidates.append(cell)
    rng.shuffle(candidates)
    for anchor in candidates:
        blocker = _grow_obstacle(
            rng,
            shape,
            arrow_cells | obstacle_cells | protected_cells,
            anchor=anchor,
            size_range=(1, 4),
        )
        if blocker is not None:
            return blocker
    return None


def _build_grid(shape, arrows, obstacle_cells):
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in obstacle_cells:
        grid[r, c] = GREY
    for arrow in arrows:
        for cell, value in arrow["paint"].items():
            grid[cell] = value
    return grid


def _visible_cross_cells(details):
    crosses = set()
    for i in range(len(details)):
        first = set(details[i]["drawn"])
        for j in range(i + 1, len(details)):
            crosses |= first & set(details[j]["drawn"])
    return crosses


def _validate_grid(grid, expect_cross, expect_block):
    arrows = _extract_arrows(grid)
    if len(arrows) < 2:
        return False
    obstacle_cells = set(map(tuple, np.argwhere(grid == GREY)))
    if not obstacle_cells:
        return False
    details = _details_for_arrows(arrows, obstacle_cells, grid.shape)
    painted = sum(len(detail["drawn"]) for detail in details)
    if painted < len(arrows) + 4:
        return False
    crosses = _visible_cross_cells(details)
    if expect_cross and not crosses:
        return False
    if expect_block and not any(detail["reason"] == "obstacle" for detail in details):
        return False
    if not any(detail["reason"] == "boundary" for detail in details):
        return False
    out = main(grid)
    if np.array_equal(grid, out):
        return False
    return True


def _gen_attempt(rng):
    shape = (rng.randint(12, 24), rng.randint(12, 24))
    include_cross = rng.random() < 0.5
    include_block = rng.random() < 0.5
    arrow_count = rng.randint(3, 5)
    if include_cross:
        arrow_count = max(arrow_count, 2)

    colors = rng.sample(BODY_COLORS, k=arrow_count)
    arrows = []
    arrow_cells = set()

    protected_crosses = set()
    color_index = 0
    if include_cross:
        pair = _place_cross_pair(rng, shape, arrow_cells, colors[:2])
        if pair is None:
            return None
        arrows.extend(pair)
        arrow_cells |= set().union(*(arrow["cells"] for arrow in pair))
        color_index = 2
        pair_details = _details_for_arrows(pair, set(), shape)
        protected_crosses = _visible_cross_cells(pair_details)
        if not protected_crosses:
            return None

    direction_names = list(DIRS)
    rng.shuffle(direction_names)
    while len(arrows) < arrow_count:
        if direction_names:
            direction_name = direction_names.pop()
        else:
            direction_name = rng.choice(list(DIRS))
        arrow = _place_random_arrow(rng, shape, arrow_cells, colors[color_index], direction_name)
        if arrow is None:
            return None
        arrows.append(arrow)
        arrow_cells |= arrow["cells"]
        color_index += 1

    obstacle_cells = set()
    if include_block:
        blocker = _place_deliberate_blocker(rng, arrows, obstacle_cells, shape, protected_crosses)
        if blocker is None:
            return None
        obstacle_cells |= blocker

    extra_obstacles = rng.randint(1, 3)
    for _ in range(extra_obstacles):
        comp = _grow_obstacle(rng, shape, arrow_cells | obstacle_cells | protected_crosses, size_range=(1, 7))
        if comp is None:
            continue
        obstacle_cells |= comp

    grid = _build_grid(shape, arrows, obstacle_cells)
    if not _validate_grid(grid, include_cross, include_block):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-arrow projection input")
