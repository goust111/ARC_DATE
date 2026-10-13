from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

_COPY_COUNT_CHOICES = (3, 4, 4)
_GRID_SIZE_RANGE = (14, 22)
_MAX_GENERATION_ATTEMPTS = 400
_TEMPLATE_ATTEMPTS = 160
_SHAPE_ATTEMPTS = 120
_POSITION_ATTEMPTS = 120
_POSITION_RADIUS = 4
_POSITION_STEPS = tuple(
    (dr, dc)
    for dr in range(-2, 3)
    for dc in range(-2, 3)
    if (dr, dc) != (0, 0) and max(abs(dr), abs(dc)) <= 2
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_topological_order(
    colors: list[int], edges: set[tuple[int, int]]
) -> list[int] | None:
    above_to_below = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for above, below in edges:
        if below not in above_to_below[above]:
            above_to_below[above].add(below)
            indegree[below] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        top = sorted(color for color in remaining if indegree[color] == 0)
        if len(top) != 1:
            return None
        color = top[0]
        order.append(color)
        remaining.remove(color)
        for below in above_to_below[color]:
            indegree[below] -= 1
    return order


def _validate_assignment(
    grid: np.ndarray,
    copy_colors: list[int],
    assigned_anchors: dict[int, tuple[int, int]],
    template: set[tuple[int, int]],
) -> list[int] | None:
    anchor_set = set(assigned_anchors.values())
    edges: set[tuple[int, int]] = set()

    for color in copy_colors:
        anchor_r, anchor_c = assigned_anchors[color]
        for dr, dc in template:
            rr = anchor_r + dr
            cc = anchor_c + dc
            if not _in_bounds(grid.shape, rr, cc):
                return None

            value = int(grid[rr, cc])
            if (rr, cc) in anchor_set:
                if value != GREY:
                    return None
                continue
            if value == color:
                continue
            if value in copy_colors:
                edges.add((value, color))
                continue
            return None

    return _unique_topological_order(copy_colors, edges)


def _solve_scene(grid: np.ndarray) -> tuple[int, set[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    copy_colors = sorted(int(color) for color in np.unique(g) if color not in (bg, GREY))
    anchors = [tuple(map(int, coord)) for coord in np.argwhere(g == GREY)]

    assert len(copy_colors) == len(anchors) >= 2, "expected one grey anchor per colored copy"

    color_coords = {
        color: [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        for color in copy_colors
    }
    assert all(color_coords[color] for color in copy_colors), "each copy color must be visible"

    solutions: dict[tuple[tuple[tuple[int, int], ...], tuple[int, ...]], tuple[set[tuple[int, int]], list[int]]] = {}

    # The correct anchor-color matching is the unique one whose aligned union forms a
    # complete template and whose hidden cells induce a unique occlusion order.
    for perm in itertools.permutations(anchors):
        assigned_anchors = {color: perm[i] for i, color in enumerate(copy_colors)}
        template: set[tuple[int, int]] = {(0, 0)}
        for color in copy_colors:
            anchor_r, anchor_c = assigned_anchors[color]
            for rr, cc in color_coords[color]:
                template.add((rr - anchor_r, cc - anchor_c))

        order = _validate_assignment(g, copy_colors, assigned_anchors, template)
        if order is None:
            continue

        key = (tuple(sorted(template)), tuple(order))
        solutions[key] = (template, order)

    assert len(solutions) == 1, "expected a unique template reconstruction and layer order"
    template, order = next(iter(solutions.values()))
    return bg, template, order


def _render_output(
    bg: int, template: set[tuple[int, int]], bottom_color: int
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(template)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in template:
        out[dr - r0, dc - c0] = int(bottom_color)
    out[-r0, -c0] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a hidden template from aligned occluded copies and repaint it with the bottom copy's color.

    Input structure:
    - The background is black. Several translated copies of one unknown monochrome template are present, each in a distinct color.
    - Every copy contains one grey anchor pixel at the same relative template location, and anchors remain visible even when copies overlap.
    - No copy is fully visible, but the visible fragments from all copies together cover every template cell.

    Rule:
    - Try the anchor-to-color matchings, align each colored fragment to its candidate grey anchor, and take their union to recover the full template shape.
    - For every reconstructed template cell missing from one copy, read the color that appears there in the scene: a colored cover implies that copy lies below the visible color, while grey anchor collisions stay grey.
    - Keep the unique matching whose occlusion evidence yields a unique top-to-bottom layer order, then output the template's tight bbox filled with the bottom copy's color and the anchor cell in grey.

    Key insight:
    - The same aligned reconstruction solves both halves of the puzzle: it reveals the template mask, and it also tells which missing cells are true occlusions versus anchor exceptions, which is enough to recover the layer order.
    """
    bg, template, order = _solve_scene(np.asarray(grid, dtype=int))
    return _render_output(bg, template, order[-1])


def _grow_polyomino(size: int, height: int, width: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = [start]

    while len(cells) < size and frontier:
        base_r, base_c = random.choice(frontier)
        neighbors = [
            (base_r - 1, base_c),
            (base_r + 1, base_c),
            (base_r, base_c - 1),
            (base_r, base_c + 1),
        ]
        random.shuffle(neighbors)
        expanded = False
        for nr, nc in neighbors:
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                expanded = True
                break
        if not expanded:
            frontier.remove((base_r, base_c))

    return cells if len(cells) == size else None


def _random_template() -> set[tuple[int, int]]:
    for _ in range(_TEMPLATE_ATTEMPTS):
        height = random.randint(3, 5)
        width = random.randint(3, 5)
        min_area = max(6, height + width - 2)
        max_area = min(height * width - 1, 10)
        if min_area > max_area:
            continue

        for _ in range(_SHAPE_ATTEMPTS):
            size = random.randint(min_area, max_area)
            cells = _grow_polyomino(size, height, width)
            if cells is None:
                continue

            rows = [r for r, _c in cells]
            cols = [c for _r, c in cells]
            if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
                continue

            anchor_candidates = list(cells)
            random.shuffle(anchor_candidates)
            for anchor_r, anchor_c in anchor_candidates:
                template = {(r - anchor_r, c - anchor_c) for r, c in cells}
                if not (
                    any(dr < 0 for dr, _dc in template)
                    and any(dr > 0 for dr, _dc in template)
                    and any(dc < 0 for _dr, dc in template)
                    and any(dc > 0 for _dr, dc in template)
                ):
                    continue
                return template

    raise RuntimeError("failed to build a non-degenerate template")


def _random_positions(copy_count: int) -> list[tuple[int, int]] | None:
    for _ in range(_POSITION_ATTEMPTS):
        positions = [(0, 0)]
        while len(positions) < copy_count:
            base_r, base_c = random.choice(positions)
            step_r, step_c = random.choice(_POSITION_STEPS)
            candidate = (base_r + step_r, base_c + step_c)
            if candidate in positions:
                continue
            if max(abs(candidate[0]), abs(candidate[1])) > _POSITION_RADIUS:
                continue
            positions.append(candidate)
        return positions
    return None


def _choose_base_anchor(
    template: set[tuple[int, int]],
    positions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    min_dr = min(dr for dr, _dc in template)
    max_dr = max(dr for dr, _dc in template)
    min_dc = min(dc for _dr, dc in template)
    max_dc = max(dc for _dr, dc in template)
    pos_rows = [r for r, _c in positions]
    pos_cols = [c for _r, c in positions]

    base_r_lo = 1 - (min(pos_rows) + min_dr)
    base_r_hi = shape[0] - 2 - (max(pos_rows) + max_dr)
    base_c_lo = 1 - (min(pos_cols) + min_dc)
    base_c_hi = shape[1] - 2 - (max(pos_cols) + max_dc)
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None
    return (
        random.randint(base_r_lo, base_r_hi),
        random.randint(base_c_lo, base_c_hi),
    )


def _render_scene(
    template: set[tuple[int, int]],
    positions: list[tuple[int, int]],
    copy_colors: list[int],
    order: list[int],
    shape: tuple[int, int],
    base_anchor: tuple[int, int],
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    grid = np.full(shape, BG, dtype=int)
    anchors = {
        copy_colors[i]: (base_anchor[0] + positions[i][0], base_anchor[1] + positions[i][1])
        for i in range(len(copy_colors))
    }

    for color in reversed(order):
        anchor_r, anchor_c = anchors[color]
        for dr, dc in template:
            if (dr, dc) == (0, 0):
                continue
            grid[anchor_r + dr, anchor_c + dc] = color

    for anchor_r, anchor_c in anchors.values():
        grid[anchor_r, anchor_c] = GREY

    return grid, anchors


def _top_candidates(
    template: set[tuple[int, int]],
    positions: list[tuple[int, int]],
    copy_colors: list[int],
) -> list[int]:
    candidates: list[int] = []
    template_offsets = template - {(0, 0)}
    for i, (base_r, base_c) in enumerate(positions):
        if any(
            i != j and (other_r - base_r, other_c - base_c) in template_offsets
            for j, (other_r, other_c) in enumerate(positions)
        ):
            candidates.append(copy_colors[i])
    return candidates


def _case_is_valid(
    grid: np.ndarray,
    template: set[tuple[int, int]],
    anchors: dict[int, tuple[int, int]],
    copy_colors: list[int],
    order: list[int],
) -> bool:
    anchor_set = set(anchors.values())
    visible_union = {(0, 0)}
    edges: set[tuple[int, int]] = set()

    for color in copy_colors:
        anchor_r, anchor_c = anchors[color]
        visible_cells = 0
        hidden_cells = 0
        for dr, dc in template:
            if (dr, dc) == (0, 0):
                continue
            cell = (anchor_r + dr, anchor_c + dc)
            if cell in anchor_set:
                hidden_cells += 1
                continue

            value = int(grid[cell])
            if value == color:
                visible_cells += 1
                visible_union.add((dr, dc))
            else:
                hidden_cells += 1
                if value in copy_colors:
                    edges.add((value, color))
                else:
                    return False

        if visible_cells == 0 or hidden_cells == 0:
            return False

    if visible_union != template:
        return False
    if _unique_topological_order(copy_colors, edges) != order:
        return False

    try:
        bg, solved_template, solved_order = _solve_scene(grid)
    except AssertionError:
        return False

    if bg != BG or solved_template != template or solved_order != order:
        return False
    return np.array_equal(main(grid), _render_output(BG, template, order[-1]))


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        copy_count = random.choice(_COPY_COUNT_CHOICES)
        template = _random_template()
        positions = _random_positions(copy_count)
        if positions is None:
            continue

        copy_colors = random.sample(PALETTE, copy_count)
        top_candidates = _top_candidates(template, positions, copy_colors)
        if not top_candidates:
            continue

        top_color = random.choice(top_candidates)
        remaining = [color for color in copy_colors if color != top_color]
        random.shuffle(remaining)
        order = [top_color] + remaining

        shape = (
            random.randint(*_GRID_SIZE_RANGE),
            random.randint(*_GRID_SIZE_RANGE),
        )
        base_anchor = _choose_base_anchor(template, positions, shape)
        if base_anchor is None:
            continue

        grid, anchors = _render_scene(template, positions, copy_colors, order, shape, base_anchor)
        if not _case_is_valid(grid, template, anchors, copy_colors, order):
            continue

        return grid, _render_output(BG, template, order[-1])

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
