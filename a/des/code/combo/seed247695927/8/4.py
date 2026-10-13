from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


MAROON = 9
BLUE = 1
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
INPUT_COLORS = [0, 2, 3, 4, 5, 6, 7, 8]
PRIMARY_SCENES = ("single_empty", "double_empty", "hit_sprite")
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components8(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIRS8:
                    rr = r + dr
                    cc = c + dc
                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and not seen[rr, cc]
                        and int(grid[rr, cc]) == color
                    ):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append({"color": color, "cells": cells})
    return comps


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _sprite_info(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    assert len(cells) == 3, "every sprite must have exactly three pixels"
    vertices = []
    for vr, vc in cells:
        neighbors = []
        for rr, cc in cells:
            if (rr, cc) == (vr, vc):
                continue
            dr = rr - vr
            dc = cc - vc
            if max(abs(dr), abs(dc)) == 1:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            continue
        (dr1, dc1), (dr2, dc2) = neighbors
        man1 = abs(dr1) + abs(dc1)
        man2 = abs(dr2) + abs(dc2)
        is_l_vertex = man1 == man2 == 1 and dr1 != dr2 and dc1 != dc2
        is_v_vertex = (
            abs(dr1) == abs(dc1) == abs(dr2) == abs(dc2) == 1
            and (dr1 == dr2 or dc1 == dc2)
        )
        if is_l_vertex or is_v_vertex:
            vertices.append(((vr, vc), neighbors))
    assert len(vertices) == 1, "sprite must have a unique two-edge vertex"

    vertex, offsets = vertices[0]
    (dr1, dc1), (dr2, dc2) = offsets
    man1 = abs(dr1) + abs(dc1)
    man2 = abs(dr2) + abs(dc2)
    if man1 == man2 == 1:
        assert dr1 != dr2 and dc1 != dc2, "L sprite arms must be orthogonal"
    else:
        assert abs(dr1) == abs(dc1) == abs(dr2) == abs(dc2) == 1, "V sprite arms must be diagonal"
        assert dr1 == dr2 or dc1 == dc2, "V sprite diagonals must open in one cardinal direction"

    sum_r = dr1 + dr2
    sum_c = dc1 + dc2
    direction = (-_sgn(sum_r), -_sgn(sum_c))
    assert direction != (0, 0), "sprite direction must be non-zero"
    if man1 == man2 == 1:
        assert direction[0] != 0 and direction[1] != 0, "L sprite must shoot diagonally"
    else:
        assert (direction[0] == 0) != (direction[1] == 0), "V sprite must shoot axially"
    return vertex, direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire a projectile from each 3-pixel triangle sprite and mark collisions.

    Input structure:
    - The dominant color is background.
    - Every non-background object is a 3-pixel 8-connected sprite, either an axis-aligned L or a diagonal V.

    Rule:
    - The sprite's degree-2 pixel is its vertex.
    - The two edge directions determine the projectile direction by their outward bisector.
    - The projectile lands exactly 5 cells from the vertex.
    - Hitting an existing sprite recolors that whole sprite maroon (9).
    - A single projectile landing on empty background paints that cell maroon (9).
    - Multiple projectiles landing on the same empty cell paint that cell blue (1).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)
    comps = _components8(g, bg)
    assert comps, "expected at least one sprite"

    cell_to_comp: dict[tuple[int, int], int] = {}
    impacts_on_empty: Counter[tuple[int, int]] = Counter()
    hit_components: set[int] = set()

    for idx, comp in enumerate(comps):
        vertex, direction = _sprite_info(comp["cells"])
        vr, vc = vertex
        dr, dc = direction
        target = (vr + 5 * dr, vc + 5 * dc)
        tr, tc = target
        assert 0 <= tr < h and 0 <= tc < w, "projectile must stay inside the grid"
        for cell in comp["cells"]:
            cell_to_comp[cell] = idx

    for idx, comp in enumerate(comps):
        vertex, direction = _sprite_info(comp["cells"])
        vr, vc = vertex
        dr, dc = direction
        target = (vr + 5 * dr, vc + 5 * dc)
        hit = cell_to_comp.get(target)
        if hit is not None:
            hit_components.add(hit)
        else:
            impacts_on_empty[target] += 1

    out = g.copy()
    for idx in hit_components:
        for r, c in comps[idx]["cells"]:
            out[r, c] = MAROON
    for (r, c), count in impacts_on_empty.items():
        out[r, c] = BLUE if count >= 2 else MAROON
    return out


SPRITE_OFFSETS = {
    (-1, -1): ((0, 0), (1, 0), (0, 1)),
    (-1, 1): ((0, 0), (1, 0), (0, -1)),
    (1, -1): ((0, 0), (-1, 0), (0, 1)),
    (1, 1): ((0, 0), (-1, 0), (0, -1)),
    (-1, 0): ((0, 0), (1, -1), (1, 1)),
    (1, 0): ((0, 0), (-1, -1), (-1, 1)),
    (0, -1): ((0, 0), (-1, 1), (1, 1)),
    (0, 1): ((0, 0), (-1, -1), (1, -1)),
}


def _sprite_cells(vertex: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    vr, vc = vertex
    return [(vr + dr, vc + dc) for dr, dc in SPRITE_OFFSETS[direction]]


def _impact(vertex: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    vr, vc = vertex
    dr, dc = direction
    return vr + 5 * dr, vc + 5 * dc


def _in_bounds(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for dr, dc in DIRS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in occupied:
                return True
    return False


def _add_sprite(
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    vertex: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> bool:
    cells = _sprite_cells(vertex, direction)
    if not _in_bounds(cells, shape):
        return False
    if not (0 <= _impact(vertex, direction)[0] < shape[0] and 0 <= _impact(vertex, direction)[1] < shape[1]):
        return False
    if _touches_existing(cells, occupied, shape):
        return False
    sprites.append({"vertex": vertex, "direction": direction, "color": color, "cells": cells})
    occupied.update(cells)
    return True


def _random_vertex_for_direction(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    offsets = SPRITE_OFFSETS[direction]
    min_r = max(-dr for dr, _ in offsets)
    max_r = min(h - 1 - dr for dr, _ in offsets)
    min_c = max(-dc for _, dc in offsets)
    max_c = min(w - 1 - dc for _, dc in offsets)
    dr, dc = direction
    min_r = max(min_r, -5 * dr)
    max_r = min(max_r, h - 1 - 5 * dr)
    min_c = max(min_c, -5 * dc)
    max_c = min(max_c, w - 1 - 5 * dc)
    assert min_r <= max_r and min_c <= max_c
    return random.randint(min_r, max_r), random.randint(min_c, max_c)


def _scene_stats(sprites: list[dict]) -> tuple[int, int, int]:
    occupied = {}
    for idx, sprite in enumerate(sprites):
        for cell in sprite["cells"]:
            occupied[cell] = idx
    hit_components: set[int] = set()
    empty_counts: Counter[tuple[int, int]] = Counter()
    for sprite in sprites:
        target = _impact(sprite["vertex"], sprite["direction"])
        hit = occupied.get(target)
        if hit is not None:
            hit_components.add(hit)
        else:
            empty_counts[target] += 1
    single_empty = sum(1 for count in empty_counts.values() if count == 1)
    blue_empty = sum(1 for count in empty_counts.values() if count >= 2)
    return len(hit_components), single_empty, blue_empty


def _render(shape: tuple[int, int], bg: int, sprites: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for sprite in sprites:
        for r, c in sprite["cells"]:
            grid[r, c] = sprite["color"]
    return grid


def _primary_single_empty(shape: tuple[int, int], sprites: list[dict], occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    for _ in range(80):
        direction = random.choice(DIRS8)
        vertex = _random_vertex_for_direction(shape, direction)
        target = _impact(vertex, direction)
        if target in occupied:
            continue
        color = colors[-1]
        if _add_sprite(sprites, occupied, shape, vertex, direction, color):
            colors.pop()
            return True
    return False


def _primary_double_empty(shape: tuple[int, int], sprites: list[dict], occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    h, w = shape
    dirs = list(DIRS8)
    for _ in range(160):
        target = (random.randrange(h), random.randrange(w))
        random.shuffle(dirs)
        chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for direction in dirs:
            dr, dc = direction
            vertex = (target[0] - 5 * dr, target[1] - 5 * dc)
            cells = _sprite_cells(vertex, direction)
            if not _in_bounds(cells, shape):
                continue
            if _touches_existing(cells, occupied, shape):
                continue
            chosen.append((vertex, direction))
            if len(chosen) == 2:
                break
        if len(chosen) != 2 or target in occupied:
            continue
        color_a = colors[-1]
        color_b = colors[-2]
        if _add_sprite(sprites, occupied, shape, chosen[0][0], chosen[0][1], color_a) and _add_sprite(
            sprites, occupied, shape, chosen[1][0], chosen[1][1], color_b
        ):
            colors.pop()
            colors.pop()
            return True
        raise AssertionError("double placement should be atomic after precheck")
    return False


def _primary_hit_sprite(shape: tuple[int, int], sprites: list[dict], occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    directions = list(DIRS8)
    for _ in range(160):
        target_dir = random.choice(directions)
        target_vertex = _random_vertex_for_direction(shape, target_dir)
        target_cells = _sprite_cells(target_vertex, target_dir)
        if _touches_existing(target_cells, occupied, shape):
            continue
        shooter_color = colors[-1]
        target_color = colors[-2]
        sprites.append({"vertex": target_vertex, "direction": target_dir, "color": target_color, "cells": target_cells})
        occupied.update(target_cells)
        target_cell = random.choice(target_cells)
        placed = False
        random.shuffle(directions)
        for direction in directions:
            vertex = (target_cell[0] - 5 * direction[0], target_cell[1] - 5 * direction[1])
            cells = _sprite_cells(vertex, direction)
            if not _in_bounds(cells, shape):
                continue
            if _touches_existing(cells, occupied, shape):
                continue
            sprites.append({"vertex": vertex, "direction": direction, "color": shooter_color, "cells": cells})
            occupied.update(cells)
            colors.pop()
            colors.pop()
            placed = True
            break
        if placed:
            return True
        sprites.pop()
        for cell in target_cells:
            occupied.remove(cell)
    return False


def _add_safe_extra(shape: tuple[int, int], sprites: list[dict], occupied: set[tuple[int, int]], colors: list[int]) -> bool:
    existing_impacts = {_impact(sprite["vertex"], sprite["direction"]) for sprite in sprites}
    for _ in range(100):
        direction = random.choice(DIRS8)
        vertex = _random_vertex_for_direction(shape, direction)
        target = _impact(vertex, direction)
        if target in occupied or target in existing_impacts:
            continue
        color = colors[-1]
        if _add_sprite(sprites, occupied, shape, vertex, direction, color):
            colors.pop()
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shape = (random.randint(12, 18), random.randint(12, 18))
        scene_kind = random.choice(PRIMARY_SCENES)
        bg = random.choice(INPUT_COLORS)
        colors = random.sample([c for c in INPUT_COLORS if c != bg], k=random.randint(2, 5))
        sprites: list[dict] = []
        occupied: set[tuple[int, int]] = set()

        ok = False
        if scene_kind == "single_empty":
            ok = _primary_single_empty(shape, sprites, occupied, colors)
        elif scene_kind == "double_empty":
            if len(colors) < 2:
                continue
            ok = _primary_double_empty(shape, sprites, occupied, colors)
        else:
            if len(colors) < 2:
                continue
            ok = _primary_hit_sprite(shape, sprites, occupied, colors)
        if not ok:
            continue

        extra_target = random.randint(0, min(2, len(colors)))
        for _extra in range(extra_target):
            if not _add_safe_extra(shape, sprites, occupied, colors):
                break

        hit_count, single_empty, blue_empty = _scene_stats(sprites)
        if scene_kind == "single_empty" and not (hit_count == 0 and blue_empty == 0 and single_empty >= 1):
            continue
        if scene_kind == "double_empty" and not (hit_count == 0 and blue_empty >= 1):
            continue
        if scene_kind == "hit_sprite" and not (hit_count >= 1 and blue_empty == 0):
            continue

        grid = _render(shape, bg, sprites)
        out = main(grid)
        assert isinstance(out, np.ndarray)
        return grid

    raise RuntimeError("failed to generate valid input")
