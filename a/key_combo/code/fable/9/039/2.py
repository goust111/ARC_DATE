from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
AXIS_KINDS = ("row", "col", "main_diag", "anti_diag")
SPRITE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def coords_to_array(coords: set[tuple[int, int]]) -> np.ndarray:
    return np.array(sorted(coords), dtype=int)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                cells.add((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "norm": normalize_coords(cells),
                    "bbox": bbox(cells),
                    "area": len(cells),
                }
            )
    return comps


def reflect_cells(
    cells: set[tuple[int, int]], axis: tuple[str, int | None], shape: tuple[int, int]
) -> set[tuple[int, int]] | None:
    h, w = shape
    kind, value = axis
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        if kind == "row":
            rr, cc = 2 * int(value) - r, c
        elif kind == "col":
            rr, cc = r, 2 * int(value) - c
        elif kind == "main_diag":
            if h != w:
                return None
            rr, cc = c, r
        elif kind == "anti_diag":
            if h != w:
                return None
            rr, cc = w - 1 - c, h - 1 - r
        else:
            raise ValueError(f"unknown axis kind: {kind}")
        if not in_bounds(shape, rr, cc):
            return None
        out.add((rr, cc))
    return out


def side_values(cells: set[tuple[int, int]], axis: tuple[str, int | None], n: int) -> list[int]:
    kind, value = axis
    if kind == "row":
        return [r - int(value) for r, _ in cells]
    if kind == "col":
        return [c - int(value) for _, c in cells]
    if kind == "main_diag":
        return [c - r for r, c in cells]
    if kind == "anti_diag":
        return [r + c - (n - 1) for r, c in cells]
    raise ValueError(f"unknown axis kind: {kind}")


def reflected_shape_variants(norm: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    cells = set(norm)
    _, _, r1, c1 = bbox(cells)
    h, w = r1 + 1, c1 + 1
    variants = []
    variants.append(normalize_coords({(h - 1 - r, c) for r, c in cells}))
    variants.append(normalize_coords({(r, w - 1 - c) for r, c in cells}))
    variants.append(normalize_coords({(c, r) for r, c in cells}))
    variants.append(normalize_coords({(w - 1 - c, h - 1 - r) for r, c in cells}))
    return set(variants)


def is_asymmetric(norm: tuple[tuple[int, int], ...]) -> bool:
    return norm not in reflected_shape_variants(norm)


def all_candidate_axes(
    color_cells: set[tuple[int, int]], ghost_cells: set[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[str, int | None]]:
    h, w = shape
    candidates: list[tuple[str, int | None]] = []
    for k in range(h):
        axis = ("row", k)
        if reflect_cells(color_cells, axis, shape) == ghost_cells:
            candidates.append(axis)
    for k in range(w):
        axis = ("col", k)
        if reflect_cells(color_cells, axis, shape) == ghost_cells:
            candidates.append(axis)
    if h == w:
        for axis in (("main_diag", None), ("anti_diag", None)):
            if reflect_cells(color_cells, axis, shape) == ghost_cells:
                candidates.append(axis)
    return candidates


def solve_scene(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, tuple[str, int | None]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = get_bg(g)
    comps = find_components(g, bg)
    ghosts = [i for i, comp in enumerate(comps) if comp["color"] == GREY]
    assert len(ghosts) == 2, "expected exactly two grey ghost components"

    non_grey_colors = sorted({comp["color"] for comp in comps if comp["color"] != GREY})
    assert len(non_grey_colors) == 2, "expected exactly two non-grey foreground colors"

    colored = [i for i, comp in enumerate(comps) if comp["color"] != GREY]
    remaining_colors = {comps[i]["color"] for i in colored}
    assert remaining_colors == set(non_grey_colors)

    edges: list[tuple[int, int, tuple[str, int | None]]] = []
    for ci in colored:
        for gi in ghosts:
            axes = all_candidate_axes(comps[ci]["cells"], comps[gi]["cells"], g.shape)
            for axis in axes:
                edges.append((ci, gi, axis))

    solutions = []
    for e1, e2 in itertools.permutations(edges, 2):
        c1, g1, a1 = e1
        c2, g2, a2 = e2
        if c1 == c2 or g1 == g2:
            continue
        if comps[c1]["color"] == comps[c2]["color"]:
            continue
        if a1[0] == a2[0]:
            continue
        used = {c1, c2, g1, g2}
        if len(used) != 4:
            continue
        canonical = tuple(sorted(((c1, g1, a1), (c2, g2, a2)), key=lambda x: x[1]))
        solutions.append(canonical)

    unique: list[tuple[int, int, tuple[str, int | None]]] = []
    for item in solutions:
        if item not in unique:
            unique.append(item)
    assert len(unique) == 1, "expected a unique sprite-ghost pairing with unique axes"
    return comps, list(unique[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer two hidden mirror axes from colored-sprite/grey-ghost pairs, recolor the ghosts, then mirror every other same-color object across its color's inferred axis.

    Input structure:
    - A dominant background contains exactly two grey ghost components and exactly two non-grey foreground colors.
    - For each of the two colors, one asymmetric sprite has a grey partner elsewhere that is its exact reflection across an undrawn full row, full column, main diagonal, or anti-diagonal axis.
    - Additional small 4-connected monochrome objects may appear, and each uses one of the two sprite colors.

    Rule:
    - Find which colored sprite matches each grey ghost by exact reflection and infer the unique hidden axis for each pair.
    - Recolor each grey ghost to its matched sprite color.
    - Leave the original sprites and small objects in place, and for every remaining small object draw a same-color reflected copy across the axis belonging to that object's color.

    Key insight:
    - The axes are never drawn; the only reliable way to recover them is to match each grey shape to the unique colored sprite whose reflected geometry lands exactly on it, then route every other object by color to the corresponding inferred axis.
    """
    g = np.asarray(grid, dtype=int)
    comps, pairs = solve_scene(g)
    bg = get_bg(g)
    out = g.copy()

    sprite_ids = set()
    color_to_axis: dict[int, tuple[str, int | None]] = {}
    for sprite_i, ghost_i, axis in pairs:
        sprite_color = int(comps[sprite_i]["color"])
        sprite_ids.add(sprite_i)
        color_to_axis[sprite_color] = axis
        for r, c in comps[ghost_i]["cells"]:
            out[r, c] = sprite_color

    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        if color == GREY or idx in sprite_ids:
            continue
        assert color in color_to_axis, "every non-generator colored object must share a sprite color"
        reflected = reflect_cells(comp["cells"], color_to_axis[color], g.shape)
        assert reflected is not None
        for r, c in reflected:
            assert int(out[r, c]) in {bg, color}, "mirrored copies must not overwrite another color"
            out[r, c] = color
    return out


def random_connected_shape(area_lo: int, area_hi: int, max_dim: int) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        target = random.randint(area_lo, area_hi)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    cell = (r + dr, c + dc)
                    if cell not in cells:
                        frontier.append(cell)
            cells.add(random.choice(frontier))
        norm = normalize_coords(cells)
        rs = [r for r, _ in norm]
        cs = [c for _, c in norm]
        if max(rs) + 1 > max_dim or max(cs) + 1 > max_dim:
            continue
        return norm
    raise RuntimeError("failed to sample connected shape")


def reserve_margin(cells: set[tuple[int, int]], n: int) -> set[tuple[int, int]]:
    reserved = set(cells)
    for r, c in list(cells):
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                reserved.add((rr, cc))
    return reserved


def place_patch_candidates(
    norm: tuple[tuple[int, int], ...],
    axis: tuple[str, int | None],
    n: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    patch = set(norm)
    _, _, r1, c1 = bbox(patch)
    h, w = r1 + 1, c1 + 1
    candidates = []
    for r0 in range(n - h + 1):
        for c0 in range(n - w + 1):
            placed = {(r + r0, c + c0) for r, c in patch}
            sides = side_values(placed, axis, n)
            if any(v == 0 for v in sides):
                continue
            if not (all(v > 0 for v in sides) or all(v < 0 for v in sides)):
                continue
            reflected = reflect_cells(placed, axis, (n, n))
            if reflected is None or placed & reflected:
                continue
            if placed & blocked or reflected & blocked:
                continue
            candidates.append((placed, reflected))
    random.shuffle(candidates)
    return candidates


def render_grid(n: int, bg: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = obj["color"]
    return grid


def expected_from_layout(
    n: int,
    bg: int,
    objects: list[dict],
    sprite_colors: tuple[int, int],
    ghost_ids: tuple[int, int],
    small_ids: list[int],
) -> np.ndarray:
    out = render_grid(n, bg, objects)
    for ghost_idx, color in zip(ghost_ids, sprite_colors):
        for r, c in objects[ghost_idx]["cells"]:
            out[r, c] = color
    for idx in small_ids:
        obj = objects[idx]
        for r, c in obj["mirror"]:
            out[r, c] = obj["color"]
    return out


def assignment_cost(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    ar0, ac0, ar1, ac1 = bbox(a)
    br0, bc0, br1, bc1 = bbox(b)
    ac = ((ar0 + ar1) / 2.0, (ac0 + ac1) / 2.0)
    bc = ((br0 + br1) / 2.0, (bc0 + bc1) / 2.0)
    return (ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2


def nearest_pairing_is_wrong(
    sprite1: set[tuple[int, int]],
    ghost1: set[tuple[int, int]],
    sprite2: set[tuple[int, int]],
    ghost2: set[tuple[int, int]],
) -> bool:
    true_cost = assignment_cost(sprite1, ghost1) + assignment_cost(sprite2, ghost2)
    swapped_cost = assignment_cost(sprite1, ghost2) + assignment_cost(sprite2, ghost1)
    return swapped_cost < true_cost


def forced_single_axis_layout_output(
    n: int,
    bg: int,
    objects: list[dict],
    sprite_colors: tuple[int, int],
    ghost_ids: tuple[int, int],
    small_ids: list[int],
    axis: tuple[str, int | None],
) -> np.ndarray | None:
    out = render_grid(n, bg, objects)
    for ghost_idx, color in zip(ghost_ids, sprite_colors):
        for r, c in objects[ghost_idx]["cells"]:
            out[r, c] = color
    for idx in small_ids:
        obj = objects[idx]
        reflected = reflect_cells(obj["cells"], axis, (n, n))
        if reflected is None:
            return None
        for r, c in reflected:
            if int(out[r, c]) not in {bg, int(obj["color"])}:
                return None
            out[r, c] = int(obj["color"])
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def generate_one() -> np.ndarray | None:
    n = random.randint(16, 24)
    bg = 0
    colors = random.sample(SPRITE_COLOR_POOL, 2)
    axis1_kind, axis2_kind = random.sample(AXIS_KINDS, 2)
    axis1 = pick_axis(axis1_kind, n)
    axis2 = pick_axis(axis2_kind, n)

    shape1 = sample_sprite_shape(None)
    shape2 = sample_sprite_shape(shape1)

    objects: list[dict] = []
    sprite_ids: list[int] = []
    ghost_ids: list[int] = []
    blocked: set[tuple[int, int]] = set()

    first_candidates = place_patch_candidates(shape1, axis1, n, blocked)
    if not first_candidates:
        return None

    chosen_pairs = None
    for sprite1, ghost1 in first_candidates[:80]:
        blocked1 = blocked | reserve_margin(sprite1 | ghost1, n)
        second_candidates = place_patch_candidates(shape2, axis2, n, blocked1)
        for sprite2, ghost2 in second_candidates[:80]:
            if nearest_pairing_is_wrong(sprite1, ghost1, sprite2, ghost2):
                chosen_pairs = ((sprite1, ghost1), (sprite2, ghost2))
                break
        if chosen_pairs is not None:
            break
    if chosen_pairs is None:
        return None

    for color, axis, pair in ((colors[0], axis1, chosen_pairs[0]), (colors[1], axis2, chosen_pairs[1])):
        sprite_cells, ghost_cells = pair
        sprite_id = len(objects)
        objects.append({"kind": "sprite", "color": color, "cells": sprite_cells, "axis": axis})
        ghost_id = len(objects)
        objects.append({"kind": "ghost", "color": GREY, "cells": ghost_cells, "axis": axis})
        sprite_ids.append(sprite_id)
        ghost_ids.append(ghost_id)
        blocked |= reserve_margin(sprite_cells | ghost_cells, n)

    small_ids: list[int] = []
    for color, axis in zip(colors, (axis1, axis2)):
        count = random.randint(1, 3)
        for _ in range(count):
            shape = sample_small_shape()
            candidates = place_patch_candidates(shape, axis, n, blocked)
            if not candidates:
                return None
            cells, mirror = random.choice(candidates)
            small_ids.append(len(objects))
            objects.append(
                {
                    "kind": "small",
                    "color": color,
                    "cells": cells,
                    "mirror": mirror,
                    "axis": axis,
                }
            )
            blocked |= reserve_margin(cells | mirror, n)

    inp = render_grid(n, bg, objects)
    expected = expected_from_layout(n, bg, objects, tuple(colors), tuple(ghost_ids), small_ids)

    try:
        comps, pairs = solve_scene(inp)
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    pair_by_color = {int(comps[sprite_i]["color"]): axis for sprite_i, _ghost_i, axis in pairs}
    if set(pair_by_color) != set(colors):
        return None

    for axis in (axis1, axis2):
        wrong = forced_single_axis_layout_output(
            n, bg, objects, tuple(colors), tuple(ghost_ids), small_ids, axis
        )
        if wrong is None or np.array_equal(wrong, expected):
            return None

    return inp


def pick_axis(kind: str, n: int) -> tuple[str, int | None]:
    if kind == "row":
        return ("row", random.randint(4, n - 5))
    if kind == "col":
        return ("col", random.randint(4, n - 5))
    if kind == "main_diag":
        return ("main_diag", None)
    if kind == "anti_diag":
        return ("anti_diag", None)
    raise ValueError(f"unknown axis kind: {kind}")


def sample_sprite_shape(other: tuple[tuple[int, int], ...] | None) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        norm = random_connected_shape(5, 8, 4)
        if not is_asymmetric(norm):
            continue
        if other is not None:
            if norm == other or norm in reflected_shape_variants(other):
                continue
        return norm
    raise RuntimeError("failed to sample sprite shape")


def sample_small_shape() -> tuple[tuple[int, int], ...]:
    if random.random() < 0.35:
        return ((0, 0),)
    return random_connected_shape(2, 3, 3)
