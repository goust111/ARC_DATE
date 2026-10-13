from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

NEIGHBORS4 = tuple(DIR_VECTORS.values())

# Road-sign templates: a 3-cell bar plus one perpendicular protrusion.
SIGN_TEMPLATES: dict[str, tuple[np.ndarray, ...]] = {
    "up": (
        np.array([(1, 0), (1, 1), (1, 2), (0, 0)], dtype=int),
        np.array([(1, 0), (1, 1), (1, 2), (0, 2)], dtype=int),
    ),
    "down": (
        np.array([(0, 0), (0, 1), (0, 2), (1, 0)], dtype=int),
        np.array([(0, 0), (0, 1), (0, 2), (1, 2)], dtype=int),
    ),
    "left": (
        np.array([(0, 1), (1, 1), (2, 1), (0, 0)], dtype=int),
        np.array([(0, 1), (1, 1), (2, 1), (2, 0)], dtype=int),
    ),
    "right": (
        np.array([(0, 0), (1, 0), (2, 0), (0, 1)], dtype=int),
        np.array([(0, 0), (1, 0), (2, 0), (2, 1)], dtype=int),
    ),
}

START_SIDES: dict[str, tuple[str, str]] = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("above", "below"),
    "right": ("above", "below"),
}

AHEAD_SIDE: dict[str, str] = {
    "up": "above",
    "down": "below",
    "left": "left",
    "right": "right",
}

GEN_BG = 0
GEN_SIGN = 5
GEN_OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_SIZE_CHOICES = (2, 3)
GEN_RETRY_LIMIT = 400
GEN_SCENARIO_TRIES = 250
GEN_MIN_SIDE = 14
GEN_MAX_SIDE = 30
GEN_OBJECT_COUNT_MIN = 2
GEN_OBJECT_COUNT_MAX = 5
GEN_MIN_NEXT_HOP = 2
GEN_MAX_NEXT_HOP = 8
GEN_MIN_EDGE_HOP = 2


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            visited[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(sorted(cells), dtype=int)
            components.append({"color": color, "coords": coords})

    return components


def _decode_sign_direction(coords: np.ndarray) -> str | None:
    cells = {(int(r), int(c)) for r, c in coords.tolist()}
    if len(cells) != 4:
        return None

    rows = sorted({r for r, _ in cells})
    for row in rows:
        cols = sorted(c for r, c in cells if r == row)
        if len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
            extra = list(cells - {(row, col) for col in cols})
            if len(extra) == 1:
                er, ec = extra[0]
                if ec in cols and abs(er - row) == 1:
                    return "up" if er < row else "down"

    cols = sorted({c for _, c in cells})
    for col in cols:
        rows_here = sorted(r for r, c in cells if c == col)
        if len(rows_here) == 3 and rows_here == list(range(rows_here[0], rows_here[0] + 3)):
            extra = list(cells - {(row, col) for row in rows_here})
            if len(extra) == 1:
                er, ec = extra[0]
                if er in rows_here and abs(ec - col) == 1:
                    return "left" if ec < col else "right"

    return None


def _shift_cells(cells: Iterable[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in NEIGHBORS4:
            if (int(r + dr), int(c + dc)) in b:
                return True
    return False


def _object_cells_from_coords(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords.tolist()}


def _max_translation_steps(
    coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]
) -> int:
    h, w = map(int, shape)
    dr, dc = map(int, direction)
    rows = coords[:, 0]
    cols = coords[:, 1]
    if dr == -1:
        return int(rows.min())
    if dr == 1:
        return int(h - 1 - rows.max())
    if dc == -1:
        return int(cols.min())
    return int(w - 1 - cols.max())


def _find_landing_step(
    coords: np.ndarray,
    start_sign_idx: int,
    sign_sets: list[set[tuple[int, int]]],
    sign_union: set[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> int:
    dr, dc = map(int, direction)
    max_step = _max_translation_steps(coords, direction, shape)
    assert max_step >= 1

    for step in range(1, max_step + 1):
        shifted = {(int(r + dr * step), int(c + dc * step)) for r, c in coords.tolist()}
        assert not (shifted & sign_union), "objects must never overlap road signs"
        hits = [i for i, sign in enumerate(sign_sets) if i != start_sign_idx and _touches(shifted, sign)]
        if hits:
            assert len(hits) == 1, "the next road sign must be unambiguous"
            return step

    return max_step


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored square one hop from its adjacent road sign to the next sign or the boundary.

    Input structure:
    - The background is a dominant black field.
    - Small solid colored blocks are the moving objects.
    - Road signs are 4-cell grey shapes made from a 3-cell straight bar plus one perpendicular protrusion.
    - Each object is 4-neighbor adjacent to exactly one road sign.

    Rule:
    - Detect every road sign and read its pointing direction from the protrusion.
    - For each colored object, find the unique touching road sign and use that sign's direction as the move axis.
    - Translate the whole object in that direction until it first becomes adjacent to a different road sign.
    - If no later road sign is encountered on that ray, keep translating until the object reaches the grid edge.
    - Road signs stay fixed, the original object positions are cleared, and each object moves only once.

    Key insight:
    - A road sign encodes only the direction; the actual travel distance is determined by the position of the next sign, or by the canvas boundary when no later sign exists.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    sign_infos: list[dict[str, object]] = []
    object_infos: list[dict[str, object]] = []
    for comp in _color_components(g, bg):
        coords = comp["coords"]
        direction_name = _decode_sign_direction(coords)
        if direction_name is not None:
            sign_infos.append(
                {
                    "color": int(comp["color"]),
                    "coords": coords,
                    "cells": _object_cells_from_coords(coords),
                    "direction": DIR_VECTORS[direction_name],
                }
            )
        else:
            object_infos.append(
                {
                    "color": int(comp["color"]),
                    "coords": coords,
                    "cells": _object_cells_from_coords(coords),
                }
            )

    assert sign_infos, "expected at least one road sign"
    assert object_infos, "expected at least one moving object"

    sign_sets = [info["cells"] for info in sign_infos]
    sign_union = set().union(*sign_sets)

    out = g.copy()
    for obj in object_infos:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    placed_cells: set[tuple[int, int]] = set()
    for obj in object_infos:
        cells = obj["cells"]
        hits = [i for i, sign in enumerate(sign_sets) if _touches(cells, sign)]
        assert len(hits) == 1, "each object must touch exactly one current road sign"

        sign_idx = hits[0]
        step = _find_landing_step(
            obj["coords"],
            sign_idx,
            sign_sets,
            sign_union,
            sign_infos[sign_idx]["direction"],
            g.shape,
        )
        dr, dc = sign_infos[sign_idx]["direction"]
        shifted = {(int(r + dr * step), int(c + dc * step)) for r, c in obj["coords"].tolist()}
        assert not (shifted & sign_union)
        assert not (shifted & placed_cells), "moved objects must not collide"
        placed_cells |= shifted
        for r, c in shifted:
            out[r, c] = int(obj["color"])

    return out


def _template_bounds(template: np.ndarray) -> tuple[int, int]:
    return int(template[:, 0].max()) + 1, int(template[:, 1].max()) + 1


def _attachment_offsets(template: np.ndarray, side: str, size: int) -> list[tuple[int, int]]:
    obj = {(r, c) for r in range(size) for c in range(size)}
    h, w = _template_bounds(template)
    out: list[tuple[int, int]] = []

    if side == "above":
        sr = -h
        shifts = [(sr, sc) for sc in range(-w + 1, size)]
    elif side == "below":
        sr = size
        shifts = [(sr, sc) for sc in range(-w + 1, size)]
    elif side == "left":
        sc = -w
        shifts = [(sr, sc) for sr in range(-h + 1, size)]
    else:
        sc = size
        shifts = [(sr, sc) for sr in range(-h + 1, size)]

    for sr, sc in shifts:
        sign = {(int(r + sr), int(c + sc)) for r, c in template.tolist()}
        if sign & obj:
            continue
        if _touches(obj, sign):
            out.append((int(sr), int(sc)))

    return out


def _all_sign_options(direction: str, side: str, size: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
    options: list[tuple[np.ndarray, tuple[int, int]]] = []
    for template in SIGN_TEMPLATES[direction]:
        for offset in _attachment_offsets(template, side, size):
            options.append((template, offset))
    return options


def _square_cells(size: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(size) for c in range(size)}


def _scenario_relative_sets(
    size: int,
    direction_name: str,
    move_steps: int,
    start_template: np.ndarray,
    start_offset: tuple[int, int],
    next_template: np.ndarray | None,
    next_offset: tuple[int, int] | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]] | None]:
    dr, dc = DIR_VECTORS[direction_name]
    start_obj = _square_cells(size)
    final_obj = {(int(r + dr * move_steps), int(c + dc * move_steps)) for r, c in start_obj}
    start_sign = _shift_cells(start_template.tolist(), start_offset[0], start_offset[1])

    next_sign = None
    if next_template is not None and next_offset is not None:
        next_sign = _shift_cells(
            next_template.tolist(),
            next_offset[0] + dr * move_steps,
            next_offset[1] + dc * move_steps,
        )

    return start_obj, final_obj, start_sign, next_sign


def _scenario_valid(
    size: int,
    direction_name: str,
    move_steps: int,
    start_sign: set[tuple[int, int]],
    next_sign: set[tuple[int, int]] | None,
) -> bool:
    dr, dc = DIR_VECTORS[direction_name]
    start_obj = _square_cells(size)
    sign_sets = [start_sign] + ([] if next_sign is None else [next_sign])
    sign_union = set().union(*sign_sets)

    if _touches(start_sign, next_sign) if next_sign is not None else False:
        return False
    if not _touches(start_obj, start_sign):
        return False
    if next_sign is not None and _touches(start_obj, next_sign):
        return False

    for step in range(1, move_steps + 1):
        obj = {(int(r + dr * step), int(c + dc * step)) for r, c in start_obj}
        if obj & sign_union:
            return False
        hits = [idx for idx, sign in enumerate(sign_sets) if _touches(obj, sign)]
        if step < move_steps:
            if hits:
                return False
        elif next_sign is not None:
            if hits != [1]:
                return False
        elif hits:
            return False

    return True


def _expand_margin(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    reserve: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    reserve.add((rr, cc))
    return reserve


def _translate_absolute(
    rel_cells: set[tuple[int, int]], origin: tuple[int, int]
) -> set[tuple[int, int]]:
    r0, c0 = map(int, origin)
    return {(int(r0 + r), int(c0 + c)) for r, c in rel_cells}


def _choose_branch_mix(count: int, rng: np.random.Generator) -> list[bool]:
    if count == 1:
        return [bool(rng.integers(0, 2))]
    extra_next = bool(rng.integers(0, 2))
    true_count = count // 2 + int(extra_next)
    flags = [True] * true_count + [False] * (count - true_count)
    rng.shuffle(flags)
    if all(flags) or not any(flags):
        flags[0] = True
        flags[-1] = False
        rng.shuffle(flags)
    return flags


def _place_scenario(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    color: int,
    has_next: bool,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    h, w = map(int, grid.shape)

    for _ in range(GEN_SCENARIO_TRIES):
        size = int(rng.choice(GEN_SIZE_CHOICES))
        direction_name = str(rng.choice(tuple(DIR_VECTORS)))
        start_side = str(rng.choice(START_SIDES[direction_name]))
        start_options = _all_sign_options(direction_name, start_side, size)
        if not start_options:
            continue
        start_template, start_offset = start_options[int(rng.integers(0, len(start_options)))]

        next_template = None
        next_offset = None

        if has_next:
            move_steps = int(rng.integers(GEN_MIN_NEXT_HOP, GEN_MAX_NEXT_HOP + 1))
            next_direction = str(rng.choice(tuple(DIR_VECTORS)))
            next_side = AHEAD_SIDE[direction_name]
            next_options = _all_sign_options(next_direction, next_side, size)
            if not next_options:
                continue
            next_template, next_offset = next_options[int(rng.integers(0, len(next_options)))]

            rel_start_obj, rel_final_obj, rel_start_sign, rel_next_sign = _scenario_relative_sets(
                size,
                direction_name,
                move_steps,
                start_template,
                start_offset,
                next_template,
                next_offset,
            )
            assert rel_next_sign is not None

            rel_all = rel_start_obj | rel_final_obj | rel_start_sign | rel_next_sign
            min_r = min(r for r, _ in rel_all)
            max_r = max(r for r, _ in rel_all)
            min_c = min(c for _, c in rel_all)
            max_c = max(c for _, c in rel_all)
            r_lo = -min_r
            r_hi = h - 1 - max_r
            c_lo = -min_c
            c_hi = w - 1 - max_c
            if r_lo > r_hi or c_lo > c_hi:
                continue

            origin = (
                int(rng.integers(r_lo, r_hi + 1)),
                int(rng.integers(c_lo, c_hi + 1)),
            )
        else:
            if direction_name in ("left", "right"):
                max_move = w - size - 1
            else:
                max_move = h - size - 1
            if max_move < GEN_MIN_EDGE_HOP:
                continue

            move_steps = int(rng.integers(GEN_MIN_EDGE_HOP, max_move + 1))
            dr, dc = DIR_VECTORS[direction_name]
            if direction_name == "up":
                base_r = move_steps
                base_c = int(rng.integers(0, w - size + 1))
            elif direction_name == "down":
                base_r = h - size - move_steps
                base_c = int(rng.integers(0, w - size + 1))
            elif direction_name == "left":
                base_r = int(rng.integers(0, h - size + 1))
                base_c = move_steps
            else:
                base_r = int(rng.integers(0, h - size + 1))
                base_c = w - size - move_steps

            rel_start_obj, rel_final_obj, rel_start_sign, rel_next_sign = _scenario_relative_sets(
                size,
                direction_name,
                move_steps,
                start_template,
                start_offset,
                None,
                None,
            )
            rel_all = rel_start_obj | rel_final_obj | rel_start_sign
            origin = (base_r, base_c)

            abs_all = _translate_absolute(rel_all, origin)
            if not all(0 <= r < h and 0 <= c < w for r, c in abs_all):
                continue

        rel_start_obj, rel_final_obj, rel_start_sign, rel_next_sign = _scenario_relative_sets(
            size,
            direction_name,
            move_steps,
            start_template,
            start_offset,
            next_template,
            next_offset,
        )
        if not _scenario_valid(size, direction_name, move_steps, rel_start_sign, rel_next_sign):
            continue

        start_obj = _translate_absolute(rel_start_obj, origin)
        final_obj = _translate_absolute(rel_final_obj, origin)
        start_sign = _translate_absolute(rel_start_sign, origin)
        next_sign = None if rel_next_sign is None else _translate_absolute(rel_next_sign, origin)

        paint = set(start_obj) | set(start_sign)
        if next_sign is not None:
            paint |= set(next_sign)

        if any(not (0 <= r < h and 0 <= c < w) for r, c in paint):
            continue
        if any(int(grid[r, c]) != GEN_BG for r, c in paint):
            continue

        sweep: set[tuple[int, int]] = set()
        dr, dc = DIR_VECTORS[direction_name]
        start_obj_rel = _square_cells(size)
        for step in range(move_steps + 1):
            sweep |= _translate_absolute(
                {(int(r + dr * step), int(c + dc * step)) for r, c in start_obj_rel},
                origin,
            )
        reserve = _expand_margin(sweep | paint, grid.shape)
        if reserve & blocked:
            continue

        if next_sign is not None and _touches(start_sign, next_sign):
            continue

        for r, c in start_sign:
            grid[r, c] = GEN_SIGN
        if next_sign is not None:
            for r, c in next_sign:
                grid[r, c] = GEN_SIGN
        for r, c in start_obj:
            grid[r, c] = color

        blocked |= reserve
        return {
            "color": color,
            "start_obj": start_obj,
            "final_obj": final_obj,
            "direction": direction_name,
            "has_next": has_next,
        }

    return None


def _expected_output(grid: np.ndarray, scenarios: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for scenario in scenarios:
        for r, c in scenario["start_obj"]:
            out[r, c] = GEN_BG
    for scenario in scenarios:
        for r, c in scenario["final_obj"]:
            out[r, c] = int(scenario["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))
    w = int(rng.integers(GEN_MIN_SIDE, GEN_MAX_SIDE + 1))
    grid = np.full((h, w), GEN_BG, dtype=int)

    object_count = int(rng.integers(GEN_OBJECT_COUNT_MIN, GEN_OBJECT_COUNT_MAX + 1))
    colors = list(GEN_OBJECT_COLORS)
    rng.shuffle(colors)
    colors = colors[:object_count]
    branch_flags = _choose_branch_mix(object_count, rng)

    blocked: set[tuple[int, int]] = set()
    scenarios: list[dict[str, object]] = []
    for color, has_next in zip(colors, branch_flags):
        scenario = _place_scenario(grid, blocked, color, has_next, rng)
        if scenario is None:
            return None
        scenarios.append(scenario)

    expected = _expected_output(grid, scenarios)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 8")
