from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


BG_COLOR = Color.BLACK
L_COLOR = Color.TEAL
FILL_COLOR = Color.GREEN
RAY_COLOR = Color.RED
BLOCKER_COLOR = Color.MAROON

CORNER_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
RAY_SPECS = (
    ((0, 0), (-1, -1)),
    ((0, 1), (-1, 1)),
    ((1, 0), (1, -1)),
    ((1, 1), (1, 1)),
)
CORNER_FOR_DIRECTION = {direction: corner for corner, direction in RAY_SPECS}
MISSING_FOR_DIRECTION = {
    (-1, -1): (1, 1),
    (-1, 1): (1, 0),
    (1, -1): (0, 1),
    (1, 1): (0, 0),
}

GRID_RANGE = range(12, 17)
L_COUNT_RANGE = range(4, 8)
WITNESS_GAP_RANGE = range(2, 5)
GEN_ATTEMPTS = 800


def _square_cells(origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, origin)
    return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]


def _missing_cell(origin: tuple[int, int], missing: tuple[int, int]) -> tuple[int, int]:
    return int(origin[0] + missing[0]), int(origin[1] + missing[1])


def _l_cells(origin: tuple[int, int], missing: tuple[int, int]) -> list[tuple[int, int]]:
    miss = _missing_cell(origin, missing)
    return [cell for cell in _square_cells(origin) if cell != miss]


def _touches_orthogonally(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) <= 1


def _square_placement_valid(
    origin: tuple[int, int],
    placed: list[dict[str, object]],
    shape: tuple[int, int],
) -> bool:
    h, w = map(int, shape)
    r0, c0 = map(int, origin)
    if not (0 <= r0 <= h - 2 and 0 <= c0 <= w - 2):
        return False

    new_cells = _square_cells(origin)
    for square in placed:
        old_cells = _square_cells(square["origin"])
        for cell_a in new_cells:
            for cell_b in old_cells:
                if _touches_orthogonally(cell_a, cell_b):
                    return False
    return True


def _trace_ray(base: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    r = int(start[0] + dr)
    c = int(start[1] + dc)
    path: list[tuple[int, int]] = []
    while in_bounds(base.shape, r, c) and int(base[r, c]) == BG_COLOR:
        path.append((r, c))
        r += dr
        c += dc
    return path


def _ray_infos(base: np.ndarray, squares: list[dict[str, object]]) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for index, square in enumerate(squares):
        r0, c0 = square["origin"]
        for corner, direction in RAY_SPECS:
            start = (r0 + corner[0], c0 + corner[1])
            cells = _trace_ray(base, start, direction)
            infos.append(
                {
                    "square_index": int(index),
                    "start": start,
                    "direction": direction,
                    "cells": cells,
                }
            )
    return infos


def _extract_l_shapes(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == BG_COLOR, "expected black background"
    assert set(map(int, np.unique(g))).issubset({BG_COLOR, L_COLOR, BLOCKER_COLOR})

    blockers = find_components(g, bg=bg, colors=[BLOCKER_COLOR], connectivity=4)
    for blocker in blockers:
        assert int(blocker["area"]) == 1, "maroon blockers must be isolated single pixels"

    shapes = find_components(g, bg=bg, colors=[L_COLOR], connectivity=4)
    assert shapes, "expected at least one teal L-shape"

    parsed: list[dict[str, object]] = []
    for comp in shapes:
        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        assert len(coords) == 3
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r1 - r0, c1 - c0) == (1, 1), "each teal component must fit a 2x2 bbox"
        box_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        missing = list(box_cells - coords)
        assert len(missing) == 1, "each teal L-shape must have one missing corner"
        mr, mc = missing[0]
        parsed.append(
            {
                "origin": (r0, c0),
                "missing": (mr - r0, mc - c0),
            }
        )
    return parsed


def _completed_base_from_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    squares = _extract_l_shapes(g)
    base = g.copy()
    for square in squares:
        base[_missing_cell(square["origin"], square["missing"])] = FILL_COLOR
    return base, squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each teal missing-corner L into a 2x2 square, then emit diagonal corner rays that stop at completed blockers.

    Input structure:
    - The background is black.
    - Teal pixels form several disjoint 4-connected three-cell L-shapes, each occupying a 2x2 bounding box with one corner missing.
    - Optional maroon single pixels are extra blockers on the background.

    Rule:
    - Simultaneously fill the unique missing corner of every teal L-shape with green, producing full 2x2 squares.
    - From each completed square corner, extend a red ray outward along that corner's diagonal direction.
    - A ray paints only black cells and stops before the first out-of-bounds or non-black cell in the completed pre-ray grid; teal, green, and maroon cells all act as blockers.

    Key insight:
    - The green completion pixels are part of the blocker set, so the squares must be completed first before any ray paths can be computed correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    base, squares = _completed_base_from_grid(g)
    ray_cells: set[tuple[int, int]] = set()
    for info in _ray_infos(base, squares):
        ray_cells.update(info["cells"])

    out = base.copy()
    for r, c in ray_cells:
        out[r, c] = RAY_COLOR
    return out


def _input_from_layout(
    shape: tuple[int, int],
    squares: list[dict[str, object]],
    blockers: list[tuple[int, int]],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for square in squares:
        for r, c in _l_cells(square["origin"], square["missing"]):
            grid[r, c] = L_COLOR
    for r, c in blockers:
        grid[r, c] = BLOCKER_COLOR
    return grid


def _oracle_output_from_layout(
    shape: tuple[int, int],
    squares: list[dict[str, object]],
    blockers: list[tuple[int, int]],
) -> np.ndarray:
    base = _completed_base_from_layout(shape, squares, blockers)
    out = base.copy()
    for info in _ray_infos(base, squares):
        for r, c in info["cells"]:
            out[r, c] = RAY_COLOR
    return out


def _completed_base_from_layout(
    shape: tuple[int, int],
    squares: list[dict[str, object]],
    blockers: list[tuple[int, int]],
) -> np.ndarray:
    base = _input_from_layout(shape, squares, blockers)
    for square in squares:
        base[_missing_cell(square["origin"], square["missing"])] = FILL_COLOR
    return base


def _witness_pair(shape: tuple[int, int]) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    h, w = map(int, shape)
    directions = list(MISSING_FOR_DIRECTION)
    random.shuffle(directions)
    gaps = list(WITNESS_GAP_RANGE)
    random.shuffle(gaps)

    for direction in directions:
        corner = CORNER_FOR_DIRECTION[direction]
        target_missing = MISSING_FOR_DIRECTION[direction]
        for gap in gaps:
            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for ar in range(h - 1):
                for ac in range(w - 1):
                    missing_target = (ar + corner[0] + direction[0] * gap, ac + corner[1] + direction[1] * gap)
                    blocker_origin = (
                        int(missing_target[0] - target_missing[0]),
                        int(missing_target[1] - target_missing[1]),
                    )
                    first = {"origin": (ar, ac)}
                    if not _square_placement_valid(first["origin"], [], shape):
                        continue
                    if not _square_placement_valid(blocker_origin, [first], shape):
                        continue
                    candidates.append(((ar, ac), blocker_origin))
            if not candidates:
                continue

            source_origin, blocker_origin = random.choice(candidates)
            source_missing = random.choice(CORNER_OFFSETS)
            squares = [
                {"origin": source_origin, "missing": source_missing},
                {"origin": blocker_origin, "missing": target_missing},
            ]
            source_corner = (
                source_origin[0] + corner[0],
                source_origin[1] + corner[1],
            )
            blocker_cell = _missing_cell(blocker_origin, target_missing)
            clear_path = [
                (source_corner[0] + direction[0] * step, source_corner[1] + direction[1] * step)
                for step in range(1, gap)
            ]
            return squares, {
                "source_index": 0,
                "target_index": 1,
                "direction": direction,
                "source_corner": source_corner,
                "blocker_cell": blocker_cell,
                "clear_path": clear_path,
            }
    return None


def _place_extra_squares(
    shape: tuple[int, int],
    placed: list[dict[str, object]],
    count: int,
    reserved_empty: set[tuple[int, int]],
) -> bool:
    h, w = map(int, shape)
    candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(candidates)

    for _ in range(count):
        placed_this_round = False
        for origin in candidates:
            if not _square_placement_valid(origin, placed, shape):
                continue
            if any(cell in reserved_empty for cell in _square_cells(origin)):
                continue
            placed.append(
                {
                    "origin": origin,
                    "missing": random.choice(CORNER_OFFSETS),
                }
            )
            placed_this_round = True
            break
        if not placed_this_round:
            return False
    return True


def _ensure_orientation_variety(squares: list[dict[str, object]]) -> None:
    orientations = {tuple(square["missing"]) for square in squares}
    if len(orientations) >= 2:
        return
    alternatives = [corner for corner in CORNER_OFFSETS if corner != squares[0]["missing"]]
    squares[0]["missing"] = random.choice(alternatives)


def _blocked_by_l_shape(cell: tuple[int, int], squares: list[dict[str, object]]) -> bool:
    for square in squares:
        for teal in _l_cells(square["origin"], square["missing"]):
            if _touches_orthogonally(cell, teal):
                return True
    return False


def _pick_blockers(
    shape: tuple[int, int],
    squares: list[dict[str, object]],
    witness: dict[str, object],
) -> list[tuple[int, int]] | None:
    desired = 0 if random.random() < 0.25 else random.randint(1, 4)
    blockers: list[tuple[int, int]] = []
    reserved = set(witness["clear_path"])
    completed_cells = {cell for square in squares for cell in _square_cells(square["origin"])}

    if desired == 0:
        return blockers

    base_without_blockers = _completed_base_from_layout(shape, squares, [])
    ray_candidates: list[tuple[int, int]] = []
    for info in _ray_infos(base_without_blockers, squares):
        for cell in info["cells"]:
            if cell in reserved or cell in completed_cells:
                continue
            if _blocked_by_l_shape(cell, squares):
                continue
            ray_candidates.append(cell)
    random.shuffle(ray_candidates)

    if not ray_candidates:
        return None
    blockers.append(ray_candidates[0])

    h, w = map(int, shape)
    free_cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(free_cells)
    for cell in free_cells:
        if len(blockers) >= desired:
            break
        if cell in completed_cells or cell in reserved or cell in blockers:
            continue
        if _blocked_by_l_shape(cell, squares):
            continue
        if any(_touches_orthogonally(cell, blocker) for blocker in blockers):
            continue
        blockers.append(cell)

    return blockers if len(blockers) == desired else None


def _validate_layout(
    shape: tuple[int, int],
    squares: list[dict[str, object]],
    blockers: list[tuple[int, int]],
    witness: dict[str, object],
) -> bool:
    if not (4 <= len(squares) <= 7):
        return False

    inp = _input_from_layout(shape, squares, blockers)
    expected = _oracle_output_from_layout(shape, squares, blockers)

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.count_nonzero(expected == RAY_COLOR) < 6:
        return False

    base = expected.copy()
    base[base == RAY_COLOR] = BG_COLOR

    source_square = squares[witness["source_index"]]
    source_corner = witness["source_corner"]
    direction = witness["direction"]
    blocker_cell = witness["blocker_cell"]

    witness_path = _trace_ray(base, source_corner, direction)
    if witness_path != witness["clear_path"]:
        return False

    probe = base.copy()
    probe[blocker_cell] = BG_COLOR
    extended = _trace_ray(probe, source_corner, direction)
    if len(extended) != len(witness_path) + 1:
        return False
    if extended[: len(witness_path)] != witness_path:
        return False
    if extended[-1] != blocker_cell:
        return False

    parsed = _extract_l_shapes(inp)
    if len(parsed) != len(squares):
        return False

    for cell in blockers:
        if _blocked_by_l_shape(cell, squares):
            return False
    if any(cell in witness["clear_path"] for cell in blockers):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(GRID_RANGE), random.choice(GRID_RANGE))
    target_count = random.choice(L_COUNT_RANGE)

    witness_pair = _witness_pair(shape)
    if witness_pair is None:
        return None
    squares, witness = witness_pair

    if not _place_extra_squares(shape, squares, target_count - len(squares), set(witness["clear_path"])):
        return None
    _ensure_orientation_variety(squares)

    blockers = _pick_blockers(shape, squares, witness)
    if blockers is None:
        return None

    if not _validate_layout(shape, squares, blockers, witness):
        return None
    return _input_from_layout(shape, squares, blockers)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
