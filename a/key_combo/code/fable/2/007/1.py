from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("shared_arc_utils", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
COLORS = tuple(range(10))
PAIR_AREA_RANGE = (3, 6)
LONE_AREA_RANGE = (3, 6)
GRID_H_RANGE = (11, 19)
GRID_W_RANGE = (11, 19)
CENTER_PARITIES = ((0, 0), (0, 1), (1, 0), (1, 1))
GENERATION_RETRIES = 500


def _coord_set(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _bbox_of_cells(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox_of_cells(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _expand8(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS8}


def _place(shape: frozenset[tuple[int, int]], top: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = top
    return {(tr + r, tc + c) for r, c in shape}


def _rotate180(cells: set[tuple[int, int]] | frozenset[tuple[int, int]], center_sum: tuple[int, int]) -> set[tuple[int, int]]:
    sr, sc = center_sum
    return {(sr - r, sc - c) for r, c in cells}


def _all_in_bounds(cells: set[tuple[int, int]] | frozenset[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _paint(grid: np.ndarray, cells: set[tuple[int, int]] | frozenset[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _same_pair_center(comp_a: dict, comp_b: dict) -> tuple[int, int] | None:
    if int(comp_a["area"]) != int(comp_b["area"]) or int(comp_a["area"]) <= 1:
        return None

    ar0, ac0, ar1, ac1 = map(int, comp_a["bbox"])
    br0, bc0, br1, bc1 = map(int, comp_b["bbox"])
    sr_candidates = {ar0 + br1, ar1 + br0}
    sc_candidates = {ac0 + bc1, ac1 + bc0}
    if len(sr_candidates) != 1 or len(sc_candidates) != 1:
        return None

    center_sum = (sr_candidates.pop(), sc_candidates.pop())
    a_cells = _coord_set(comp_a["coords"])
    b_cells = _coord_set(comp_b["coords"])
    if _rotate180(a_cells, center_sum) != set(b_cells):
        return None
    return center_sum


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected foreground objects"

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)
    assert len(by_color) == 2, "expected exactly two foreground colors"

    pair_candidates = []
    lone_candidates = []
    for color, comps in by_color.items():
        multi = [comp for comp in comps if int(comp["area"]) > 1]
        singles = [comp for comp in comps if int(comp["area"]) == 1]
        if len(multi) == 2 and len(singles) == 1:
            center_sum = _same_pair_center(multi[0], multi[1])
            if center_sum is not None:
                pair_candidates.append(
                    {
                        "pair_color": int(color),
                        "pair_components": multi,
                        "singleton": singles[0],
                        "center_sum": center_sum,
                    }
                )
        elif len(comps) == 1 and len(multi) == 1:
            lone_candidates.append({"lone_color": int(color), "lone_component": multi[0]})

    assert len(pair_candidates) == 1, "expected one valid same-color reflected pair"
    assert len(lone_candidates) == 1, "expected one lone multicolor object"

    pair_info = pair_candidates[0]
    lone_info = lone_candidates[0]
    assert pair_info["pair_color"] != lone_info["lone_color"]

    pair_cells = set()
    for comp in pair_info["pair_components"]:
        pair_cells.update(_coord_set(comp["coords"]))
    lone_cells = _coord_set(lone_info["lone_component"]["coords"])
    singleton_cell = next(iter(_coord_set(pair_info["singleton"]["coords"])))

    return {
        "bg": bg,
        "pair_color": pair_info["pair_color"],
        "pair_cells": frozenset(pair_cells),
        "singleton_cell": singleton_cell,
        "center_sum": pair_info["center_sum"],
        "lone_color": lone_info["lone_color"],
        "lone_cells": lone_cells,
    }


def _expected_output(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    rotated = _rotate180(scene["lone_cells"], scene["center_sum"])
    if _all_in_bounds(rotated, out.shape):
        _paint(out, rotated, scene["lone_color"])
    else:
        _paint(out, scene["lone_cells"], scene["bg"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unseen 180-degree rotation center from a same-color shape pair, then add or erase the lone shape by that symmetry.

    Input structure:
    - A dominant background fills most of the grid.
    - Exactly one foreground color forms two disconnected multi-cell components plus one isolated single-pixel distractor.
    - Exactly one other foreground color forms one disconnected multi-cell lone shape.

    Rule:
    - Identify the two multi-cell same-color components that are 180-degree rotations of each other and infer their common midpoint as the hidden rotation center.
    - Rotate the lone shape 180 degrees around that center.
    - If every rotated cell stays inside the grid, draw the rotated copy in the lone shape's color; otherwise erase the original lone shape to background and draw nothing new.
    - Keep the reflected pair and the single-pixel distractor unchanged.

    Key insight:
    - The center is never marked directly: it must be reconstructed from the paired shape geometry, while the same-color singleton is only a distractor and the out-of-bounds case flips completion into deletion.
    """
    scene = _analyze_scene(np.asarray(grid, dtype=int))
    return _expected_output(grid, scene)


def _random_shape(area_range: tuple[int, int], max_height: int = 4, max_width: int = 4) -> frozenset[tuple[int, int]]:
    for _ in range(200):
        target_area = random.randint(*area_range)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nb = (r + dr, c + dc)
                    if nb in cells:
                        continue
                    candidate = cells | {nb}
                    nr0, nc0, nr1, nc1 = _bbox_of_cells(candidate)
                    if nr1 - nr0 + 1 <= max_height and nc1 - nc0 + 1 <= max_width:
                        frontier.append(nb)
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue
        shape = _normalize(cells)
        r0, c0, r1, c1 = _bbox_of_cells(shape)
        if r1 == 0 or c1 == 0:
            continue
        return shape
    raise RuntimeError("failed to sample a connected shape")


def _sample_center_sum(limit: int, parity: int) -> int:
    values = [v for v in range(2, 2 * limit - 2) if v % 2 == parity]
    return random.choice(values)


def _pair_placements(
    shape: frozenset[tuple[int, int]],
    center_sum: tuple[int, int],
    grid_shape: tuple[int, int],
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = grid_shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    placements = []
    for top_r in range(h - max_r):
        for top_c in range(w - max_c):
            left = _place(shape, (top_r, top_c))
            right = _rotate180(left, center_sum)
            if not _all_in_bounds(right, grid_shape):
                continue
            if left & right:
                continue
            if not _expand8(left).isdisjoint(right):
                continue
            placements.append((left, right))
    random.shuffle(placements)
    return placements


def _lone_placements(
    shape: frozenset[tuple[int, int]],
    center_sum: tuple[int, int],
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    want_in_bounds: bool,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    h, w = grid_shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    blocked = _expand8(occupied)
    placements = []
    for top_r in range(h - max_r):
        for top_c in range(w - max_c):
            source = _place(shape, (top_r, top_c))
            if not source.isdisjoint(blocked):
                continue

            target = _rotate180(source, center_sum)
            in_bounds = _all_in_bounds(target, grid_shape)
            if in_bounds != want_in_bounds:
                continue

            if want_in_bounds:
                if source & target:
                    continue
                if not source.isdisjoint(_expand8(target)):
                    continue
                if not target.isdisjoint(_expand8(occupied)):
                    continue
            placements.append((source, target))
    random.shuffle(placements)
    return placements


def _singleton_choices(
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    blocked = _expand8(occupied | reserved)
    choices = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    random.shuffle(choices)
    return choices


def _manual_from_constructed(
    grid_shape: tuple[int, int],
    bg: int,
    pair_color: int,
    lone_color: int,
    pair_a: set[tuple[int, int]],
    pair_b: set[tuple[int, int]],
    singleton: tuple[int, int],
    lone: set[tuple[int, int]],
    lone_target: set[tuple[int, int]],
    want_in_bounds: bool,
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    _paint(out, pair_a, pair_color)
    _paint(out, pair_b, pair_color)
    out[singleton] = pair_color
    _paint(out, lone, lone_color)
    if want_in_bounds:
        _paint(out, lone_target, lone_color)
    else:
        _paint(out, lone, bg)
    return out


def _validate_generated_case(
    grid: np.ndarray,
    *,
    expected_center_sum: tuple[int, int],
    expected_pair_color: int,
    expected_lone_color: int,
    want_in_bounds: bool,
    manual_output: np.ndarray,
) -> bool:
    scene = _analyze_scene(grid)
    if scene["center_sum"] != expected_center_sum:
        return False
    if scene["pair_color"] != expected_pair_color or scene["lone_color"] != expected_lone_color:
        return False

    rotated = _rotate180(scene["lone_cells"], scene["center_sum"])
    if _all_in_bounds(rotated, grid.shape) != want_in_bounds:
        return False

    if want_in_bounds:
        occupied = set(scene["pair_cells"])
        occupied.add(scene["singleton_cell"])
        if not rotated.isdisjoint(occupied):
            return False
        if rotated == set(scene["lone_cells"]):
            return False

    return np.array_equal(main(grid), manual_output)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(COLORS)
    pair_color, lone_color = random.sample([c for c in COLORS if c != bg], 2)
    want_in_bounds = random.random() < 0.5

    pair_shape = _random_shape(PAIR_AREA_RANGE)
    lone_shape = _random_shape(LONE_AREA_RANGE)
    sr_parity, sc_parity = random.choice(CENTER_PARITIES)

    for _ in range(120):
        center_sum = (_sample_center_sum(h, sr_parity), _sample_center_sum(w, sc_parity))
        pair_options = _pair_placements(pair_shape, center_sum, (h, w))
        if not pair_options:
            continue
        pair_a, pair_b = random.choice(pair_options)

        occupied = set(pair_a) | set(pair_b)
        lone_options = _lone_placements(lone_shape, center_sum, (h, w), occupied, want_in_bounds)
        if not lone_options:
            continue
        lone_cells, lone_target = random.choice(lone_options)

        reserved = set(lone_target) if want_in_bounds else set()
        singleton_options = _singleton_choices((h, w), occupied | set(lone_cells), reserved)
        if not singleton_options:
            continue
        singleton = singleton_options[0]

        grid = np.full((h, w), bg, dtype=int)
        _paint(grid, pair_a, pair_color)
        _paint(grid, pair_b, pair_color)
        grid[singleton] = pair_color
        _paint(grid, lone_cells, lone_color)

        manual_output = _manual_from_constructed(
            (h, w),
            bg,
            pair_color,
            lone_color,
            pair_a,
            pair_b,
            singleton,
            lone_cells,
            lone_target,
            want_in_bounds,
        )

        if _validate_generated_case(
            grid,
            expected_center_sum=center_sum,
            expected_pair_color=pair_color,
            expected_lone_color=lone_color,
            want_in_bounds=want_in_bounds,
            manual_output=manual_output,
        ):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-center puzzle")
