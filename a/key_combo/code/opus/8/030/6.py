from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, in_bounds, neighbors4


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

EDGE_TO_DIR = {
    "top": DIRS["D"],
    "bottom": DIRS["U"],
    "left": DIRS["R"],
    "right": DIRS["L"],
}

SHAPES = [
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_variants(shape: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    pts = np.array(sorted(shape), dtype=int)
    out: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = pts.copy()
        for _ in range(k):
            rot = np.stack([rot[:, 1], -rot[:, 0]], axis=1)
        for arr in (rot, np.stack([rot[:, 0], -rot[:, 1]], axis=1)):
            out.add(_normalize({(int(r), int(c)) for r, c in arr}))
    return sorted(out)


SHAPE_VARIANTS = [variant for shape in SHAPES for variant in _shape_variants(shape)]


def _touching_edge(comp: dict, shape: tuple[int, int]) -> str:
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    edges = []
    if r0 == 0:
        edges.append("top")
    if r1 == H - 1:
        edges.append("bottom")
    if c0 == 0:
        edges.append("left")
    if c1 == W - 1:
        edges.append("right")
    assert len(edges) == 1, f"expected exactly one touched edge, got {edges}"
    return edges[0]


def _start_pixel(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    H, W = map(int, shape)
    center_r = (H - 1) / 2.0
    center_c = (W - 1) / 2.0
    return min(
        ((int(r), int(c)) for r, c in coords),
        key=lambda rc: ((rc[0] - center_r) ** 2 + (rc[1] - center_c) ** 2, rc[0], rc[1]),
    )


def _turn_left(dr: int, dc: int) -> tuple[int, int]:
    return -int(dc), int(dr)


def _turn_right(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), -int(dr)


def _component_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _trace_particle(
    occupied: np.ndarray,
    *,
    start: tuple[int, int],
    edge: str,
    turn_mode: str = "alternate",
) -> set[tuple[int, int]]:
    H, W = occupied.shape
    r, c = map(int, start)
    dr, dc = EDGE_TO_DIR[edge]
    left_next = True
    path: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, int, int, bool]] = set()

    for _ in range(H * W * 4):
        state = (int(r), int(c), int(dr), int(dc), bool(left_next))
        if state in seen:
            break
        seen.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not in_bounds((H, W), nr, nc):
            if turn_mode == "always_left":
                tdr, tdc = _turn_left(dr, dc)
            elif turn_mode == "always_right":
                tdr, tdc = _turn_right(dr, dc)
            else:
                tdr, tdc = _turn_left(dr, dc) if left_next else _turn_right(dr, dc)
            rr = int(r + tdr)
            cc = int(c + tdc)
            if not in_bounds((H, W), rr, cc) or occupied[rr, cc]:
                break
            dr, dc = int(tdr), int(tdc)
            if turn_mode == "alternate":
                left_next = not left_next
            continue

        if occupied[nr, nc]:
            break

        r, c = int(nr), int(nc)
        occupied[r, c] = True
        path.add((int(r), int(c)))

    return path


def _analyze_and_solve(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    list[dict],
    list[dict],
    set[tuple[int, int]],
    dict[tuple[tuple[int, int], ...], set[tuple[int, int]]],
]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == 0, "expected black background"

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected foreground objects"

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    repeated_colors = [color for color, parts in by_color.items() if len(parts) >= 2]
    assert len(repeated_colors) == 1, f"expected one repeated color, got {repeated_colors}"
    repeated_color = int(repeated_colors[0])
    repeated = sorted(
        by_color[repeated_color],
        key=lambda comp: _start_pixel(comp["coords"], g.shape),
    )
    singleton = []
    for color, parts in by_color.items():
        if color == repeated_color:
            continue
        assert len(parts) == 1, f"non-repeated colors must appear once, got color {color}"
        singleton.append(parts[0])

    occupied = g != bg
    out = g.copy()
    all_path: set[tuple[int, int]] = set()
    per_comp_path: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {}
    for comp in repeated:
        edge = _touching_edge(comp, g.shape)
        start = _start_pixel(comp["coords"], g.shape)
        path = _trace_particle(occupied, start=start, edge=edge, turn_mode="alternate")
        key = _component_key(comp["coords"])
        assert not (all_path & path), "repeated-color trajectories must be disjoint"
        per_comp_path[key] = path
        all_path |= path
        for r, c in path:
            out[r, c] = repeated_color

    for comp in singleton:
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        touched = any(
            (nr, nc) in all_path
            for r, c in coords
            for nr, nc in neighbors4(r, c)
        )
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        out[rr, cc] = repeated_color if touched else bg

    return out, repeated_color, repeated, singleton, all_path, per_comp_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit repeated-color particles from boundary objects; recolor the singleton objects they can reach and erase the rest.

    Input structure:
    - A black background contains several small 4-connected monochrome objects.
    - Exactly one foreground color appears in 2 or 3 separate components; each of those repeated-color components touches exactly one grid edge.
    - All other foreground colors appear exactly once as interior 4-connected objects.

    Rule:
    - Identify the repeated color by counting 4-connected components per color.
    - For each repeated-color component, choose the pixel nearest the grid center as the particle start and launch it perpendicular to the touched edge toward the interior.
    - The particle paints repeated-color trajectory cells over background; whenever the next step would leave the grid it turns 90 degrees, alternating CCW then CW, and stops if the turned step is also blocked or out of bounds.
    - After all trajectories are drawn, each singleton object is recolored to the repeated color if any of its pixels is 4-neighbor adjacent to any trajectory pixel; otherwise that singleton object is erased to black.
    - The original repeated-color objects remain unchanged.

    Key insight:
    - The singleton objects are not classified by their own shape or color; their fate is determined by whether the repeated-color particle geometry can reach their neighborhood.
    """
    out, _repeated_color, _repeated, _singleton, _all_path, _per_comp_path = _analyze_and_solve(grid)
    return out


def _make_bar(edge: str, idx: int, length: int, H: int, W: int) -> set[tuple[int, int]]:
    if edge == "top":
        return {(r, idx) for r in range(length)}
    if edge == "bottom":
        return {(H - length + r, idx) for r in range(length)}
    if edge == "left":
        return {(idx, c) for c in range(length)}
    if edge == "right":
        return {(idx, W - length + c) for c in range(length)}
    raise ValueError(edge)


def _layout_vertical(H: int, W: int, count: int) -> list[dict]:
    if count == 2:
        bottom_col = random.randint(2, W - 6)
        top_col = random.randint(bottom_col + 2, W - 3)
        return [
            {"edge": "top", "idx": top_col, "length": random.randint(3, 5), "free": True},
            {"edge": "bottom", "idx": bottom_col, "length": random.randint(3, 5), "free": True},
        ]

    block_col = random.randint(1, W - 8)
    free_col = random.randint(block_col + 4, W - 3)
    return [
        {"edge": "top", "idx": block_col, "length": random.randint(3, 5), "free": False},
        {"edge": "bottom", "idx": block_col + 1, "length": random.randint(3, 5), "free": False},
        {"edge": "top", "idx": free_col, "length": random.randint(3, 5), "free": True},
    ]


def _layout_horizontal(H: int, W: int, count: int) -> list[dict]:
    if count == 2:
        left_row = random.randint(2, H - 6)
        right_row = random.randint(left_row + 2, H - 3)
        return [
            {"edge": "left", "idx": left_row, "length": random.randint(3, 5), "free": True},
            {"edge": "right", "idx": right_row, "length": random.randint(3, 5), "free": True},
        ]

    block_row = random.randint(4, H - 3)
    free_row = random.randint(1, block_row - 2)
    return [
        {"edge": "left", "idx": free_row, "length": random.randint(3, 5), "free": True},
        {"edge": "right", "idx": block_row - 1, "length": random.randint(3, 5), "free": False},
        {"edge": "left", "idx": block_row, "length": random.randint(3, 5), "free": False},
    ]


def _build_repeated_only() -> tuple[np.ndarray, list[dict], dict]:
    H = random.randint(15, 22)
    W = random.randint(15, 22)
    family = random.choice(["vf2", "vf3", "hf2", "hf3"])
    count = 2 if family.endswith("2") else 3
    repeated_color = random.choice(list(range(1, 10)))
    grid = np.zeros((H, W), dtype=int)

    specs = _layout_vertical(H, W, count) if family.startswith("v") else _layout_horizontal(H, W, count)
    placed = []
    for spec in specs:
        cells = _make_bar(spec["edge"], spec["idx"], spec["length"], H, W)
        for r, c in cells:
            assert grid[r, c] == 0
            grid[r, c] = repeated_color
        placed.append({**spec, "cells": cells})

    out, repeated_color_check, repeated, _singleton, all_path, per_comp_path = _analyze_and_solve(grid)
    assert repeated_color == repeated_color_check

    by_key = {_component_key(comp["coords"]): comp for comp in repeated}
    for spec in placed:
        key = tuple(sorted(spec["cells"]))
        spec["start"] = _start_pixel(by_key[key]["coords"], grid.shape)
        spec["path"] = per_comp_path[key]

    free_specs = [spec for spec in placed if spec["free"]]
    chosen_free = random.choice(free_specs)
    occupied = grid != 0
    wrong_path = _trace_particle(occupied.copy(), start=chosen_free["start"], edge=chosen_free["edge"], turn_mode="always_left")
    wrong_only = wrong_path - chosen_free["path"]
    assert wrong_only, "free path must differ from a non-alternating left-turn shortcut"

    return grid, placed, {"family": family, "repeated_color": repeated_color, "all_path": all_path, "chosen_free": chosen_free, "wrong_only": wrong_only, "solver_out": out}


def _adjacent_to(cells: set[tuple[int, int]], targets: set[tuple[int, int]]) -> bool:
    return any((nr, nc) in targets for r, c in cells for nr, nc in neighbors4(r, c))


def _valid_shape_placement(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    *,
    trajectory: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    if not cells:
        return False
    if min(r for r, _c in cells) < 1 or max(r for r, _c in cells) > H - 2:
        return False
    if min(c for _r, c in cells) < 1 or max(c for _r, c in cells) > W - 2:
        return False
    if any(grid[r, c] != 0 for r, c in cells):
        return False
    if cells & trajectory:
        return False
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells:
                continue
            if grid[nr, nc] != 0:
                return False
    return True


def _find_singleton_placement(
    grid: np.ndarray,
    *,
    trajectory: set[tuple[int, int]],
    require_adjacent_to: set[tuple[int, int]] | None,
    forbid_adjacent_to_trajectory: bool,
) -> set[tuple[int, int]] | None:
    H, W = grid.shape
    variants = SHAPE_VARIANTS[:]
    random.shuffle(variants)
    placements: list[set[tuple[int, int]]] = []

    for variant in variants:
        h = max(r for r, _c in variant) + 1
        w = max(c for _r, c in variant) + 1
        row_order = list(range(1, H - h))
        col_order = list(range(1, W - w))
        random.shuffle(row_order)
        random.shuffle(col_order)
        for r0 in row_order:
            for c0 in col_order:
                cells = {(r0 + r, c0 + c) for r, c in variant}
                if not _valid_shape_placement(grid, cells, trajectory=trajectory):
                    continue
                touches_traj = _adjacent_to(cells, trajectory)
                if forbid_adjacent_to_trajectory and touches_traj:
                    continue
                if require_adjacent_to is not None and not _adjacent_to(cells, require_adjacent_to):
                    continue
                if require_adjacent_to is None and not forbid_adjacent_to_trajectory and not touches_traj:
                    continue
                placements.append(cells)
        if placements:
            break

    return random.choice(placements) if placements else None


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _singleton_statuses(grid: np.ndarray) -> tuple[int, list[tuple[int, bool, int]]]:
    out, repeated_color, _repeated, singleton, all_path, _per_comp_path = _analyze_and_solve(grid)
    statuses = []
    for comp in singleton:
        color = int(comp["color"])
        touched = any(
            (nr, nc) in all_path
            for r, c in comp["coords"]
            for nr, nc in neighbors4(int(r), int(c))
        )
        values = {int(out[int(r), int(c)]) for r, c in comp["coords"]}
        assert len(values) == 1
        statuses.append((color, touched, values.pop()))
    return repeated_color, statuses


def _validate_generated(grid: np.ndarray) -> None:
    out, repeated_color, repeated, singleton, all_path, per_comp_path = _analyze_and_solve(grid)
    assert 2 <= len(repeated) <= 3
    assert 3 <= len(singleton) <= 4
    assert len(all_path) > 0
    assert any(len(per_comp_path[_component_key(comp["coords"])]) > 0 for comp in repeated)

    touched_seen = False
    erased_seen = False
    recolored_seen = False
    for comp in repeated:
        assert 3 <= int(comp["area"]) <= 5
        edge = _touching_edge(comp, grid.shape)
        assert edge in EDGE_TO_DIR

    for comp in singleton:
        assert 3 <= int(comp["area"]) <= 5
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert 0 < r0 and r1 < grid.shape[0] - 1
        assert 0 < c0 and c1 < grid.shape[1] - 1
        touched = any(
            (nr, nc) in all_path
            for r, c in comp["coords"]
            for nr, nc in neighbors4(int(r), int(c))
        )
        values = {int(out[int(r), int(c)]) for r, c in comp["coords"]}
        assert len(values) == 1
        value = values.pop()
        if touched:
            touched_seen = True
            recolored_seen = True
            assert value == repeated_color
        else:
            erased_seen = True
            assert value == 0

    assert touched_seen and erased_seen and recolored_seen
    assert np.any(grid != out), "generator must not produce identity examples"

    status_repeated_color, statuses = _singleton_statuses(grid)
    assert status_repeated_color == repeated_color
    assert any(touched and value == repeated_color for _color, touched, value in statuses)
    assert any((not touched) and value == 0 for _color, touched, value in statuses)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, repeated_specs, meta = _build_repeated_only()
            trajectory = set(meta["all_path"])
            repeated_color = int(meta["repeated_color"])
            wrong_only = set(meta["wrong_only"])

            colors = [color for color in range(1, 10) if color != repeated_color]
            random.shuffle(colors)
            singleton_count = random.randint(3, 4)
            statuses = ["touch", "erase"]
            while len(statuses) < singleton_count:
                statuses.append(random.choice(["touch", "erase"]))
            random.shuffle(statuses)

            anti_cells = _find_singleton_placement(
                grid,
                trajectory=trajectory,
                require_adjacent_to=wrong_only,
                forbid_adjacent_to_trajectory=True,
            )
            if anti_cells is None:
                continue
            anti_color = colors.pop()
            _paint_cells(grid, anti_cells, anti_color)

            touched_cells = _find_singleton_placement(
                grid,
                trajectory=trajectory,
                require_adjacent_to=trajectory,
                forbid_adjacent_to_trajectory=False,
            )
            if touched_cells is None:
                continue
            touched_color = colors.pop()
            _paint_cells(grid, touched_cells, touched_color)

            remaining = singleton_count - 2
            while remaining > 0:
                want_touch = statuses.pop()
                if want_touch == "touch":
                    cells = _find_singleton_placement(
                        grid,
                        trajectory=trajectory,
                        require_adjacent_to=trajectory,
                        forbid_adjacent_to_trajectory=False,
                    )
                else:
                    cells = _find_singleton_placement(
                        grid,
                        trajectory=trajectory,
                        require_adjacent_to=None,
                        forbid_adjacent_to_trajectory=True,
                    )
                if cells is None:
                    break
                _paint_cells(grid, cells, colors.pop())
                remaining -= 1
            if remaining != 0:
                continue

            _validate_generated(grid)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid puzzle instance")
