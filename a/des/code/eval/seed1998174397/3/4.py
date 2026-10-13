from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
TOP_COLOR = 2
BOTTOM_COLOR = 1
LEFT_COLOR = 3
RIGHT_COLOR = 4
OBJECT_COLORS = (5, 6, 7, 8, 9)

DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_COLORS = {
    "up": TOP_COLOR,
    "down": BOTTOM_COLOR,
    "left": LEFT_COLOR,
    "right": RIGHT_COLOR,
}
MODES = ("free", "blocked", "truncated")

GRID_MIN = 11
GRID_MAX = 18
OBJECT_COUNT = {
    "free": (2, 4),
    "blocked": (2, 3),
    "truncated": (2, 3),
}
AREA_RANGE = (3, 8)
MAX_SHAPE_BBOX = 4
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 200
MIN_NEW_PIXELS = 6


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _inside_inner(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _direction_from_pointer(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    pointer: tuple[int, int],
) -> str:
    arr = np.array(coords, dtype=float)
    mean_r, mean_c = np.mean(arr, axis=0)
    center_r = _round_half_up(float(mean_r))
    center_c = _round_half_up(float(mean_c))
    pr, pc = pointer

    dr = int(pr - center_r)
    dc = int(pc - center_c)
    raw_dr = float(pr - mean_r)
    raw_dc = float(pc - mean_c)

    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"

    if abs(raw_dr) >= abs(raw_dc):
        return "down" if raw_dr > 0 else "up"
    return "right" if raw_dc > 0 else "left"


def _trace_component(
    cells: set[tuple[int, int]],
    direction: str,
    length: int,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool, bool]:
    dr, dc = DIR_STEPS[direction]
    other = occupied - cells
    painted: set[tuple[int, int]] = set()
    blocked = False
    truncated = False

    for start_r, start_c in sorted(cells):
        r, c = start_r, start_c
        while True:
            nr, nc = r + dr, c + dc
            if not _inside_inner(shape, nr, nc):
                truncated = True
                break
            if (nr, nc) in cells:
                r, c = nr, nc
                continue
            rr, cc = r, c
            for _ in range(length):
                nr, nc = rr + dr, cc + dc
                if not _inside_inner(shape, nr, nc):
                    truncated = True
                    break
                if (nr, nc) in other:
                    blocked = True
                    break
                painted.add((nr, nc))
                rr, cc = nr, nc
            break

    return painted, blocked, truncated


def _global_cells(obj: dict[str, object]) -> set[tuple[int, int]]:
    r0, c0 = obj["anchor"]
    return {(r0 + r, c0 + c) for r, c in obj["cells"]}


def _global_pointer(obj: dict[str, object]) -> tuple[int, int]:
    r0, c0 = obj["anchor"]
    pr, pc = obj["pointer"]
    return r0 + pr, c0 + pc


def _frontier_cells(
    cells: tuple[tuple[int, int], ...] | set[tuple[int, int]],
    direction: str,
) -> list[tuple[int, int]]:
    cell_set = set(cells)
    dr, dc = DIR_STEPS[direction]
    return sorted((r, c) for r, c in cell_set if (r + dr, c + dc) not in cell_set)


def _make_empty_grid(h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = TOP_COLOR
    grid[-1, :] = BOTTOM_COLOR
    grid[:, 0] = LEFT_COLOR
    grid[:, -1] = RIGHT_COLOR
    return grid


def _random_shape(area: int) -> tuple[tuple[int, int], ...] | None:
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        weighted: list[tuple[int, int]] = []
        for cand in sorted(frontier):
            touches = sum((cand[0] + dr, cand[1] + dc) in cells for dr, dc in DIR_STEPS.values())
            weighted.extend([cand] * (2 if touches >= 2 else 1))
        cells.add(random.choice(weighted))

    shape = _normalize(cells)
    height = 1 + max(r for r, _ in shape)
    width = 1 + max(c for _, c in shape)
    if height > MAX_SHAPE_BBOX or width > MAX_SHAPE_BBOX:
        return None
    return shape


def _pointer_score(
    coords: tuple[tuple[int, int], ...],
    pointer: tuple[int, int],
    direction: str,
) -> tuple[float, float]:
    arr = np.array(coords, dtype=float)
    mean_r, mean_c = np.mean(arr, axis=0)
    pr, pc = pointer
    if direction == "up":
        return (mean_r - pr, -abs(mean_c - pc))
    if direction == "down":
        return (pr - mean_r, -abs(mean_c - pc))
    if direction == "left":
        return (mean_c - pc, -abs(mean_r - pr))
    return (pc - mean_c, -abs(mean_r - pr))


def _sample_object(direction: str) -> dict[str, object] | None:
    for _ in range(SHAPE_ATTEMPTS):
        area = random.randint(*AREA_RANGE)
        shape = _random_shape(area)
        if shape is None:
            continue

        candidates = [
            cell for cell in shape
            if _direction_from_pointer(shape, cell) == direction
        ]
        if not candidates:
            continue

        pointer = max(candidates, key=lambda cell: _pointer_score(shape, cell, direction))
        main_color = random.choice(OBJECT_COLORS)
        pointer_color = random.choice([c for c in OBJECT_COLORS if c != main_color])
        height = 1 + max(r for r, _ in shape)
        width = 1 + max(c for _, c in shape)
        return {
            "cells": shape,
            "pointer": pointer,
            "dir": direction,
            "area": area,
            "height": height,
            "width": width,
            "main": main_color,
            "pointer_color": pointer_color,
        }
    return None


def _collect_occupied(objects: list[dict[str, object]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    occupied: set[tuple[int, int]] = set()
    for obj in objects:
        occupied |= _global_cells(obj)
    moat: set[tuple[int, int]] = set()
    for r, c in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                moat.add((r + dr, c + dc))
    return occupied, moat


def _can_place(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    moat: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if any(not _inside_inner(shape, r, c) for r, c in cells):
        return False
    if cells & occupied:
        return False
    if cells & moat:
        return False
    return True


def _place_random(
    obj: dict[str, object],
    shape: tuple[int, int],
    placed: list[dict[str, object]],
    predicate=None,
) -> tuple[int, int] | None:
    h, w = shape
    occupied, moat = _collect_occupied(placed)
    anchors = [
        (r0, c0)
        for r0 in range(1, h - obj["height"])
        for c0 in range(1, w - obj["width"])
    ]
    random.shuffle(anchors)
    for anchor in anchors:
        cells = {(anchor[0] + r, anchor[1] + c) for r, c in obj["cells"]}
        if not _can_place(cells, occupied, moat, shape):
            continue
        if predicate is not None and not predicate(cells):
            continue
        return anchor
    return None


def _render_grid(objects: list[dict[str, object]], h: int, w: int) -> np.ndarray:
    grid = _make_empty_grid(h, w)
    for obj in objects:
        for r, c in _global_cells(obj):
            grid[r, c] = int(obj["main"])
        pr, pc = _global_pointer(obj)
        grid[pr, pc] = int(obj["pointer_color"])
    return grid


def _projection_analysis(
    objects: list[dict[str, object]],
    h: int,
    w: int,
) -> dict[str, object] | None:
    occupied, _moat = _collect_occupied(objects)
    paint_sets: list[set[tuple[int, int]]] = []
    blocked_flags: list[bool] = []
    truncated_flags: list[bool] = []
    paint_map: dict[tuple[int, int], int] = {}

    for obj in objects:
        cells = _global_cells(obj)
        painted, blocked, truncated = _trace_component(
            cells,
            str(obj["dir"]),
            int(obj["area"]),
            occupied,
            (h, w),
        )
        for cell in painted:
            if cell in paint_map:
                return None
            paint_map[cell] = DIR_COLORS[str(obj["dir"])]
        paint_sets.append(painted)
        blocked_flags.append(blocked)
        truncated_flags.append(truncated)

    return {
        "paint_sets": paint_sets,
        "blocked": blocked_flags,
        "truncated": truncated_flags,
        "paint_map": paint_map,
        "total_new": len(paint_map),
    }


def _expected_output(grid: np.ndarray, objects: list[dict[str, object]], analysis: dict[str, object]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for obj in objects:
        pr, pc = _global_pointer(obj)
        out[pr, pc] = int(obj["main"])
    for cell, color in analysis["paint_map"].items():
        out[cell] = color
    return out


def _place_blocker(
    subject: dict[str, object],
    blocker: dict[str, object],
    shape: tuple[int, int],
    placed: list[dict[str, object]],
) -> tuple[int, int] | None:
    h, w = shape
    occupied, moat = _collect_occupied(placed)
    subject_cells = _global_cells(subject)
    dr, dc = DIR_STEPS[str(subject["dir"])]
    frontiers = _frontier_cells(subject["cells"], str(subject["dir"]))
    random.shuffle(frontiers)

    target_steps = list(range(2, int(subject["area"]) + 1))
    random.shuffle(target_steps)
    blocker_locals = list(blocker["cells"])
    random.shuffle(blocker_locals)

    for fr, fc in frontiers:
        base_r, base_c = subject["anchor"][0] + fr, subject["anchor"][1] + fc
        for step in target_steps:
            target = (base_r + dr * step, base_c + dc * step)
            if not _inside_inner((h, w), *target):
                continue
            for br, bc in blocker_locals:
                anchor = (target[0] - br, target[1] - bc)
                cells = {(anchor[0] + r, anchor[1] + c) for r, c in blocker["cells"]}
                if not _can_place(cells, occupied, moat, (h, w)):
                    continue
                occupied2 = occupied | cells
                _painted, blocked, truncated = _trace_component(
                    subject_cells,
                    str(subject["dir"]),
                    int(subject["area"]),
                    occupied2,
                    (h, w),
                )
                if blocked and not truncated:
                    return anchor
    return None


def _validate_mode(
    mode: str,
    analysis: dict[str, object],
    subject_idx: int,
) -> bool:
    blocked = analysis["blocked"]
    truncated = analysis["truncated"]
    if mode == "free":
        return not any(blocked) and not any(truncated)
    if mode == "blocked":
        return bool(blocked[subject_idx]) and not bool(truncated[subject_idx])
    return bool(truncated[subject_idx]) and not bool(blocked[subject_idx])


def _gen_direction_list(count: int) -> list[str]:
    dirs = random.sample(list(DIR_STEPS), k=min(count, 4))
    while len(dirs) < count:
        dirs.append(random.choice(list(DIR_STEPS)))
    random.shuffle(dirs)
    return dirs


def _attempt_generate() -> np.ndarray | None:
    mode = random.choice(MODES)
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.randint(*OBJECT_COUNT[mode])
    directions = _gen_direction_list(count)

    placed: list[dict[str, object]] = []

    subject = _sample_object(directions[0])
    if subject is None:
        return None

    if mode == "truncated":
        def subject_predicate(cells: set[tuple[int, int]]) -> bool:
            _painted, blocked, truncated = _trace_component(
                cells,
                str(subject["dir"]),
                int(subject["area"]),
                set(cells),
                (h, w),
            )
            return truncated and not blocked
    else:
        subject_predicate = None

    anchor = _place_random(subject, (h, w), placed, predicate=subject_predicate)
    if anchor is None:
        return None
    subject = dict(subject)
    subject["anchor"] = anchor
    placed.append(subject)
    subject_idx = 0

    start_idx = 1
    if mode == "blocked":
        blocker = _sample_object(directions[1])
        if blocker is None:
            return None
        blocker_anchor = _place_blocker(subject, blocker, (h, w), placed)
        if blocker_anchor is None:
            return None
        blocker = dict(blocker)
        blocker["anchor"] = blocker_anchor
        placed.append(blocker)
        start_idx = 2

    for idx in range(start_idx, count):
        obj = _sample_object(directions[idx])
        if obj is None:
            return None
        anchor = _place_random(obj, (h, w), placed)
        if anchor is None:
            return None
        obj = dict(obj)
        obj["anchor"] = anchor
        placed.append(obj)

    analysis = _projection_analysis(placed, h, w)
    if analysis is None:
        return None
    if int(analysis["total_new"]) < MIN_NEW_PIXELS:
        return None
    if not _validate_mode(mode, analysis, subject_idx):
        return None

    grid = _render_grid(placed, h, w)
    expected = _expected_output(grid, placed, analysis)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3

    top_color = int(g[0, 1]) if w > 2 else int(g[0, 0])
    bottom_color = int(g[-1, 1]) if w > 2 else int(g[-1, -1])
    left_color = int(g[1, 0]) if h > 2 else int(g[0, 0])
    right_color = int(g[1, -1]) if h > 2 else int(g[-1, -1])
    if w > 2:
        assert np.all(g[0, 1:-1] == top_color)
        assert np.all(g[-1, 1:-1] == bottom_color)
    if h > 2:
        assert np.all(g[1:-1, 0] == left_color)
        assert np.all(g[1:-1, -1] == right_color)
    border_by_dir = {
        "up": top_color,
        "down": bottom_color,
        "left": left_color,
        "right": right_color,
    }

    mask = np.zeros_like(g, dtype=bool)
    mask[1:-1, 1:-1] = g[1:-1, 1:-1] != BG
    comps = _components(mask)
    occupied: set[tuple[int, int]] = set()
    analyzed: list[tuple[set[tuple[int, int]], int, tuple[int, int], str, int]] = []

    for comp in comps:
        colors = Counter(int(g[r, c]) for r, c in comp)
        main_color, _count = max(colors.items(), key=lambda item: item[1])
        pointer_cells = [(r, c) for r, c in comp if int(g[r, c]) != int(main_color)]
        assert len(pointer_cells) == 1, "each object must contain exactly one pointer pixel"
        direction = _direction_from_pointer(comp, pointer_cells[0])
        cell_set = set(comp)
        occupied |= cell_set
        analyzed.append((cell_set, int(main_color), pointer_cells[0], direction, len(comp)))

    out = np.array(g, copy=True)
    for _cells, main_color, pointer, _direction, _area in analyzed:
        out[pointer] = int(main_color)

    for cells, _main_color, _pointer, direction, area in analyzed:
        painted, _blocked, _truncated = _trace_component(cells, direction, area, occupied, (h, w))
        for r, c in painted:
            out[r, c] = border_by_dir[direction]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection puzzle input")
