import random
from collections import Counter

import numpy as np


BG = 0
TARGET = 4
ENDPOINT = 5
HINGE = 9
ROD_PALETTE = (1, 2, 3, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sort_line(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        return sorted(coords, key=lambda p: p[1])
    if len(cols) == 1:
        return sorted(coords, key=lambda p: p[0])
    raise AssertionError("rod must be straight")


def _perp_options(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _trace_targets(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = components_mask(grid == TARGET)
    targets: list[list[tuple[int, int]]] = []
    target_by_cell: dict[tuple[int, int], int] = {}
    for tid, comp in enumerate(comps):
        cells = [tuple(map(int, p)) for p in comp]
        targets.append(cells)
        for cell in cells:
            target_by_cell[cell] = tid
    return targets, target_by_cell


def _parse_rods(grid: np.ndarray) -> list[dict]:
    fg = grid != BG
    rods = []
    for comp in components_mask(fg):
        cells = [tuple(map(int, p)) for p in comp]
        colors = {int(grid[r, c]) for r, c in cells}
        if colors == {TARGET}:
            continue
        assert ENDPOINT in colors and HINGE in colors and len(colors) == 3, "expected multicolor rod"
        assert sum(int(grid[r, c]) == ENDPOINT for r, c in cells) == 1
        assert sum(int(grid[r, c]) == HINGE for r, c in cells) == 1
        body_colors = colors - {ENDPOINT, HINGE}
        assert len(body_colors) == 1
        body = next(iter(body_colors))
        assert body not in {BG, TARGET, ENDPOINT, HINGE}
        assert 5 <= len(cells) <= 8

        ordered = _sort_line(cells)
        line_set = set(ordered)
        (r0, c0), (r1, c1) = ordered[0], ordered[-1]
        if r0 == r1:
            expected = [(r0, cc) for cc in range(c0, c1 + 1)]
        else:
            expected = [(rr, c0) for rr in range(r0, r1 + 1)]
        assert set(expected) == line_set and len(expected) == len(cells), "rod cells must be contiguous"

        endpoint = next((p for p in expected if int(grid[p]) == ENDPOINT), None)
        hinge = next((p for p in expected if int(grid[p]) == HINGE), None)
        assert endpoint in {expected[0], expected[-1]}
        hinge_idx = expected.index(hinge)
        endpoint_idx = expected.index(endpoint)
        assert 0 < hinge_idx < len(expected) - 1

        if endpoint_idx == 0:
            axis = (expected[1][0] - expected[0][0], expected[1][1] - expected[0][1])
            active_old = expected[hinge_idx + 1 :]
            fixed_cells = expected[: hinge_idx + 1]
        else:
            axis = (expected[-2][0] - expected[-1][0], expected[-2][1] - expected[-1][1])
            active_old = expected[:hinge_idx]
            fixed_cells = expected[hinge_idx:]
        axis = (int(axis[0]), int(axis[1]))
        assert len(active_old) >= 1
        assert len(fixed_cells) >= 2

        hr, hc = hinge
        if axis[0] == 0:
            up_space = hr
            down_space = grid.shape[0] - 1 - hr
            assert up_space != down_space
            fold_dir = (-1, 0) if up_space > down_space else (1, 0)
        else:
            left_space = hc
            right_space = grid.shape[1] - 1 - hc
            assert left_space != right_space
            fold_dir = (0, -1) if left_space > right_space else (0, 1)

        fixed_colors = {}
        for cell in fixed_cells:
            if cell == endpoint:
                fixed_colors[cell] = ENDPOINT
            elif cell == hinge:
                fixed_colors[cell] = HINGE
            else:
                fixed_colors[cell] = body

        rods.append(
            {
                "body": int(body),
                "endpoint": endpoint,
                "hinge": hinge,
                "axis": axis,
                "fold_dir": fold_dir,
                "fixed_cells": fixed_cells,
                "fixed_colors": fixed_colors,
                "active_old": active_old,
                "active_len": len(active_old),
            }
        )

    rods.sort(key=lambda rod: rod["endpoint"])
    return rods


def _simulate(grid: np.ndarray, trace: bool = False):
    g = np.array(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    rods = _parse_rods(g)
    assert 3 <= len(rods) <= 4
    targets, target_by_cell = _trace_targets(g)
    assert 3 <= len(targets) <= 5

    current = g.copy()
    rod_traces = []

    for rod in rods:
        for cell in rod["active_old"]:
            current[cell] = BG
        for cell, color in rod["fixed_colors"].items():
            current[cell] = int(color)

        fold_cells = []
        hr, hc = rod["hinge"]
        dr, dc = rod["fold_dir"]
        for step in range(1, rod["active_len"] + 1):
            cell = (hr + dr * step, hc + dc * step)
            assert in_bounds(current.shape, *cell)
            fold_cells.append(cell)
            current[cell] = rod["body"]

        new_end = fold_cells[-1]
        ray_painted = []
        hit_target = None
        stop_reason = "edge"
        r = new_end[0] + dr
        c = new_end[1] + dc
        while in_bounds(current.shape, r, c):
            if int(current[r, c]) != BG:
                stop_reason = "blocked"
                if int(current[r, c]) == TARGET:
                    hit_target = target_by_cell[(r, c)]
                    for tr, tc in targets[hit_target]:
                        current[tr, tc] = rod["body"]
                break
            current[r, c] = rod["body"]
            ray_painted.append((r, c))
            r += dr
            c += dc

        if trace:
            rod_traces.append(
                {
                    "endpoint": rod["endpoint"],
                    "hinge": rod["hinge"],
                    "fold_dir": rod["fold_dir"],
                    "body": rod["body"],
                    "ray_painted": ray_painted,
                    "hit_target": hit_target,
                    "stop_reason": stop_reason,
                    "new_end": new_end,
                }
            )

    return (current, rod_traces) if trace else current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked rod, then cast a ray from the folded free end to recolor the hit target.

    Input structure:
    - The grid is a black background with 3-4 straight rods and 3-5 compact yellow target objects.
    - Each rod is horizontal or vertical, uses one unique body color, has one gray endpoint marker,
      and one maroon hinge marker somewhere strictly inside the rod.
    - The gray marker lies on the fixed side of the hinge; the opposite side is the moving tail.

    Rule:
    - Process rods in row-major order of their gray endpoint markers.
    - For each rod, keep the gray-to-hinge segment fixed and rotate the tail 90 degrees around the hinge
      toward the perpendicular side that has more space to the grid edge.
    - From the new tail endpoint, extend a same-colored ray until the first non-black cell or the edge;
      paint only the traversed black cells, and if the blocking cell is yellow, recolor that entire target
      object to the rod color.

    Key insight:
    - The ray direction is not visible in the input: it is created by first identifying the fixed side,
      folding the free tail toward the larger open side, and only then continuing in that new direction.
    """
    return _simulate(grid)


TARGET_SHAPES = [
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
]


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _target_cells(
    anchor: tuple[int, int],
    fold_dir: tuple[int, int],
    shape: list[tuple[int, int]],
    lateral_sign: int,
) -> list[tuple[int, int]]:
    pr, pc = _turn_right(fold_dir)
    pr *= lateral_sign
    pc *= lateral_sign
    ar, ac = anchor
    cells = []
    for depth, lateral in shape:
        cells.append((ar + depth * fold_dir[0] + lateral * pr, ac + depth * fold_dir[1] + lateral * pc))
    return cells


def _rod_input_cells(plan: dict) -> set[tuple[int, int]]:
    hr, hc = plan["hinge"]
    fr, fc = plan["fixed_dir"]
    cells = {(hr, hc)}
    for step in range(1, plan["fixed_steps"] + 1):
        cells.add((hr + fr * step, hc + fc * step))
    for step in range(1, plan["active_len"] + 1):
        cells.add((hr - fr * step, hc - fc * step))
    return cells


def _rod_fold_cells(plan: dict) -> list[tuple[int, int]]:
    hr, hc = plan["hinge"]
    dr, dc = plan["fold_dir"]
    return [(hr + dr * step, hc + dc * step) for step in range(1, plan["active_len"] + 1)]


def _rod_new_end(plan: dict) -> tuple[int, int]:
    return _rod_fold_cells(plan)[-1]


def _paint_rod(grid: np.ndarray, plan: dict) -> None:
    hr, hc = plan["hinge"]
    fr, fc = plan["fixed_dir"]
    body = plan["body"]
    for step in range(1, plan["fixed_steps"] + 1):
        grid[hr + fr * step, hc + fc * step] = body
    grid[hr, hc] = HINGE
    grid[hr + fr * plan["fixed_steps"], hc + fc * plan["fixed_steps"]] = ENDPOINT
    for step in range(1, plan["active_len"] + 1):
        grid[hr - fr * step, hc - fc * step] = body


def _line_target_count(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    target_by_cell: dict[tuple[int, int], int],
) -> set[int]:
    out: set[int] = set()
    r, c = start
    dr, dc = direction
    while in_bounds(grid.shape, r, c):
        if int(grid[r, c]) == TARGET:
            out.add(target_by_cell[(r, c)])
        r += dr
        c += dc
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    rods = _parse_rods(grid)
    targets, target_by_cell = _trace_targets(grid)
    if not (3 <= len(rods) <= 4 and 3 <= len(targets) <= 5):
        return False
    if len({rod["body"] for rod in rods}) != len(rods):
        return False

    for target in targets:
        if not (3 <= len(target) <= 6):
            return False
        coords = np.array(target, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        if max(r1 - r0 + 1, c1 - c0 + 1) > 3:
            return False

    for rod in rods:
        start = (rod["hinge"][0] + rod["fold_dir"][0], rod["hinge"][1] + rod["fold_dir"][1])
        if len(_line_target_count(grid, start, rod["fold_dir"], target_by_cell)) != 1:
            return False
        opposite = (-rod["fold_dir"][0], -rod["fold_dir"][1])
        opp_start = (rod["hinge"][0] + opposite[0], rod["hinge"][1] + opposite[1])
        if _line_target_count(grid, opp_start, opposite, target_by_cell):
            return False

        body_or_mark = {rod["body"], ENDPOINT, HINGE}
        new_end = _rod_new_end(
            {
                "hinge": rod["hinge"],
                "fold_dir": rod["fold_dir"],
                "active_len": rod["active_len"],
            }
        )
        ray_probe = (new_end[0] + rod["fold_dir"][0], new_end[1] + rod["fold_dir"][1])
        seen_target = False
        while in_bounds(grid.shape, *ray_probe):
            value = int(grid[ray_probe])
            if value != BG:
                if value != TARGET:
                    return False
                seen_target = True
                break
            ray_probe = (ray_probe[0] + rod["fold_dir"][0], ray_probe[1] + rod["fold_dir"][1])
        if not seen_target:
            return False

    out, traces = _simulate(grid, trace=True)
    if out.shape != grid.shape:
        return False
    if max(grid.shape) > 30:
        return False

    hit_count = sum(trace["hit_target"] is not None for trace in traces)
    blocked_count = sum(trace["stop_reason"] == "blocked" and trace["hit_target"] is None for trace in traces)
    if hit_count < 2:
        return False
    if max(len(trace["ray_painted"]) for trace in traces) < 2:
        return False
    if blocked_count and not any(len(trace["ray_painted"]) == 0 for trace in traces):
        return False
    return True


def _generic_target_and_reservations(
    plan: dict,
    shape: list[tuple[int, int]],
    gap: int,
    lateral_sign: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    new_end = _rod_new_end(plan)
    anchor = (new_end[0] + plan["fold_dir"][0] * gap, new_end[1] + plan["fold_dir"][1] * gap)
    target = _target_cells(anchor, plan["fold_dir"], shape, lateral_sign)
    fold_cells = set(_rod_fold_cells(plan))
    ray_cells = set()
    r = new_end[0] + plan["fold_dir"][0]
    c = new_end[1] + plan["fold_dir"][1]
    while (r, c) != anchor:
        ray_cells.add((r, c))
        r += plan["fold_dir"][0]
        c += plan["fold_dir"][1]
    return target, fold_cells | ray_cells | set(target), anchor


def _can_place_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    guarded: set[tuple[int, int]],
    allow_touch: set[tuple[int, int]] | None = None,
) -> bool:
    allow_touch = allow_touch or set()
    for cell in cells:
        if not in_bounds(shape, *cell):
            return False
        if cell in occupied:
            return False
        if cell in guarded and cell not in allow_touch:
            return False
    return True


def _try_place_random_rod(
    h: int,
    w: int,
    body: int,
    occupied_input: set[tuple[int, int]],
    guarded_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[dict, list[tuple[int, int]], set[tuple[int, int]]] | None:
    shape_hw = (h, w)
    for _ in range(250):
        fold_dir = random.choice(DIRS4)
        fixed_dir = random.choice(_perp_options(fold_dir))
        active_len = random.randint(2, 4)
        fixed_steps = random.randint(2, 4)
        if active_len + fixed_steps + 1 > 8:
            continue

        plan = {
            "body": body,
            "fold_dir": fold_dir,
            "fixed_dir": fixed_dir,
            "active_len": active_len,
            "fixed_steps": fixed_steps,
        }

        shape = random.choice(TARGET_SHAPES)
        gap = random.randint(2, 4)
        lateral_sign = random.choice((-1, 1))

        for _ in range(100):
            hinge = (random.randint(1, h - 2), random.randint(1, w - 2))
            plan["hinge"] = hinge
            rod_cells = _rod_input_cells(plan)
            target_cells, out_reserved, _anchor = _generic_target_and_reservations(plan, shape, gap, lateral_sign)
            target_set = set(target_cells)
            all_input = rod_cells | target_set
            if not _can_place_cells(all_input, shape_hw, occupied_input, guarded_input):
                continue
            if any(cell in reserved_output for cell in rod_cells):
                continue
            if any(not in_bounds(shape_hw, *cell) for cell in out_reserved):
                continue
            if any(cell in occupied_input for cell in out_reserved):
                continue
            if any(cell in reserved_output for cell in out_reserved):
                continue

            hr, hc = hinge
            dr, dc = fold_dir
            if dr != 0:
                if (dr == -1 and hr <= h - 1 - hr) or (dr == 1 and h - 1 - hr <= hr):
                    continue
            else:
                if (dc == -1 and hc <= w - 1 - hc) or (dc == 1 and w - 1 - hc <= hc):
                    continue
            return plan.copy(), target_cells, out_reserved
    return None


def _try_place_blocked_pair(
    h: int,
    w: int,
    body_a: int,
    body_b: int,
    occupied_input: set[tuple[int, int]],
    guarded_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[list[dict], list[list[tuple[int, int]]], list[set[tuple[int, int]]]] | None:
    shape_hw = (h, w)
    for _ in range(300):
        active_b = 2
        fixed_b = random.randint(2, 4)
        active_a = random.randint(2, 3)
        fixed_a = 4

        c_a = random.randint(1, max(1, w // 4))
        c_b_min = c_a + active_a + 4
        c_b_max = w - 5
        if c_b_min > c_b_max:
            continue
        c_b = random.randint(c_b_min, c_b_max)
        r_b_max = min(h // 3, h - 9)
        if r_b_max < 2:
            continue
        r_b = random.randint(2, r_b_max)
        r_a = r_b + active_b + 1

        a = {
            "body": body_a,
            "hinge": (r_a, c_a),
            "fixed_dir": (-1, 0),
            "fixed_steps": fixed_a,
            "fold_dir": (0, 1),
            "active_len": active_a,
        }
        b = {
            "body": body_b,
            "hinge": (r_b, c_b),
            "fixed_dir": (0, 1),
            "fixed_steps": fixed_b,
            "fold_dir": (1, 0),
            "active_len": active_b,
        }

        a_input = _rod_input_cells(a)
        b_input = _rod_input_cells(b)
        if a_input & b_input:
            continue
        if any(not in_bounds(shape_hw, *cell) for cell in a_input | b_input):
            continue
        a_shape = random.choice(TARGET_SHAPES)
        b_shape = random.choice(TARGET_SHAPES)
        a_gap = random.randint(c_b - (c_a + active_a) + 1, c_b - (c_a + active_a) + 2)
        b_gap = random.randint((r_a + 2) - (r_b + active_b), (r_a + 4) - (r_b + active_b))
        a_target, a_reserved, a_anchor = _generic_target_and_reservations(a, a_shape, a_gap, random.choice((-1, 1)))
        b_target, b_reserved, b_anchor = _generic_target_and_reservations(b, b_shape, b_gap, random.choice((-1, 1)))

        intersection = (r_a, c_b)
        if intersection not in a_reserved:
            continue
        if intersection not in b_reserved:
            continue
        if intersection in set(b_target):
            continue
        if b_anchor[0] <= r_a + 1:
            continue
        if a_anchor[1] <= c_b + 1:
            continue

        all_new_input = a_input | b_input | set(a_target) | set(b_target)
        if not _can_place_cells(all_new_input, shape_hw, occupied_input, guarded_input):
            continue
        if any(cell in reserved_output for cell in a_input | b_input):
            continue

        a_reserved_minus = set(a_reserved)
        b_reserved_minus = set(b_reserved)
        if any(not in_bounds(shape_hw, *cell) for cell in a_reserved_minus | b_reserved_minus):
            continue
        if (a_reserved_minus - {intersection}) & occupied_input:
            continue
        if (b_reserved_minus - {intersection}) & occupied_input:
            continue
        if (a_reserved_minus - {intersection}) & reserved_output:
            continue
        if (b_reserved_minus - {intersection}) & reserved_output:
            continue
        if (a_reserved_minus - {intersection}) & (b_reserved_minus - {intersection}):
            continue

        endpoints = [(r_a - fixed_a, c_a), (r_b, c_b + fixed_b)]
        if not (endpoints[0] < endpoints[1]):
            continue

        return [a, b], [a_target, b_target], [a_reserved_minus, b_reserved_minus]
    return None


def _try_place_distractor_target(
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
    guarded_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    shape_hw = (h, w)
    for _ in range(120):
        shape = random.choice(TARGET_SHAPES)
        fold_dir = random.choice(DIRS4)
        lateral_sign = random.choice((-1, 1))
        anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
        cells = _target_cells(anchor, fold_dir, shape, lateral_sign)
        cell_set = set(cells)
        if not _can_place_cells(cell_set, shape_hw, occupied_input, guarded_input):
            continue
        if cell_set & reserved_output:
            continue
        return cells
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(16, 20)
        w = random.randint(16, 20)
        rod_count = random.choice([3, 4])
        use_blocked_pair = random.random() < 0.55

        grid = np.zeros((h, w), dtype=int)
        occupied_input: set[tuple[int, int]] = set()
        guarded_input: set[tuple[int, int]] = set()
        reserved_output: set[tuple[int, int]] = set()
        plans: list[dict] = []
        targets: list[list[tuple[int, int]]] = []

        colors = random.sample(list(ROD_PALETTE), rod_count)

        if use_blocked_pair:
            blocked = _try_place_blocked_pair(h, w, colors[0], colors[1], occupied_input, guarded_input, reserved_output)
            if blocked is None:
                continue
            pair_plans, pair_targets, pair_reserved = blocked
            for plan, target, reserve in zip(pair_plans, pair_targets, pair_reserved):
                rod_cells = _rod_input_cells(plan)
                target_set = set(target)
                occupied_input |= rod_cells | target_set
                guarded_input = _expand8(occupied_input, (h, w))
                reserved_output |= reserve
                plans.append(plan)
                targets.append(target)
            start_index = 2
        else:
            start_index = 0

        ok = True
        for idx in range(start_index, rod_count):
            placed = _try_place_random_rod(h, w, colors[idx], occupied_input, guarded_input, reserved_output)
            if placed is None:
                ok = False
                break
            plan, target, reserve = placed
            rod_cells = _rod_input_cells(plan)
            target_set = set(target)
            occupied_input |= rod_cells | target_set
            guarded_input = _expand8(occupied_input, (h, w))
            reserved_output |= reserve
            plans.append(plan)
            targets.append(target)
        if not ok:
            continue

        if len(targets) < 5 and random.random() < 0.45:
            extra = _try_place_distractor_target(h, w, occupied_input, guarded_input, reserved_output)
            if extra is not None:
                occupied_input |= set(extra)
                guarded_input = _expand8(occupied_input, (h, w))
                targets.append(extra)

        for plan in plans:
            _paint_rod(grid, plan)
        for target in targets:
            for r, c in target:
                grid[r, c] = TARGET

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate valid puzzle input")
