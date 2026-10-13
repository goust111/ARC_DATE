from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CROSS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (15, 23)
COMPONENT_COUNT_MIN = 2
COMPONENT_COUNT_MAX = 5
TAIL_COUNT_RANGE = (1, 3)
TAIL_LENGTH_RANGE = (1, 5)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _find_components(g: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or bool(seen[r, c]):
                continue

            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            top_left = min(coords)
            components.append({"color": color, "coords": coords, "top_left": top_left})

    components.sort(key=lambda comp: comp["top_left"])
    return components


def _local_density(g: np.ndarray, r: int, c: int, color: int) -> int:
    r0 = max(0, int(r) - 1)
    r1 = min(int(g.shape[0]), int(r) + 2)
    c0 = max(0, int(c) - 1)
    c1 = min(int(g.shape[1]), int(c) + 2)
    return int(np.count_nonzero(g[r0:r1, c0:c1] == int(color)))


def _visual_centroid(g: np.ndarray, coords: list[tuple[int, int]], color: int) -> tuple[int, int]:
    best_pos = coords[0]
    best_score = -1
    for r, c in coords:
        score = _local_density(g, int(r), int(c), int(color))
        if score > best_score or (score == best_score and (int(r), int(c)) < best_pos):
            best_score = score
            best_pos = (int(r), int(c))
    return best_pos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the densest 3x3 neighborhood inside each foreground component and place a same-color cross there.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _find_components(g)

    jobs: list[tuple[int, int, int, set[tuple[int, int]]]] = []
    for comp in components:
        color = int(comp["color"])
        coords = list(comp["coords"])
        center = _visual_centroid(g, coords, color)
        jobs.append((int(center[0]), int(center[1]), color, set(coords)))

    jobs.sort(key=lambda job: (job[0], job[1], job[2]))
    h, w = map(int, g.shape)

    for cr, cc, color, own_cells in jobs:
        for dr, dc in CROSS:
            rr = int(cr + dr)
            cc2 = int(cc + dc)
            if not (0 <= rr < h and 0 <= cc2 < w):
                continue
            if int(g[rr, cc2]) == BG or (rr, cc2) in own_cells:
                out[rr, cc2] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _unique_core() -> tuple[set[tuple[int, int]], tuple[int, int], bool]:
    cells = {(r, c) for r in range(3) for c in range(3)}
    arm_choices = [(0, 1), (1, 0), (1, 2), (2, 1)]
    random.shuffle(arm_choices)
    removed = arm_choices[: random.choice([1, 2])]
    for cell in removed:
        cells.remove(cell)
    return cells, (3, 3), False


def _tie_core() -> tuple[set[tuple[int, int]], tuple[int, int], bool]:
    if random.choice([False, True]):
        cells = {(r, c) for r in range(3) for c in range(4)}
        notch = [(0, 1), (0, 2)] if random.choice([False, True]) else [(2, 1), (2, 2)]
        core_shape = (3, 4)
    else:
        cells = {(r, c) for r in range(4) for c in range(3)}
        notch = [(1, 0), (2, 0)] if random.choice([False, True]) else [(1, 2), (2, 2)]
        core_shape = (4, 3)
    for cell in notch:
        cells.remove(cell)
    return cells, core_shape, True


def _tail_start_options(core_h: int, core_w: int) -> list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]]:
    options: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []

    for c in range(core_w):
        options.append(((0, c), (-1, 0), [(0, -1), (0, 1)]))
        options.append(((core_h - 1, c), (1, 0), [(0, -1), (0, 1)]))
    for r in range(core_h):
        options.append(((r, 0), (0, -1), [(-1, 0), (1, 0)]))
        options.append(((r, core_w - 1), (0, 1), [(-1, 0), (1, 0)]))
    return options


def _add_tail(cells: set[tuple[int, int]], core_h: int, core_w: int) -> None:
    options = _tail_start_options(core_h, core_w)
    random.shuffle(options)

    for start, direction, turns in options:
        cur = start
        dr, dc = direction
        steps = random.randint(*TAIL_LENGTH_RANGE)
        bend_step = random.randint(1, steps - 1) if steps >= 3 and random.random() < 0.55 else None
        added: list[tuple[int, int]] = []

        for step in range(steps):
            if bend_step is not None and step == bend_step:
                turn_choices = turns[:]
                random.shuffle(turn_choices)
                changed = False
                for cand_dr, cand_dc in turn_choices:
                    nr = int(cur[0] + cand_dr)
                    nc = int(cur[1] + cand_dc)
                    if (nr, nc) not in cells and (nr, nc) not in added:
                        dr, dc = cand_dr, cand_dc
                        changed = True
                        break
                if not changed:
                    bend_step = None

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if (nr, nc) in cells or (nr, nc) in added:
                break
            added.append((nr, nc))
            cur = (nr, nc)

        if not added:
            continue

        cells.update(added)
        if random.random() < 0.4:
            nub_dirs = turns[:]
            random.shuffle(nub_dirs)
            end_r, end_c = added[-1]
            for nub_dr, nub_dc in nub_dirs:
                extra = (int(end_r + nub_dr), int(end_c + nub_dc))
                if extra not in cells:
                    cells.add(extra)
                    break
        return


def _build_component(force_branch: str | None = None) -> dict[str, object]:
    for _ in range(80):
        if force_branch == "tie" or (force_branch is None and random.choice([False, True])):
            cells, (core_h, core_w), tie_branch = _tie_core()
        else:
            cells, (core_h, core_w), tie_branch = _unique_core()

        for _ in range(random.randint(*TAIL_COUNT_RANGE)):
            _add_tail(cells, core_h, core_w)

        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        cells = _normalize(cells)
        h = max(r for r, _c in cells) + 1
        w = max(c for _r, c in cells) + 1
        local_grid = np.zeros((h, w), dtype=int)
        for r, c in cells:
            local_grid[r, c] = 1

        center = _visual_centroid(local_grid, sorted(cells), 1)
        expected_center = (int(1 - min_r), int(1 - min_c))
        if center != expected_center:
            continue

        cross = {(int(center[0] + dr), int(center[1] + dc)) for dr, dc in CROSS}
        visible_additions = {pos for pos in cross if pos not in cells}
        if not visible_additions:
            continue

        return {
            "cells": cells,
            "height": h,
            "width": w,
            "center": center,
            "cross": cross,
            "branch": "tie" if tie_branch else "unique",
            "footprint_size": len(cells | cross),
        }

    raise RuntimeError("failed to build a valid component")


def _dilate(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _place_component(
    grid: np.ndarray,
    comp: dict[str, object],
    blocked: set[tuple[int, int]],
) -> tuple[dict[str, object], set[tuple[int, int]]] | None:
    h, w = map(int, grid.shape)
    ch = int(comp["height"])
    cw = int(comp["width"])
    if ch > h or cw > w:
        return None

    candidate_offsets = [(r0, c0) for r0 in range(h - ch + 1) for c0 in range(w - cw + 1)]
    random.shuffle(candidate_offsets)

    cells = set(comp["cells"])
    cross = set(comp["cross"])
    center_r, center_c = map(int, comp["center"])

    for r0, c0 in candidate_offsets[:PLACEMENT_ATTEMPTS]:
        abs_cells = {(int(r0 + r), int(c0 + c)) for r, c in cells}
        abs_cross = {
            (int(r0 + r), int(c0 + c))
            for r, c in cross
            if 0 <= int(r0 + r) < h and 0 <= int(c0 + c) < w
        }
        footprint = abs_cells | abs_cross
        if footprint & blocked:
            continue

        placed = {
            "cells": abs_cells,
            "cross": abs_cross,
            "center": (int(r0 + center_r), int(c0 + center_c)),
        }
        return placed, _dilate(footprint)

    return None


def _component_budget(height: int, width: int) -> int:
    return min(COMPONENT_COUNT_MAX, max(COMPONENT_COUNT_MIN, (height * width) // 85))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    max_components = min(_component_budget(height, width), 9)
    n_components = random.randint(COMPONENT_COUNT_MIN, max_components)
    colors = random.sample(range(1, 10), n_components)

    branch_plan = ["tie", "unique"] + [random.choice(["tie", "unique"]) for _ in range(n_components - 2)]
    random.shuffle(branch_plan)
    raw_components = [_build_component(branch) for branch in branch_plan]

    raw_components.sort(key=lambda comp: int(comp["footprint_size"]), reverse=True)

    grid = np.zeros((height, width), dtype=int)
    blocked: set[tuple[int, int]] = set()
    placed_components: list[dict[str, object]] = []

    for comp, color in zip(raw_components, colors):
        placed = _place_component(grid, comp, blocked)
        if placed is None:
            return None
        placed_info, blocked_cells = placed
        for r, c in placed_info["cells"]:
            grid[r, c] = int(color)
        blocked |= blocked_cells
        placed_components.append(placed_info)

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    for placed in placed_components:
        cr, cc = placed["center"]
        if out[cr, cc] == BG:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visual-centroid ARC input")
