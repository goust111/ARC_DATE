from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
EMITTER_COLORS = (1, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400
MIN_CONTACT_STEP = 2
MIN_EXIT_STEPS = 2


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_boundary(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _ray_cells(
    shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, (r, c)):
            return out
        out.append((r, c))


def _step_on_ray(
    pos: tuple[int, int], direction: tuple[int, int], cell: tuple[int, int]
) -> int | None:
    pr, pc = pos
    dr, dc = direction
    r, c = cell
    if dr != 0:
        if c != pc:
            return None
        delta = r - pr
        if delta * dr <= 0:
            return None
        return delta // dr
    if r != pr:
        return None
    delta = c - pc
    if delta * dc <= 0:
        return None
    return delta // dc


def _rg_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = (grid == RED) | (grid == GREEN)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cell = stack.pop()
                comp.append(cell)
                for nr, nc in _neighbors4(cell):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _identify_emitter(
    grid: np.ndarray,
) -> tuple[tuple[int, int], int, tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BG, RED, GREEN):
                continue
            if not _is_boundary((h, w), (r, c)):
                continue
            if np.count_nonzero(grid == value) != 1:
                continue
            if (r in (0, h - 1)) and (c in (0, w - 1)):
                continue
            if r == 0:
                direction = (1, 0)
            elif r == h - 1:
                direction = (-1, 0)
            elif c == 0:
                direction = (0, 1)
            else:
                direction = (0, -1)
            candidates.append(((r, c), value, direction))
    assert len(candidates) == 1, "expected exactly one boundary emitter"
    return candidates[0]


def _find_dominoes(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], set[tuple[int, int]]]:
    dominoes: list[dict] = []
    cell_to_domino: dict[tuple[int, int], int] = {}
    distractors: set[tuple[int, int]] = set()
    for comp in _rg_components(grid):
        if len(comp) == 1:
            distractors.add(comp[0])
            continue
        assert len(comp) == 2, "red/green objects must be singles or dominoes"
        a, b = comp
        va = int(grid[a])
        vb = int(grid[b])
        assert {va, vb} == {RED, GREEN}, "domino must contain one red and one green cell"
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "domino must be edge-connected"
        red = a if va == RED else b
        green = a if va == GREEN else b
        idx = len(dominoes)
        dominoes.append(
            {
                "id": idx,
                "red": red,
                "green": green,
                "dir": (green[0] - red[0], green[1] - red[1]),
                "cells": {red, green},
            }
        )
        cell_to_domino[red] = idx
        cell_to_domino[green] = idx
    return dominoes, cell_to_domino, distractors


def _trace_beam(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    shape = tuple(map(int, g.shape))
    emitter_pos, emitter_color, direction = _identify_emitter(g)
    dominoes, cell_to_domino, distractors = _find_dominoes(g)

    used: set[int] = set()
    visited_cells: list[tuple[int, int]] = []
    bg_path: list[tuple[int, int]] = []
    passed_distractors: list[tuple[int, int]] = []
    revisited_domino_cells: list[tuple[int, int]] = []
    used_order: list[int] = []
    state_seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]] = set()

    pos = emitter_pos
    max_steps = g.size * max(4, len(dominoes) * 8)
    for _ in range(max_steps):
        state = (pos, direction, tuple(sorted(used)))
        assert state not in state_seen, "beam entered a loop instead of exiting"
        state_seen.add(state)

        nxt = (pos[0] + direction[0], pos[1] + direction[1])
        if not _in_bounds(shape, nxt):
            return {
                "emitter_pos": emitter_pos,
                "emitter_color": emitter_color,
                "dominoes": dominoes,
                "distractors": distractors,
                "used_order": used_order,
                "bg_path": bg_path,
                "visited_cells": visited_cells,
                "passed_distractors": passed_distractors,
                "revisited_domino_cells": revisited_domino_cells,
            }

        visited_cells.append(nxt)
        value = int(g[nxt])
        domino_id = cell_to_domino.get(nxt)
        if domino_id is not None and domino_id not in used:
            domino = dominoes[domino_id]
            used.add(domino_id)
            used_order.append(domino_id)
            green = domino["green"]
            if green != nxt:
                visited_cells.append(green)
            pos = green
            direction = domino["dir"]
            continue

        if value == BG:
            bg_path.append(nxt)
        elif nxt in distractors:
            passed_distractors.append(nxt)
        elif domino_id is not None:
            revisited_domino_cells.append(nxt)
        pos = nxt

    raise AssertionError("beam did not exit within a safe step budget")


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the beam from the unique boundary emitter and paint the background cells on its full one-use domino path.

    Input structure:
    - The grid is mostly black background.
    - One non-red/non-green singleton lies on a non-corner boundary cell; it is the emitter.
    - The remaining foreground consists of isolated red/green single pixels and separated red-green dominoes.

    Rule:
    - Emit a ray perpendicular to the boundary at the emitter.
    - When the ray first touches an unused red-green domino, jump to that domino's green cell and continue in the red-to-green direction.
    - Each domino can deflect only once; later passes through its cells and through isolated red/green singles do nothing.
    - The ray eventually exits the grid; recolor every visited background cell to the emitter color and leave all non-background pixels unchanged.

    Key insight:
    - Only genuine two-cell red-green pairs are optical elements, and their red-to-green orientation fully determines the next ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    trace = _trace_beam(g)
    out = g.copy()
    emitter_color = int(trace["emitter_color"])
    for r, c in trace["bg_path"]:
        out[r, c] = emitter_color
    return out


def _can_place_pair(
    shape: tuple[int, int],
    red: tuple[int, int],
    green: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    pair = {red, green}
    if red == green:
        return False
    if not (_in_bounds(shape, red) and _in_bounds(shape, green)):
        return False
    if _is_boundary(shape, red) or _is_boundary(shape, green):
        return False
    if red in blocked or green in blocked:
        return False
    for cell in pair:
        for nb in _neighbors4(cell):
            if nb in blocked and nb not in pair:
                return False
    return True


def _can_place_single(
    shape: tuple[int, int],
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]] | None = None,
) -> bool:
    if not _in_bounds(shape, cell) or _is_boundary(shape, cell):
        return False
    if cell in blocked:
        return False
    if forbidden is not None and cell in forbidden:
        return False
    for nb in _neighbors4(cell):
        if nb in blocked:
            return False
    return True


def _first_contact_step(
    pos: tuple[int, int],
    direction: tuple[int, int],
    red: tuple[int, int],
    green: tuple[int, int],
) -> int | None:
    steps = [
        step
        for step in (
            _step_on_ray(pos, direction, red),
            _step_on_ray(pos, direction, green),
        )
        if step is not None
    ]
    if not steps:
        return None
    return min(steps)


def _candidate_dominoes(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
    previous_domino_cells: set[tuple[int, int]],
) -> list[dict]:
    ray = _ray_cells(shape, pos, direction)
    candidates: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for step, contact in enumerate(ray, start=1):
        if step < MIN_CONTACT_STEP:
            continue
        for new_dir in DIRS:
            for touch_green in (False, True):
                if touch_green:
                    green = contact
                    red = (green[0] - new_dir[0], green[1] - new_dir[1])
                else:
                    red = contact
                    green = (red[0] + new_dir[0], red[1] + new_dir[1])
                key = (red, green)
                if key in seen:
                    continue
                seen.add(key)
                if not _can_place_pair(shape, red, green, blocked):
                    continue
                first_step = _first_contact_step(pos, direction, red, green)
                if first_step != step:
                    continue
                segment_cells = ray[:step]
                candidates.append(
                    {
                        "red": red,
                        "green": green,
                        "dir": new_dir,
                        "step": step,
                        "segment_revisit": any(
                            cell in previous_domino_cells for cell in segment_cells
                        ),
                    }
                )
    return candidates


def _search_path(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    remaining: int,
    blocked: set[tuple[int, int]],
    previous_domino_cells: set[tuple[int, int]],
    require_revisit: bool,
    revisited: bool,
) -> list[dict] | None:
    ray = _ray_cells(shape, pos, direction)
    if remaining == 0:
        if len(ray) < MIN_EXIT_STEPS:
            return None
        revisited = revisited or any(cell in previous_domino_cells for cell in ray)
        if require_revisit and not revisited:
            return None
        return []

    candidates = _candidate_dominoes(shape, pos, direction, blocked, previous_domino_cells)
    random.shuffle(candidates)
    if require_revisit and not revisited:
        candidates.sort(key=lambda item: (not item["segment_revisit"], abs(item["step"] - 4)))
    else:
        candidates.sort(key=lambda item: abs(item["step"] - 4))

    for cand in candidates[:80]:
        next_blocked = blocked | {cand["red"], cand["green"]}
        next_prev = previous_domino_cells | {cand["red"], cand["green"]}
        next_revisited = revisited or bool(cand["segment_revisit"])
        tail = _search_path(
            shape,
            cand["green"],
            cand["dir"],
            remaining - 1,
            next_blocked,
            next_prev,
            require_revisit,
            next_revisited,
        )
        if tail is not None:
            return [cand] + tail
    return None


def _unused_domino_candidates(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            red = (r, c)
            for dr, dc in DIRS:
                green = (r + dr, c + dc)
                if not _can_place_pair(shape, red, green, blocked):
                    continue
                if red in forbidden_cells or green in forbidden_cells:
                    continue
                options.append((red, green))
    random.shuffle(options)
    return options


def _place_extra_dominoes(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    forbidden_cells: set[tuple[int, int]],
    count: int,
) -> int:
    placed = 0
    for red, green in _unused_domino_candidates(grid.shape, blocked, forbidden_cells):
        grid[red] = RED
        grid[green] = GREEN
        blocked |= {red, green}
        placed += 1
        if placed == count:
            return placed
    return placed


def _place_singles_from_candidates(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    candidates: list[tuple[int, int]],
    target: int,
) -> list[tuple[int, int]]:
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for cell in candidates:
        if not _can_place_single(grid.shape, cell, blocked):
            continue
        grid[cell] = random.choice((RED, GREEN))
        blocked.add(cell)
        placed.append(cell)
        if len(placed) == target:
            break
    return placed


def _emitter_position(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randint(1, w - 2)), (1, 0)
    if side == "bottom":
        return (h - 1, random.randint(1, w - 2)), (-1, 0)
    if side == "left":
        return (random.randint(1, h - 2), 0), (0, 1)
    return (random.randint(1, h - 2), w - 1), (0, -1)


def _validate_scene(
    grid: np.ndarray,
    planned_deflections: int,
    require_revisit: bool,
    path_distractors: set[tuple[int, int]],
) -> bool:
    try:
        trace = _trace_beam(grid)
        dominoes = trace["dominoes"]
        distractors = trace["distractors"]
        if len(trace["used_order"]) != planned_deflections:
            return False
        if len(dominoes) <= planned_deflections:
            return False
        if len(distractors) < 2:
            return False
        if len(trace["bg_path"]) < 6:
            return False
        if require_revisit and not trace["revisited_domino_cells"]:
            return False
        if not path_distractors.issubset(set(trace["passed_distractors"])):
            return False
        out = main(grid)
        diff = {(int(r), int(c)) for r, c in np.argwhere(out != grid)}
        bg_path = set(trace["bg_path"])
        if diff != bg_path:
            return False
        emitter_color = int(trace["emitter_color"])
        for r, c in diff:
            if int(grid[r, c]) != BG or int(out[r, c]) != emitter_color:
                return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    shape = (h, w)
    emitter_pos, emitter_dir = _emitter_position(shape)
    emitter_color = random.choice(EMITTER_COLORS)

    require_revisit = random.random() < 0.4
    planned_deflections = random.randint(3, 4) if require_revisit else random.randint(1, 4)
    path = _search_path(
        shape,
        emitter_pos,
        emitter_dir,
        planned_deflections,
        set(),
        set(),
        require_revisit,
        False,
    )
    if path is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    grid[emitter_pos] = emitter_color
    blocked: set[tuple[int, int]] = set()
    for domino in path:
        grid[domino["red"]] = RED
        grid[domino["green"]] = GREEN
        blocked |= {domino["red"], domino["green"]}

    trace = _trace_beam(grid)
    if len(trace["used_order"]) != planned_deflections:
        return None
    if require_revisit and not trace["revisited_domino_cells"]:
        return None

    visited = set(trace["visited_cells"])
    path_bg = list(dict.fromkeys(trace["bg_path"]))

    want_path_distractors = random.random() < 0.75
    max_path_distractors = 2 if require_revisit else 1
    path_target = random.randint(1, max_path_distractors) if want_path_distractors else 0
    path_distractors: set[tuple[int, int]] = set(
        _place_singles_from_candidates(grid, blocked, path_bg, path_target)
    )
    if path_target > 0 and len(path_distractors) == 0:
        return None

    extra_domino_target = random.randint(1, 3)
    extra_dominoes = _place_extra_dominoes(grid, blocked, visited, extra_domino_target)
    if extra_dominoes == 0:
        return None

    off_path_candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in visited and int(grid[r, c]) == BG
    ]
    off_path_target = max(2 - len(path_distractors), 1) + random.randint(0, 2)
    _place_singles_from_candidates(grid, blocked, off_path_candidates, off_path_target)

    if not _validate_scene(grid, planned_deflections, require_revisit, path_distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid one-use reflective domino input")
