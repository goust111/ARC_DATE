from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
AZURE = 8
MAROON = 9

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

REFLECT_BACKSLASH = {
    RIGHT: DOWN,
    DOWN: RIGHT,
    LEFT: UP,
    UP: LEFT,
}
REFLECT_SLASH = {
    RIGHT: UP,
    UP: RIGHT,
    LEFT: DOWN,
    DOWN: LEFT,
}

SOURCE_COLORS = (1, 2, 3, 4, 6, 7)

GEN_H_RANGE = (11, 17)
GEN_W_RANGE = (11, 17)
GEN_MAX_ATTEMPTS = 400
GEN_PATH_ATTEMPTS = 80
GEN_EXTRA_RAY_PROB = 0.55
GEN_EXTRA_TARGET_PROB = 0.65
GEN_EXTRA_MIRROR_PROB = 0.6
GEN_EXTRA_TARGET_MAX = 2
GEN_EXTRA_MIRROR_MAX = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _source_direction(shape: tuple[int, int], r: int, c: int) -> int:
    h, w = shape
    assert _is_edge(shape, r, c), "sources must lie on the grid edge"
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "corner sources are ambiguous"
    if r == 0:
        return DOWN
    if r == h - 1:
        return UP
    if c == 0:
        return RIGHT
    return LEFT


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mirror_patch(orientation: str) -> np.ndarray:
    if orientation == "\\":
        return np.array([[MAROON, GREY], [GREY, MAROON]], dtype=int)
    assert orientation == "/"
    return np.array([[GREY, MAROON], [MAROON, GREY]], dtype=int)


def _reflect(direction: int, orientation: str) -> int:
    if orientation == "\\":
        return REFLECT_BACKSLASH[direction]
    return REFLECT_SLASH[direction]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    g = np.asarray(grid, dtype=int)
    mask = (g == GREY) | (g == MAROON)
    mirror_by_cell: dict[tuple[int, int], str] = {}
    for comp in _component_cells(mask):
        assert len(comp) == 4, "each mirror must be a separate 2x2 component"
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(rows) == 2 and len(cols) == 2, "mirror component must span 2 rows and 2 cols"
        r0, r1 = rows
        c0, c1 = cols
        assert r1 == r0 + 1 and c1 == c0 + 1, "mirror bbox must be exactly 2x2"
        assert 0 < r0 and r1 < g.shape[0] - 1 and 0 < c0 and c1 < g.shape[1] - 1, "mirrors must be internal"

        patch = g[r0 : r0 + 2, c0 : c0 + 2]
        maroon_positions = {(int(r), int(c)) for r, c in np.argwhere(patch == MAROON)}
        grey_positions = {(int(r), int(c)) for r, c in np.argwhere(patch == GREY)}
        assert len(maroon_positions) == 2 and len(grey_positions) == 2, "mirror patch must use two maroon and two grey pixels"
        if maroon_positions == {(0, 0), (1, 1)}:
            orientation = "\\"
        else:
            assert maroon_positions == {(0, 1), (1, 0)}, "mirror encoding must use a diagonal"
            orientation = "/"

        for cell in comp:
            mirror_by_cell[cell] = orientation
    return mirror_by_cell


def _find_sources(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources: list[tuple[int, int, int, int]] = []
    seen_colors: set[int] = set()
    for r in range(h):
        for c in range(w):
            if not _is_edge(g.shape, r, c):
                continue
            color = int(g[r, c])
            if color in (BG, GREY, AZURE, MAROON):
                continue
            direction = _source_direction(g.shape, r, c)
            assert color not in seen_colors, "source colors must be unique"
            seen_colors.add(color)
            sources.append((r, c, color, direction))
    assert sources, "expected at least one edge source"
    sources.sort()
    return sources


def _find_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    targets = {(int(r), int(c)) for r, c in np.argwhere(g == AZURE)}
    assert targets, "expected at least one azure target"
    assert all(not _is_edge(g.shape, r, c) for r, c in targets), "targets must be internal"
    return targets


def _validate_scene(
    grid: np.ndarray,
    mirrors: dict[tuple[int, int], str],
    sources: list[tuple[int, int, int, int]],
    targets: set[tuple[int, int]],
) -> None:
    source_cells = {(r, c) for r, c, _color, _direction in sources}
    mirror_cells = set(mirrors.keys())
    role_cells = source_cells | mirror_cells | targets
    non_bg = {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}
    assert role_cells == non_bg, "all non-background pixels must be sources, mirrors, or targets"
    assert source_cells.isdisjoint(mirror_cells)
    assert source_cells.isdisjoint(targets)
    assert mirror_cells.isdisjoint(targets)


def _trace_ray(
    grid: np.ndarray,
    out: np.ndarray,
    source: tuple[int, int, int, int],
    mirror_by_cell: dict[tuple[int, int], str],
    source_cells: set[tuple[int, int]],
    targets: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    r, c, color, direction = source
    seen: set[tuple[int, int, int]] = set()
    while True:
        state = (r, c, direction)
        if state in seen:
            break
        seen.add(state)

        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if (nr, nc) in source_cells:
            break
        if (nr, nc) in targets:
            out[nr, nc] = color
            break
        if (nr, nc) in mirror_by_cell:
            direction = _reflect(direction, mirror_by_cell[(nr, nc)])
            continue
        if int(grid[nr, nc]) == BG:
            out[nr, nc] = color
            r, c = nr, nc
            continue
        break


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect colored edge rays with diagonal-coded 2x2 mirrors and repaint any hit azure targets.

    Input structure
    - Background is black.
    - Sources are unique non-grey, non-maroon, non-azure single pixels on the edge.
    - Mirrors are internal 2x2 blocks made of grey and maroon, where the maroon diagonal encodes "/" vs "\".
    - Targets are internal azure singleton pixels.

    Rule
    - Each source emits inward from its edge side.
    - When a ray would enter a mirror cell, it reflects according to the mirror orientation and continues from the same side.
    - Rays paint only background cells, recolor the first target they hit, and stop on targets, sources, or the grid boundary.
    - Mirrors and sources stay unchanged; untouched targets stay azure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mirrors = _parse_mirrors(g)
    sources = _find_sources(g)
    targets = _find_targets(g)
    _validate_scene(g, mirrors, sources, targets)

    out = g.copy()
    source_cells = {(r, c) for r, c, _color, _direction in sources}
    for source in sources:
        _trace_ray(g, out, source, mirrors, source_cells, targets)
    return out


def _orthogonal_dirs(direction: int) -> tuple[int, int]:
    return ((direction + 1) % 4, (direction + 3) % 4)


def _mirror_for_turn(turn_point: tuple[int, int], incoming: int, outgoing: int) -> tuple[int, int, str, tuple[tuple[int, int], ...]]:
    r, c = turn_point
    key = (incoming, outgoing)
    if key == (RIGHT, UP):
        r0, c0, orientation = r - 1, c + 1, "/"
    elif key == (RIGHT, DOWN):
        r0, c0, orientation = r, c + 1, "\\"
    elif key == (LEFT, UP):
        r0, c0, orientation = r - 1, c - 2, "\\"
    elif key == (LEFT, DOWN):
        r0, c0, orientation = r, c - 2, "/"
    elif key == (UP, RIGHT):
        r0, c0, orientation = r - 2, c, "/"
    elif key == (UP, LEFT):
        r0, c0, orientation = r - 2, c - 1, "\\"
    elif key == (DOWN, RIGHT):
        r0, c0, orientation = r + 1, c, "\\"
    else:
        assert key == (DOWN, LEFT)
        r0, c0, orientation = r + 1, c - 1, "/"
    cells = ((r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1))
    return r0, c0, orientation, cells


def _run_ahead(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    run: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        if (r, c) in occupied:
            break
        run.append((r, c))
    return run


def _expanded(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _random_source(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = shape
    candidates = [(0, c) for c in range(1, w - 1)]
    candidates += [(h - 1, c) for c in range(1, w - 1)]
    candidates += [(r, 0) for r in range(1, h - 1)]
    candidates += [(r, w - 1) for r in range(1, h - 1)]
    random.shuffle(candidates)
    for cell in candidates:
        if cell not in occupied:
            return cell
    return None


def _search_path(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: int,
    mode: str,
    turns: int,
    occupied: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
) -> dict[str, object] | None:
    assert mode in {"hit", "miss"}

    def rec(
        current: tuple[int, int],
        current_direction: int,
        turns_left: int,
        used: set[tuple[int, int]],
        mirror_used: set[tuple[int, int]],
        path_cells: list[tuple[int, int]],
        mirrors: list[tuple[int, int, str]],
    ) -> dict[str, object] | None:
        run = _run_ahead(shape, current, current_direction, used)
        if turns_left == 0:
            finish_order: list[tuple[str, int]] = []
            if mode == "hit":
                for idx in range(1, len(run)):
                    tr, tc = run[idx]
                    if not _is_edge(shape, tr, tc):
                        finish_order.append(("hit", idx))
            else:
                if run and _is_edge(shape, *run[-1]):
                    finish_order.append(("miss", len(run) - 1))
            random.shuffle(finish_order)
            for kind, idx in finish_order:
                if kind == "hit":
                    target = run[idx]
                    return {
                        "source": source,
                        "path_cells": path_cells + run[:idx],
                        "mirrors": mirrors,
                        "target": target,
                        "mode": "hit",
                    }
                return {
                    "source": source,
                    "path_cells": path_cells + run[: idx + 1],
                    "mirrors": mirrors,
                    "target": None,
                    "mode": "miss",
                }
            return None

        turn_candidates: list[tuple[int, int, tuple[int, int], tuple[tuple[int, int], ...], str]] = []
        orthogonal = list(_orthogonal_dirs(current_direction))
        random.shuffle(orthogonal)
        for out_direction in orthogonal:
            for idx, turn_point in enumerate(run):
                r0, c0, orientation, block_cells = _mirror_for_turn(turn_point, current_direction, out_direction)
                if not all(_in_bounds(shape, r, c) for r, c in block_cells):
                    continue
                if any(_is_edge(shape, r, c) for r, c in block_cells):
                    continue
                if any(cell in used for cell in block_cells):
                    continue
                if _expanded(block_cells) & mirror_used:
                    continue
                next_dr, next_dc = DIRS[out_direction]
                next_cell = (turn_point[0] + next_dr, turn_point[1] + next_dc)
                if not _in_bounds(shape, *next_cell) or next_cell in used or next_cell in block_cells:
                    continue
                turn_candidates.append((idx, out_direction, turn_point, block_cells, orientation))

        random.shuffle(turn_candidates)
        for idx, out_direction, turn_point, block_cells, orientation in turn_candidates:
            seg = run[: idx + 1]
            next_used = used | set(seg) | set(block_cells)
            next_mirror_cells = mirror_used | set(block_cells)
            result = rec(
                turn_point,
                out_direction,
                turns_left - 1,
                next_used,
                next_mirror_cells,
                path_cells + seg,
                mirrors + [(block_cells[0][0], block_cells[0][1], orientation)],
            )
            if result is not None:
                return result
        return None

    return rec(source, direction, turns, occupied | {source}, set(mirror_cells), [], [])


def _plan_ray(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    mirror_cells: set[tuple[int, int]],
    mode: str,
    turns_range: tuple[int, int],
) -> dict[str, object] | None:
    for _ in range(GEN_PATH_ATTEMPTS):
        source = _random_source(shape, occupied)
        if source is None:
            return None
        direction = _source_direction(shape, *source)
        turns = random.randint(turns_range[0], turns_range[1])
        plan = _search_path(shape, source, direction, mode, turns, occupied, mirror_cells)
        if plan is not None:
            return plan
    return None


def _commit_ray(
    scene: dict[str, object],
    plan: dict[str, object],
    color: int,
) -> None:
    sources = scene["sources"]
    targets = scene["targets"]
    hit_targets = scene["hit_targets"]
    path_cells = scene["path_cells"]
    used = scene["used"]
    mirrors = scene["mirrors"]
    mirror_cells = scene["mirror_cells"]

    source = tuple(plan["source"])
    sources.append((source[0], source[1], color))
    used.add(source)

    for r0, c0, orientation in plan["mirrors"]:
        mirrors.append((r0, c0, orientation))
        block = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
        mirror_cells.update(block)
        used.update(block)

    for cell in plan["path_cells"]:
        path_cells.add(tuple(cell))
        used.add(tuple(cell))

    if plan["target"] is not None:
        target = tuple(plan["target"])
        targets.append(target)
        hit_targets[target] = color
        used.add(target)


def _place_extra_targets(scene: dict[str, object], shape: tuple[int, int]) -> None:
    used = scene["used"]
    targets = scene["targets"]
    untouched = scene["untouched_targets"]
    mirror_cells = scene["mirror_cells"]

    max_count = random.randint(0, GEN_EXTRA_TARGET_MAX)
    for _ in range(max_count):
        if random.random() > GEN_EXTRA_TARGET_PROB:
            continue
        candidates = []
        for r in range(1, shape[0] - 1):
            for c in range(1, shape[1] - 1):
                cell = (r, c)
                if cell in used:
                    continue
                if cell in _expanded(mirror_cells):
                    continue
                candidates.append(cell)
        if not candidates:
            return
        cell = random.choice(candidates)
        targets.append(cell)
        untouched.add(cell)
        used.add(cell)


def _place_extra_mirrors(scene: dict[str, object], shape: tuple[int, int]) -> None:
    used = scene["used"]
    mirrors = scene["mirrors"]
    mirror_cells = scene["mirror_cells"]

    max_count = random.randint(0, GEN_EXTRA_MIRROR_MAX)
    for _ in range(max_count):
        if random.random() > GEN_EXTRA_MIRROR_PROB:
            continue
        candidates: list[tuple[int, int, str]] = []
        for r0 in range(1, shape[0] - 2):
            for c0 in range(1, shape[1] - 2):
                block = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
                if block & used:
                    continue
                if _expanded(block) & mirror_cells:
                    continue
                for orientation in ("\\", "/"):
                    candidates.append((r0, c0, orientation))
        if not candidates:
            return
        r0, c0, orientation = random.choice(candidates)
        mirrors.append((r0, c0, orientation))
        block = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
        mirror_cells.update(block)
        used.update(block)


def _render_scene(shape: tuple[int, int], scene: dict[str, object]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r0, c0, orientation in scene["mirrors"]:
        grid[r0 : r0 + 2, c0 : c0 + 2] = _mirror_patch(orientation)
    for r, c in scene["targets"]:
        grid[r, c] = AZURE
    for r, c, color in scene["sources"]:
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    shape = (h, w)
    scene: dict[str, object] = {
        "used": set(),
        "mirror_cells": set(),
        "mirrors": [],
        "sources": [],
        "targets": [],
        "path_cells": set(),
        "hit_targets": {},
        "untouched_targets": set(),
    }

    colors = list(SOURCE_COLORS)
    random.shuffle(colors)

    hit_plan = _plan_ray(shape, scene["used"], scene["mirror_cells"], "hit", (1, 4))
    if hit_plan is None:
        return None
    _commit_ray(scene, hit_plan, colors.pop())

    miss_plan = _plan_ray(shape, scene["used"], scene["mirror_cells"], "miss", (0, 2))
    if miss_plan is None:
        return None
    _commit_ray(scene, miss_plan, colors.pop())

    if colors and random.random() < GEN_EXTRA_RAY_PROB:
        extra_mode = random.choice(("hit", "miss"))
        extra_turns = (0, 3) if extra_mode == "hit" else (0, 2)
        extra_plan = _plan_ray(shape, scene["used"], scene["mirror_cells"], extra_mode, extra_turns)
        if extra_plan is not None:
            _commit_ray(scene, extra_plan, colors.pop())

    _place_extra_targets(scene, shape)
    _place_extra_mirrors(scene, shape)

    grid = _render_scene(shape, scene)
    out = main(grid)

    hit_targets: dict[tuple[int, int], int] = scene["hit_targets"]
    untouched_targets: set[tuple[int, int]] = scene["untouched_targets"]
    for target, color in hit_targets.items():
        if int(out[target]) != color:
            return None
    for target in untouched_targets:
        if int(out[target]) != AZURE:
            return None
    if not np.any(out != grid):
        return None
    if len(scene["sources"]) < 2:
        return None
    if len(scene["mirrors"]) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirror-ray input")
