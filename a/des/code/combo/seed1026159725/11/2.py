from __future__ import annotations

import random
from typing import Iterable

import numpy as np


YELLOW = 4
PRISM = 5
PALETTE = [0, 1, 2, 3, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
SOURCE_COUNT_RANGE = (3, 5)
MAX_GENERATE_ATTEMPTS = 500
MAX_FILL_FRACTION = 0.42


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _edge_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    assert _is_edge(shape, r, c), "sources must lie on the grid edge"
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "corner sources are ambiguous"
    if r == 0:
        return (1, 0)
    if r == h - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    return (0, -1)


def _split_directions(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if dr == 1 and dc == 0:
        return ((1, -1), (1, 1))
    if dr == -1 and dc == 0:
        return ((-1, -1), (-1, 1))
    if dr == 0 and dc == 1:
        return ((-1, 1), (1, 1))
    if dr == 0 and dc == -1:
        return ((-1, -1), (1, -1))
    raise AssertionError("primary rays must be cardinal")


def _trace_ray(
    origin: tuple[int, int],
    step: tuple[int, int],
    shape: tuple[int, int],
    prisms: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    r, c = origin
    dr, dc = step
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells, None
        cell = (r, c)
        if cell in source_cells:
            return cells, None
        if cell in prisms:
            return cells, cell
        cells.append(cell)


def _simulate(
    shape: tuple[int, int],
    sources: list[dict[str, object]],
    prisms: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], set[int]], dict[str, int]]:
    source_cells = {tuple(src["pos"]) for src in sources}
    visits: dict[tuple[int, int], set[int]] = {}
    stats = {"direct_sources": 0, "split_sources": 0, "blocked_branches": 0}

    for src in sources:
        pos = tuple(src["pos"])
        color = int(src["color"])
        step = tuple(src["dir"])

        main_cells, hit_prism = _trace_ray(pos, step, shape, prisms, source_cells)
        for cell in main_cells:
            visits.setdefault(cell, set()).add(color)

        if hit_prism is None:
            stats["direct_sources"] += 1
            continue

        stats["split_sources"] += 1
        for diag_step in _split_directions(*step):
            diag_cells, second_prism = _trace_ray(hit_prism, diag_step, shape, prisms, source_cells)
            for cell in diag_cells:
                visits.setdefault(cell, set()).add(color)
            if second_prism is not None:
                stats["blocked_branches"] += 1

    stats["yellow_cells"] = sum(len(colors) >= 2 for colors in visits.values())
    stats["painted_cells"] = len(visits)
    return visits, stats


def _parse_grid(grid: np.ndarray) -> tuple[int, list[dict[str, object]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid must be large enough for sources and prisms"

    bg = _background_color(g)
    prisms = {tuple(map(int, coord)) for coord in np.argwhere(g == PRISM)}
    for r, c in prisms:
        assert not _is_edge(g.shape, r, c), "prisms must be interior cells"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            assert (nr, nc) not in prisms, "prisms must remain visually separated"

    sources: list[dict[str, object]] = []
    seen_colors: set[int] = set()
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, PRISM):
                continue
            assert _is_edge(g.shape, r, c), "all non-background non-prism pixels must be edge sources"
            step = _edge_direction(g.shape, r, c)
            assert color not in seen_colors, "source colors must be distinct"
            seen_colors.add(color)
            sources.append({"pos": (r, c), "dir": step, "color": color})

    assert sources, "expected at least one signal source"
    return bg, sources, prisms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-launched rays. A primary ray stops at a gray prism and splits once into
    two forward diagonals. Cells visited by multiple source colors become yellow.
    """
    g = np.asarray(grid, dtype=int)
    bg, sources, prisms = _parse_grid(g)
    visits, _stats = _simulate(g.shape, sources, prisms)

    out = g.copy()
    source_cells = {tuple(src["pos"]) for src in sources}
    for (r, c), colors in visits.items():
        if (r, c) in source_cells or (r, c) in prisms:
            continue
        out[r, c] = YELLOW if len(colors) >= 2 else next(iter(colors))
    out[g == PRISM] = PRISM
    for src in sources:
        r, c = tuple(src["pos"])
        out[r, c] = int(src["color"])
    assert np.array_equal(out[g == PRISM], g[g == PRISM])
    return out


def _source_cell(side: str, index: int, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return (0, index)
    if side == "bottom":
        return (h - 1, index)
    if side == "left":
        return (index, 0)
    if side == "right":
        return (index, w - 1)
    raise AssertionError("unknown side")


def _edge_candidates(side: str, h: int, w: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _line_cells_from_source(pos: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = step
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _diagonal_cells_from_prism(
    prism: tuple[int, int],
    step: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = prism
    dr, dc = step
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _pick_source_slot(
    h: int,
    w: int,
    used: set[tuple[int, int]],
    allowed_sides: Iterable[str],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for side in allowed_sides:
        for idx in _edge_candidates(side, h, w):
            pos = _source_cell(side, idx, h, w)
            if pos in used:
                continue
            candidates.append((pos, _edge_direction((h, w), *pos)))
    return random.choice(candidates) if candidates else None


def _choose_prism_for_source(
    pos: tuple[int, int],
    step: tuple[int, int],
    shape: tuple[int, int],
    prisms: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    line = _line_cells_from_source(pos, step, shape)
    candidates = []
    for i, cell in enumerate(line):
        forward_room = len(line) - i - 1
        if i < 1 or forward_room < 1:
            continue
        if cell in prisms or cell in forbidden:
            continue
        candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _add_crossing_pair(
    h: int,
    w: int,
    used_sources: set[tuple[int, int]],
    protected: set[tuple[int, int]],
    planned: list[dict[str, object]],
) -> bool:
    cross_r = random.randint(2, h - 3)
    cross_c = random.randint(2, w - 3)
    vertical_side = random.choice(("top", "bottom"))
    horizontal_side = random.choice(("left", "right"))
    first = _source_cell(vertical_side, cross_c, h, w)
    second = _source_cell(horizontal_side, cross_r, h, w)
    if first in used_sources or second in used_sources:
        return False

    for pos in (first, second):
        step = _edge_direction((h, w), *pos)
        planned.append({"pos": pos, "dir": step, "needs_prism": False})
        used_sources.add(pos)
        protected.update(_line_cells_from_source(pos, step, (h, w)))
    return True


def _assign_colors(bg: int, count: int) -> list[int]:
    pool = [c for c in PALETTE if c != bg]
    return random.sample(pool, count)


def _build_grid(
    h: int,
    w: int,
    bg: int,
    sources: list[dict[str, object]],
    prisms: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for r, c in prisms:
        grid[r, c] = PRISM
    for src in sources:
        r, c = tuple(src["pos"])
        grid[r, c] = int(src["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)
    n_sources = random.randint(*SOURCE_COUNT_RANGE)
    target_cross = random.choice([False, True])
    target_block = random.choice([False, True])

    used_sources: set[tuple[int, int]] = set()
    prisms: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    planned: list[dict[str, object]] = []

    if target_cross:
        if not _add_crossing_pair(h, w, used_sources, protected, planned):
            return None
    else:
        orientation = random.choice(["vertical", "horizontal"])
        sides = ("top", "bottom") if orientation == "vertical" else ("left", "right")
        slot = _pick_source_slot(h, w, used_sources, sides)
        if slot is None:
            return None
        pos, step = slot
        planned.append({"pos": pos, "dir": step, "needs_prism": False})
        used_sources.add(pos)
        protected.update(_line_cells_from_source(pos, step, (h, w)))

    while len(planned) < n_sources:
        needs_prism = len([p for p in planned if p["needs_prism"]]) == 0 or random.random() < 0.6
        if target_cross:
            allowed = SIDES
        else:
            if random.random() < 0.75:
                allowed = ("top", "bottom") if planned[0]["dir"][0] != 0 else ("left", "right")
            else:
                allowed = SIDES
        slot = _pick_source_slot(h, w, used_sources, allowed)
        if slot is None:
            return None
        pos, step = slot
        plan = {"pos": pos, "dir": step, "needs_prism": needs_prism}
        if needs_prism:
            prism = _choose_prism_for_source(pos, step, (h, w), prisms, protected)
            if prism is None:
                continue
            plan["prism"] = prism
            prisms.add(prism)
        else:
            protected.update(_line_cells_from_source(pos, step, (h, w)))

        planned.append(plan)
        used_sources.add(pos)

    if not any(plan["needs_prism"] for plan in planned):
        return None
    if not any(not plan["needs_prism"] for plan in planned):
        return None

    split_plans = [plan for plan in planned if plan["needs_prism"]]
    if target_block and split_plans:
        random.shuffle(split_plans)
        blocker_added = False
        for plan in split_plans:
            prism = tuple(plan["prism"])
            branch_steps = list(_split_directions(*tuple(plan["dir"])))
            random.shuffle(branch_steps)
            for diag in branch_steps:
                cells = _diagonal_cells_from_prism(prism, diag, (h, w))
                candidates = [cell for i, cell in enumerate(cells) if i >= 1 and cell not in prisms and cell not in protected]
                if candidates:
                    prisms.add(random.choice(candidates))
                    blocker_added = True
                    break
            if blocker_added:
                break
        if not blocker_added:
            return None

    colors = _assign_colors(bg, len(planned))
    sources = []
    for color, plan in zip(colors, planned):
        sources.append({"pos": tuple(plan["pos"]), "dir": tuple(plan["dir"]), "color": color})

    grid = _build_grid(h, w, bg, sources, prisms)
    visits, stats = _simulate((h, w), sources, prisms)
    if stats["split_sources"] == 0 or stats["direct_sources"] == 0:
        return None
    if bool(stats["yellow_cells"]) != target_cross:
        return None
    if target_block and stats["blocked_branches"] == 0:
        return None
    if stats["painted_cells"] == 0 or stats["painted_cells"] / grid.size > MAX_FILL_FRACTION:
        return None
    if _background_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if bool(np.any(out == YELLOW)) != target_cross:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATE_ATTEMPTS} attempts")
