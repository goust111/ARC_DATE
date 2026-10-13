from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_ATTEMPTS = 500

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _is_corner(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _initial_dir(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    assert not _is_corner(shape, r, c), "corner light sources are ambiguous"
    if r == 0:
        return DIR_DOWN
    if r == h - 1:
        return DIR_UP
    if c == 0:
        return DIR_RIGHT
    if c == w - 1:
        return DIR_LEFT
    raise AssertionError("source must lie on the grid edge")


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _steps_to_edge(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    if dc == 1:
        return w - 1 - c
    raise AssertionError("invalid direction")


def _ray_cells_to_edge(pos: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        out.append((r, c))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT):
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _extract_walls(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]]]:
    walls: list[dict[str, object]] = []
    wall_lookup: dict[tuple[int, int], dict[str, object]] = {}
    marker_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == MARKER)}

    for comp in _components(grid == WALL):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(comp) >= 2, "mirror wall must have length >= 2"
        if len(rows) == 1:
            row = next(iter(rows))
            ordered = sorted(comp, key=lambda p: p[1])
            expected = list(range(ordered[0][1], ordered[-1][1] + 1))
            assert [c for _, c in ordered] == expected, "horizontal wall must be contiguous"
            endpoints = [ordered[0], ordered[-1]]
            orientation = "horizontal"
        else:
            assert len(cols) == 1, "mirror wall must be straight"
            col = next(iter(cols))
            ordered = sorted(comp, key=lambda p: p[0])
            expected = list(range(ordered[0][0], ordered[-1][0] + 1))
            assert [r for r, _ in ordered] == expected, "vertical wall must be contiguous"
            endpoints = [ordered[0], ordered[-1]]
            orientation = "vertical"

        hits: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        for er, ec in endpoints:
            for mr in (-1, 1):
                for mc in (-1, 1):
                    marker = (er + mr, ec + mc)
                    if marker in marker_cells:
                        hits.append(((er, ec), (mr, mc), marker))

        assert len(hits) == 1, "each wall must have exactly one diagonal maroon marker"
        endpoint, offset, marker = hits[0]
        wall = {
            "cells": tuple(ordered),
            "orientation": orientation,
            "endpoint": endpoint,
            "marker_offset": offset,
            "marker": marker,
        }
        walls.append(wall)
        for cell in ordered:
            wall_lookup[cell] = wall

    assert len(marker_cells) == len(walls), "every maroon marker must label exactly one wall"
    return walls, wall_lookup


def _reflect(wall: dict[str, object], direction: tuple[int, int]) -> tuple[int, int]:
    mr, mc = wall["marker_offset"]  # type: ignore[index]
    orientation = wall["orientation"]
    if orientation == "horizontal":
        expected = (-int(mr), 0)
        assert direction == expected, "horizontal wall hit from the wrong side"
        return (0, int(mc))
    assert orientation == "vertical"
    expected = (0, -int(mc))
    assert direction == expected, "vertical wall hit from the wrong side"
    return (int(mr), 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-emitted light rays through marked mirror walls and paint every traversed background pixel with the emitting source color.

    Input structure:
    - The background is the most common color. Grey pixels form several separate straight horizontal or vertical wall segments.
    - Single colored light sources sit on the outer border, one pixel each, with distinct colors. Their edge positions encode the initial ray directions: top->down, bottom->up, left->right, right->left.
    - Each grey wall has exactly one maroon marker diagonally adjacent to one endpoint. That diagonal marker encodes which side a ray may enter from and which orthogonal direction it turns toward.

    Rule:
    - Process the sources in left-to-right, then top-to-bottom order (column-major scan of source positions).
    - Advance each ray one pixel at a time through original background cells. When the next cell is a labelled mirror wall, do not paint the wall; instead turn the ray 90 degrees according to the wall marker and continue from the current cell.
    - Maroon marker pixels are annotations rather than blockers: a ray may pass through them, but they keep their original maroon color.
    - Stop when the next step would leave the grid or would enter another source pixel.
    - Recolor every visited background pixel with that source color. Later sources overwrite earlier rays on crossed cells, while grey walls, maroon markers, and source pixels remain unchanged.

    Key insight:
    - The puzzle has two independent spatial codes: border placement gives each source its launch direction, and the maroon marker's diagonal position relative to a wall endpoint gives the wall's allowed incoming side and outgoing turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    _walls, wall_lookup = _extract_walls(g)

    sources: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            value = int(g[r, c])
            if value in (bg, WALL, MARKER):
                continue
            sources.append((r, c))

    assert sources, "expected at least one light source"
    assert len({int(g[r, c]) for r, c in sources}) == len(sources), "source colors must be distinct"
    source_set = set(sources)

    out = g.copy()
    for src in sorted(sources, key=lambda p: (p[1], p[0])):
        color = int(g[src])
        dr, dc = _initial_dir(g.shape, *src)
        r, c = src
        seen_states: set[tuple[int, int, int, int]] = set()
        while True:
            state = (int(r), int(c), int(dr), int(dc))
            assert state not in seen_states, "ray path must not loop"
            seen_states.add(state)

            nr, nc = int(r + dr), int(c + dc)
            if not _in_bounds(g.shape, nr, nc):
                break

            if (nr, nc) in source_set:
                break

            value = int(g[nr, nc])
            if value == bg:
                out[nr, nc] = color
                r, c = nr, nc
                continue

            if value == WALL:
                dr, dc = _reflect(wall_lookup[(nr, nc)], (dr, dc))
                continue

            assert value == MARKER, "unexpected non-background cell on a ray path"
            r, c = nr, nc
            continue

    return out


def _guard_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _reserve_structure(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    guard: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if any(cell in guard for cell in cells):
        return False
    occupied.update(cells)
    guard.update(_guard_cells(cells, shape))
    return True


def _paint_path_cells(
    path_cells: set[tuple[int, int]],
    path_orients: dict[tuple[int, int], set[str]],
    cells: list[tuple[int, int]],
    direction: tuple[int, int],
) -> None:
    orient = "h" if direction[1] else "v"
    for cell in cells:
        path_cells.add(cell)
        path_orients.setdefault(cell, set()).add(orient)


def _turn_object(
    turn_cell: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    dr, dc = incoming
    tr, tc = turn_cell
    wall_hit = (tr + dr, tc + dc)

    if dr != 0:
        if outgoing == DIR_LEFT:
            wall = [(wall_hit[0], wall_hit[1] + i) for i in range(length)]
        else:
            assert outgoing == DIR_RIGHT
            wall = [(wall_hit[0], wall_hit[1] - i) for i in range(length)]
        marker = (wall_hit[0] - dr, wall_hit[1] + outgoing[1])
        return wall, marker

    if outgoing == DIR_UP:
        wall = [(wall_hit[0] + i, wall_hit[1]) for i in range(length)]
    else:
        assert outgoing == DIR_DOWN
        wall = [(wall_hit[0] - i, wall_hit[1]) for i in range(length)]
    marker = (wall_hit[0] + outgoing[0], wall_hit[1] - dc)
    return wall, marker


def _straight_finish_is_clear(
    pos: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_paths: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> bool:
    cells = _ray_cells_to_edge(pos, direction, shape)
    return bool(cells) and not any(cell in blocked_paths or cell in blockers for cell in cells)


def _candidate_turns(
    pos: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked_paths: set[tuple[int, int]],
    local_path: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    guard: set[tuple[int, int]],
) -> list[dict[str, object]]:
    max_run = _steps_to_edge(shape, pos, direction) - 2
    if max_run < 1:
        return []

    candidates: list[dict[str, object]] = []
    for outgoing in (_turn_left(direction), _turn_right(direction)):
        for run_len in range(1, max_run + 1):
            run_cells = [
                (pos[0] + direction[0] * step, pos[1] + direction[1] * step)
                for step in range(1, run_len + 1)
            ]
            if any(cell in blockers or cell in blocked_paths or cell in local_path for cell in run_cells):
                break

            turn_cell = run_cells[-1]
            for length in range(2, 7):
                wall_cells, marker = _turn_object(turn_cell, direction, outgoing, length)
                object_cells = set(wall_cells)
                object_cells.add(marker)
                if any(not _in_bounds(shape, r, c) for r, c in object_cells):
                    continue
                if marker[0] in (0, shape[0] - 1) or marker[1] in (0, shape[1] - 1):
                    continue
                if any(cell in guard for cell in object_cells):
                    continue
                if any(cell in blocked_paths or cell in local_path for cell in object_cells):
                    continue
                wall_blockers = set(wall_cells)
                if not _straight_finish_is_clear(turn_cell, outgoing, shape, blocked_paths | local_path, blockers | wall_blockers):
                    continue

                candidates.append(
                    {
                        "run_cells": run_cells,
                        "turn_cell": turn_cell,
                        "outgoing": outgoing,
                        "wall_cells": wall_cells,
                        "marker": marker,
                    }
                )
    return candidates


def _plan_mirror_path(
    source: tuple[int, int],
    turns: int,
    shape: tuple[int, int],
    blocked_paths: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    guard: set[tuple[int, int]],
) -> dict[str, object] | None:
    direction = _initial_dir(shape, *source)
    pos = source
    local_path: set[tuple[int, int]] = set()
    segments: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    path_runs: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []

    for _ in range(turns):
        choices = _candidate_turns(
            pos,
            direction,
            shape,
            blocked_paths,
            local_path,
            blockers,
            guard,
        )
        if not choices:
            return None
        choice = random.choice(choices)
        run_cells = choice["run_cells"]  # type: ignore[index]
        wall_cells = choice["wall_cells"]  # type: ignore[index]
        marker = choice["marker"]  # type: ignore[index]
        outgoing = choice["outgoing"]  # type: ignore[index]
        turn_cell = choice["turn_cell"]  # type: ignore[index]

        local_path.update(run_cells)
        path_runs.append((run_cells, direction))
        segments.append((wall_cells, marker))
        pos = turn_cell
        direction = outgoing

    finish_cells = _ray_cells_to_edge(pos, direction, shape)
    if not finish_cells:
        return None
    if any(cell in blockers or cell in blocked_paths or cell in local_path for cell in finish_cells):
        return None
    local_path.update(finish_cells)
    path_runs.append((finish_cells, direction))

    return {
        "source": source,
        "segments": segments,
        "path_runs": path_runs,
        "path_cells": local_path,
    }


def _edge_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    random.shuffle(cells)
    return cells


def _add_pair_corridor(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    guard: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    path_orients: dict[tuple[int, int], set[str]],
    sources: list[tuple[int, int]],
) -> bool:
    h, w = shape
    options: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], tuple[int, int]]] = []

    for r in range(1, h - 1):
        left = (r, 0)
        right = (r, w - 1)
        corridor = [(r, c) for c in range(1, w - 1)]
        if (
            left not in guard
            and right not in guard
            and not any(cell in occupied or cell in path_cells for cell in corridor)
        ):
            options.append((left, right, corridor, DIR_RIGHT))

    for c in range(1, w - 1):
        top = (0, c)
        bottom = (h - 1, c)
        corridor = [(r, c) for r in range(1, h - 1)]
        if (
            top not in guard
            and bottom not in guard
            and not any(cell in occupied or cell in path_cells for cell in corridor)
        ):
            options.append((top, bottom, corridor, DIR_DOWN))

    if not options:
        return False

    start, end, corridor, direction = random.choice(options)
    if not _reserve_structure({start, end}, occupied, guard, shape):
        return False
    sources.extend([start, end])
    blockers.update({start, end})
    _paint_path_cells(path_cells, path_orients, corridor, direction)
    _paint_path_cells(path_cells, path_orients, corridor, (-direction[0], -direction[1]))
    return True


def _add_mirror_source(
    turns: int,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    guard: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    path_orients: dict[tuple[int, int], set[str]],
    sources: list[tuple[int, int]],
    walls: list[list[tuple[int, int]]],
    markers: list[tuple[int, int]],
) -> bool:
    for source in _edge_candidates(shape):
        if source in guard or source in path_cells:
            continue
        planned = _plan_mirror_path(source, turns, shape, path_cells, blockers, guard)
        if planned is None:
            continue

        obj_cells: set[tuple[int, int]] = set()
        wall_cells_all: set[tuple[int, int]] = set()
        for wall_cells, marker in planned["segments"]:  # type: ignore[index]
            obj_cells.update(wall_cells)
            obj_cells.add(marker)
            wall_cells_all.update(wall_cells)
        obj_cells.add(source)
        if not _reserve_structure(obj_cells, occupied, guard, shape):
            continue

        sources.append(source)
        blockers.add(source)
        blockers.update(wall_cells_all)
        for wall_cells, marker in planned["segments"]:  # type: ignore[index]
            walls.append(list(wall_cells))
            markers.append(marker)
        for cells, direction in planned["path_runs"]:  # type: ignore[index]
            _paint_path_cells(path_cells, path_orients, cells, direction)
        return True
    return False


def _add_crossing_source(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    guard: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    path_orients: dict[tuple[int, int], set[str]],
    sources: list[tuple[int, int]],
) -> bool:
    candidates = list(path_orients.items())
    random.shuffle(candidates)
    h, w = shape

    for cell, orients in candidates:
        r, c = cell
        if r in (0, h - 1) or c in (0, w - 1):
            continue

        if "h" in orients:
            top = (0, c)
            path = [(rr, c) for rr in range(1, h)]
            if top not in guard and top not in path_cells and not any(p in blockers for p in path):
                if _reserve_structure({top}, occupied, guard, shape):
                    sources.append(top)
                    blockers.add(top)
                    _paint_path_cells(path_cells, path_orients, path, DIR_DOWN)
                    return True
            bottom = (h - 1, c)
            path = [(rr, c) for rr in range(h - 2, -1, -1)]
            if bottom not in guard and bottom not in path_cells and not any(p in blockers for p in path):
                if _reserve_structure({bottom}, occupied, guard, shape):
                    sources.append(bottom)
                    blockers.add(bottom)
                    _paint_path_cells(path_cells, path_orients, path, DIR_UP)
                    return True

        if "v" in orients:
            left = (r, 0)
            path = [(r, cc) for cc in range(1, w)]
            if left not in guard and left not in path_cells and not any(p in blockers for p in path):
                if _reserve_structure({left}, occupied, guard, shape):
                    sources.append(left)
                    blockers.add(left)
                    _paint_path_cells(path_cells, path_orients, path, DIR_RIGHT)
                    return True
            right = (r, w - 1)
            path = [(r, cc) for cc in range(w - 2, -1, -1)]
            if right not in guard and right not in path_cells and not any(p in blockers for p in path):
                if _reserve_structure({right}, occupied, guard, shape):
                    sources.append(right)
                    blockers.add(right)
                    _paint_path_cells(path_cells, path_orients, path, DIR_LEFT)
                    return True

    return False


def _render_grid(
    shape: tuple[int, int],
    sources: list[tuple[int, int]],
    walls: list[list[tuple[int, int]]],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for wall_cells in walls:
        for r, c in wall_cells:
            grid[r, c] = WALL
    for r, c in markers:
        grid[r, c] = MARKER
    colors = random.sample(SOURCE_COLORS, k=len(sources))
    for source, color in zip(sources, colors):
        grid[source] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 26)
    w = random.randint(12, 26)
    shape = (h, w)

    occupied: set[tuple[int, int]] = set()
    guard: set[tuple[int, int]] = set()
    blockers: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    path_orients: dict[tuple[int, int], set[str]] = {}
    sources: list[tuple[int, int]] = []
    walls: list[list[tuple[int, int]]] = []
    markers: list[tuple[int, int]] = []

    if random.random() < 0.5:
        if not _add_pair_corridor(shape, occupied, guard, blockers, path_cells, path_orients, sources):
            return None

    mirror_path_count = random.randint(1, 3)
    for _ in range(mirror_path_count):
        turns = random.randint(1, 3)
        if not _add_mirror_source(turns, shape, occupied, guard, blockers, path_cells, path_orients, sources, walls, markers):
            return None

    while len(sources) < 2:
        if not _add_mirror_source(random.randint(1, 2), shape, occupied, guard, blockers, path_cells, path_orients, sources, walls, markers):
            return None

    if random.random() < 0.45:
        _add_crossing_source(shape, occupied, guard, blockers, path_cells, path_orients, sources)

    if len(sources) > len(SOURCE_COLORS):
        return None

    if len(walls) != len(markers):
        return None

    grid = _render_grid(shape, sources, walls, markers)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(np.unique(grid)) < 4:
        return None
    if not walls:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirror-light input")
