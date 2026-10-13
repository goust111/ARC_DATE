import numpy as np


BLACK = 0
GRAY = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

GEN_RETRIES = 400
PATH_RETRIES = 120
DISTRACTOR_RETRIES = 60


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _step(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return (cell[0] + direction[0], cell[1] + direction[1])


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_source(grid: np.ndarray) -> tuple[tuple[int, int], int, tuple[int, int]]:
    coords = np.argwhere((grid != BLACK) & (grid != GRAY))
    assert coords.shape[0] == 1, "expected one colored source pixel"
    r, c = map(int, coords[0])
    color = int(grid[r, c])
    h, w = grid.shape
    on_edges = [r == 0, r == h - 1, c == 0, c == w - 1]
    assert sum(on_edges) == 1, "source must lie on exactly one edge"
    if r == 0:
        direction = DOWN
    elif r == h - 1:
        direction = UP
    elif c == 0:
        direction = RIGHT
    else:
        direction = LEFT
    return (r, c), color, direction


def _extract_mirrors(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], tuple[int, int]]]:
    mirrors: list[dict] = []
    cell_to_hit: dict[tuple[int, int], tuple[int, int]] = {}
    for coords in _components4(grid == GRAY):
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        assert len(coords) in (2, 3, 4), "mirror length must be 2-4"
        if len(rows) == 1:
            row = rows[0]
            cells = [(row, c) for c in cols]
            assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal mirror must be contiguous"
            orientation = "h"
        else:
            assert len(cols) == 1, "mirror must be horizontal or vertical"
            col = cols[0]
            cells = [(r, col) for r in rows]
            assert rows == list(range(rows[0], rows[-1] + 1)), "vertical mirror must be contiguous"
            orientation = "v"
        mirror_index = len(mirrors)
        mirrors.append({"orientation": orientation, "cells": cells})
        for hit_index, cell in enumerate(cells):
            cell_to_hit[cell] = (mirror_index, hit_index)
    return mirrors, cell_to_hit


def _trace(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    source, source_color, direction = _find_source(g)
    mirrors, mirror_lookup = _extract_mirrors(g)

    out = g.copy()
    r, c = source
    dr, dc = direction
    out[r, c] = source_color
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        if (nr, nc) in mirror_lookup:
            mirror_index, hit_index = mirror_lookup[(nr, nc)]
            mirror = mirrors[mirror_index]
            length = len(mirror["cells"])
            if mirror["orientation"] == "h":
                assert dr != 0 and dc == 0, "horizontal mirror must be hit by a vertical ray"
                dr, dc = (0, -1) if 2 * hit_index < length else (0, 1)
            else:
                assert dc != 0 and dr == 0, "vertical mirror must be hit by a horizontal ray"
                dr, dc = (-1, 0) if 2 * hit_index < length else (1, 0)
            continue

        r, c = nr, nc
        out[r, c] = source_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the colored ray from the unique edge source, turning at gray mirrors.

    Reflection happens one cell before the mirror: the ray never enters the gray
    segment itself. For odd-length mirrors, the center cell is treated as part of
    the latter half via the split `2 * hit_index < length`.
    """
    return _trace(grid)


def _distance_to_boundary(cell: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    r, c = cell
    h, w = shape
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(start[0] + dr * step, start[1] + dc * step) for step in range(1, steps + 1)]


def _turn_options(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction in (UP, DOWN):
        return [LEFT, RIGHT]
    return [UP, DOWN]


def _half_offsets(length: int, branch_is_negative: bool) -> list[int]:
    return [idx for idx in range(length) if (2 * idx < length) == branch_is_negative]


def _mirror_cells_for_hit(
    hit: tuple[int, int],
    incoming: tuple[int, int],
    length: int,
    offset: int,
) -> list[tuple[int, int]] | None:
    hr, hc = hit
    if incoming in (UP, DOWN):
        start_c = hc - offset
        cells = [(hr, start_c + i) for i in range(length)]
    else:
        start_r = hr - offset
        cells = [(start_r + i, hc) for i in range(length)]
    return cells


def _touches_existing_mirror(candidate: list[tuple[int, int]], mirror_cells: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in mirror_cells:
                return True
    return False


def _path_expected_output(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    source_color: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in path_cells:
        if int(out[r, c]) != GRAY:
            out[r, c] = source_color
    return out


def _extend_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    used_path: set[tuple[int, int]],
    mirror_segments: list[list[tuple[int, int]]],
    mirror_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]] | None:
    distance = _distance_to_boundary(current, direction, shape)
    if turns_left == 0:
        tail = _segment_cells(current, direction, distance)
        for cell in tail:
            if cell in used_path or cell in mirror_cells:
                return None
        return used_path | set(tail), mirror_segments

    if distance < 2:
        return None

    step_choices = list(range(1, distance))
    rng.shuffle(step_choices)
    length_choices = [2, 3, 4]

    for steps in step_choices:
        segment = _segment_cells(current, direction, steps)
        if any(cell in used_path or cell in mirror_cells for cell in segment):
            continue
        next_path = used_path | set(segment)
        pivot = segment[-1]
        hit = _step(pivot, direction)

        turn_choices = _turn_options(direction)
        rng.shuffle(turn_choices)
        for out_dir in turn_choices:
            next_distance = _distance_to_boundary(pivot, out_dir, shape)
            if turns_left > 1 and next_distance < 2:
                continue

            rng.shuffle(length_choices)
            branch_is_negative = out_dir in (UP, LEFT)
            for length in length_choices:
                offsets = _half_offsets(length, branch_is_negative)
                rng.shuffle(offsets)
                for offset in offsets:
                    mirror = _mirror_cells_for_hit(hit, direction, length, offset)
                    assert mirror is not None
                    if not all(_in_bounds(cell, shape) for cell in mirror):
                        continue
                    if any(cell in next_path or cell in mirror_cells for cell in mirror):
                        continue
                    if _touches_existing_mirror(mirror, mirror_cells):
                        continue

                    result = _extend_path(
                        shape,
                        pivot,
                        out_dir,
                        turns_left - 1,
                        next_path,
                        mirror_segments + [mirror],
                        mirror_cells | set(mirror),
                        rng,
                    )
                    if result is not None:
                        return result

    return None


def _source_candidates(shape: tuple[int, int], edge: str) -> list[tuple[int, int]]:
    h, w = shape
    if edge == "top":
        return [(0, c) for c in range(1, w - 1)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _edge_direction(edge: str) -> tuple[int, int]:
    return {"top": DOWN, "bottom": UP, "left": RIGHT, "right": LEFT}[edge]


def _path_isolation_ok(candidate: list[tuple[int, int]], path_cells: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in path_cells:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in path_cells:
                return False
    return True


def _try_place_distractors(
    mirror_segments: list[list[tuple[int, int]]],
    path_cells: set[tuple[int, int]],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]]:
    mirrors = [list(seg) for seg in mirror_segments]
    mirror_cells = {cell for seg in mirrors for cell in seg}
    distractor_target = int(rng.integers(0, 4))

    for _ in range(distractor_target):
        placed = False
        for _ in range(DISTRACTOR_RETRIES):
            length = int(rng.integers(2, 5))
            horizontal = bool(rng.integers(0, 2))
            if horizontal:
                row = int(rng.integers(1, shape[0] - 1))
                col0 = int(rng.integers(0, shape[1] - length + 1))
                candidate = [(row, col0 + i) for i in range(length)]
            else:
                col = int(rng.integers(1, shape[1] - 1))
                row0 = int(rng.integers(0, shape[0] - length + 1))
                candidate = [(row0 + i, col) for i in range(length)]

            if any(cell in mirror_cells for cell in candidate):
                continue
            if _touches_existing_mirror(candidate, mirror_cells):
                continue
            if not _path_isolation_ok(candidate, path_cells):
                continue

            mirrors.append(candidate)
            mirror_cells |= set(candidate)
            placed = True
            break
        if not placed:
            break

    return mirrors


def _build_grid(
    shape: tuple[int, int],
    source: tuple[int, int],
    source_color: int,
    mirror_segments: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for segment in mirror_segments:
        for r, c in segment:
            grid[r, c] = GRAY
    grid[source] = source_color
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 21))
    w = int(rng.integers(12, 21))
    shape = (h, w)

    edge = str(rng.choice(["top", "bottom", "left", "right"]))
    source_options = _source_candidates(shape, edge)
    if not source_options:
        return None
    source = source_options[int(rng.integers(0, len(source_options)))]
    source_color = int(rng.choice(SOURCE_COLORS))
    direction = _edge_direction(edge)

    turn_count = int(rng.choice([1, 2, 3, 4]))
    for _ in range(PATH_RETRIES):
        result = _extend_path(
            shape,
            source,
            direction,
            turn_count,
            {source},
            [],
            set(),
            rng,
        )
        if result is None:
            continue

        path_cells, mirror_segments = result
        mirror_segments = _try_place_distractors(mirror_segments, path_cells, shape, rng)
        grid = _build_grid(shape, source, source_color, mirror_segments)
        expected = _path_expected_output(grid, path_cells, source_color)

        try:
            actual = main(grid)
        except AssertionError:
            continue

        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(actual, grid):
            continue
        if len(path_cells) < max(h, w) // 2:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray puzzle")
