import random

import numpy as np


BG_COLOR = 0
STAMP_COLOR = 9
FRAME_COLOR = 5
RAIL_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: index for index, direction in enumerate(DIRS)}
TURN_ACTIONS = ["L", "S", "R"]
MAX_GENERATION_ATTEMPTS = 1000


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _neighbors4_unbounded(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _rotate_clockwise(mask: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    return np.rot90(mask, -turns)


def _extract_stamp(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    comps = _components(grid == STAMP_COLOR)
    assert len(comps) == 1, "expected one maroon stamp"
    mask, _ = _coords_to_mask(comps[0])
    return mask, set(comps[0])


def _extract_frame(grid: np.ndarray) -> dict:
    for comp in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        ok = True
        for c in range(c0, c1 + 1):
            ok &= int(grid[r0, c]) == FRAME_COLOR
            ok &= int(grid[r1, c]) == FRAME_COLOR
        for r in range(r0, r1 + 1):
            ok &= int(grid[r, c0]) == FRAME_COLOR
            ok &= int(grid[r, c1]) == FRAME_COLOR
        if ok:
            return {
                "bbox": (r0, c0, r1, c1),
                "cells": set(comp),
                "inner_shape": (h - 2, w - 2),
            }
    raise AssertionError("expected one grey rectangular frame")


def _extract_path(grid: np.ndarray) -> list[tuple[int, int]]:
    rail_mask = (grid != BG_COLOR) & (grid != STAMP_COLOR) & (grid != FRAME_COLOR)
    comps = _components(rail_mask)
    assert len(comps) == 1, "expected one rail path"
    cells = set(comps[0])
    h, w = grid.shape
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in cells:
        neighbor_map[cell] = [nb for nb in _neighbors4(cell[0], cell[1], h, w) if nb in cells]
        assert 1 <= len(neighbor_map[cell]) <= 2, "rail path must be a simple chain"

    endpoints = [cell for cell, nbs in neighbor_map.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "rail path must have exactly two endpoints"

    stamp_cells = set(map(tuple, np.argwhere(grid == STAMP_COLOR)))
    start_candidates = [
        cell
        for cell in endpoints
        if any((cell[0] + dr, cell[1] + dc) in stamp_cells for dr, dc in DIRS)
    ]
    assert len(start_candidates) == 1, "exactly one endpoint must touch the stamp"

    path = [start_candidates[0]]
    prev = None
    cur = path[0]
    while True:
        next_cells = [nb for nb in neighbor_map[cur] if nb != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "rail path cannot branch"
        prev, cur = cur, next_cells[0]
        path.append(cur)
    return path


def _rotation_from_path(path: list[tuple[int, int]]) -> int:
    steps = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = (r1 - r0, c1 - c0)
        assert step in DIR_TO_INDEX, "rail steps must be cardinal"
        steps.append(step)

    turns = 0
    for prev_step, next_step in zip(steps, steps[1:]):
        delta = (DIR_TO_INDEX[next_step] - DIR_TO_INDEX[prev_step]) % 4
        assert delta in (0, 1, 3), "rail path cannot reverse inside the chain"
        if delta == 1:
            turns += 1
        elif delta == 3:
            turns -= 1
    return turns % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pass a maroon stamp along a colored rail path and stamp its turned copy into the grey frame.

    Input structure:
    - bg = Black.
    - One small connected Maroon asymmetric shape acts as the prototype stamp.
    - Colored one-cell-thick horizontal and vertical rail segments form a single 4-connected simple path starting next to the stamp.
    - A Grey hollow rectangle sits next to the far rail endpoint; its inner cavity matches the stamp bbox.

    Rule:
    - Follow the rail path from the endpoint that touches the stamp.
    - Each right turn contributes one clockwise quarter-turn, each left turn contributes one counterclockwise quarter-turn, and straight continuation contributes nothing.
    - Accumulate those turns, rotate the stamp mask by the net amount, and paint it inside the grey frame.
    - Use the color of the last rail segment for the painted stamp while leaving the original stamp, rails, and frame unchanged.

    Key insight:
    - The rail geometry encodes a sequence of turns, so solving the task means tracing one simple path, converting local turn direction into rotation, and then recoloring the rotated stamp at the destination.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    stamp_mask, _ = _extract_stamp(g)
    frame = _extract_frame(g)
    path = _extract_path(g)
    turns = _rotation_from_path(path)
    painted = _rotate_clockwise(stamp_mask, turns)

    r0, c0, r1, c1 = frame["bbox"]
    assert painted.shape == frame["inner_shape"], "frame must fit the rotated stamp"

    out = g.copy()
    last_color = int(g[path[-1]])
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    inner[painted] = last_color
    return out


def _turn_dir(dir_index: int, action: str) -> int:
    if action == "L":
        return (dir_index - 1) % 4
    if action == "R":
        return (dir_index + 1) % 4
    return dir_index


def _can_extend_path(candidate: tuple[int, int], prev: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if candidate in occupied:
        return False
    touching = {cell for cell in _neighbors4_unbounded(candidate) if cell in occupied}
    return touching == {prev}


def _random_stamp_mask() -> np.ndarray:
    for _ in range(300):
        size = random.choice([3, 4])
        target_area = random.randint(size + 1, size * size - 2)
        start = (random.randrange(size), random.randrange(size))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for cell in cells:
                for nr, nc in _neighbors4_unbounded(cell):
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        if (
            int(coords[:, 0].min()) != 0
            or int(coords[:, 0].max()) != size - 1
            or int(coords[:, 1].min()) != 0
            or int(coords[:, 1].max()) != size - 1
        ):
            continue

        signatures = {tuple(np.rot90(mask, k).ravel().tolist()) for k in range(4)}
        if len(signatures) != 4:
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric stamp")


def _build_path() -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]], list[int], list[str]] | None:
    for _ in range(400):
        segment_count = random.randint(2, 5)
        lengths = [random.randint(2, 4) for _ in range(segment_count)]
        actions = [random.choice(TURN_ACTIONS) for _ in range(segment_count - 1)]

        direction = random.randrange(4)
        directions = [direction]
        path = [(0, 0)]
        segments = [[(0, 0)]]
        occupied = {(0, 0)}
        cur = (0, 0)
        ok = True

        for index in range(segment_count):
            if index > 0:
                segments.append([])
            dr, dc = DIRS[direction]
            steps = lengths[index] - 1 if index == 0 else lengths[index]
            for _step in range(steps):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not _can_extend_path(nxt, cur, occupied):
                    ok = False
                    break
                occupied.add(nxt)
                path.append(nxt)
                segments[-1].append(nxt)
                cur = nxt
            if not ok:
                break
            if index + 1 < segment_count:
                direction = _turn_dir(direction, actions[index])
                directions.append(direction)

        if ok and all(segment for segment in segments):
            return path, segments, directions, actions
    return None


def _side_cells(mask: np.ndarray, dir_index: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    if dir_index == 0:
        return [(0, c) for c in range(w) if mask[0, c]]
    if dir_index == 1:
        return [(r, w - 1) for r in range(h) if mask[r, w - 1]]
    if dir_index == 2:
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    return [(r, 0) for r in range(h) if mask[r, 0]]


def _frame_border_cells(outer_size: int, dir_index: int) -> list[tuple[int, int]]:
    last = outer_size - 1
    if dir_index == 0:
        return [(last, c) for c in range(outer_size)]
    if dir_index == 1:
        return [(r, 0) for r in range(outer_size)]
    if dir_index == 2:
        return [(0, c) for c in range(outer_size)]
    return [(r, last) for r in range(outer_size)]


def _frame_cells_from_top_left(top_left: tuple[int, int], outer_size: int) -> set[tuple[int, int]]:
    r0, c0 = top_left
    last = outer_size - 1
    cells = set()
    for offset in range(outer_size):
        cells.add((r0, c0 + offset))
        cells.add((r0 + last, c0 + offset))
        cells.add((r0 + offset, c0))
        cells.add((r0 + offset, c0 + last))
    return cells


def _shift_cells(cells: set[tuple[int, int]], shift: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = shift
    return {(r + dr, c + dc) for r, c in cells}


def _touching_path_cells(shape_cells: set[tuple[int, int]], path_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    touches = set()
    for cell in path_cells:
        if any(nb in shape_cells for nb in _neighbors4_unbounded(cell)):
            touches.add(cell)
    return touches


def _place_stamp(mask: np.ndarray, path: list[tuple[int, int]], first_dir: int) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    start = path[0]
    dr, dc = DIRS[first_dir]
    connection_abs = (start[0] - dr, start[1] - dc)
    path_cells = set(path)
    candidates = _side_cells(mask, first_dir)
    random.shuffle(candidates)
    stamp_offsets = [tuple(map(int, cell)) for cell in np.argwhere(mask)]

    for sr, sc in candidates:
        top_left = (connection_abs[0] - sr, connection_abs[1] - sc)
        stamp_cells = {(top_left[0] + r, top_left[1] + c) for r, c in stamp_offsets}
        if stamp_cells & path_cells:
            continue
        if _touching_path_cells(stamp_cells, path_cells) != {start}:
            continue
        return top_left, stamp_cells
    return None


def _place_frame(
    stamp_size: int,
    stamp_cells: set[tuple[int, int]],
    path: list[tuple[int, int]],
    last_dir: int,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    outer_size = stamp_size + 2
    end = path[-1]
    dr, dc = DIRS[last_dir]
    border_abs = (end[0] + dr, end[1] + dc)
    path_cells = set(path)
    candidates = _frame_border_cells(outer_size, last_dir)
    random.shuffle(candidates)

    for fr, fc in candidates:
        top_left = (border_abs[0] - fr, border_abs[1] - fc)
        frame_cells = _frame_cells_from_top_left(top_left, outer_size)
        if frame_cells & path_cells:
            continue
        if frame_cells & stamp_cells:
            continue
        if _touching_path_cells(frame_cells, path_cells) != {end}:
            continue
        return top_left, frame_cells
    return None


def _build_expected(
    grid: np.ndarray,
    stamp_mask: np.ndarray,
    frame_top_left: tuple[int, int],
    turns: int,
    last_color: int,
) -> np.ndarray:
    out = grid.copy()
    rotated = _rotate_clockwise(stamp_mask, turns)
    r0, c0 = frame_top_left
    window = out[r0 + 1 : r0 + 1 + rotated.shape[0], c0 + 1 : c0 + 1 + rotated.shape[1]]
    window[rotated] = last_color
    return out


def _gen_attempt() -> np.ndarray | None:
    stamp_mask = _random_stamp_mask()
    path_bundle = _build_path()
    if path_bundle is None:
        return None
    path, segments, directions, actions = path_bundle

    stamp_placement = _place_stamp(stamp_mask, path, directions[0])
    if stamp_placement is None:
        return None
    stamp_top_left, stamp_cells = stamp_placement

    frame_placement = _place_frame(stamp_mask.shape[0], stamp_cells, path, directions[-1])
    if frame_placement is None:
        return None
    frame_top_left, frame_cells = frame_placement

    all_cells = set(path) | stamp_cells | frame_cells
    min_r, min_c, max_r, max_c = _bbox(all_cells)
    base_h = max_r - min_r + 1
    base_w = max_c - min_c + 1
    if base_h > 30 or base_w > 30:
        return None

    top = random.randint(0, 30 - base_h)
    bottom = random.randint(0, 30 - base_h - top)
    left = random.randint(0, 30 - base_w)
    right = random.randint(0, 30 - base_w - left)
    shift = (top - min_r, left - min_c)
    final_h = base_h + top + bottom
    final_w = base_w + left + right

    grid = np.full((final_h, final_w), BG_COLOR, dtype=int)
    stamp_offsets = [tuple(map(int, cell)) for cell in np.argwhere(stamp_mask)]
    shifted_stamp_top_left = (stamp_top_left[0] + shift[0], stamp_top_left[1] + shift[1])
    shifted_frame_top_left = (frame_top_left[0] + shift[0], frame_top_left[1] + shift[1])

    for r, c in stamp_offsets:
        grid[shifted_stamp_top_left[0] + r, shifted_stamp_top_left[1] + c] = STAMP_COLOR

    segment_colors = random.sample(RAIL_COLORS, len(segments))
    for color, segment in zip(segment_colors, segments):
        for r, c in segment:
            grid[r + shift[0], c + shift[1]] = color

    for r, c in _shift_cells(frame_cells, shift):
        grid[r, c] = FRAME_COLOR

    turns = sum(1 if action == "R" else -1 if action == "L" else 0 for action in actions) % 4
    expected = _build_expected(grid, stamp_mask, shifted_frame_top_left, turns, segment_colors[-1])

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rail-stamp puzzle")
