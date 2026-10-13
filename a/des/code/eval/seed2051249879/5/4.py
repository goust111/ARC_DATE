from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _turn(direction: tuple[int, int], turn_sign: int) -> tuple[int, int]:
    idx = DIR_INDEX[direction]
    return DIRS[(idx + turn_sign) % 4]


def _block_top_left(endpoint: tuple[int, int], outward: tuple[int, int]) -> tuple[int, int]:
    r, c = endpoint
    dr, dc = outward
    if (dr, dc) == (-1, 0):
        return r - 3, c - 1
    if (dr, dc) == (1, 0):
        return r + 1, c - 1
    if (dr, dc) == (0, -1):
        return r - 1, c - 3
    assert (dr, dc) == (0, 1)
    return r - 1, c + 1


def _block_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}


def _read_block(grid: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    r0, c0 = top_left
    h, w = grid.shape
    assert 0 <= r0 <= h - 3 and 0 <= c0 <= w - 3
    return np.asarray(grid[r0 : r0 + 3, c0 : c0 + 3], dtype=int)


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = grid == GREY
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, grid.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _terminal_info(
    grid: np.ndarray,
    endpoint: tuple[int, int],
    interior_neighbor: tuple[int, int],
) -> tuple[str, tuple[int, int], np.ndarray]:
    er, ec = endpoint
    nr, nc = interior_neighbor
    outward = (er - nr, ec - nc)
    top_left = _block_top_left(endpoint, outward)
    block = _read_block(grid, top_left)
    assert not np.any(block == GREY)
    if np.all(block == BLACK):
        return "receiver", top_left, block

    colors = {int(v) for v in np.unique(block)}
    assert BLACK not in colors and GREY not in colors
    assert 2 <= len(colors) <= 3
    return "source", top_left, block


def _ordered_path(
    component: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    comp_set = set(component)
    neighbors = {
        cell: [(cell[0] + dr, cell[1] + dc) for dr, dc in DIRS if (cell[0] + dr, cell[1] + dc) in comp_set]
        for cell in comp_set
    }
    endpoints = [cell for cell, nbs in neighbors.items() if len(nbs) == 1]
    assert len(endpoints) == 2
    assert all(len(nbs) in (1, 2) for nbs in neighbors.values())
    return endpoints, neighbors


def _turn_rotated(pattern: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(pattern, dtype=int).copy()
    for idx in range(1, len(path) - 1):
        prev_dir = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        next_dir = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if prev_dir == next_dir:
            continue
        delta = (DIR_INDEX[next_dir] - DIR_INDEX[prev_dir]) % 4
        assert delta in (1, 3)
        out = np.rot90(out, -1 if delta == 1 else 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each 3x3 source pattern along its grey channel and rotate it by
    90 degrees at every turn: clockwise on right turns, counter-clockwise on
    left turns. The final rotated pattern fills the 3x3 black receiver area.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for component in _gray_components(g):
        endpoints, neighbors = _ordered_path(component)

        terminal_meta = []
        for endpoint in endpoints:
            terminal_meta.append((endpoint, * _terminal_info(g, endpoint, neighbors[endpoint][0])))

        sources = [(endpoint, top_left, block) for endpoint, kind, top_left, block in terminal_meta if kind == "source"]
        receivers = [(endpoint, top_left, block) for endpoint, kind, top_left, block in terminal_meta if kind == "receiver"]
        assert len(sources) == 1 and len(receivers) == 1

        source_endpoint, _, source_block = sources[0]
        receiver_endpoint, receiver_top_left, _ = receivers[0]

        path = [source_endpoint]
        prev = None
        curr = source_endpoint
        while curr != receiver_endpoint:
            next_cells = [cell for cell in neighbors[curr] if cell != prev]
            assert len(next_cells) == 1
            prev, curr = curr, next_cells[0]
            path.append(curr)
        assert len(path) == len(component)

        rotated = _turn_rotated(source_block, path)
        rr, rc = receiver_top_left
        out[rr : rr + 3, rc : rc + 3] = rotated

    return out


def _pattern_is_asymmetric(pattern: np.ndarray) -> bool:
    for k in (1, 2, 3):
        if np.array_equal(pattern, np.rot90(pattern, k)):
            return False
    return not np.array_equal(pattern, np.fliplr(pattern)) and not np.array_equal(pattern, np.flipud(pattern))


def _make_pattern() -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(2, 3)
        colors = random.sample(PALETTE, color_count)
        cells = random.choices(colors, k=9)
        if len(set(cells)) != color_count:
            continue
        pattern = np.array(cells, dtype=int).reshape(3, 3)
        if _pattern_is_asymmetric(pattern):
            return pattern
    raise RuntimeError("failed to build asymmetric 3x3 pattern")


def _mark_blocked(blocked: set[tuple[int, int]], cells: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    for r, c in cells:
        blocked.add((r, c))
        for nr, nc in _neighbors4(r, c, shape):
            blocked.add((nr, nc))


def _cell_ok(
    cell: tuple[int, int],
    blocked: set[tuple[int, int]],
    current_used: set[tuple[int, int]],
    shape: tuple[int, int],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c) or cell in blocked or cell in current_used:
        return False
    for nb in _neighbors4(r, c, shape):
        if nb in current_used and nb not in allowed_neighbors:
            return False
    return True


def _block_ok(
    block_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    current_used: set[tuple[int, int]],
    shape: tuple[int, int],
    allowed_external_neighbors: set[tuple[int, int]],
) -> bool:
    for cell in block_cells:
        r, c = cell
        if not _in_bounds(shape, r, c) or cell in blocked or cell in current_used:
            return False
        for nb in _neighbors4(r, c, shape):
            if nb in current_used and nb not in allowed_external_neighbors:
                return False
    return True


def _random_start(shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return random.randint(2, h - 4), random.randint(1, w - 2)
    if (dr, dc) == (1, 0):
        return random.randint(3, h - 3), random.randint(1, w - 2)
    if (dr, dc) == (0, -1):
        return random.randint(1, h - 2), random.randint(2, w - 4)
    assert (dr, dc) == (0, 1)
    return random.randint(1, h - 2), random.randint(3, w - 3)


def _build_path_candidate(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], np.ndarray, tuple[int, int]] | None:
    for _ in range(120):
        start_dir = random.choice(DIRS)
        start = _random_start(shape, start_dir)

        source_top_left = _block_top_left(start, (-start_dir[0], -start_dir[1]))
        source_cells = _block_cells(source_top_left)
        if not _block_ok(source_cells, blocked, set(), shape, set()):
            continue

        current_used = set(source_cells)
        source_touch = (start[0] - start_dir[0], start[1] - start_dir[1])
        if not _cell_ok(start, blocked, current_used, shape, {source_touch}):
            continue
        current_used.add(start)
        path = [start]

        turn_count = random.choice((0, 1, 2, 3, 4))
        turn_sequence = [random.choice((-1, 1)) for _ in range(turn_count)]
        direction = start_dir
        ok = True

        for seg_idx in range(turn_count + 1):
            seg_len = random.randint(2, 6)
            for _ in range(seg_len):
                prev = path[-1]
                nxt = (prev[0] + direction[0], prev[1] + direction[1])
                if not _cell_ok(nxt, blocked, current_used, shape, {prev}):
                    ok = False
                    break
                current_used.add(nxt)
                path.append(nxt)
            if not ok:
                break
            if seg_idx < turn_count:
                direction = _turn(direction, turn_sequence[seg_idx])

        if not ok:
            continue

        end = path[-1]
        receiver_top_left = _block_top_left(end, direction)
        receiver_cells = _block_cells(receiver_top_left)
        receiver_touch = (end[0] + direction[0], end[1] + direction[1])
        if not _block_ok(receiver_cells, blocked, current_used, shape, {end}):
            continue
        if receiver_touch not in receiver_cells:
            continue

        source_pattern = _make_pattern()
        receiver_pattern = np.asarray(source_pattern, dtype=int).copy()
        for turn_sign in turn_sequence:
            receiver_pattern = np.rot90(receiver_pattern, -1 if turn_sign == 1 else 1)

        return source_pattern, path, receiver_top_left, receiver_pattern, source_top_left
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    shape = (h, w)
    grid = np.full(shape, BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()

    max_paths = 1 + (min(h, w) >= 22) + (min(h, w) >= 26)
    path_count = random.randint(1, max_paths)
    expected = grid.copy()

    for _ in range(path_count):
        candidate = _build_path_candidate(shape, blocked)
        if candidate is None:
            return None

        source_pattern, path, receiver_top_left, receiver_pattern, source_top_left = candidate
        sr, sc = source_top_left
        rr, rc = receiver_top_left

        grid[sr : sr + 3, sc : sc + 3] = source_pattern
        expected[sr : sr + 3, sc : sc + 3] = source_pattern

        for r, c in path:
            grid[r, c] = GREY
            expected[r, c] = GREY

        expected[rr : rr + 3, rc : rc + 3] = receiver_pattern

        reserved = _block_cells(source_top_left) | set(path) | _block_cells(receiver_top_left)
        _mark_blocked(blocked, reserved, shape)

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
