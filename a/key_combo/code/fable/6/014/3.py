from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds


BLACK = int(Color.BLACK)
RED = int(Color.RED)
GREY = int(Color.GREY)
BEAD_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) not in {RED, GREY}]
MAX_GENERATION_ATTEMPTS = 800

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] - b[0]), int(a[1] - b[1])


def _edge_direction(start: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, start)
    h, w = map(int, shape)
    if r == 0:
        return DIRS["D"]
    if r == h - 1:
        return DIRS["U"]
    if c == 0:
        return DIRS["R"]
    if c == w - 1:
        return DIRS["L"]
    raise AssertionError("start marker must lie on an edge")


def _red_coords(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(np.asarray(grid, dtype=int) == RED)}


def _parse_elbows(grid: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]:
    elbows: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = {}
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[RED],
    )

    for comp in comps:
        coords = [tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)]
        if len(coords) != 3:
            continue

        neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cell in coords:
            linked = []
            for other in coords:
                if cell == other:
                    continue
                if abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1:
                    linked.append(other)
            neighbors[cell] = linked

        corners = [cell for cell, linked in neighbors.items() if len(linked) == 2]
        arms = [cell for cell, linked in neighbors.items() if len(linked) == 1]
        if len(corners) != 1 or len(arms) != 2:
            continue

        corner = corners[0]
        for arm in arms:
            other_arm = arms[0] if arms[1] == arm else arms[1]
            incoming_dir = _sub(corner, arm)
            outgoing_dir = _sub(other_arm, corner)
            elbows[arm] = (corner, other_arm, incoming_dir, outgoing_dir)

    return elbows


def _trace_pipe(input_grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    grid = np.asarray(input_grid, dtype=int)
    h, w = map(int, grid.shape)

    starts = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    edge_starts = [cell for cell in starts if cell[0] in {0, h - 1} or cell[1] in {0, w - 1}]
    assert len(edge_starts) == 1, "expected exactly one grey edge start marker"
    start = edge_starts[0]
    direction = _edge_direction(start, grid.shape)
    elbows = _parse_elbows(grid)

    current = start
    path: list[tuple[int, int]] = []
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        state = (current, direction)
        assert state not in seen_states, "pipe should not loop"
        seen_states.add(state)

        nxt = _add(current, direction)
        if not in_bounds(grid.shape, nxt[0], nxt[1]):
            break

        if int(grid[nxt]) == RED and nxt in elbows:
            corner, exit_arm, incoming_dir, outgoing_dir = elbows[nxt]
            assert direction == incoming_dir, "encountered elbow from the wrong side"
            path.extend([nxt, corner, exit_arm])
            current = exit_arm
            direction = outgoing_dir
            continue

        path.append(nxt)
        current = nxt

    return start, path


def _tail_cells(path: list[tuple[int, int]], fixed_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    tail: list[tuple[int, int]] = []
    for cell in reversed(path):
        if cell in fixed_cells:
            break
        tail.append(cell)
    tail.reverse()
    return tail


def _apply_slide(input_grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    fixed_cells = _red_coords(grid)
    movable = [cell for cell in path if cell not in fixed_cells]

    beads: list[int] = []
    occupied: list[tuple[int, int]] = []
    for cell in movable:
        value = int(out[cell])
        if value != BLACK:
            beads.append(value)
            occupied.append(cell)

    for cell in occupied:
        out[cell] = BLACK

    if not beads:
        return out

    tail = _tail_cells(path, fixed_cells)
    targets = tail[-len(beads) :] if len(tail) >= len(beads) else movable[-len(beads) :]
    for cell, color in zip(targets, beads):
        out[cell] = int(color)

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Trace the invisible orthogonal pipe from the grey edge marker and red elbows.

    Only beads that lie exactly on the traced pipe move. Their path order is
    preserved, and they compact against the pipe's exit end while the start
    marker and all elbows remain fixed.
    """
    _start, path = _trace_pipe(np.asarray(input_grid, dtype=int))
    return _apply_slide(np.asarray(input_grid, dtype=int), path)


def _partition_total(total: int, minimums: list[int], rng: np.random.Generator) -> list[int]:
    extra = int(total - sum(minimums))
    assert extra >= 0
    if extra == 0:
        return list(minimums)
    bonus = rng.multinomial(extra, [1.0 / len(minimums)] * len(minimums))
    return [int(base + add) for base, add in zip(minimums, bonus.tolist())]


def _build_path(
    start: tuple[int, int],
    segment_lengths: list[int],
    directions: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]]:
    current = start
    path: list[tuple[int, int]] = []
    corners: list[tuple[int, int]] = []
    for idx, (length, direction) in enumerate(zip(segment_lengths, directions)):
        for _ in range(int(length)):
            current = _add(current, direction)
            path.append(current)
        if idx < len(segment_lengths) - 1:
            corners.append(current)

    red_cells: set[tuple[int, int]] = set()
    for idx, corner in enumerate(corners):
        incoming = directions[idx]
        outgoing = directions[idx + 1]
        red_cells.add(corner)
        red_cells.add(_sub(corner, incoming))
        red_cells.add(_add(corner, outgoing))

    return path, corners, red_cells


def _canonical_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(12, 23))
        w = int(rng.integers(12, 23))
        turn_count = int(rng.integers(1, 5))
        directions = [DIRS["D"] if idx % 2 == 0 else DIRS["R"] for idx in range(turn_count + 1)]

        vertical_count = (turn_count + 2) // 2
        horizontal_count = (turn_count + 1) // 2

        if turn_count % 2 == 0:
            min_verticals = [2] + [4] * max(0, vertical_count - 2) + [2]
            min_horizontals = [4] * horizontal_count
        else:
            min_verticals = [2] + [4] * (vertical_count - 1)
            min_horizontals = [4] * max(0, horizontal_count - 1) + [2]

        if h - 1 < sum(min_verticals):
            continue
        if w - 2 < sum(min_horizontals):
            continue

        max_start_col = int(w - 1 - sum(min_horizontals))
        if max_start_col < 1:
            continue
        start_col = int(rng.integers(1, max_start_col + 1))
        start = (0, start_col)

        if turn_count % 2 == 0:
            total_vertical = h - 1
            max_horizontal = (w - 1) - start_col
            if max_horizontal < sum(min_horizontals):
                continue
            total_horizontal = int(rng.integers(sum(min_horizontals), max_horizontal + 1))
        else:
            total_horizontal = (w - 1) - start_col
            if total_horizontal < sum(min_horizontals):
                continue
            total_vertical = int(rng.integers(sum(min_verticals), h))

        vertical_lengths = _partition_total(total_vertical, min_verticals, rng)
        horizontal_lengths = _partition_total(total_horizontal, min_horizontals, rng)

        segment_lengths: list[int] = []
        v_idx = 0
        h_idx = 0
        for direction in directions:
            if direction == DIRS["D"]:
                segment_lengths.append(vertical_lengths[v_idx])
                v_idx += 1
            else:
                segment_lengths.append(horizontal_lengths[h_idx])
                h_idx += 1

        path, _corners, red_cells = _build_path(start, segment_lengths, directions)
        if any(not in_bounds((h, w), r, c) for r, c in path):
            continue

        fixed_path = [cell for cell in path if cell in red_cells]
        if len(fixed_path) != len(red_cells):
            continue

        movable = [cell for cell in path if cell not in red_cells]
        tail = _tail_cells(path, red_cells)
        if len(tail) < 2 or len(movable) < 4:
            continue

        max_beads = min(len(tail), len(movable) // 2, 6)
        if max_beads < 1:
            continue
        bead_count = int(rng.integers(1, max_beads + 1))

        target_suffix = set(tail[-bead_count:])
        source_candidates = [cell for cell in movable if cell not in target_suffix]
        if len(source_candidates) < bead_count:
            continue

        path_index = {cell: idx for idx, cell in enumerate(path)}
        source_candidates.sort(key=lambda cell: path_index[cell])
        choose_idx = np.sort(rng.choice(len(source_candidates), size=bead_count, replace=False))
        sources = [source_candidates[int(idx)] for idx in choose_idx.tolist()]

        grid = np.full((h, w), BLACK, dtype=int)
        expected = np.full((h, w), BLACK, dtype=int)
        grid[start] = GREY
        expected[start] = GREY

        for cell in red_cells:
            grid[cell] = RED
            expected[cell] = RED

        bead_colors = rng.choice(np.array(BEAD_COLORS, dtype=int), size=bead_count, replace=True).tolist()
        for cell, color in zip(sources, bead_colors):
            grid[cell] = int(color)
        for cell, color in zip(tail[-bead_count:], bead_colors):
            expected[cell] = int(color)

        unused = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) != start and (r, c) not in red_cells and (r, c) not in path
        ]
        rng.shuffle(unused)
        off_path_count = int(min(len(unused), rng.integers(2, 8)))
        for cell in unused[:off_path_count]:
            color = int(rng.choice(np.array(BEAD_COLORS, dtype=int)))
            grid[cell] = color
            expected[cell] = color

        return grid, expected

    raise AssertionError("failed to construct a canonical case")


def _apply_transform(grid: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(np.asarray(grid, dtype=int), k=int(code % 4))
    if code >= 4:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    grid, expected = _canonical_case(rng)
    transform_code = int(rng.integers(0, 8))
    return _apply_transform(grid, transform_code), _apply_transform(expected, transform_code)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, expected = _generate_case(rng)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise AssertionError("failed to generate a self-consistent input")
