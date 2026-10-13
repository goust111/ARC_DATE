from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, in_bounds


_DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_BLOCKER_COLOR = 2

# Generator palette and geometry.
_BG_COLOR = 0
_TRAIL_PALETTE = (1, 3, 4, 6, 8, 9)
_HEIGHT_RANGE = (9, 19)
_WIDTH_RANGE = (9, 19)
_MAX_ATTEMPTS = 400


def _is_edge(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _diag_neighbors(cells: set[tuple[int, int]], cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in _DIAG_DIRS:
        nxt = (r + dr, c + dc)
        if nxt in cells:
            out.append(nxt)
    return out


def _ordered_path(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    assert cells, "expected a non-empty trail"

    neighbors = {cell: _diag_neighbors(cells, cell) for cell in cells}
    endpoints = [cell for cell, adj in neighbors.items() if len(adj) == 1]
    assert len(endpoints) == 2, "trail must have exactly two endpoints"
    assert all(_is_edge(mask.shape, cell) for cell in endpoints), "trail endpoints must lie on the border"
    assert all(len(adj) in (1, 2) for adj in neighbors.values()), "trail must be a simple path"

    start = min(endpoints)
    path = [start]
    prev = None
    cur = start
    while True:
        options = [nxt for nxt in neighbors[cur] if nxt != prev]
        if not options:
            break
        assert len(options) == 1, "trail ordering must be unique"
        prev, cur = cur, options[0]
        path.append(cur)

    assert len(path) == len(cells), "trail must be one connected component"
    return path


def _turn_blockers(path: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    blockers: list[tuple[int, int]] = []
    for idx in range(1, len(path) - 1):
        pr, pc = path[idx - 1]
        cr, cc = path[idx]
        nr, nc = path[idx + 1]
        d_in = (cr - pr, cc - pc)
        d_out = (nr - cr, nc - cc)
        if d_in == d_out:
            continue
        assert d_in[0] * d_out[0] + d_in[1] * d_out[1] == 0, "bends must be right-angle turns"
        blocker = (cr + d_in[0], cc + d_in[1])
        assert in_bounds(shape, blocker[0], blocker[1]), "a bend's hidden blocker must be in bounds"
        blockers.append(blocker)
    return blockers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the invisible blockers that forced a diagonal light trail to bend.

    Input structure:
    - The grid has one dominant background color and exactly one visible foreground color.
    - That foreground forms a single simple path whose consecutive cells touch only diagonally.
    - The path starts on one border cell, bends zero or more times, and ends on another border cell.

    Rule:
    - Reconstruct the trail in traversal order from one endpoint to the other.
    - Whenever the incoming diagonal direction and outgoing diagonal direction differ at a trail cell,
      treat that cell as a bend.
    - For each bend, mark the cell one more step ahead in the incoming direction with the fixed blocker color,
      and leave the original trail unchanged.

    Key insight:
    - A bend cell is not the blocker itself; the missing blocker sits in the diagonal cell the light would have
      entered if it had continued straight.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one visible trail color"
    trail_color = colors[0]

    trail_mask = g == trail_color
    path = _ordered_path(trail_mask)
    blockers = _turn_blockers(path, g.shape)

    out = g.copy()
    for r, c in blockers:
        out[r, c] = _BLOCKER_COLOR
    return out


def _perpendicular_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return ((-dr, dc), (dr, -dc))


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    r, c = start
    steps = 0
    while in_bounds(shape, r + dr, c + dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _segment_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    return [(r + step * dr, c + step * dc) for step in range(1, length + 1)]


def _touches_other_trail(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> bool:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = (r + dr, c + dc)
            if neighbor in occupied and neighbor not in allowed:
                return True
    return False


def _can_append_segment(
    segment: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    prev: tuple[int, int],
) -> bool:
    staged: set[tuple[int, int]] = set()
    last = prev
    for cell in segment:
        if cell in occupied or cell in blockers or cell in staged:
            return False
        if _touches_other_trail(cell, occupied | staged, {last}):
            return False
        staged.add(cell)
        last = cell
    return True


def _blocker_is_clear(
    blocker: tuple[int, int],
    bend: tuple[int, int],
    occupied: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> bool:
    if blocker in occupied or blocker in blockers:
        return False
    br, bc = blocker
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor = (br + dr, bc + dc)
            if neighbor in occupied and neighbor != bend:
                return False
    return True


def _build_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    bends_left: int,
    occupied: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
    path: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    if bends_left == 0:
        final_len = _steps_to_edge(shape, current, direction)
        if final_len < 1:
            return None
        final_segment = _segment_cells(current, direction, final_len)
        if not _can_append_segment(final_segment, occupied, blockers, current):
            return None
        return path + final_segment, blockers

    max_len = _steps_to_edge(shape, current, direction)
    if max_len < 2:
        return None

    bend_lengths = list(range(1, max_len))
    random.shuffle(bend_lengths)
    for bend_len in bend_lengths:
        segment = _segment_cells(current, direction, bend_len)
        if not _can_append_segment(segment, occupied, blockers, current):
            continue

        bend = segment[-1]
        blocker = (bend[0] + direction[0], bend[1] + direction[1])
        occupied_after = occupied | set(segment)
        if not _blocker_is_clear(blocker, bend, occupied_after, blockers):
            continue

        next_dirs = list(_perpendicular_dirs(direction))
        random.shuffle(next_dirs)
        for next_dir in next_dirs:
            if _steps_to_edge(shape, bend, next_dir) < 1:
                continue
            result = _build_path(
                shape=shape,
                current=bend,
                direction=next_dir,
                bends_left=bends_left - 1,
                occupied=occupied_after,
                blockers=blockers | {blocker},
                path=path + segment,
            )
            if result is not None:
                return result
    return None


def _start_side_and_direction(
    shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (0, random.randrange(1, w - 1)), random.choice(((1, -1), (1, 1)))
    if side == "bottom":
        return (h - 1, random.randrange(1, w - 1)), random.choice(((-1, -1), (-1, 1)))
    if side == "left":
        return (random.randrange(1, h - 1), 0), random.choice(((-1, 1), (1, 1)))
    return (random.randrange(1, h - 1), w - 1), random.choice(((-1, -1), (1, -1)))


def _choose_bend_count(shape: tuple[int, int]) -> int:
    min_dim = min(shape)
    if min_dim <= 11:
        choices = (1, 2, 3)
        weights = (2, 3, 2)
    elif min_dim <= 14:
        choices = (1, 2, 3, 4)
        weights = (1, 3, 3, 2)
    else:
        choices = (1, 2, 3, 4, 5)
        weights = (1, 2, 3, 2, 1)
    return random.choices(choices, weights=weights, k=1)[0]


def _render_output(
    shape: tuple[int, int],
    trail: list[tuple[int, int]],
    blockers: set[tuple[int, int]],
    trail_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, _BG_COLOR, dtype=int)
    out = np.full(shape, _BG_COLOR, dtype=int)
    for r, c in trail:
        inp[r, c] = trail_color
        out[r, c] = trail_color
    for r, c in blockers:
        out[r, c] = _BLOCKER_COLOR
    return inp, out


def _generate_example() -> tuple[np.ndarray, np.ndarray] | None:
    shape = (
        random.randint(_HEIGHT_RANGE[0], _HEIGHT_RANGE[1]),
        random.randint(_WIDTH_RANGE[0], _WIDTH_RANGE[1]),
    )
    bends = _choose_bend_count(shape)
    start, direction = _start_side_and_direction(shape)

    built = _build_path(
        shape=shape,
        current=start,
        direction=direction,
        bends_left=bends,
        occupied={start},
        blockers=set(),
        path=[start],
    )
    if built is None:
        return None

    trail, blockers = built
    if len(trail) <= bends + 2:
        return None
    if len(trail) >= (shape[0] * shape[1]) // 2:
        return None

    trail_set = set(trail)
    if blockers & trail_set:
        return None

    expected_blockers = _turn_blockers(trail, shape)
    if len(expected_blockers) != bends or set(expected_blockers) != blockers:
        return None

    if not _is_edge(shape, trail[0]) or not _is_edge(shape, trail[-1]):
        return None

    trail_color = random.choice(_TRAIL_PALETTE)
    inp, out = _render_output(shape, trail, blockers, trail_color)
    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        example = _generate_example()
        if example is not None:
            return example[0]
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
