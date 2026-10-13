from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
CYAN = 8

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

GRID_MIN = 12
GRID_MAX = 24
MAX_GENERATION_ATTEMPTS = 800


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(cell: tuple[int, int]):
    r, c = cell
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _step(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction in (UP, DOWN):
        return LEFT, RIGHT
    return UP, DOWN


def _distance_to_boundary(shape: tuple[int, int], cell: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _red_green_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = (grid == RED) | (grid == GREEN)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _true_domino_maps(
    grid: np.ndarray,
) -> tuple[dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]]]:
    redirects: dict[tuple[int, int], tuple[int, int]] = {}
    stop_reds: set[tuple[int, int]] = set()
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for comp in _red_green_components(grid):
        if len(comp) != 2:
            continue
        colors = {int(grid[cell]) for cell in comp}
        if colors != {RED, GREEN}:
            continue
        red = next(cell for cell in comp if int(grid[cell]) == RED)
        green = next(cell for cell in comp if int(grid[cell]) == GREEN)
        redirects[green] = (green[0] - red[0], green[1] - red[1])
        stop_reds.add(red)
        pairs.add((red, green))
    return redirects, stop_reds, pairs


def _start_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    starts = np.argwhere(grid == CYAN)
    assert starts.shape[0] == 1, "expected exactly one cyan start pixel"
    sr, sc = map(int, starts[0])
    h, w = grid.shape
    on_border = sr in (0, h - 1) or sc in (0, w - 1)
    assert on_border, "cyan start must lie on the border"
    if sr == 0:
        return (sr, sc), DOWN
    if sr == h - 1:
        return (sr, sc), UP
    if sc == 0:
        return (sr, sc), RIGHT
    return (sr, sc), LEFT


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a cyan beam inward from the border, turning only at the green ends of true red-green dominoes.

    Input structure:
    - The grid has a black background and exactly one cyan start pixel on the border.
    - Some red/green cells form 4-connected two-cell dominoes; others are isolated singles or diagonal red-green distractors.
    - True dominoes are the only red-green components of size two under 4-neighbor connectivity.

    Rule:
    - Launch the beam from the cyan border pixel straight inward; every visited background cell becomes cyan.
    - If the beam enters the green cell of a true domino, it continues from that cell in the direction from the domino's red cell to its green cell.
    - If the beam enters the red cell of a true domino, it stops immediately.
    - Isolated red cells, isolated green cells, and diagonal red-green pairs do not affect the beam, and all original red/green cells keep their colors.

    Key insight:
    - Orthogonal red-green adjacency defines a polarized redirector: first identify genuine two-cell dominoes, then read each one as a direction vector from red to green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    redirects, stop_reds, _pairs = _true_domino_maps(g)
    pos, direction = _start_and_direction(g)
    out = g.copy()
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen_states, "beam entered a loop"
        seen_states.add(state)

        nxt = _step(pos, direction)
        if not _in_bounds(g.shape, nxt):
            break

        if nxt in redirects:
            pos = nxt
            direction = redirects[nxt]
            continue

        if nxt in stop_reds:
            break

        if int(out[nxt]) == BLACK:
            out[nxt] = CYAN
        pos = nxt

    return out


def _pair_cells_are_clean(
    first: tuple[int, int],
    second: tuple[int, int],
    color_cells: dict[tuple[int, int], int],
) -> bool:
    if first == second:
        return False
    if first in color_cells or second in color_cells:
        return False
    if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
        return False
    for cell, partner in ((first, second), (second, first)):
        for nbr in _neighbors4(cell):
            if nbr == partner:
                continue
            if nbr in color_cells:
                return False
    return True


def _single_cell_is_clean(cell: tuple[int, int], color_cells: dict[tuple[int, int], int]) -> bool:
    if cell in color_cells:
        return False
    return all(nbr not in color_cells for nbr in _neighbors4(cell))


def _segment_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    return [_step(start, direction, k) for k in range(1, length + 1)]


def _choose_start(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        return (0, random.randint(1, w - 2)), DOWN
    if edge == "bottom":
        return (h - 1, random.randint(1, w - 2)), UP
    if edge == "left":
        return (random.randint(1, h - 2), 0), RIGHT
    return (random.randint(1, h - 2), w - 1), LEFT


def _turn_candidates(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]]:
    max_steps = _distance_to_boundary(shape, pos, direction)
    candidates: list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for length in range(2, max_steps + 1):
        segment = _segment_cells(pos, direction, length)
        if any((not _in_bounds(shape, cell)) or cell in path_set or cell in color_cells for cell in segment):
            continue
        green = segment[-1]
        for next_dir in _perpendiculars(direction):
            red = _step(green, next_dir, -1)
            after = _step(green, next_dir)
            if not _in_bounds(shape, red) or not _in_bounds(shape, after):
                continue
            if red in path_set or red in color_cells or after in path_set or after in color_cells:
                continue
            if not _pair_cells_are_clean(red, green, color_cells):
                continue
            candidates.append((length, next_dir, green, red))
    return candidates


def _terminal_red_candidates(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    max_steps = _distance_to_boundary(shape, pos, direction)
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for length in range(2, max_steps + 1):
        segment = _segment_cells(pos, direction, length)
        if any((not _in_bounds(shape, cell)) or cell in path_set or cell in color_cells for cell in segment):
            continue
        red = segment[-1]
        for side in _perpendiculars(direction):
            green = _step(red, side)
            if not _in_bounds(shape, green):
                continue
            if green in path_set or green in color_cells:
                continue
            if not _pair_cells_are_clean(red, green, color_cells):
                continue
            candidates.append((length, red, green))
    return candidates


def _exit_segment(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    length = _distance_to_boundary(shape, pos, direction)
    if length < 2:
        return None
    segment = _segment_cells(pos, direction, length)
    if any(cell in path_set or cell in color_cells for cell in segment):
        return None
    return segment


def _off_path_single_candidates(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in path_set:
                continue
            if _single_cell_is_clean(cell, color_cells):
                cells.append(cell)
    return cells


def _path_single_candidates(
    traversed: list[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for cell in traversed[1:]:
        if cell in color_cells:
            continue
        if _single_cell_is_clean(cell, color_cells):
            cells.append(cell)
    return cells


def _off_path_domino_candidates(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            first = (r, c)
            if first in path_set or first in color_cells:
                continue
            for direction in (DOWN, RIGHT):
                second = _step(first, direction)
                if not _in_bounds(shape, second):
                    continue
                if second in path_set or second in color_cells:
                    continue
                if _pair_cells_are_clean(first, second, color_cells):
                    pairs.append((first, second))
    return pairs


def _diagonal_pair_candidates(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    color_cells: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            options = [((r, c), (r + 1, c + 1)), ((r + 1, c), (r, c + 1))]
            for first, second in options:
                if first in path_set or second in path_set:
                    continue
                if first in color_cells or second in color_cells:
                    continue
                if not _single_cell_is_clean(first, color_cells):
                    continue
                if not _single_cell_is_clean(second, color_cells):
                    continue
                pairs.append((first, second))
    return pairs


def _render_input(shape: tuple[int, int], start: tuple[int, int], color_cells: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[start] = CYAN
    for cell, color in color_cells.items():
        grid[cell] = color
    return grid


def _render_expected(inp: np.ndarray, traversed: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for cell in traversed[1:]:
        if int(out[cell]) == BLACK:
            out[cell] = CYAN
    return out


def _validate_generated_scene(
    inp: np.ndarray,
    traversed: list[tuple[int, int]],
    true_pairs: set[tuple[tuple[int, int], tuple[int, int]]],
    turn_greens: list[tuple[int, int]],
    path_distractors: set[tuple[int, int]],
    end_mode: str,
    final_direction: tuple[int, int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(turn_greens) == 0 or len(path_distractors) == 0:
        return False

    expected = _render_expected(inp, traversed)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    redirects, stop_reds, parsed_pairs = _true_domino_maps(inp)
    if parsed_pairs != true_pairs:
        return False

    for comp in _red_green_components(inp):
        if len(comp) > 2:
            return False
        if len(comp) == 2 and {int(inp[cell]) for cell in comp} != {RED, GREEN}:
            return False

    if any(green not in redirects for green in turn_greens):
        return False
    if not path_distractors.issubset(set(traversed)):
        return False

    last = traversed[-1]
    if end_mode == "red":
        if last not in stop_reds or int(inp[last]) != RED:
            return False
    else:
        if _in_bounds(inp.shape, _step(last, final_direction)):
            return False

    if np.array_equal(inp, expected):
        return False
    return True


def _generate_path_scene() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]], tuple[int, int], str, set[tuple[int, int]]] | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    start, direction = _choose_start(shape)
    traversed = [start]
    path_set = {start}
    color_cells: dict[tuple[int, int], int] = {}
    true_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    turn_greens: list[tuple[int, int]] = []

    pos = start
    turn_count = random.randint(1, 3)
    end_mode = random.choice(("exit", "red"))

    for _ in range(turn_count):
        candidates = _turn_candidates(shape, pos, direction, path_set, color_cells)
        if not candidates:
            return None
        length, next_dir, green, red = random.choice(candidates)
        segment = _segment_cells(pos, direction, length)
        traversed.extend(segment)
        path_set.update(segment)
        color_cells[red] = RED
        color_cells[green] = GREEN
        true_pairs.add((red, green))
        turn_greens.append(green)
        pos = green
        direction = next_dir

    if end_mode == "exit":
        segment = _exit_segment(shape, pos, direction, path_set, color_cells)
        if segment is None:
            return None
        traversed.extend(segment)
        path_set.update(segment)
    else:
        candidates = _terminal_red_candidates(shape, pos, direction, path_set, color_cells)
        if not candidates:
            return None
        length, red, green = random.choice(candidates)
        segment = _segment_cells(pos, direction, length)
        traversed.extend(segment)
        path_set.update(segment)
        color_cells[red] = RED
        color_cells[green] = GREEN
        true_pairs.add((red, green))

    path_distractors: set[tuple[int, int]] = set()
    distractor_count = random.randint(1, 2)
    for _ in range(distractor_count):
        candidates = _path_single_candidates(traversed, color_cells)
        if not candidates:
            return None
        cell = random.choice(candidates)
        color_cells[cell] = random.choice((RED, GREEN))
        path_distractors.add(cell)

    diag_candidates = _diagonal_pair_candidates(shape, path_set, color_cells)
    if not diag_candidates:
        return None
    first, second = random.choice(diag_candidates)
    if random.random() < 0.5:
        color_cells[first], color_cells[second] = RED, GREEN
    else:
        color_cells[first], color_cells[second] = GREEN, RED

    min_extra_dominoes = max(0, 2 - len(true_pairs))
    extra_dominoes = random.randint(min_extra_dominoes, min_extra_dominoes + 2)
    for _ in range(extra_dominoes):
        candidates = _off_path_domino_candidates(shape, path_set, color_cells)
        if not candidates:
            break
        first, second = random.choice(candidates)
        if random.random() < 0.5:
            red, green = first, second
        else:
            red, green = second, first
        color_cells[red] = RED
        color_cells[green] = GREEN
        true_pairs.add((red, green))

    extra_singles = random.randint(0, 2)
    for _ in range(extra_singles):
        candidates = _off_path_single_candidates(shape, path_set, color_cells)
        if not candidates:
            break
        cell = random.choice(candidates)
        color_cells[cell] = random.choice((RED, GREEN))

    inp = _render_input(shape, start, color_cells)
    if not _validate_generated_scene(inp, traversed, true_pairs, turn_greens, path_distractors, end_mode, direction):
        return None
    return inp, traversed, true_pairs, turn_greens, direction, end_mode, path_distractors


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_path_scene()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid red-green redirector puzzle")
