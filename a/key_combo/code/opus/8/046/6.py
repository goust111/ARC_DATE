from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

GRID_MIN = 8
GRID_MAX = 10
MAX_OUTPUT_SIZE = 30
MAX_BLOCKS_PER_COLOR = 5
MAX_DISTRACTORS = 3
GENERATION_ATTEMPTS = 400
ODD_BLUE_COUNTS = (1, 3, 5)
EVEN_BLUE_COUNTS = (2, 4)

_GEN_CALL_COUNT = 0


def _count_solid_blocks(grid: np.ndarray, color: int) -> int:
    h, w = grid.shape
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == color):
                count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count blue 2x2 squares and use their parity to choose a mirror-stacking direction.

    Input structure:
    - A black background grid contains several isolated solid blue 2x2 blocks.
    - It also contains several isolated solid red 2x2 blocks plus optional isolated
      single blue or red distractor pixels.
    - Objects do not overlap, and only fully blue 2x2 windows are relevant to the count.

    Rule:
    - Scan every 2x2 window and count those filled completely with blue.
    - If the blue-block count is odd, keep the input on the left and append its
      left-right mirror on the right.
    - If the blue-block count is even, keep the input on the top and append its
      top-bottom mirror below.

    Key insight:
    - The same blue blocks serve two roles at once: they are the objects being
      counted, and their parity chooses which whole-grid mirror copy is produced.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blue_block_count = _count_solid_blocks(g, BLUE)
    if blue_block_count % 2 == 1:
        return np.hstack((g, np.fliplr(g)))
    return np.vstack((g, np.flipud(g)))


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _component_is_square_block(component: list[tuple[int, int]]) -> bool:
    if len(component) != 4:
        return False
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return (r1 - r0 == 1) and (c1 - c0 == 1) and set(component) == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _cell_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    for nr, nc in _neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BLACK:
            return False
    return True


def _candidate_block_slots(height: int, width: int, needed: int) -> list[tuple[int, int]]:
    slot_sets: list[list[tuple[int, int]]] = []
    for row_start in range(3):
        row_positions = list(range(row_start, height - 1, 3))
        for col_start in range(3):
            col_positions = list(range(col_start, width - 1, 3))
            slots = [(r, c) for r in row_positions for c in col_positions]
            if len(slots) >= needed:
                slot_sets.append(slots)
    if not slot_sets:
        return []
    return random.choice(slot_sets)


def _place_distractors(grid: np.ndarray, count: int) -> None:
    h, w = grid.shape
    for _ in range(count):
        candidates = []
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) != BLACK:
                    continue
                if _cell_is_isolated(grid, r, c):
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = random.choice(candidates)
        grid[r, c] = random.choice((BLUE, RED))


def _random_transform(grid: np.ndarray) -> np.ndarray:
    transformed = np.array(grid, copy=True)
    rotations = random.randrange(4)
    if rotations:
        transformed = np.rot90(transformed, rotations)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    return np.asarray(transformed, dtype=int)


def _validate_input(grid: np.ndarray, intended_blue_blocks: int, intended_red_blocks: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if min(grid.shape) < GRID_MIN or max(grid.shape) > GRID_MAX:
        return False

    blue_components = _components_of_color(grid, BLUE)
    red_components = _components_of_color(grid, RED)

    blue_block_components = [comp for comp in blue_components if _component_is_square_block(comp)]
    red_block_components = [comp for comp in red_components if _component_is_square_block(comp)]
    blue_singletons = [comp for comp in blue_components if len(comp) == 1]
    red_singletons = [comp for comp in red_components if len(comp) == 1]

    if len(blue_block_components) != intended_blue_blocks:
        return False
    if len(red_block_components) != intended_red_blocks:
        return False
    if _count_solid_blocks(grid, BLUE) != intended_blue_blocks:
        return False
    if _count_solid_blocks(grid, RED) != intended_red_blocks:
        return False

    if any(len(comp) not in (1, 4) for comp in blue_components + red_components):
        return False
    if any(not _component_is_square_block(comp) for comp in blue_block_components + red_block_components):
        return False

    distractor_total = len(blue_singletons) + len(red_singletons)
    if distractor_total > MAX_DISTRACTORS:
        return False
    for comp in blue_singletons + red_singletons:
        r, c = comp[0]
        if not _cell_is_isolated(grid, r, c):
            return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > MAX_OUTPUT_SIZE:
        return False

    if intended_blue_blocks % 2 == 1:
        if out.shape != (grid.shape[0], grid.shape[1] * 2):
            return False
        if np.array_equal(grid, np.fliplr(grid)):
            return False
        if not np.array_equal(out[:, : grid.shape[1]], grid):
            return False
        if not np.array_equal(out[:, grid.shape[1] :], np.fliplr(grid)):
            return False
    else:
        if out.shape != (grid.shape[0] * 2, grid.shape[1]):
            return False
        if np.array_equal(grid, np.flipud(grid)):
            return False
        if not np.array_equal(out[: grid.shape[0], :], grid):
            return False
        if not np.array_equal(out[grid.shape[0] :, :], np.flipud(grid)):
            return False

    return True


def _attempt_generate(force_odd: bool) -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    blue_count = random.choice(ODD_BLUE_COUNTS if force_odd else EVEN_BLUE_COUNTS)
    red_count = random.randint(1, min(MAX_BLOCKS_PER_COLOR, 9 - blue_count))
    total_blocks = blue_count + red_count

    slots = _candidate_block_slots(height, width, total_blocks)
    if len(slots) < total_blocks:
        return None

    grid = np.zeros((height, width), dtype=int)
    chosen_slots = random.sample(slots, total_blocks)
    random.shuffle(chosen_slots)

    for r, c in chosen_slots[:blue_count]:
        grid[r : r + 2, c : c + 2] = BLUE
    for r, c in chosen_slots[blue_count:]:
        grid[r : r + 2, c : c + 2] = RED

    distractor_count = random.choice((0, 1, 1, 2, 2, 3))
    _place_distractors(grid, distractor_count)
    grid = _random_transform(grid)

    if not _validate_input(grid, blue_count, red_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    force_odd = (_GEN_CALL_COUNT % 2 == 0)
    _GEN_CALL_COUNT += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(force_odd=force_odd)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
