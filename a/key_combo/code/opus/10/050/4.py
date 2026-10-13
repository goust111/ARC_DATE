import random
from collections import Counter

import numpy as np


GRID_SIZE = 9
BG = 0
OBSTACLE = 1
SPIRAL = 2
BASE_STEP_LIMIT = 4
GEN_ATTEMPTS = 500
MAX_COMPONENTS = 2
MAX_COMPONENT_SIZE = 2
MAX_SKIP = MAX_COMPONENTS * MAX_COMPONENT_SIZE
START_POSITIONS = (
    (0, 2),
    (2, 0),
    (2, 2),
    (2, 4),
    (4, 2),
)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def _is_node_pixel(cell: tuple[int, int]) -> bool:
    return cell[0] % 2 == 0 and cell[1] % 2 == 0


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc):
            yield rr, cc


def _extract_start(grid: np.ndarray) -> tuple[int, int]:
    start_cells = np.argwhere(grid == SPIRAL)
    assert start_cells.shape == (1, 2), "expected exactly one red start pixel"
    start = tuple(int(v) for v in start_cells[0])
    assert start in START_POSITIONS, "start must be one of the valid mapped lattice nodes"
    return start


def _spiral_sequences(
    start: tuple[int, int],
    *,
    base_step_limit: int = BASE_STEP_LIMIT,
    extra_after: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Trace the doubled-lattice spiral in time order.

    The base footprint is the 5x5 lattice spiral clipped to the first four step-length
    pairs (1,1,2,2,3,3,4,4). Compensation continues from the same spiral afterwards.
    """
    lr, lc = start[0] // 2, start[1] // 2
    r, c = lr, lc
    direction_index = 0

    base_pixels = [start]
    tail_pixels: list[tuple[int, int]] = []

    for step_len in range(1, base_step_limit + 1):
        for _ in range(2):
            dr, dc = DIRS[direction_index % 4]
            for _ in range(step_len):
                nr, nc = r + dr, c + dc
                candidates = ((2 * r + dr, 2 * c + dc), (2 * nr, 2 * nc))
                for pr, pc in candidates:
                    if _in_bounds(pr, pc):
                        base_pixels.append((pr, pc))
                r, c = nr, nc
            direction_index += 1

    if extra_after <= 0:
        return base_pixels, tail_pixels

    step_len = base_step_limit + 1
    max_step_len = base_step_limit + 12
    while len(tail_pixels) < extra_after and step_len <= max_step_len:
        for _ in range(2):
            dr, dc = DIRS[direction_index % 4]
            for _ in range(step_len):
                nr, nc = r + dr, c + dc
                candidates = ((2 * r + dr, 2 * c + dc), (2 * nr, 2 * nc))
                for pr, pc in candidates:
                    if _in_bounds(pr, pc):
                        tail_pixels.append((pr, pc))
                        if len(tail_pixels) >= extra_after:
                            break
                r, c = nr, nc
                if len(tail_pixels) >= extra_after:
                    break
            direction_index += 1
            if len(tail_pixels) >= extra_after:
                break
        step_len += 1

    return base_pixels, tail_pixels


def _paint_output(start: tuple[int, int], obstacles: set[tuple[int, int]]) -> np.ndarray:
    base_pixels, _ = _spiral_sequences(start)
    skipped = sum(pixel in obstacles for pixel in base_pixels)
    _, tail_pixels = _spiral_sequences(start, extra_after=skipped)

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c in obstacles:
        out[r, c] = OBSTACLE

    for r, c in base_pixels:
        if (r, c) not in obstacles:
            out[r, c] = SPIRAL
    for r, c in tail_pixels:
        if (r, c) not in obstacles:
            out[r, c] = SPIRAL
    return out


def _paint_without_compensation(start: tuple[int, int], obstacles: set[tuple[int, int]]) -> np.ndarray:
    base_pixels, _ = _spiral_sequences(start)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c in obstacles:
        out[r, c] = OBSTACLE
    for r, c in base_pixels:
        if (r, c) not in obstacles:
            out[r, c] = SPIRAL
    return out


def _blue_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(grid == OBSTACLE)):
        if (r, c) in seen:
            continue
        stack = [(int(r), int(c))]
        seen.add((int(r), int(c)))
        comp: set[tuple[int, int]] = set()
        while stack:
            cr, cc = stack.pop()
            comp.add((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if grid[nr, nc] == OBSTACLE and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        components.append(comp)
    return components


def _valid_component_cells(base_pixels: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    eligible = [pixel for pixel in base_pixels if pixel != base_pixels[0]]
    singles = [(pixel,) for pixel in eligible]

    pairs: list[tuple[tuple[int, int], ...]] = []
    for i, a in enumerate(eligible):
        for b in eligible[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                ordered = tuple(sorted((a, b)))
                pairs.append(ordered)

    uniq = []
    seen = set()
    for comp in singles + pairs:
        if comp not in seen:
            seen.add(comp)
            uniq.append(comp)
    return uniq


def _sample_obstacles(
    start: tuple[int, int],
) -> set[tuple[int, int]] | None:
    base_pixels, tail_probe = _spiral_sequences(start, extra_after=MAX_SKIP)
    candidates = _valid_component_cells(base_pixels)
    if not candidates:
        return None

    want_two_components = random.random() < 0.5
    want_midpoint_hit = random.random() < 0.5
    want_node_hit = random.random() < 0.5

    def score(component: tuple[tuple[int, int], ...]) -> int:
        node_hits = sum(_is_node_pixel(cell) for cell in component)
        midpoint_hits = len(component) - node_hits
        total = 0
        if want_midpoint_hit and midpoint_hits:
            total += 2
        if want_node_hit and node_hits:
            total += 2
        if len(component) == 2:
            total += 1
        return total

    weighted = sorted(candidates, key=score, reverse=True)
    pool = weighted[: max(6, len(weighted) // 2)]

    for _ in range(80):
        n_components = 2 if want_two_components else 1
        chosen: list[tuple[tuple[int, int], ...]] = []
        used: set[tuple[int, int]] = set()
        shuffled = pool[:]
        random.shuffle(shuffled)
        for comp in shuffled:
            comp_set = set(comp)
            if used & comp_set:
                continue
            chosen.append(comp)
            used |= comp_set
            if len(chosen) == n_components:
                break
        if len(chosen) != n_components:
            continue

        skipped = len(used)
        if skipped == 0 or skipped > MAX_SKIP:
            continue

        tail_cells = tail_probe[:skipped]
        if len(tail_cells) < skipped:
            continue
        if used & set(tail_cells):
            continue

        has_node = any(_is_node_pixel(cell) for cell in used)
        has_midpoint = any(not _is_node_pixel(cell) for cell in used)
        if want_node_hit and not has_node:
            continue
        if want_midpoint_hit and not has_midpoint:
            continue

        return used

    return None


def _validate_input_semantics(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert set(map(int, np.unique(grid))) <= {BG, OBSTACLE, SPIRAL}

    start = _extract_start(grid)
    obstacles = {tuple(map(int, cell)) for cell in np.argwhere(grid == OBSTACLE)}
    base_pixels, tail_probe = _spiral_sequences(start, extra_after=MAX_SKIP)
    base_set = set(base_pixels)

    assert start not in obstacles
    assert obstacles
    assert obstacles <= base_set, "every obstacle must lie on the original spiral footprint"

    components = _blue_components(grid)
    assert 1 <= len(components) <= MAX_COMPONENTS
    assert all(1 <= len(comp) <= MAX_COMPONENT_SIZE for comp in components)

    skipped = sum(pixel in obstacles for pixel in base_pixels)
    assert skipped == len(obstacles)
    assert skipped > 0

    tail_cells = tail_probe[:skipped]
    assert len(tail_cells) == skipped, "generator must expose visible compensation"
    assert not (set(tail_cells) & obstacles), "tail compensation should stay unobstructed in generated cases"

    out = _paint_output(start, obstacles)
    baseline = _paint_without_compensation(start, obstacles)
    assert not np.array_equal(out, baseline), "generated cases must distinguish compensation from simple skipping"

    expected_red = len(base_pixels) - skipped + len(tail_cells)
    assert int((out == SPIRAL).sum()) == expected_red
    assert Counter(map(int, out.flatten()))[OBSTACLE] == len(obstacles)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a clipped doubled-lattice spiral, skip blue obstacle pixels, then extend the tail to conserve the skipped count.

    Input structure:
    - A 9x9 grid has black background, one red start marker, and one or two blue obstacle components.
    - The red marker sits on one of the valid even-even lattice nodes obtained by mapping a 3x3 source position into the 5x5 doubled lattice.
    - Blue obstacle pixels occupy cells of the canonical spiral footprint; they may sit on lattice nodes or edge-midpoint cells.

    Rule:
    - Use the red marker as the spiral origin on the doubled 5x5 lattice.
    - Trace the outward square spiral in up, right, down, left order with step lengths 1, 1, 2, 2, 3, 3, 4, 4; each lattice move paints its midpoint and destination node when those pixels are inside the 9x9 grid.
    - Whenever a spiral pixel is occupied by blue, leave that cell blue and count it as skipped; after the base spiral footprint ends, continue along the same spiral for exactly that many additional in-bounds pixels, again leaving blue cells unchanged.

    Key insight:
    - The obstacles do not reroute the spiral; they only punch holes in its fixed time-ordered footprint, and the missing count reappears as extra pixels at the far tail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    start = _extract_start(g)
    obstacles = {tuple(map(int, cell)) for cell in np.argwhere(g == OBSTACLE)}
    return _paint_output(start, obstacles)


def _gen_attempt() -> np.ndarray | None:
    start = random.choice(START_POSITIONS)
    obstacles = _sample_obstacles(start)
    if obstacles is None:
        return None

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    grid[start] = SPIRAL

    try:
        _validate_input_semantics(grid)
    except AssertionError:
        return None

    if not np.array_equal(main(grid), _paint_output(start, obstacles)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-obstacle ARC input")
