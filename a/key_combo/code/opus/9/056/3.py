from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
BLOCK = 2

TARGET_COUNT_RANGE = (5, 7)
GRID_SIZE_RANGE = (16, 26)
STEP_RANGE = (4, 8)  # top-left deltas; bbox gaps are 2..5 cells.
MAX_GENERATION_ATTEMPTS = 300

PATTERNS: dict[str, dict[int, list[tuple[int, int]]]] = {
    "ladder": {
        5: [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 2)],
        6: [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 2), (3, 2)],
        7: [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 2), (3, 2), (3, 3)],
    },
    "hook": {
        5: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)],
        6: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0), (3, 0)],
        7: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0), (3, 0), (3, 3)],
    },
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _extract_blocks(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    components = _connected_components(grid)
    starts: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []
    for comp in components:
        color = int(comp["color"])
        assert color in {RED, BLUE}, "expected only red start and blue target blocks"
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == 4, "every object must be a 2x2 block"
        assert r1 - r0 == 1 and c1 - c0 == 1, "every object must occupy a 2x2 bbox"
        pos = (r0, c0)
        if color == RED:
            starts.append(pos)
        else:
            targets.append(pos)
    assert len(starts) == 1, "expected exactly one red start block"
    assert TARGET_COUNT_RANGE[0] <= len(targets) <= TARGET_COUNT_RANGE[1], "expected 5-7 blue target blocks"
    return starts[0], targets


def _visible_step(grid: np.ndarray, current: tuple[int, int], target: tuple[int, int]) -> tuple[str, int] | None:
    ar, ac = current
    br, bc = target
    if ar == br:
        c0 = min(ac, bc) + BLOCK
        c1 = max(ac, bc) - 1
        if c0 <= c1 and not np.all(grid[ar : ar + BLOCK, c0 : c1 + 1] == BG):
            return None
        return "h", abs(bc - ac)
    if ac == bc:
        r0 = min(ar, br) + BLOCK
        r1 = max(ar, br) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac : ac + BLOCK] == BG):
            return None
        return "v", abs(br - ar)
    return None


def _paint_corridor(
    out: np.ndarray,
    current: tuple[int, int],
    target: tuple[int, int],
    axis: str,
    color: int,
) -> None:
    ar, ac = current
    br, bc = target
    if axis == "h":
        c0 = min(ac, bc) + BLOCK
        c1 = max(ac, bc) - 1
        if c0 <= c1:
            out[ar : ar + BLOCK, c0 : c1 + 1] = int(color)
        return
    assert axis == "v"
    r0 = min(ar, br) + BLOCK
    r1 = max(ar, br) - 1
    if r0 <= r1:
        out[r0 : r1 + 1, ac : ac + BLOCK] = int(color)


def _paint_block(out: np.ndarray, pos: tuple[int, int], color: int) -> None:
    r, c = pos
    out[r : r + BLOCK, c : c + BLOCK] = int(color)


def _walk_steps(
    grid: np.ndarray,
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], str]]:
    current = start
    remaining = list(targets)
    steps: list[tuple[tuple[int, int], str]] = []
    while remaining:
        candidates: list[tuple[int, str, tuple[int, int]]] = []
        for target in remaining:
            visible = _visible_step(grid, current, target)
            if visible is None:
                continue
            axis, distance = visible
            candidates.append((distance, axis, target))
        assert candidates, "walk terminated before all targets were visited"
        best_distance = min(distance for distance, _axis, _target in candidates)
        nearest = [item for item in candidates if item[0] == best_distance]
        assert len(nearest) == 1, "nearest visible target must be unique"
        _distance, axis, target = nearest[0]
        steps.append((target, axis))
        remaining.remove(target)
        current = target
    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse the blue 2x2 blocks by greedy line-of-sight order and recolor each visited target and corridor by visit parity.

    Input structure:
    - The grid has a black background, exactly one red 2x2 start block, and 5-7 blue 2x2 target blocks.
    - Two blocks can interact only when they share the same two-row band or the same two-column band.
    - Any non-black pixels inside the gap between aligned blocks block visibility.

    Rule:
    - Start from the red block and repeatedly inspect all unvisited blue blocks.
    - Among blocks that are visible in the same row-band or column-band, choose the unique nearest one by Manhattan distance.
    - Paint the background corridor between the current block and that chosen block.
    - Number the visited targets from 1 upward; odd-indexed targets become green and their incoming corridors become orange, while even-indexed targets become yellow and their incoming corridors become grey.
    - Move to the chosen target and continue until all targets have been visited; the red start block stays unchanged.

    Key insight:
    - Final colors are determined by the greedy traversal order, not by spatial parity, so the solver must mentally simulate the whole nearest-visible walk before recoloring anything.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    start, targets = _extract_blocks(g)
    steps = _walk_steps(g, start, targets)

    out = g.copy()
    current = start
    for index, (target, axis) in enumerate(steps, start=1):
        if index % 2 == 1:
            corridor_color = ORANGE
            block_color = GREEN
        else:
            corridor_color = GREY
            block_color = YELLOW
        _paint_corridor(out, current, target, axis, corridor_color)
        _paint_block(out, target, block_color)
        current = target
    _paint_block(out, start, RED)
    return out


def _positions_from_pattern(
    rng: np.random.Generator,
    pattern: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    row_labels = sorted({r for r, _ in pattern})
    col_labels = sorted({c for _, c in pattern})

    row_steps = [int(rng.integers(*STEP_RANGE)) for _ in range(len(row_labels) - 1)]
    col_steps = [int(rng.integers(*STEP_RANGE)) for _ in range(len(col_labels) - 1)]

    if pattern[:4] == PATTERNS["ladder"][5][:4]:
        if row_steps:
            row_steps[0] = max(row_steps[0], col_steps[0] + 1)
            if row_steps[0] >= STEP_RANGE[1]:
                row_steps[0] = STEP_RANGE[1] - 1
                col_steps[0] = min(col_steps[0], row_steps[0] - 1)

    row_pos = {row_labels[0]: 0}
    col_pos = {col_labels[0]: 0}
    for idx, step in enumerate(row_steps, start=1):
        row_pos[row_labels[idx]] = row_pos[row_labels[idx - 1]] + step
    for idx, step in enumerate(col_steps, start=1):
        col_pos[col_labels[idx]] = col_pos[col_labels[idx - 1]] + step
    return [(row_pos[r], col_pos[c]) for r, c in pattern]


def _transform_positions(
    rng: np.random.Generator,
    positions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = list(positions)
    if bool(rng.integers(0, 2)):
        out = [(c, r) for r, c in out]
    max_r = max(r for r, _ in out)
    max_c = max(c for _, c in out)
    if bool(rng.integers(0, 2)):
        out = [(max_r - r, c) for r, c in out]
    if bool(rng.integers(0, 2)):
        out = [(r, max_c - c) for r, c in out]
    return out


def _shift_into_grid(
    rng: np.random.Generator,
    positions: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]]:
    max_r = max(r for r, _ in positions)
    max_c = max(c for _, c in positions)
    span_h = max_r + BLOCK
    span_w = max_c + BLOCK

    min_h = max(GRID_SIZE_RANGE[0], span_h + 2)
    min_w = max(GRID_SIZE_RANGE[0], span_w + 2)
    max_h = min(GRID_SIZE_RANGE[1], span_h + 8)
    max_w = min(GRID_SIZE_RANGE[1], span_w + 8)
    assert min_h <= max_h and min_w <= max_w

    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    row_offset = int(rng.integers(1, h - span_h))
    col_offset = int(rng.integers(1, w - span_w))

    shifted = [(r + row_offset, c + col_offset) for r, c in positions]
    return shifted[0], shifted[1:], (h, w)


def _bbox_gap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + BLOCK - 1, ac0 + BLOCK - 1
    br1, bc1 = br0 + BLOCK - 1, bc0 + BLOCK - 1
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap, col_gap


def _separated_enough(positions: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(positions):
        for b in positions[i + 1 :]:
            row_gap, col_gap = _bbox_gap(a, b)
            if row_gap < 2 and col_gap < 2:
                return False
    return True


def _render_input(
    shape: tuple[int, int],
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _paint_block(grid, start, RED)
    for target in targets:
        _paint_block(grid, target, BLUE)
    return grid


def _expected_output_from_order(
    shape: tuple[int, int],
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    out = _render_input(shape, start, targets)
    current = start
    probe = out.copy()
    for index, target in enumerate(targets, start=1):
        visible = _visible_step(probe, current, target)
        assert visible is not None
        axis, _distance = visible
        if index % 2 == 1:
            corridor_color = ORANGE
            block_color = GREEN
        else:
            corridor_color = GREY
            block_color = YELLOW
        _paint_corridor(out, current, target, axis, corridor_color)
        _paint_block(out, target, block_color)
        current = target
    return out


def _validate_case(
    grid: np.ndarray,
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> bool:
    all_positions = [start] + targets
    if not _separated_enough(all_positions):
        return False

    current = start
    remaining = list(targets)
    axes: list[str] = []
    branch_steps = 0
    for expected in targets:
        candidates: list[tuple[int, str, tuple[int, int]]] = []
        for target in remaining:
            visible = _visible_step(grid, current, target)
            if visible is None:
                continue
            axis, distance = visible
            candidates.append((distance, axis, target))
        if not candidates:
            return False
        if len(candidates) > 1:
            branch_steps += 1
        best_distance = min(distance for distance, _axis, _target in candidates)
        nearest = [item for item in candidates if item[0] == best_distance]
        if len(nearest) != 1 or nearest[0][2] != expected:
            return False
        axes.append(nearest[0][1])
        remaining.remove(expected)
        current = expected

    if branch_steps == 0 or len(set(axes)) < 2:
        return False

    expected = _expected_output_from_order(grid.shape, start, targets)
    try:
        actual = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    target_count = int(rng.integers(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    family = str(rng.choice(sorted(PATTERNS)))
    pattern = PATTERNS[family][target_count]
    positions = _positions_from_pattern(rng, pattern)
    positions = _transform_positions(rng, positions)
    start, targets, shape = _shift_into_grid(rng, positions)
    grid = _render_input(shape, start, targets)
    if not _validate_case(grid, start, targets):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid greedy-parity ARC input")
