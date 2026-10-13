from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MOVE_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GRID_HEIGHT_RANGE = (14, 25)
GRID_WIDTH_RANGE = (14, 25)
OBJECT_COUNT_RANGE = (2, 6)
OBJECT_SIZE_RANGE = (2, 7)
GAP_RANGE = (2, 8)
MAX_GENERATION_ATTEMPTS = 300
SCENARIO_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 180
PALETTE = tuple(range(10))


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(comp)
    return components


def _translation_for_pair(
    grid: np.ndarray, bg: int, obj: list[tuple[int, int]], beacon: tuple[int, int]
) -> tuple[int, int]:
    rows = [r for r, _ in obj]
    cols = [c for _, c in obj]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    br, bc = beacon

    candidates: list[tuple[int, int]] = []

    if br < min_r:
        aligned_rows = [r for r, c in obj if c == bc]
        if aligned_rows:
            top = min(aligned_rows)
            if all(int(grid[r, bc]) == bg for r in range(br + 1, top)):
                candidates.append((br + 1 - top, 0))

    if br > max_r:
        aligned_rows = [r for r, c in obj if c == bc]
        if aligned_rows:
            bottom = max(aligned_rows)
            if all(int(grid[r, bc]) == bg for r in range(bottom + 1, br)):
                candidates.append((br - 1 - bottom, 0))

    if bc < min_c:
        aligned_cols = [c for r, c in obj if r == br]
        if aligned_cols:
            left = min(aligned_cols)
            if all(int(grid[br, c]) == bg for c in range(bc + 1, left)):
                candidates.append((0, bc + 1 - left))

    if bc > max_c:
        aligned_cols = [c for r, c in obj if r == br]
        if aligned_cols:
            right = max(aligned_cols)
            if all(int(grid[br, c]) == bg for c in range(right + 1, bc)):
                candidates.append((0, bc - 1 - right))

    assert len(candidates) == 1, f"expected exactly one beacon direction, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each colored object until it sits immediately beside its matching same-color beacon.

    Input structure:
    - The background is the dominant color.
    - Each non-background color appears as one 4-connected object of size 2-6 plus one isolated singleton beacon.
    - Every beacon lies strictly above, below, left, or right of its matching object, sharing a row or column with a clear background gap.

    Rule:
    - For each color, identify the large component as the movable object and the size-1 component as its beacon.
    - Determine whether the beacon is above, below, left, or right of the object, then translate the object along that axis.
    - Stop when the object's nearest aligned edge cell is exactly one step from the beacon; keep the beacon in place and clear the object's old location.

    Key insight:
    - The singleton is not another object fragment: it is an alignment anchor that encodes both the movement direction and the exact stopping distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = np.full_like(g, bg)

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps = _components_of_color(g, color)
        obj_comps = [comp for comp in comps if len(comp) > 1]
        beacon_comps = [comp for comp in comps if len(comp) == 1]

        assert len(obj_comps) == 1, f"color {color} should have one movable object"
        assert len(beacon_comps) == 1, f"color {color} should have one beacon"

        obj = obj_comps[0]
        beacon = beacon_comps[0][0]
        dr, dc = _translation_for_pair(g, bg, obj, beacon)
        shifted = [(r + dr, c + dc) for r, c in obj]

        assert beacon not in shifted
        assert any(abs(r - beacon[0]) + abs(c - beacon[1]) == 1 for r, c in shifted)

        br, bc = beacon
        out[br, bc] = color
        for r, c in shifted:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert out[r, c] == bg, "translated objects must stay disjoint"
            out[r, c] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_shape(rng: np.random.Generator, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nbr = (r + dr, c + dc)
                if nbr not in cells:
                    frontier.append(nbr)
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    return _normalize(cells)


def _rotate90(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _normalize({(c, -r) for r, c in cells})


def _random_transform(rng: np.random.Generator, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = list(cells)
    for _ in range(int(rng.integers(0, 4))):
        out = _rotate90(out)
    transformed = {(r, c) for r, c in out}
    if rng.random() < 0.5:
        transformed = {(-r, c) for r, c in transformed}
    if rng.random() < 0.5:
        transformed = {(r, -c) for r, c in transformed}
    return _normalize(transformed)


def _anchor_candidates(shape: list[tuple[int, int]], move: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = move
    if dr == -1:
        edge = min(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge]
    if dr == 1:
        edge = max(r for r, _ in shape)
        return [(r, c) for r, c in shape if r == edge]
    if dc == -1:
        edge = min(c for _, c in shape)
        return [(r, c) for r, c in shape if c == edge]
    edge = max(c for _, c in shape)
    return [(r, c) for r, c in shape if c == edge]


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _place_output_shape(
    shape: list[tuple[int, int]],
    anchor: tuple[int, int],
    move: tuple[int, int],
    beacon: tuple[int, int],
) -> list[tuple[int, int]]:
    ar, ac = anchor
    br, bc = beacon
    dr, dc = move
    base_r = br - dr - ar
    base_c = bc - dc - ac
    return [(base_r + r, base_c + c) for r, c in shape]


def _line_gap_cells(beacon: tuple[int, int], move: tuple[int, int], gap: int) -> list[tuple[int, int]]:
    br, bc = beacon
    dr, dc = move
    return [(br - dr * step, bc - dc * step) for step in range(1, gap + 1)]


def _in_bounds(cells: list[tuple[int, int]] | set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                expanded.add((rr, cc))
    return expanded


def _direction_sequence(rng: np.random.Generator, count: int) -> list[tuple[int, int]]:
    indices = list(rng.permutation(len(MOVE_DIRS)))
    chosen = indices[: min(count, len(indices))]
    while len(chosen) < count:
        chosen.append(int(rng.integers(0, len(MOVE_DIRS))))
    rng.shuffle(chosen)
    return [MOVE_DIRS[idx] for idx in chosen]


def _expected_grid(shape: tuple[int, int], bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        beacon = spec["beacon"]
        output_cells = spec["output_cells"]
        br, bc = beacon
        out[br, bc] = color
        for r, c in output_cells:
            out[r, c] = color
    return out


def _try_place_scenario(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    move: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = grid_shape
    for _ in range(SCENARIO_ATTEMPTS):
        size = int(rng.integers(*OBJECT_SIZE_RANGE))
        shape = _random_transform(rng, _random_shape(rng, size))
        anchors = _anchor_candidates(shape, move)
        anchor = anchors[int(rng.integers(0, len(anchors)))]
        gap = int(rng.integers(*GAP_RANGE))

        for _ in range(PLACEMENT_ATTEMPTS):
            beacon = (int(rng.integers(0, h)), int(rng.integers(0, w)))
            output_cells = _place_output_shape(shape, anchor, move, beacon)
            input_cells = _shift_cells(output_cells, -move[0] * gap, -move[1] * gap)
            gap_cells = _line_gap_cells(beacon, move, gap)

            input_colored = set(input_cells)
            output_colored = set(output_cells)
            input_all = input_colored | {beacon}
            output_all = output_colored | {beacon}
            if len(input_colored) != len(input_cells) or len(output_colored) != len(output_cells):
                continue
            if beacon in input_colored or beacon in output_colored:
                continue
            if not _in_bounds(input_all, grid_shape):
                continue
            if not _in_bounds(output_all, grid_shape):
                continue
            if not _in_bounds(gap_cells, grid_shape):
                continue
            if input_all & reserved_input:
                continue
            if set(gap_cells) & reserved_input:
                continue
            if output_all & reserved_output:
                continue
            if input_all & occupied_input:
                continue
            if output_all & occupied_output:
                continue

            return {
                "beacon": beacon,
                "input_cells": input_cells,
                "output_cells": output_cells,
                "gap_cells": gap_cells,
            }
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    obj_count = int(rng.integers(*OBJECT_COUNT_RANGE))
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    bg = int(rng.integers(0, len(PALETTE)))
    colors = [color for color in PALETTE if color != bg]
    rng.shuffle(colors)
    colors = colors[:obj_count]
    moves = _direction_sequence(rng, obj_count)

    occupied_input: set[tuple[int, int]] = set()
    reserved_input: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    for color, move in zip(colors, moves):
        spec = _try_place_scenario(
            rng, (h, w), move, occupied_input, reserved_input, occupied_output, reserved_output
        )
        if spec is None:
            return None

        spec["color"] = color
        specs.append(spec)

        input_cells = set(spec["input_cells"])
        output_cells = set(spec["output_cells"])
        beacon = spec["beacon"]
        gap_cells = set(spec["gap_cells"])

        input_colored = input_cells | {beacon}
        output_colored = output_cells | {beacon}

        occupied_input |= input_colored
        reserved_input |= _expand_cells(input_colored, (h, w))
        reserved_input |= gap_cells
        occupied_output |= output_colored
        reserved_output |= _expand_cells(output_colored, (h, w))

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        beacon = spec["beacon"]
        br, bc = beacon
        grid[br, bc] = color
        for r, c in spec["input_cells"]:
            grid[r, c] = color

    expected = _expected_grid((h, w), bg, specs)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid beacon-translation input")
