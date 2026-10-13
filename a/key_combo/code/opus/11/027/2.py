from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7

BLOCK_SIZE = 3
MACRO_SIDE = 5
GRID_SIDE = BLOCK_SIZE * MACRO_SIDE
GENERATION_RETRIES = 400


def _is_four_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _rotate_pattern(mask: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "h":
        return np.rot90(mask, k=-1)
    if orientation == "v":
        return np.rot90(mask, k=2)
    raise ValueError(f"unknown orientation: {orientation}")


def _domino_candidates(orientation: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if orientation == "h":
        return [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((2, 0), (2, 1)),
            ((2, 1), (2, 2)),
        ]
    if orientation == "v":
        return [
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((0, 2), (1, 2)),
            ((1, 2), (2, 2)),
        ]
    raise ValueError(f"unknown orientation: {orientation}")


def _first_domino_placement(mask: np.ndarray, orientation: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for a, b in _domino_candidates(orientation):
        if not mask[a] and not mask[b]:
            return a, b
    return None


def _enumerate_patterns() -> list[dict[str, object]]:
    patterns: list[dict[str, object]] = []
    for area in (2, 3, 4):
        for cells in itertools.combinations(range(BLOCK_SIZE * BLOCK_SIZE), area):
            mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
            for idx in cells:
                mask[idx // BLOCK_SIZE, idx % BLOCK_SIZE] = True
            if not _is_four_connected(mask):
                continue
            rot_h = _rotate_pattern(mask, "h")
            rot_v = _rotate_pattern(mask, "v")
            if np.array_equal(mask, rot_h) or np.array_equal(mask, rot_v):
                continue
            if np.array_equal(rot_h, rot_v):
                continue
            dom_h = _first_domino_placement(mask, "h")
            dom_v = _first_domino_placement(mask, "v")
            if dom_h is None or dom_v is None:
                continue
            patterns.append(
                {
                    "mask": mask,
                    "area": area,
                    "rot_h": rot_h,
                    "rot_v": rot_v,
                    "dom_h": dom_h,
                    "dom_v": dom_v,
                }
            )
    return patterns


PATTERN_LIBRARY = _enumerate_patterns()
PATTERNS_BY_AREA = {
    area: [entry for entry in PATTERN_LIBRARY if int(entry["area"]) == area]
    for area in (2, 3, 4)
}


def _parse_orientation(orange_coords: np.ndarray) -> str:
    cells = sorted(tuple(map(int, coord)) for coord in orange_coords)
    (r0, c0), (r1, c1) = cells
    if r0 == r1 and c1 == c0 + 1 and r0 in (0, 2):
        return "h"
    if c0 == c1 and r1 == r0 + 1 and c0 in (0, 2):
        return "v"
    raise AssertionError("orange marker must be a horizontal edge domino or vertical edge domino")


def _render_active_block(pattern: np.ndarray, orientation: str) -> np.ndarray:
    block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    block[pattern] = GREY
    placement = _first_domino_placement(pattern, orientation)
    assert placement is not None
    for r, c in placement:
        block[r, c] = ORANGE
    return block


def _output_from_assignments(
    assignments: list[tuple[int, int, np.ndarray, str]],
) -> np.ndarray:
    out = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)
    for br, bc, pattern, orientation in assignments:
        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE][_rotate_pattern(pattern, orientation)] = GREY
    return out


def _remove_markers_only(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[out == ORANGE] = BLACK
    return out


def _uniform_rotation_output(grid: np.ndarray, forced_orientation: str) -> np.ndarray:
    out = np.zeros_like(grid)
    for br in range(0, grid.shape[0], BLOCK_SIZE):
        for bc in range(0, grid.shape[1], BLOCK_SIZE):
            block = grid[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE]
            grey = block == GREY
            if grey.any():
                out[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE][_rotate_pattern(grey, forced_orientation)] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each marked 3x3 macro-block's grey pattern according to its orange domino orientation.

    Input structure:
    - The grid is a black-background 5x5 array of 3x3 macro-blocks.
    - Some macro-blocks contain a 2-4 cell grey 4-connected pattern plus one orange
      two-cell domino on an edge row or edge column of that same 3x3 block.
    - Empty macro-blocks are fully black, and the orange domino never overlaps the grey pattern.

    Rule:
    - Process each 3x3 block independently.
    - A horizontal orange domino means rotate that block's grey pattern 90 degrees clockwise
      around the block center; a vertical orange domino means rotate it 180 degrees.
    - Remove the orange domino from the output and leave empty blocks black.

    Key insight:
    - The marker's macro-block selects which local pattern to transform, while the domino's
      orientation encodes the rotation angle rather than a movement direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % BLOCK_SIZE == 0 and g.shape[1] % BLOCK_SIZE == 0

    out = np.zeros_like(g)
    for r0 in range(0, g.shape[0], BLOCK_SIZE):
        for c0 in range(0, g.shape[1], BLOCK_SIZE):
            block = g[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
            colors = set(map(int, np.unique(block)))
            assert colors <= {BLACK, GREY, ORANGE}

            grey = block == GREY
            orange = np.argwhere(block == ORANGE)
            if orange.size == 0:
                assert not grey.any(), "grey patterns must have an orange domino marker"
                continue

            assert orange.shape == (2, 2), "each active block must contain exactly one two-cell orange domino"
            assert 2 <= int(grey.sum()) <= 4, "grey pattern must contain 2-4 cells"
            assert _is_four_connected(grey), "grey pattern must be 4-connected"
            assert not np.any(grey[orange[:, 0], orange[:, 1]]), "orange and grey cells must not overlap"

            orientation = _parse_orientation(orange)
            rotated = _rotate_pattern(grey, orientation)
            out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE][rotated] = GREY
    return out


def _choose_positions(active_count: int) -> list[tuple[int, int]]:
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    interior = [(r, c) for r in range(1, 4) for c in range(1, 4)]
    all_positions = [(r, c) for r in range(MACRO_SIDE) for c in range(MACRO_SIDE)]

    chosen: list[tuple[int, int]] = []
    chosen.append(random.choice(corners))
    chosen.append(random.choice(interior))

    remaining = [pos for pos in all_positions if pos not in chosen]
    chosen.extend(random.sample(remaining, active_count - len(chosen)))
    random.shuffle(chosen)
    return chosen


def _choose_areas(active_count: int) -> list[int]:
    areas = [2, 3, 4]
    while len(areas) < active_count:
        areas.append(random.choice((2, 3, 4)))
    random.shuffle(areas)
    return areas[:active_count]


def _choose_patterns(areas: list[int]) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    used = set()
    for area in areas:
        pool = PATTERNS_BY_AREA[area]
        fresh = [entry for entry in pool if id(entry["mask"]) not in used]
        entry = random.choice(fresh or pool)
        used.add(id(entry["mask"]))
        chosen.append(np.array(entry["mask"], copy=True))
    return chosen


def _build_assignments() -> list[tuple[int, int, np.ndarray, str]]:
    active_count = random.randint(4, 7)
    positions = _choose_positions(active_count)
    areas = _choose_areas(active_count)
    patterns = _choose_patterns(areas)

    orientations = ["h", "v"]
    while len(orientations) < active_count:
        orientations.append(random.choice(("h", "v")))
    random.shuffle(orientations)

    return [
        (br, bc, pattern, orientation)
        for (br, bc), pattern, orientation in zip(positions, patterns, orientations)
    ]


def _render_input(assignments: list[tuple[int, int, np.ndarray, str]]) -> np.ndarray:
    grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)
    for br, bc, pattern, orientation in assignments:
        block = _render_active_block(pattern, orientation)
        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
    return grid


def _validate_assignments(assignments: list[tuple[int, int, np.ndarray, str]]) -> None:
    orientations = [orientation for _, _, _, orientation in assignments]
    assert "h" in orientations and "v" in orientations
    for _, _, pattern, orientation in assignments:
        actual = _rotate_pattern(pattern, orientation)
        alternate = _rotate_pattern(pattern, "v" if orientation == "h" else "h")
        assert not np.array_equal(actual, pattern), "rotation must visibly change the pattern"
        assert not np.array_equal(actual, alternate), "domino orientation must affect the output"
        assert _first_domino_placement(pattern, orientation) is not None


def _verify_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    assignments: list[tuple[int, int, np.ndarray, str]],
) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, _remove_markers_only(inp)):
        return False
    if np.array_equal(actual, _uniform_rotation_output(inp, "h")):
        return False
    if np.array_equal(actual, _uniform_rotation_output(inp, "v")):
        return False

    seen_positions = set()
    for br, bc, pattern, orientation in assignments:
        if (br, bc) in seen_positions:
            return False
        seen_positions.add((br, bc))

        r0 = br * BLOCK_SIZE
        c0 = bc * BLOCK_SIZE
        block_in = inp[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        block_out = actual[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        if np.count_nonzero(block_in == ORANGE) != 2:
            return False
        if np.count_nonzero(block_out == GREY) != int(pattern.sum()):
            return False
        if np.any(block_out == ORANGE):
            return False
        if np.array_equal(block_out == GREY, block_in == GREY):
            return False
        alternate = _rotate_pattern(pattern, "v" if orientation == "h" else "h")
        if np.array_equal(block_out == GREY, alternate):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        assignments = _build_assignments()
        _validate_assignments(assignments)
        inp = _render_input(assignments)
        expected = _output_from_assignments(assignments)
        if _verify_generated(inp, expected, assignments):
            return inp
    raise RuntimeError("failed to generate a valid macro-rotation ARC input")
