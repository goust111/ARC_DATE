from __future__ import annotations

import itertools
import random

import numpy as np


MACRO_SIZE = 3
MACRO_SIDE = 5
GRID_SIDE = MACRO_SIZE * MACRO_SIDE

BG = 0
LONGEST = 1
SECOND = 2
THIRD = 3
SHORTEST = 4
WALL = 5
SOURCE = 8

DIRECTIONS = (
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
)
RANK_COLORS = (LONGEST, SECOND, THIRD, SHORTEST)
GEN_RETRIES = 300


def _macro_bounds(mr: int, mc: int) -> tuple[slice, slice]:
    r0 = mr * MACRO_SIZE
    c0 = mc * MACRO_SIZE
    return slice(r0, r0 + MACRO_SIZE), slice(c0, c0 + MACRO_SIZE)


def _fill_macro(grid: np.ndarray, mr: int, mc: int, color: int) -> None:
    rs, cs = _macro_bounds(mr, mc)
    grid[rs, cs] = int(color)


def _source_pixel(source_macro: tuple[int, int]) -> tuple[int, int]:
    mr, mc = source_macro
    return mr * MACRO_SIZE + 1, mc * MACRO_SIZE + 1


def _classify_input_macros(grid: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIDE, GRID_SIDE), f"expected {GRID_SIDE}x{GRID_SIDE} grid"

    source: tuple[int, int] | None = None
    walls: set[tuple[int, int]] = set()
    for mr in range(MACRO_SIDE):
        for mc in range(MACRO_SIDE):
            rs, cs = _macro_bounds(mr, mc)
            block = g[rs, cs]
            if np.all(block == WALL):
                walls.add((mr, mc))
                continue
            if int(block[1, 1]) == SOURCE and int(np.count_nonzero(block)) == 1:
                assert source is None, "expected exactly one source macro"
                source = (mr, mc)
                continue
            assert np.all(block == BG), "every non-wall macro must be empty or the unique source macro"

    assert source is not None, "expected one source macro"
    return source, walls


def _ray_paths(
    source: tuple[int, int], walls: set[tuple[int, int]]
) -> dict[str, list[tuple[int, int]]]:
    sr, sc = source
    paths: dict[str, list[tuple[int, int]]] = {}
    for name, (dr, dc) in DIRECTIONS:
        r = sr + dr
        c = sc + dc
        covered: list[tuple[int, int]] = []
        while 0 <= r < MACRO_SIDE and 0 <= c < MACRO_SIDE:
            if (r, c) in walls:
                break
            covered.append((r, c))
            r += dr
            c += dc
        paths[name] = covered
    return paths


def _ray_caps(source: tuple[int, int]) -> dict[str, int]:
    sr, sc = source
    return {
        "up": sr,
        "down": MACRO_SIDE - 1 - sr,
        "left": sc,
        "right": MACRO_SIDE - 1 - sc,
    }


def _rank_colors_by_length(paths: dict[str, list[tuple[int, int]]]) -> dict[str, int]:
    lengths = {name: len(coords) for name, coords in paths.items()}
    assert len(set(lengths.values())) == 4, "ray lengths must be pairwise distinct"

    ordered = sorted(lengths.items(), key=lambda item: item[1], reverse=True)
    return {name: color for (name, _length), color in zip(ordered, RANK_COLORS)}


def main(grid: np.ndarray) -> np.ndarray:
    """Project the cyan source across macro blocks, rank the four ray lengths, and recolor each ray by its global length rank.

    Input structure:
    - The grid is a 15x15 canvas interpreted as a 5x5 macro grid of 3x3 blocks on black background.
    - Some macros are solid grey 3x3 wall blocks, and exactly one non-wall macro contains a single cyan center pixel.
    - Every remaining macro is entirely black.

    Rule:
    - Treat the cyan pixel's containing macro as the source and cast four cardinal rays in macro coordinates.
    - A ray covers consecutive empty macros until the next macro would be a grey wall or leave the 5x5 macro grid; the wall macro itself is not covered.
    - Count the covered macros on each of the four rays, rank those four lengths from longest to shortest, and assign colors blue, red, green, yellow to ranks 1 through 4.
    - Fill every covered macro on a ray with that ray's rank color, while preserving the cyan source pixel, grey wall macros, and untouched black macros.

    Key insight:
    - Direction does not determine color: the wall layout first determines the four propagation lengths, and only the global ordering of those lengths determines how each direction is recolored.
    """
    g = np.asarray(grid, dtype=int)
    source, walls = _classify_input_macros(g)
    paths = _ray_paths(source, walls)
    color_for_dir = _rank_colors_by_length(paths)

    out = g.copy()
    for name, coords in paths.items():
        color = color_for_dir[name]
        for mr, mc in coords:
            _fill_macro(out, mr, mc, color)

    sr, sc = _source_pixel(source)
    out[sr, sc] = SOURCE
    return out


def _valid_sources() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(1, MACRO_SIDE - 1)
        for c in range(1, MACRO_SIDE - 1)
        if (r, c) != (MACRO_SIDE // 2, MACRO_SIDE // 2)
    ]


def _valid_length_assignments(source: tuple[int, int]) -> list[dict[str, int]]:
    caps = _ray_caps(source)
    names = [name for name, _delta in DIRECTIONS]
    assignments = []
    for perm in itertools.permutations((0, 1, 2, 3)):
        candidate = dict(zip(names, perm))
        if not all(candidate[name] <= caps[name] for name in names):
            continue
        wall_limited = sum(candidate[name] < caps[name] for name in names)
        boundary_limited = sum(candidate[name] == caps[name] for name in names)
        if wall_limited >= 2 and boundary_limited >= 1:
            assignments.append(candidate)
    return assignments


def _build_input(source: tuple[int, int], lengths: dict[str, int], extra_walls: set[tuple[int, int]]) -> np.ndarray:
    walls = set(extra_walls)
    caps = _ray_caps(source)
    sr, sc = source
    for name, (dr, dc) in DIRECTIONS:
        length = lengths[name]
        cap = caps[name]
        if length < cap:
            walls.add((sr + dr * (length + 1), sc + dc * (length + 1)))

    grid = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)
    for mr, mc in walls:
        _fill_macro(grid, mr, mc, WALL)
    pr, pc = _source_pixel(source)
    grid[pr, pc] = SOURCE
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    source, walls = _classify_input_macros(grid)
    sr, sc = source
    assert 0 < sr < MACRO_SIDE - 1 and 0 < sc < MACRO_SIDE - 1, "source must be off the macro border"
    assert source != (MACRO_SIDE // 2, MACRO_SIDE // 2), "center source cannot yield four distinct lengths"

    paths = _ray_paths(source, walls)
    lengths = {name: len(coords) for name, coords in paths.items()}
    assert sorted(lengths.values()) == [0, 1, 2, 3], "15x15 layout requires ray lengths to be exactly 0,1,2,3"

    caps = _ray_caps(source)
    wall_limited = sum(lengths[name] < caps[name] for name, _delta in DIRECTIONS)
    boundary_limited = sum(lengths[name] == caps[name] for name, _delta in DIRECTIONS)
    assert wall_limited >= 2, "examples should require reading multiple stopping walls"
    assert boundary_limited >= 1, "examples should also show boundary stopping"

    out = main(grid)
    color_for_dir = _rank_colors_by_length(paths)

    path_cells = {cell for coords in paths.values() for cell in coords}
    for mr in range(MACRO_SIDE):
        for mc in range(MACRO_SIDE):
            rs, cs = _macro_bounds(mr, mc)
            block = out[rs, cs]
            cell = (mr, mc)
            if cell == source:
                assert int(block[1, 1]) == SOURCE
                block_wo_source = block.copy()
                block_wo_source[1, 1] = BG
                assert np.all(block_wo_source == BG), "source macro should otherwise remain black"
            elif cell in walls:
                assert np.all(block == WALL), "wall macro must stay grey"
            elif cell in path_cells:
                expected = next(color_for_dir[name] for name, coords in paths.items() if cell in coords)
                assert np.all(block == expected), "every covered macro must be filled with its ray rank color"
            else:
                assert np.all(block == BG), "non-ray empty macros must remain black"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        source = random.choice(_valid_sources())
        assignments = _valid_length_assignments(source)
        if not assignments:
            continue
        lengths = random.choice(assignments)

        provisional = _build_input(source, lengths, set())
        path_cells = {cell for coords in _ray_paths(source, _classify_input_macros(provisional)[1]).values() for cell in coords}

        candidates = [
            (mr, mc)
            for mr in range(MACRO_SIDE)
            for mc in range(MACRO_SIDE)
            if (mr, mc) != source and (mr, mc) not in path_cells
        ]
        random.shuffle(candidates)
        extra_count = random.randint(0, min(3, len(candidates)))
        extra_walls = set(candidates[:extra_count])

        grid = _build_input(source, lengths, extra_walls)
        _validate_generated(grid)
        return grid

    raise RuntimeError("failed to generate valid macro-ray ranking input")
