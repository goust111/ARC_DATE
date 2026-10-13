from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

SIDE_MIN = 2
SIDE_MAX = 4
GENERATOR_SIDE_MIN = 2
GRID_FACTOR_MIN = 4
GRID_FACTOR_MAX = 10
MAX_ATTEMPTS = 400
DUST_MAX = 8
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _line_positions(side: int, size: int) -> tuple[int, ...]:
    return tuple(range(int(side), int(size), int(side) + 1))


def _find_red_square(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == RED)
    assert coords.size > 0, "expected one red square"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    assert side_r == side_c, "expected a square red object"
    side = side_r
    assert SIDE_MIN <= side <= SIDE_MAX, "red square side must be 2-4"
    assert int(coords.shape[0]) == side * side, "red square must be solid"
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == RED), "red square must fill its bbox"
    return side, (int(r0), int(c0), int(r1), int(c1))


def _find_teal_segment(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, p)) for p in np.argwhere(grid == TEAL)]
    assert len(coords) == 2, "expected exactly two teal pixels"
    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) == 1 and abs(c0 - c1) == 1, "teal pixels must form a diagonal segment"
    return coords[0], coords[1]


def _segment_source_and_direction(
    p0: tuple[int, int], p1: tuple[int, int], shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    center_r = (shape[0] - 1) / 2.0
    center_c = (shape[1] - 1) / 2.0

    def dist2(pt: tuple[int, int]) -> float:
        return (pt[0] - center_r) ** 2 + (pt[1] - center_c) ** 2

    ordered = sorted((p0, p1), key=lambda pt: (dist2(pt), pt[0], pt[1]))
    source = ordered[0]
    other = ordered[1]
    dr = int(source[0] - other[0])
    dc = int(source[1] - other[1])
    assert abs(dr) == 1 and abs(dc) == 1
    return source, (dr, dc)


def _paint_lattice(shape: tuple[int, int], side: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    separators = _line_positions(side, shape[0])
    out[list(separators), :] = GREY
    out[:, list(separators)] = GREY
    return out


def _trace_ray(
    base: np.ndarray, source: tuple[int, int], direction: tuple[int, int]
) -> tuple[set[tuple[int, int]], str, dict[str, int]]:
    out = base.copy()
    row_lines = set(np.where(np.any(base == GREY, axis=1))[0].tolist())
    col_lines = set(np.where(np.any(base == GREY, axis=0))[0].tolist())
    r, c = source
    dr, dc = direction
    visited: set[tuple[int, int, int, int]] = set()
    green_cells: set[tuple[int, int]] = set()
    hits = {"horizontal": 0, "vertical": 0, "cross": 0}

    while True:
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in visited, "ray entered a non-start cycle"
        visited.add(state)

        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
            return green_cells, "boundary", hits

        target = int(out[nr, nc])
        if target == GREY:
            hit_h = nr in row_lines
            hit_v = nc in col_lines
            if hit_h and hit_v:
                dr, dc = -dr, -dc
                hits["cross"] += 1
            elif hit_h:
                dr = -dr
                hits["horizontal"] += 1
            else:
                assert hit_v
                dc = -dc
                hits["vertical"] += 1
            continue

        r, c = nr, nc
        if (r, c) == source:
            return green_cells, "loop", hits
        if out[r, c] == BLACK:
            out[r, c] = GREEN
            green_cells.add((int(r), int(c)))


def main(grid: np.ndarray) -> np.ndarray:
    """Generate a grey lattice from the red square size, then cast one diagonal green ray that bounces on the lattice lines.

    Input structure:
    - The background is black.
    - One red solid square encodes the cell side length s, where s is 2-4.
    - One teal 2-pixel diagonal segment sits away from the square and away from future lattice lines.
    - Optional isolated blue distractor pixels may appear; they are ignored by the construction.

    Rule:
    - Read the red square side length s.
    - Draw full grey rows and columns at indices s, 2s+1, 3s+2, ... on a copy of the input-sized canvas.
    - Restore the original red square on top of the lattice and keep the teal seed segment unchanged.
    - Drop any blue distractor pixels.
    - Use the teal segment orientation as the ray direction, choosing the pixel closer to the grid center as the emitter, then advance one diagonal step at a time.
    - If the next step would enter a grey row, flip the vertical direction; if it would enter a grey column, flip the horizontal direction; if it would enter a grey intersection, flip both; the ray never enters grey cells.
    - Paint only black visited cells green and stop when the ray returns to the emitter or leaves the grid.

    Key insight:
    - The red square does not directly block the ray: it only sets the lattice spacing, and those generated lattice lines become the reflecting geometry that fully determines the green path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    side, (r0, c0, r1, c1) = _find_red_square(g)
    p0, p1 = _find_teal_segment(g)
    source, direction = _segment_source_and_direction(p0, p1, g.shape)

    out = _paint_lattice(g.shape, side)
    out[g == RED] = RED
    out[g == TEAL] = TEAL

    green_cells, _termination, _hits = _trace_ray(out, source, direction)
    for r, c in green_cells:
        out[r, c] = GREEN

    assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == RED)
    assert out[p0] == TEAL and out[p1] == TEAL
    return out


def _block_start(block_index: int, side: int) -> int:
    return int(block_index) * (int(side) + 1)


def _edge_blocks(k: int) -> list[tuple[int, int]]:
    blocks = []
    for br in range(k):
        for bc in range(k):
            if br in (0, k - 1) or bc in (0, k - 1):
                blocks.append((br, bc))
    return blocks


def _segment_candidates(
    size: int, side: int, k: int, square_block: tuple[int, int]
) -> list[dict[str, object]]:
    row_lines = set(_line_positions(side, size))
    col_lines = set(_line_positions(side, size))
    square_br, square_bc = square_block
    center_r = (size - 1) / 2.0
    center_c = (size - 1) / 2.0
    candidates: list[dict[str, object]] = []

    for br in range(k):
        for bc in range(k):
            if (br, bc) == square_block:
                continue
            if abs(br - square_br) + abs(bc - square_bc) < 2:
                continue

            r_start = _block_start(br, side)
            c_start = _block_start(bc, side)
            rows = range(r_start, r_start + side)
            cols = range(c_start, c_start + side)

            for source_r in rows:
                for source_c in cols:
                    if source_r in row_lines or source_c in col_lines:
                        continue
                    if not (1 <= source_r < size - 1 and 1 <= source_c < size - 1):
                        continue

                    for dr, dc in DIRS:
                        other_r = source_r - dr
                        other_c = source_c - dc
                        if not (1 <= other_r < size - 1 and 1 <= other_c < size - 1):
                            continue
                        if other_r not in rows or other_c not in cols:
                            continue
                        if other_r in row_lines or other_c in col_lines:
                            continue

                        source_d = (source_r - center_r) ** 2 + (source_c - center_c) ** 2
                        other_d = (other_r - center_r) ** 2 + (other_c - center_c) ** 2
                        if not source_d < other_d:
                            continue

                        candidate = {
                            "source": (int(source_r), int(source_c)),
                            "other": (int(other_r), int(other_c)),
                            "direction": (int(dr), int(dc)),
                            "block": (int(br), int(bc)),
                        }
                        candidates.append(candidate)
    return candidates


def _candidate_profile(termination: str, hits: dict[str, int]) -> str:
    if termination == "loop":
        if hits["cross"] > 0:
            return "loop_cross"
        return "loop_split"
    if hits["cross"] > 0:
        return "boundary_cross"
    if hits["horizontal"] > 0 and hits["vertical"] > 0:
        return "boundary_split_both"
    if hits["horizontal"] > 0:
        return "boundary_horizontal"
    if hits["vertical"] > 0:
        return "boundary_vertical"
    return "boundary_direct"


def _profile_weights(profiles: list[str]) -> list[float]:
    weight_map = {
        "loop_cross": 2.0,
        "loop_split": 1.7,
        "boundary_cross": 2.0,
        "boundary_split_both": 1.6,
        "boundary_horizontal": 1.3,
        "boundary_vertical": 1.3,
        "boundary_direct": 1.0,
    }
    return [weight_map[p] for p in profiles]


def _choose_layout() -> tuple[int, int]:
    side = random.randint(GENERATOR_SIDE_MIN, SIDE_MAX)
    max_k = min(GRID_FACTOR_MAX, 30 // (side + 1))
    k = random.randint(GRID_FACTOR_MIN, max_k)
    return side, k


def _build_input_from_choice(
    side: int, k: int, square_block: tuple[int, int], choice: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    size = k * (side + 1)
    inp = np.zeros((size, size), dtype=int)

    square_r = _block_start(square_block[0], side)
    square_c = _block_start(square_block[1], side)
    inp[square_r : square_r + side, square_c : square_c + side] = RED

    source = choice["source"]
    other = choice["other"]
    inp[source] = TEAL
    inp[other] = TEAL

    occupied = {tuple(map(int, p)) for p in np.argwhere(inp != BLACK)}
    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if (r, c) not in occupied
    ]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(0, DUST_MAX)]:
        inp[r, c] = BLUE

    out = main(inp)
    lattice = _paint_lattice(inp.shape, side)
    lattice[inp == RED] = RED
    lattice[inp == TEAL] = TEAL
    green_cells, termination, hits = _trace_ray(lattice, source, choice["direction"])

    info = {
        "output": out,
        "termination": termination,
        "hits": hits,
        "profile": _candidate_profile(termination, hits),
        "green_cells": green_cells,
        "source": source,
        "other": other,
        "direction": choice["direction"],
        "square_block": square_block,
    }
    return inp, info


def _valid_square_blocks(k: int) -> list[tuple[int, int]]:
    blocks = _edge_blocks(k)
    random.shuffle(blocks)
    return blocks


def _semantic_checks(inp: np.ndarray, out: np.ndarray, info: dict[str, object], side: int) -> bool:
    try:
        recovered_side, _bbox = _find_red_square(inp)
        if recovered_side != side:
            return False
        p0, p1 = _find_teal_segment(inp)
        source, direction = _segment_source_and_direction(p0, p1, inp.shape)
        if source != info["source"] or direction != info["direction"]:
            return False

        separators = _line_positions(side, inp.shape[0])
        for idx in separators:
            row = out[idx]
            col = out[:, idx]
            if not np.all(np.isin(row, [GREY, RED])):
                return False
            if not np.all(np.isin(col, [GREY, RED])):
                return False

        expected = _paint_lattice(inp.shape, side)
        expected[inp == RED] = RED
        expected[inp == TEAL] = TEAL
        green_cells, termination, hits = _trace_ray(expected, source, direction)
        for r, c in green_cells:
            expected[r, c] = GREEN
        if not np.array_equal(expected, out):
            return False
        if termination != info["termination"] or hits != info["hits"]:
            return False
        if len(green_cells) == 0:
            return False
        if any(out[r, c] != GREEN for r, c in green_cells):
            return False
        if np.any((out == GREEN) & ((expected == GREY) | (inp == RED) | (inp == TEAL))):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        side, k = _choose_layout()
        size = k * (side + 1)
        row_lines = set(_line_positions(side, size))
        if not row_lines:
            continue

        square_blocks = _valid_square_blocks(k)
        random.shuffle(square_blocks)
        for square_block in square_blocks:
            candidates = _segment_candidates(size, side, k, square_block)
            if not candidates:
                continue

            profiled: dict[str, list[dict[str, object]]] = {}
            for cand in candidates:
                lattice = _paint_lattice((size, size), side)
                green_cells, termination, hits = _trace_ray(lattice, cand["source"], cand["direction"])
                if not green_cells:
                    continue
                profile = _candidate_profile(termination, hits)
                profiled.setdefault(profile, []).append(
                    {
                        **cand,
                        "termination": termination,
                        "hits": hits,
                        "profile": profile,
                    }
                )

            if not profiled:
                continue

            profiles = sorted(profiled)
            chosen_profile = random.choices(profiles, weights=_profile_weights(profiles), k=1)[0]
            choice = random.choice(profiled[chosen_profile])
            inp, info = _build_input_from_choice(side, k, square_block, choice)
            if _semantic_checks(inp, info["output"], info, side):
                return inp

    raise RuntimeError("failed to generate a valid input")
