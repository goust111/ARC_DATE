from __future__ import annotations

from collections import Counter

import numpy as np


DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE: dict[str, str] = {"U": "D", "D": "U", "L": "R", "R": "L"}

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (10, 17)  # high exclusive
GRID_W_RANGE = (10, 17)  # high exclusive
NUM_COLORS_RANGE = (2, 5)  # high exclusive
PALETTE = tuple(range(10))


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.flat)
    assert counts, "expected at least one color"
    return int(counts.most_common(1)[0][0])


def _expected_bg(even_bg: int, odd_bg: int, r: int, c: int) -> int:
    return int(even_bg if (r + c) % 2 == 0 else odd_bg)


def _extract_seeds(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    parity = np.fromfunction(lambda r, c: (r + c) % 2, (H, W), dtype=int)
    even_bg = _mode_color(g[parity == 0])
    odd_bg = _mode_color(g[parity == 1])
    assert even_bg != odd_bg, "checkerboard needs two distinct background colors"

    seeds: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            expected = _expected_bg(even_bg, odd_bg, r, c)
            value = int(g[r, c])
            if value == expected:
                continue
            assert value not in {even_bg, odd_bg}, "seed colors must differ from both checker colors"
            seeds.append((r, c, value))

    assert seeds, "expected at least one seed"
    return int(even_bg), int(odd_bg), seeds


def _seed_direction(seed_pos: tuple[int, int], seed_set: set[tuple[int, int]], shape: tuple[int, int]) -> str:
    r, c = map(int, seed_pos)
    H, W = map(int, shape)

    missing: list[str] = []
    if r == 0:
        missing.append("U")
    if r == H - 1:
        missing.append("D")
    if c == 0:
        missing.append("L")
    if c == W - 1:
        missing.append("R")

    occupied = [
        name
        for name, (dr, dc) in DIRS.items()
        if (r + dr, c + dc) in seed_set
    ]

    if missing:
        assert len(missing) == 1, "corner seeds are ambiguous"
        assert not occupied, "boundary seeds should not also use adjacency clues"
        return OPPOSITE[missing[0]]

    assert len(occupied) == 1, "interior seeds need exactly one adjacent seed clue"
    return OPPOSITE[occupied[0]]


def _paint_branch(out: np.ndarray, seed: tuple[int, int], direction: str, length: int, color: int) -> None:
    H, W = out.shape
    r, c = map(int, seed)
    dr, dc = DIRS[direction]
    for step in range(1, int(length) + 1):
        nr = r + dr * step
        nc = c + dc * step
        assert 0 <= nr < H and 0 <= nc < W, "branch must stay inside the board"
        out[nr, nc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow monochrome branches on a checkerboard.

    The checkerboard colors are recovered from the majority color on each parity.
    Every non-checker pixel is a seed. A boundary seed grows inward from its unique
    touched edge; an interior seed grows away from its unique orthogonally adjacent
    seed. The number of same-colored seeds is the branch length for every seed of
    that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H >= 3 and W >= 3

    _even_bg, _odd_bg, seeds = _extract_seeds(g)
    seed_set = {(int(r), int(c)) for r, c, _color in seeds}
    color_counts = Counter(int(color) for _r, _c, color in seeds)

    out = g.copy()
    for r, c, color in seeds:
        direction = _seed_direction((int(r), int(c)), seed_set, (H, W))
        _paint_branch(out, (int(r), int(c)), direction, int(color_counts[int(color)]), int(color))
    return out


def _build_checkerboard(H: int, W: int, color_a: int, color_b: int, phase: int) -> np.ndarray:
    grid = np.empty((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            grid[r, c] = int(color_a if (r + c + phase) % 2 == 0 else color_b)
    return grid


def _line_cells(shape: tuple[int, int], seed: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]] | None:
    H, W = map(int, shape)
    r, c = map(int, seed)
    dr, dc = DIRS[direction]
    cells: list[tuple[int, int]] = []
    for step in range(0, int(length) + 1):
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < H and 0 <= nc < W):
            return None
        cells.append((int(nr), int(nc)))
    return cells


def _touches_existing_seed(
    seed: tuple[int, int],
    seed_positions: set[tuple[int, int]],
    allowed_neighbors: set[tuple[int, int]] | None = None,
) -> bool:
    allowed = set() if allowed_neighbors is None else set(allowed_neighbors)
    r, c = map(int, seed)
    for dr, dc in DIRS.values():
        neighbor = (int(r + dr), int(c + dc))
        if neighbor in seed_positions and neighbor not in allowed:
            return True
    return False


def _edge_candidates(
    shape: tuple[int, int],
    length: int,
    occupied: set[tuple[int, int]],
    seed_positions: set[tuple[int, int]],
) -> list[dict[str, object]]:
    H, W = map(int, shape)
    candidates: list[dict[str, object]] = []

    for c in range(1, W - 1):
        seed = (0, c)
        if _touches_existing_seed(seed, seed_positions):
            continue
        cells = _line_cells((H, W), seed, "D", int(length))
        if cells is not None and all(cell not in occupied for cell in cells):
            candidates.append(
                {
                    "cells": tuple(cells),
                    "placements": ((seed, "D"),),
                }
            )

        seed = (H - 1, c)
        if _touches_existing_seed(seed, seed_positions):
            continue
        cells = _line_cells((H, W), seed, "U", int(length))
        if cells is not None and all(cell not in occupied for cell in cells):
            candidates.append(
                {
                    "cells": tuple(cells),
                    "placements": ((seed, "U"),),
                }
            )

    for r in range(1, H - 1):
        seed = (r, 0)
        if _touches_existing_seed(seed, seed_positions):
            continue
        cells = _line_cells((H, W), seed, "R", int(length))
        if cells is not None and all(cell not in occupied for cell in cells):
            candidates.append(
                {
                    "cells": tuple(cells),
                    "placements": ((seed, "R"),),
                }
            )

        seed = (r, W - 1)
        if _touches_existing_seed(seed, seed_positions):
            continue
        cells = _line_cells((H, W), seed, "L", int(length))
        if cells is not None and all(cell not in occupied for cell in cells):
            candidates.append(
                {
                    "cells": tuple(cells),
                    "placements": ((seed, "L"),),
                }
            )

    return candidates


def _pair_candidates(
    shape: tuple[int, int],
    left_or_top_length: int,
    right_or_bottom_length: int,
    occupied: set[tuple[int, int]],
    seed_positions: set[tuple[int, int]],
) -> list[dict[str, object]]:
    H, W = map(int, shape)
    candidates: list[dict[str, object]] = []

    for r in range(1, H - 1):
        for c in range(1, W - 2):
            seed_a = (r, c)
            seed_b = (r, c + 1)
            if _touches_existing_seed(seed_a, seed_positions, {seed_b}):
                continue
            if _touches_existing_seed(seed_b, seed_positions, {seed_a}):
                continue
            cells_a = _line_cells((H, W), seed_a, "L", int(left_or_top_length))
            cells_b = _line_cells((H, W), seed_b, "R", int(right_or_bottom_length))
            if cells_a is None or cells_b is None:
                continue
            cells = tuple(cells_a + cells_b)
            if len(set(cells)) != len(cells):
                continue
            if all(cell not in occupied for cell in cells):
                candidates.append(
                    {
                        "cells": cells,
                        "placements": ((seed_a, "L"), (seed_b, "R")),
                    }
                )

    for r in range(1, H - 2):
        for c in range(1, W - 1):
            seed_a = (r, c)
            seed_b = (r + 1, c)
            if _touches_existing_seed(seed_a, seed_positions, {seed_b}):
                continue
            if _touches_existing_seed(seed_b, seed_positions, {seed_a}):
                continue
            cells_a = _line_cells((H, W), seed_a, "U", int(left_or_top_length))
            cells_b = _line_cells((H, W), seed_b, "D", int(right_or_bottom_length))
            if cells_a is None or cells_b is None:
                continue
            cells = tuple(cells_a + cells_b)
            if len(set(cells)) != len(cells):
                continue
            if all(cell not in occupied for cell in cells):
                candidates.append(
                    {
                        "cells": cells,
                        "placements": ((seed_a, "U"), (seed_b, "D")),
                    }
                )

    return candidates


def _manual_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        seed = item["seed"]
        direction = item["direction"]
        length = item["length"]
        color = item["color"]
        _paint_branch(out, seed, direction, length, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = int(RNG.integers(*GRID_H_RANGE))
    W = int(RNG.integers(*GRID_W_RANGE))

    num_colors = int(RNG.integers(*NUM_COLORS_RANGE))
    branch_lengths = list(RNG.choice(np.array([1, 2, 3, 4], dtype=int), size=num_colors, replace=False))
    RNG.shuffle(branch_lengths)

    if max(branch_lengths) + 2 > max(H, W):
        return None

    colors = list(RNG.choice(np.array(PALETTE, dtype=int), size=num_colors + 2, replace=False))
    bg_a, bg_b = int(colors[0]), int(colors[1])
    seed_colors = [int(v) for v in colors[2:]]

    seed_specs: list[dict[str, int]] = []
    for color, length in zip(seed_colors, branch_lengths):
        for _ in range(int(length)):
            seed_specs.append({"color": int(color), "length": int(length)})

    total_seeds = len(seed_specs)
    pair_seed_options = [count for count in range(2, total_seeds, 2) if total_seeds - count >= 1]
    if not pair_seed_options:
        return None

    RNG.shuffle(seed_specs)
    pair_seed_count = int(RNG.choice(np.array(pair_seed_options, dtype=int)))
    pair_group_count = pair_seed_count // 2
    edge_group_count = total_seeds - pair_seed_count

    groups: list[dict[str, object]] = []
    index = 0
    for _ in range(pair_group_count):
        groups.append({"kind": "pair", "specs": seed_specs[index : index + 2]})
        index += 2
    for _ in range(edge_group_count):
        groups.append({"kind": "edge", "specs": seed_specs[index : index + 1]})
        index += 1

    RNG.shuffle(groups)
    groups.sort(
        key=lambda g: sum(int(spec["length"]) for spec in g["specs"]) + len(g["specs"]),
        reverse=True,
    )

    occupied: set[tuple[int, int]] = set()
    seed_positions: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []
    seen_kinds: Counter[str] = Counter()

    for group in groups:
        specs = group["specs"]
        if group["kind"] == "edge":
            candidates = _edge_candidates((H, W), int(specs[0]["length"]), occupied, seed_positions)
        else:
            candidates = _pair_candidates(
                (H, W),
                int(specs[0]["length"]),
                int(specs[1]["length"]),
                occupied,
                seed_positions,
            )

        if not candidates:
            return None

        chosen = candidates[int(RNG.integers(len(candidates)))]
        chosen_cells = [tuple(map(int, cell)) for cell in chosen["cells"]]
        chosen_placements = chosen["placements"]

        if group["kind"] == "edge":
            seed, direction = chosen_placements[0]
            spec = specs[0]
            placements.append(
                {
                    "seed": tuple(map(int, seed)),
                    "direction": str(direction),
                    "length": int(spec["length"]),
                    "color": int(spec["color"]),
                    "kind": "edge",
                }
            )
            seed_positions.add(tuple(map(int, seed)))
            seen_kinds["edge"] += 1
        else:
            for (seed, direction), spec in zip(chosen_placements, specs):
                placements.append(
                    {
                        "seed": tuple(map(int, seed)),
                        "direction": str(direction),
                        "length": int(spec["length"]),
                        "color": int(spec["color"]),
                        "kind": "pair",
                    }
                )
                seed_positions.add(tuple(map(int, seed)))
                seen_kinds["pair"] += 1

        occupied.update(chosen_cells)

    if seen_kinds["edge"] == 0 or seen_kinds["pair"] == 0:
        return None

    directions = {str(item["direction"]) for item in placements}
    if not (directions & {"U", "D"}) or not (directions & {"L", "R"}):
        return None

    phase = int(RNG.integers(0, 2))
    grid = _build_checkerboard(H, W, bg_a, bg_b, phase)
    for item in placements:
        r, c = item["seed"]
        grid[r, c] = int(item["color"])

    try:
        produced = main(grid)
    except AssertionError:
        return None

    expected = _manual_output(grid, placements)
    if not np.array_equal(produced, expected):
        return None

    if np.array_equal(produced, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard seed puzzle")
