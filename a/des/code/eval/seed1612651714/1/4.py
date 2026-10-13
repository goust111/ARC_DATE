from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
COMPASS_DIRS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)

GRID_SIDE_RANGE = (12, 21)
OBJECT_COUNT_RANGE = (3, 6)
GEN_ATTEMPTS = 400
TRUNCATION_BIAS = 0.35


def _sign(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _marker_offsets(grid: np.ndarray, r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = grid.shape
    offsets = []
    for dr, dc in NEIGHBORS8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
            offsets.append((dr, dc))
    return tuple(sorted(offsets))


def _same_side_direction(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    sum_dr = sum(dr for dr, _ in offsets)
    sum_dc = sum(dc for _, dc in offsets)
    direction = (_sign(sum_dr), _sign(sum_dc))
    if direction == (0, 0):
        return None
    dir_dr, dir_dc = direction
    for dr, dc in offsets:
        if dr * dir_dr + dc * dir_dc <= 0:
            return None
    return direction


def _is_negation_symmetric(offsets: tuple[tuple[int, int], ...]) -> bool:
    cells = set(offsets)
    return all((-dr, -dc) in cells for dr, dc in cells)


def _classify_offsets(offsets: tuple[tuple[int, int], ...]) -> tuple[str, tuple[int, int] | None]:
    assert 1 <= len(offsets) <= 4, "each seed must have 1..4 grey markers"
    direction = _same_side_direction(offsets)
    if direction is not None:
        return "bar", direction
    if all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in offsets) and _is_negation_symmetric(offsets):
        return "x", None
    if all(abs(dr) + abs(dc) == 1 for dr, dc in offsets) and _is_negation_symmetric(offsets):
        return "cross", None
    raise AssertionError("unrecognized marker symmetry pattern")


def _grow_cells(
    shape: tuple[int, int],
    seed: tuple[int, int],
    kind: str,
    length: int,
    direction: tuple[int, int] | None,
) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    cells = {(r, c)}
    if kind == "cross":
        dirs = ORTHO_DIRS
    elif kind == "x":
        dirs = DIAG_DIRS
    else:
        assert direction is not None
        dirs = (direction,)

    for dr, dc in dirs:
        for step in range(1, length + 1):
            nr = r + dr * step
            nc = c + dc * step
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed according to the local symmetry of grey markers in
    its 8-neighborhood. Grey markers encode both shape type and arm length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full(g.shape, BLACK, dtype=int)
    seed_positions = np.argwhere((g != BLACK) & (g != GREY))

    for r, c in seed_positions:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        offsets = _marker_offsets(g, r, c)
        kind, direction = _classify_offsets(offsets)
        for rr, cc in _grow_cells(g.shape, (r, c), kind, len(offsets), direction):
            out[rr, cc] = color

    return out


def _build_pattern_catalog() -> tuple[
    dict[int, list[tuple[tuple[int, int], ...]]],
    dict[int, list[tuple[tuple[int, int], ...]]],
    dict[tuple[int, int], dict[int, list[tuple[tuple[int, int], ...]]]],
]:
    cross_patterns: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    x_patterns: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    bar_patterns: dict[tuple[int, int], dict[int, list[tuple[tuple[int, int], ...]]]] = {
        direction: {} for direction in COMPASS_DIRS
    }

    for size in range(1, 5):
        for combo in combinations(NEIGHBORS8, size):
            offsets = tuple(sorted(combo))
            try:
                kind, direction = _classify_offsets(offsets)
            except AssertionError:
                continue
            if kind == "cross":
                cross_patterns.setdefault(size, []).append(offsets)
            elif kind == "x":
                x_patterns.setdefault(size, []).append(offsets)
            else:
                assert direction is not None
                bar_patterns[direction].setdefault(size, []).append(offsets)

    return cross_patterns, x_patterns, bar_patterns


CROSS_PATTERNS, X_PATTERNS, BAR_PATTERNS = _build_pattern_catalog()


def _sample_object_specs(n_objects: int) -> list[dict]:
    kinds = ["cross", "x", "bar"]
    branch_pool = kinds + [random.choice(kinds) for _ in range(n_objects - len(kinds))]
    random.shuffle(branch_pool)
    colors = random.sample(SEED_COLORS, n_objects)

    specs = []
    for color, kind in zip(colors, branch_pool):
        if kind == "cross":
            count = random.choice(sorted(CROSS_PATTERNS))
            offsets = random.choice(CROSS_PATTERNS[count])
            direction = None
        elif kind == "x":
            count = random.choice(sorted(X_PATTERNS))
            offsets = random.choice(X_PATTERNS[count])
            direction = None
        else:
            direction = random.choice(COMPASS_DIRS)
            count = random.choice(sorted(BAR_PATTERNS[direction]))
            offsets = random.choice(BAR_PATTERNS[direction][count])
        specs.append(
            {
                "kind": kind,
                "color": color,
                "count": count,
                "offsets": offsets,
                "direction": direction,
            }
        )

    specs.sort(key=lambda spec: (-spec["count"], spec["kind"]))
    return specs


def _seed_halo(shape: tuple[int, int], seed: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    r, c = seed
    halo = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                halo.add((nr, nc))
    return halo


def _full_output_size(kind: str, count: int) -> int:
    if kind == "bar":
        return 1 + count
    return 1 + 4 * count


def _choose_seed(
    shape: tuple[int, int],
    spec: dict,
    blocked: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            seed = (r, c)
            halo = _seed_halo(shape, seed)
            output = _grow_cells(shape, seed, spec["kind"], spec["count"], spec["direction"])
            if halo & blocked or output & blocked:
                continue
            truncated = len(output) < _full_output_size(spec["kind"], spec["count"])
            edge_distance = min(r - 1, h - 2 - r, c - 1, w - 2 - c)
            candidates.append((seed, halo, output, truncated, edge_distance))

    if not candidates:
        return None

    if random.random() < TRUNCATION_BIAS:
        truncated_candidates = [item for item in candidates if item[3]]
        if truncated_candidates:
            candidates = truncated_candidates

    min_edge = min(item[4] for item in candidates)
    near_edge = [item for item in candidates if item[4] == min_edge]
    pool = near_edge if random.random() < 0.45 else candidates
    seed, halo, output, _, _ = random.choice(pool)
    blocked.update(halo)
    blocked.update(output)
    return seed


def _render_input(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for item in placements:
        r, c = item["seed"]
        grid[r, c] = item["color"]
        for dr, dc in item["offsets"]:
            grid[r + dr, c + dc] = GREY
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False

    seed_positions = np.argwhere((grid != BLACK) & (grid != GREY))
    if seed_positions.shape[0] < 3:
        return False

    seen_kinds = set()
    used_grey = set()
    for r, c in seed_positions:
        r = int(r)
        c = int(c)
        try:
            offsets = _marker_offsets(grid, r, c)
            kind, _ = _classify_offsets(offsets)
        except AssertionError:
            return False
        seen_kinds.add(kind)
        for dr, dc in offsets:
            pos = (r + dr, c + dc)
            if pos in used_grey:
                return False
            used_grey.add(pos)

    actual_grey = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    if used_grey != actual_grey:
        return False
    if seen_kinds != {"bar", "cross", "x"}:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    specs = _sample_object_specs(n_objects)

    blocked: set[tuple[int, int]] = set()
    placements = []
    for spec in specs:
        seed = _choose_seed((h, w), spec, blocked)
        if seed is None:
            return None
        placements.append({**spec, "seed": seed})

    grid = _render_input((h, w), placements)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
