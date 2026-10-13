from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
INPUT_COLORS = (3, 4, 5, 6, 7, 8, 9)

ORTH_FAMILY_A = ((0, -1), (0, 1))   # horizontal
ORTH_FAMILY_B = ((-1, 0), (1, 0))   # vertical
DIAG_FAMILY_A = ((-1, -1), (1, 1))  # "\"
DIAG_FAMILY_B = ((-1, 1), (1, -1))  # "/"

SCENARIO_CYCLE = (
    (3, True),
    (2, False),
    (3, False),
    (3, True),
)
MAX_GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sorted_coords(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted({(int(r), int(c)) for r, c in cells})


def _validate_initial_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(g != BLACK)]
    assert 2 <= len(seeds) <= 3, "expected two or three seed pixels"
    colors = [color for _, _, color in seeds]
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    assert all(color in INPUT_COLORS for color in colors), "red and blue are reserved for descendants"
    rows = [r for r, _, _ in seeds]
    cols = [c for _, c, _ in seeds]
    assert len(set(rows)) == len(rows), "seed rows must be distinct"
    assert len(set(cols)) == len(cols), "seed columns must be distinct"
    return sorted(seeds)


def _march(
    shape: tuple[int, int],
    occupied: np.ndarray,
    start: tuple[int, int],
    dr: int,
    dc: int,
    *,
    stop_on_color: bool,
) -> tuple[list[tuple[int, int]], bool]:
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return cells, False
        if stop_on_color and occupied[nr, nc]:
            return cells, True
        cells.append((nr, nc))
        r, c = nr, nc


def _regime_for_generation(generation: int, regime_mode: str) -> str:
    if generation == 1:
        return "orth"
    if regime_mode == "alternate":
        return "diag" if generation % 2 == 0 else "orth"
    if regime_mode == "orth_only_after_first":
        return "orth"
    if regime_mode == "diag_only_after_first":
        return "diag"
    raise ValueError(f"unknown regime_mode: {regime_mode}")


def _child_color(generation: int) -> int:
    return RED if generation % 2 == 1 else BLUE


def _family_dirs(regime: str) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    if regime == "orth":
        return ORTH_FAMILY_A, ORTH_FAMILY_B
    if regime == "diag":
        return DIAG_FAMILY_A, DIAG_FAMILY_B
    raise ValueError(f"unknown regime: {regime}")


def _simulate(
    grid: np.ndarray,
    *,
    block_after_first: bool = True,
    regime_mode: str = "alternate",
    max_generations: int | None = None,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    seeds = _validate_initial_seeds(g)
    out = g.copy()
    current_seeds = [(r, c) for r, c, _ in seeds]
    generations: list[dict] = []
    seen_seed_sets: set[tuple[tuple[int, int], ...]] = set()

    generation = 1
    while current_seeds:
        key = tuple(current_seeds)
        if key in seen_seed_sets:
            break
        seen_seed_sets.add(key)
        regime = _regime_for_generation(generation, regime_mode)
        family_a_dirs, family_b_dirs = _family_dirs(regime)
        stop_on_color = generation > 1 and block_after_first
        occupied_snapshot = out != BLACK

        family_a_cells: list[set[tuple[int, int]]] = []
        family_b_cells: list[set[tuple[int, int]]] = []
        blocked_rays = 0
        painted_this_generation: set[tuple[int, int]] = set()

        for r, c in current_seeds:
            seed_color = int(out[r, c])
            cells_a: list[tuple[int, int]] = []
            cells_b: list[tuple[int, int]] = []

            for dr, dc in family_a_dirs:
                ray_cells, blocked = _march(
                    out.shape,
                    occupied_snapshot,
                    (r, c),
                    dr,
                    dc,
                    stop_on_color=stop_on_color,
                )
                cells_a.extend(ray_cells)
                blocked_rays += int(blocked)

            for dr, dc in family_b_dirs:
                ray_cells, blocked = _march(
                    out.shape,
                    occupied_snapshot,
                    (r, c),
                    dr,
                    dc,
                    stop_on_color=stop_on_color,
                )
                cells_b.extend(ray_cells)
                blocked_rays += int(blocked)

            set_a = set(cells_a)
            set_b = set(cells_b)
            family_a_cells.append(set_a)
            family_b_cells.append(set_b)
            for rr, cc in set_a | set_b:
                out[rr, cc] = seed_color
            painted_this_generation.update(set_a)
            painted_this_generation.update(set_b)

        next_seed_cells: set[tuple[int, int]] = set()
        for i in range(len(current_seeds)):
            for j in range(len(current_seeds)):
                if i == j:
                    continue
                next_seed_cells.update(family_a_cells[i] & family_b_cells[j])

        next_seeds = _sorted_coords(next_seed_cells)
        child_color = _child_color(generation)
        for rr, cc in next_seeds:
            out[rr, cc] = child_color

        generations.append(
            {
                "index": generation,
                "regime": regime,
                "seed_positions": list(current_seeds),
                "seed_count": len(current_seeds),
                "ray_cell_count": len(painted_this_generation),
                "blocked_rays": blocked_rays,
                "total_rays": 4 * len(current_seeds),
                "next_seeds": next_seeds,
                "child_color": child_color,
            }
        )

        if not next_seeds:
            break
        if max_generations is not None and generation >= max_generations:
            break
        current_seeds = next_seeds
        generation += 1

    info = {
        "generations": generations,
        "generation_count": len(generations),
        "initial_seed_count": len(seeds),
        "initial_positions": [(r, c) for r, c, _ in seeds],
        "final_colored": int(np.count_nonzero(out)),
        "final_density": float(np.count_nonzero(out) / out.size),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow alternating orthogonal and diagonal cross-chains from the seed pixels, recoloring each new generation's intersections.

    Input structure:
    - The grid is black except for two or three single-pixel seeds.
    - Seed colors are all different, and none of them is red or blue.
    - No two seeds share a row or a column.

    Rule:
    - First, each seed draws a full same-colored orthogonal cross to the grid edges.
    - Every cell where one cross's horizontal arm meets another cross's vertical arm becomes a red second-generation seed.
    - From the second generation onward, all seeds of the current generation fire same-colored rays in the current regime:
      diagonal on even-numbered generations, orthogonal on odd-numbered generations after the first.
    - These later rays stop just before any already colored cell and never overwrite it; all rays of one generation are drawn simultaneously.
    - After a generation is drawn, every cell shared by a family-A ray of one seed and a family-B ray of a different seed becomes the next generation's seed color
      (red after orthogonal generations, blue after diagonal generations). If no such new intersections exist, stop.

    Key insight:
    - The output is a chain reaction: intersections are not endpoints but fresh seeds, and the propagation geometry alternates between "+" and "X" while older paint blocks later rays.
    """
    solved, _ = _simulate(grid, block_after_first=True, regime_mode="alternate")
    return solved


def _pick_scenario() -> tuple[int, bool]:
    global _GEN_CALL_COUNT
    scenario = SCENARIO_CYCLE[_GEN_CALL_COUNT % len(SCENARIO_CYCLE)]
    _GEN_CALL_COUNT += 1
    return scenario


def _sample_axis(length: int, count: int, margin: int, min_gap: int) -> list[int] | None:
    candidates = list(range(margin, length - margin))
    if len(candidates) < count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_seed_grid(seed_count: int, require_deep: bool) -> np.ndarray | None:
    if seed_count == 2:
        h = random.randint(13, 20) if require_deep else random.randint(11, 18)
        w = random.randint(13, 20) if require_deep else random.randint(11, 18)
        margin = 2 if require_deep else 1
        min_gap = 3 if require_deep else 2
    else:
        h = random.randint(16, 22) if require_deep else random.randint(14, 20)
        w = random.randint(16, 22) if require_deep else random.randint(14, 20)
        margin = 3 if require_deep else 2
        min_gap = 3

    rows = _sample_axis(h, seed_count, margin, min_gap)
    cols = _sample_axis(w, seed_count, margin, min_gap)
    if rows is None or cols is None:
        return None

    perm = list(range(seed_count))
    random.shuffle(perm)
    colors = random.sample(list(INPUT_COLORS), seed_count)
    grid = np.zeros((h, w), dtype=int)
    for i, color in enumerate(colors):
        grid[rows[i], cols[perm[i]]] = color
    return grid


def _has_interior_descendant(info: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    for gen in info["generations"]:
        for r, c in gen["next_seeds"]:
            if 0 < r < h - 1 and 0 < c < w - 1:
                return True
    return False


def _semantic_checks(grid: np.ndarray, *, seed_count: int, require_deep: bool) -> bool:
    true_out, info = _simulate(grid, block_after_first=True, regime_mode="alternate")
    no_stop_out, _ = _simulate(
        grid,
        block_after_first=False,
        regime_mode="alternate",
        max_generations=4,
    )
    orth_only_out, _ = _simulate(
        grid,
        block_after_first=True,
        regime_mode="orth_only_after_first",
        max_generations=4,
    )
    diag_only_out, _ = _simulate(
        grid,
        block_after_first=True,
        regime_mode="diag_only_after_first",
        max_generations=4,
    )

    generations = info["generations"]
    if info["initial_seed_count"] != seed_count:
        return False
    if len(generations) < 2:
        return False
    if not generations[0]["next_seeds"]:
        return False
    if not _has_interior_descendant(info, grid.shape):
        return False
    if info["final_density"] < 0.08 or info["final_density"] > 0.5:
        return False
    if int(np.count_nonzero(true_out != grid)) < 10:
        return False
    if len(set(map(int, true_out.flatten())) - {BLACK}) < 3:
        return False
    if sum(gen["blocked_rays"] for gen in generations[1:]) == 0:
        return False
    if np.array_equal(true_out, no_stop_out):
        return False

    if require_deep:
        if info["generation_count"] < 3 or info["generation_count"] > 4:
            return False
        if not generations[1]["next_seeds"]:
            return False
        if generations[1]["child_color"] != BLUE:
            return False
        if np.array_equal(true_out, orth_only_out):
            return False
        if np.array_equal(true_out, diag_only_out):
            return False
    else:
        if info["generation_count"] != 2:
            return False
        if generations[1]["next_seeds"]:
            return False
        if np.array_equal(true_out, orth_only_out):
            return False

    return True


def generate_input() -> np.ndarray:
    seed_count, require_deep = _pick_scenario()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_seed_grid(seed_count, require_deep)
        if grid is None:
            continue
        if _semantic_checks(grid, seed_count=seed_count, require_deep=require_deep):
            return grid
    raise RuntimeError("failed to generate a valid cross-chain input")
