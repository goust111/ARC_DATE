from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
COLLISION = Color.GREY
SEED_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE)

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
SEED_ROW_RANGE = (0, 4)
MAX_GENERATION_ATTEMPTS = 400

TEMPLATES_BY_COUNT = {
    3: ("pair_iso_far", "pair_safe_adj"),
    4: ("pair_two_iso", "chain_iso", "pair_safe_adj_iso"),
}

RELATIVE_COLUMNS = {
    "pair_iso_far": (0, 2, 7),
    "pair_safe_adj": (0, 2, 4),
    "pair_two_iso": (0, 2, 7, 10),
    "chain_iso": (0, 2, 4, 9),
    "pair_safe_adj_iso": (0, 2, 4, 9),
}

TEMPLATE_SEED_COUNT = {
    template: seed_count
    for seed_count, templates in TEMPLATES_BY_COUNT.items()
    for template in templates
}

_TEMPLATE_CYCLE: list[str] = []


def _seed_rows_for_parity(parity: int) -> list[int]:
    lo, hi = SEED_ROW_RANGE
    return [row for row in range(lo, hi + 1) if row % 2 == parity]


def _next_template() -> str:
    global _TEMPLATE_CYCLE
    if not _TEMPLATE_CYCLE:
        _TEMPLATE_CYCLE = list(TEMPLATE_SEED_COUNT)
        random.shuffle(_TEMPLATE_CYCLE)
    return _TEMPLATE_CYCLE.pop()


def _extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected black background"

    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 3 <= len(components) <= 4, "expected three or four isolated seeds"

    seeds: list[tuple[int, int, int]] = []
    for comp in components:
        assert int(comp["area"]) == 1, "each foreground component must be a single seed pixel"
        coords = np.asarray(comp["coords"], dtype=int)
        r, c = map(int, coords[0])
        color = int(comp["color"])
        assert color != BG, "seed color must be non-background"
        seeds.append((r, c, color))

    colors = [color for _r, _c, color in seeds]
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    return sorted(seeds)


def _trail_coords(seed: tuple[int, int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, _color = seed
    coords: set[tuple[int, int]] = set()
    for start_r, start_c in (
        (r0, c0),
        (r0 - 1, c0 - 1),
        (r0 - 1, c0 + 1),
        (r0 + 1, c0 - 1),
        (r0 + 1, c0 + 1),
    ):
        r = int(start_r)
        c = int(start_c)
        while 0 <= r < h and 0 <= c < w:
            coords.add((r, c))
            r += 2
    return coords


def _analyze_case(
    seeds: list[tuple[int, int, int]], shape: tuple[int, int]
) -> dict[str, object]:
    owners: dict[tuple[int, int], list[int]] = {}
    seed_trails: list[set[tuple[int, int]]] = []
    for idx, seed in enumerate(seeds):
        trail = _trail_coords(seed, shape)
        seed_trails.append(trail)
        for pos in trail:
            owners.setdefault(pos, []).append(idx)

    collision_positions = {pos for pos, idxs in owners.items() if len(idxs) > 1}
    per_seed: list[dict[str, object]] = []
    for idx, seed in enumerate(seeds):
        seed_r, seed_c, color = seed
        survivor_coords = sorted(
            [pos for pos in seed_trails[idx] if pos not in collision_positions],
            key=lambda pos: (
                abs(pos[0] - seed_r) + abs(pos[1] - seed_c),
                pos[0],
                pos[1],
            ),
        )
        collision_count = int(sum(pos in collision_positions for pos in seed_trails[idx]))
        keep_step = collision_count + 1
        kept_coords = [pos for order, pos in enumerate(survivor_coords) if order % keep_step == 0]
        per_seed.append(
            {
                "seed": (seed_r, seed_c),
                "color": color,
                "trail_coords": seed_trails[idx],
                "survivor_coords": survivor_coords,
                "collision_count": collision_count,
                "kept_coords": kept_coords,
            }
        )

    return {
        "collision_positions": collision_positions,
        "per_seed": per_seed,
    }


def _render_output(shape: tuple[int, int], analysis: dict[str, object]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    per_seed = analysis["per_seed"]
    assert isinstance(per_seed, list)
    for info in per_seed:
        color = int(info["color"])
        for r, c in info["kept_coords"]:
            out[int(r), int(c)] = color

    for r, c in analysis["collision_positions"]:
        out[int(r), int(c)] = COLLISION

    for info in per_seed:
        seed_r, seed_c = info["seed"]
        out[int(seed_r), int(seed_c)] = int(info["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each top seed into five period-2 downward trails, turn inter-seed overlaps grey, and sparsify each surviving trail by its own collision count.

    Input structure:
    - The background is black.
    - Three or four isolated colored seed pixels appear in the upper half of the canvas.
    - Seed colors are distinct and each seed is the only input pixel of its color.

    Rule:
    - For each seed at (r,c), generate period-2 downward trails from (r,c), (r-1,c-1), (r-1,c+1), (r+1,c-1), and (r+1,c+1); each trail keeps the same column and adds pixels every two rows while in bounds.
    - Any pixel reached by trails from two or more different seeds becomes grey and counts as one collision pixel for every seed that reached it.
    - For each seed, sort its non-collision trail pixels by Manhattan distance from the seed, breaking ties by row then column, and keep only indices divisible by C+1 where C is that seed's collision count.
    - Keep every seed pixel and every grey collision pixel.

    Key insight:
    - Collision pixels do not just erase overlaps: the number of grey collisions attached to a seed becomes the spacing parameter that thins out the rest of that seed's surviving trail.
    """
    g = np.asarray(grid, dtype=int)
    seeds = _extract_seeds(g)
    analysis = _analyze_case(seeds, g.shape)
    return _render_output(g.shape, analysis)


def _embed_relative_columns(rel_cols: tuple[int, ...], width: int) -> list[int]:
    shifted = [int(col - min(rel_cols)) for col in rel_cols]
    span = max(shifted)
    if random.random() < 0.5:
        shifted = [span - col for col in shifted]
    min_col = 1
    max_shift = width - 2 - span
    shift = random.randint(min_col, max_shift)
    return [shift + col for col in shifted]


def _seed_rows_for_template(name: str) -> list[int]:
    base_parity = random.randint(0, 1)
    same = _seed_rows_for_parity(base_parity)
    other = _seed_rows_for_parity(1 - base_parity)

    if name == "pair_iso_far":
        return [random.choice(same), random.choice(same), random.randint(*SEED_ROW_RANGE)]
    if name == "pair_safe_adj":
        return [random.choice(same), random.choice(same), random.choice(other)]
    if name == "pair_two_iso":
        return [
            random.choice(same),
            random.choice(same),
            random.randint(*SEED_ROW_RANGE),
            random.randint(*SEED_ROW_RANGE),
        ]
    if name == "chain_iso":
        return [random.choice(same), random.choice(same), random.choice(same), random.randint(*SEED_ROW_RANGE)]
    if name == "pair_safe_adj_iso":
        return [
            random.choice(same),
            random.choice(same),
            random.choice(other),
            random.randint(*SEED_ROW_RANGE),
        ]
    raise ValueError(f"unknown template: {name}")


def _build_seed_specs(template: str, width: int) -> list[tuple[int, int, int]]:
    cols = _embed_relative_columns(RELATIVE_COLUMNS[template], width)
    rows = _seed_rows_for_template(template)
    colors = random.sample(SEED_COLORS, len(cols))
    seeds = [(int(rows[i]), int(cols[i]), int(colors[i])) for i in range(len(cols))]
    assert len({(r, c) for r, c, _color in seeds}) == len(seeds), "seed positions must be unique"
    return seeds


def _is_semantically_valid(
    grid: np.ndarray,
    seeds: list[tuple[int, int, int]],
    analysis: dict[str, object],
    output: np.ndarray,
) -> bool:
    per_seed = analysis["per_seed"]
    assert isinstance(per_seed, list)
    collision_positions = set(analysis["collision_positions"])

    counts = [int(info["collision_count"]) for info in per_seed]
    if sum(count > 0 for count in counts) < 2:
        return False
    if sum(count == 0 for count in counts) < 1:
        return False
    if not collision_positions:
        return False

    grey_positions = {tuple(map(int, rc)) for rc in np.argwhere(output == COLLISION)}
    if grey_positions != collision_positions:
        return False

    for i, (r0, c0, _color0) in enumerate(seeds):
        for r1, c1, _color1 in seeds[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    for info in per_seed:
        color = int(info["color"])
        seed_r, seed_c = map(int, info["seed"])
        expected_positions = set(info["kept_coords"])
        actual_positions = {tuple(map(int, rc)) for rc in np.argwhere(output == color)}
        if actual_positions != expected_positions:
            return False
        if int(output[seed_r, seed_c]) != color:
            return False

        survivor_count = len(info["survivor_coords"])
        kept_count = len(info["kept_coords"])
        collision_count = int(info["collision_count"])
        if collision_count == 0 and kept_count != survivor_count:
            return False
        if collision_count > 0 and kept_count >= survivor_count:
            return False

    if np.array_equal(grid, output):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    template = _next_template()
    seed_count = TEMPLATE_SEED_COUNT[template]
    min_width = max(GRID_WIDTH_RANGE[0], max(RELATIVE_COLUMNS[template]) - min(RELATIVE_COLUMNS[template]) + 3)
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])

    seeds = _build_seed_specs(template, width)
    grid = np.full((height, width), BG, dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color

    analysis = _analyze_case(seeds, grid.shape)
    expected = _render_output(grid.shape, analysis)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not _is_semantically_valid(grid, seeds, analysis, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-collision sparsification case")
