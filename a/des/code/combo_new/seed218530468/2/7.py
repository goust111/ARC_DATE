import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_bfs(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    towers: list[dict] = []
    samples: list[dict] = []

    for color in map(int, np.unique(grid)):
        if color in (BG, GREY):
            continue
        for coords in _component_bfs(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if len(coords) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
                if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                    towers.append({"color": color, "top": r0, "left": c0})
                    continue
            if len(coords) == 1:
                rr, cc = map(int, coords[0])
                samples.append({"color": color, "pos": (rr, cc)})
                continue
            raise AssertionError("expected only 2x2 towers and 1x1 mix samples")

    tower_colors = [tower["color"] for tower in towers]
    assert towers, "expected at least one tower"
    assert len(set(tower_colors)) == len(towers), "tower colors must be distinct"
    assert all(sample["color"] not in tower_colors for sample in samples), "mix colors must differ from tower colors"
    return towers, samples


def _tower_cell_map(towers: list[dict]) -> dict[tuple[int, int], dict]:
    mapping: dict[tuple[int, int], dict] = {}
    for tower in towers:
        r0, c0 = tower["top"], tower["left"]
        for rr in range(r0, r0 + 2):
            for cc in range(c0, c0 + 2):
                mapping[(rr, cc)] = tower
    return mapping


def _first_tower_in_direction(
    grid: np.ndarray,
    tower_by_cell: dict[tuple[int, int], dict],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> dict | None:
    while _in_bounds(grid.shape, r, c):
        value = int(grid[r, c])
        if value in (BG, GREY):
            r += dr
            c += dc
            continue
        return tower_by_cell.get((r, c))
    return None


def _build_mix_lookup(grid: np.ndarray, towers: list[dict], samples: list[dict]) -> dict[tuple[int, int], int]:
    tower_by_cell = _tower_cell_map(towers)
    lookup: dict[tuple[int, int], int] = {}

    for sample in samples:
        r, c = sample["pos"]
        pair_candidates: list[tuple[int, int]] = []

        left = _first_tower_in_direction(grid, tower_by_cell, r, c - 1, 0, -1)
        right = _first_tower_in_direction(grid, tower_by_cell, r, c + 1, 0, 1)
        if left is not None and right is not None and left["color"] != right["color"]:
            pair_candidates.append(tuple(sorted((left["color"], right["color"]))))

        up = _first_tower_in_direction(grid, tower_by_cell, r - 1, c, -1, 0)
        down = _first_tower_in_direction(grid, tower_by_cell, r + 1, c, 1, 0)
        if up is not None and down is not None and up["color"] != down["color"]:
            pair_candidates.append(tuple(sorted((up["color"], down["color"]))))

        pair_candidates = list(dict.fromkeys(pair_candidates))
        assert len(pair_candidates) == 1, "each mix sample must identify exactly one tower pair"

        pair = pair_candidates[0]
        assert pair not in lookup, "each tower pair may have at most one mix sample"
        lookup[pair] = sample["color"]

    return lookup


def _ray_starts(tower: dict) -> list[tuple[int, int, int, int]]:
    r0, c0 = tower["top"], tower["left"]
    r1, c1 = r0 + 1, c0 + 1
    return [
        (r0 - 1, c0, -1, 0),
        (r0 - 1, c1, -1, 0),
        (r1 + 1, c0, 1, 0),
        (r1 + 1, c1, 1, 0),
        (r0, c0 - 1, 0, -1),
        (r1, c0 - 1, 0, -1),
        (r0, c1 + 1, 0, 1),
        (r1, c1 + 1, 0, 1),
    ]


def _trace_coverages(grid: np.ndarray, towers: list[dict]) -> dict[tuple[int, int], set[int]]:
    coverages: dict[tuple[int, int], set[int]] = defaultdict(set)
    for tower in towers:
        color = tower["color"]
        for r, c, dr, dc in _ray_starts(tower):
            while _in_bounds(grid.shape, r, c):
                value = int(grid[r, c])
                if value in (BG, GREY):
                    coverages[(r, c)].add(color)
                    r += dr
                    c += dc
                    continue
                break
    return coverages


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray receivers according to which 2x2 signal towers can reach them along unobstructed cardinal rays.

    Input structure:
    - The background is black.
    - Each signal tower is a distinct-colored solid 2x2 block.
    - Gray 1x1 pixels are receivers, and other non-black non-gray 1x1 pixels are mix samples.

    Rule:
    - Identify all 2x2 towers and all gray receivers.
    - From each tower, cast rays outward from the two cells on every side; rays pass through black and gray cells, but stop at the grid edge or at the first non-black non-gray pixel.
    - A receiver reached by exactly one tower becomes that tower's color; a receiver reached by exactly two different towers becomes the color of the sample pixel lying between that same color pair; untouched receivers stay gray.

    Key insight:
    The singleton colored samples encode the tower-pair mixing table inside the grid itself, so solving the task requires combining ray tracing with a lookup inferred from the sample placements.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    towers, samples = _parse_scene(g)
    mix_lookup = _build_mix_lookup(g, towers, samples)
    coverages = _trace_coverages(g, towers)

    out = g.copy()
    for r, c in np.argwhere(g == GREY):
        colors = sorted(coverages.get((int(r), int(c)), set()))
        if len(colors) == 1:
            out[r, c] = colors[0]
        elif len(colors) == 2:
            pair = tuple(colors)
            assert pair in mix_lookup, "missing mix sample for a doubly covered receiver"
            out[r, c] = mix_lookup[pair]
        else:
            assert len(colors) <= 2, "generator should avoid three-way overlaps"
    return out


def _place_tower(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 2, left : left + 2] = color


def _pair_canvas() -> tuple[np.ndarray, set[int], set[int]]:
    orientation = random.choice(("H", "V"))
    gap = random.randint(4, 10)
    tower_colors = set(random.sample(PALETTE, 2))
    mix_color = next(color for color in random.sample(PALETTE, len(PALETTE)) if color not in tower_colors)
    color_a, color_b = random.sample(tuple(tower_colors), 2)

    core_h, core_w = (2, gap + 4) if orientation == "H" else (gap + 4, 2)
    height = random.randint(core_h + 4, min(30, core_h + 14))
    width = random.randint(core_w + 4, min(30, core_w + 14))
    top = random.randint(2, height - core_h - 2)
    left = random.randint(2, width - core_w - 2)

    grid = np.full((height, width), BG, dtype=int)
    if orientation == "H":
        a_top, a_left = top, left
        b_top, b_left = top, left + gap + 2
        _place_tower(grid, a_top, a_left, color_a)
        _place_tower(grid, b_top, b_left, color_b)
        sample_r = top + random.randint(0, 1)
        sample_c = random.randint(a_left + 3, b_left - 2)
        grid[sample_r, sample_c] = mix_color
    else:
        a_top, a_left = top, left
        b_top, b_left = top + gap + 2, left
        _place_tower(grid, a_top, a_left, color_a)
        _place_tower(grid, b_top, b_left, color_b)
        sample_r = random.randint(a_top + 3, b_top - 2)
        sample_c = left + random.randint(0, 1)
        grid[sample_r, sample_c] = mix_color

    return grid, {color_a, color_b}, {mix_color}


def _l_canvas() -> tuple[np.ndarray, set[int], set[int]]:
    gap_h = random.randint(4, 8)
    gap_v = random.randint(4, 8)
    color_a, color_b, color_c, mix_ab, mix_bc = random.sample(PALETTE, 5)

    core_h, core_w = gap_v + 4, gap_h + 4
    height = random.randint(core_h + 4, min(30, core_h + 14))
    width = random.randint(core_w + 4, min(30, core_w + 14))
    top = random.randint(2, height - core_h - 2)
    left = random.randint(2, width - core_w - 2)
    a_on_left = random.choice((True, False))
    c_below = random.choice((True, False))

    b_top = top if c_below else top + gap_v + 2
    c_top = top + gap_v + 2 if c_below else top
    a_left = left if a_on_left else left + gap_h + 2
    b_left = left + gap_h + 2 if a_on_left else left

    grid = np.full((height, width), BG, dtype=int)
    _place_tower(grid, b_top, b_left, color_b)
    _place_tower(grid, b_top, a_left, color_a)
    _place_tower(grid, c_top, b_left, color_c)

    gap_left = min(a_left, b_left) + 2
    gap_right = max(a_left, b_left) - 1
    sample_ab_r = b_top + random.randint(0, 1)
    sample_ab_c = random.randint(gap_left + 1, gap_right - 1)
    grid[sample_ab_r, sample_ab_c] = mix_ab

    gap_top = min(b_top, c_top) + 2
    gap_bottom = max(b_top, c_top) - 1
    sample_bc_r = random.randint(gap_top + 1, gap_bottom - 1)
    sample_bc_c = b_left + random.randint(0, 1)
    grid[sample_bc_r, sample_bc_c] = mix_bc

    return grid, {color_a, color_b, color_c}, {mix_ab, mix_bc}


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _pick_receivers(
    candidates: list[tuple[int, int]],
    minimum: int,
    maximum: int,
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if len(candidates) < minimum:
        return None
    target = random.randint(minimum, min(maximum, len(candidates)))
    pool = candidates[:]
    random.shuffle(pool)
    picked: list[tuple[int, int]] = []
    for cell in pool:
        if cell in chosen or any(_adjacent(cell, other) for other in chosen):
            continue
        if any(_adjacent(cell, other) for other in picked):
            continue
        picked.append(cell)
        if len(picked) == target:
            break
    if len(picked) < minimum:
        return None
    return picked


def _add_receivers(
    grid: np.ndarray,
    tower_colors: set[int],
    mix_colors: set[int],
) -> np.ndarray | None:
    towers, samples = _parse_scene(grid)
    mix_lookup = _build_mix_lookup(grid, towers, samples)
    coverages = _trace_coverages(grid, towers)

    double_cells: list[tuple[int, int]] = []
    single_cells: list[tuple[int, int]] = []
    none_cells: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != BG:
                continue
            colors = tuple(sorted(coverages.get((r, c), set())))
            degree = len(colors)
            if degree == 2 and colors in mix_lookup:
                double_cells.append((r, c))
            elif degree == 1:
                single_cells.append((r, c))
            elif degree == 0:
                none_cells.append((r, c))

    chosen: list[tuple[int, int]] = []
    picked_double = _pick_receivers(double_cells, 1, 3, chosen)
    if picked_double is None:
        return None
    chosen.extend(picked_double)

    picked_single = _pick_receivers(single_cells, 2, 5, chosen)
    if picked_single is None:
        return None
    chosen.extend(picked_single)

    picked_none = _pick_receivers(none_cells, 1, 3, chosen)
    if picked_none is None:
        return None
    chosen.extend(picked_none)

    result = grid.copy()
    for r, c in chosen:
        result[r, c] = GREY

    out = main(result)
    receiver_values = [int(out[r, c]) for r, c in chosen]
    if not any(value in tower_colors for value in receiver_values):
        return None
    if not any(value in mix_colors for value in receiver_values):
        return None
    if not any(value == GREY for value in receiver_values):
        return None
    return result


def _gen_attempt() -> np.ndarray | None:
    base, tower_colors, mix_colors = (_pair_canvas() if random.random() < 0.5 else _l_canvas())
    return _add_receivers(base, tower_colors, mix_colors)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is None:
            continue
        try:
            output = main(candidate)
        except Exception:
            continue
        if output.shape != candidate.shape:
            continue
        if np.array_equal(output, candidate):
            continue
        return candidate
    raise RuntimeError("failed to generate a valid signal-tower puzzle input")
