import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LOOKUP_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_RETRIES = 400
SCENARIOS = ("open", "obstacle", "competition", "mixed")

PATTERN_LIBRARY = {
    "right": ((0, 1),),
    "left": ((0, -1),),
    "down": ((1, 0),),
    "up": ((-1, 0),),
    "horiz": ((0, -1), (0, 1)),
    "vert": ((-1, 0), (1, 0)),
    "diag_main": ((-1, -1), (1, 1)),
    "diag_anti": ((-1, 1), (1, -1)),
    "arrow_r": ((0, 1), (-1, 1), (1, 1)),
    "arrow_l": ((0, -1), (-1, -1), (1, -1)),
    "arrow_u": ((-1, 0), (-1, -1), (-1, 1)),
    "arrow_d": ((1, 0), (1, -1), (1, 1)),
    "l_ne": ((-1, 0), (0, 1), (-1, 1)),
    "l_sw": ((1, 0), (0, -1), (1, -1)),
    "l_nw": ((-1, 0), (0, -1), (-1, -1)),
    "l_se": ((1, 0), (0, 1), (1, 1)),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a color lookup table and let matching seeds grow simultaneously until blocked.

    Input structure:
    - The background is black.
    - One full grey row splits the grid into an upper lookup area and a lower work area.
    - Each lookup entry is a 3x5 block: a single colored index pixel on the left, one black gap column, and a 3x3 pattern on the right whose center matches the index color.
    - The work area contains colored single-pixel seeds and optional maroon obstacle blocks.

    Rule:
    - Read each lookup entry as a color -> direction-set mapping, using the non-center colored cells of the 3x3 pattern as relative growth offsets.
    - In the work area, every seed grows in synchronous rounds: only the current frontier expands, and a frontier pixel proposes one new same-colored pixel for every offset in its lookup pattern.
    - A proposal fails if it leaves the work area, hits any non-black cell, or shares a target cell with a different color in the same round.
    - Keep iterating until no color can add a new pixel; leave the lookup area, separator row, and obstacle cells unchanged.

    Key insight:
    - The top half is a color-keyed motion legend, and the bottom half is a simultaneous competitive wavefront process rather than independent flood fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _find_separator_row(g)
    mapping = _extract_lookup(g, sep)
    return _simulate_growth(g, mapping, sep, with_stats=False)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]
    assert len(rows) == 1, "expected exactly one grey separator row"
    return int(rows[0])


def _extract_lookup(grid: np.ndarray, sep: int) -> dict[int, tuple[tuple[int, int], ...]]:
    upper = grid[:sep, :]
    upper_colors = {
        int(v) for v in np.unique(upper) if int(v) not in (BLACK, GREY)
    }
    assert upper_colors, "lookup area must contain at least one color"

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for r in range(max(0, sep - 2)):
        for c in range(max(0, grid.shape[1] - 4)):
            color = int(grid[r + 1, c])
            if color not in upper_colors or color == MAROON:
                continue
            window = grid[r : r + 3, c : c + 5]
            allowed = np.zeros((3, 5), dtype=bool)
            allowed[1, 0] = True
            allowed[:, 2:5] = True
            if bool(np.any(window[~allowed] != BLACK)):
                continue
            patch = window[:, 2:5]
            if int(patch[1, 1]) != color:
                continue
            vals = {int(v) for v in np.unique(patch)}
            if not vals.issubset({BLACK, color}):
                continue
            offsets = tuple(
                sorted(
                    (rr - 1, cc - 1)
                    for rr in range(3)
                    for cc in range(3)
                    if (rr, cc) != (1, 1) and int(patch[rr, cc]) == color
                )
            )
            if not offsets:
                continue
            if color in mapping:
                assert mapping[color] == offsets, "duplicate lookup color with inconsistent pattern"
            else:
                mapping[color] = offsets

    assert set(mapping) == upper_colors, "failed to recover every lookup entry"
    return mapping


def _simulate_growth(
    grid: np.ndarray,
    mapping: dict[int, tuple[tuple[int, int], ...]],
    sep: int,
    with_stats: bool,
):
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape

    frontier: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in range(sep + 1, h):
        for c in range(w):
            color = int(out[r, c])
            if color in (BLACK, GREY, MAROON):
                continue
            assert color in mapping, "work-area seed color missing from lookup table"
            frontier[color].add((r, c))

    assert frontier, "work area must contain at least one seed"

    stats = {
        "steps": 0,
        "total_added": 0,
        "conflict_cells": 0,
        "boundary_blocks": 0,
        "occupied_blocks": 0,
        "maroon_blocks": 0,
        "colors_with_growth": set(),
    }

    while frontier:
        proposals: dict[tuple[int, int], set[int]] = defaultdict(set)
        for color, cells in frontier.items():
            offsets = mapping[color]
            for r, c in cells:
                for dr, dc in offsets:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (sep + 1 <= rr < h and 0 <= cc < w):
                        stats["boundary_blocks"] += 1
                        continue
                    target = int(out[rr, cc])
                    if target != BLACK:
                        stats["occupied_blocks"] += 1
                        if target == MAROON:
                            stats["maroon_blocks"] += 1
                        continue
                    proposals[(rr, cc)].add(int(color))

        additions: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for cell, colors in proposals.items():
            if len(colors) == 1:
                color = next(iter(colors))
                additions[color].add(cell)
            else:
                stats["conflict_cells"] += 1

        if not additions:
            break

        for color, cells in additions.items():
            stats["colors_with_growth"].add(int(color))
            stats["total_added"] += len(cells)
            for r, c in cells:
                out[r, c] = int(color)

        frontier = {int(color): set(cells) for color, cells in additions.items() if cells}
        stats["steps"] += 1

    if with_stats:
        return out, stats
    return out


def _choose_lookup_spec(n_colors: int) -> dict[str, object]:
    for _ in range(100):
        table_rows = 1 if (n_colors <= 3 and random.random() < 0.55) else random.choice((1, 2))
        if table_rows == 1:
            row_counts = [n_colors]
        else:
            split = n_colors // 2
            row_counts = [split, n_colors - split]
            if random.random() < 0.5:
                row_counts.reverse()

        row_gaps = [random.randint(1, 2) for _ in row_counts]
        min_width = max(5 * count + gap * (count - 1) for count, gap in zip(row_counts, row_gaps))
        if min_width > 30:
            continue

        top_pad = random.randint(0, 1)
        row_gap = random.randint(1, 2) if len(row_counts) == 2 else 0
        row_starts = [top_pad]
        if len(row_counts) == 2:
            row_starts.append(top_pad + 3 + row_gap)

        return {
            "row_counts": row_counts,
            "row_gaps": row_gaps,
            "row_starts": row_starts,
            "top_h": row_starts[-1] + 3,
            "min_w": min_width,
        }
    raise RuntimeError("failed to sample a valid lookup layout")


def _lookup_positions(spec: dict[str, object], width: int) -> list[tuple[int, int]]:
    row_counts = spec["row_counts"]
    row_gaps = spec["row_gaps"]
    row_starts = spec["row_starts"]

    positions: list[tuple[int, int]] = []
    for count, gap, row in zip(row_counts, row_gaps, row_starts):
        required = 5 * count + gap * (count - 1)
        left = random.randint(0, width - required)
        cursor = left
        for _ in range(count):
            positions.append((int(row), int(cursor)))
            cursor += 5 + gap
    random.shuffle(positions)
    return positions


def _paint_lookup_entry(grid: np.ndarray, row: int, col: int, color: int, offsets: tuple[tuple[int, int], ...]) -> None:
    grid[row + 1, col] = int(color)
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = int(color)
    for dr, dc in offsets:
        patch[dr + 1, dc + 1] = int(color)
    grid[row : row + 3, col + 2 : col + 5] = patch


def _assign_patterns(colors: list[int], scenario: str):
    assigned: dict[int, tuple[tuple[int, int], ...]] = {}
    used_names: set[str] = set()
    meta: dict[str, object] = {}

    if scenario in ("competition", "mixed"):
        assigned[colors[0]] = PATTERN_LIBRARY["right"]
        assigned[colors[1]] = PATTERN_LIBRARY["left"]
        used_names.update({"right", "left"})
        meta["competition_colors"] = (colors[0], colors[1])

    if scenario == "obstacle":
        obstacle_name = random.choice(("right", "down"))
        obstacle_color = colors[0]
        assigned[obstacle_color] = PATTERN_LIBRARY[obstacle_name]
        used_names.add(obstacle_name)
        meta["obstacle"] = (obstacle_color, obstacle_name)
    elif scenario == "mixed":
        obstacle_name = "down"
        obstacle_color = colors[2]
        assigned[obstacle_color] = PATTERN_LIBRARY[obstacle_name]
        used_names.add(obstacle_name)
        meta["obstacle"] = (obstacle_color, obstacle_name)

    available = [name for name in PATTERN_LIBRARY if name not in used_names]
    random.shuffle(available)
    for color in colors:
        if color in assigned:
            continue
        assigned[color] = PATTERN_LIBRARY[available.pop()]
    return assigned, meta


def _region_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _mark(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _place_competition_pair(
    mask: np.ndarray, seeds: list[tuple[int, int, int]], color_a: int, color_b: int
) -> bool:
    h, w = mask.shape
    distances = [d for d in (4, 6, 8, 10, 12) if d < w - 2]
    if h < 5 or not distances:
        return False
    options = []
    for dist in distances:
        for row in range(1, h - 1):
            for c1 in range(1, w - dist - 1):
                c2 = c1 + dist
                options.append((row, c1, c2))
    random.shuffle(options)
    for row, c1, c2 in options:
        if not _region_clear(mask, row, c1, 1, 1, pad=1):
            continue
        if not _region_clear(mask, row, c2, 1, 1, pad=1):
            continue
        _mark(mask, row, c1, 1, 1)
        _mark(mask, row, c2, 1, 1)
        seeds.append((row, c1, color_a))
        seeds.append((row, c2, color_b))
        return True
    return False


def _place_obstacle_ray(
    mask: np.ndarray,
    seeds: list[tuple[int, int, int]],
    obstacles: list[tuple[int, int, int, int]],
    color: int,
    direction: str,
) -> bool:
    h, w = mask.shape
    if direction == "right":
        options = []
        for row in range(1, h - 1):
            for seed_c in range(1, max(2, w - 7)):
                for dist in (3, 4, 5, 6):
                    for obs_h in (1, 2, 3):
                        obs_r0_min = max(0, row - obs_h + 1)
                        obs_r0_max = min(row, h - obs_h)
                        for obs_r0 in range(obs_r0_min, obs_r0_max + 1):
                            for obs_w in (1, 2, 3):
                                obs_c0 = seed_c + dist
                                if obs_c0 + obs_w > w:
                                    continue
                                options.append((row, seed_c, obs_r0, obs_c0, obs_h, obs_w))
        random.shuffle(options)
        for row, seed_c, obs_r0, obs_c0, obs_h, obs_w in options:
            if not _region_clear(mask, row, seed_c, 1, 1, pad=1):
                continue
            if not _region_clear(mask, obs_r0, obs_c0, obs_h, obs_w, pad=1):
                continue
            _mark(mask, row, seed_c, 1, 1)
            _mark(mask, obs_r0, obs_c0, obs_h, obs_w)
            seeds.append((row, seed_c, color))
            obstacles.append((obs_r0, obs_c0, obs_h, obs_w))
            return True
        return False

    if direction == "down":
        options = []
        for col in range(1, w - 1):
            for seed_r in range(1, max(2, h - 7)):
                for dist in (3, 4, 5, 6):
                    for obs_w in (1, 2, 3):
                        obs_c0_min = max(0, col - obs_w + 1)
                        obs_c0_max = min(col, w - obs_w)
                        for obs_c0 in range(obs_c0_min, obs_c0_max + 1):
                            for obs_h in (1, 2, 3):
                                obs_r0 = seed_r + dist
                                if obs_r0 + obs_h > h:
                                    continue
                                options.append((seed_r, col, obs_r0, obs_c0, obs_h, obs_w))
        random.shuffle(options)
        for seed_r, col, obs_r0, obs_c0, obs_h, obs_w in options:
            if not _region_clear(mask, seed_r, col, 1, 1, pad=1):
                continue
            if not _region_clear(mask, obs_r0, obs_c0, obs_h, obs_w, pad=1):
                continue
            _mark(mask, seed_r, col, 1, 1)
            _mark(mask, obs_r0, obs_c0, obs_h, obs_w)
            seeds.append((seed_r, col, color))
            obstacles.append((obs_r0, obs_c0, obs_h, obs_w))
            return True
        return False

    return False


def _place_extra_seed(mask: np.ndarray, seeds: list[tuple[int, int, int]], color: int) -> bool:
    h, w = mask.shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if not _region_clear(mask, r, c, 1, 1, pad=1):
            continue
        _mark(mask, r, c, 1, 1)
        seeds.append((r, c, color))
        return True
    return False


def _paint_work_area(
    grid: np.ndarray,
    sep: int,
    seeds: list[tuple[int, int, int]],
    obstacles: list[tuple[int, int, int, int]],
) -> None:
    for r0, c0, h, w in obstacles:
        grid[sep + 1 + r0 : sep + 1 + r0 + h, c0 : c0 + w] = MAROON
    for r, c, color in seeds:
        grid[sep + 1 + r, c] = int(color)


def _scenario_accepts(scenario: str, stats: dict[str, object], work_shape: tuple[int, int]) -> bool:
    area = work_shape[0] * work_shape[1]
    if stats["steps"] < 2:
        return False
    if stats["total_added"] < 4:
        return False
    if stats["total_added"] > int(area * 0.55):
        return False
    if len(stats["colors_with_growth"]) < 2:
        return False

    maroon_hits = int(stats["maroon_blocks"])
    conflicts = int(stats["conflict_cells"])
    if scenario == "open":
        return maroon_hits == 0 and conflicts == 0
    if scenario == "obstacle":
        return maroon_hits > 0 and conflicts == 0
    if scenario == "competition":
        return conflicts > 0 and maroon_hits == 0
    return maroon_hits > 0 and conflicts > 0


def _build_attempt(scenario: str) -> np.ndarray | None:
    if scenario == "mixed":
        n_colors = random.randint(4, 5)
    else:
        n_colors = random.randint(3, 5)
    colors = random.sample(list(LOOKUP_COLORS), n_colors)
    pattern_map, meta = _assign_patterns(colors, scenario)

    spec = _choose_lookup_spec(n_colors)
    top_h = int(spec["top_h"])
    min_w = int(spec["min_w"])

    work_h_low = 10 if scenario != "competition" else 9
    work_h_high = 18 if scenario == "open" else 16
    work_w_low = 14 if scenario != "mixed" else 16
    work_w_high = 24 if scenario == "mixed" else 22

    work_h = random.randint(work_h_low, work_h_high)
    width_low = max(work_w_low, min_w)
    width_high = max(width_low, min(30, work_w_high))
    work_w = random.randint(width_low, width_high)
    total_h = top_h + 1 + work_h
    if total_h > 30:
        return None

    grid = np.zeros((total_h, work_w), dtype=int)
    sep = top_h
    grid[sep, :] = GREY

    positions = _lookup_positions(spec, work_w)
    shuffled_colors = colors[:]
    random.shuffle(shuffled_colors)
    for color, (row, col) in zip(shuffled_colors, positions):
        _paint_lookup_entry(grid, row, col, color, pattern_map[color])

    mask = np.zeros((work_h, work_w), dtype=bool)
    seeds: list[tuple[int, int, int]] = []
    obstacles: list[tuple[int, int, int, int]] = []

    if scenario in ("competition", "mixed"):
        color_a, color_b = meta["competition_colors"]
        if not _place_competition_pair(mask, seeds, color_a, color_b):
            return None

    if scenario in ("obstacle", "mixed"):
        obstacle_color, direction = meta["obstacle"]
        if not _place_obstacle_ray(mask, seeds, obstacles, obstacle_color, direction):
            return None

    for color in colors:
        if any(seed_color == color for _r, _c, seed_color in seeds):
            continue
        if not _place_extra_seed(mask, seeds, color):
            return None

    _paint_work_area(grid, sep, seeds, obstacles)

    try:
        recovered = _extract_lookup(grid, sep)
        assert recovered == {
            int(color): tuple(sorted(pattern_map[color])) for color in pattern_map
        }
        out, stats = _simulate_growth(grid, recovered, sep, with_stats=True)
        if np.array_equal(out, grid):
            return None
        if not _scenario_accepts(scenario, stats, (work_h, work_w)):
            return None
        solved = main(grid)
        if not np.array_equal(solved, out):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        scenario = random.choice(SCENARIOS)
        grid = _build_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid growth-table input")
