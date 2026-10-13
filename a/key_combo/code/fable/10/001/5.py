from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
TERRAIN_CHOICES = (1, 5, 6, 7, 8, 9)
SEED_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (13, 20)
BASIN_COUNT_CHOICES = (2, 3, 4)
MAX_ATTEMPTS = 500


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _infer_terrain_color(grid: np.ndarray) -> int:
    bottom = np.unique(grid[-1, :])
    assert bottom.size == 1, "bottom row must be solid terrain"
    return int(bottom[0])


def _infer_bg_color(grid: np.ndarray, terrain: int) -> int:
    non_terrain = grid[grid != terrain]
    assert non_terrain.size > 0, "expected air above the terrain"
    return _most_common_color(non_terrain)


def _terrain_starts(grid: np.ndarray, terrain: int) -> np.ndarray:
    h, w = grid.shape
    starts = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == terrain)
        assert rows.size > 0 and int(rows[-1]) == h - 1, "terrain must reach the bottom in every column"
        start = int(rows[0])
        assert np.all(grid[start:, c] == terrain), "terrain must be a contiguous suffix"
        assert not np.any(grid[:start, c] == terrain), "terrain cannot float above air"
        starts[c] = start
    return starts


def _water_levels(heights: np.ndarray) -> np.ndarray:
    left = np.maximum.accumulate(heights)
    right = np.maximum.accumulate(heights[::-1])[::-1]
    return np.minimum(left, right)


def _basins_from_heights(heights: np.ndarray, grid_height: int) -> list[dict[str, int]]:
    levels = _water_levels(heights)
    basins: list[dict[str, int]] = []
    i = 0
    while i < len(heights):
        if int(levels[i]) > int(heights[i]):
            water_h = int(levels[i])
            j = i
            while (
                j + 1 < len(heights)
                and int(levels[j + 1]) == water_h
                and int(levels[j + 1]) > int(heights[j + 1])
            ):
                j += 1
            basins.append(
                {
                    "l": int(i),
                    "r": int(j),
                    "water_h": water_h,
                    "surface_r": int(grid_height - water_h),
                }
            )
            i = j + 1
        else:
            i += 1
    return basins


def _seed_pixels(grid: np.ndarray, bg: int, terrain: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != bg) & (grid != terrain))
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert len({color for _r, _c, color in seeds}) == len(seeds), "seed colors must be distinct"
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the seeded terrain basins up to the lower of their spill rim and seed row.

    Input structure:
    - The lower part of the grid is one solid terrain color, with each column made of air above a contiguous terrain suffix.
    - The terrain contour forms one or more bowl-like basins between higher rim columns.
    - Distinct single-pixel seeds float in the air over specific basins; there is at most one seed per basin.

    Rule:
    - Compute each basin's natural water height from the lower of its two enclosing rims.
    - If the basin's seed sits below that natural surface, lower the water surface to the seed row instead.
    - Recolor every non-terrain cell inside that basin from the final surface row downward with the seed color.
    - Remove all seed pixels as standalone markers; unseeded basins stay dry and the terrain itself is unchanged.

    Key insight:
    - The seed does not choose the basin shape, only whether that basin fills and whether its row caps the otherwise rim-limited water surface.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    terrain = _infer_terrain_color(g)
    bg = _infer_bg_color(g, terrain)
    starts = _terrain_starts(g, terrain)
    heights = g.shape[0] - starts
    basins = _basins_from_heights(heights, g.shape[0])
    seeds = _seed_pixels(g, bg, terrain)
    assert seeds, "expected at least one seed"

    out = g.copy()
    out[(out != bg) & (out != terrain)] = bg

    assigned: list[tuple[int, int, int] | None] = [None] * len(basins)
    for seed_r, seed_c, seed_color in seeds:
        matches = [
            basin_i
            for basin_i, basin in enumerate(basins)
            if basin["l"] <= seed_c <= basin["r"] and seed_r < int(starts[seed_c])
        ]
        assert len(matches) == 1, "each seed must belong to exactly one basin"
        basin_i = matches[0]
        assert assigned[basin_i] is None, "expected at most one seed per basin"
        assigned[basin_i] = (seed_r, seed_c, seed_color)

    for basin, seed in zip(basins, assigned):
        if seed is None:
            continue
        seed_r, _seed_c, seed_color = seed
        fill_top = max(int(basin["surface_r"]), int(seed_r))
        for c in range(int(basin["l"]), int(basin["r"]) + 1):
            if fill_top < int(starts[c]):
                out[fill_top : int(starts[c]), c] = seed_color

    return out


def _monotone_segment(start: int, end: int, steps: int) -> list[int]:
    assert steps >= 1
    raw = np.rint(np.linspace(start, end, steps + 1)[1:]).astype(int).tolist()
    out: list[int] = []
    prev = int(start)
    if end >= start:
        for value in raw:
            prev = max(prev, int(value))
            out.append(prev)
        out[-1] = int(end)
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1])
    else:
        for value in raw:
            prev = min(prev, int(value))
            out.append(prev)
        out[-1] = int(end)
        for i in range(len(out) - 2, -1, -1):
            out[i] = max(out[i], out[i + 1])
    return out


def _sample_profile() -> tuple[int, np.ndarray, list[dict[str, int]]] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    basin_count = random.choice(BASIN_COUNT_CHOICES)
    deltas = [random.randint(1, 2) for _ in range(basin_count)]
    span = sum(deltas)

    peak_lo = max(5, h // 2 - 1)
    peak_hi = h - 4
    if peak_hi - peak_lo < span:
        return None

    first_peak = random.randint(peak_lo, peak_hi - span)
    peaks = [first_peak]
    for delta in deltas:
        peaks.append(peaks[-1] + delta)
    if random.random() < 0.5:
        peaks = peaks[::-1]

    heights = [peaks[0]] * random.randint(1, 2)
    for basin_i in range(basin_count):
        water_h = min(peaks[basin_i], peaks[basin_i + 1])
        if water_h < 5:
            return None
        valley = random.randint(2, water_h - 2)
        down_len = random.randint(2, 4)
        flat_len = random.randint(1, 3)
        up_len = random.randint(2, 4)
        rim_hold = random.randint(1, 2)

        heights.extend(_monotone_segment(heights[-1], valley, down_len))
        heights.extend([valley] * (flat_len - 1))
        heights.extend(_monotone_segment(heights[-1], peaks[basin_i + 1], up_len))
        heights.extend([peaks[basin_i + 1]] * rim_hold)

    if not (12 <= len(heights) <= 30):
        return None

    heights_arr = np.array(heights, dtype=int)
    basins = _basins_from_heights(heights_arr, h)
    if len(basins) != basin_count:
        return None
    if len({basin["water_h"] for basin in basins}) == 1 and basin_count >= 3:
        return None
    return h, heights_arr, basins


def _low_cap_columns(basin: dict[str, int], starts: np.ndarray) -> list[int]:
    surface_r = int(basin["surface_r"])
    return [c for c in range(int(basin["l"]), int(basin["r"]) + 1) if int(starts[c]) >= surface_r + 2]


def _pick_seed_plan(
    basins: list[dict[str, int]], starts: np.ndarray, terrain: int
) -> list[dict[str, int | str]] | None:
    basin_count = len(basins)
    if basin_count == 2:
        seeded_count = 1
    elif basin_count == 3:
        seeded_count = random.choice((1, 2))
    else:
        seeded_count = random.choice((2, 3))

    seeded_idxs = sorted(random.sample(range(basin_count), seeded_count))
    seed_plan: list[dict[str, int | str]] = []

    forced_low = None
    forced_high = None
    low_ready = [idx for idx in seeded_idxs if _low_cap_columns(basins[idx], starts)]
    high_ready = [idx for idx in seeded_idxs if int(basins[idx]["surface_r"]) >= 1]
    if seeded_count >= 2 and low_ready and high_ready and random.random() < 0.7:
        forced_low = random.choice(low_ready)
        high_options = [idx for idx in high_ready if idx != forced_low] or high_ready
        forced_high = random.choice(high_options)

    used_colors = set()
    for basin_idx in seeded_idxs:
        basin = basins[basin_idx]
        low_cols = _low_cap_columns(basin, starts)
        modes: list[str] = []
        if int(basin["surface_r"]) >= 1:
            modes.append("high")
        if low_cols:
            modes.append("low")
        if not modes:
            return None

        if basin_idx == forced_low:
            mode = "low"
        elif basin_idx == forced_high:
            mode = "high"
        else:
            mode = random.choice(modes)

        if mode == "low":
            seed_c = random.choice(low_cols)
            seed_r = random.randint(int(basin["surface_r"]) + 1, int(starts[seed_c]) - 1)
        else:
            seed_c = random.randint(int(basin["l"]), int(basin["r"]))
            seed_r = random.randint(0, int(basin["surface_r"]) - 1)

        color_choices = [color for color in SEED_PALETTE if color not in used_colors and color != terrain]
        if not color_choices:
            return None
        seed_color = random.choice(color_choices)
        used_colors.add(seed_color)
        seed_plan.append(
            {
                "basin_idx": basin_idx,
                "r": seed_r,
                "c": seed_c,
                "color": seed_color,
                "mode": mode,
            }
        )

    return seed_plan


def _render_input(height: int, heights: np.ndarray, terrain: int, seed_plan: list[dict[str, int | str]]) -> np.ndarray:
    width = len(heights)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    starts = height - heights
    for c in range(width):
        grid[int(starts[c]) :, c] = terrain
    for seed in seed_plan:
        r = int(seed["r"])
        c = int(seed["c"])
        color = int(seed["color"])
        assert grid[r, c] == BG_COLOR, "seed must occupy empty air"
        grid[r, c] = color
    return grid


def _expected_from_plan(
    grid: np.ndarray,
    terrain: int,
    basins: list[dict[str, int]],
    seed_plan: list[dict[str, int | str]],
) -> np.ndarray:
    starts = _terrain_starts(grid, terrain)
    out = grid.copy()
    out[(out != BG_COLOR) & (out != terrain)] = BG_COLOR
    for seed in seed_plan:
        basin = basins[int(seed["basin_idx"])]
        fill_top = max(int(basin["surface_r"]), int(seed["r"]))
        color = int(seed["color"])
        for c in range(int(basin["l"]), int(basin["r"]) + 1):
            if fill_top < int(starts[c]):
                out[fill_top : int(starts[c]), c] = color
    return out


def _validate_semantics(
    grid: np.ndarray,
    terrain: int,
    heights: np.ndarray,
    basins: list[dict[str, int]],
    seed_plan: list[dict[str, int | str]],
    expected: np.ndarray,
) -> bool:
    starts = _terrain_starts(grid, terrain)
    redetected = _basins_from_heights(heights, grid.shape[0])
    if redetected != basins:
        return False

    seeded_idxs = {int(seed["basin_idx"]) for seed in seed_plan}
    if not seeded_idxs or len(seeded_idxs) >= len(basins):
        return False

    any_low = False
    natural_only = grid.copy()
    natural_only[(natural_only != BG_COLOR) & (natural_only != terrain)] = BG_COLOR

    for seed in seed_plan:
        basin_idx = int(seed["basin_idx"])
        basin = basins[basin_idx]
        seed_r = int(seed["r"])
        seed_c = int(seed["c"])
        seed_color = int(seed["color"])
        mode = str(seed["mode"])

        matches = [
            idx
            for idx, candidate in enumerate(basins)
            if candidate["l"] <= seed_c <= candidate["r"] and seed_r < int(starts[seed_c])
        ]
        if matches != [basin_idx]:
            return False

        for c in range(int(basin["l"]), int(basin["r"]) + 1):
            if int(basin["surface_r"]) < int(starts[c]):
                natural_only[int(basin["surface_r"]) : int(starts[c]), c] = seed_color

        if mode == "high":
            if seed_r >= int(basin["surface_r"]) or expected[seed_r, seed_c] != BG_COLOR:
                return False
        else:
            any_low = True
            if seed_r <= int(basin["surface_r"]):
                return False
            natural_cells = sum(
                max(0, int(starts[c]) - int(basin["surface_r"]))
                for c in range(int(basin["l"]), int(basin["r"]) + 1)
            )
            capped_cells = sum(
                max(0, int(starts[c]) - seed_r) for c in range(int(basin["l"]), int(basin["r"]) + 1)
            )
            if capped_cells >= natural_cells:
                return False

    if any_low and np.array_equal(natural_only, expected):
        return False

    for basin_idx, basin in enumerate(basins):
        wet_mask_present = False
        for c in range(int(basin["l"]), int(basin["r"]) + 1):
            if int(basin["surface_r"]) < int(starts[c]):
                wet_mask_present = True
                if basin_idx not in seeded_idxs and not np.all(
                    expected[int(basin["surface_r"]) : int(starts[c]), c] == BG_COLOR
                ):
                    return False
        if not wet_mask_present:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    sampled = _sample_profile()
    if sampled is None:
        return None
    height, heights, basins = sampled
    terrain = random.choice([color for color in TERRAIN_CHOICES if color != BG_COLOR])
    starts = height - heights
    seed_plan = _pick_seed_plan(basins, starts, terrain)
    if seed_plan is None:
        return None

    grid = _render_input(height, heights, terrain, seed_plan)
    expected = _expected_from_plan(grid, terrain, basins, seed_plan)
    if not _validate_semantics(grid, terrain, heights, basins, seed_plan, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seeded-basin input")
