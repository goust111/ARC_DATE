from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


BG = 0
MARKER = 5
MAIN_COLOR_POOL = (1, 2, 3, 4, 7)
NOISE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
OUTPUT_WIDTH = 3
GEN_ATTEMPTS = 400


def _marker_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        if int(np.count_nonzero(grid[r] == MARKER)) >= 3:
            rows.append(int(r))
    return rows


def _region_ranges(height: int, marker_rows: list[int]) -> list[tuple[int, int]]:
    cuts = [-1, *marker_rows, height]
    ranges: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        r0 = int(a + 1)
        r1 = int(b)
        if r0 < r1:
            ranges.append((r0, r1))
    return ranges


def _region_summary(region: np.ndarray) -> tuple[int, int]:
    fg_mask = (region != BG) & (region != MARKER)
    fg = region[fg_mask]
    assert fg.size > 0, "each region must contain foreground pixels"
    counts = Counter(map(int, fg.tolist()))
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "region main color must be unique"
    return int(fg.size), int(ranked[0][0])


def _snake_fill(colors: list[int]) -> np.ndarray:
    rows = int(math.ceil(len(colors) / OUTPUT_WIDTH))
    out = np.full((rows, OUTPUT_WIDTH), BG, dtype=int)
    for idx, color in enumerate(colors):
        r = idx // OUTPUT_WIDTH
        offset = idx % OUTPUT_WIDTH
        c = offset if r % 2 == 0 else OUTPUT_WIDTH - 1 - offset
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress horizontal bands separated by sparse gray marker rows into a snake-ordered color ranking.

    Input structure:
    - The background is black.
    - Two or three sparse gray rows act as horizontal separators; each such row has at least three gray pixels.
    - The non-marker rows form horizontal bands containing scattered non-black, non-gray foreground pixels, with one uniquely dominant color per band.

    Rule:
    - Detect every sparse gray marker row.
    - Use those rows as boundaries; the marker rows themselves belong to no band.
    - For each band, count all non-black, non-gray foreground pixels and record the band's most frequent foreground color.
    - Sort the bands by total foreground-pixel count from largest to smallest.
    - Write the sorted dominant colors into a compact matrix with 3 columns in boustrophedon order: even rows left-to-right, odd rows right-to-left, padding unused cells with black.

    Key insight:
    - The gray rows are partition landmarks rather than content, and the output order comes from per-band foreground density before being folded into a snake layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _marker_rows(g)
    assert len(markers) >= 1, "expected sparse marker rows"

    summaries = []
    for r0, r1 in _region_ranges(g.shape[0], markers):
        total, main_color = _region_summary(g[r0:r1, :])
        summaries.append((total, main_color))

    totals = [total for total, _color in summaries]
    assert len(set(totals)) == len(totals), "band foreground counts must be distinct"

    ranked_colors = [color for total, color in sorted(summaries, key=lambda item: item[0], reverse=True)]
    return _snake_fill(ranked_colors)


def _random_partition(total: int, parts: int, min_each: int) -> list[int]:
    vals = [min_each] * parts
    for _ in range(total - parts * min_each):
        vals[random.randrange(parts)] += 1
    random.shuffle(vals)
    return vals


def _pick_marker_rows(height: int, marker_count: int) -> tuple[list[int], list[int]]:
    region_count = marker_count + 1
    band_height_sum = height - marker_count
    heights = _random_partition(band_height_sum, region_count, min_each=2)
    rows: list[int] = []
    cursor = 0
    for band_h in heights[:-1]:
        cursor += int(band_h)
        rows.append(cursor)
        cursor += 1
    return rows, heights


def _pick_totals(region_areas: list[int]) -> list[int] | None:
    common_max = min(min(area - 1, max(8, int(area * 0.6))) for area in region_areas)
    if common_max < 8:
        return None
    choices = list(range(5, common_max + 1))
    if len(choices) < len(region_areas):
        return None
    totals = random.sample(choices, len(region_areas))
    random.shuffle(totals)
    if totals == sorted(totals, reverse=True):
        return None
    return totals


def _split_noise(total: int, main_count: int, forced_two_way: bool) -> list[int]:
    noise = total - main_count
    if noise == 0:
        return []
    if forced_two_way and noise >= 2:
        left = random.randint(1, noise - 1)
        right = noise - left
        if max(left, right) < main_count:
            return [left, right]
    if noise < main_count:
        return [noise]

    lo = max(1, noise - (main_count - 1))
    hi = min(main_count - 1, noise - 1)
    if lo <= hi:
        left = random.randint(lo, hi)
        right = noise - left
        if max(left, right) < main_count:
            return [left, right]
    return []


def _pick_region_specs(main_colors: list[int], totals: list[int]) -> list[dict] | None:
    region_count = len(totals)
    pair_candidates = [(i, j) for i in range(region_count) for j in range(region_count) if totals[i] > totals[j]]
    if not pair_candidates:
        return None
    big_idx, small_idx = random.choice(pair_candidates)

    specs: list[dict] = []
    main_counts: list[int] = [0] * region_count
    noisy_regions = set()

    big_total = totals[big_idx]
    small_total = totals[small_idx]
    big_main_lo = max(4, big_total // 3 + 1)
    big_main_hi = max(big_main_lo, int(math.floor(big_total * 0.58)))
    small_main_lo = max(big_main_lo + 1, int(math.ceil(small_total * 0.72)))
    small_main_hi = small_total
    if big_main_lo > big_main_hi or small_main_lo > small_main_hi:
        return None

    main_counts[big_idx] = random.randint(big_main_lo, big_main_hi)
    main_counts[small_idx] = random.randint(small_main_lo, small_main_hi)
    if not (main_counts[big_idx] < main_counts[small_idx]):
        return None

    for idx, total in enumerate(totals):
        if idx in (big_idx, small_idx):
            continue
        lo = max(4, total // 2)
        hi = total
        if lo > hi:
            return None
        main_counts[idx] = random.randint(lo, hi)

    for idx, (main_color, total, main_count) in enumerate(zip(main_colors, totals, main_counts)):
        forced_two_way = idx == big_idx and total - main_count >= 2
        noise_parts = _split_noise(total, main_count, forced_two_way=forced_two_way)
        if total - main_count > 0 and not noise_parts:
            return None
        if noise_parts:
            noisy_regions.add(idx)

        noise_palette = [color for color in NOISE_COLOR_POOL if color != main_color]
        random.shuffle(noise_palette)
        if len(noise_palette) < len(noise_parts):
            return None
        noise_colors = noise_palette[: len(noise_parts)]
        specs.append(
            {
                "main_color": int(main_color),
                "total": int(total),
                "main_count": int(main_count),
                "noise": list(zip(noise_colors, noise_parts)),
            }
        )

    if not noisy_regions:
        return None
    return specs


def _sample_cells(shape: tuple[int, int], count: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
    if len(available) < count:
        raise ValueError("not enough free cells in region")
    return random.sample(available, count)


def _paint_region(grid: np.ndarray, row_start: int, row_stop: int, spec: dict) -> None:
    h = row_stop - row_start
    w = grid.shape[1]
    occupied: set[tuple[int, int]] = set()

    main_cells = _sample_cells((h, w), spec["main_count"], occupied)
    for rr, cc in main_cells:
        occupied.add((rr, cc))
        grid[row_start + rr, cc] = int(spec["main_color"])

    for color, count in spec["noise"]:
        cells = _sample_cells((h, w), int(count), occupied)
        for rr, cc in cells:
            occupied.add((rr, cc))
            grid[row_start + rr, cc] = int(color)


def _draw_marker_rows(grid: np.ndarray, marker_rows: list[int]) -> None:
    width = grid.shape[1]
    lo = max(3, int(math.ceil(width * 0.3)))
    hi = max(lo, int(math.floor(width * 0.7)))
    for r in marker_rows:
        count = random.randint(lo, hi)
        cols = random.sample(range(width), count)
        grid[int(r), cols] = MARKER


def _expected_from_specs(region_specs: list[dict]) -> np.ndarray:
    ranked = [spec["main_color"] for spec in sorted(region_specs, key=lambda spec: spec["total"], reverse=True)]
    return _snake_fill(ranked)


def _validate_generated_grid(
    grid: np.ndarray,
    marker_rows: list[int],
    region_specs: list[dict],
    expected: np.ndarray,
) -> bool:
    found_markers = _marker_rows(grid)
    if found_markers != marker_rows:
        return False

    width = grid.shape[1]
    for r in marker_rows:
        grey_count = int(np.count_nonzero(grid[r] == MARKER))
        if grey_count < 3:
            return False
        if not (math.ceil(width * 0.3) <= grey_count <= math.floor(width * 0.7)):
            return False
        if np.any((grid[r] != BG) & (grid[r] != MARKER)):
            return False

    ranges = _region_ranges(grid.shape[0], marker_rows)
    if len(ranges) != len(region_specs):
        return False

    seen_main_colors = set()
    seen_totals = set()
    noise_present = False
    main_counts = []
    actual_totals = []
    for (r0, r1), spec in zip(ranges, region_specs):
        region = grid[r0:r1, :]
        total, main_color = _region_summary(region)
        counts = Counter(map(int, region[(region != BG) & (region != MARKER)].tolist()))
        main_count = int(counts[main_color])

        if total != spec["total"] or main_color != spec["main_color"]:
            return False
        if main_count != spec["main_count"]:
            return False
        if sum(count for _color, count in spec["noise"]) != total - main_count:
            return False
        if any(count >= main_count for color, count in spec["noise"] if color != main_color):
            return False

        seen_main_colors.add(main_color)
        seen_totals.add(total)
        actual_totals.append(total)
        main_counts.append(main_count)
        noise_present |= bool(spec["noise"])

    if len(seen_main_colors) != len(region_specs):
        return False
    if len(seen_totals) != len(region_specs):
        return False
    if not noise_present:
        return False
    if actual_totals == sorted(actual_totals, reverse=True):
        return False

    mismatch_found = any(
        actual_totals[i] > actual_totals[j] and main_counts[i] < main_counts[j]
        for i in range(len(region_specs))
        for j in range(len(region_specs))
    )
    if not mismatch_found:
        return False

    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    marker_count = 3 if random.random() < 0.65 else 2
    height = random.randint(14, 18)
    width = random.randint(12, 17)
    region_count = marker_count + 1
    if width < 10:
        return None

    marker_rows, band_heights = _pick_marker_rows(height, marker_count)
    region_areas = [band_h * width for band_h in band_heights]
    totals = _pick_totals(region_areas)
    if totals is None:
        return None

    main_colors = random.sample(MAIN_COLOR_POOL, region_count)
    region_specs = _pick_region_specs(main_colors, totals)
    if region_specs is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    _draw_marker_rows(grid, marker_rows)
    for (r0, r1), spec in zip(_region_ranges(height, marker_rows), region_specs):
        _paint_region(grid, r0, r1, spec)

    expected = _expected_from_specs(region_specs)
    if not _validate_generated_grid(grid, marker_rows, region_specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
