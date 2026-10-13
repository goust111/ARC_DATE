from __future__ import annotations

import random
from itertools import combinations

import numpy as np


MIN_REGION_SIDE = 4
GEN_SIDE_RANGE = (4, 7)
GEN_ATTEMPTS = 400

LAYOUTS_BY_REGION_COUNT = {
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
}


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    vals, cnts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist(), strict=False)}


def _seed_map(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    seeds: dict[int, list[tuple[int, int]]] = {}
    counts = _color_counts(grid)
    for color, count in counts.items():
        if count != 2:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) == 2:
            seeds[int(color)] = sorted(coords)
    return seeds


def _partitions(total: int, parts: int, min_size: int) -> list[tuple[tuple[int, int], ...]]:
    if parts == 1:
        if total >= min_size:
            return [((0, total),)]
        return []

    out: list[tuple[tuple[int, int], ...]] = []
    max_first = total - min_size * (parts - 1)
    for first in range(min_size, max_first + 1):
        for rest in _partitions(total - first, parts - 1, min_size):
            shifted = tuple((first + a, first + b) for a, b in rest)
            out.append(((0, first),) + shifted)
    return out


def _alternates_two(colors: list[int]) -> bool:
    if len(colors) < 2 or len(set(colors)) != 2:
        return False
    return all(colors[i] != colors[i + 1] for i in range(len(colors) - 1)) and all(
        colors[i] == colors[i + 2] for i in range(len(colors) - 2)
    )


def _is_horizontal_stripes(block: np.ndarray, seed_color: int) -> bool:
    row_colors: list[int] = []
    for r in range(block.shape[0]):
        vals = block[r][block[r] != seed_color]
        if vals.size == 0 or np.unique(vals).size != 1:
            return False
        row_colors.append(int(vals[0]))
    return _alternates_two(row_colors)


def _is_vertical_stripes(block: np.ndarray, seed_color: int) -> bool:
    col_colors: list[int] = []
    for c in range(block.shape[1]):
        vals = block[:, c][block[:, c] != seed_color]
        if vals.size == 0 or np.unique(vals).size != 1:
            return False
        col_colors.append(int(vals[0]))
    return _alternates_two(col_colors)


def _analyze_block(block: np.ndarray, seed_colors: set[int]) -> dict | None:
    present_seed_colors = [color for color in seed_colors if np.any(block == color)]
    if len(present_seed_colors) != 1:
        return None

    seed_color = int(present_seed_colors[0])
    coords = [tuple(map(int, rc)) for rc in np.argwhere(block == seed_color)]
    if len(coords) != 2:
        return None

    bg_vals = block[block != seed_color]
    if bg_vals.size == 0 or np.unique(bg_vals).size != 2:
        return None

    is_h = _is_horizontal_stripes(block, seed_color)
    is_v = _is_vertical_stripes(block, seed_color)
    if is_h == is_v:
        return None

    return {
        "seed_color": seed_color,
        "seeds_rel": sorted(coords),
        "orientation": "H" if is_h else "V",
    }


def _candidate_signature(regions: list[dict]) -> tuple:
    return tuple(
        sorted(
            (
                int(region["r0"]),
                int(region["r1"]),
                int(region["c0"]),
                int(region["c1"]),
                int(region["seed_color"]),
                str(region["orientation"]),
            )
            for region in regions
        )
    )


def _find_region_candidates(grid: np.ndarray) -> list[list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seeds = _seed_map(g)
    seed_colors = set(seeds)
    region_count = len(seeds)
    layouts = LAYOUTS_BY_REGION_COUNT.get(region_count, ())
    candidates: dict[tuple, tuple[tuple[int, int, int], list[dict]]] = {}

    for nr, nc in layouts:
        for row_parts in _partitions(h, nr, MIN_REGION_SIDE):
            for col_parts in _partitions(w, nc, MIN_REGION_SIDE):
                used: set[int] = set()
                regions: list[dict] = []
                ok = True

                for r0, r1 in row_parts:
                    for c0, c1 in col_parts:
                        info = _analyze_block(g[r0:r1, c0:c1], seed_colors)
                        if info is None or int(info["seed_color"]) in used:
                            ok = False
                            break

                        used.add(int(info["seed_color"]))
                        seeds_abs = [(r0 + rr, c0 + cc) for rr, cc in info["seeds_rel"]]
                        regions.append(
                            {
                                "r0": int(r0),
                                "r1": int(r1),
                                "c0": int(c0),
                                "c1": int(c1),
                                "seed_color": int(info["seed_color"]),
                                "orientation": str(info["orientation"]),
                                "seeds": sorted(seeds_abs),
                            }
                        )
                    if not ok:
                        break

                if not ok or used != seed_colors:
                    continue

                score = (
                    min((region["r1"] - region["r0"]) * (region["c1"] - region["c0"]) for region in regions),
                    min(region["r1"] - region["r0"] for region in regions),
                    min(region["c1"] - region["c0"] for region in regions),
                )
                signature = _candidate_signature(regions)
                prev = candidates.get(signature)
                if prev is None or score > prev[0]:
                    candidates[signature] = (score, regions)

    ordered = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
    return [regions for _score, regions in ordered]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise ValueError("segment must be axis-aligned")


def _choose_elbow(
    seed_a: tuple[int, int],
    seed_b: tuple[int, int],
    region: dict,
) -> tuple[int, int] | None:
    if seed_a[0] == seed_b[0] or seed_a[1] == seed_b[1]:
        return None

    cand_h = (seed_a[0], seed_b[1])
    cand_v = (seed_b[0], seed_a[1])
    center_r = (region["r0"] + region["r1"] - 1) / 2.0
    center_c = (region["c0"] + region["c1"] - 1) / 2.0

    def dist2(cell: tuple[int, int]) -> float:
        return (cell[0] - center_r) ** 2 + (cell[1] - center_c) ** 2

    d_h = dist2(cand_h)
    d_v = dist2(cand_v)
    if d_h < d_v:
        return cand_h
    if d_v < d_h:
        return cand_v

    first, second = sorted((seed_a, seed_b))
    if region["orientation"] == "H":
        return (second[0], first[1])
    return (first[0], second[1])


def _apply_region_path(out: np.ndarray, region: dict) -> None:
    seed_a, seed_b = sorted(region["seeds"])
    color = int(region["seed_color"])
    elbow = _choose_elbow(seed_a, seed_b, region)

    if elbow is None:
        cells = _segment_cells(seed_a, seed_b)
    else:
        cells = _segment_cells(seed_a, elbow) + _segment_cells(elbow, seed_b)

    for r, c in cells:
        out[int(r), int(c)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored seed pair inside its striped rectangular region.

    The grid is partitioned into hidden rectangles. Every rectangle has a two-color
    stripe texture and exactly one pair of same-colored seed pixels. Regions are
    recovered by enumerating plausible row/column partitions and validating that each
    block contains one seed pair plus either horizontal or vertical stripes.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _find_region_candidates(g)
    assert candidates, "failed to recover any valid rectangular partition"

    out = g.copy()
    for region in candidates[0]:
        _apply_region_path(out, region)
    return out


def _render_stripes(height: int, width: int, colors: tuple[int, int], orientation: str, phase: int) -> np.ndarray:
    if orientation == "H":
        rows = (np.arange(height) + phase) % 2
        return np.where(np.repeat(rows[:, None], width, axis=1) == 0, colors[0], colors[1]).astype(int)

    cols = (np.arange(width) + phase) % 2
    return np.where(np.repeat(cols[None, :], height, axis=0) == 0, colors[0], colors[1]).astype(int)


def _sample_sizes(parts: int) -> list[int]:
    return [random.randint(*GEN_SIDE_RANGE) for _ in range(parts)]


def _elbow_dist_tie(height: int, width: int, a: tuple[int, int], b: tuple[int, int]) -> bool:
    region = {"r0": 0, "r1": height, "c0": 0, "c1": width}
    cand_h = (a[0], b[1])
    cand_v = (b[0], a[1])
    center_r = (region["r0"] + region["r1"] - 1) / 2.0
    center_c = (region["c0"] + region["c1"] - 1) / 2.0
    d_h = (cand_h[0] - center_r) ** 2 + (cand_h[1] - center_c) ** 2
    d_v = (cand_v[0] - center_r) ** 2 + (cand_v[1] - center_c) ** 2
    return abs(d_h - d_v) < 1e-9


def _sample_seed_pair(height: int, width: int, case_name: str) -> list[tuple[int, int]] | None:
    if case_name == "row":
        for _ in range(40):
            row = random.randrange(height)
            cols = sorted(random.sample(range(width), 2))
            if cols[1] - cols[0] >= 2 or width == MIN_REGION_SIDE:
                return [(row, cols[0]), (row, cols[1])]
        return None

    if case_name == "col":
        for _ in range(40):
            col = random.randrange(width)
            rows = sorted(random.sample(range(height), 2))
            if rows[1] - rows[0] >= 2 or height == MIN_REGION_SIDE:
                return [(rows[0], col), (rows[1], col)]
        return None

    if case_name == "tie":
        for _ in range(80):
            r0 = random.randrange(height)
            c0 = random.randrange(width)
            r1 = height - 1 - r0
            c1 = width - 1 - c0
            if (r0, c0) == (r1, c1):
                continue
            if r0 == r1 or c0 == c1:
                continue
            return sorted([(r0, c0), (r1, c1)])
        return None

    for _ in range(100):
        a = (random.randrange(height), random.randrange(width))
        b = (random.randrange(height), random.randrange(width))
        if a == b or a[0] == b[0] or a[1] == b[1]:
            continue
        if _elbow_dist_tie(height, width, a, b):
            continue
        return sorted([a, b])
    return None


def _layout_for_region_count(region_count: int) -> tuple[int, int]:
    return random.choice(LAYOUTS_BY_REGION_COUNT[region_count])


def _direct_expected(inp: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        _apply_region_path(out, region)
    return out


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice((2, 3, 4, 6))
    nr, nc = _layout_for_region_count(region_count)
    row_sizes = _sample_sizes(nr)
    col_sizes = _sample_sizes(nc)
    h, w = sum(row_sizes), sum(col_sizes)

    colors = list(range(10))
    random.shuffle(colors)
    seed_colors = colors[:region_count]
    bg_palette = colors[region_count:]
    if len(list(combinations(bg_palette, 2))) < region_count:
        return None

    bg_pairs = random.sample(list(combinations(bg_palette, 2)), k=region_count)
    orientations = [random.choice(("H", "V")) for _ in range(region_count)]
    phases = [random.randint(0, 1) for _ in range(region_count)]

    grid = np.zeros((h, w), dtype=int)
    regions: list[dict] = []
    row_starts = [0]
    col_starts = [0]
    for size in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + size)
    for size in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + size)

    region_index = 0
    for rr, row_size in enumerate(row_sizes):
        for cc, col_size in enumerate(col_sizes):
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            r1 = r0 + row_size
            c1 = c0 + col_size

            seed_color = int(seed_colors[region_index])
            bg_pair = tuple(map(int, bg_pairs[region_index]))
            orientation = str(orientations[region_index])
            phase = int(phases[region_index])

            grid[r0:r1, c0:c1] = _render_stripes(row_size, col_size, bg_pair, orientation, phase)

            case_name = random.choice(("row", "col", "bend", "bend", "tie"))
            seeds_rel = _sample_seed_pair(row_size, col_size, case_name)
            if seeds_rel is None:
                return None

            seeds_abs = [(r0 + r, c0 + c) for r, c in seeds_rel]
            for r, c in seeds_abs:
                grid[r, c] = seed_color

            regions.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "seed_color": seed_color,
                    "orientation": orientation,
                    "seeds": sorted(seeds_abs),
                }
            )
            region_index += 1

    counts = _color_counts(grid)
    if any(color not in seed_colors and count == 2 for color, count in counts.items()):
        return None

    candidates = _find_region_candidates(grid)
    if len(candidates) != 1:
        return None

    expected = _direct_expected(grid, regions)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-region sample")
