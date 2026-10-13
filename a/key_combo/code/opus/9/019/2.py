from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC = SourceFileLoader("_arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py").load_module()
find_components = _ARC.find_components
get_bg = _ARC.get_bg


BG = 0
WALL = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_RETRIES = 400
MIN_REGION = 5
GRID_RANGE = (17, 26)


def _rect_size(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _extract_regions(grid: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for comp in find_components(grid, bg=WALL, connectivity=4, monochromatic=False, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h, w = _rect_size((r0, c0, r1, c1))
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert int(comp["area"]) == h * w, "wall-bounded regions must be rectangles"
        assert np.all(patch != WALL), "region interior cannot contain walls"

        fg_mask = (patch != BG) & (patch != WALL)
        colors = sorted(set(map(int, np.unique(patch[fg_mask]))))
        assert len(colors) <= 1, "each region must use at most one foreground color"
        color = colors[0] if colors else None
        local_sources = [tuple(map(int, coord)) for coord in np.argwhere(fg_mask)]
        abs_sources = [(r0 + rr, c0 + cc) for rr, cc in local_sources]
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "h": h,
                "w": w,
                "patch": patch,
                "color": color,
                "local_sources": local_sources,
                "sources": abs_sources,
                "count": len(abs_sources),
                "area": h * w,
            }
        )
    regions.sort(key=lambda region: region["bbox"])
    return regions


def _is_isolated(coords: list[tuple[int, int]]) -> bool:
    seen = set(coords)
    for r, c in coords:
        if (r - 1, c) in seen or (r + 1, c) in seen or (r, c - 1) in seen or (r, c + 1) in seen:
            return False
    return True


def _winner_index(regions: list[dict]) -> int:
    counts = [int(region["count"]) for region in regions]
    max_count = max(counts)
    winners = [idx for idx, count in enumerate(counts) if count == max_count]
    assert len(winners) == 1, "foreground count winner must be unique"
    return winners[0]


def _cast_ray(grid: np.ndarray, source: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    r, c = source
    dr, dc = direction
    rr = r + dr
    cc = c + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == WALL:
            return cells, abs(rr - r) + abs(cc - c)
        cells.append((rr, cc))
        rr += dr
        cc += dc
    raise AssertionError("every ray must terminate on a wall")


def _collect_candidates(grid: np.ndarray, sources: list[tuple[int, int]], color: int) -> dict[tuple[int, int], list[dict]]:
    cell_to_candidates: dict[tuple[int, int], list[dict]] = {}
    for src_r, src_c in sources:
        for dir_idx, direction in enumerate(DIRS):
            cells, wall_distance = _cast_ray(grid, (src_r, src_c), direction)
            priority = (int(wall_distance), int(src_r), int(src_c), int(dir_idx))
            for cell in cells:
                cell_to_candidates.setdefault(cell, []).append(
                    {
                        "priority": priority,
                        "distance": int(wall_distance),
                        "source": (int(src_r), int(src_c)),
                        "color": int(color),
                        "direction": direction,
                    }
                )
    return cell_to_candidates


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)
    winner = regions[_winner_index(regions)]
    if winner["count"] == 0:
        return g.copy()

    out = g.copy()
    candidates = _collect_candidates(g, winner["sources"], int(winner["color"]))
    for cell, options in candidates.items():
        best = min(options, key=lambda option: option["priority"])
        out[cell] = int(best["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let only the densest wall-bounded region emit same-color rays until the surrounding gray walls.

    Input structure:
    - Black cells are background and gray cells form one-cell-thick partition walls.
    - The non-wall cells are split into 4 to 6 rectangular regions.
    - Each region contains 2 to 7 isolated foreground pixels in one non-black, non-gray color, and different regions use different colors.

    Rule:
    - Partition the grid by the gray walls and count foreground pixels inside each rectangular region.
    - Identify the unique region with the strictly largest foreground count.
    - For every foreground pixel in that winning region, cast rays of the same color up, down, left, and right until the cell just before a gray wall.
    - When multiple rays cover the same cell, the ray whose source is closer to its target wall wins; ties prefer the smaller source row.
    - All non-winning regions stay unchanged and the gray walls remain unchanged.

    Key insight:
    - Counting decides which region is allowed to act; the visible fill is then produced by row/column ray emission inside only that wall-defined compartment.
    """
    return _solve_grid(grid)


def _random_partition() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    target_regions = random.randint(4, 6)
    rects = [(1, 1, height - 2, width - 2)]

    while len(rects) < target_regions:
        feasible: list[int] = []
        for idx, rect in enumerate(rects):
            h, w = _rect_size(rect)
            if h >= 2 * MIN_REGION + 1 or w >= 2 * MIN_REGION + 1:
                feasible.append(idx)
        if not feasible:
            return None

        rect_idx = random.choice(feasible)
        r0, c0, r1, c1 = rects.pop(rect_idx)
        h, w = _rect_size((r0, c0, r1, c1))
        orientations: list[str] = []
        if h >= 2 * MIN_REGION + 1:
            orientations.append("h")
        if w >= 2 * MIN_REGION + 1:
            orientations.append("v")
        orientation = random.choice(orientations)
        if orientation == "h":
            wall_r = random.randint(r0 + MIN_REGION, r1 - MIN_REGION)
            rects.append((r0, c0, wall_r - 1, c1))
            rects.append((wall_r + 1, c0, r1, c1))
        else:
            wall_c = random.randint(c0 + MIN_REGION, c1 - MIN_REGION)
            rects.append((r0, c0, r1, wall_c - 1))
            rects.append((r0, wall_c + 1, r1, c1))

    grid = np.full((height, width), WALL, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    return grid, sorted(rects)


def _region_capacity(rect: tuple[int, int, int, int]) -> int:
    h, w = _rect_size(rect)
    return (h * w + 1) // 2


def _overlap_signature(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> tuple[bool, bool]:
    ra, ca = a
    rb, cb = b
    if ra == rb or ca == cb:
        return False, False

    dist_a = h - ra if rb > ra else ra + 1
    dist_b = cb + 1 if ca < cb else w - cb
    if dist_a == dist_b:
        return True, True

    dist_c = w - ca if cb > ca else ca + 1
    dist_d = rb + 1 if ra < rb else h - rb
    if dist_c == dist_d:
        return True, True
    return True, False


def _find_seed_pair(h: int, w: int, tie_mode: bool) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    fallback: list[tuple[int, int]] | None = None
    for a in cells:
        for b in cells:
            if a == b:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                continue
            has_cross, is_tie = _overlap_signature(a, b, h, w)
            if not has_cross:
                continue
            pair = [a, b]
            if tie_mode and is_tie:
                return pair
            if not tie_mode and not is_tie:
                return pair
            if fallback is None:
                fallback = pair
    return fallback


def _fill_random_isolated(h: int, w: int, count: int, required: list[tuple[int, int]] | None = None) -> list[tuple[int, int]] | None:
    chosen = [] if required is None else list(required)
    if not _is_isolated(chosen):
        return None

    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in chosen]
    random.shuffle(cells)
    for cell in cells:
        if len(chosen) >= count:
            break
        if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
    if len(chosen) != count or not _is_isolated(chosen):
        return None
    return chosen


def _paint_region_points(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    count: int,
    mode: str,
) -> list[tuple[int, int]] | None:
    r0, c0, _r1, _c1 = rect
    h, w = _rect_size(rect)
    required: list[tuple[int, int]] = []
    if mode in {"cross", "tie"}:
        pair = _find_seed_pair(h, w, tie_mode=(mode == "tie"))
        if pair is None:
            return None
        required = pair
    coords = _fill_random_isolated(h, w, count, required=required)
    if coords is None:
        return None
    for rr, cc in coords:
        grid[r0 + rr, c0 + cc] = int(color)
    return coords


def _winner_rows_cols(local_sources: list[tuple[int, int]]) -> tuple[set[int], set[int]]:
    return {r for r, _ in local_sources}, {c for _, c in local_sources}


def _all_regions_emit(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for region in _extract_regions(out):
        if region["count"] == 0:
            continue
        candidates = _collect_candidates(out, region["sources"], int(region["color"]))
        for cell, options in candidates.items():
            best = min(options, key=lambda option: option["priority"])
            out[cell] = int(best["color"])
    return out


def _validate_semantics(grid: np.ndarray, mode: str, prefer_non_area_winner: bool) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(get_bg(grid)) == BG

    regions = _extract_regions(grid)
    assert 4 <= len(regions) <= 6
    colors = [region["color"] for region in regions]
    assert None not in colors
    assert len(set(colors)) == len(colors)

    winner_idx = _winner_index(regions)
    winner = regions[winner_idx]
    counts = [region["count"] for region in regions]
    assert min(counts) >= 2 and max(counts) <= 7
    assert winner["count"] > max(count for idx, count in enumerate(counts) if idx != winner_idx)

    for region in regions:
        assert _is_isolated(region["local_sources"])
        patch = region["patch"]
        fg_mask = (patch != BG) & (patch != WALL)
        assert int(np.sum(fg_mask)) == region["count"]

    if prefer_non_area_winner:
        max_area = max(region["area"] for region in regions)
        assert winner["area"] < max_area

    out = main(grid)
    assert np.array_equal(out == WALL, grid == WALL)
    assert not np.array_equal(out, grid)

    r0, c0, r1, c1 = winner["bbox"]
    expected_patch = winner["patch"].copy()
    rows, cols = _winner_rows_cols(winner["local_sources"])
    for rr in range(winner["h"]):
        for cc in range(winner["w"]):
            if rr in rows or cc in cols:
                expected_patch[rr, cc] = int(winner["color"])
    assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected_patch)

    for idx, region in enumerate(regions):
        if idx == winner_idx:
            continue
        rr0, cc0, rr1, cc1 = region["bbox"]
        assert np.array_equal(out[rr0 : rr1 + 1, cc0 : cc1 + 1], grid[rr0 : rr1 + 1, cc0 : cc1 + 1])

    all_emit = _all_regions_emit(grid)
    assert not np.array_equal(all_emit, out)

    overlaps = []
    candidates = _collect_candidates(grid, winner["sources"], int(winner["color"]))
    for cell, options in candidates.items():
        if len(options) >= 2:
            ordered = sorted(options, key=lambda option: option["priority"])
            overlaps.append((cell, ordered))
    assert overlaps
    assert any(grid[cell] == BG for cell in candidates)

    if mode == "tie":
        assert any(
            len(options) >= 2
            and options[0]["distance"] == options[1]["distance"]
            and options[0]["source"][0] != options[1]["source"][0]
            for _, options in overlaps
        )
    elif mode == "cross":
        assert any(
            len(options) >= 2 and options[0]["distance"] != options[1]["distance"]
            for _, options in overlaps
        )


def _generate_attempt() -> np.ndarray | None:
    partition = _random_partition()
    if partition is None:
        return None
    grid, rects = partition

    capacities = [_region_capacity(rect) for rect in rects]
    if min(capacities) < 2:
        return None

    winner_mode = random.choices(["tie", "cross", "plain"], weights=(4, 4, 2), k=1)[0]
    prefer_non_area_winner = random.random() < 0.6
    area_order = sorted(range(len(rects)), key=lambda idx: _rect_size(rects[idx])[0] * _rect_size(rects[idx])[1], reverse=True)

    eligible = [idx for idx, cap in enumerate(capacities) if cap >= 4]
    if not eligible:
        return None
    if prefer_non_area_winner:
        eligible = [idx for idx in eligible if idx != area_order[0]] or eligible
        prefer_non_area_winner = bool([idx for idx in eligible if idx != area_order[0]])

    winner_idx = random.choice(eligible)
    winner_cap = capacities[winner_idx]
    winner_count = random.randint(4, min(7, winner_cap))

    counts: list[int] = []
    for idx, cap in enumerate(capacities):
        if idx == winner_idx:
            counts.append(winner_count)
            continue
        upper = min(7, winner_count - 1, cap)
        if upper < 2:
            return None
        counts.append(random.randint(2, upper))

    if counts[winner_idx] != max(counts) or counts.count(max(counts)) != 1:
        return None

    colors = random.sample(FG_COLORS, len(rects))
    for idx, rect in enumerate(rects):
        mode = "plain"
        if idx == winner_idx:
            mode = winner_mode
        coords = _paint_region_points(grid, rect, colors[idx], counts[idx], mode)
        if coords is None:
            return None

    try:
        _validate_semantics(grid, winner_mode, prefer_non_area_winner)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid density-ray ARC input")
