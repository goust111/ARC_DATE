from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BLOCKER_COLORS = (RED, GREEN, YELLOW, ORANGE)
FILL_COLORS = (BLUE, PINK, TEAL, MAROON)
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _segment_between(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac
    assert dr != 0 and abs(dr) == abs(dc)
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(ar + i * sr, ac + i * sc) for i in range(abs(dr) + 1)], (sr, sc)


def _legend_from_corner_frame(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], tuple[int, int, int, int]]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 != 0:
            continue
        if c0 != 0 and c1 != w - 1:
            continue
        rows: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            source_positions = []
            fill_positions = []
            for c in range(c0, c1 + 1):
                color = int(g[r, c])
                if color in BLOCKER_COLORS:
                    source_positions.append((c, color))
                elif color in FILL_COLORS:
                    fill_positions.append((c, color))
            if len(source_positions) == 1 and len(fill_positions) == 1:
                sc, source = source_positions[0]
                fc, fill = fill_positions[0]
                if sc < fc:
                    rows.append((source, fill))
        if 2 <= len(rows) <= 4:
            area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            candidates.append((area, rows, (r0, c0, r1, c1)))

    assert candidates, "expected a grey corner legend"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _area, rows, box = candidates[0]
    legend_mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = box
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return rows, legend_mask


def _parse_puzzle(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]], tuple[int, int], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    pairs, legend_mask = _legend_from_corner_frame(g)
    fill_map = {source: fill for source, fill in pairs}

    grey_pts = np.argwhere((g == GREY) & ~legend_mask)
    assert grey_pts.shape[0] == 2, "expected exactly two grey endpoints outside the legend"
    endpoints = [(int(r), int(c)) for r, c in grey_pts]
    segment, step = _segment_between(endpoints[0], endpoints[1])

    blockers: list[tuple[int, int]] = []
    blocker_positions: dict[int, tuple[int, int]] = {}
    for r, c in segment[1:-1]:
        color = int(g[r, c])
        if color == BG:
            continue
        assert color in fill_map, "unexpected color on the main diagonal"
        blockers.append((r, c))
        blocker_positions[color] = (r, c)

    assert len(blockers) == len(pairs), "legend rows and blocker count must match"
    assert set(blocker_positions) == set(fill_map), "legend source colors must match the blockers"
    return pairs, legend_mask, segment, step, blocker_positions


def _render_structure(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    pairs, _legend_mask, segment, step, blocker_positions = _parse_puzzle(g)
    fill_map = {source: fill for source, fill in pairs}
    priority = {source: idx for idx, (source, _fill) in enumerate(pairs)}

    out = g.copy()
    for r, c in segment:
        if int(out[r, c]) in (BG, GREY):
            out[r, c] = GREY

    ray_dirs = [(step[0], -step[1]), (-step[0], step[1])]
    rays_by_source: dict[int, list[tuple[int, int]]] = {}
    for source, (r, c) in blocker_positions.items():
        coords = [(r, c)]
        for dr, dc in ray_dirs:
            rr, cc = r + dr, c + dc
            while _in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == BG:
                out[rr, cc] = source
                coords.append((rr, cc))
                rr += dr
                cc += dc
        rays_by_source[source] = coords

    ray_arrays = {source: np.array(coords, dtype=int) for source, coords in rays_by_source.items()}
    return out, ray_arrays, fill_map, priority


def _min_chebyshev_distance(region: np.ndarray, ray: np.ndarray) -> int:
    delta = np.abs(region[:, None, :] - ray[None, :, :])
    return int(np.max(delta, axis=2).min())


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the main diagonal, emit perpendicular blocker rays, and fill each partition by the closest legend-mapped blocker.

    Input structure:
    - The background is black.
    - A grey corner legend frame contains 2-4 rows of `(blocker_color, fill_color)` pairs.
    - Outside the legend, two grey endpoints mark a main diagonal segment, and the segment interior contains one uniquely colored blocker for each legend row.

    Rule:
    - Complete the grey main diagonal between the two endpoints while preserving blocker pixels on it.
    - Each blocker emits the full perpendicular diagonal in both directions until the grid edge or an existing non-background pixel.
    - Treat the drawn diagonal and blocker rays as barriers, split the remaining background into 4-connected regions, and fill each region with the fill color paired to the closest blocker ray; equal distances use earlier legend rows first.

    Key insight:
    - The diagonal and emitted rays define the geometry, but the legend order decides who wins disputed regions, so solving requires combining spatial partitioning with symbolic priority.
    """
    structure, rays_by_source, fill_map, priority = _render_structure(np.asarray(grid, dtype=int))
    out = structure.copy()

    for region in _components(out == BG):
        best_source = None
        best_key = None
        for source, ray in rays_by_source.items():
            key = (_min_chebyshev_distance(region, ray), priority[source])
            if best_key is None or key < best_key:
                best_key = key
                best_source = source
        assert best_source is not None
        out[region[:, 0], region[:, 1]] = fill_map[best_source]
    return out


def _random_spaced_indices(length: int, count: int) -> list[int]:
    interior = list(range(1, length))
    for _ in range(100):
        picks = sorted(random.sample(interior, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    return sorted(random.sample(interior, count))


def _legend_mask_for_box(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _ray_cells_from_point(
    shape: tuple[int, int],
    point: tuple[int, int],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = point
    cells = [(r, c)]
    for dr, dc in ((step[0], -step[1]), (-step[0], step[1])):
        rr, cc = r + dr, c + dc
        while _in_bounds(shape, rr, cc):
            cells.append((rr, cc))
            rr += dr
            cc += dc
    return cells


def _region_winners(
    structure: np.ndarray,
    rays_by_source: dict[int, np.ndarray],
    fill_map: dict[int, int],
    priority: dict[int, int],
) -> tuple[list[int], bool]:
    winners: list[int] = []
    tie_seen = False
    for region in _components(structure == BG):
        scored = sorted(
            (_min_chebyshev_distance(region, ray), priority[source], source)
            for source, ray in rays_by_source.items()
        )
        if len(scored) >= 2 and scored[0][0] == scored[1][0]:
            tie_seen = True
        winners.append(fill_map[scored[0][2]])
    return winners, tie_seen


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    blocker_count = random.randint(2, 4)

    grid = np.full((h, w), BG, dtype=int)
    legend_height = blocker_count + 2
    legend_width = 4
    legend_left = random.choice([True, False])
    legend_c0 = 0 if legend_left else w - legend_width
    legend_box = (0, legend_c0, legend_height - 1, legend_c0 + legend_width - 1)
    legend_mask = _legend_mask_for_box(grid.shape, legend_box)

    grid[0:legend_height, legend_c0 : legend_c0 + legend_width] = GREY
    sources = random.sample(list(BLOCKER_COLORS), blocker_count)
    fills = random.sample(list(FILL_COLORS), blocker_count)
    pairs = list(zip(sources, fills))
    for idx, (source, fill) in enumerate(pairs):
        rr = idx + 1
        grid[rr, legend_c0 + 1] = source
        grid[rr, legend_c0 + 2] = fill

    min_steps = max(8, blocker_count * 2 + 3)
    max_steps = min(14, h - legend_height - 4, w - 5)
    if max_steps < min_steps:
        return None

    for _ in range(160):
        step = (1, random.choice([-1, 1]))
        diag_steps = random.randint(min_steps, max_steps)
        top_r = random.randint(legend_height + 1, h - diag_steps - 2)
        if step[1] == 1:
            top_c = random.randint(2, w - diag_steps - 3)
        else:
            top_c = random.randint(diag_steps + 2, w - 3)

        a = (top_r, top_c)
        b = (top_r + diag_steps, top_c + step[1] * diag_steps)
        segment, _ = _segment_between(a, b)
        if any(legend_mask[r, c] for r, c in segment):
            continue

        blocker_steps = _random_spaced_indices(diag_steps, blocker_count)
        blocker_points = [segment[i] for i in blocker_steps]
        all_non_bg = {a, b, *blocker_points}

        blocked = False
        for point in blocker_points:
            ray_cells = _ray_cells_from_point(grid.shape, point, step)
            for rr, cc in ray_cells[1:]:
                if legend_mask[rr, cc] or (rr, cc) in all_non_bg:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue

        candidate = grid.copy()
        candidate[a] = GREY
        candidate[b] = GREY
        blocker_order = random.sample(sources, blocker_count)
        for color, point in zip(blocker_order, blocker_points):
            candidate[point] = color

        try:
            structure, rays_by_source, fill_map, priority = _render_structure(candidate)
        except AssertionError:
            continue

        regions = _components(structure == BG)
        if len(regions) < blocker_count + 2:
            continue

        winners, tie_seen = _region_winners(structure, rays_by_source, fill_map, priority)
        if not tie_seen:
            continue
        if len(set(winners)) < 2:
            continue

        solved = main(candidate)
        used_fills = set(map(int, solved[np.isin(solved, list(FILL_COLORS))]))
        if len(used_fills) < 2:
            continue
        return candidate
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
