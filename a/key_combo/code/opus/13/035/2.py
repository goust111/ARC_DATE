from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
WALL = 5
RAY_COLORS = (1, 2, 3, 4)


def _find_wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == WALL)) > w // 2]
    wall_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == WALL)) > h // 2]
    return wall_rows, wall_cols


def _intervals(size: int, walls: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + walls + [size]
    spans = []
    for a, b in zip(bounds, bounds[1:]):
        lo = a + 1
        hi = b - 1
        if lo <= hi:
            spans.append((lo, hi))
    return spans


def _border_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    seeds = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color not in (BG, WALL):
                seeds.append((r, c, color))
    return sorted(seeds)


def _seed_direction(seed: tuple[int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c, _color = seed
    h, w = shape
    if c == 0:
        return 0, 1
    if c == w - 1:
        return 0, -1
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    raise AssertionError("seed must lie on the grid border")


def _trace_ray(grid: np.ndarray, seed: tuple[int, int, int]) -> dict[str, Any]:
    h, w = grid.shape
    r, c, color = seed
    dr, dc = _seed_direction(seed, grid.shape)
    path: list[tuple[int, int]] = []
    dist: dict[tuple[int, int], int] = {}
    steps = 0

    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != WALL:
        path.append((r, c))
        dist[(r, c)] = steps
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) == WALL:
            break
        r, c = nr, nc
        steps += 1

    return {"seed": seed, "color": color, "path": path, "dist": dist}


def _chambers(grid: np.ndarray) -> list[dict[str, Any]]:
    wall_rows, wall_cols = _find_wall_lines(grid)
    row_spans = _intervals(grid.shape[0], wall_rows)
    col_spans = _intervals(grid.shape[1], wall_cols)
    chambers = []
    for rb, (r0, r1) in enumerate(row_spans):
        for cb, (c0, c1) in enumerate(col_spans):
            chambers.append(
                {
                    "row_band": rb,
                    "col_band": cb,
                    "bounds": (r0, c0, r1, c1),
                }
            )
    return chambers


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rays = [_trace_ray(g, seed) for seed in _border_seeds(g)]

    for ray in rays:
        for r, c in ray["path"]:
            out[r, c] = ray["color"]

    chambers = _chambers(g)
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bounds"]
        entrants: list[tuple[int, int]] = []
        for ray in rays:
            inside = [d for (rr, cc), d in ray["dist"].items() if r0 <= rr <= r1 and c0 <= cc <= c1]
            if inside:
                entrants.append((min(inside), int(ray["color"])))

        if len(entrants) == 1:
            fill_color = entrants[0][1]
        elif len(entrants) >= 2:
            best_distance = max(distance for distance, _color in entrants)
            winners = [color for distance, color in entrants if distance == best_distance]
            assert len(winners) == 1, "competitive chamber must have a unique farthest ray"
            fill_color = winners[0]
        else:
            continue

        block = out[r0 : r1 + 1, c0 : c1 + 1]
        block[block == BG] = fill_color

    return out, rays, chambers


def main(grid: np.ndarray) -> np.ndarray:
    """Extend border rays through wall gaps and flood-fill each chamber with the farthest entering ray color.

    Input structure:
    - A black grid is partitioned by grey orthogonal walls into rectangular chambers.
    - Every wall segment between neighboring crossings has exactly one one-cell gap.
    - Colored single pixels on the outer border start straight horizontal or vertical rays.

    Rule:
    - From each border pixel, extend a ray inward along its row or column through every non-wall cell until the next solid wall or the opposite border.
    - Keep all wall pixels grey and keep all traversed ray pixels in the ray's own color.
    - For each rectangular chamber, collect the rays that place at least one pixel inside it; if exactly one enters, flood-fill the chamber's remaining black cells with that color, and if multiple enter, use the color of the ray whose first in-chamber pixel is farthest from its border source.

    Key insight:
    - The chamber color is not decided globally per ray: a single ray can travel through several chambers, and each chamber must compare the rays again using that chamber's own entry distance.
    """
    return _solve(grid)[0]


def _build_plan() -> dict[str, Any]:
    cols = random.choice([3, 4])
    heights = [random.randint(3, 5), random.randint(3, 5)]
    widths = [random.randint(3, 5) for _ in range(cols)]

    row_spans = []
    r = 0
    for idx, height in enumerate(heights):
        row_spans.append((r, r + height - 1))
        r += height + (1 if idx == 0 else 0)

    wall_rows = [heights[0]]
    wall_cols = []
    c = widths[0]
    for width in widths[1:]:
        wall_cols.append(c)
        c += width + 1

    h = sum(heights) + 1
    w = sum(widths) + (cols - 1)

    gap_band = random.randint(0, 1)
    overlap_band = 1 - gap_band

    patterns = [None, None]
    if cols == 3:
        patterns[overlap_band] = {"kind": "overlap", "left_reach": 2, "right_reach": 2}
        patterns[gap_band] = {"kind": "gap", "left_reach": 1, "right_reach": 1}
    else:
        overlap_col = random.choice([1, 2])
        patterns[overlap_band] = {
            "kind": "overlap",
            "left_reach": overlap_col + 1,
            "right_reach": cols - overlap_col,
        }
        patterns[gap_band] = {
            "kind": "gap",
            "left_reach": random.choice([1, 2]),
            "right_reach": random.choice([1, 2]),
        }
        if patterns[gap_band]["left_reach"] + patterns[gap_band]["right_reach"] >= cols:
            if random.choice([False, True]):
                patterns[gap_band]["left_reach"] = 1
            else:
                patterns[gap_band]["right_reach"] = 1
        if patterns[gap_band]["left_reach"] + patterns[gap_band]["right_reach"] >= cols:
            patterns[gap_band]["left_reach"] = 1
            patterns[gap_band]["right_reach"] = 1

    seed_rows = []
    for r0, r1 in row_spans:
        seed_rows.append(tuple(sorted(random.sample(range(r0, r1 + 1), 2))))

    colors = list(RAY_COLORS)
    random.shuffle(colors)
    band_colors = [
        {"left": colors[0], "right": colors[1]},
        {"left": colors[2], "right": colors[3]},
    ]

    return {
        "shape": (h, w),
        "cols": cols,
        "widths": widths,
        "heights": heights,
        "row_spans": row_spans,
        "col_spans": _intervals(w, wall_cols),
        "wall_rows": wall_rows,
        "wall_cols": wall_cols,
        "patterns": patterns,
        "seed_rows": seed_rows,
        "band_colors": band_colors,
    }


def _make_vertical_gap_rows(plan: dict[str, Any], band: int) -> list[int]:
    cols = int(plan["cols"])
    left_row, right_row = plan["seed_rows"][band]
    left_reach = int(plan["patterns"][band]["left_reach"])
    right_reach = int(plan["patterns"][band]["right_reach"])
    r0, r1 = plan["row_spans"][band]
    all_rows = list(range(r0, r1 + 1))

    gap_rows: list[int | None] = [None] * (cols - 1)
    for wall_idx in range(cols - 1):
        cross_left = wall_idx < left_reach - 1
        cross_right = wall_idx >= cols - right_reach
        assert not (cross_left and cross_right)
        if cross_left:
            gap_rows[wall_idx] = left_row
        elif cross_right:
            gap_rows[wall_idx] = right_row

    for required_row in (left_row, right_row):
        if required_row not in gap_rows:
            options = [idx for idx, row in enumerate(gap_rows) if row is None]
            assert options
            gap_rows[random.choice(options)] = required_row

    spare_rows = [row for row in all_rows if row not in (left_row, right_row)]
    assert spare_rows
    for idx, row in enumerate(gap_rows):
        if row is None:
            gap_rows[idx] = random.choice(spare_rows)

    return [int(row) for row in gap_rows]


def _render_from_plan(plan: dict[str, Any]) -> np.ndarray:
    h, w = plan["shape"]
    grid = np.full((h, w), BG, dtype=int)

    for r in plan["wall_rows"]:
        grid[r, :] = WALL
    for c in plan["wall_cols"]:
        grid[:, c] = WALL

    vertical_gaps = []
    for band in range(2):
        gap_rows = _make_vertical_gap_rows(plan, band)
        vertical_gaps.append(gap_rows)
        for wall_idx, gap_row in enumerate(gap_rows):
            grid[gap_row, plan["wall_cols"][wall_idx]] = BG

    for wall_row in plan["wall_rows"]:
        for c0, c1 in plan["col_spans"]:
            grid[wall_row, random.randint(c0, c1)] = BG

    for band, (r0, _r1) in enumerate(plan["row_spans"]):
        left_row, right_row = plan["seed_rows"][band]
        grid[left_row, 0] = plan["band_colors"][band]["left"]
        grid[right_row, w - 1] = plan["band_colors"][band]["right"]

    plan["vertical_gaps"] = vertical_gaps
    return grid


def _expected_from_plan(inp: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    out = inp.copy()
    h, w = inp.shape
    wall_cols = plan["wall_cols"]
    col_spans = plan["col_spans"]

    for band in range(2):
        left_row, right_row = plan["seed_rows"][band]
        left_color = plan["band_colors"][band]["left"]
        right_color = plan["band_colors"][band]["right"]
        left_reach = int(plan["patterns"][band]["left_reach"])
        right_reach = int(plan["patterns"][band]["right_reach"])

        if left_reach == plan["cols"]:
            left_stop = w - 1
        else:
            left_stop = wall_cols[left_reach - 1] - 1
        out[left_row, 0 : left_stop + 1] = left_color

        if right_reach == plan["cols"]:
            right_stop = 0
        else:
            right_stop = wall_cols[plan["cols"] - right_reach - 1] + 1
        out[right_row, right_stop:w] = right_color

        r0, r1 = plan["row_spans"][band]
        for col_band, (c0, c1) in enumerate(col_spans):
            entrants = []
            if col_band < left_reach:
                entrants.append((c0, left_color))
            if col_band >= plan["cols"] - right_reach:
                entrants.append((w - 1 - c1, right_color))

            if not entrants:
                continue
            if len(entrants) == 1:
                fill_color = entrants[0][1]
            else:
                best_dist = max(distance for distance, _color in entrants)
                winners = [color for distance, color in entrants if distance == best_dist]
                assert len(winners) == 1
                fill_color = winners[0]

            block = out[r0 : r1 + 1, c0 : c1 + 1]
            block[block == BG] = fill_color

    return out


def _validate_generated(inp: np.ndarray, plan: dict[str, Any]) -> bool:
    try:
        out, rays, chambers = _solve(inp)
        expected = _expected_from_plan(inp, plan)
    except Exception:
        return False

    if not np.array_equal(out, expected):
        return False

    if len(rays) != 4:
        return False

    used = set()
    for ray in rays:
        for cell in ray["path"]:
            if cell in used:
                return False
            used.add(cell)

    chamber_counts = []
    competitive_winners = set()
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bounds"]
        entrants = []
        for ray in rays:
            inside = [d for (rr, cc), d in ray["dist"].items() if r0 <= rr <= r1 and c0 <= cc <= c1]
            if inside:
                entrants.append((min(inside), int(ray["color"])))
        chamber_counts.append(len(entrants))
        if len(entrants) >= 2:
            dmax = max(distance for distance, _color in entrants)
            winners = [color for distance, color in entrants if distance == dmax]
            if len(winners) != 1:
                return False
            competitive_winners.add(winners[0])

    if 0 not in chamber_counts or 1 not in chamber_counts or 2 not in chamber_counts:
        return False
    if not competitive_winners:
        return False

    rows, cols = _find_wall_lines(inp)
    if len(rows) != 1 or len(cols) != plan["cols"] - 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        plan = _build_plan()
        grid = _render_from_plan(plan)
        if _validate_generated(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid ray-corridor flood-fill puzzle")
