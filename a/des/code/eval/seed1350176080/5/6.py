from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
SEED_COLORS = (BLUE, RED, GREEN, YELLOW)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _seed_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    seed_mask = (g != BG) & (g != GREY)
    coords = np.argwhere(seed_mask)
    seeds = [(int(r), int(c), int(g[r, c])) for r, c in coords]
    assert all(color in SEED_COLORS for _r, _c, color in seeds)
    return seeds


def _pairs_from_grid(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in _seed_cells(grid):
        coords_by_color.setdefault(color, []).append((r, c))

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, coords in coords_by_color.items():
        assert len(coords) in (2, 4)
        row_groups: dict[int, list[tuple[int, int]]] = {}
        col_groups: dict[int, list[tuple[int, int]]] = {}
        for coord in coords:
            row_groups.setdefault(coord[0], []).append(coord)
            col_groups.setdefault(coord[1], []).append(coord)

        color_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        usage = {coord: 0 for coord in coords}

        for group in row_groups.values():
            assert len(group) in (1, 2)
            if len(group) == 2:
                a, b = sorted(group)
                color_pairs.append((a, b))
                usage[a] += 1
                usage[b] += 1

        for group in col_groups.values():
            assert len(group) in (1, 2)
            if len(group) == 2:
                a, b = sorted(group)
                color_pairs.append((a, b))
                usage[a] += 1
                usage[b] += 1

        assert len(color_pairs) == len(coords) // 2
        assert all(count == 1 for count in usage.values())
        for a, b in color_pairs:
            pairs.append((color, a, b))

    return pairs


def _grow_from_seed(
    out: np.ndarray,
    base: np.ndarray,
    start: tuple[int, int],
    partner: tuple[int, int],
    color: int,
) -> tuple[int, int, str] | None:
    sr, sc = start
    pr, pc = partner
    if sr == pr:
        axis = "h"
        step_r, step_c = 0, 1 if sc < pc else -1
    else:
        assert sc == pc
        axis = "v"
        step_r, step_c = 1 if sr < pr else -1, 0

    cur_r, cur_c = sr, sc
    nr, nc = sr + step_r, sc + step_c
    while (nr, nc) != (pr, pc):
        cell = int(base[nr, nc])
        if cell == GREY:
            return cur_r, cur_c, axis
        if cell != BG:
            return None
        out[nr, nc] = color
        cur_r, cur_c = nr, nc
        nr += step_r
        nc += step_c
    return None


def _paint_orange_branches(out: np.ndarray, bounce: tuple[int, int, str]) -> None:
    r, c, axis = bounce
    directions = [(-1, 0), (1, 0)] if axis == "h" else [(0, -1), (0, 1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        while _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
            out[nr, nc] = ORANGE
            nr += dr
            nc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow paired same-color seed pixels toward each other, bouncing off grey blockers
    into orthogonal orange branches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(np.isin(g, (BG, GREY, *SEED_COLORS)))

    out = g.copy()
    bounces: list[tuple[int, int, str]] = []
    for color, a, b in _pairs_from_grid(g):
        bounce = _grow_from_seed(out, g, a, b, color)
        if bounce is not None:
            bounces.append(bounce)
        bounce = _grow_from_seed(out, g, b, a, color)
        if bounce is not None:
            bounces.append(bounce)

    for bounce in bounces:
        _paint_orange_branches(out, bounce)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (14, 20)
TOTAL_PAIR_RANGE = (2, 3)
MAX_SCENE_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200
BLOCKER_PROB = 0.6


def _ordered_pair(axis: str, a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if axis == "h":
        return (a, b) if a[1] < b[1] else (b, a)
    return (a, b) if a[0] < b[0] else (b, a)


def _orth_step(axis: str, sign: int) -> tuple[int, int]:
    if axis == "h":
        return sign, 0
    return 0, sign


def _corridor_sources(pair: dict) -> list[tuple[int, int]]:
    if pair["obstacle"] is None:
        return []
    axis = pair["axis"]
    obs_r, obs_c = pair["obstacle"]
    if axis == "h":
        return [(obs_r, obs_c - 1), (obs_r, obs_c + 1)]
    return [(obs_r - 1, obs_c), (obs_r + 1, obs_c)]


def _steps_to_boundary(shape: tuple[int, int], source: tuple[int, int], axis: str, sign: int) -> int:
    r, c = source
    if axis == "h":
        return r if sign < 0 else shape[0] - 1 - r
    return c if sign < 0 else shape[1] - 1 - c


def _blocker_cell(source: tuple[int, int], axis: str, sign: int, steps: int) -> tuple[int, int]:
    dr, dc = _orth_step(axis, sign)
    return source[0] + dr * steps, source[1] + dc * steps


def _render_input_from_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for pair in specs:
        (r1, c1), (r2, c2) = pair["seeds"]
        color = pair["color"]
        grid[r1, c1] = color
        grid[r2, c2] = color
        if pair["obstacle"] is not None:
            orow, ocol = pair["obstacle"]
            grid[orow, ocol] = GREY
            for key, steps in pair["blockers"].items():
                if steps is None:
                    continue
                source = _corridor_sources(pair)[key[0]]
                br, bc = _blocker_cell(source, pair["axis"], key[1], steps)
                grid[br, bc] = GREY
    return grid


def _paint_pair_from_spec(out: np.ndarray, pair: dict) -> None:
    axis = pair["axis"]
    color = pair["color"]
    (r1, c1), (r2, c2) = pair["seeds"]
    obstacle = pair["obstacle"]

    if axis == "h":
        if obstacle is None:
            out[r1, c1 : c2 + 1] = color
        else:
            obs_c = obstacle[1]
            out[r1, c1:obs_c] = color
            out[r1, obs_c + 1 : c2 + 1] = color
    else:
        if obstacle is None:
            out[r1 : r2 + 1, c1] = color
        else:
            obs_r = obstacle[0]
            out[r1:obs_r, c1] = color
            out[obs_r + 1 : r2 + 1, c1] = color

    if obstacle is None:
        return

    sources = _corridor_sources(pair)
    for key, steps in pair["blockers"].items():
        source_idx, sign = key
        source = sources[source_idx]
        max_steps = _steps_to_boundary(out.shape, source, axis, sign)
        limit = max_steps if steps is None else steps - 1
        dr, dc = _orth_step(axis, sign)
        for k in range(1, limit + 1):
            rr = source[0] + dr * k
            cc = source[1] + dc * k
            out[rr, cc] = ORANGE


def _render_output_from_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    out = _render_input_from_specs(height, width, specs)
    for pair in specs:
        _paint_pair_from_spec(out, pair)
    return out


def _occupied_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
    }


def _dilate_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        out.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                out.add((nr, nc))
    return out


def _sample_color_plan() -> list[tuple[int, int]]:
    total_pairs = random.randint(*TOTAL_PAIR_RANGE)
    valid_color_counts = [count for count in range(1, min(3, total_pairs) + 1) if total_pairs <= 2 * count]
    num_colors = random.choice(valid_color_counts)
    colors = random.sample(SEED_COLORS, num_colors)
    counts = [1] * num_colors
    extra = total_pairs - num_colors
    for idx in random.sample(range(num_colors), extra):
        counts[idx] += 1
    return list(zip(colors, counts))


def _sample_branch_blockers(shape: tuple[int, int], pair: dict) -> dict[tuple[int, int], int | None]:
    blockers: dict[tuple[int, int], int | None] = {}
    if pair["obstacle"] is None:
        return blockers
    for source_idx, source in enumerate(_corridor_sources(pair)):
        for sign in (-1, 1):
            max_steps = _steps_to_boundary(shape, source, pair["axis"], sign)
            if max_steps >= 2 and random.random() < BLOCKER_PROB:
                blockers[(source_idx, sign)] = random.randint(2, max_steps)
            else:
                blockers[(source_idx, sign)] = None
    return blockers


def _sample_pair_spec(
    height: int,
    width: int,
    color: int,
    axis: str,
    with_obstacle: bool,
    used_rows: set[int],
    used_cols: set[int],
) -> dict | None:
    if axis == "h":
        valid_rows = [r for r in range(1, height - 1) if r not in used_rows]
        valid_cols = [c for c in range(1, width - 1) if c not in used_cols]
        if not valid_rows or len(valid_cols) < 2:
            return None
        min_span = 4 if with_obstacle else 3
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            row = random.choice(valid_rows)
            c1, c2 = sorted(random.sample(valid_cols, 2))
            if c2 - c1 < min_span:
                continue
            seeds = _ordered_pair(axis, (row, c1), (row, c2))
            obstacle = None
            if with_obstacle:
                obstacle = (row, random.randint(c1 + 2, c2 - 2))
            pair = {"color": color, "axis": axis, "seeds": seeds, "obstacle": obstacle, "blockers": {}}
            pair["blockers"] = _sample_branch_blockers((height, width), pair)
            return pair
    else:
        valid_rows = [r for r in range(1, height - 1) if r not in used_rows]
        valid_cols = [c for c in range(1, width - 1) if c not in used_cols]
        if len(valid_rows) < 2 or not valid_cols:
            return None
        min_span = 4 if with_obstacle else 3
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            col = random.choice(valid_cols)
            r1, r2 = sorted(random.sample(valid_rows, 2))
            if r2 - r1 < min_span:
                continue
            seeds = _ordered_pair(axis, (r1, col), (r2, col))
            obstacle = None
            if with_obstacle:
                obstacle = (random.randint(r1 + 2, r2 - 2), col)
            pair = {"color": color, "axis": axis, "seeds": seeds, "obstacle": obstacle, "blockers": {}}
            pair["blockers"] = _sample_branch_blockers((height, width), pair)
            return pair
    return None


def _scene_requests() -> list[dict]:
    requests: list[dict] = []
    for color, count in _sample_color_plan():
        axis = random.choice(("h", "v"))
        requests.extend({"color": color, "axis": axis} for _ in range(count))

    obstacle_count = random.randint(1, len(requests) - 1)
    obstacle_flags = [True] * obstacle_count + [False] * (len(requests) - obstacle_count)
    random.shuffle(obstacle_flags)
    for req, flag in zip(requests, obstacle_flags, strict=False):
        req["with_obstacle"] = flag

    requests.sort(key=lambda req: int(req["with_obstacle"]), reverse=True)
    return requests


def _branch_stats(specs: list[dict]) -> tuple[int, int]:
    blocked = 0
    open_ended = 0
    for pair in specs:
        for steps in pair["blockers"].values():
            if steps is None:
                open_ended += 1
            else:
                blocked += 1
    return blocked, open_ended


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    specs: list[dict] = []
    forbidden: set[tuple[int, int]] = set()
    same_color_rows: dict[int, set[int]] = {}
    same_color_cols: dict[int, set[int]] = {}

    for request in _scene_requests():
        color = request["color"]
        axis = request["axis"]
        used_rows = same_color_rows.setdefault(color, set())
        used_cols = same_color_cols.setdefault(color, set())

        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            pair = _sample_pair_spec(height, width, color, axis, request["with_obstacle"], used_rows, used_cols)
            if pair is None:
                break
            pair_output = _render_output_from_specs(height, width, [pair])
            cells = _occupied_cells(pair_output)
            if cells & forbidden:
                continue
            specs.append(pair)
            for sr, sc in pair["seeds"]:
                used_rows.add(sr)
                used_cols.add(sc)
            forbidden |= _dilate_cells(cells, height, width)
            placed = True
            break
        if not placed:
            return None

    if not any(pair["obstacle"] is not None for pair in specs):
        return None
    if not any(pair["obstacle"] is None for pair in specs):
        return None

    blocked_rays, open_rays = _branch_stats(specs)
    if blocked_rays == 0 or open_rays == 0:
        return None

    inp = _render_input_from_specs(height, width, specs)
    expected = _render_output_from_specs(height, width, specs)

    try:
        inferred_pairs = _pairs_from_grid(inp)
    except AssertionError:
        return None

    if len(inferred_pairs) != len(specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
