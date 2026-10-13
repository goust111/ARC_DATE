from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LOOP_DIM_CANDIDATES = [
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps.append(comp)
            next_id += 1

    return labels, comps


def _closed_cavities_by_wall(grid: np.ndarray) -> tuple[np.ndarray, dict[int, list[list[tuple[int, int]]]]]:
    h, w = grid.shape
    wall_labels, _wall_comps = _label_components(grid == WALL)
    bg_labels, bg_comps = _label_components(grid == BG)
    _ = bg_labels
    cavities_by_wall: dict[int, list[list[tuple[int, int]]]] = {}

    for comp in bg_comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        boundary_ids = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == WALL:
                    boundary_ids.add(int(wall_labels[nr, nc]))
        boundary_ids.discard(0)
        if len(boundary_ids) == 1:
            wall_id = next(iter(boundary_ids))
            cavities_by_wall.setdefault(wall_id, []).append(comp)

    return wall_labels, cavities_by_wall


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-edge seeds straight down and fill only the enclosed cavity whose grey loop they first reach.

    Input structure:
    - Background is black.
    - Grey pixels form several disconnected objects: closed loop outlines in the lower area and obstacle pixels or open segments above them.
    - The top row contains several colored single-pixel seeds in distinct columns.

    Rule:
    - Each seed falls vertically within its own column until the first non-background pixel below it; it stops one cell above that hit.
    - If the hit grey pixel belongs to a grey component that encloses a background cavity, flood-fill that cavity with the seed color and remove the seed itself.
    - Otherwise the seed is stranded on its landing cell and remains visible there; all grey structure stays unchanged.

    Key insight:
    - A seed does not color the first grey object it touches directly; its vertical route only decides whether it reaches a true closed loop boundary or gets intercepted by a non-enclosing obstacle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_labels, cavities_by_wall = _closed_cavities_by_wall(g)

    out = g.copy()
    seeds = [(0, c, int(g[0, c])) for c in range(w) if int(g[0, c]) not in (BG, WALL)]
    for _r, c, _color in seeds:
        out[0, c] = BG

    for _r0, c, color in seeds:
        hit_r = None
        for r in range(1, h):
            if int(g[r, c]) != BG:
                hit_r = r
                break

        if hit_r is None:
            out[h - 1, c] = color
            continue

        landing_r = hit_r - 1
        wall_id = int(wall_labels[hit_r, c]) if int(g[hit_r, c]) == WALL else 0
        if wall_id in cavities_by_wall:
            for cavity in cavities_by_wall[wall_id]:
                for rr, cc in cavity:
                    out[rr, cc] = color
        else:
            out[landing_r, c] = color

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_loop_specs(count: int) -> list[tuple[int, int]]:
    candidates = LOOP_DIM_CANDIDATES[:]
    random.shuffle(candidates)
    specs: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for inner_h, inner_w in candidates:
        area = inner_h * inner_w
        if area in used_areas:
            continue
        specs.append((inner_h, inner_w))
        used_areas.add(area)
        if len(specs) == count:
            return specs
    raise RuntimeError("not enough distinct loop sizes")


def _draw_rectangle_loop(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> dict[str, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _paint_vertical_obstacle(grid: np.ndarray, col: int, top: int, length: int) -> list[tuple[int, int]]:
    cells = [(r, col) for r in range(top, top + length)]
    for r, c in cells:
        grid[r, c] = WALL
    return cells


def _paint_horizontal_obstacle(grid: np.ndarray, row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    cells = [(row, c) for c in range(c0, c1 + 1)]
    for r, c in cells:
        grid[r, c] = WALL
    return cells


def _oracle_from_scene(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    for seed in scene["seeds"]:
        out[0, seed["col"]] = BG
    for seed in scene["seeds"]:
        if seed["blocked"]:
            out[seed["landing_row"], seed["col"]] = seed["color"]
        else:
            loop = scene["loops"][seed["loop_idx"]]
            out[loop["top"] + 1 : loop["bottom"], loop["left"] + 1 : loop["right"]] = seed["color"]
    return out


def _validate_scene(scene: dict) -> None:
    grid = scene["grid"]
    h, w = grid.shape
    assert h <= 30 and w <= 30
    assert 3 <= len(scene["loops"]) <= 5
    assert 3 <= len(scene["seeds"]) <= 5

    loop_spans = [(loop["left"], loop["right"]) for loop in scene["loops"]]
    assert len({seed["color"] for seed in scene["seeds"]}) == len(scene["seeds"])
    assert len({seed["col"] for seed in scene["seeds"]}) == len(scene["seeds"])
    assert any(seed["blocked"] for seed in scene["seeds"])
    assert any(not seed["blocked"] for seed in scene["seeds"])

    for i, (a0, a1) in enumerate(loop_spans):
        for j, (b0, b1) in enumerate(loop_spans):
            if i >= j:
                continue
            assert a1 < b0 or b1 < a0

    for seed in scene["seeds"]:
        col = seed["col"]
        covered = [i for i, loop in enumerate(scene["loops"]) if loop["left"] < col < loop["right"]]
        assert covered == [seed["loop_idx"]]

        first_hit = next(r for r in range(1, h) if int(grid[r, col]) != BG)
        if seed["blocked"]:
            assert first_hit == seed["hit_row"]
            assert int(grid[first_hit, col]) == WALL
            assert seed["landing_row"] == first_hit - 1
            assert first_hit < scene["loops"][seed["loop_idx"]]["top"]
        else:
            loop = scene["loops"][seed["loop_idx"]]
            assert first_hit == loop["top"]
            assert seed["landing_row"] == loop["top"] - 1

    wall_labels, cavities_by_wall = _closed_cavities_by_wall(grid)
    closed_loop_ids = set(cavities_by_wall)
    assert len(closed_loop_ids) == len(scene["loops"])
    for loop in scene["loops"]:
        wall_id = int(wall_labels[loop["top"], loop["left"]])
        assert wall_id in closed_loop_ids
        cavity_lists = cavities_by_wall[wall_id]
        assert len(cavity_lists) == 1
        cavity = cavity_lists[0]
        assert len(cavity) == loop["inner_h"] * loop["inner_w"]

    expected = _oracle_from_scene(scene)
    actual = main(grid)
    assert np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    seed_count = random.randint(3, 5)
    blocked_count = random.randint(1, min(2, seed_count - 1))
    success_count = seed_count - blocked_count
    _ = success_count

    loop_specs = _sample_loop_specs(seed_count)
    outer_widths = [inner_w + 2 for _inner_h, inner_w in loop_specs]
    width_min = sum(outer_widths) + (seed_count - 1) + 2
    if width_min > 30:
        return None
    width = random.randint(max(20, width_min), 30)

    height = random.randint(18, 26)
    top_band = height // 3
    lower_start = top_band + 3
    if lower_start >= height - 4:
        return None

    max_outer_h = max(inner_h + 2 for inner_h, _inner_w in loop_specs)
    if lower_start + max_outer_h >= height:
        return None

    slack = width - (sum(outer_widths) + (seed_count - 1) + 2)
    extras = _random_composition(slack, seed_count + 1)
    gaps = [1 + extras[i + 1] for i in range(seed_count - 1)]
    left_margin = 1 + extras[0]

    grid = np.zeros((height, width), dtype=int)
    loops: list[dict[str, int]] = []
    left = left_margin
    for idx, (inner_h, inner_w) in enumerate(loop_specs):
        outer_h = inner_h + 2
        top_max = height - outer_h - 1
        if lower_start > top_max:
            return None
        top = random.randint(lower_start, top_max)
        loops.append(_draw_rectangle_loop(grid, top, left, inner_h, inner_w))
        if idx < seed_count - 1:
            left += inner_w + 2 + gaps[idx]

    loop_indices = list(range(seed_count))
    random.shuffle(loop_indices)
    blocked_loops = set(loop_indices[:blocked_count])
    seed_colors = random.sample(SEED_COLORS, seed_count)

    planned_cols: set[int] = set()
    for loop in loops:
        cols = [c for c in range(loop["left"] + 1, loop["right"]) if c not in planned_cols]
        if not cols:
            return None
        loop["seed_col"] = random.choice(cols)
        planned_cols.add(loop["seed_col"])

    all_seed_cols = [loop["seed_col"] for loop in loops]
    obstacle_cells: list[list[tuple[int, int]]] = []
    seeds = []

    for loop_idx, color in zip(range(seed_count), seed_colors):
        loop = loops[loop_idx]
        col = loop["seed_col"]
        blocked = loop_idx in blocked_loops
        landing_row = loop["top"] - 1
        hit_row = loop["top"]
        if blocked:
            obstacle_top_max = min(top_band + 1, loop["top"] - 2)
            if obstacle_top_max < 1:
                return None
            obstacle_top = random.randint(1, obstacle_top_max)
            if random.random() < 0.5:
                max_len = min(3, loop["top"] - obstacle_top - 1)
                if max_len < 1:
                    return None
                obs_len = random.randint(1, max_len)
                cells = _paint_vertical_obstacle(grid, col, obstacle_top, obs_len)
                hit_row = obstacle_top
            else:
                span = random.randint(1, 2)
                c0 = max(0, col - span)
                c1 = min(width - 1, col + span)
                used_other = {x for x in all_seed_cols if x != col}
                valid_cols = [c for c in range(c0, c1 + 1) if c not in used_other]
                if col not in valid_cols:
                    return None
                c0 = min(valid_cols)
                c1 = max(valid_cols)
                cells = _paint_horizontal_obstacle(grid, obstacle_top, c0, c1)
                hit_row = obstacle_top
            landing_row = hit_row - 1
            obstacle_cells.append(cells)

        grid[0, col] = color
        seeds.append(
            {
                "col": col,
                "color": color,
                "loop_idx": loop_idx,
                "blocked": blocked,
                "hit_row": hit_row,
                "landing_row": landing_row,
            }
        )

    scene = {
        "grid": grid,
        "loops": loops,
        "seeds": seeds,
        "obstacles": obstacle_cells,
    }

    try:
        _validate_scene(scene)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-seed cavity puzzle")
