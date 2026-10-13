import random
from typing import Any

import numpy as np


BG = 0
SEED = 5
OBSTACLE = 7
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SAFE_CYCLE_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_cycle_bar(grid: np.ndarray) -> tuple[tuple[int, int], list[int]]:
    h, w = grid.shape
    for r in range(min(3, h)):
        for c in range(min(4, w - 2)):
            vals = [int(x) for x in grid[r, c : c + 3]]
            if BG in vals or len(set(vals)) != 3:
                continue
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 4)
            halo = grid[r0:r1, c0:c1].copy()
            halo[r - r0, c - c0 : c - c0 + 3] = BG
            if int(np.count_nonzero(halo)) == 0:
                return (r, c), vals
    raise AssertionError("expected an isolated 1x3 cycle bar in the top-left area")


def _static_length(
    shape: tuple[int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
    seed_set: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> int:
    r, c = seed
    dr, dc = direction
    steps = 0
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return steps
        if (nr, nc) in seed_set or (nr, nc) in obstacles:
            return steps
        steps += 1
        r, c = nr, nc


def _simulate(grid: np.ndarray, *, strict_counts: bool = True) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    (bar_r, bar_c), cycle = _find_cycle_bar(g)
    bar_cells = {(bar_r, bar_c + i) for i in range(3)}
    seeds = sorted(
        (int(r), int(c))
        for r, c in np.argwhere(g == SEED)
        if (int(r), int(c)) not in bar_cells
    )
    obstacles = {(int(r), int(c)) for r, c in np.argwhere(g == OBSTACLE)}
    seed_set = set(seeds)
    if strict_counts:
        assert 3 <= len(seeds) <= 5
        assert 1 <= len(obstacles) <= 2

    rays: list[dict[str, Any]] = []
    for seed_idx, seed in enumerate(seeds):
        for dir_idx, direction in enumerate(DIRS):
            rays.append(
                {
                    "seed_idx": seed_idx,
                    "seed": seed,
                    "dir_idx": dir_idx,
                    "dir": direction,
                    "cells": [],
                    "active": True,
                    "stop_reason": None,
                    "stop_info": None,
                    "static_len": _static_length((h, w), seed, direction, seed_set, obstacles),
                }
            )

    painted_owner: dict[tuple[int, int], int] = {}
    collision_events: list[dict[str, Any]] = []
    for step in range(1, h + w + 1):
        proposals: dict[tuple[int, int], list[int]] = {}
        for ray_idx, ray in enumerate(rays):
            if not ray["active"]:
                continue
            sr, sc = ray["seed"]
            dr, dc = ray["dir"]
            nr = sr + dr * step
            nc = sc + dc * step
            cell = (nr, nc)
            if not _in_bounds((h, w), nr, nc):
                ray["active"] = False
                ray["stop_reason"] = "boundary"
                continue
            if cell in obstacles:
                ray["active"] = False
                ray["stop_reason"] = "obstacle"
                continue
            if cell in seed_set:
                ray["active"] = False
                ray["stop_reason"] = "seed"
                continue
            if cell in painted_owner:
                ray["active"] = False
                ray["stop_reason"] = "painted"
                ray["stop_info"] = painted_owner[cell]
                continue
            proposals.setdefault(cell, []).append(ray_idx)

        if not proposals and not any(ray["active"] for ray in rays):
            break

        for cell, ray_ids in proposals.items():
            if len(ray_ids) > 1:
                seeds_here = tuple(sorted({rays[i]["seed_idx"] for i in ray_ids}))
                collision_events.append(
                    {"cell": cell, "step": step, "rays": tuple(ray_ids), "seed_ids": seeds_here}
                )
                for ray_idx in ray_ids:
                    rays[ray_idx]["active"] = False
                    rays[ray_idx]["stop_reason"] = "collision"
                    rays[ray_idx]["stop_info"] = tuple(i for i in ray_ids if i != ray_idx)
                continue
            ray_idx = ray_ids[0]
            rays[ray_idx]["cells"].append(cell)
            painted_owner[cell] = ray_idx

        if not any(ray["active"] for ray in rays):
            break

    out = g.copy()
    for ray in rays:
        for dist, (r, c) in enumerate(ray["cells"], start=1):
            out[r, c] = cycle[(dist - 1) % len(cycle)]
        ray["actual_len"] = len(ray["cells"])
        ray["center_color"] = cycle[(len(ray["cells"]) - 1) % len(cycle)] if ray["cells"] else None

    stamps: list[dict[str, Any]] = []
    stamped_rays = sorted(
        (ray for ray in rays if ray["cells"]),
        key=lambda ray: (ray["seed"][0], ray["seed"][1], ray["dir_idx"]),
    )
    for ray in stamped_rays:
        er, ec = ray["cells"][-1]
        r0, r1 = er - 1, er + 1
        c0, c1 = ec - 1, ec + 1
        clipped = False
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if _in_bounds((h, w), rr, cc):
                    out[rr, cc] = SEED
                else:
                    clipped = True
        out[er, ec] = int(ray["center_color"])
        stamps.append(
            {
                "seed_idx": ray["seed_idx"],
                "seed": ray["seed"],
                "endpoint": (er, ec),
                "center_color": int(ray["center_color"]),
                "clipped": clipped,
                "bbox": (r0, c0, r1, c1),
            }
        )

    for seed in seeds:
        out[seed] = SEED
    for i in range(3):
        out[bar_r, bar_c + i] = cycle[i]

    return {
        "output": out,
        "cycle": cycle,
        "bar_origin": (bar_r, bar_c),
        "bar_cells": bar_cells,
        "seeds": seeds,
        "obstacles": obstacles,
        "rays": rays,
        "stamps": stamps,
        "collisions": collision_events,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend four cycle-colored rays from every seed, then stamp each ray endpoint.

    Input structure:
    - Background is black, with one isolated 1x3 horizontal color bar near the top-left that defines a 3-color cycle.
    - Several grey seed pixels are scattered in the main field, plus one or two orange obstacle pixels.
    - Seeds are the emitters; the bar is only a visible legend for the repeating ray colors.

    Rule:
    - Every grey seed emits rays simultaneously in the four cardinal directions.
    - A ray colors its 1st, 2nd, 3rd, ... painted cells by cycling through the top-left bar left-to-right, then repeating.
    - A ray stops before coloring the next cell if that next cell would leave the grid, is an orange obstacle, is another seed, or has already been colored by a different ray; if two rays reach the same new cell in the same step, neither colors it.
    - After all rays stop, stamp a grey 3x3 square centered on each ray's last colored cell, then recolor that center back to the endpoint's cycle color; clipped stamps are cropped to the grid, later seeds in row-major order overwrite earlier stamps, and the original seeds and color bar remain unchanged.

    Key insight:
    - The endpoint color encodes the ray travel distance modulo the 3-color cycle, so collisions must be resolved first because they shorten rays and change the stamped center colors.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _dist_to_bar(seed: tuple[int, int], bar_cells: list[tuple[int, int]]) -> int:
    return min(abs(seed[0] - r) + abs(seed[1] - c) for r, c in bar_cells)


def _valid_seed(
    seed: tuple[int, int],
    seeds: list[tuple[int, int]],
    bar_cells: list[tuple[int, int]],
    *,
    forbid_rows: set[int] | None = None,
    forbid_cols: set[int] | None = None,
) -> bool:
    r, c = seed
    if _dist_to_bar(seed, bar_cells) < 4:
        return False
    if forbid_rows is not None and r in forbid_rows:
        return False
    if forbid_cols is not None and c in forbid_cols:
        return False
    return all(abs(r - rr) + abs(c - cc) >= 4 for rr, cc in seeds)


def _ray_cells_until_static_stop(
    shape: tuple[int, int],
    seed: tuple[int, int],
    direction: tuple[int, int],
    seed_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = seed
    dr, dc = direction
    cells = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc) or (nr, nc) in seed_set:
            return cells
        cells.append((nr, nc))
        r, c = nr, nc


def _pick_collision_pair(h: int, w: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    horizontal = random.choice([False, True])
    protected: set[tuple[int, int]] = set()
    if horizontal:
        gaps = [gap for gap in [4, 7, 8] if w - gap - 6 >= 4]
        if not gaps:
            raise AssertionError("grid too narrow for a protected horizontal collision pair")
        gap = random.choice(gaps)
        r = random.randint(4, h - 5)
        c1 = random.randint(4, w - gap - 6)
        c2 = c1 + gap + 1
        for cc in range(c1 + 1, c2):
            protected.add((r, cc))
        return [(r, c1), (r, c2)], protected
    gaps = [gap for gap in [4, 7, 8] if h - gap - 6 >= 4]
    if not gaps:
        raise AssertionError("grid too short for a protected vertical collision pair")
    gap = random.choice(gaps)
    c = random.randint(4, w - 5)
    r1 = random.randint(4, h - gap - 6)
    r2 = r1 + gap + 1
    for rr in range(r1 + 1, r2):
        protected.add((rr, c))
    return [(r1, c), (r2, c)], protected


def _pick_edge_seed(
    h: int,
    w: int,
    seeds: list[tuple[int, int]],
    bar_cells: list[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
) -> tuple[int, int] | None:
    candidates = []
    for r, c in (
        *[(2, cc) for cc in range(4, w - 3)],
        *[(h - 3, cc) for cc in range(4, w - 3)],
        *[(rr, 2) for rr in range(4, h - 3)],
        *[(rr, w - 3) for rr in range(4, h - 3)],
    ):
        if _valid_seed((r, c), seeds, bar_cells, forbid_rows=used_rows, forbid_cols=used_cols):
            candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _place_extra_seeds(
    h: int,
    w: int,
    seeds: list[tuple[int, int]],
    bar_cells: list[tuple[int, int]],
    total: int,
    need_clipped: bool,
) -> list[tuple[int, int]] | None:
    used_rows = {r for r, _ in seeds}
    used_cols = {c for _, c in seeds}
    if need_clipped and len(seeds) < total:
        edge_seed = _pick_edge_seed(h, w, seeds, bar_cells, used_rows, used_cols)
        if edge_seed is None:
            return None
        seeds.append(edge_seed)
        used_rows.add(edge_seed[0])
        used_cols.add(edge_seed[1])
    elif random.random() < 0.5 and len(seeds) < total:
        edge_seed = _pick_edge_seed(h, w, seeds, bar_cells, used_rows, used_cols)
        if edge_seed is not None:
            seeds.append(edge_seed)
            used_rows.add(edge_seed[0])
            used_cols.add(edge_seed[1])

    candidates = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if _valid_seed((r, c), seeds, bar_cells, forbid_rows=used_rows, forbid_cols=used_cols)
    ]
    random.shuffle(candidates)
    while len(seeds) < total and candidates:
        seed = candidates.pop()
        if not _valid_seed(seed, seeds, bar_cells, forbid_rows=used_rows, forbid_cols=used_cols):
            continue
        seeds.append(seed)
        used_rows.add(seed[0])
        used_cols.add(seed[1])
    return seeds if len(seeds) == total else None


def _is_clip_endpoint(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r in (0, h - 1) or c in (0, w - 1)


def _clip_protected_cells(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    seed_set = set(seeds)
    clip_paths = []
    for seed in seeds:
        for direction in DIRS:
            cells = _ray_cells_until_static_stop(shape, seed, direction, seed_set)
            if cells and _is_clip_endpoint(shape, cells[-1]):
                clip_paths.append(cells)
    return set(random.choice(clip_paths)) if clip_paths else set()


def _candidate_obstacle_cells(
    h: int,
    w: int,
    cycle: list[int],
    seeds: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    bar_cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    preview = _simulate(_build_grid(h, w, cycle, seeds, []), strict_counts=False)
    candidates = []
    for ray in preview["rays"]:
        for cell in ray["cells"][:4]:
            if cell in protected or cell in set(seeds):
                continue
            if _dist_to_bar(cell, bar_cells) < 4:
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    dedup = []
    seen = set()
    for cell in candidates:
        if cell not in seen:
            dedup.append(cell)
            seen.add(cell)
    return dedup


def _random_obstacle_cells(
    h: int,
    w: int,
    seeds: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    bar_cells: list[tuple[int, int]],
    taken: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    seed_set = set(seeds)
    cells = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            cell = (r, c)
            if cell in seed_set or cell in protected or cell in taken:
                continue
            if _dist_to_bar(cell, bar_cells) < 4:
                continue
            cells.append(cell)
    random.shuffle(cells)
    return cells


def _build_grid(
    h: int,
    w: int,
    cycle: list[int],
    seeds: list[tuple[int, int]],
    obstacles: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, 0:3] = np.array(cycle, dtype=int)
    for r, c in seeds:
        grid[r, c] = SEED
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    return grid


def _dynamic_mod_change(ray: dict[str, Any], rays: list[dict[str, Any]]) -> bool:
    if ray["actual_len"] <= 0 or ray["actual_len"] >= ray["static_len"]:
        return False
    if ray["actual_len"] % 3 == ray["static_len"] % 3:
        return False
    if ray["stop_reason"] == "collision":
        other_ids = {rays[i]["seed_idx"] for i in ray["stop_info"]}
        return any(seed_idx != ray["seed_idx"] for seed_idx in other_ids)
    if ray["stop_reason"] == "painted":
        return rays[ray["stop_info"]]["seed_idx"] != ray["seed_idx"]
    return False


def _stamps_avoid_sources(
    stamps: list[dict[str, Any]],
    seeds: list[tuple[int, int]],
    bar_cells: set[tuple[int, int]],
) -> bool:
    protected = set(seeds) | set(bar_cells)
    for stamp in stamps:
        r0, c0, r1, c1 = stamp["bbox"]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) in protected and (rr, cc) != stamp["endpoint"]:
                    return False
    return True


def _analyze_grid(grid: np.ndarray) -> dict[str, Any] | None:
    try:
        sim = _simulate(grid)
    except Exception:
        return None

    rays = sim["rays"]
    dynamic_change = any(_dynamic_mod_change(ray, rays) for ray in rays)
    obstacle_hit = any(ray["stop_reason"] == "obstacle" for ray in rays)
    clipped_stamp = any(stamp["clipped"] for stamp in sim["stamps"])
    stamp_colors = {stamp["center_color"] for stamp in sim["stamps"]}
    overlap_count = 0
    for i, a in enumerate(sim["stamps"]):
        ar, ac = a["endpoint"]
        for b in sim["stamps"][i + 1 :]:
            br, bc = b["endpoint"]
            if max(abs(ar - br), abs(ac - bc)) <= 2:
                overlap_count += 1

    return {
        "sim": sim,
        "dynamic_change": dynamic_change,
        "obstacle_hit": obstacle_hit,
        "clipped_stamp": clipped_stamp,
        "stamp_color_count": len(stamp_colors),
        "stamp_overlap_count": overlap_count,
        "collisions": len(sim["collisions"]),
        "stamps_ok": _stamps_avoid_sources(sim["stamps"], sim["seeds"], sim["bar_cells"]),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 19)
    w = random.randint(14, 19)
    cycle = random.sample(SAFE_CYCLE_COLORS, 3)
    bar_cells = [(0, 0), (0, 1), (0, 2)]
    need_clipped = random.random() < 0.5

    pair, protected = _pick_collision_pair(h, w)
    seed_total = random.randint(3, 5)
    seeds = _place_extra_seeds(h, w, pair.copy(), bar_cells, seed_total, need_clipped)
    if seeds is None:
        return None
    protected |= _clip_protected_cells((h, w), seeds)

    obstacle_total = random.randint(1, 2)
    active_obstacle_target = True
    obstacles: list[tuple[int, int]] = []
    taken = set(seeds)

    if active_obstacle_target:
        active_candidates = _candidate_obstacle_cells(h, w, cycle, seeds, protected, bar_cells)
        if not active_candidates:
            return None
        first = active_candidates[0]
        obstacles.append(first)
        taken.add(first)

    extra_candidates = _random_obstacle_cells(h, w, seeds, protected, bar_cells, taken)
    while len(obstacles) < obstacle_total and extra_candidates:
        cell = extra_candidates.pop()
        obstacles.append(cell)
        taken.add(cell)
    if len(obstacles) != obstacle_total:
        return None

    grid = _build_grid(h, w, cycle, seeds, obstacles)
    analysis = _analyze_grid(grid)
    if analysis is None:
        return None

    sim = analysis["sim"]
    if sim["seeds"] != sorted(seeds):
        return None
    if not analysis["dynamic_change"]:
        return None
    if analysis["collisions"] <= 0:
        return None
    if active_obstacle_target and not analysis["obstacle_hit"]:
        return None
    if need_clipped and not analysis["clipped_stamp"]:
        return None
    if analysis["stamp_color_count"] < 2:
        return None
    if len(sim["stamps"]) < 6:
        return None
    if not analysis["stamps_ok"]:
        return None
    if np.array_equal(grid, sim["output"]):
        return None
    if not np.array_equal(main(grid), sim["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid non-degenerate sample")
