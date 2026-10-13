from __future__ import annotations

import heapq
import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

PLAY_H_RANGE = (11, 17)
WIDTH_RANGE = (14, 22)
WALL_COUNT_RANGE = (3, 6)
WALL_LEN_RANGE = (3, 8)
GEN_RETRIES = 500


def _legend_runs(row: np.ndarray) -> list[tuple[int, int]] | None:
    runs: list[tuple[int, int]] = []
    c = 0
    while c < len(row):
        val = int(row[c])
        if val == BLACK:
            c += 1
            continue
        if val == GREY:
            return None
        start = c
        while c < len(row) and int(row[c]) != BLACK:
            if int(row[c]) == GREY:
                return None
            c += 1
        if c - start != 2:
            return None
        runs.append((start, c - 1))
    if 2 <= len(runs) <= 3:
        return runs
    return None


def _extract_legend(grid: np.ndarray) -> tuple[int, dict[int, int], list[tuple[int, int]]]:
    h = int(grid.shape[0])
    for r in range(h - 1, -1, -1):
        runs = _legend_runs(grid[r])
        if runs is None:
            continue
        mapping: dict[int, int] = {}
        for c0, c1 in runs:
            mapping[int(grid[r, c0])] = int(grid[r, c1])
        if len(mapping) == len(runs):
            return int(r), mapping, runs
    raise AssertionError("failed to locate legend row")


def _in_play(r: int, c: int, play_h: int, width: int) -> bool:
    return 0 <= int(r) < int(play_h) and 0 <= int(c) < int(width)


def _record_stop(
    meta: dict | None,
    base: np.ndarray,
    play_h: int,
    claim_dist: np.ndarray,
    claim_ray: np.ndarray,
    arrival_dist: int,
    ray_id: int,
    r: int,
    c: int,
) -> None:
    if meta is None:
        return
    width = int(base.shape[1])
    if not _in_play(r, c, play_h, width):
        meta["boundary_stops"] += 1
        return
    if int(base[r, c]) == GREY:
        meta["second_wall_stops"] += 1
        return
    if int(claim_dist[r, c]) != -1:
        meta["collision_stops"] += 1
        if int(claim_dist[r, c]) == int(arrival_dist) and int(claim_ray[r, c]) != int(ray_id):
            meta["ambiguous"] += 1
        return
    meta["static_block_stops"] += 1


def _simulate(grid: np.ndarray, *, collect_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    legend_r, mapping, _runs = _extract_legend(g)
    play_h = int(legend_r)
    assert play_h > 0, "legend row must leave room for a play area"

    seeds: list[tuple[int, int, int]] = []
    for r in range(play_h):
        for c in range(w):
            val = int(g[r, c])
            if val not in (BLACK, GREY):
                seeds.append((int(r), int(c), int(val)))
    assert seeds, "expected at least one colored seed"

    out = g.copy()
    claim_dist = np.full((play_h, w), -1, dtype=int)
    claim_ray = np.full((play_h, w), -1, dtype=int)
    events: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []

    meta = None
    if collect_meta:
        meta = {
            "seed_count": len(seeds),
            "legend_pairs": len(mapping),
            "drawn": 0,
            "bounced_cells": 0,
            "bounced_rays": set(),
            "boundary_stops": 0,
            "collision_stops": 0,
            "second_wall_stops": 0,
            "static_block_stops": 0,
            "unmapped_wall_stops": 0,
            "ambiguous": 0,
        }

    def schedule_from(
        cur_r: int,
        cur_c: int,
        dist: int,
        seed_id: int,
        dir_id: int,
        ray_id: int,
        dr: int,
        dc: int,
        bounced: int,
        color: int,
    ) -> None:
        nr = int(cur_r + dr)
        nc = int(cur_c + dc)
        if not _in_play(nr, nc, play_h, w):
            if meta is not None:
                meta["boundary_stops"] += 1
            return

        if int(g[nr, nc]) == GREY:
            if bounced:
                if meta is not None:
                    meta["second_wall_stops"] += 1
                return
            if color not in mapping:
                if meta is not None:
                    meta["unmapped_wall_stops"] += 1
                return
            ar = int(nr + dr)
            ac = int(nc + dc)
            if not _in_play(ar, ac, play_h, w):
                if meta is not None:
                    meta["boundary_stops"] += 1
                return
            if int(out[ar, ac]) != BLACK or int(g[ar, ac]) != BLACK:
                _record_stop(meta, g, play_h, claim_dist, claim_ray, dist + 2, ray_id, ar, ac)
                return
            heapq.heappush(
                events,
                (dist + 2, seed_id, dir_id, ray_id, ar, ac, dr, dc, 1, int(mapping[color])),
            )
            return

        if int(out[nr, nc]) != BLACK or int(g[nr, nc]) != BLACK:
            _record_stop(meta, g, play_h, claim_dist, claim_ray, dist + 1, ray_id, nr, nc)
            return

        heapq.heappush(events, (dist + 1, seed_id, dir_id, ray_id, nr, nc, dr, dc, bounced, color))

    ray_id = 0
    for seed_id, (r, c, color) in enumerate(seeds):
        for dir_id, (dr, dc) in enumerate(DIRS):
            schedule_from(int(r), int(c), 0, int(seed_id), int(dir_id), int(ray_id), int(dr), int(dc), 0, int(color))
            ray_id += 1

    while events:
        dist, seed_id, dir_id, ray_id, r, c, dr, dc, bounced, color = heapq.heappop(events)
        if int(out[r, c]) != BLACK:
            _record_stop(meta, g, play_h, claim_dist, claim_ray, dist, ray_id, r, c)
            continue
        out[r, c] = int(color)
        claim_dist[r, c] = int(dist)
        claim_ray[r, c] = int(ray_id)
        if meta is not None:
            meta["drawn"] += 1
            if bounced:
                meta["bounced_cells"] += 1
                meta["bounced_rays"].add(int(ray_id))
        schedule_from(int(r), int(c), int(dist), int(seed_id), int(dir_id), int(ray_id), int(dr), int(dc), int(bounced), int(color))

    if meta is not None:
        meta["bounced_ray_count"] = len(meta["bounced_rays"])
    return out if meta is None else (out, meta)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom legend, then cast four orthogonal rays from every seed.
    Rays paint black cells, may pass through one grey wall, and use the legend
    to change color after that single bounce.
    """
    return _simulate(grid, collect_meta=False)


def _distance_to_edge(pos: tuple[int, int], direction: tuple[int, int], play_h: int, width: int) -> int:
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    if dr == -1:
        return int(r)
    if dr == 1:
        return int(play_h - 1 - r)
    if dc == -1:
        return int(c)
    return int(width - 1 - c)


def _ray_cell(pos: tuple[int, int], direction: tuple[int, int], step: int) -> tuple[int, int]:
    r, c = map(int, pos)
    dr, dc = map(int, direction)
    return int(r + dr * step), int(c + dc * step)


def _ray_cells(pos: tuple[int, int], direction: tuple[int, int], start_step: int, end_step: int) -> list[tuple[int, int]]:
    return [_ray_cell(pos, direction, step) for step in range(int(start_step), int(end_step) + 1)]


def _sample_seed_positions(rng: random.Random, play_h: int, width: int, seed_count: int) -> list[tuple[int, int]] | None:
    rows = list(range(2, play_h - 2))
    cols = list(range(2, width - 2))
    if len(rows) < seed_count or len(cols) < seed_count:
        return None
    for _ in range(300):
        chosen_rows = sorted(rng.sample(rows, seed_count))
        chosen_cols = rng.sample(cols, seed_count)
        rng.shuffle(chosen_cols)
        positions = [(int(r), int(c)) for r, c in zip(chosen_rows, chosen_cols)]
        ok = True
        for (r1, c1), (r2, c2) in combinations(positions, 2):
            if abs(int(r1) - int(r2)) < 2 or abs(int(c1) - int(c2)) < 2:
                ok = False
                break
            if abs(int(r1) - int(r2)) == abs(int(c1) - int(c2)):
                ok = False
                break
        if ok:
            return positions
    return None


def _collision_plan(rng: random.Random, seeds: list[tuple[int, int]]) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    pairs = list(combinations(range(len(seeds)), 2))
    rng.shuffle(pairs)
    for ia, ib in pairs:
        ra, ca = map(int, seeds[ia])
        rb, cb = map(int, seeds[ib])
        inter = (int(ra), int(cb))
        path = set()
        step_c = 1 if cb > ca else -1
        for cc in range(ca + step_c, cb + step_c, step_c):
            path.add((int(ra), int(cc)))
        step_r = 1 if ra > rb else -1
        for rr in range(rb + step_r, ra + step_r, step_r):
            path.add((int(rr), int(cb)))
        if inter not in path:
            path.add(inter)
        return (ia, ib), path
    return None


def _axis_conflict(
    seeds: list[tuple[int, int]],
    seed_idx: int,
    direction: tuple[int, int],
    start_step: int,
    end_step: int,
) -> bool:
    r0, c0 = map(int, seeds[seed_idx])
    dr, dc = map(int, direction)
    for other_idx, (r, c) in enumerate(seeds):
        if other_idx == seed_idx:
            continue
        if dr == 0:
            delta = int((c - c0) * dc)
        else:
            delta = int((r - r0) * dr)
        if int(start_step) <= delta <= int(end_step):
            return True
    return False


def _pick_bounce_plan(
    rng: random.Random,
    seeds: list[tuple[int, int]],
    play_h: int,
    width: int,
    collision_cells: set[tuple[int, int]],
) -> dict | None:
    modes = ["boundary", "second_wall"]
    rng.shuffle(modes)
    for mode in modes:
        plans: list[dict] = []
        for seed_idx, pos in enumerate(seeds):
            for dir_id, direction in enumerate(DIRS):
                edge_dist = _distance_to_edge(pos, direction, play_h, width)
                if mode == "boundary":
                    if edge_dist < 4:
                        continue
                    for first_step in range(2, min(4, edge_dist - 1) + 1):
                        path = set(_ray_cells(pos, direction, 1, edge_dist))
                        if path & collision_cells:
                            continue
                        if _axis_conflict(seeds, seed_idx, direction, 1, edge_dist):
                            continue
                        plans.append(
                            {
                                "mode": mode,
                                "seed_idx": int(seed_idx),
                                "dir_id": int(dir_id),
                                "direction": direction,
                                "first_step": int(first_step),
                            }
                        )
                else:
                    if edge_dist < 6:
                        continue
                    for first_step in range(2, min(4, edge_dist - 2) + 1):
                        for second_step in range(first_step + 2, edge_dist):
                            path = set(_ray_cells(pos, direction, 1, second_step))
                            if path & collision_cells:
                                continue
                            if _axis_conflict(seeds, seed_idx, direction, 1, second_step - 1):
                                continue
                            plans.append(
                                {
                                    "mode": mode,
                                    "seed_idx": int(seed_idx),
                                    "dir_id": int(dir_id),
                                    "direction": direction,
                                    "first_step": int(first_step),
                                    "second_step": int(second_step),
                                }
                            )
        if plans:
            return rng.choice(plans)
    return None


def _seed_buffer(seeds: list[tuple[int, int]], play_h: int, width: int) -> set[tuple[int, int]]:
    buf: set[tuple[int, int]] = set()
    for r, c in seeds:
        for rr in range(int(r) - 1, int(r) + 2):
            for cc in range(int(c) - 1, int(c) + 2):
                if _in_play(rr, cc, play_h, width):
                    buf.add((int(rr), int(cc)))
    return buf


def _can_place_segment(
    cells: list[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    seed_buf: set[tuple[int, int]],
    allow: set[tuple[int, int]] | None = None,
) -> bool:
    allow = allow or set()
    for cell in cells:
        if cell in wall_cells:
            return False
        if cell in seed_buf:
            return False
        if cell in blocked and cell not in allow:
            return False
    return True


def _crossing_segment(
    rng: random.Random,
    anchor: tuple[int, int],
    orientation: str,
    play_h: int,
    width: int,
    wall_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    seed_buf: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    ar, ac = map(int, anchor)
    max_len = min(WALL_LEN_RANGE[1], play_h if orientation == "v" else width)
    lengths = list(range(WALL_LEN_RANGE[0], max_len + 1))
    rng.shuffle(lengths)
    for length in lengths:
        if orientation == "v":
            min_top = max(0, ar - length + 1)
            max_top = min(ar, play_h - length)
            starts = list(range(min_top, max_top + 1))
            rng.shuffle(starts)
            for top in starts:
                cells = [(int(rr), int(ac)) for rr in range(int(top), int(top + length))]
                if _can_place_segment(cells, wall_cells, blocked, seed_buf, allow={anchor}):
                    return cells
        else:
            min_left = max(0, ac - length + 1)
            max_left = min(ac, width - length)
            starts = list(range(min_left, max_left + 1))
            rng.shuffle(starts)
            for left in starts:
                cells = [(int(ar), int(cc)) for cc in range(int(left), int(left + length))]
                if _can_place_segment(cells, wall_cells, blocked, seed_buf, allow={anchor}):
                    return cells
    return None


def _random_segment(
    rng: random.Random,
    play_h: int,
    width: int,
    wall_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    seed_buf: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    for _ in range(200):
        orientation = rng.choice(["h", "v"])
        max_len = min(WALL_LEN_RANGE[1], width if orientation == "h" else play_h)
        if max_len < WALL_LEN_RANGE[0]:
            return None
        length = rng.randint(WALL_LEN_RANGE[0], max_len)
        if orientation == "h":
            r = rng.randint(0, play_h - 1)
            c0 = rng.randint(0, width - length)
            cells = [(int(r), int(c)) for c in range(int(c0), int(c0 + length))]
        else:
            c = rng.randint(0, width - 1)
            r0 = rng.randint(0, play_h - length)
            cells = [(int(r), int(c)) for r in range(int(r0), int(r0 + length))]
        if _can_place_segment(cells, wall_cells, blocked, seed_buf):
            return cells
    return None


def _place_legend(rng: random.Random, width: int, mapping: dict[int, int]) -> np.ndarray:
    row = np.full(width, BLACK, dtype=int)
    pairs = list(mapping.items())
    rng.shuffle(pairs)
    gaps = [rng.randint(1, 3) for _ in range(max(0, len(pairs) - 1))]
    needed = 2 * len(pairs) + sum(gaps)
    left = rng.randint(0, width - needed)
    c = left
    for idx, (src, dst) in enumerate(pairs):
        row[c] = int(src)
        row[c + 1] = int(dst)
        c += 2
        if idx < len(gaps):
            c += int(gaps[idx])
    return row


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3])
    seed_count = rng.randint(pair_count, 4)
    play_h = rng.randint(*PLAY_H_RANGE)
    width = rng.randint(*WIDTH_RANGE)

    min_legend_width = 2 * pair_count + (pair_count - 1)
    if width < min_legend_width + 2:
        return None

    colors = rng.sample(PALETTE, 2 * pair_count)
    incident = colors[:pair_count]
    reflected = colors[pair_count:]
    mapping = {int(src): int(dst) for src, dst in zip(incident, reflected)}

    seeds = _sample_seed_positions(rng, play_h, width, seed_count)
    if seeds is None:
        return None

    collision = _collision_plan(rng, seeds)
    if collision is None:
        return None
    _pair, collision_cells = collision

    bounce_plan = _pick_bounce_plan(rng, seeds, play_h, width, collision_cells)
    if bounce_plan is None:
        return None

    blocked = set(collision_cells)
    bounce_seed = seeds[int(bounce_plan["seed_idx"])]
    direction = bounce_plan["direction"]
    first_step = int(bounce_plan["first_step"])
    if bounce_plan["mode"] == "boundary":
        stop_step = _distance_to_edge(bounce_seed, direction, play_h, width)
    else:
        stop_step = int(bounce_plan["second_step"])
    for cell in _ray_cells(bounce_seed, direction, 1, stop_step):
        blocked.add(cell)
    first_anchor = _ray_cell(bounce_seed, direction, first_step)
    blocked.discard(first_anchor)

    second_anchor = None
    if bounce_plan["mode"] == "second_wall":
        second_anchor = _ray_cell(bounce_seed, direction, int(bounce_plan["second_step"]))
        blocked.discard(second_anchor)

    seed_buf = _seed_buffer(seeds, play_h, width)
    wall_cells: set[tuple[int, int]] = set()
    wall_segments: list[list[tuple[int, int]]] = []

    primary_orientation = "v" if int(direction[0]) == 0 else "h"
    first_segment = _crossing_segment(
        rng,
        first_anchor,
        primary_orientation,
        play_h,
        width,
        wall_cells,
        blocked,
        seed_buf,
    )
    if first_segment is None:
        return None
    wall_segments.append(first_segment)
    wall_cells.update(first_segment)

    if second_anchor is not None:
        second_segment = _crossing_segment(
            rng,
            second_anchor,
            primary_orientation,
            play_h,
            width,
            wall_cells,
            blocked,
            seed_buf,
        )
        if second_segment is None:
            return None
        wall_segments.append(second_segment)
        wall_cells.update(second_segment)

    target_wall_count = rng.randint(max(WALL_COUNT_RANGE[0], len(wall_segments)), WALL_COUNT_RANGE[1])
    while len(wall_segments) < target_wall_count:
        extra = _random_segment(rng, play_h, width, wall_cells, blocked, seed_buf)
        if extra is None:
            return None
        wall_segments.append(extra)
        wall_cells.update(extra)

    seed_colors = incident.copy()
    while len(seed_colors) < seed_count:
        seed_colors.append(rng.choice(incident))
    rng.shuffle(seed_colors)

    grid = np.full((play_h + 2, width), BLACK, dtype=int)
    for seg in wall_segments:
        for r, c in seg:
            grid[r, c] = GREY
    for (r, c), color in zip(seeds, seed_colors):
        if int(grid[r, c]) != BLACK:
            return None
        grid[r, c] = int(color)
    grid[play_h + 1] = _place_legend(rng, width, mapping)

    simulated, meta = _simulate(grid, collect_meta=True)
    if meta["ambiguous"] != 0:
        return None
    if meta["drawn"] < 10 or meta["drawn"] > (play_h * width) // 2:
        return None
    if meta["bounced_cells"] < 2 or meta["bounced_ray_count"] < 1:
        return None
    if meta["collision_stops"] < 1:
        return None
    if meta["boundary_stops"] < 1:
        return None
    if bounce_plan["mode"] == "second_wall" and meta["second_wall_stops"] < 1:
        return None
    if np.array_equal(simulated, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
