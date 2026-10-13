from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8

TOWER_COLORS = (RED, BLUE, GREEN)
INPUT_COLORS = {BLACK, BLUE, RED, GREEN, GREY}
RAY_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_CHAINS = {
    RED: (RED, ORANGE, YELLOW),
    BLUE: (BLUE, AZURE, GREY),
    GREEN: (GREEN, YELLOW, ORANGE),
}

GEN_RETRIES = 400
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
EXTRA_WALLS_RANGE = (0, 3)
EXTRA_TOWER_PROB = 0.35


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _tower_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(np.isin(grid, TOWER_COLORS)):
        towers.append((int(r), int(c), int(grid[r, c])))
    return towers


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    direction: tuple[int, int],
    tower_id: int,
) -> list[tuple[int, int, int, int, int, int]]:
    h, w = grid.shape
    dr, dc = direction
    chain = COLOR_CHAINS[color]
    attenuation = 0
    dist = 0
    r, c = start
    claims: list[tuple[int, int, int, int, int, int]] = []

    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds(h, w, nr, nc):
            break
        r, c = nr, nc
        dist += 1

        if int(grid[r, c]) == GREY:
            while True:
                nr, nc = r + dr, c + dc
                if not _in_bounds(h, w, nr, nc):
                    return claims
                if int(grid[nr, nc]) != GREY:
                    break
                r, c = nr, nc
                dist += 1

            attenuation += 1
            if attenuation >= len(chain):
                break

            r, c = nr, nc
            dist += 1

        if int(grid[r, c]) == BLACK:
            claims.append((int(r), int(c), int(attenuation), int(dist), int(chain[attenuation]), int(tower_id)))

    return claims


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= INPUT_COLORS, "unexpected input colors"

    towers = _tower_cells(g)
    assert towers, "expected at least one signal tower"

    claims_by_cell: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    tower_claim_counts = Counter()
    for tower_id, (r, c, color) in enumerate(towers):
        for direction in RAY_DIRS:
            for rr, cc, attenuation, dist, ray_color, source_id in _trace_ray(g, (r, c), color, direction, tower_id):
                claims_by_cell[(int(rr), int(cc))].append(
                    (int(source_id), int(attenuation), int(dist), int(ray_color))
                )
                tower_claim_counts[int(source_id)] += 1

    out = g.copy()
    winning_attenuations: Counter[int] = Counter()
    freshness_conflicts = 0
    distance_conflicts = 0
    tie_conflicts = 0
    contested_cells = 0

    for (r, c), cell_claims in claims_by_cell.items():
        if len(cell_claims) > 1:
            contested_cells += 1

        best_att, best_dist = min((att, dist) for _tower_id, att, dist, _color in cell_claims)
        best_claims = [
            claim for claim in cell_claims if (int(claim[1]), int(claim[2])) == (int(best_att), int(best_dist))
        ]

        if len(best_claims) > 1:
            tie_conflicts += 1
            continue

        winner = best_claims[0]
        out[r, c] = int(winner[3])
        winning_attenuations[int(winner[1])] += 1

        if len(cell_claims) > 1:
            other_claims = [claim for claim in cell_claims if claim is not winner]
            if any(int(claim[1]) > int(best_att) for claim in other_claims):
                freshness_conflicts += 1
            same_att_farther = [
                claim for claim in other_claims if int(claim[1]) == int(best_att) and int(claim[2]) > int(best_dist)
            ]
            if same_att_farther:
                distance_conflicts += 1

    stats = {
        "towers": towers,
        "winning_attenuations": winning_attenuations,
        "freshness_conflicts": int(freshness_conflicts),
        "distance_conflicts": int(distance_conflicts),
        "tie_conflicts": int(tie_conflicts),
        "contested_cells": int(contested_cells),
        "colored_count": int(np.sum((g == BLACK) & (out != BLACK))),
        "tower_claim_counts": tower_claim_counts,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast orthogonal rays from colored towers, attenuating only when rays pass through grey walls.

    Rays paint only original black cells. Lower attenuation beats higher attenuation; if the
    attenuation is tied, the closer tower wins; if both are tied, the pixel stays black.
    """
    out, _stats = _simulate(grid)
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _axis_path_exclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(int(ar), int(c)) for c in range(int(ac + step), int(bc), int(step))]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(int(r), int(ac)) for r in range(int(ar + step), int(br), int(step))]
    raise AssertionError("points must be axis-aligned")


def _path_to_edge(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(h, w, r, c):
            break
        cells.append((int(r), int(c)))
    return cells


def _segment_cells(anchor: tuple[int, int], orientation: str, length: int) -> list[tuple[int, int]]:
    r, c = anchor
    if orientation == "h":
        return [(int(r), int(c + i)) for i in range(length)]
    return [(int(r + i), int(c)) for i in range(length)]


def _can_place_tower(grid: np.ndarray, pos: tuple[int, int], towers: list[tuple[int, int]]) -> bool:
    r, c = pos
    if int(grid[r, c]) != BLACK:
        return False
    return all(_chebyshev((r, c), other) >= 2 for other in towers)


def _can_place_wall(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    towers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        if (int(r), int(c)) in protected:
            return False
        for tr, tc in towers:
            if _chebyshev((int(r), int(c)), (int(tr), int(tc))) <= 1:
                return False
        for dr, dc in RAY_DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) == GREY:
                return False
    return True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _place_perpendicular_wall(
    grid: np.ndarray,
    intersection: tuple[int, int],
    ray_dir: tuple[int, int],
    towers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    ir, ic = intersection
    orientation = "v" if ray_dir[0] == 0 else "h"
    lengths = list(range(2, 6))
    rng.shuffle(lengths)

    for length in lengths:
        offsets = list(range(length))
        rng.shuffle(offsets)
        for offset in offsets:
            if orientation == "v":
                start = (int(ir - offset), int(ic))
            else:
                start = (int(ir), int(ic - offset))
            cells = _segment_cells(start, orientation, int(length))
            if not any((int(r), int(c)) == (int(ir), int(ic)) for r, c in cells):
                continue
            if _can_place_wall(grid, cells, towers, protected):
                _paint_cells(grid, cells, GREY)
                return cells
    return None


def _choose_intervals(path_len: int, kinds: list[str], rng: random.Random) -> list[tuple[int, int, str]] | None:
    lengths: list[int] = []
    for kind in kinds:
        if kind == "collinear":
            lengths.append(rng.randint(2, 4))
        else:
            lengths.append(1)
    required = sum(lengths) + len(kinds) + 1
    if path_len < required:
        return None

    gaps = [1] * (len(kinds) + 1)
    extra = path_len - required
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1
    rng.shuffle(gaps)

    intervals: list[tuple[int, int, str]] = []
    pos = gaps[0]
    for i, (kind, length) in enumerate(zip(kinds, lengths)):
        start = pos
        end = start + length - 1
        intervals.append((int(start), int(end), kind))
        pos = end + 1 + gaps[i + 1]
    return intervals


def _place_path_walls(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    ray_dir: tuple[int, int],
    towers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    kinds: list[str],
    rng: random.Random,
) -> list[tuple[int, int]]:
    for _ in range(80):
        intervals = _choose_intervals(len(path), kinds, rng)
        if intervals is None:
            return []

        snapshot = grid.copy()
        placed: list[tuple[int, int]] = []
        ok = True
        for start, end, kind in intervals:
            if kind == "collinear":
                cells = path[start : end + 1]
                if not _can_place_wall(snapshot, cells, towers, protected):
                    ok = False
                    break
                _paint_cells(snapshot, cells, GREY)
                placed.extend(cells)
            else:
                intersection = path[start]
                wall = _place_perpendicular_wall(snapshot, intersection, ray_dir, towers, protected, rng)
                if wall is None:
                    ok = False
                    break
                placed.extend(wall)
        if ok:
            grid[:, :] = snapshot
            return placed
    return []


def _random_focus_pair(
    shape: tuple[int, int],
    mode: str,
    rng: random.Random,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    list[str],
    list[str],
] | None:
    h, w = shape
    for _ in range(200):
        focus = (rng.randint(3, h - 4), rng.randint(3, w - 4))
        a_side = rng.choice(("left", "right"))
        b_side = rng.choice(("up", "down"))

        max_da = focus[1] - 1 if a_side == "left" else w - focus[1] - 2
        max_db = focus[0] - 1 if b_side == "up" else h - focus[0] - 2
        if max_da < 2 or max_db < 2:
            continue

        if mode == "tie":
            if min(max_da, max_db) < 3:
                continue
            shared = rng.randint(3, min(max_da, max_db))
            da = db = shared
            a_kinds: list[str] = []
            b_kinds = []
        elif mode == "distance":
            d_lo = 2
            d_hi = min(max_da, max_db, 5)
            if d_hi < d_lo:
                continue
            da = rng.randint(d_lo, min(max_da, 5))
            choices = [d for d in range(d_lo, min(max_db, 5) + 1) if d != da]
            if not choices:
                continue
            db = rng.choice(choices)
            a_kinds = []
            b_kinds = []
        else:
            fresh_is_a = rng.choice((True, False))
            a_kinds = []
            b_kinds = []
            if fresh_is_a:
                if max_da < 3:
                    continue
                da = rng.randint(3, min(max_da, 6))
                db = rng.randint(2, min(max_db, 5))
                b_kinds = [rng.choice(["perp", "collinear"]) if db >= 4 else "perp"]
            else:
                if max_db < 3:
                    continue
                db = rng.randint(3, min(max_db, 6))
                da = rng.randint(2, min(max_da, 5))
                a_kinds = [rng.choice(["perp", "collinear"]) if da >= 4 else "perp"]

        a_pos = (focus[0], focus[1] - da) if a_side == "left" else (focus[0], focus[1] + da)
        b_pos = (focus[0] - db, focus[1]) if b_side == "up" else (focus[0] + db, focus[1])

        if a_pos == b_pos:
            continue
        return focus, a_pos, b_pos, a_kinds, b_kinds
    return None


def _dir_toward(start: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    sr, sc = start
    tr, tc = target
    if sr == tr:
        return (0, 1) if tc > sc else (0, -1)
    return (1, 0) if tr > sr else (-1, 0)


def _place_conflict_structure(
    grid: np.ndarray,
    mode: str,
    colors: list[int],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    sampled = _random_focus_pair(grid.shape, mode, rng)
    if sampled is None:
        return None
    focus, a_pos, b_pos, a_kinds, b_kinds = sampled

    towers = [a_pos, b_pos]
    if not _can_place_tower(grid, a_pos, []):
        return None
    if not _can_place_tower(grid, b_pos, [a_pos]):
        return None

    grid[a_pos] = colors[0]
    grid[b_pos] = colors[1]

    protected = {focus}
    path_a = _axis_path_exclusive(a_pos, focus)
    path_b = _axis_path_exclusive(b_pos, focus)
    dir_a = _dir_toward(a_pos, focus)
    dir_b = _dir_toward(b_pos, focus)

    if a_kinds:
        placed = _place_path_walls(grid, path_a, dir_a, towers, protected, a_kinds, rng)
        if not placed:
            return None
        protected.update(placed)
    if b_kinds:
        placed = _place_path_walls(grid, path_b, dir_b, towers, protected, b_kinds, rng)
        if not placed:
            return None
        protected.update(placed)

    protected.update(path_a)
    protected.update(path_b)
    protected.add(a_pos)
    protected.add(b_pos)
    return protected, towers


def _candidate_lane_towers(
    grid: np.ndarray,
    towers: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape
    out: list[tuple[int, int, tuple[int, int], list[tuple[int, int]]]] = []
    rows = list(range(1, h - 1))
    cols = list(range(1, w - 1))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r in rows:
        for c in cols:
            if not _can_place_tower(grid, (r, c), towers):
                continue
            dirs = list(RAY_DIRS)
            rng.shuffle(dirs)
            for direction in dirs:
                path = _path_to_edge((r, c), direction, grid.shape)
                if len(path) < 8:
                    continue
                if all(int(grid[rr, cc]) == BLACK for rr, cc in path):
                    out.append((int(r), int(c), direction, path))
    return out


def _place_two_wall_lane(
    grid: np.ndarray,
    tower_pos: tuple[int, int],
    color: int,
    direction: tuple[int, int],
    path: list[tuple[int, int]],
    towers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    patterns = [
        ["perp", "perp"],
        ["collinear", "perp"],
        ["perp", "collinear"],
        ["collinear", "collinear"],
    ]
    rng.shuffle(patterns)

    grid[tower_pos] = int(color)
    protected = set(protected)
    protected.add(tower_pos)

    for pattern in patterns:
        snapshot = grid.copy()
        local_protected = set(protected)
        placed = _place_path_walls(snapshot, path, direction, towers + [tower_pos], local_protected, pattern, rng)
        if not placed:
            continue
        grid[:, :] = snapshot
        local_protected.update(path)
        local_protected.update(placed)
        return local_protected, placed
    return None


def _place_extra_tower(
    grid: np.ndarray,
    towers: list[tuple[int, int]],
    color: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if _can_place_tower(grid, (r, c), towers)]
    if not candidates:
        return None
    r, c = rng.choice(candidates)
    grid[r, c] = int(color)
    return int(r), int(c)


def _place_extra_walls(
    grid: np.ndarray,
    towers: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    extra_count = rng.randint(EXTRA_WALLS_RANGE[0], EXTRA_WALLS_RANGE[1])
    for _ in range(extra_count):
        for _attempt in range(80):
            orientation = rng.choice(["h", "v"])
            length = rng.randint(2, 5)
            if orientation == "h":
                r = rng.randint(0, h - 1)
                c = rng.randint(0, w - length)
            else:
                r = rng.randint(0, h - length)
                c = rng.randint(0, w - 1)
            cells = _segment_cells((r, c), orientation, length)
            if _can_place_wall(grid, cells, towers, protected):
                _paint_cells(grid, cells, GREY)
                protected.update(cells)
                break


def _build_candidate(mode: str, rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = list(TOWER_COLORS)
    rng.shuffle(colors)

    conflict = _place_conflict_structure(grid, mode, colors, rng)
    if conflict is None:
        return None
    protected, towers = conflict

    lane_candidates = _candidate_lane_towers(grid, towers, rng)
    if not lane_candidates:
        return None

    lane_pick = rng.choice(lane_candidates)
    lane_pos = (lane_pick[0], lane_pick[1])
    lane_dir = lane_pick[2]
    lane_path = lane_pick[3]
    lane_result = _place_two_wall_lane(
        grid,
        lane_pos,
        colors[2],
        lane_dir,
        lane_path,
        towers,
        protected,
        rng,
    )
    if lane_result is None:
        return None
    protected, lane_walls = lane_result
    towers = towers + [lane_pos]
    protected.update(lane_walls)

    if rng.random() < EXTRA_TOWER_PROB:
        extra_color = rng.choice(TOWER_COLORS)
        extra_pos = _place_extra_tower(grid, towers, extra_color, rng)
        if extra_pos is not None:
            towers.append(extra_pos)

    _place_extra_walls(grid, towers, protected, rng)
    return grid


def _is_good_example(grid: np.ndarray, mode: str) -> bool:
    out, stats = _simulate(grid)
    h, w = grid.shape
    colored = int(stats["colored_count"])
    winning = stats["winning_attenuations"]

    if colored < 18 or colored > (h * w) // 2:
        return False
    if int(winning.get(1, 0)) == 0 or int(winning.get(2, 0)) == 0:
        return False
    if len(stats["towers"]) < 3:
        return False
    if mode == "freshness" and int(stats["freshness_conflicts"]) == 0:
        return False
    if mode == "distance" and int(stats["distance_conflicts"]) == 0:
        return False
    if mode == "tie" and int(stats["tie_conflicts"]) == 0:
        return False
    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    modes = ["freshness", "distance", "tie"]
    for _ in range(GEN_RETRIES):
        mode = rng.choice(modes)
        grid = _build_candidate(mode, rng)
        if grid is None:
            continue
        if _is_good_example(grid, mode):
            return grid
    raise RuntimeError(f"failed to generate valid signal-tower input after {GEN_RETRIES} attempts")
