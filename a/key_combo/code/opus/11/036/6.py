from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
TEAL = 8

SEED_COLORS = (RED, YELLOW)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_single_seed(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert len(pts) == 1, f"expected exactly one seed of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _wall_distances(grid: np.ndarray, pos: tuple[int, int]) -> list[tuple[int, int, int, tuple[int, int]]]:
    """Return wall distance data in priority order: (distance, dir_idx, dr, dc, wall_pos)."""
    r, c = pos
    out = []
    for idx, (dr, dc) in enumerate(DIRS):
        rr, cc = r + dr, c + dc
        dist = 1
        while _in_bounds(grid.shape, rr, cc):
            if int(grid[rr, cc]) == TEAL:
                out.append((dist, idx, dr, dc, (rr, cc)))
                break
            rr += dr
            cc += dc
            dist += 1
    assert len(out) == 4, "each seed must see a wall in all four directions"
    return out


def _compress_info(grid: np.ndarray, pos: tuple[int, int]) -> dict[str, object]:
    dists = _wall_distances(grid, pos)
    best = min(dists, key=lambda item: (item[0], item[1]))
    dist, dir_idx, dr, dc, wall = best
    landing = (pos[0] + dr * (dist - 1), pos[1] + dc * (dist - 1))
    tied = sum(item[0] == dist for item in dists) > 1
    return {
        "pos": pos,
        "landing": landing,
        "distance": dist,
        "move": dist - 1,
        "dir_idx": dir_idx,
        "dir": (dr, dc),
        "wall": wall,
        "distances": [item[0] for item in dists],
        "tied": tied,
    }


def _apply_compression(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]], dict[int, dict[str, object]]]:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    seed_infos: dict[int, dict[str, object]] = {}
    landings: dict[int, tuple[int, int]] = {}

    for color in SEED_COLORS:
        pos = _find_single_seed(base, color)
        info = _compress_info(base, pos)
        landing = info["landing"]
        assert landing != pos or info["move"] == 0
        assert int(base[landing]) in (BLACK, color), "landing cell must remain corridor"
        seed_infos[color] = info
        landings[color] = landing

    assert landings[RED] != landings[YELLOW], "compressed seeds must stay distinct"
    for color in SEED_COLORS:
        out[_find_single_seed(base, color)] = BLACK
    for color, landing in landings.items():
        out[landing] = color
    return out, landings, seed_infos


def _bfs_distances(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(grid.shape, -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not _in_bounds(grid.shape, rr, cc):
                continue
            if dist[rr, cc] != -1 or int(grid[rr, cc]) == TEAL:
                continue
            dist[rr, cc] = dist[r, c] + 1
            q.append((rr, cc))
    return dist


def _fill_from_landings(grid: np.ndarray, landings: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    red_dist = _bfs_distances(out, landings[RED])
    yellow_dist = _bfs_distances(out, landings[YELLOW])

    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            value = int(out[r, c])
            if value == TEAL:
                continue
            if (r, c) == landings[RED]:
                out[r, c] = RED
                continue
            if (r, c) == landings[YELLOW]:
                out[r, c] = YELLOW
                continue

            dr = int(red_dist[r, c])
            dy = int(yellow_dist[r, c])
            if dr == -1 and dy == -1:
                continue
            if dr == dy and dr > 0:
                out[r, c] = GREY
                continue
            if dy == -1 or (dr != -1 and dr < dy):
                out[r, c] = RED if dr % 2 == 1 else YELLOW
            else:
                out[r, c] = YELLOW if dy % 2 == 1 else RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each corridor seed to its nearest wall and then alternate-color the maze by BFS path parity.

    Input structure:
    - The grid contains teal wall cells, black corridor cells, and exactly one red seed plus one yellow seed.
    - Each seed lies inside a straight corridor region with teal walls visible in all four orthogonal directions.
    - After removing the seeds, the remaining non-wall corridor stays 4-connected.

    Rule:
    - For each seed, look outward in the four orthogonal directions and find the nearest teal wall; ties are resolved by the fixed priority up, right, down, left.
    - Clear the original seed cell to black and move the seed to the corridor cell immediately adjacent to that chosen wall along the same ray.
    - From the two compressed seed positions, run simultaneous 4-neighbor BFS through all non-wall cells.
    - Each seed keeps its own color at distance 0; at odd distances a source paints with its own color, at even distances it paints with the other seed's color; equal-time arrivals paint grey.

    Key insight:
    - The tiny pre-BFS seed shift changes path parity everywhere downstream, so moving a seed by only 1-3 cells can flip the coloring of large corridor regions.
    """
    compressed, landings, _ = _apply_compression(np.asarray(grid, dtype=int))
    return _fill_from_landings(compressed, landings)


def _sample_band_sizes() -> list[int]:
    for _ in range(200):
        count = random.randint(2, 3)
        bands = [random.randint(3, 5) for _ in range(count)]
        total = sum(bands) + count + 1
        if 11 <= total <= 17:
            return bands
    raise RuntimeError("failed to sample band sizes")


def _wall_indices(bands: list[int]) -> list[int]:
    walls = [0]
    cur = 0
    for size in bands:
        cur += size + 1
        walls.append(cur)
    return walls


def _band_ranges(walls: list[int]) -> list[list[int]]:
    return [list(range(walls[i] + 1, walls[i + 1])) for i in range(len(walls) - 1)]


def _build_partition() -> tuple[np.ndarray, list[int], list[int], dict[int, set[int]], dict[int, set[int]]]:
    row_bands = _sample_band_sizes()
    col_bands = _sample_band_sizes()
    wall_rows = _wall_indices(row_bands)
    wall_cols = _wall_indices(col_bands)
    h, w = wall_rows[-1] + 1, wall_cols[-1] + 1

    grid = np.full((h, w), BLACK, dtype=int)
    grid[wall_rows, :] = TEAL
    grid[:, wall_cols] = TEAL

    row_ranges = _band_ranges(wall_rows)
    col_ranges = _band_ranges(wall_cols)
    h_gaps = {r: set() for r in wall_rows[1:-1]}
    v_gaps = {c: set() for c in wall_cols[1:-1]}

    extra_gap_budget = 0
    for r in wall_rows[1:-1]:
        gap_count = 1 + (random.random() < 0.45)
        extra_gap_budget += gap_count - 1
        bands = random.sample(range(len(col_ranges)), gap_count)
        for j in bands:
            c = random.choice(col_ranges[j])
            grid[r, c] = BLACK
            h_gaps[r].add(c)

    for c in wall_cols[1:-1]:
        gap_count = 1 + (random.random() < 0.45)
        extra_gap_budget += gap_count - 1
        bands = random.sample(range(len(row_ranges)), gap_count)
        for i in bands:
            r = random.choice(row_ranges[i])
            grid[r, c] = BLACK
            v_gaps[c].add(r)

    if extra_gap_budget == 0:
        if h_gaps:
            r = random.choice(list(h_gaps))
            c = random.choice([cc for cc in range(1, w - 1) if cc not in wall_cols and cc not in h_gaps[r]])
            grid[r, c] = BLACK
            h_gaps[r].add(c)
        elif v_gaps:
            c = random.choice(list(v_gaps))
            r = random.choice([rr for rr in range(1, h - 1) if rr not in wall_rows and rr not in v_gaps[c]])
            grid[r, c] = BLACK
            v_gaps[c].add(r)

    return grid, wall_rows, wall_cols, h_gaps, v_gaps


def _component_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(r), int(c)) for r, c in pts]


def _is_connected_non_wall(grid: np.ndarray) -> bool:
    cells = _component_cells(grid != TEAL)
    if not cells:
        return False
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) == TEAL or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            q.append((rr, cc))
    return len(seen) == len(cells)


def _locate_band(walls: list[int], value: int) -> int:
    for i in range(len(walls) - 1):
        if walls[i] < value < walls[i + 1]:
            return i
    raise AssertionError("value does not lie in any band")


def _enumerate_candidates(
    base: np.ndarray, wall_rows: list[int], wall_cols: list[int]
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)
    for r in range(1, base.shape[0] - 1):
        for c in range(1, base.shape[1] - 1):
            if int(base[r, c]) != BLACK:
                continue
            if r in wall_row_set or c in wall_col_set:
                continue
            chamber = (_locate_band(wall_rows, r), _locate_band(wall_cols, c))
            info = _compress_info(base, (r, c))
            move = int(info["move"])
            if move < 1 or move > 3:
                continue

            row_band, col_band = chamber
            top, bottom = wall_rows[row_band], wall_rows[row_band + 1]
            left, right = wall_cols[col_band], wall_cols[col_band + 1]
            wall_r, wall_c = info["wall"]
            dir_idx = int(info["dir_idx"])
            if dir_idx == 0 and wall_r != top:
                continue
            if dir_idx == 1 and wall_c != right:
                continue
            if dir_idx == 2 and wall_r != bottom:
                continue
            if dir_idx == 3 and wall_c != left:
                continue

            d_up, d_right, d_down, d_left = map(int, info["distances"])
            vert_open = d_up + d_down - 1
            horiz_open = d_left + d_right - 1
            if dir_idx in (0, 2):
                if vert_open < horiz_open or vert_open < 4:
                    continue
            else:
                if horiz_open < vert_open or horiz_open < 4:
                    continue

            candidates.append(
                {
                    "pos": (r, c),
                    "landing": info["landing"],
                    "move": move,
                    "dir_idx": dir_idx,
                    "tied": bool(info["tied"]),
                    "chamber": chamber,
                }
            )
    return candidates


def _place_seeds(base: np.ndarray, pair: tuple[dict[str, object], dict[str, object]]) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    grid = base.copy()
    placed: dict[int, dict[str, object]] = {}
    for color, meta in zip(colors, pair):
        pos = meta["pos"]
        grid[pos] = color
        placed[color] = meta
    return grid, placed


def _collision_exists(base: np.ndarray, landings: dict[int, tuple[int, int]]) -> bool:
    red_dist = _bfs_distances(base, landings[RED])
    yellow_dist = _bfs_distances(base, landings[YELLOW])
    for r in range(base.shape[0]):
        for c in range(base.shape[1]):
            if int(base[r, c]) == TEAL or (r, c) in landings.values():
                continue
            if int(red_dist[r, c]) > 0 and int(red_dist[r, c]) == int(yellow_dist[r, c]):
                return True
    return False


def _validate_generated(grid: np.ndarray, base: np.ndarray, chosen: dict[int, dict[str, object]]) -> bool:
    compressed, landings, infos = _apply_compression(grid)
    for color in SEED_COLORS:
        if int(infos[color]["move"]) not in (1, 2, 3):
            return False
        if tuple(landings[color]) != tuple(chosen[color]["landing"]):
            return False

    if (landings[RED][0] + landings[RED][1]) % 2 != (landings[YELLOW][0] + landings[YELLOW][1]) % 2:
        return False

    red_dist = _bfs_distances(compressed, landings[RED])
    yellow_dist = _bfs_distances(compressed, landings[YELLOW])
    corridor_mask = compressed != TEAL
    if not np.all(red_dist[corridor_mask] >= 0) or not np.all(yellow_dist[corridor_mask] >= 0):
        return False
    if not _collision_exists(compressed, landings):
        return False

    out = _fill_from_landings(compressed, landings)
    if not np.array_equal(main(grid), out):
        return False

    no_compress = _fill_from_landings(grid, {RED: _find_single_seed(grid, RED), YELLOW: _find_single_seed(grid, YELLOW)})
    if np.count_nonzero(out != no_compress) < 8:
        return False
    if np.count_nonzero(out != compressed) < 12:
        return False

    color_counts = {color: int(np.sum(out == color)) for color in (RED, YELLOW, GREY)}
    if color_counts[GREY] == 0 or color_counts[RED] < 3 or color_counts[YELLOW] < 3:
        return False
    return True


def _choose_pair(candidates: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]] | None:
    def pick(require_tie: bool) -> tuple[dict[str, object], dict[str, object]] | None:
        order = candidates[:]
        random.shuffle(order)
        for i, first in enumerate(order):
            for second in order[i + 1 :]:
                if first["chamber"] == second["chamber"]:
                    continue
                if first["landing"] == second["landing"]:
                    continue
                if (first["landing"][0] + first["landing"][1]) % 2 != (second["landing"][0] + second["landing"][1]) % 2:
                    continue
                if require_tie and not (first["tied"] or second["tied"]):
                    continue
                if first["move"] % 2 == 0 and second["move"] % 2 == 0:
                    continue
                return first, second
        return None

    want_tie = random.random() < 0.4
    return pick(want_tie) or pick(False)


def _generate_attempt() -> np.ndarray | None:
    base, wall_rows, wall_cols, _h_gaps, _v_gaps = _build_partition()
    if not _is_connected_non_wall(base):
        return None
    if int(np.sum(base == BLACK)) < 40:
        return None

    candidates = _enumerate_candidates(base, wall_rows, wall_cols)
    if len(candidates) < 2:
        return None
    pair = _choose_pair(candidates)
    if pair is None:
        return None

    grid, chosen = _place_seeds(base, pair)
    if not _validate_generated(grid, base, chosen):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
