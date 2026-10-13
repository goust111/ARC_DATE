from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_004_9",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
in_bounds = _ARC_UTILS.in_bounds
neighbors4 = _ARC_UTILS.neighbors4


BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
TEAL = Color.TEAL

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (18, 26)
BOX_HEIGHT = 8
GEN_RETRIES = 500


def _runs(cols: list[int]) -> list[list[int]]:
    if not cols:
        return []
    cols = sorted(cols)
    out = [[cols[0]]]
    for c in cols[1:]:
        if c == out[-1][-1] + 1:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def _guide_info(grid: np.ndarray) -> tuple[int, set[int], set[int]]:
    red_rows = np.flatnonzero(np.any(grid == RED, axis=1))
    assert len(red_rows) == 1, "expected exactly one red guide row"
    guide_r = int(red_rows[0])

    red_cols = set(int(c) for c in np.flatnonzero(grid[guide_r] == RED))
    assert red_cols, "expected red guide pixels"
    c0, c1 = min(red_cols), max(red_cols)

    assert all(int(grid[guide_r, c]) in (BLACK, RED) for c in range(c0, c1 + 1))
    gap_cols = {c for c in range(c0, c1 + 1) if int(grid[guide_r, c]) == BLACK}
    gap_runs = _runs(sorted(gap_cols))
    assert 2 <= len(gap_runs) <= 3, "expected 2-3 guide gaps"
    return guide_r, red_cols, gap_cols


def _seed_positions(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    seeds: dict[int, tuple[int, int]] = {}
    for color in (BLUE, GREEN):
        pts = np.argwhere(grid == color)
        assert len(pts) == 1, "expected one blue seed and one green seed"
        seeds[color] = tuple(int(v) for v in pts[0])
    return seeds


def _bfs_from_seed(grid: np.ndarray, seed: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {seed: 0}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        base = dist[(r, c)]
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) != BLACK:
                continue
            if (nr, nc) in dist:
                continue
            dist[(nr, nc)] = base + 1
            q.append((nr, nc))
    return dist


def _yellow_analysis(
    grid: np.ndarray,
    pos: tuple[int, int],
    guide_r: int,
    solid_cols: set[int],
    seed_dists: dict[int, dict[tuple[int, int], int]],
    seeds: dict[int, tuple[int, int]],
) -> dict:
    r, c = pos
    assert r != guide_r, "yellow pixels must not lie on the guide row"

    best: tuple[int, int, int, int] | None = None
    for color, seed in seeds.items():
        sr, sc = seed
        cand = None
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) == seed:
                cand = 1 if cand is None else min(cand, 1)
            elif int(grid[nr, nc]) == BLACK and (nr, nc) in seed_dists[color]:
                step = 1 + seed_dists[color][(nr, nc)]
                cand = step if cand is None else min(cand, step)
        if cand is None:
            continue
        key = (cand, sr, sc, color)
        if best is None or key < best:
            best = key

    info = {
        "pos": pos,
        "A": c in solid_cols,
        "B": best is not None,
        "best_color": None,
        "best_dist": None,
        "seed_dists": {},
    }
    for color, seed in seeds.items():
        sr, sc = seed
        cand = None
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) == seed:
                cand = 1 if cand is None else min(cand, 1)
            elif int(grid[nr, nc]) == BLACK and (nr, nc) in seed_dists[color]:
                step = 1 + seed_dists[color][(nr, nc)]
                cand = step if cand is None else min(cand, step)
        info["seed_dists"][color] = cand

    if best is not None:
        info["best_dist"] = int(best[0])
        info["best_color"] = int(best[3])
    return info


def _expected_output_and_analysis(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    guide_r, solid_cols, _gap_cols = _guide_info(g)
    seeds = _seed_positions(g)
    yellow_positions = [tuple(int(v) for v in pt) for pt in np.argwhere(g == YELLOW)]
    assert 5 <= len(yellow_positions) <= 8, "expected 5-8 yellow pixels"

    seed_dists = {color: _bfs_from_seed(g, pos) for color, pos in seeds.items()}
    analysis = {
        pos: _yellow_analysis(g, pos, guide_r, solid_cols, seed_dists, seeds)
        for pos in yellow_positions
    }

    out = g.copy()
    out[out == YELLOW] = BLACK

    painted: dict[tuple[int, int], int] = {}
    for pos, info in analysis.items():
        if not (info["A"] and info["B"]):
            continue
        r, c = pos
        target = (guide_r - 1, c) if r < guide_r else (guide_r + 1, c)
        assert in_bounds(g.shape, *target), "projection target must stay inside the grid"
        assert int(g[target]) == BLACK, "projection target must be empty in the input"
        color = int(info["best_color"])
        if target in painted:
            assert painted[target] == color, "conflicting projections to the same target"
        painted[target] = color

    for target, color in painted.items():
        out[target] = color
    return out, analysis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only scattered pixels that satisfy both guide-bar alignment and seed reachability, then project them onto the guide-adjacent row with the winning seed color.

    Input structure:
    - Black background contains one horizontal red guide bar with 2-3 gaps on a single middle row.
    - Cyan wall boxes above and below the guide row carve narrow black corridor networks.
    - Exactly one blue seed and one green seed sit in one corridor network, while 5-8 yellow scattered pixels are split between seeded and unseeded corridors.

    Rule:
    - For each yellow pixel at column c, condition A is true when the guide row still has a red pixel at column c.
    - Condition B is true when the yellow pixel can reach a seed by 4-neighbor motion whose intermediate cells are black; cyan walls, the red guide bar, seeds, and other yellow pixels are not traversable.
    - If both conditions hold, erase the original yellow pixel and place the winning seed color on the guide-adjacent row in the same column: above-guide yellows project to one row above the guide, below-guide yellows project to one row below.
    - If both seeds are reachable, use the one with the shorter valid path; if those lengths tie, use the seed with the smaller row index.

    Key insight:
    - Guide-bar occupancy filters columns, while corridor reachability filters topology; a yellow survives only when the same pixel passes both independent tests, and its output color comes from the reachability winner.
    """
    out, _analysis = _expected_output_and_analysis(np.asarray(grid, dtype=int))
    return out


def _sample_guide_columns(width: int) -> tuple[list[int], list[int]]:
    span = list(range(1, width - 1))
    for _ in range(200):
        run_count = random.randint(2, 3)
        runs: list[tuple[int, int]] = []
        while len(runs) < run_count:
            run_len = random.randint(1, 2)
            start = random.randint(2, width - 4 - run_len)
            end = start + run_len - 1
            if any(not (end + 1 < s or e + 1 < start) for s, e in runs):
                continue
            runs.append((start, end))
        gap_cols = sorted(c for s, e in runs for c in range(s, e + 1))
        solid_cols = sorted(c for c in span if c not in set(gap_cols))
        if len(gap_cols) >= 2 and solid_cols:
            return solid_cols, gap_cols
    raise RuntimeError("failed to sample guide gaps")


def _pick_column_roles(
    solid_cols: list[int],
    gap_cols: list[int],
    width: int,
) -> dict[str, int] | None:
    solid_set = set(solid_cols)
    gap_pool = [g for g in gap_cols if 5 <= g <= width - 6]
    random.shuffle(gap_pool)

    for g_seed in gap_pool:
        left = [s for s in solid_cols if max(3, g_seed - 4) <= s < g_seed]
        right = [s for s in solid_cols if g_seed < s <= min(width - 4, g_seed + 4)]
        random.shuffle(left)
        random.shuffle(right)
        for s_blue in left:
            for s_green in right:
                if (s_blue + s_green) % 2 != 0:
                    continue
                bcol = s_blue - 1
                gcol = s_green + 1
                if bcol < 2 or gcol > width - 3:
                    continue
                mid = (bcol + gcol) // 2
                tie_ok = mid in solid_set and mid not in {s_blue, s_green}

                other_gaps = [g for g in gap_pool if g != g_seed]
                random.shuffle(other_gaps)
                for g_unseed in other_gaps:
                    nearby = [
                        s
                        for s in solid_cols
                        if 1 <= abs(s - g_unseed) <= 4 and s not in {s_blue, s_green}
                    ]
                    if not nearby:
                        continue
                    u_solid = random.choice(nearby)
                    return {
                        "blue_seed": bcol,
                        "ab_blue": s_blue,
                        "b_only": g_seed,
                        "ab_green": s_green,
                        "green_seed": gcol,
                        "tie_col": mid,
                        "tie_ok": int(tie_ok),
                        "a_only": u_solid,
                        "neither": g_unseed,
                    }
    return None


def _box_rows(side: str, guide_r: int) -> tuple[int, int]:
    if side == "top":
        r1 = guide_r - 2
        r0 = r1 - BOX_HEIGHT + 1
    else:
        r0 = guide_r + 2
        r1 = r0 + BOX_HEIGHT - 1
    return r0, r1


def _build_box(
    side: str,
    guide_r: int,
    specs: list[dict[str, int | str]],
) -> dict:
    r0, r1 = _box_rows(side, guide_r)
    cols = [int(spec["col"]) for spec in specs]
    c0 = min(cols) - 2
    c1 = max(cols) + 2
    trunk_r = r0 + 3
    slot_rows = {
        "upper_far": trunk_r - 2,
        "upper_near": trunk_r - 1,
        "lower_near": trunk_r + 1,
        "lower_far": trunk_r + 2,
    }
    leaf_cells: dict[str, tuple[int, int]] = {}
    black_cells = {(trunk_r, c) for c in range(c0 + 1, c1)}

    for spec in specs:
        label = str(spec["label"])
        col = int(spec["col"])
        leaf_r = slot_rows[str(spec["slot"])]
        leaf_cells[label] = (leaf_r, col)
        step = 1 if leaf_r < trunk_r else -1
        for rr in range(trunk_r, leaf_r, -step):
            black_cells.add((rr, col))

    spare = [c for c in range(c0 + 1, c1) if c not in cols]
    random.shuffle(spare)
    for c in spare[: random.randint(1, min(2, len(spare)))]:
        depth = random.choice([1, 2])
        if random.random() < 0.5:
            for d in range(1, depth + 1):
                black_cells.add((trunk_r - d, c))
        else:
            for d in range(1, depth + 1):
                black_cells.add((trunk_r + d, c))

    return {
        "bbox": (r0, c0, r1, c1),
        "leaf_cells": leaf_cells,
        "black_cells": black_cells,
        "specs": specs,
    }


def _paint_box(grid: np.ndarray, box: dict) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = box["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = TEAL
    for r, c in box["black_cells"]:
        grid[r, c] = BLACK
    for spec in box["specs"]:
        label = str(spec["label"])
        color = int(spec["color"])
        r, c = box["leaf_cells"][label]
        grid[r, c] = color
    return box["leaf_cells"]


def _project_row_for(side: str, guide_r: int) -> int:
    return guide_r - 1 if side == "top" else guide_r + 1


def _validate_generated(
    grid: np.ndarray,
    labels: dict[str, tuple[int, int]],
    expect_tie: bool,
) -> bool:
    try:
        expected, analysis = _expected_output_and_analysis(grid)
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(expected, actual):
        return False

    info = {name: analysis[pos] for name, pos in labels.items() if pos in analysis}
    required = {"ab_blue", "ab_green", "b_only", "a_only", "neither"}
    if not required.issubset(info):
        return False

    if not (info["ab_blue"]["A"] and info["ab_blue"]["B"] and info["ab_blue"]["best_color"] == BLUE):
        return False
    if not (info["ab_green"]["A"] and info["ab_green"]["B"] and info["ab_green"]["best_color"] == GREEN):
        return False
    if not ((not info["b_only"]["A"]) and info["b_only"]["B"]):
        return False
    if not (info["a_only"]["A"] and (not info["a_only"]["B"])):
        return False
    if not ((not info["neither"]["A"]) and (not info["neither"]["B"])):
        return False

    projected = []
    for name in ("ab_blue", "ab_green"):
        r, c = labels[name]
        guide_r, _solid_cols, _gap_cols = _guide_info(grid)
        target = (guide_r - 1, c) if r < guide_r else (guide_r + 1, c)
        projected.append((target, int(expected[target])))
    if len({color for _target, color in projected}) != 2:
        return False

    if expect_tie:
        if "ab_tie" not in info:
            return False
        tie = info["ab_tie"]
        if not (tie["A"] and tie["B"] and tie["best_color"] == BLUE):
            return False
        if tie["seed_dists"][BLUE] != tie["seed_dists"][GREEN]:
            return False

    yellow_count = int(np.count_nonzero(grid == YELLOW))
    if not (5 <= yellow_count <= 8):
        return False
    return True


def _attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    if width < 18:
        return None
    guide_r = random.randint(10, height - 10)

    solid_cols, gap_cols = _sample_guide_columns(width)
    roles = _pick_column_roles(solid_cols, gap_cols, width)
    if roles is None:
        return None

    want_tie = bool(roles["tie_ok"]) and random.random() < 0.4
    seeded_side = random.choice(["top", "bottom"])
    unseeded_side = "bottom" if seeded_side == "top" else "top"

    seeded_specs: list[dict[str, int | str]] = [
        {"label": "blue_seed", "color": BLUE, "col": roles["blue_seed"], "slot": "upper_far"},
        {"label": "ab_blue", "color": YELLOW, "col": roles["ab_blue"], "slot": "upper_near"},
        {"label": "b_only", "color": YELLOW, "col": roles["b_only"], "slot": "lower_near"},
        {"label": "ab_green", "color": YELLOW, "col": roles["ab_green"], "slot": "lower_near"},
        {"label": "green_seed", "color": GREEN, "col": roles["green_seed"], "slot": "lower_far"},
    ]
    if want_tie:
        seeded_specs.append({"label": "ab_tie", "color": YELLOW, "col": roles["tie_col"], "slot": "upper_near"})

    unseeded_specs: list[dict[str, int | str]] = [
        {"label": "a_only", "color": YELLOW, "col": roles["a_only"], "slot": "upper_near"},
        {"label": "neither", "color": YELLOW, "col": roles["neither"], "slot": "lower_near"},
    ]

    seeded_box = _build_box(seeded_side, guide_r, seeded_specs)
    unseeded_box = _build_box(unseeded_side, guide_r, unseeded_specs)

    for box in (seeded_box, unseeded_box):
        r0, c0, r1, c1 = box["bbox"]
        if not (1 <= r0 < r1 < height - 1 and 1 <= c0 < c1 < width - 1):
            return None

    grid = np.zeros((height, width), dtype=int)
    grid[guide_r, 1 : width - 1] = RED
    grid[guide_r, gap_cols] = BLACK

    labels = {}
    labels.update(_paint_box(grid, seeded_box))
    labels.update(_paint_box(grid, unseeded_box))

    for name, pos in labels.items():
        if "seed" in name:
            continue
        project_r = _project_row_for(seeded_side if name.startswith("ab_") or name == "b_only" else unseeded_side, guide_r)
        pr = project_r
        pc = pos[1]
        if int(grid[pr, pc]) != BLACK:
            return None

    if not _validate_generated(grid, labels, expect_tie=want_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-constraint ARC input")
