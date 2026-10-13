from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

SIDE_RANGE = (3, 5)
GRID_RANGE = (18, 24)
GEN_RETRIES = 400

KIND_TO_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}

QUADRANT_TO_KIND = {
    "nw": "br",
    "ne": "bl",
    "sw": "tr",
    "se": "tl",
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_gap_at_least_two(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(row_gap, col_gap) >= 2


def _group_from_missing(missing: tuple[int, int], kind: str, side: int) -> dict:
    r, c = map(int, missing)
    sr, sc = KIND_TO_SIGN[kind]
    other_r = r - sr * int(side)
    other_c = c - sc * int(side)

    corners = {
        "tl": (min(r, other_r), min(c, other_c)),
        "tr": (min(r, other_r), max(c, other_c)),
        "bl": (max(r, other_r), min(c, other_c)),
        "br": (max(r, other_r), max(c, other_c)),
    }
    markers = tuple(corners[name] for name in ("tl", "tr", "bl", "br") if name != kind)
    bbox = (
        min(r, other_r),
        min(c, other_c),
        max(r, other_r),
        max(c, other_c),
    )
    return {
        "missing": (r, c),
        "missing_kind": kind,
        "side": int(side),
        "markers": markers,
        "bbox": bbox,
    }


def _candidate_groups(marker_coords: set[tuple[int, int]]) -> list[dict]:
    rows = sorted({r for r, _ in marker_coords})
    row_set = set(rows)
    cols = sorted({c for _, c in marker_coords})
    col_set = set(cols)
    candidates: list[dict] = []

    for r0 in rows:
        for side in range(SIDE_RANGE[0], SIDE_RANGE[1] + 1):
            r1 = r0 + side
            if r1 not in row_set:
                continue
            for c0 in cols:
                c1 = c0 + side
                if c1 not in col_set:
                    continue
                corners = {
                    "tl": (r0, c0),
                    "tr": (r0, c1),
                    "bl": (r1, c0),
                    "br": (r1, c1),
                }
                present = [name for name, pos in corners.items() if pos in marker_coords]
                if len(present) != 3:
                    continue
                missing_kind = next(name for name in corners if name not in present)
                markers = tuple(corners[name] for name in ("tl", "tr", "bl", "br") if name != missing_kind)
                candidates.append(
                    {
                        "missing": corners[missing_kind],
                        "missing_kind": missing_kind,
                        "side": side,
                        "markers": markers,
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    return candidates


def _unique_partition(marker_coords: set[tuple[int, int]]) -> list[dict]:
    candidates = _candidate_groups(marker_coords)
    marker_list = sorted(marker_coords)
    solutions: list[list[dict]] = []

    def backtrack(remaining: set[tuple[int, int]], chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if 3 <= len(chosen) <= 4:
                solutions.append(chosen.copy())
            return
        if len(chosen) >= 4:
            return

        pivot = min(remaining)
        for cand in candidates:
            marker_set = set(cand["markers"])
            if pivot not in marker_set:
                continue
            if not marker_set <= remaining:
                continue
            backtrack(remaining - marker_set, chosen + [cand])

    backtrack(set(marker_list), [])
    assert len(solutions) == 1, "expected a unique grouping into 3-4 three-corner squares"
    groups = solutions[0]
    groups.sort(key=lambda g: (g["missing"][0], g["missing"][1], g["missing_kind"]))
    return groups


def _trace_ray(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int], bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r = int(start[0]) + int(step[0])
    c = int(start[1]) + int(step[1])
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
        out.append((r, c))
        r += int(step[0])
        c += int(step[1])
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.flat)))
    assert set(colors) <= {bg, YELLOW}, "input must contain only background and yellow markers"

    marker_coords = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}
    assert 9 <= len(marker_coords) <= 12 and len(marker_coords) % 3 == 0
    groups = _unique_partition(marker_coords)
    assert len(groups) * 3 == len(marker_coords)

    for i, gi in enumerate(groups):
        r0, c0, r1, c1 = gi["bbox"]
        assert r1 - r0 == c1 - c0
        assert SIDE_RANGE[0] <= gi["side"] <= SIDE_RANGE[1]
        for j in range(i + 1, len(groups)):
            assert _bbox_gap_at_least_two(gi["bbox"], groups[j]["bbox"])

    per_group: list[dict] = []
    cover_counts: dict[tuple[int, int], set[int]] = {}
    for gid, group in enumerate(groups):
        missing = group["missing"]
        sr, sc = KIND_TO_SIGN[group["missing_kind"]]
        horizontal = _trace_ray(g, missing, (0, sc), bg)
        vertical = _trace_ray(g, missing, (sr, 0), bg)
        per_group.append(
            {
                "missing": missing,
                "kind": group["missing_kind"],
                "side": group["side"],
                "horizontal": horizontal,
                "vertical": vertical,
            }
        )
        for cell in horizontal + vertical:
            cover_counts.setdefault(cell, set()).add(gid)

    intersections = sorted(cell for cell, gids in cover_counts.items() if len(gids) >= 2)
    singles = sorted(cell for cell, gids in cover_counts.items() if len(gids) == 1)
    return {
        "bg": bg,
        "groups": groups,
        "rays": per_group,
        "cover_counts": cover_counts,
        "intersections": intersections,
        "singles": singles,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend two outward rays from each three-corner square's missing corner and recolor cross-group overlaps red.

    Input structure:
    - The grid is mostly black background.
    - Yellow singleton pixels form 3-4 separate groups; each group is exactly three corners of a square bbox, with one corner missing.
    - Different groups do not overlap, and the missing corner position identifies one of the four corner types.

    Rule:
    - Keep all input pixels unchanged.
    - For each three-corner group, infer the missing square corner.
    - From that missing corner, launch one horizontal ray and one vertical ray outward from the square center, starting at the adjacent cell and continuing through background until the grid edge or the first non-background cell.
    - Paint cells covered by exactly one group's rays green; repaint any cell covered by rays from two or more different groups red.

    Key insight:
    - The absent corner is not just a hole: its geometric quadrant encodes the two ray directions, and red pixels only emerge after combining independently inferred rays from different groups.
    """
    analysis = _analyze(grid)
    out = np.asarray(grid, dtype=int).copy()
    for cell, gids in analysis["cover_counts"].items():
        out[cell] = GREEN if len(gids) == 1 else RED
    return out


def _choose_quadrants() -> list[str]:
    if random.random() < 0.7:
        return ["nw", "ne", "sw", "se"]
    return random.sample(["nw", "ne", "sw", "se"], 3)


def _ranges_for_quadrant(
    quadrant: str, side: int, height: int, width: int
) -> tuple[list[int], list[int]]:
    mid_r = height // 2
    mid_c = width // 2

    if quadrant == "nw":
        rows = list(range(side + 1, mid_r - 1))
        cols = list(range(side + 1, mid_c - 1))
    elif quadrant == "ne":
        rows = list(range(side + 1, mid_r - 1))
        cols = list(range(mid_c + 1, width - side - 1))
    elif quadrant == "sw":
        rows = list(range(mid_r + 1, height - side - 1))
        cols = list(range(side + 1, mid_c - 1))
    else:
        rows = list(range(mid_r + 1, height - side - 1))
        cols = list(range(mid_c + 1, width - side - 1))
    return rows, cols


def _attempt_groups(height: int, width: int) -> list[dict] | None:
    quadrants = _choose_quadrants()
    groups: list[dict] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    used_markers: set[tuple[int, int]] = set()

    order = quadrants[:]
    random.shuffle(order)
    for quadrant in order:
        placed = False
        for _ in range(80):
            side = random.randint(SIDE_RANGE[0], SIDE_RANGE[1])
            rows, cols = _ranges_for_quadrant(quadrant, side, height, width)
            rows = [r for r in rows if r not in used_rows]
            cols = [c for c in cols if c not in used_cols]
            if not rows or not cols:
                continue

            missing = (random.choice(rows), random.choice(cols))
            group = _group_from_missing(missing, QUADRANT_TO_KIND[quadrant], side)
            bbox = group["bbox"]
            marker_set = set(group["markers"])
            if len(marker_set) != 3:
                continue
            if marker_set & used_markers:
                continue
            if any(not _bbox_gap_at_least_two(bbox, old["bbox"]) for old in groups):
                continue

            groups.append(group)
            used_rows.add(missing[0])
            used_cols.add(missing[1])
            used_markers |= marker_set
            placed = True
            break
        if not placed:
            return None

    groups.sort(key=lambda g: (g["missing"][0], g["missing"][1]))
    return groups


def _render_input(height: int, width: int, groups: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for group in groups:
        for r, c in group["markers"]:
            grid[r, c] = YELLOW
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
    except AssertionError:
        return False

    groups = analysis["groups"]
    rays = analysis["rays"]
    intersections = analysis["intersections"]
    singles = analysis["singles"]
    cover_counts = analysis["cover_counts"]

    if len(groups) not in (3, 4):
        return False
    if any(len(ray["horizontal"]) == 0 or len(ray["vertical"]) == 0 for ray in rays):
        return False
    if len(intersections) < 2:
        return False
    if not singles:
        return False
    if len({r for r, _ in intersections}) < 2:
        return False
    if len({c for _, c in intersections}) < 2:
        return False
    participating = set()
    for cell in intersections:
        participating |= cover_counts[cell]
    if len(participating) < min(3, len(groups)):
        return False

    ray_cells = {cell for ray in rays for cell in (ray["horizontal"] + ray["vertical"])}
    marker_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)}
    if ray_cells & marker_cells:
        return False

    result = main(grid)
    if RED not in result or GREEN not in result:
        return False

    red_count = int(np.count_nonzero(result == RED))
    green_count = int(np.count_nonzero(result == GREEN))
    if red_count < 2 or green_count <= red_count:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*GRID_RANGE)
        width = random.randint(*GRID_RANGE)
        groups = _attempt_groups(height, width)
        if groups is None:
            continue
        grid = _render_input(height, width, groups)
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid missing-corner ray puzzle")
