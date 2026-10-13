import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
CYAN = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
SIDE_TO_DIAGONALS = {
    "N": ("NW", "NE"),
    "S": ("SW", "SE"),
    "W": ("NW", "SW"),
    "E": ("NE", "SE"),
}
COLOR_BY_D = {0: BLUE, 1: RED, 2: GREEN, 3: YELLOW, 4: ORANGE}

GEN_ATTEMPTS = 500
GEN_SIDE_RANGE = (14, 19)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            if all(grid[r + dr, c + dc] == BLACK for dr, dc in DIR4):
                centers.append((r, c))
    return centers


def _diagonal_cyan_count(grid: np.ndarray, center: tuple[int, int]) -> int:
    r, c = center
    return sum(int(grid[r + dr, c + dc] == CYAN) for dr, dc in DIAGONALS.values())


def _claim_cells(grid: np.ndarray, center: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Reach black cells by monotone shortest L1 paths; cyan blocks propagation."""
    h, w = grid.shape
    cr, cc = center
    claims: dict[tuple[int, int], int] = {(cr, cc): 0}
    max_k = h + w
    for k in range(1, max_k + 1):
        any_added = False
        r_lo = max(0, cr - k)
        r_hi = min(h - 1, cr + k)
        for r in range(r_lo, r_hi + 1):
            rem = k - abs(r - cr)
            cols = [cc - rem]
            if rem:
                cols.append(cc + rem)
            for c in cols:
                if not (0 <= c < w) or grid[r, c] != BLACK or (r, c) in claims:
                    continue
                predecessors: list[tuple[int, int]] = []
                if r < cr:
                    predecessors.append((r + 1, c))
                elif r > cr:
                    predecessors.append((r - 1, c))
                if c < cc:
                    predecessors.append((r, c + 1))
                elif c > cc:
                    predecessors.append((r, c - 1))
                if any(claims.get(prev) == k - 1 for prev in predecessors):
                    claims[(r, c)] = k
                    any_added = True
        if not any_added:
            farthest = max(claims.values())
            if farthest < k - 1:
                break
    return claims


def _solve_metadata(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[tuple[int, int], list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers = _find_centers(g)
    info: list[dict] = []
    contested: dict[tuple[int, int], list[int]] = defaultdict(list)
    owners: dict[tuple[int, int], tuple[int, int, int]] = {}

    for idx, center in enumerate(centers):
        d = _diagonal_cyan_count(g, center)
        claims = _claim_cells(g, center)
        color = COLOR_BY_D[d]
        info.append(
            {
                "center": center,
                "d": d,
                "color": color,
                "claims": claims,
            }
        )
        priority = (d, -center[0], -center[1])
        for cell in claims:
            contested[cell].append(idx)
            if cell not in owners or priority > owners[cell]:
                owners[cell] = priority

    out = g.copy()
    for cell, indices in contested.items():
        best = max(indices, key=lambda i: (info[i]["d"], -info[i]["center"][0], -info[i]["center"][1]))
        out[cell] = info[best]["color"]

    assert out.shape == (h, w)
    return out, info, contested


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect black plus centers and expand each one across the black scaffold with a color chosen by its diagonal cyan count.

    Input structure:
    - The grid is a cyan-and-black canvas where cyan cells are fixed barriers and black cells form several local 5-cell pluses together with surrounding black corridors.
    - A valid seed is a black cell whose north, south, west, and east neighbors are also black.
    - Around each seed, the four diagonal cells may independently be cyan or black; that visible diagonal pattern is the seed's parameter.

    Rule:
    - Find every valid black plus center and count how many of its four diagonal neighbors are cyan, calling that number d.
    - Map d to the expansion color d=0->blue, d=1->red, d=2->green, d=3->yellow, d=4->orange.
    - From each center, propagate through black cells by increasing Manhattan distance, but only along shortest monotone paths that stay on black cells; cyan cells and the grid edge stop growth and are never overwritten.
    - If multiple centers can color the same black cell, keep the one with larger d; break equal-d ties by smaller center row, then smaller center column.

    Key insight:
    - The same cyan layout serves three roles at once: it decides which black pixels qualify as plus centers, it sets each center's color and priority through the diagonal cyan count, and it clips the concentric Manhattan expansion so overlaps must be resolved by that parameter.
    """
    out, _info, _contested = _solve_metadata(np.asarray(grid, dtype=int))
    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells: list[tuple[int, int]] = []
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac, bc + step, step):
            cells.append((ar, c))
    else:
        assert ac == bc
        step = 1 if br > ar else -1
        for r in range(ar, br + step, step):
            cells.append((r, ac))
    return cells


def _template_layout(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g1 = rng.randint(4, 5)
    g2 = rng.randint(4, 5)
    g3 = rng.randint(4, 5)
    g4 = rng.randint(4, 5)
    templates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = [
        (
            [(0, 0), (0, g1), (0, g1 + g2)],
            [(0, 1), (1, 2)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1)],
            [(0, 1), (1, 2)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1)],
            [(0, 1), (1, 2), (1, 3)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1), (2 * g3, g1)],
            [(0, 1), (1, 2), (1, 3), (3, 4)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1), (g3, g1 + g2 + g4)],
            [(0, 1), (1, 2), (1, 3)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1)],
            [(0, 1), (1, 2)],
        ),
        (
            [(0, 0), (0, g1), (0, g1 + g2), (g3, g1), (g3, g1 + g2 + g4)],
            [(0, 1), (1, 2), (1, 3)],
        ),
        (
            [(0, 0), (g3, 0), (g3, g1), (2 * g3, g1)],
            [(0, 1), (1, 2), (2, 3)],
        ),
    ]
    return rng.choice(templates)


def _shift_layout(
    points: list[tuple[int, int]], h: int, w: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    margin = 2
    room_r = h - (max_r - min_r) - 1 - 2 * margin
    room_c = w - (max_c - min_c) - 1 - 2 * margin
    if room_r < 0 or room_c < 0:
        return None
    shift_r = rng.randint(margin - min_r, margin - min_r + room_r)
    shift_c = rng.randint(margin - min_c, margin - min_c + room_c)
    return [(r + shift_r, c + shift_c) for r, c in points]


def _center_used_sides(
    centers: list[tuple[int, int]], edges: list[tuple[int, int]]
) -> dict[int, set[str]]:
    used = {i: set() for i in range(len(centers))}
    for i, j in edges:
        r1, c1 = centers[i]
        r2, c2 = centers[j]
        if r1 == r2:
            if c1 < c2:
                used[i].add("E")
                used[j].add("W")
            else:
                used[i].add("W")
                used[j].add("E")
        else:
            assert c1 == c2
            if r1 < r2:
                used[i].add("S")
                used[j].add("N")
            else:
                used[i].add("N")
                used[j].add("S")
    return used


def _valid_cyan_subsets(required_sides: set[str]) -> dict[int, list[set[str]]]:
    names = list(DIAGONALS)
    out: dict[int, list[set[str]]] = defaultdict(list)
    for mask in range(1 << 4):
        cyan = {names[i] for i in range(4) if (mask >> i) & 1}
        if all(any(diag in cyan for diag in SIDE_TO_DIAGONALS[side]) for side in required_sides):
            out[len(cyan)].append(cyan)
    return out


def _choose_diagonals(
    used_sides: dict[int, set[str]], count: int, rng: random.Random
) -> tuple[list[set[str]], list[int]] | None:
    candidates = [_valid_cyan_subsets(used_sides[i]) for i in range(count)]
    possible_values = [sorted(dmap) for dmap in candidates]
    if any(not vals for vals in possible_values):
        return None

    for _ in range(200):
        ds = [rng.choice(vals) for vals in possible_values]
        if len(set(ds)) < 2:
            continue
        if not any(d == 4 for d in ds) and rng.random() < 0.35:
            extra = [i for i, vals in enumerate(possible_values) if 4 in vals]
            if extra:
                ds[rng.choice(extra)] = 4
        if not any(d == 0 for d in ds):
            iso = [i for i, sides in used_sides.items() if not sides and 0 in possible_values[i]]
            if iso:
                ds[rng.choice(iso)] = 0
        if len(set(ds)) < 2:
            continue
        subsets = [set(rng.choice(candidates[i][ds[i]])) for i in range(count)]
        return subsets, ds
    return None


def _add_center_structure(
    black: set[tuple[int, int]],
    center: tuple[int, int],
    cyan_diagonals: set[str],
) -> None:
    r, c = center
    black.add((r, c))
    for dr, dc in DIR4:
        black.add((r + dr, c + dc))
    for name, (dr, dc) in DIAGONALS.items():
        if name not in cyan_diagonals:
            black.add((r + dr, c + dc))


def _safe_extension_sides(cyan_diagonals: set[str]) -> set[str]:
    safe = set()
    for side, diags in SIDE_TO_DIAGONALS.items():
        if any(diag in cyan_diagonals for diag in diags):
            safe.add(side)
    return safe


def _extend_side(
    black: set[tuple[int, int]],
    center: tuple[int, int],
    side: str,
    length: int,
) -> None:
    dr, dc = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}[side]
    r, c = center
    for step in range(1, length + 1):
        black.add((r + dr * step, c + dc * step))


def _plan_extra_spokes(
    centers: list[tuple[int, int]],
    used_sides: dict[int, set[str]],
    cyan_subsets: list[set[str]],
    rng: random.Random,
) -> list[tuple[int, str, int]]:
    plans: list[tuple[int, str, int]] = []
    for i, center in enumerate(centers):
        safe = _safe_extension_sides(cyan_subsets[i]) - used_sides[i]
        if not safe:
            continue
        side_count = 1 if rng.random() < 0.75 else min(2, len(safe))
        chosen = rng.sample(sorted(safe), k=side_count)
        for side in chosen:
            length = rng.randint(2, 4)
            plans.append((i, side, length))
        if not used_sides[i] and rng.random() < 0.55:
            for side in sorted(safe - set(chosen))[:1]:
                plans.append((i, side, 2))
    return plans


def _paint_black(shape: tuple[int, int], black: set[tuple[int, int]]) -> np.ndarray | None:
    grid = np.full(shape, CYAN, dtype=int)
    for r, c in black:
        if not _in_bounds(shape, r, c):
            return None
        grid[r, c] = BLACK
    return grid


def _grid_from_plan(
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    cyan_subsets: list[set[str]],
    rng: random.Random,
) -> np.ndarray | None:
    black: set[tuple[int, int]] = set()
    for center, subset in zip(centers, cyan_subsets):
        _add_center_structure(black, center, subset)

    used_sides = _center_used_sides(centers, edges)
    for i, j in edges:
        for cell in _segment_cells(centers[i], centers[j]):
            black.add(cell)

    for i, side, length in _plan_extra_spokes(centers, used_sides, cyan_subsets, rng):
        _extend_side(black, centers[i], side, length)

    grid = _paint_black((h, w), black)
    if grid is None:
        return None
    return grid


def _priority_winner(indices: list[int], info: list[dict]) -> int:
    return max(indices, key=lambda i: (info[i]["d"], -info[i]["center"][0], -info[i]["center"][1]))


def _validate_sample(grid: np.ndarray, planned_centers: list[tuple[int, int]]) -> bool:
    found = _find_centers(grid)
    if found != sorted(planned_centers):
        return False

    out, info, contested = _solve_metadata(grid)
    ds = [item["d"] for item in info]
    if not (3 <= len(info) <= 5):
        return False
    if len(set(ds)) < 2:
        return False
    if np.array_equal(grid, out):
        return False

    used_colors = set(int(v) for v in np.unique(out)) - {CYAN, BLACK}
    if len(used_colors) < 2:
        return False

    if not all(max(item["claims"].values()) >= 2 for item in info):
        return False

    if not all(any(out[cell] == item["color"] for cell in item["claims"]) for item in info):
        return False

    different_d_conflict = False
    equal_d_conflict = False
    for cell, indices in contested.items():
        if len(indices) < 2:
            continue
        dvals = {info[i]["d"] for i in indices}
        if len(dvals) >= 2:
            winner = _priority_winner(indices, info)
            if all(
                (info[winner]["d"], -info[winner]["center"][0], -info[winner]["center"][1])
                >= (info[i]["d"], -info[i]["center"][0], -info[i]["center"][1])
                for i in indices
            ):
                different_d_conflict = True
        else:
            ordered = sorted((info[i]["center"], i) for i in indices)
            expected = ordered[0][1]
            if _priority_winner(indices, info) == expected:
                equal_d_conflict = True

    if not different_d_conflict:
        return False

    if len(set(ds)) < len(ds) and not equal_d_conflict:
        return False

    if any(_diagonal_cyan_count(grid, item["center"]) != item["d"] for item in info):
        return False

    for item in info:
        r, c = item["center"]
        if not all(grid[r + dr, c + dc] == BLACK for dr, dc in DIR4):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_SIDE_RANGE)
    w = rng.randint(*GEN_SIDE_RANGE)
    base_points, edges = _template_layout(rng)
    centers = _shift_layout(base_points, h, w, rng)
    if centers is None:
        return None

    planned_centers = sorted(centers)
    index_of = {center: i for i, center in enumerate(centers)}
    edges = [(index_of[centers[i]], index_of[centers[j]]) for i, j in edges]

    used_sides = _center_used_sides(centers, edges)
    chosen = _choose_diagonals(used_sides, len(centers), rng)
    if chosen is None:
        return None
    cyan_subsets, _ds = chosen

    grid = _grid_from_plan(h, w, centers, edges, cyan_subsets, rng)
    if grid is None:
        return None

    if not _validate_sample(grid, planned_centers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid sample")
