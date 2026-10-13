from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
PASSABLE = {BLACK, GREY}
GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
        (c, r),
        (c, n - 1 - r),
        (n - 1 - c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(pts))


def all_d4_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = d4_orbit(r, c, n)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def is_d4_symmetric(grid: np.ndarray) -> bool:
    n = int(grid.shape[0])
    return all(len({int(grid[r, c]) for r, c in orbit}) == 1 for orbit in all_d4_orbits(n))


def repair_orbit_errors(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]
    n = int(g.shape[0])
    out = g.copy()
    repaired: list[tuple[int, int]] = []
    inconsistent_orbits: list[tuple[tuple[int, int], ...]] = []

    for orbit in all_d4_orbits(n):
        vals = [int(g[r, c]) for r, c in orbit]
        uniq = set(vals)
        if len(uniq) == 1:
            continue
        assert uniq <= {GREY, TEAL}, "only grey/teal disagreement may break symmetry"
        counts = Counter(vals)
        assert counts[GREY] != counts[TEAL], "broken orbit must have a unique majority"
        minority = GREY if counts[GREY] < counts[TEAL] else TEAL
        inconsistent_orbits.append(orbit)
        for r, c in orbit:
            if int(g[r, c]) == minority:
                out[r, c] = TEAL if minority == GREY else GREY
                repaired.append((r, c))
    return out, repaired, inconsistent_orbits


def fill_enclosed_regions(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    passable = np.isin(g, tuple(PASSABLE))
    visited = np.zeros((h, w), dtype=bool)
    out = g.copy()
    filled: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or not passable[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            cells: list[tuple[int, int]] = []
            cell_set: set[tuple[int, int]] = set()
            touches_edge = False
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                cell_set.add((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            if touches_edge:
                continue

            black_cells = [(r, c) for r, c in cells if int(g[r, c]) == BLACK]
            if not black_cells:
                continue

            border_counts: Counter[int] = Counter()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in cell_set:
                        continue
                    val = int(g[nr, nc])
                    if val not in PASSABLE:
                        border_counts[val] += 1

            assert border_counts, "every enclosed region needs a colored boundary"
            most_common = border_counts.most_common()
            top_color, top_count = most_common[0]
            second_count = most_common[1][1] if len(most_common) > 1 else 0
            assert top_count > second_count and top_count * 2 > sum(border_counts.values())
            for r, c in black_cells:
                out[r, c] = top_color
            filled.append(
                {
                    "cells": tuple(black_cells),
                    "fill_color": int(top_color),
                    "boundary_counts": dict(border_counts),
                }
            )

    return out, filled


def swap_entire_mixed_orbits(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    n = int(g.shape[0])
    for orbit in all_d4_orbits(n):
        vals = {int(g[r, c]) for r, c in orbit}
        if len(vals) == 1 or not vals <= {GREY, TEAL}:
            continue
        for r, c in orbit:
            if int(g[r, c]) == GREY:
                out[r, c] = TEAL
            elif int(g[r, c]) == TEAL:
                out[r, c] = GREY
    return out


def swap_all_grey_teal(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == GREY] = TEAL
    out[g == TEAL] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair asymmetric grey/teal orbit errors, then fill enclosed black rooms by their repaired boundary majority.

    Input structure:
    - The input is an odd square grid with a nearly D4-symmetric multicolor pattern on a black background.
    - Grey and teal are the only colors that may disagree within a D4 orbit; grey cells act as leaky gaps when testing enclosure.
    - Black rooms may become enclosed only after the minority grey/teal mistakes are repaired.

    Rule:
    - Partition the square into D4 reflection orbits.
    - For every non-uniform orbit whose disagreement is only between grey and teal, swap the minority-colored orbit members so the orbit matches its majority.
    - On the repaired grid, find 4-connected passable regions made of black or grey; any such region that does not touch the outer border is enclosed.
    - For each enclosed region, count the colors on its 4-neighbor boundary excluding black and grey, take the unique strict majority color, and fill that region's black cells with it.

    Key insight:
    - The task is not a global grey/teal substitution: D4 orbit analysis isolates exactly which minority error pixels to repair, and only that repair closes the leaks that determine which black rooms get filled.
    """
    g = np.asarray(grid, dtype=int)
    repaired, _repaired_cells, _bad_orbits = repair_orbit_errors(g)
    filled, _regions = fill_enclosed_regions(repaired)
    return filled


def paint_orbit(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    *,
    require_empty: bool = True,
) -> tuple[tuple[int, int], ...]:
    n = int(grid.shape[0])
    orbit = d4_orbit(start[0], start[1], n)
    for r, c in orbit:
        if require_empty:
            assert int(grid[r, c]) == BLACK
        else:
            assert int(grid[r, c]) in {BLACK, color}
        grid[r, c] = color
    return orbit


def draw_ring(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def add_side_pockets(
    grid: np.ndarray,
    pocket_dist: int,
    pocket_color: int,
    accent_color: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    n = int(grid.shape[0])
    m = n // 2
    centers = [
        (pocket_dist, m),
        (m, n - 1 - pocket_dist),
        (n - 1 - pocket_dist, m),
        (m, pocket_dist),
    ]
    gate_orbit = (
        (pocket_dist - 1, m),
        (m, n - pocket_dist),
        (n - pocket_dist, m),
        (m, pocket_dist - 1),
    )
    corner_orbit: list[tuple[int, int]] = []
    for cr, cc in centers:
        r0, c0 = cr - 1, cc - 1
        r1, c1 = cr + 1, cc + 1
        draw_ring(grid, r0, c0, r1, c1, pocket_color)
        grid[cr, cc] = BLACK
        for rr, cc2 in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            grid[rr, cc2] = accent_color
            corner_orbit.append((rr, cc2))
    for r, c in gate_orbit:
        grid[r, c] = TEAL
    return tuple(gate_orbit), tuple(sorted(set(corner_orbit)))


def build_fixed_grid() -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    n = random.choice([11, 13])
    grid = np.zeros((n, n), dtype=int)
    colors = random.sample([1, 2, 3, 6, 7, 9], 4)
    central_major, pocket_major, accent_color, outer_color = colors
    inner_radius = 1 if n == 11 else random.choice([1, 2])
    pocket_dist = 1 if n == 11 or inner_radius == 2 else random.choice([1, 2])
    m = n // 2

    orbits: dict[str, dict[str, object]] = {}

    top = m - inner_radius - 1
    bottom = m + inner_radius + 1
    left = top
    right = bottom
    draw_ring(grid, top, left, bottom, right, central_major)
    central_corners = ((top, left), (top, right), (bottom, left), (bottom, right))
    for r, c in central_corners:
        grid[r, c] = accent_color
    central_gate_orbit = ((top, m), (m, right), (bottom, m), (m, left))
    for r, c in central_gate_orbit:
        grid[r, c] = TEAL
    orbits["central_gate"] = {"cells": central_gate_orbit, "base_color": TEAL, "critical": True}

    pocket_gate_orbit, pocket_corner_cells = add_side_pockets(
        grid,
        pocket_dist,
        pocket_major,
        accent_color,
    )
    orbits["pocket_gate"] = {"cells": pocket_gate_orbit, "base_color": TEAL, "critical": True}

    orbits["corner_grey"] = {
        "cells": paint_orbit(grid, (1, 1), GREY),
        "base_color": GREY,
        "critical": False,
    }
    orbits["near_corner_teal"] = {
        "cells": paint_orbit(grid, (1, 2), TEAL),
        "base_color": TEAL,
        "critical": False,
    }
    orbits["diag_grey"] = {
        "cells": paint_orbit(grid, (2, 2), GREY),
        "base_color": GREY,
        "critical": False,
    }

    extra_candidates = [
        ((1, 3), outer_color),
        ((2, 3), TEAL),
        ((3, 1), outer_color),
    ]
    extra_index = 0
    for start, color in extra_candidates:
        orbit = d4_orbit(start[0], start[1], n)
        if all(int(grid[r, c]) == BLACK for r, c in orbit):
            name = f"extra_{extra_index}"
            orbits[name] = {
                "cells": paint_orbit(grid, start, color),
                "base_color": color,
                "critical": False,
            }
            extra_index += 1

    grid[m, m] = BLACK
    assert is_d4_symmetric(grid)
    assert set(pocket_corner_cells)
    return grid, orbits


def inject_errors(
    fixed: np.ndarray,
    orbit_defs: dict[str, dict[str, object]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    inp = fixed.copy()
    changed: list[tuple[int, int]] = []

    selections = [
        ("central_gate", 1),
        ("corner_grey", 1),
        ("pocket_gate", 1),
    ]

    optional = ["near_corner_teal", "diag_grey"] + [name for name in orbit_defs if name.startswith("extra_")]
    random.shuffle(optional)
    target_errors = random.randint(3, 6)
    current = sum(count for _, count in selections)
    for name in optional:
        orbit = orbit_defs[name]["cells"]
        limit = max(1, len(orbit) // 3)
        count = 1 if current >= target_errors - 1 else random.randint(1, min(2, limit))
        if current + count > target_errors:
            count = target_errors - current
        if count <= 0:
            continue
        selections.append((name, count))
        current += count
        if current == target_errors:
            break

    if current < target_errors:
        return inp, []

    for name, count in selections:
        orbit = list(orbit_defs[name]["cells"])
        base_color = int(orbit_defs[name]["base_color"])
        wrong_color = TEAL if base_color == GREY else GREY
        assert count < len(orbit) / 2
        for r, c in random.sample(orbit, count):
            inp[r, c] = wrong_color
            changed.append((r, c))

    return inp, changed


def edge_connected_black_sets(grid: np.ndarray) -> set[frozenset[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    passable = np.isin(g, tuple(PASSABLE))
    visited = np.zeros((h, w), dtype=bool)
    comps: set[frozenset[tuple[int, int]]] = set()
    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or not passable[r0, c0]:
                continue
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            cells: set[tuple[int, int]] = set()
            touches_edge = False
            has_black = False
            while q:
                r, c = q.popleft()
                cells.add((r, c))
                if int(g[r, c]) == BLACK:
                    has_black = True
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if has_black and not touches_edge:
                comps.add(frozenset((r, c) for r, c in cells if int(g[r, c]) == BLACK))
    return comps


def validate_instance(inp: np.ndarray, fixed: np.ndarray, expected_errors: list[tuple[int, int]]) -> bool:
    if not (inp.ndim == 2 and inp.shape[0] == inp.shape[1] and int(inp.shape[0]) <= 30):
        return False
    if not (3 <= len(expected_errors) <= 6):
        return False
    if is_d4_symmetric(inp) or not is_d4_symmetric(fixed):
        return False

    try:
        repaired, repaired_cells, bad_orbits = repair_orbit_errors(inp)
    except AssertionError:
        return False

    if repaired_cells != sorted(repaired_cells):
        repaired_cells = sorted(repaired_cells)
    if sorted(repaired_cells) != sorted(expected_errors):
        return False
    if not np.array_equal(repaired, fixed):
        return False
    if len(bad_orbits) < 3:
        return False

    solved = main(inp)
    expected_out, filled_info = fill_enclosed_regions(fixed)
    if not np.array_equal(solved, expected_out):
        return False
    if np.array_equal(repaired, solved):
        return False
    if not filled_info:
        return False

    try_fill_only, _ = fill_enclosed_regions(inp)
    if np.array_equal(try_fill_only, solved):
        return False

    wrong_orbit_swap, _ = fill_enclosed_regions(swap_entire_mixed_orbits(inp))
    if np.array_equal(wrong_orbit_swap, solved):
        return False

    wrong_global_swap, _ = fill_enclosed_regions(swap_all_grey_teal(inp))
    if np.array_equal(wrong_global_swap, solved):
        return False

    before = edge_connected_black_sets(inp)
    after = edge_connected_black_sets(fixed)
    if before == after:
        return False
    if len(after) <= len(before):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        fixed, orbit_defs = build_fixed_grid()
        inp, errors = inject_errors(fixed, orbit_defs)
        if errors and validate_instance(inp, fixed, errors):
            return inp
    raise RuntimeError("failed to generate a valid symmetric swap-and-fill input")
