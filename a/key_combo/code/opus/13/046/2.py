from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

REGION_COLORS = (BLUE, RED, GREEN)
ROLE_OUTER = 1
ROLE_MIDDLE = 2
ROLE_INNER = 3

GRID_SIZES = (13, 15)
RECT_DIMS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _yellow_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == YELLOW)
    assert coords.size > 0, "expected a yellow occluder"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = grid == YELLOW
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "yellow cells must form a solid rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _recover(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, m = map(int, g.shape)
    assert n == m, "expected a square grid"
    r0, c0, r1, c1 = _yellow_bbox(g)

    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            candidates = (
                (r, n - 1 - c),          # horizontal mirror: left-right reflection
                (n - 1 - r, c),          # vertical mirror: top-bottom reflection
                (n - 1 - r, n - 1 - c),  # diagonal mirror: both reflections
            )
            for rr, cc in candidates:
                value = int(g[rr, cc])
                if value != YELLOW:
                    out[r, c] = value
                    break
            else:
                raise AssertionError("each yellow cell must be recoverable from symmetry")
    return out


def _marker_owner(grid: np.ndarray, r: int, c: int) -> int:
    counts: Counter[int] = Counter(
        int(grid[rr, cc])
        for rr, cc in _neighbors4(r, c)
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) in REGION_COLORS
    )
    assert counts, "every marker must touch a colored region"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "marker ownership must be unique"
    return int(ranked[0][0])


def _count_markers_and_areas(grid: np.ndarray) -> tuple[Counter[int], Counter[int]]:
    g = np.asarray(grid, dtype=int)
    marker_counts: Counter[int] = Counter()
    for r, c in np.argwhere(g == GREY):
        marker_counts[_marker_owner(g, int(r), int(c))] += 1

    area_counts: Counter[int] = Counter(int(v) for v in g.flatten() if int(v) in REGION_COLORS)
    for color in REGION_COLORS:
        marker_counts.setdefault(color, 0)
        area_counts.setdefault(color, 0)
    return marker_counts, area_counts


def _winner_from_counts(marker_counts: Counter[int], area_counts: Counter[int]) -> int:
    present = [color for color in REGION_COLORS if area_counts[color] > 0]
    assert len(present) == 3, "expected exactly three region colors"
    return min(present, key=lambda color: (-marker_counts[color], area_counts[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the symmetric patch hidden by the yellow rectangle, then return the color whose region wins the restored marker vote.

    Input structure:
    - A square grid with black background contains a biaxially symmetric colored pattern.
    - The pattern is made from three large 4-connected regions using blue, red, and green.
    - Grey single-pixel markers sit inside those regions, and one solid yellow rectangle hides
      part of the pattern together with some of the markers.

    Rule:
    - Identify the yellow solid rectangle as the occluder.
    - Replace each yellow cell by its first available mirror value: use the horizontal mirror
      first, otherwise the vertical mirror, otherwise the diagonal mirror.
    - After recovery, count grey markers by the region color that dominates their 4-neighborhood.
    - Output a 1x1 grid containing the color with the largest recovered marker count; if marker
      counts tie, choose the color whose recovered region area is smaller.

    Key insight:
    - The visible markers alone are misleading: the winner is determined only after symmetry
      reconstruction reveals the hidden markers under the yellow rectangle.
    """
    restored = _recover(np.asarray(grid, dtype=int))
    marker_counts, area_counts = _count_markers_and_areas(restored)
    return np.array([[_winner_from_counts(marker_counts, area_counts)]], dtype=int)


def _build_role_grid(n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for role, size in ((ROLE_OUTER, n - 2), (ROLE_MIDDLE, n - 4), (ROLE_INNER, n - 8)):
        start = (n - size) // 2
        stop = start + size
        grid[start:stop, start:stop] = role
    return grid


def _orbit_cells(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
    }
    return tuple(sorted(cells))


def _candidate_orbits(n: int) -> list[dict]:
    role_grid = _build_role_grid(n)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out = []

    for r in range(1, n - 1):
        for c in range(1, n - 1):
            cells = _orbit_cells(n, r, c)
            if cells in seen or len(cells) == 1:
                continue
            seen.add(cells)

            roles = {int(role_grid[rr, cc]) for rr, cc in cells}
            if len(roles) != 1:
                continue
            role = roles.pop()
            if role not in (ROLE_MIDDLE, ROLE_INNER):
                continue

            valid = True
            for rr, cc in cells:
                counts = Counter(
                    int(role_grid[nr, nc])
                    for nr, nc in _neighbors4(rr, cc)
                    if _in_bounds(role_grid.shape, nr, nc) and int(role_grid[nr, nc]) in (ROLE_OUTER, ROLE_MIDDLE, ROLE_INNER)
                )
                ranked = counts.most_common()
                if not ranked or ranked[0][0] != role or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
                    valid = False
                    break
            if not valid:
                continue

            out.append(
                {
                    "role": role,
                    "cells": cells,
                    "size": len(cells),
                    "rep": cells[0],
                }
            )
    return out


def _orbit_conflict(a: dict, b: dict) -> bool:
    for r, c in a["cells"]:
        for rr, cc in b["cells"]:
            if abs(r - rr) + abs(c - cc) <= 1:
                return True
    return False


def _subset_is_independent(orbits: tuple[dict, ...]) -> bool:
    return all(not _orbit_conflict(a, b) for a, b in itertools.combinations(orbits, 2))


def _independent_subsets(orbits: list[dict], max_size: int) -> list[tuple[dict, ...]]:
    subsets = []
    for k in range(1, min(max_size, len(orbits)) + 1):
        for subset in itertools.combinations(orbits, k):
            if _subset_is_independent(subset):
                subsets.append(subset)
    return subsets


def _rectangles_for_branch(n: int, branch: str) -> list[tuple[int, int, int, int]]:
    mid = n // 2
    rects = []
    for h in RECT_DIMS:
        for w in RECT_DIMS:
            for r0 in range(1, n - 1 - h + 1):
                r1 = r0 + h - 1
                for c0 in range(1, n - 1 - w + 1):
                    c1 = c0 + w - 1
                    if branch == "horizontal":
                        if not (r0 <= mid <= r1):
                            continue
                        if c0 <= mid <= c1:
                            continue
                    else:
                        if not (c0 <= mid <= c1):
                            continue
                        if r0 <= mid <= r1:
                            continue
                    rects.append((r0, c0, r1, c1))
    return rects


_ROLE_GRID_CACHE = {n: _build_role_grid(n) for n in GRID_SIZES}
_ORBIT_CACHE = {n: _candidate_orbits(n) for n in GRID_SIZES}
_ORBIT_BY_REP = {n: {orbit["rep"]: orbit for orbit in _ORBIT_CACHE[n]} for n in GRID_SIZES}
_SUBSET_CACHE = {
    n: {
        ROLE_MIDDLE: _independent_subsets([o for o in _ORBIT_CACHE[n] if o["role"] == ROLE_MIDDLE], 3),
        ROLE_INNER: _independent_subsets([o for o in _ORBIT_CACHE[n] if o["role"] == ROLE_INNER], 3),
    }
    for n in GRID_SIZES
}
_RECT_CACHE = {
    n: {
        "horizontal": _rectangles_for_branch(n, "horizontal"),
        "vertical": _rectangles_for_branch(n, "vertical"),
    }
    for n in GRID_SIZES
}


def _hidden_count(rect: tuple[int, int, int, int], orbit: dict) -> int:
    r0, c0, r1, c1 = rect
    return sum(r0 <= r <= r1 and c0 <= c <= c1 for r, c in orbit["cells"])


def _subsets_compatible(a: tuple[dict, ...], b: tuple[dict, ...]) -> bool:
    return all(not _orbit_conflict(x, y) for x in a for y in b)


def _evaluate_subset(rect: tuple[int, int, int, int], subset: tuple[dict, ...]) -> tuple[int, int]:
    full = sum(orbit["size"] for orbit in subset)
    hidden = sum(_hidden_count(rect, orbit) for orbit in subset)
    return full, hidden


_SCENARIO_TEMPLATES = [
    {
        "n": 13,
        "branch": "horizontal",
        "target": "inner_tie",
        "rect": (4, 3, 8, 5),
        "inner_reps": ((4, 5), (5, 4)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
    {
        "n": 13,
        "branch": "vertical",
        "target": "inner_tie",
        "rect": (3, 4, 5, 8),
        "inner_reps": ((4, 5), (5, 4)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
    {
        "n": 13,
        "branch": "horizontal",
        "target": "middle_strict",
        "rect": (4, 1, 6, 3),
        "inner_reps": ((4, 5), (5, 4)),
        "middle_reps": ((4, 3), (5, 2), (6, 3)),
        "final_role": ROLE_MIDDLE,
        "visible_role": ROLE_INNER,
    },
    {
        "n": 13,
        "branch": "vertical",
        "target": "middle_strict",
        "rect": (1, 4, 3, 8),
        "inner_reps": ((4, 5), (5, 4)),
        "middle_reps": ((2, 5), (3, 4), (3, 6)),
        "final_role": ROLE_MIDDLE,
        "visible_role": ROLE_INNER,
    },
    {
        "n": 15,
        "branch": "horizontal",
        "target": "inner_tie",
        "rect": (6, 4, 8, 6),
        "inner_reps": ((6, 4), (6, 6)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
    {
        "n": 15,
        "branch": "vertical",
        "target": "inner_tie",
        "rect": (4, 6, 6, 8),
        "inner_reps": ((4, 6), (6, 6)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
    {
        "n": 15,
        "branch": "horizontal",
        "target": "inner_strict",
        "rect": (5, 4, 9, 6),
        "inner_reps": ((5, 4), (5, 6), (6, 5)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
    {
        "n": 15,
        "branch": "vertical",
        "target": "inner_strict",
        "rect": (4, 5, 6, 9),
        "inner_reps": ((4, 5), (5, 6), (6, 5)),
        "middle_reps": ((2, 3), (2, 5)),
        "final_role": ROLE_INNER,
        "visible_role": ROLE_MIDDLE,
    },
]


def _mirror_rect_lr(n: int, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0, n - 1 - c1, r1, n - 1 - c0


def _mirror_rect_ud(n: int, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return n - 1 - r1, c0, n - 1 - r0, c1


def _materialize_templates() -> list[dict]:
    scenarios = []
    for template in _SCENARIO_TEMPLATES:
        n = template["n"]
        base = {
            "n": n,
            "branch": template["branch"],
            "target": template["target"],
            "final_role": template["final_role"],
            "visible_role": template["visible_role"],
            "inner_subset": tuple(_ORBIT_BY_REP[n][rep] for rep in template["inner_reps"]),
            "middle_subset": tuple(_ORBIT_BY_REP[n][rep] for rep in template["middle_reps"]),
        }
        scenarios.append({**base, "rect": template["rect"]})
        if template["branch"] == "horizontal":
            scenarios.append({**base, "rect": _mirror_rect_lr(n, template["rect"])})
        else:
            scenarios.append({**base, "rect": _mirror_rect_ud(n, template["rect"])})
    return scenarios


_SCENARIO_POOL = _materialize_templates()


def _paint_clean_grid(n: int, role_colors: dict[int, int], inner_subset: tuple[dict, ...], middle_subset: tuple[dict, ...]) -> np.ndarray:
    role_grid = _ROLE_GRID_CACHE[n]
    clean = np.zeros((n, n), dtype=int)
    for role, color in role_colors.items():
        clean[role_grid == role] = int(color)

    for orbit in inner_subset + middle_subset:
        for r, c in orbit["cells"]:
            clean[r, c] = GREY
    return clean


def _visible_vote(grid: np.ndarray) -> int:
    marker_counts, area_counts = _count_markers_and_areas(grid)
    return _winner_from_counts(marker_counts, area_counts)


def _role_from_color(role_colors: dict[int, int], color: int) -> int:
    for role, mapped in role_colors.items():
        if mapped == color:
            return role
    raise AssertionError("unknown role color")


def _validate_generated(grid: np.ndarray, clean: np.ndarray, scenario: dict, role_colors: dict[int, int]) -> bool:
    rect = scenario["rect"]
    r0, c0, r1, c1 = rect

    hidden_marker_count = int(np.sum((clean == GREY) & (grid == YELLOW)))
    if hidden_marker_count <= 0:
        return False

    restored = _recover(grid)
    if not np.array_equal(restored, clean):
        return False

    visible_color = _visible_vote(grid)
    final_color = int(main(grid)[0, 0])
    visible_role = _role_from_color(role_colors, visible_color)
    final_role = _role_from_color(role_colors, final_color)
    if visible_role != scenario["visible_role"] or final_role != scenario["final_role"]:
        return False
    if visible_role == final_role:
        return False

    visible_counts, _ = _count_markers_and_areas(grid)
    final_counts, final_areas = _count_markers_and_areas(clean)

    if visible_counts[visible_color] <= max(
        count for color, count in visible_counts.items() if color != visible_color
    ):
        return False

    final_winner = _winner_from_counts(final_counts, final_areas)
    if final_winner != final_color:
        return False

    if scenario["target"] == "inner_tie":
        inner_color = role_colors[ROLE_INNER]
        middle_color = role_colors[ROLE_MIDDLE]
        if final_counts[inner_color] != final_counts[middle_color]:
            return False
        if final_areas[inner_color] >= final_areas[middle_color]:
            return False
    elif scenario["target"] == "inner_strict":
        if final_counts[role_colors[ROLE_INNER]] <= final_counts[role_colors[ROLE_MIDDLE]]:
            return False
    else:
        if final_counts[role_colors[ROLE_MIDDLE]] <= final_counts[role_colors[ROLE_INNER]]:
            return False

    uses_vertical_fallback = scenario["branch"] == "vertical"
    if uses_vertical_fallback:
        if not any(grid[r, clean.shape[0] - 1 - c] == YELLOW for r, c in zip(*np.where(grid == YELLOW))):
            return False
    else:
        if any(grid[r, clean.shape[0] - 1 - c] == YELLOW for r, c in zip(*np.where(grid == YELLOW))):
            return False

    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == YELLOW):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scenario = rng.choice(_SCENARIO_POOL)

        colors = list(REGION_COLORS)
        rng.shuffle(colors)
        role_colors = {
            ROLE_OUTER: colors[0],
            ROLE_MIDDLE: colors[1],
            ROLE_INNER: colors[2],
        }

        clean = _paint_clean_grid(
            scenario["n"],
            role_colors,
            scenario["inner_subset"],
            scenario["middle_subset"],
        )

        inp = clean.copy()
        r0, c0, r1, c1 = scenario["rect"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = YELLOW

        if _validate_generated(inp, clean, scenario, role_colors):
            return inp

    raise RuntimeError("failed to generate a valid symmetric-vote input")
