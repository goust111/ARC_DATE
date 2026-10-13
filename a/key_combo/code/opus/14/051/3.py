from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

GRID_SIZES = (9, 11, 13, 15)
MAX_GENERATION_ATTEMPTS = 500
EXTRA_BLOCKED_MARKER_PROB = 0.45
PREFER_MARKER_MASK_ORBIT_PROB = 0.5


def _orbit4(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        orbit.add((rr, cc))
        rr, cc = cc, n - 1 - rr
    return tuple(sorted(orbit))


def _toward_center_step(n: int, r: int, c: int) -> tuple[int, int]:
    center = n // 2
    dr = (center > r) - (center < r)
    dc = (center > c) - (center < c)
    assert dr != 0 and dc != 0, "marker must not lie on the center row or column"
    return int(dr), int(dc)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _apply_marker_instruction(
    out: np.ndarray,
    marker: tuple[int, int],
    expected_color: int,
) -> None:
    r, c = marker
    dr, dc = _toward_center_step(out.shape[0], r, c)
    if expected_color == RED:
        nr, nc = r + dr, c + dc
        paint = GREEN
    else:
        nr, nc = r - dr, c - dc
        paint = YELLOW
    assert _in_bounds(out.shape, nr, nc)
    if int(out[nr, nc]) == BLACK:
        out[nr, nc] = paint


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a masked fourfold-symmetric pattern, then decode each orange marker into a diagonal offset instruction.

    Input structure:
    - The grid is a square black canvas containing a centrally placed pattern made only of red and blue orbit-cells.
    - The full pattern is invariant under 90-degree rotation around the grid center, but some pattern cells are replaced by grey masks.
    - Two to four orange marker pixels also replace pattern cells; each marker belongs to a four-cell rotation orbit whose true color is still visible elsewhere in that orbit.

    Rule:
    - Treat every non-black cell as belonging to a fourfold rotation orbit about the grid center.
    - For each orbit, use its visible non-grey non-orange cells to recover the unique red-or-blue color of that orbit, and fill every grey cell in that orbit with that color.
    - Keep orange markers unchanged, then decode each marker's recovered orbit color: red means paint one green pixel one diagonal step toward the center, blue means paint one yellow pixel one diagonal step away from the center.
    - If that destination is already non-black after restoration, do not paint there.

    Key insight:
    - Orange markers do not encode direction by themselves; their hidden original orbit color, revealed only through rotational symmetry, is the actual movement instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, w = g.shape
    assert n == w and n % 2 == 1, "expected an odd square grid"

    out = g.copy()
    orbit_color: dict[tuple[tuple[int, int], ...], int] = {}
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in range(n):
        for c in range(n):
            orbit = _orbit4(n, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            values = [int(g[rr, cc]) for rr, cc in orbit]
            if not any(v != BLACK for v in values):
                continue

            visible = {v for v in values if v not in (BLACK, GREY, ORANGE)}
            assert len(visible) == 1, "each occupied orbit must expose one recoverable color"
            color = visible.pop()
            assert color in (BLUE, RED)
            orbit_color[orbit] = color

            for rr, cc in orbit:
                assert int(g[rr, cc]) != BLACK, "pattern orbits must stay fully present"
                if int(out[rr, cc]) == GREY:
                    out[rr, cc] = color

    for r, c in np.argwhere(g == ORANGE):
        orbit = _orbit4(n, int(r), int(c))
        assert orbit in orbit_color, "marker must belong to a visible pattern orbit"
        _apply_marker_instruction(out, (int(r), int(c)), orbit_color[orbit])

    return out


def _candidate_anchors(n: int, radius: int) -> list[tuple[int, int]]:
    center = n // 2
    anchors = []
    for r in range(max(1, center - radius), center):
        for c in range(center + 1, min(n - 2, center + radius) + 1):
            anchors.append((r, c))
    return anchors


def _grow_anchor_cluster(candidates: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    candidate_set = set(candidates)
    selected = {random.choice(candidates)}
    while len(selected) < target:
        frontier = [
            cell
            for cell in candidate_set - selected
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in selected)
        ]
        pool = frontier if frontier else list(candidate_set - selected)
        selected.add(random.choice(pool))
    return sorted(selected)


def _build_clean_pattern() -> tuple[np.ndarray, dict[tuple[tuple[int, int], ...], int]]:
    n = random.choice(GRID_SIZES)
    center = n // 2
    radius = random.randint(2, max(2, center - 1))
    candidates = _candidate_anchors(n, radius)
    if len(candidates) < 4:
        raise AssertionError("need at least four candidate orbits")
    orbit_count = random.randint(4, min(10, len(candidates)))
    anchors = _grow_anchor_cluster(candidates, orbit_count)

    colors = [RED] * random.randint(1, orbit_count - 1)
    colors += [BLUE] * (orbit_count - len(colors))
    random.shuffle(colors)

    grid = np.zeros((n, n), dtype=int)
    orbit_colors: dict[tuple[tuple[int, int], ...], int] = {}
    for anchor, color in zip(anchors, colors):
        orbit = _orbit4(n, *anchor)
        orbit_colors[orbit] = color
        for r, c in orbit:
            grid[r, c] = color
    return grid, orbit_colors


def _marker_candidates(
    clean: np.ndarray,
    orbit_colors: dict[tuple[tuple[int, int], ...], int],
) -> list[dict]:
    candidates = []
    for orbit, color in orbit_colors.items():
        for r, c in orbit:
            dr, dc = _toward_center_step(clean.shape[0], r, c)
            if color == RED:
                nr, nc = r + dr, c + dc
            else:
                nr, nc = r - dr, c - dc
            if not _in_bounds(clean.shape, nr, nc):
                continue
            candidates.append(
                {
                    "orbit": orbit,
                    "marker": (r, c),
                    "expected": color,
                    "target": (nr, nc),
                    "blocked": int(clean[nr, nc]) != BLACK,
                }
            )
    return candidates


def _pick_required_markers(candidates: list[dict]) -> list[dict] | None:
    reds = [cand for cand in candidates if cand["expected"] == RED and not cand["blocked"]]
    blues = [cand for cand in candidates if cand["expected"] == BLUE and not cand["blocked"]]
    random.shuffle(reds)
    random.shuffle(blues)

    for red in reds:
        for blue in blues:
            if red["orbit"] == blue["orbit"]:
                continue
            if red["target"] == blue["target"]:
                continue
            return [red, blue]
    return None


def _extend_markers(base_markers: list[dict], candidates: list[dict]) -> list[dict]:
    markers = list(base_markers)
    used_orbits = {cand["orbit"] for cand in markers}
    used_markers = {cand["marker"] for cand in markers}
    used_targets = {cand["target"] for cand in markers}

    extras = [cand for cand in candidates if cand["orbit"] not in used_orbits]
    blocked_pool = [cand for cand in extras if cand["blocked"]]
    open_pool = [cand for cand in extras if not cand["blocked"]]

    extra_count = random.randint(0, 2)
    chosen_extra = 0
    while chosen_extra < extra_count:
        want_blocked = blocked_pool and random.random() < EXTRA_BLOCKED_MARKER_PROB
        pool = blocked_pool if want_blocked else open_pool
        pool = [
            cand
            for cand in pool
            if cand["orbit"] not in used_orbits
            and cand["marker"] not in used_markers
            and cand["target"] not in used_markers
            and cand["target"] not in used_targets
        ]
        if not pool:
            break
        cand = random.choice(pool)
        markers.append(cand)
        used_orbits.add(cand["orbit"])
        used_markers.add(cand["marker"])
        used_targets.add(cand["target"])
        chosen_extra += 1

    random.shuffle(markers)
    return markers


def _choose_masks(
    orbit_colors: dict[tuple[tuple[int, int], ...], int],
    markers: list[dict],
) -> set[tuple[int, int]]:
    marker_by_orbit = {marker["orbit"]: marker["marker"] for marker in markers}
    orbit_cells = {orbit: list(orbit) for orbit in orbit_colors}
    mask_counts = {orbit: 0 for orbit in orbit_colors}
    masks: set[tuple[int, int]] = set()

    heavy_orbits = [
        orbit
        for orbit in orbit_colors
        if len([cell for cell in orbit_cells[orbit] if cell != marker_by_orbit.get(orbit)]) >= 3
        or orbit in marker_by_orbit
    ]
    if random.random() < PREFER_MARKER_MASK_ORBIT_PROB:
        preferred = [orbit for orbit in heavy_orbits if orbit in marker_by_orbit]
        if preferred:
            heavy_orbits = preferred
    heavy_orbit = random.choice(heavy_orbits)

    heavy_choices = [cell for cell in orbit_cells[heavy_orbit] if cell != marker_by_orbit.get(heavy_orbit)]
    random.shuffle(heavy_choices)
    for cell in heavy_choices[:2]:
        masks.add(cell)
        mask_counts[heavy_orbit] += 1

    capacity = {
        orbit: len([cell for cell in orbit_cells[orbit] if cell != marker_by_orbit.get(orbit)]) - 1
        for orbit in orbit_colors
    }
    remaining_capacity = sum(max(0, capacity[orbit] - mask_counts[orbit]) for orbit in orbit_colors)
    extra_mask_count = random.randint(0, min(4, remaining_capacity))

    for _ in range(extra_mask_count):
        eligible = [
            orbit
            for orbit in orbit_colors
            if mask_counts[orbit] < capacity[orbit]
        ]
        if not eligible:
            break
        orbit = random.choice(eligible)
        choices = [
            cell
            for cell in orbit_cells[orbit]
            if cell != marker_by_orbit.get(orbit) and cell not in masks
        ]
        if not choices:
            continue
        cell = random.choice(choices)
        masks.add(cell)
        mask_counts[orbit] += 1

    return masks


def _render_input_and_output(
    clean: np.ndarray,
    orbit_colors: dict[tuple[tuple[int, int], ...], int],
    markers: list[dict],
    masks: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = clean.copy()
    for r, c in masks:
        inp[r, c] = GREY
    for marker in markers:
        r, c = marker["marker"]
        inp[r, c] = ORANGE

    expected = clean.copy()
    for marker in markers:
        r, c = marker["marker"]
        expected[r, c] = ORANGE
    for marker in markers:
        _apply_marker_instruction(expected, marker["marker"], marker["expected"])
    return inp, expected


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    clean: np.ndarray,
    orbit_colors: dict[tuple[tuple[int, int], ...], int],
    markers: list[dict],
    masks: set[tuple[int, int]],
) -> bool:
    n = inp.shape[0]
    if inp.shape != expected.shape or inp.shape != clean.shape:
        return False
    if inp.ndim != 2 or inp.shape[0] != inp.shape[1] or n > 30:
        return False
    if not set(map(int, np.unique(inp))).issubset({BLACK, BLUE, RED, GREY, ORANGE}):
        return False

    expected_colors = {marker["expected"] for marker in markers}
    if expected_colors != {BLUE, RED}:
        return False
    if not (2 <= len(markers) <= 4):
        return False
    if len(masks) < 2:
        return False

    heavy_mask_orbits = 0
    for orbit, color in orbit_colors.items():
        orbit_values = [int(clean[r, c]) for r, c in orbit]
        if any(v != color for v in orbit_values):
            return False

        inp_values = [int(inp[r, c]) for r, c in orbit]
        grey_count = sum(v == GREY for v in inp_values)
        orange_count = sum(v == ORANGE for v in inp_values)
        visible_count = sum(v in (BLUE, RED) for v in inp_values)
        if grey_count > 0 and visible_count < 1:
            return False
        if orange_count > 0 and visible_count < 1:
            return False
        if grey_count >= 2:
            heavy_mask_orbits += 1

    if heavy_mask_orbits < 1:
        return False

    red_open = 0
    blue_open = 0
    blocked_markers = 0
    for marker in markers:
        r, c = marker["marker"]
        if r == n // 2 or c == n // 2:
            return False
        tr, tc = marker["target"]
        if not _in_bounds(inp.shape, tr, tc):
            return False
        if marker["blocked"]:
            blocked_markers += 1
        elif marker["expected"] == RED:
            red_open += 1
        else:
            blue_open += 1

    if red_open < 1 or blue_open < 1:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    all_toward = clean.copy()
    all_away = clean.copy()
    for marker in markers:
        r, c = marker["marker"]
        all_toward[r, c] = ORANGE
        all_away[r, c] = ORANGE
    for marker in markers:
        _apply_marker_instruction(all_toward, marker["marker"], RED)
        _apply_marker_instruction(all_away, marker["marker"], BLUE)
    if np.array_equal(all_toward, expected) or np.array_equal(all_away, expected):
        return False

    return True


def _generate_once() -> np.ndarray | None:
    clean, orbit_colors = _build_clean_pattern()
    candidates = _marker_candidates(clean, orbit_colors)
    required = _pick_required_markers(candidates)
    if required is None:
        return None
    markers = _extend_markers(required, candidates)
    masks = _choose_masks(orbit_colors, markers)
    inp, expected = _render_input_and_output(clean, orbit_colors, markers, masks)

    if not _validate_semantics(inp, expected, clean, orbit_colors, markers, masks):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
