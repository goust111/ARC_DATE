from __future__ import annotations

import numpy as np

from arc_utils import bbox, get_bg
import itertools
import math
import random


def _rot90_orbit(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = side - 1
    return (
        (r, c),
        (c, last - r),
        (last - r, last - c),
        (last - c, r),
    )


def _rotation_closure(
    g: np.ndarray,
    bg: int,
    side: int,
    r0: int,
    c0: int,
    coords: np.ndarray,
) -> dict[tuple[int, int], int] | None:
    painted: dict[tuple[int, int], int] = {}
    for r, c in coords:
        color = int(g[r, c])
        assert color != bg, "closure coordinates must be foreground"
        rr, cc = int(r - r0), int(c - c0)
        for pos in _rot90_orbit(side, rr, cc):
            old = painted.get(pos)
            if old is not None and old != color:
                return None
            painted[pos] = color
    return painted


def _find_rotation_square(g: np.ndarray, bg: int) -> tuple[int, int, dict[tuple[int, int], int]]:
    mask = g != bg
    b = bbox(mask)
    assert b is not None, "expected at least one foreground pixel"
    rmin, cmin, rmax, cmax = b
    coords = np.argwhere(mask)

    min_side = max(rmax - rmin + 1, cmax - cmin + 1)
    max_side = min(g.shape)
    for side in range(min_side, max_side + 1):
        candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
        for r0 in range(g.shape[0] - side + 1):
            if not (r0 <= rmin and rmax < r0 + side):
                continue
            for c0 in range(g.shape[1] - side + 1):
                if not (c0 <= cmin and cmax < c0 + side):
                    continue
                painted = _rotation_closure(g, bg, side, r0, c0, coords)
                if painted is not None:
                    candidates.append((r0, c0, painted))

        if candidates:
            assert len(candidates) == 1, "expected unique smallest rotation square"
            r0, c0, painted = candidates[0]
            return r0, c0, painted

    raise AssertionError("expected a rotation-consistent square containing the foreground")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the foreground pattern by rotating it into a square with four-way rotational symmetry.

    Input structure
    - The grid has a dominant background and a partial foreground pattern inside one hidden square region.
    - The intended square is the unique smallest square containing all foreground pixels whose 90-degree rotation closure has no color conflict.

    Rule
    - Keep the grid shape and background.
    - For each foreground pixel in the inferred square, paint its positions under 0, 90, 180, and 270 degree rotations about the square center.

    Key insight
    The missing cells are not mirrored; they are exactly the orbit needed to complete quarter-turn rotational symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    r0, c0, painted = _find_rotation_square(g, bg)

    out = np.full_like(g, bg)
    for (rr, cc), color in painted.items():
        out[r0 + rr, c0 + cc] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and placement.
CANVAS_SIZE, BACKGROUND = 10, 0
FOREGROUND_COLORS = range(1, 10)
SIDE_MIN, SIDE_MAX = 4, 7
PLACEMENT_MARGIN = 1

# Fill, partial-orbit, and accent controls.
ORBIT_SIZE = 4
FILL_MIN_FRAC, FILL_MAX_FRAC = 0.50, 0.75
PARTIAL_ORBITS_MIN, PARTIAL_ORBITS_MAX = 1, 5
ACCENT_ORBITS_MIN, ACCENT_ORBITS_MAX = 1, 2

# Ranking, retention, and retry controls.
ORBIT_RANK_JITTER = 0.1
SECTOR_COUNT = 4
PREFERRED_RETAIN_PROB = 0.8
SPANNING_SCORE_TOLERANCE = 1
GEN_MAX_ATTEMPTS = 300


def _gen_orbits(side: int) -> list[dict]:
    """Return unique C4 orbits with ranking fields used by layout selection."""
    seen = set()
    orbits = []
    center = (side - 1) / 2
    for r in range(side):
        for c in range(side):
            cells = tuple(sorted(set(_rot90_orbit(side, r, c))))
            if cells in seen:
                continue
            seen.add(cells)
            reps = [p for p in cells if p[0] <= p[1] and p[0] + p[1] <= side - 1]
            rep = min(reps) if reps else min(cells)
            maxd = max(max(abs(rr - center), abs(cc - center)) for rr, cc in cells)
            dist = sum((rr - center) ** 2 + (cc - center) ** 2 for rr, cc in cells) / len(cells)
            edge = sum(rr in (0, side - 1) or cc in (0, side - 1) for rr, cc in cells)
            corner = sum(rr in (0, side - 1) and cc in (0, side - 1) for rr, cc in cells)
            orbits.append({"cells": cells, "rep": rep, "size": len(cells), "maxd": maxd, "dist": dist, "edge": edge, "corner": corner})
    return orbits


def _gen_design_layout(side: int) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Choose center, full, and partial orbits for one square."""
    orbits = _gen_orbits(side)
    center_orb = [orb for orb in orbits if orb["size"] == 1]
    size4_orbs = [orb for orb in orbits if orb["size"] == ORBIT_SIZE]

    configs = []
    min_fill = math.ceil(FILL_MIN_FRAC * side * side)
    max_fill = math.floor(FILL_MAX_FRAC * side * side)
    center_fill = 1 if center_orb else 0
    for n_foreground in range(1, len(size4_orbs) + 1):
        fill = ORBIT_SIZE * n_foreground + center_fill
        if min_fill <= fill <= max_fill:
            for n_partial in range(PARTIAL_ORBITS_MIN, min(PARTIAL_ORBITS_MAX, n_foreground - 1) + 1):
                configs.append((n_foreground, n_partial))
    if not configs:
        return None
    n_foreground, n_partial = random.choice(configs)
    n_full = n_foreground - n_partial

    inner_first = sorted(
        size4_orbs,
        key=lambda o: (o["maxd"], o["dist"] + random.random() * ORBIT_RANK_JITTER),
    )
    full_orbs = inner_first[:n_full]
    partial_pool = inner_first[n_full:]

    partial_orbs = []
    corner_orbs = [orb for orb in partial_pool if orb["corner"]]
    if corner_orbs:
        partial_orbs.append(corner_orbs[0])

    remaining = [orb for orb in partial_pool if orb not in partial_orbs]
    remaining = sorted(
        remaining,
        key=lambda o: (o["maxd"], o["edge"], o["dist"] + random.random() * ORBIT_RANK_JITTER),
        reverse=True,
    )
    partial_orbs.extend(remaining[: n_partial - len(partial_orbs)])
    if len(partial_orbs) != n_partial:
        return None
    return center_orb, full_orbs, partial_orbs


def _gen_choose_retained(side: int, full_orbs: list[dict], center_orb: list[dict], partial_orbs: list[dict]) -> list[tuple[int, int]]:
    """Choose one retained cell from each partial orbit, preferring spanning sectors."""
    base = []
    for orb in full_orbs:
        base.extend(orb["cells"])
    if center_orb:
        base.extend(center_orb[0]["cells"])

    preferred_sector = random.randrange(SECTOR_COUNT)
    choices = []
    for orb in partial_orbs:
        opts = []
        seen = set()
        for idx, pos in enumerate(_rot90_orbit(side, *orb["rep"])):
            if pos not in seen:
                seen.add(pos)
                opts.append((pos, 1 if idx == preferred_sector else 0))
        choices.append(opts)

    spanning = []
    for combo in itertools.product(*choices):
        pts = base + [p for p, _ in combo]
        rows, cols = zip(*pts)
        if (min(rows), min(cols), max(rows), max(cols)) == (0, 0, side - 1, side - 1):
            spanning.append((sum(score for _, score in combo), combo))

    if spanning:
        best = max(score for score, _ in spanning)
        pool = [combo for score, combo in spanning if score >= best - SPANNING_SCORE_TOLERANCE]
        return [p for p, _ in random.choice(pool)]

    retained = []
    for opts in choices:
        preferred = [p for p, score in opts if score]
        if preferred and random.random() < PREFERRED_RETAIN_PROB:
            retained.append(preferred[0])
        else:
            retained.append(random.choice(opts)[0])
    return retained


def _gen_render_patch_and_mask(side: int, center_orb: list[dict], full_orbs: list[dict], partial_orbs: list[dict], body_color: int, accent_color: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the completed patch and the subset mask kept in the input."""
    patch = np.zeros((side, side), dtype=int)
    n_accent = random.randint(ACCENT_ORBITS_MIN, min(ACCENT_ORBITS_MAX, len(partial_orbs)))
    accent_orbs = sorted(
        partial_orbs,
        key=lambda o: (o["corner"], o["maxd"], o["dist"]),
        reverse=True,
    )[:n_accent]
    accent_ids = {id(orb) for orb in accent_orbs}

    for orb in full_orbs + partial_orbs:
        color = accent_color if id(orb) in accent_ids else body_color
        for pos in orb["cells"]:
            patch[pos] = color
    if center_orb:
        patch[center_orb[0]["cells"][0]] = body_color

    retain = np.zeros((side, side), dtype=bool)
    for orb in full_orbs:
        for pos in orb["cells"]:
            retain[pos] = True
    if center_orb:
        retain[center_orb[0]["cells"][0]] = True
    for pos in _gen_choose_retained(side, full_orbs, center_orb, partial_orbs):
        retain[pos] = True
    return patch, retain


def _gen_place_patch(side: int, patch: np.ndarray, retain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place the square with the configured margin."""
    max_anchor = CANVAS_SIZE - PLACEMENT_MARGIN - side
    r0 = random.randint(PLACEMENT_MARGIN, max_anchor)
    c0 = random.randint(PLACEMENT_MARGIN, max_anchor)
    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    expected = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    inp[r0 : r0 + side, c0 : c0 + side] = np.where(retain, patch, 0)
    expected[r0 : r0 + side, c0 : c0 + side] = patch
    return inp, expected


def _gen_solver_accepts(inp: np.ndarray, expected: np.ndarray, body_color: int, accent_color: int) -> bool:
    """Check required colors and solver recovery."""
    if set(np.unique(inp)) - {BACKGROUND} != {body_color, accent_color}:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(SIDE_MIN, SIDE_MAX)
    body_color, accent_color = random.sample(FOREGROUND_COLORS, 2)

    layout = _gen_design_layout(side)
    if layout is None:
        return None
    center_orb, full_orbs, partial_orbs = layout

    patch, retain = _gen_render_patch_and_mask(side, center_orb, full_orbs, partial_orbs, body_color, accent_color)
    inp, expected = _gen_place_patch(side, patch, retain)
    if not _gen_solver_accepts(inp, expected, body_color, accent_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid e40b9e2f input")
