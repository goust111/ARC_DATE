from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
BAR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _full_cross(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        for r in rows:
            for c in cols:
                if 0 < r < h - 1 and 0 < c < w - 1:
                    candidates.append((int(r), int(c), int(color)))
    assert len(candidates) == 1, "expected one internal full-row/full-column cross"
    return candidates[0]


def _unique_most_common(values: np.ndarray, role: str) -> int:
    counts = Counter(map(int, values.ravel()))
    assert counts, f"expected at least one {role} color"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, f"expected a unique {role} color"
    return int(winners[0])


def _marker_constraints(
    g: np.ndarray, center: tuple[int, int], bg: int, bar: int
) -> dict[int, int]:
    cr, cc = center
    dist_to_color: dict[int, int] = {}
    for r, c in np.argwhere((g != int(bg)) & (g != int(bar))):
        d = max(abs(int(r) - cr), abs(int(c) - cc))
        assert d > 0, "the center is occupied by the cross, not a marker"
        color = int(g[int(r), int(c)])
        if d in dist_to_color:
            assert dist_to_color[d] == color, "same-distance markers must agree"
        else:
            dist_to_color[d] = color
    assert dist_to_color, "expected colored marker pixels"
    return dist_to_color


def _infer_period(dist_to_color: dict[int, int]) -> tuple[int, dict[int, int]]:
    max_sampled = max(dist_to_color)
    for period in range(1, max_sampled + 1):
        residues: dict[int, int] = {}
        ok = True
        for d, color in dist_to_color.items():
            residue = int(d % period)
            if residue in residues and residues[residue] != int(color):
                ok = False
                break
            residues[residue] = int(color)
        if ok and len(residues) == period:
            return int(period), residues
    raise AssertionError("markers do not define a complete consistent period")


def _ring_grid(shape: tuple[int, int], center: tuple[int, int], colors_by_residue: dict[int, int]) -> np.ndarray:
    period = len(colors_by_residue)
    assert period > 0 and set(colors_by_residue) == set(range(period))
    palette = np.array([int(colors_by_residue[i]) for i in range(period)], dtype=int)
    rr, cc = np.indices(shape)
    cr, c0 = center
    distances = np.maximum(np.abs(rr - int(cr)), np.abs(cc - int(c0)))
    return palette[distances % period]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the periodic Chebyshev-distance target rings from sparse marker pixels.

    Input structure:
    - One full row bar and one full column bar of the same color cross at a single internal center.
    - The remaining most common color is background.
    - Sparse colored marker pixels sit off the bars; markers at the same Chebyshev distance from the center agree in color.

    Rule:
    - Use the full-row/full-column cross to locate the center.
    - Convert each marker into a constraint on the whole ring at that Chebyshev distance.
    - Find the smallest period whose distance residues are complete and color-consistent, then color every cell by distance modulo that period.

    Key insight:
    - The bars only point to the center; the full image is the hidden periodic sequence of ring colors, not a protected cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    cr, cc, bar = _full_cross(g)
    non_bar = g[g != int(bar)]
    bg = _unique_most_common(non_bar, "background")
    dist_to_color = _marker_constraints(g, (cr, cc), bg, bar)
    _period, colors_by_residue = _infer_period(dist_to_color)
    return _ring_grid(g.shape, (cr, cc), colors_by_residue)


def _max_distance(shape: tuple[int, int], center: tuple[int, int]) -> int:
    h, w = shape
    cr, cc = center
    return max(cr, h - 1 - cr, cc, w - 1 - cc)


def _ring_positions(shape: tuple[int, int], center: tuple[int, int], distance: int) -> list[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    d = int(distance)
    coords = []
    for r in range(max(0, cr - d), min(h - 1, cr + d) + 1):
        for c in range(max(0, cc - d), min(w - 1, cc + d) + 1):
            if r == cr or c == cc:
                continue
            if max(abs(r - cr), abs(c - cc)) == d:
                coords.append((int(r), int(c)))
    random.shuffle(coords)
    return coords


def _spaced_choice(options: list[tuple[int, int]], taken: set[tuple[int, int]]) -> tuple[int, int] | None:
    spaced = [
        (r, c)
        for r, c in options
        if (r, c) not in taken
        and all(max(abs(r - pr), abs(c - pc)) >= 2 for pr, pc in taken)
    ]
    if spaced:
        return random.choice(spaced)
    free = [(r, c) for r, c in options if (r, c) not in taken]
    if free:
        return random.choice(free)
    return None


def _sample_distances(shape: tuple[int, int], center: tuple[int, int], period: int) -> dict[int, list[int]]:
    max_d = _max_distance(shape, center)
    by_residue: dict[int, list[int]] = {}
    for residue in range(period):
        possible = [
            d
            for d in range(1, max_d + 1)
            if d % period == residue and _ring_positions(shape, center, d)
        ]
        if len(possible) < 2:
            return {}

        first = period if residue == 0 else residue
        if first not in possible:
            first = possible[0]
        second = first + period if first + period in possible else possible[1]
        chosen = [first, second]

        extras = [d for d in possible if d not in chosen]
        if extras and random.random() < 0.35:
            chosen.append(random.choice(extras))
        chosen.sort()
        by_residue[residue] = chosen
    return by_residue


def _place_markers(
    inp: np.ndarray,
    center: tuple[int, int],
    colors_by_residue: dict[int, int],
    distances_by_residue: dict[int, list[int]],
) -> bool:
    taken: set[tuple[int, int]] = set()
    for residue, distances in distances_by_residue.items():
        for distance in distances:
            options = _ring_positions(inp.shape, center, distance)
            pos = _spaced_choice(options, taken)
            if pos is None:
                return False
            r, c = pos
            inp[r, c] = int(colors_by_residue[residue])
            taken.add(pos)

            if random.random() < 0.25:
                extra = _spaced_choice(options, taken)
                if extra is not None:
                    er, ec = extra
                    inp[er, ec] = int(colors_by_residue[residue])
                    taken.add(extra)
    return True


def _gen_semantics_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    center: tuple[int, int],
    period: int,
    colors_by_residue: dict[int, int],
    distances_by_residue: dict[int, list[int]],
) -> bool:
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if len(set(colors_by_residue.values())) != period:
        return False
    if np.any(expected == BG) or np.any(expected == BAR):
        return False

    sampled = {d for distances in distances_by_residue.values() for d in distances}
    max_d = _max_distance(inp.shape, center)
    if not sampled or len(sampled) >= max_d:
        return False
    if not any(d not in sampled for d in range(1, max_d + 1)):
        return False
    if any(len(distances_by_residue.get(residue, ())) < 2 for residue in range(period)):
        return False

    try:
        cr, cc, bar = _full_cross(inp)
        if (cr, cc) != center or bar != BAR:
            return False
        bg = _unique_most_common(inp[inp != BAR], "background")
        if bg != BG:
            return False
        constraints = _marker_constraints(inp, center, BG, BAR)
        inferred, residues = _infer_period(constraints)
        if inferred != period or residues != colors_by_residue:
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    period = random.randint(2, 5)
    h = random.randint(max(14, 2 * period + 5), 30)
    w = random.randint(max(14, 2 * period + 5), 30)

    valid_centers = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _max_distance((h, w), (r, c)) >= 2 * period + 2
    ]
    if not valid_centers:
        return None
    center = random.choice(valid_centers)

    sequence = random.sample(COLOR_POOL, period)
    colors_by_residue = {i: int(sequence[i]) for i in range(period)}
    expected = _ring_grid((h, w), center, colors_by_residue)

    inp = np.full((h, w), BG, dtype=int)
    cr, cc = center
    inp[cr, :] = BAR
    inp[:, cc] = BAR

    distances_by_residue = _sample_distances(inp.shape, center, period)
    if set(distances_by_residue) != set(range(period)):
        return None
    if not _place_markers(inp, center, colors_by_residue, distances_by_residue):
        return None
    if not _gen_semantics_ok(inp, expected, center, period, colors_by_residue, distances_by_residue):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic target-ring input")
