import random

import numpy as np


MASK = 6
ORANGE = 7
BASE_COLORS = (1, 2, 3, 4)
GRID_SIZES = (9, 11, 13)
GENERATION_ATTEMPTS = 500


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    s = n - 1
    pts = {
        (r, c),
        (c, r),
        (s - r, c),
        (r, s - c),
        (s - r, s - c),
        (s - c, s - r),
        (c, s - r),
        (s - c, r),
    }
    return tuple(sorted(pts))


def _all_orbits(n: int) -> tuple[list[tuple[tuple[int, int], ...]], dict[tuple[int, int], int]]:
    seen: dict[tuple[tuple[int, int], ...], int] = {}
    orbits: list[tuple[tuple[int, int], ...]] = []
    orbit_id: dict[tuple[int, int], int] = {}
    for r in range(n):
        for c in range(n):
            orb = _orbit(n, r, c)
            if orb not in seen:
                seen[orb] = len(orbits)
                orbits.append(orb)
            orbit_id[(r, c)] = seen[orb]
    return orbits, orbit_id


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = _sign(r1 - r0)
    dc = _sign(c1 - c0)
    assert abs(r1 - r0) == abs(c1 - c0) and dr != 0 and dc != 0
    dist = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(dist + 1)]


def _line_kind(markers: np.ndarray) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in markers]
    assert abs(r1 - r0) == abs(c1 - c0) and abs(r1 - r0) >= 2
    if (r1 - r0) * (c1 - c0) > 0:
        return _segment_cells((r0, c0), (r1, c1))
    return _segment_cells((r0, c0), (r1, c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill masked cells by first drawing the marker diagonal segment and then propagating colors across D4 symmetry orbits.

    Input structure:
    - The input is an odd-sized square grid.
    - Non-pink, non-orange cells form a D4-symmetric multicolor pattern around the unique center.
    - Exactly two orange marker pixels lie on one 45-degree diagonal, while pink cells mark missing values.

    Rule:
    - Find the two orange markers and traverse every cell on the diagonal segment between them.
    - On that segment, keep existing non-pink colors unchanged, but replace pink cells with orange.
    - For every remaining pink cell, collect its full D4 orbit under horizontal, vertical, diagonal, and rotational symmetries of the square.
    - Replace the pink cell with the unique visible non-pink color already present in that orbit; orange anchors propagate orange as well.

    Key insight:
    - The diagonal is not decoration: it reveals hidden orange anchor pixels whose symmetry orbits then determine remote masked cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]
    n = g.shape[0]
    assert n % 2 == 1

    markers = np.argwhere(g == ORANGE)
    assert len(markers) == 2

    out = g.copy()
    for r, c in _line_kind(markers):
        if out[r, c] == MASK:
            out[r, c] = ORANGE

    for r, c in zip(*np.where(out == MASK)):
        colors = {int(out[rr, cc]) for rr, cc in _orbit(n, int(r), int(c)) if out[rr, cc] != MASK}
        assert len(colors) == 1
        out[r, c] = colors.pop()

    return out


def _diag_families(n: int) -> list[list[tuple[int, int]]]:
    families: list[list[tuple[int, int]]] = []
    for d in range(-(n - 1), n):
        cells = [(r, r + d) for r in range(n) if 0 <= r + d < n]
        if len(cells) >= 5:
            families.append(cells)
    for s in range(0, 2 * n - 1):
        cells = [(r, s - r) for r in range(n) if 0 <= s - r < n]
        if len(cells) >= 5:
            families.append(cells)
    return families


def _feature_pair(n: int, r: int, c: int) -> tuple[int, int]:
    ctr = n // 2
    a, b = abs(r - ctr), abs(c - ctr)
    return tuple(sorted((a, b)))


def _base_pattern(n: int, orbits: list[tuple[tuple[int, int], ...]], palette: tuple[int, ...]) -> np.ndarray:
    k = len(palette)
    formulas = [
        lambda a, b, s: a + b + s,
        lambda a, b, s: b + s,
        lambda a, b, s: a + 2 * b + s,
        lambda a, b, s: a * b + a + b + s,
        lambda a, b, s: 2 * (a % 2) + b + s,
    ]
    for _ in range(60):
        shift = random.randrange(k)
        formula = random.choice(formulas)
        grid = np.zeros((n, n), dtype=int)
        used = set()
        for orb in orbits:
            a, b = _feature_pair(n, *orb[0])
            color = palette[formula(a, b, shift) % k]
            used.add(color)
            for r, c in orb:
                grid[r, c] = color
        if len(used) == k:
            return grid

    grid = np.zeros((n, n), dtype=int)
    for i, orb in enumerate(orbits):
        color = palette[i % k]
        for r, c in orb:
            grid[r, c] = color
    return grid


def _choose_segment(
    n: int,
    orbits: list[tuple[tuple[int, int], ...]],
    orbit_id: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], set[int]]:
    families = _diag_families(n)
    random.shuffle(families)

    for cells in families:
        pairs = [(i, j) for i in range(len(cells) - 3) for j in range(i + 3, len(cells))]
        random.shuffle(pairs)
        for i, j in pairs:
            segment = cells[i : j + 1]
            segset = set(segment)
            endpoint_ids = {orbit_id[segment[0]], orbit_id[segment[-1]]}

            segment_ids: list[int] = []
            expandable_ids: list[int] = []
            internal_anchor_ids: list[int] = []
            for idx, cell in enumerate(segment):
                oid = orbit_id[cell]
                if oid not in segment_ids:
                    segment_ids.append(oid)
                if any(p not in segset for p in orbits[oid]):
                    if oid not in expandable_ids:
                        expandable_ids.append(oid)
                    if 0 < idx < len(segment) - 1 and oid not in endpoint_ids and oid not in internal_anchor_ids:
                        internal_anchor_ids.append(oid)

            if not internal_anchor_ids:
                continue

            orange_ids = set(endpoint_ids)
            orange_ids.add(random.choice(internal_anchor_ids))

            extras = [oid for oid in expandable_ids if oid not in orange_ids]
            random.shuffle(extras)
            for oid in extras:
                if random.random() < 0.35:
                    trial = orange_ids | {oid}
                    if any(orbit_id[cell] not in trial for cell in segment[1:-1]):
                        orange_ids = trial

            if not any(orbit_id[cell] in orange_ids for cell in segment[1:-1]):
                continue
            if not any(orbit_id[cell] not in orange_ids for cell in segment[1:-1]):
                continue

            masked_off_line = sum(1 for oid in orange_ids for p in orbits[oid] if p not in segset)
            if masked_off_line == 0:
                continue

            return segment, orange_ids

    raise AssertionError("failed to choose a valid diagonal segment")


def _render_input() -> tuple[np.ndarray, np.ndarray]:
    n = random.choice(GRID_SIZES)
    orbits, orbit_id = _all_orbits(n)
    palette = tuple(random.sample(BASE_COLORS, random.choice((2, 3))))
    out = _base_pattern(n, orbits, palette)

    segment, orange_ids = _choose_segment(n, orbits, orbit_id)
    for oid in orange_ids:
        for r, c in orbits[oid]:
            out[r, c] = ORANGE

    inp = out.copy()
    endpoints = {segment[0], segment[-1]}
    for oid in orange_ids:
        for r, c in orbits[oid]:
            inp[r, c] = MASK
    for r, c in endpoints:
        inp[r, c] = ORANGE

    assert np.count_nonzero(inp == ORANGE) == 2
    assert np.any(inp == MASK)
    assert any(inp[r, c] == MASK for r, c in segment[1:-1])
    assert any(inp[r, c] not in (MASK, ORANGE) for r, c in segment[1:-1])
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp, out = _render_input()
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
