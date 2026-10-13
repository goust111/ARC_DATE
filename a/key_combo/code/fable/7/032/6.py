from __future__ import annotations

import itertools
import random

import numpy as np


Rect = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color (the sparse lattice canvas background)."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _solutions(grid: np.ndarray) -> list[tuple[tuple[int, int, int], Rect]]:
    """Find all (cycle, lattice-index rectangle) explanations of the input."""
    bg = _background(grid)
    coords = np.argwhere(grid != bg)  # np.argwhere is already in reading order.
    assert len(coords) >= 12, "expected a substantial periodic point lattice"

    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    assert len(rows) >= 4 and len(cols) >= 4
    if len(rows) > 1:
        assert len(set(map(int, np.diff(rows)))) == 1, "point rows must be periodic"
    if len(cols) > 1:
        assert len(set(map(int, np.diff(cols)))) == 1, "point columns must be periodic"

    colors = tuple(map(int, np.unique(grid[grid != bg])))
    assert len(colors) == 3, "the threading cycle must contain exactly three colors"
    row_index = {int(r): i for i, r in enumerate(rows)}
    col_index = {int(c): i for i, c in enumerate(cols)}
    lattice_coords = np.array(
        [(row_index[int(r)], col_index[int(c)]) for r, c in coords], dtype=int
    )

    found: list[tuple[tuple[int, int, int], Rect]] = []
    for cycle in itertools.permutations(colors):
        relation = np.full(len(coords), -1, dtype=int)
        for i, (r, c) in enumerate(coords):
            observed = int(grid[int(r), int(c)])
            if observed == cycle[i % 3]:
                relation[i] = 0
            elif observed == cycle[(i + 1) % 3]:
                relation[i] = 1
        if np.any(relation < 0) or not np.any(relation == 1) or not np.any(relation == 0):
            continue

        shifted = lattice_coords[relation == 1]
        sr0, sc0 = map(int, shifted.min(axis=0))
        sr1, sc1 = map(int, shifted.max(axis=0))
        # A valid rectangle must enclose the shifted bbox.  Vacant border sites
        # can otherwise make a larger rectangle possible, so enumerate them too.
        for r0 in range(sr0 + 1):
            for c0 in range(sc0 + 1):
                for r1 in range(sr1, len(rows)):
                    for c1 in range(sc1, len(cols)):
                        if r1 == r0 or c1 == c0:
                            continue
                        inside = (
                            (lattice_coords[:, 0] >= r0)
                            & (lattice_coords[:, 0] <= r1)
                            & (lattice_coords[:, 1] >= c0)
                            & (lattice_coords[:, 1] <= c1)
                        )
                        if np.array_equal(inside, relation == 1):
                            found.append((tuple(map(int, cycle)), (r0, c0, r1, c1)))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the one-step phase slip inside the uniquely identifiable rectangular lattice region.

    Input structure:
    - A background canvas contains singleton point sites on periodic rows and
      columns, with irregularly vacant sites left as background.
    - The present points use exactly three foreground colors.  Reading across
      the whole canvas, their colors form one continuous three-color cycle,
      except within one unmarked rectangular region.

    Rule:
    - Thread only the present points in global row-major order; a vacant site
      is skipped without consuming a beat and rows do not restart the cycle.
    - Find the unique rectangle whose present points are all one cycle step
      early while every present point outside it follows the global cycle.
      Recolor those points to their globally prescribed colors and preserve
      every vacancy and all other cells.

    Key insight:
    - Lattice coordinates do not encode phase: irregular vacancies change the
      beat of every later present point, so the two-dimensional pattern must be
      reconstructed as a single vacancy-skipping one-dimensional sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    explanations = _solutions(g)
    assert len(explanations) == 1, "the phase-slip rectangle and cycle must be unique"
    cycle, rect = explanations[0]

    bg = _background(g)
    coords = np.argwhere(g != bg)
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    row_index = {int(r): i for i, r in enumerate(rows)}
    col_index = {int(c): i for i, c in enumerate(cols)}
    r0, c0, r1, c1 = rect

    out = g.copy()
    changed = 0
    for beat, (r, c) in enumerate(coords):
        lr, lc = row_index[int(r)], col_index[int(c)]
        if r0 <= lr <= r1 and c0 <= lc <= c1:
            out[int(r), int(c)] = cycle[beat % 3]
            changed += 1
    assert changed >= 4
    return out


def _has_simple_coordinate_phase(
    present: np.ndarray, expected_phase: np.ndarray
) -> bool:
    """Whether an affine function of lattice (row,col) replaces threading."""
    coords = np.argwhere(present)
    phases = expected_phase[present]
    for a in range(3):
        for b in range(3):
            for k in range(3):
                guessed = (a * coords[:, 0] + b * coords[:, 1] + k) % 3
                if np.array_equal(guessed, phases):
                    return True
    return False


def _semantic_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    present: np.ndarray,
    rect: Rect,
    cycle: tuple[int, int, int],
    bg: int,
) -> bool:
    """Validate the stated semantics independently of a mere generator round-trip."""
    nr, nc = present.shape
    r0, c0, r1, c1 = rect
    region = np.zeros_like(present)
    region[r0 : r1 + 1, c0 : c1 + 1] = True
    defect_points = present & region
    vacancies = ~present

    # The rectangle is visibly constrained on all four sides, contains a lure
    # vacancy, and has ample changed and unchanged evidence.
    if defect_points.sum() < 6 or (vacancies & region).sum() < 1:
        return False
    if min(defect_points[r0].sum(), defect_points[r1].sum()) < 1:
        return False
    if min(defect_points[:, c0].sum(), defect_points[:, c1].sum()) < 1:
        return False
    if present.sum() - defect_points.sum() < 12:
        return False

    flat_sites = np.arange(nr * nc).reshape(nr, nc)
    beat_phase = np.full((nr, nc), -1, dtype=int)
    beat_phase[present] = np.arange(int(present.sum())) % 3
    # Vacancies must make site-coordinate counting wrong in all major portions.
    before = flat_sites < r0 * nc + c0
    middle = region
    after = flat_sites > r1 * nc + c1
    for portion in (before, middle, after):
        q = present & portion
        if not np.any(q) or not np.any(beat_phase[q] != flat_sites[q] % 3):
            return False
    if _has_simple_coordinate_phase(present, beat_phase):
        return False

    # Cross-row continuity must be evidenced rather than compatible with a
    # cycle reset on every row.
    first_phases = [int(beat_phase[r, np.flatnonzero(present[r])[0]]) for r in range(nr)]
    if len(set(first_phases)) < 3:
        return False

    # Exact semantic oracle: only present points in the known rectangle change;
    # vacancies stay background, and every corrected point follows threaded beat.
    if not np.array_equal(inp == expected, ~defect_points):
        return False
    if not np.all(inp[vacancies] == bg) or not np.all(expected[vacancies] == bg):
        return False
    coords = np.argwhere(present)
    for beat, (lr, lc) in enumerate(coords):
        if int(expected[int(lr), int(lc)]) != cycle[beat % 3]:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    nr = random.randint(6, 9)
    nc = random.randint(7, 11)
    row_step = random.choice((2, 3))
    col_step = random.choice((2, 3))
    used_h = (nr - 1) * row_step + 1
    used_w = (nc - 1) * col_step + 1
    if used_h > 28 or used_w > 28:
        return None
    top = random.randint(1, min(3, 30 - used_h - 1))
    left = random.randint(1, min(3, 30 - used_w - 1))
    height = used_h + top + random.randint(1, min(3, 30 - used_h - top))
    width = used_w + left + random.randint(1, min(3, 30 - used_w - left))

    bg = 0
    cycle = tuple(random.sample(range(1, 10), 3))
    vacancy_rate = random.uniform(0.16, 0.30)
    present = np.random.default_rng().random((nr, nc)) >= vacancy_rate
    if np.any(present.sum(axis=0) == 0) or np.any(present.sum(axis=1) == 0):
        return None
    if (~present).sum() < 7:
        return None

    r0 = random.randint(1, nr - 4)
    r1 = random.randint(r0 + 1, min(nr - 2, r0 + 3))
    c0 = random.randint(1, nc - 4)
    c1 = random.randint(c0 + 1, min(nc - 2, c0 + 4))
    rect = (r0, c0, r1, c1)

    lattice_expected = np.full((nr, nc), bg, dtype=int)
    lattice_input = np.full((nr, nc), bg, dtype=int)
    beat = 0
    for lr in range(nr):
        for lc in range(nc):
            if not present[lr, lc]:
                continue
            lattice_expected[lr, lc] = cycle[beat % 3]
            shifted = r0 <= lr <= r1 and c0 <= lc <= c1
            lattice_input[lr, lc] = cycle[(beat + int(shifted)) % 3]
            beat += 1

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    for lr in range(nr):
        for lc in range(nc):
            rr, cc = top + lr * row_step, left + lc * col_step
            inp[rr, cc] = lattice_input[lr, lc]
            expected[rr, cc] = lattice_expected[lr, lc]

    if not _semantic_valid(lattice_input, lattice_expected, present, rect, cycle, bg):
        return None
    explanations = _solutions(inp)
    if explanations != [(cycle, rect)]:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse, oracle-checked lattices with non-degenerate vacancies."""
    for _ in range(1000):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely decodable phase-slip lattice")
