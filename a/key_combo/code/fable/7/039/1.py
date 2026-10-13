"""ARC task: reveal a sprite by detecting and folding D4-symmetry defects."""

from collections import Counter

import numpy as np


def _orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    """Return the distinct D4 orbit of ``(r, c)`` in an n-by-n grid."""
    z = n - 1
    return tuple(sorted({
        (r, c), (c, r),
        (r, z - c), (z - r, c), (z - r, z - c),
        (c, z - r), (z - c, r), (z - c, z - r),
    }))


def _canonical(n: int, r: int, c: int) -> tuple[int, int]:
    """Fold through the two midlines, then above the main diagonal."""
    a = min(r, n - 1 - r)
    b = min(c, n - 1 - c)
    return min(a, b), max(a, b)


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate each square-symmetry orbit once, in canonical order."""
    result = []
    for r in range((n + 1) // 2):
        for c in range(r, (n + 1) // 2):
            orbit = _orbit(n, r, c)
            if len(orbit) > 1 or (r, c) == (n // 2, n // 2):
                result.append(orbit)
    return result


def _deviations(grid: np.ndarray) -> list[tuple[int, int, int, tuple[int, int]]]:
    """Find (row, col, deviant color, canonical coordinate) records."""
    n = int(grid.shape[0])
    deviations = []
    for orbit in _all_orbits(n):
        values = [int(grid[r, c]) for r, c in orbit]
        counts = Counter(values)
        ranked = counts.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], (
            "each orbit must have a unique majority color"
        )
        normal = ranked[0][0]
        odd = [(r, c) for r, c in orbit if int(grid[r, c]) != normal]
        assert len(odd) <= 1, "each orbit may contain at most one deviant pixel"
        if odd:
            r, c = odd[0]
            deviations.append((r, c, int(grid[r, c]), _canonical(n, r, c)))
    return deviations


def main(grid: np.ndarray) -> np.ndarray:
    """Detect D4-symmetry defects, fold them into the upper-left wedge, and crop the revealed sprite.

    Input structure:
    - The input is a square, densely multicolored texture that is constant on
      almost every orbit of the square's four rotations and four reflections.
    - A used orbit has exactly one deviant pixel; its color is also common in
      normal texture, and each orbit has at most one such deviation.

    Rule:
    - Take the unique majority color independently within every D4 orbit and
      collect the cells that differ from it.
    - Fold each collected cell across the horizontal and vertical midlines into
      the upper-left quadrant, then across the main diagonal to row <= column.
      Paint its original color at that canonical position on black.
    - Return the inclusive minimum bounding-box crop of the painted positions.

    Key insight:
    - Orbit voting reveals which pixels are anomalous, while the same symmetry
      supplies the folding map that reassembles their scattered sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    n, w = map(int, g.shape)
    assert n == w, "expected a square grid"

    deviations = _deviations(g)
    assert deviations, "expected at least one deviant pixel"

    folded: dict[tuple[int, int], int] = {}
    for _r, _c, color, target in deviations:
        assert target not in folded, "distinct anomaly orbits must fold separately"
        folded[target] = color

    rows = [r for r, _c in folded]
    cols = [c for _r, c in folded]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in folded.items():
        out[r - r0, c - c0] = color
    return out


def _hidden_shape(rng: np.random.Generator, n: int) -> set[tuple[int, int]] | None:
    """Make a connected, holey sprite wholly inside the fundamental wedge."""
    m = (n + 1) // 2
    h = int(rng.integers(2, min(4, m - 2) + 1))
    w = int(rng.integers(3, min(6, m) + 1))

    # Sometimes touch the diagonal, top edge, or outer quadrant boundary.
    diagonal = bool(rng.integers(0, 2))
    r0 = 0 if rng.random() < 0.35 else int(rng.integers(0, max(1, m - h)))
    min_c0 = r0 + h - 1
    if min_c0 + w > m:
        return None
    if diagonal:
        c0 = min_c0
    elif rng.random() < 0.35:
        c0 = m - w
        if c0 < min_c0:
            return None
    else:
        c0 = int(rng.integers(min_c0, m - w + 1))

    # A random monotone backbone guarantees connectivity and all four bbox sides.
    lr = lc = 0
    local = {(0, 0)}
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    for step in steps:
        if step == 0:
            lr += 1
        else:
            lc += 1
        local.add((lr, lc))

    for rr in range(h):
        for cc in range(w):
            if rng.random() < 0.28:
                local.add((rr, cc))

    if diagonal:
        # Join the wedge-diagonal cell to the backbone along the bottom row.
        bottom_cols = [cc for rr, cc in local if rr == h - 1]
        for cc in range(0, min(bottom_cols) + 1):
            local.add((h - 1, cc))

    # Keep at least one black cell inside the eventual crop, so bbox/crop is visible.
    if len(local) >= h * w:
        removable = [p for p in local if p not in {(0, 0), (h - 1, w - 1)}]
        if removable:
            local.remove(removable[int(rng.integers(0, len(removable)))])

    shape = {(r0 + rr, c0 + cc) for rr, cc in local}
    center = (n // 2, n // 2)
    if n % 2 == 1 and center in shape:  # the singleton orbit cannot have a minority
        shape.remove(center)
    if len(shape) < 5:
        return None
    return shape


def _render_expected(hidden: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _c in hidden]
    cols = [c for _r, c in hidden]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in hidden.items():
        out[r - r0, c - c0] = color
    return out


def _validate_generated(
    inp: np.ndarray,
    pristine: np.ndarray,
    hidden: dict[tuple[int, int], int],
) -> bool:
    """Check the stated semantics and reject visually or logically degenerate cases."""
    n = int(inp.shape[0])
    changed = [tuple(map(int, rc)) for rc in np.argwhere(inp != pristine)]
    if len(changed) != len(hidden):
        return False

    # Every orbit is symmetric before injection and has no more than one defect.
    for orbit in _all_orbits(n):
        if len({int(pristine[r, c]) for r, c in orbit}) != 1:
            return False
        if sum(inp[r, c] != pristine[r, c] for r, c in orbit) > 1:
            return False

    records = _deviations(inp)
    recovered = {target: color for _r, _c, color, target in records}
    if recovered != hidden:
        return False

    expected = _render_expected(hidden)
    if not np.array_equal(main(inp), expected):
        return False
    if min(expected.shape) < 2 or not np.any(expected == 0):
        return False

    # Folding must materially move and compact the scattered anomaly positions.
    raw_rows, raw_cols = zip(*changed)
    raw_area = (max(raw_rows) - min(raw_rows) + 1) * (max(raw_cols) - min(raw_cols) + 1)
    if any(pos in hidden for pos in changed) or raw_area <= 2 * expected.size:
        return False
    quadrants = {(r >= n / 2, c >= n / 2) for r, c in changed}
    if len(quadrants) < 3:
        return False

    # No anomaly-only color: each deviant color occurs many times normally.
    normal_counts = Counter(map(int, pristine.ravel()))
    if any(normal_counts[color] < 8 for color in hidden.values()):
        return False
    return expected.shape[0] < (n + 1) // 2 and expected.shape[1] < (n + 1) // 2


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(11, 26))
    shape = _hidden_shape(rng, n)
    if shape is None:
        return None

    palette_size = int(rng.integers(3, 6))
    palette = [int(x) for x in rng.choice(np.arange(1, 10), palette_size, replace=False)]
    orbits = _all_orbits(n)
    assignments = (palette * ((len(orbits) + palette_size - 1) // palette_size))[:len(orbits)]
    rng.shuffle(assignments)

    pristine = np.empty((n, n), dtype=int)
    orbit_by_canonical = {}
    for orbit, color in zip(orbits, assignments):
        canonical = _canonical(n, *orbit[0])
        orbit_by_canonical[canonical] = orbit
        for r, c in orbit:
            pristine[r, c] = color

    # Odd grids have a singleton center; it is included and initialized above.
    assert np.all((pristine >= 1) & (pristine <= 9))

    hidden = {}
    for target in shape:
        orbit = orbit_by_canonical[target]
        base = int(pristine[orbit[0]])
        choices = [color for color in palette if color != base]
        hidden[target] = int(choices[int(rng.integers(0, len(choices)))])
    if len(set(hidden.values())) < 2:
        return None

    # Scatter to arbitrary orbit members, but require distinct broad regions and
    # never leave a defect already at its canonical location.
    inp = pristine.copy()
    for target, color in hidden.items():
        candidates = [pos for pos in orbit_by_canonical[target] if pos != target]
        pos = candidates[int(rng.integers(0, len(candidates)))]
        inp[pos] = color

    if not _validate_generated(inp, pristine, hidden):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse valid near-eightfold-symmetric texture."""
    rng = np.random.default_rng()
    for _ in range(500):
        result = _generate_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid symmetry-fold input")
