from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _dihedral(a: np.ndarray, index: int) -> np.ndarray:
    """One of the eight square-grid symmetries, in a fixed visible order."""
    k, reflected = divmod(int(index), 2)
    turned = np.rot90(a, k)
    return np.fliplr(turned) if reflected else turned.copy()


def _inverse_index(index: int) -> int:
    """Return the index undoing a dihedral transform."""
    probe = np.arange(25, dtype=int).reshape(5, 5)
    transformed = _dihedral(probe, index)
    matches = [j for j in range(8) if np.array_equal(_dihedral(transformed, j), probe)]
    assert len(matches) == 1
    return int(matches[0])


def _split_candidates(g: np.ndarray) -> list[tuple[str, int, int, np.ndarray, np.ndarray]]:
    """Find full monochrome framework lines which make two equal square regions."""
    h, w = g.shape
    candidates: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []

    for r in range(h):
        if np.all(g[r, :] == g[r, 0]) and r == h - r - 1 and w == r:
            candidates.append(("row", r, int(g[r, 0]), g[:r, :], g[r + 1 :, :]))

    for c in range(w):
        if np.all(g[:, c] == g[0, c]) and c == w - c - 1 and h == c:
            candidates.append(("col", c, int(g[0, c]), g[:, :c], g[:, c + 1 :]))

    return candidates


def _period_vote(a: np.ndarray, b_aligned: np.ndarray, period: int):
    """Recover a tile and its consensus strength from two aligned regions."""
    n = int(a.shape[0])
    tile = np.empty((period, period), dtype=int)
    support = 0
    weakest_num = 1
    weakest_den = 1

    for r in range(period):
        for c in range(period):
            values = np.concatenate(
                [a[r:n:period, c:n:period].ravel(), b_aligned[r:n:period, c:n:period].ravel()]
            )
            counts = Counter(map(int, values))
            ordered = counts.most_common()
            if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
                return None
            color, count = ordered[0]
            # Every residue must have a visibly strong cross-region consensus.
            if 3 * int(count) < 2 * int(values.size):
                return None
            if int(count) * weakest_den < weakest_num * int(values.size):
                weakest_num, weakest_den = int(count), int(values.size)
            tile[r, c] = int(color)
            support += int(count)

    return tile, int(support), int(weakest_num), int(weakest_den)


def _solution_candidates(g: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for axis, divider, frame, a, b in _split_candidates(g):
        n = int(a.shape[0])
        for period in range(2, 5):
            if n % period != 0 or n // period < 3:
                continue
            for transform in range(8):
                b_aligned = _dihedral(b, transform)
                voted = _period_vote(a, b_aligned, period)
                if voted is None:
                    continue
                tile, support, weakest_num, weakest_den = voted
                candidates.append(
                    {
                        "axis": axis,
                        "divider": divider,
                        "frame": frame,
                        "a": a,
                        "b": b,
                        "period": period,
                        "transform": transform,
                        "tile": tile,
                        "support": support,
                        "weakest_num": weakest_num,
                        "weakest_den": weakest_den,
                    }
                )
    return candidates


def _choose_solution(g: np.ndarray) -> dict:
    candidates = _solution_candidates(g)
    assert candidates, "expected a framework, a small period, and a cross-region alignment"

    # A fundamental period is the smallest valid period.  At that period the
    # alignment with the most agreeing visible cells must be unique.
    min_period = min(int(c["period"]) for c in candidates)
    candidates = [c for c in candidates if int(c["period"]) == min_period]
    best_support = max(int(c["support"]) for c in candidates)
    best = [c for c in candidates if int(c["support"]) == best_support]
    assert len(best) == 1, "period and dihedral alignment must be uniquely determined"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align two transformed periodic backups and repair every corrupted tile copy.

    Input structure:
    - One monochrome full row or column divides the grid into two equal square regions.
    - Each region is a noisy periodic tiling of the same small square tile, but the
      second region has a different rotation or reflection and corruptions occur at
      different cells in the two regions.

    Rule:
    - Find the dihedral transform that aligns the second region with the first and
      infer their smallest shared period.  For every tile residue, combine all
      aligned copies from both regions and take its unique strongly supported color.
    - Retile the repaired tile in the first orientation, undo the alignment for the
      second region, and leave the separating framework unchanged.

    Key insight:
    - Each region has a residue where its own majority is corrupted; only orientation
      alignment followed by cross-region voting lets the two backups repair each other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    solution = _choose_solution(g)
    period = int(solution["period"])
    n = int(solution["a"].shape[0])
    clean_a = np.tile(solution["tile"], (n // period, n // period))
    clean_b = _dihedral(clean_a, _inverse_index(int(solution["transform"])))

    out = g.copy()
    divider = int(solution["divider"])
    if solution["axis"] == "row":
        out[:divider, :] = clean_a
        out[divider + 1 :, :] = clean_b
    else:
        out[:, :divider] = clean_a
        out[:, divider + 1 :] = clean_b
    return out


# ------------------------------ generator ------------------------------

_GEN_ATTEMPTS = 500
_PERIOD_REPEATS = {2: (4, 5, 6, 7), 3: (3, 4), 4: (3,)}


def _random_asymmetric_tile(period: int, palette: list[int]) -> np.ndarray | None:
    tile = np.random.choice(palette, size=(period, period)).astype(int)
    # All selected colors must be visibly relevant, and orientation must matter.
    if set(map(int, tile.ravel())) != set(palette):
        return None
    variants = {_dihedral(tile, i).tobytes() for i in range(8)}
    if len(variants) != 8:
        return None
    return tile


def _occurrences(n: int, period: int, residue: tuple[int, int]) -> list[tuple[int, int]]:
    rr, cc = residue
    return [(r, c) for r in range(rr, n, period) for c in range(cc, n, period)]


def _wrong_color(clean: int, palette: list[int]) -> int:
    return int(random.choice([color for color in palette if int(color) != int(clean)]))


def _single_region_tile(region: np.ndarray, period: int) -> np.ndarray:
    """Ordinary per-region majority, used only to reject semantic shortcuts."""
    n = int(region.shape[0])
    tile = np.empty((period, period), dtype=int)
    for r in range(period):
        for c in range(period):
            counts = Counter(map(int, region[r:n:period, c:n:period].ravel()))
            ordered = counts.most_common()
            if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
                return np.full((period, period), -1, dtype=int)
            tile[r, c] = int(ordered[0][0])
    return tile


def _assemble(a: np.ndarray, b: np.ndarray, frame: int, axis: str) -> np.ndarray:
    n = int(a.shape[0])
    if axis == "row":
        return np.concatenate([a, np.full((1, n), frame, dtype=int), b], axis=0)
    return np.concatenate([a, np.full((n, 1), frame, dtype=int), b], axis=1)


def _gen_attempt(
    period_choice: int | None = None,
    axis_choice: str | None = None,
    alignment_choice: int | None = None,
) -> np.ndarray | None:
    period = int(period_choice) if period_choice is not None else random.choice([2, 3, 4])
    repeats = random.choice(_PERIOD_REPEATS[period])
    n = period * repeats
    axis = str(axis_choice) if axis_choice is not None else random.choice(["row", "col"])

    all_colors = list(range(10))
    random.shuffle(all_colors)
    color_count = min(period * period, random.choice([4, 5]))
    palette = [int(c) for c in all_colors[:color_count]]
    frame = int(all_colors[color_count])
    tile = _random_asymmetric_tile(period, palette)
    if tile is None:
        return None

    clean_a = np.tile(tile, (repeats, repeats))
    # The two visible orientations always differ.
    alignment = int(alignment_choice) if alignment_choice is not None else random.randrange(1, 8)
    inverse = _inverse_index(alignment)
    clean_b = _dihedral(clean_a, inverse)

    # Work in A's coordinate system while choosing disjoint corruption sites.
    a_noisy = clean_a.copy()
    b_aligned_noisy = clean_a.copy()
    residues = [(r, c) for r in range(period) for c in range(period)]
    critical_a, critical_b = random.sample(residues, 2)
    copies = repeats * repeats
    majority_corruptions = copies // 2 + 1
    mask_a: set[tuple[int, int]] = set()
    mask_b: set[tuple[int, int]] = set()

    cells_a = random.sample(_occurrences(n, period, critical_a), majority_corruptions)
    bad_a = _wrong_color(int(tile[critical_a]), palette)
    for pos in cells_a:
        a_noisy[pos] = bad_a
        mask_a.add(pos)

    cells_b = random.sample(_occurrences(n, period, critical_b), majority_corruptions)
    bad_b = _wrong_color(int(tile[critical_b]), palette)
    for pos in cells_b:
        b_aligned_noisy[pos] = bad_b
        mask_b.add(pos)

    # Sparse extra defects cover non-critical residues without ever masking the
    # same aligned position in both backups.
    extra_count = random.randint(2, 5)
    positions = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(positions)
    for pos in positions:
        if extra_count <= 0:
            break
        target_a = bool(random.getrandbits(1))
        target_mask, other_mask = (mask_a, mask_b) if target_a else (mask_b, mask_a)
        target = a_noisy if target_a else b_aligned_noisy
        if pos in target_mask or pos in other_mask:
            continue
        residue = (pos[0] % period, pos[1] % period)
        if residue in (critical_a, critical_b):
            continue
        target[pos] = _wrong_color(int(clean_a[pos]), palette)
        target_mask.add(pos)
        extra_count -= 1
    if extra_count:
        return None

    b_noisy = _dihedral(b_aligned_noisy, inverse)
    inp = _assemble(a_noisy, b_noisy, frame, axis)
    expected = _assemble(clean_a, clean_b, frame, axis)

    # Structural clarity: the divider must be the only constant full line.
    if len(_split_candidates(inp)) != 1:
        return None

    # Both independent majority shortcuts must fail, at different residues.
    vote_a = _single_region_tile(a_noisy, period)
    vote_b = _single_region_tile(b_aligned_noisy, period)
    if np.array_equal(vote_a, tile) or np.array_equal(vote_b, tile):
        return None
    wrong_a = set(map(tuple, np.argwhere(vote_a != tile)))
    wrong_b = set(map(tuple, np.argwhere(vote_b != tile)))
    if critical_a not in wrong_a or critical_b not in wrong_b or wrong_a == wrong_b:
        return None

    # Alignment and cross-region combination must each affect the result.
    unaligned_vote = _period_vote(a_noisy, b_noisy, period)
    if unaligned_vote is not None and np.array_equal(unaligned_vote[0], tile):
        return None

    try:
        solution = _choose_solution(inp)
        actual = main(inp)
    except (AssertionError, ValueError):
        return None
    if int(solution["period"]) != period or int(solution["transform"]) != alignment:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None

    # Each backup contributes necessary clean evidence for the other's bad residue.
    combined = _period_vote(a_noisy, b_aligned_noisy, period)
    if combined is None or not np.array_equal(combined[0], tile):
        return None
    if not np.any(a_noisy != clean_a) or not np.any(b_noisy != clean_b):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely solvable pair of cross-repairing periodic regions."""
    # Hold the high-level branch choices fixed during rejection sampling so hard
    # periods and orientations are not silently underrepresented.
    period = random.choice([2, 3, 4])
    axis = random.choice(["row", "col"])
    alignment = random.randrange(1, 8)
    for _ in range(_GEN_ATTEMPTS):
        result = _gen_attempt(period, axis, alignment)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid cross-backup tiling")
