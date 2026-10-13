from __future__ import annotations

import numpy as np


PLACEHOLDER = 0
PALETTE = np.arange(1, 10, dtype=int)
MIN_COLORS = 4
MAX_COLORS = 6
MAX_GENERATION_ATTEMPTS = 400


def _exact_skew_vector(grid: np.ndarray, k: int, s: int) -> bool:
    """Return True iff row r+k is row r shifted right by s, for every valid r."""
    h, _w = grid.shape
    for r in range(h - int(k)):
        if not np.array_equal(grid[r], np.roll(grid[r + int(k)], -int(s))):
            return False
    return True


def _compatible_partial(grid: np.ndarray, k: int, s: int, placeholder: int) -> bool:
    """Check the skew relation while ignoring placeholder cells on either side."""
    h, _w = grid.shape
    saw_visible_pair = False
    for r in range(h - int(k)):
        lower = np.roll(grid[r + int(k)], -int(s))
        lower_ok = np.roll(grid[r + int(k)] != int(placeholder), -int(s))
        visible = (grid[r] != int(placeholder)) & lower_ok
        if not np.any(visible):
            continue
        saw_visible_pair = True
        if np.any(grid[r, visible] != lower[visible]):
            return False
    return saw_visible_pair


def _repair_with_vector(
    grid: np.ndarray, k: int, s: int, placeholder: int
) -> np.ndarray | None:
    """Fill each placeholder from the unique visible color on its skew orbit."""
    h, w = grid.shape
    out = grid.copy()
    repaired = 0

    for r0 in range(int(k)):
        orbit_len = (h - 1 - r0) // int(k) + 1
        for c0 in range(w):
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            for t in range(orbit_len):
                r = r0 + t * int(k)
                c = (c0 + t * int(s)) % w
                coords.append((r, c))
                value = int(grid[r, c])
                if value != int(placeholder):
                    colors.add(value)
            if not colors or len(colors) != 1:
                return None
            fill = colors.pop()
            for r, c in coords:
                if out[r, c] == int(placeholder):
                    out[r, c] = fill
                    repaired += 1

    if repaired != int(np.count_nonzero(grid == int(placeholder))):
        return None
    return out


def _solve_for_placeholder(
    grid: np.ndarray, placeholder: int
) -> tuple[int, int, np.ndarray] | None:
    h, w = grid.shape
    for k in range(1, h):
        candidates: list[tuple[int, np.ndarray]] = []
        for s in range(1, w):
            if not _compatible_partial(grid, k, s, placeholder):
                continue
            repaired = _repair_with_vector(grid, k, s, placeholder)
            if repaired is None:
                continue
            if _exact_skew_vector(repaired, k, s):
                candidates.append((s, repaired))
        if candidates:
            if len(candidates) != 1:
                raise AssertionError(
                    f"ambiguous skew repair for k={k}: {[s for s, _ in candidates]}"
                )
            s, repaired = candidates[0]
            return k, s, repaired
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every placeholder by following the grid's unique skew wrapped translation orbit.

    Input structure:
    - The grid is fully tiled by a colored pattern, except for several cells overwritten by one uniform placeholder color.
    - Ignoring placeholders, the intact pattern is invariant under one primitive skew translation:
      move down by k rows and simultaneously wrap right by s columns.
    - Pure vertical periodicity, pure horizontal wrap periodicity, and mirror symmetries do not explain the intact cells.

    Rule:
    - Find the smallest row step k for which exactly one wrapped horizontal shift s keeps every visible cell pair consistent.
    - Partition the grid into skew orbits of cells linked by repeated applications of (down k, wrap-right s).
    - Recolor each placeholder with the unique visible color already present somewhere on its orbit; leave other cells unchanged.

    Key insight:
    - The missing value is not copied from the same column or from a mirror mate: it comes from a diagonal-with-wrap translation chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    placeholder_candidates = [PLACEHOLDER]
    if PLACEHOLDER not in set(map(int, g.ravel())):
        colors, counts = np.unique(g, return_counts=True)
        order = np.argsort(counts)
        placeholder_candidates = [int(colors[i]) for i in order]

    solutions: list[tuple[int, int, int, np.ndarray]] = []
    for placeholder in placeholder_candidates:
        solved = _solve_for_placeholder(g, int(placeholder))
        if solved is not None:
            k, s, out = solved
            solutions.append((int(placeholder), k, s, out))
        if placeholder == PLACEHOLDER and solutions:
            break

    assert solutions, "no skew wrapped translation explains the damaged grid"
    best = min(solutions, key=lambda item: (item[1], item[2], item[0]))
    placeholder, k, s, out = best
    assert int(np.count_nonzero(g == placeholder)) > 0, "expected at least one placeholder"
    assert _exact_skew_vector(out, k, s)
    return out


def _make_seed_rows(
    rng: np.random.Generator, k: int, w: int, palette: np.ndarray
) -> np.ndarray | None:
    base = int(rng.choice(palette))
    seed = np.full((k, w), base, dtype=int)
    used_colors = {base}

    for r in range(k):
        seg_count = int(rng.integers(3, min(6, w) + 1))
        cuts = (
            sorted(
                rng.choice(np.arange(1, w), size=seg_count - 1, replace=False).tolist()
            )
            if seg_count > 1
            else []
        )
        bounds = [0] + cuts + [w]
        prev = None
        for c0, c1 in zip(bounds, bounds[1:]):
            choices = [int(color) for color in palette if int(color) != prev]
            color = int(rng.choice(choices))
            seed[r, c0:c1] = color
            used_colors.add(color)
            prev = color

        for _ in range(int(rng.integers(1, 4))):
            run = int(rng.integers(1, min(4, w) + 1))
            c0 = int(rng.integers(0, w - run + 1))
            choices = [int(color) for color in palette if int(color) != int(seed[r, c0])]
            color = int(rng.choice(choices))
            seed[r, c0 : c0 + run] = color
            used_colors.add(color)

        if len(set(map(int, seed[r]))) < 3:
            picks = np.atleast_1d(rng.choice(w, size=min(2, w), replace=False))
            for idx in picks:
                c = int(idx)
                choices = [int(color) for color in palette if int(color) != int(seed[r, c])]
                color = int(rng.choice(choices))
                seed[r, c] = color
                used_colors.add(color)

        if any(np.array_equal(seed[r], np.roll(seed[r], shift)) for shift in range(1, w)):
            return None

    if len(used_colors) < min(MIN_COLORS, len(palette)):
        return None
    return seed


def _build_complete_pattern(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int] | None:
    for _ in range(200):
        w = int(rng.integers(8, 19))
        k = int(rng.integers(2, 6))
        blocks = int(rng.integers(3, 6))
        extra = int(rng.integers(0, k))
        h = blocks * k + extra
        if h > 28:
            continue

        s = int(rng.integers(1, w))
        n_colors = int(rng.integers(MIN_COLORS, MAX_COLORS + 1))
        palette = rng.choice(PALETTE, size=n_colors, replace=False)
        seed = _make_seed_rows(rng, k, w, palette)
        if seed is None:
            continue

        out = np.empty((h, w), dtype=int)
        out[:k] = seed
        for r in range(k, h):
            out[r] = np.roll(out[r - k], s)

        exact_vectors = [
            (kk, ss)
            for kk in range(1, h)
            for ss in range(1, w)
            if _exact_skew_vector(out, kk, ss)
        ]
        if not exact_vectors or min(exact_vectors) != (k, s):
            continue
        if sum(1 for kk, _ss in exact_vectors if kk == k) != 1:
            continue
        if any(_exact_skew_vector(out, kk, 0) for kk in range(1, h)):
            continue
        if any(np.array_equal(out, np.roll(out, shift, axis=1)) for shift in range(1, w)):
            continue
        if np.array_equal(out, np.fliplr(out)) or np.array_equal(out, np.flipud(out)):
            continue
        if h == w and (
            np.array_equal(out, out.T) or np.array_equal(out, np.fliplr(out).T)
        ):
            continue
        return out, k, s
    return None


def _choose_damage_mask(
    rng: np.random.Generator, out: np.ndarray, k: int, s: int
) -> np.ndarray | None:
    h, w = out.shape
    total = h * w
    min_cells = max(8, total // 10)
    max_cells = min(28, total // 4)

    for _ in range(300):
        mask = np.zeros((h, w), dtype=bool)
        rect_count = int(rng.integers(2, 5))
        for _ in range(rect_count):
            rh = int(rng.integers(1, min(5, h) + 1))
            rw = int(rng.integers(1, min(6, w) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            mask[r0 : r0 + rh, c0 : c0 + rw] = True

        damaged = int(mask.sum())
        if damaged < min_cells or damaged > max_cells:
            continue

        damaged_orbits = 0
        okay = True
        for r0 in range(k):
            orbit_len = (h - 1 - r0) // k + 1
            for c0 in range(w):
                hits = 0
                for t in range(orbit_len):
                    r = r0 + t * k
                    c = (c0 + t * s) % w
                    hits += int(mask[r, c])
                if hits == orbit_len:
                    okay = False
                    break
                if hits:
                    damaged_orbits += 1
            if not okay:
                break
        if not okay or damaged_orbits < max(4, k):
            continue

        vertical_counterexamples = 0
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            if r >= k and int(out[r - k, c]) != int(out[r, c]):
                vertical_counterexamples += 1
            elif r + k < h and int(out[r + k, c]) != int(out[r, c]):
                vertical_counterexamples += 1
        if vertical_counterexamples < max(3, damaged // 5):
            continue
        return mask
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_complete_pattern(rng)
        if built is None:
            continue
        out, k, s = built

        damage = _choose_damage_mask(rng, out, k, s)
        if damage is None:
            continue

        inp = out.copy()
        inp[damage] = PLACEHOLDER

        solved = _solve_for_placeholder(inp, PLACEHOLDER)
        if solved is None:
            continue
        found_k, found_s, repaired = solved
        if (found_k, found_s) != (k, s):
            continue
        if not np.array_equal(repaired, out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid skew-wrapped repair input")
