from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _class_id(r: int, c: int, v_period: int, h_period: int, row_shift: int) -> int:
    q, t = divmod(int(c), int(h_period))
    u = (int(r) - q * int(row_shift)) % int(v_period)
    return u * int(h_period) + t


def _pair_stats(grid: np.ndarray, dr: int, dc: int) -> tuple[int, int]:
    h, w = grid.shape
    if dr < 0 or dc < 0 or dr >= h or dc >= w:
        return 0, 0
    a = grid[: h - dr, : w - dc]
    b = grid[dr:, dc:]
    mask = (a != BLACK) & (b != BLACK)
    if not np.any(mask):
        return 0, 0
    same = int(np.count_nonzero(mask & (a == b)))
    diff = int(np.count_nonzero(mask & (a != b)))
    return same, diff


def _evaluate_candidate(
    grid: np.ndarray,
    v_period: int,
    h_period: int,
    row_shift: int,
    *,
    require_direct_pairs: bool,
) -> tuple[np.ndarray, int] | None:
    h, w = grid.shape

    vertical_same, vertical_diff = _pair_stats(grid, v_period, 0)
    shift_same, shift_diff = _pair_stats(grid, row_shift, h_period)
    if vertical_diff or shift_diff:
        return None
    if require_direct_pairs and (vertical_same == 0 or shift_same == 0):
        return None

    palette = np.full(v_period * h_period, -1, dtype=int)
    seen = np.zeros(v_period * h_period, dtype=int)
    repeat_pairs = 0

    rows, cols = np.where(grid != BLACK)
    for r, c in zip(rows.tolist(), cols.tolist()):
        idx = _class_id(r, c, v_period, h_period, row_shift)
        color = int(grid[r, c])
        prev = int(palette[idx])
        if prev == -1:
            palette[idx] = color
        elif prev != color:
            return None
        repeat_pairs += int(seen[idx])
        seen[idx] += 1

    if repeat_pairs == 0:
        return None

    for r in range(-1, h + 1):
        for c in range(-1, w + 1):
            idx = _class_id(r, c, v_period, h_period, row_shift)
            if int(palette[idx]) == -1:
                return None

    return palette, repeat_pairs


def _enumerate_candidates(grid: np.ndarray, *, require_direct_pairs: bool) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []

    max_v = h - 1 if require_direct_pairs else h + 1
    max_p = w - 1 if require_direct_pairs else w + 1

    for v_period in range(2, max_v + 1):
        for h_period in range(1, max_p + 1):
            for row_shift in range(1, v_period):
                result = _evaluate_candidate(
                    grid,
                    v_period,
                    h_period,
                    row_shift,
                    require_direct_pairs=require_direct_pairs,
                )
                if result is None:
                    continue
                palette, repeat_pairs = result
                candidates.append(
                    {
                        "v_period": v_period,
                        "h_period": h_period,
                        "row_shift": row_shift,
                        "palette": palette,
                        "repeat_pairs": repeat_pairs,
                    }
                )
    return candidates


def _unique_candidate(grid: np.ndarray) -> dict:
    strict = _enumerate_candidates(grid, require_direct_pairs=True)
    if strict:
        best = min(strict, key=lambda c: (c["v_period"], c["h_period"], c["row_shift"]))
        best_key = (best["v_period"], best["h_period"], best["row_shift"])
        matches = [c for c in strict if (c["v_period"], c["h_period"], c["row_shift"]) == best_key]
        assert len(matches) == 1, "expected a unique minimal strict candidate"
        return matches[0]

    relaxed = _enumerate_candidates(grid, require_direct_pairs=False)
    assert relaxed, "expected at least one shifted-period candidate"
    best = min(relaxed, key=lambda c: (c["v_period"], c["h_period"], c["row_shift"]))
    best_key = (best["v_period"], best["h_period"], best["row_shift"])
    matches = [c for c in relaxed if (c["v_period"], c["h_period"], c["row_shift"]) == best_key]
    assert len(matches) == 1, "expected a unique minimal relaxed candidate"
    return matches[0]


def _render_from_candidate(shape: tuple[int, int], candidate: dict) -> np.ndarray:
    h, w = shape
    out = np.zeros((h + 2, w + 2), dtype=int)
    palette = candidate["palette"]
    v_period = int(candidate["v_period"])
    h_period = int(candidate["h_period"])
    row_shift = int(candidate["row_shift"])

    for r in range(-1, h + 1):
        for c in range(-1, w + 1):
            idx = _class_id(r, c, v_period, h_period, row_shift)
            out[r + 1, c + 1] = int(palette[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a window of a brick-shifted periodic pattern by one cell on every side and repair its black holes.

    Input structure:
    - The non-black cells are a rectangular window cut from an infinite periodic pattern.
    - The pattern repeats after a fixed vertical period, and moving right by one horizontal period also shifts the pattern downward by a fixed nonzero row offset.
    - A few cells inside the window are erased to black; black is not a valid pattern color.

    Rule:
    - Infer the unique vertical period, horizontal period, and row shift that explain all visible non-black cells.
    - Use that brick-shifted periodic structure to refill every black hole in the window.
    - Return a grid that is two rows taller and two columns wider, with the repaired input centered and a one-cell border extrapolated from the same infinite pattern rather than copied from the edge.

    Key insight:
    - The window is not governed by an ordinary rectangular tile: the decisive invariant is the lattice generated by a pure vertical repeat and a rightward period that carries a matching downward row shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidate = _unique_candidate(g)
    return _render_from_candidate(g.shape, candidate)


def _full_horizontal_period(v_period: int, h_period: int, row_shift: int) -> int:
    return h_period * v_period // math.gcd(v_period, row_shift)


def _pattern_color(
    tile: np.ndarray,
    v_period: int,
    h_period: int,
    row_shift: int,
    row: int,
    col: int,
) -> int:
    q, t = divmod(int(col), int(h_period))
    return int(tile[(int(row) - q * int(row_shift)) % int(v_period), t])


def _sample_tile(v_period: int, h_period: int) -> np.ndarray | None:
    n_cells = v_period * h_period
    n_colors = random.randint(3, min(5, n_cells))
    colors = random.sample(PALETTE, n_colors)

    for _ in range(80):
        tile = np.array([random.choice(colors) for _ in range(n_cells)], dtype=int).reshape(v_period, h_period)

        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if any(len(set(map(int, tile[r]))) == 1 for r in range(v_period)):
            continue
        if v_period > 1 and any(len(set(map(int, tile[:, c]))) == 1 for c in range(h_period)):
            continue
        if any(np.array_equal(tile[r], tile[(r + 1) % v_period]) for r in range(v_period - 1)):
            continue

        smaller_vertical = False
        for dv in range(1, v_period):
            if np.array_equal(tile, np.roll(tile, -dv, axis=0)):
                smaller_vertical = True
                break
        if smaller_vertical:
            continue
        return tile

    return None


def _make_clean_window(
    tile: np.ndarray,
    v_period: int,
    h_period: int,
    row_shift: int,
    row0: int,
    col0: int,
    height: int,
    width: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            out[r, c] = _pattern_color(tile, v_period, h_period, row_shift, row0 + r, col0 + c)
    return out


def _choose_window_shape(v_period: int, h_period: int, row_shift: int) -> tuple[int, int] | None:
    full_w = _full_horizontal_period(v_period, h_period, row_shift)

    min_h = max(v_period + 2, 2 * v_period)
    max_h = min(16, 3 * v_period + 1)
    min_w = max(h_period + 2, 2 * h_period + 1)
    max_w = min(12, 4 * h_period, full_w - 2)

    if min_h > max_h or min_w > max_w:
        return None
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _try_holes(clean: np.ndarray, v_period: int, h_period: int, row_shift: int) -> np.ndarray | None:
    h, w = clean.shape
    if h < 3 or w < 3:
        return None

    ids = np.zeros_like(clean, dtype=int)
    counts = np.zeros(v_period * h_period, dtype=int)
    for r in range(h):
        for c in range(w):
            idx = _class_id(r, c, v_period, h_period, row_shift)
            ids[r, c] = idx
            counts[idx] += 1

    eligible = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if counts[ids[r, c]] >= 2
    ]
    if len(eligible) < 2:
        return None

    max_holes = min(6, len(eligible))
    target_holes = random.randint(2, max_holes)

    random.shuffle(eligible)
    remaining = counts.copy()
    selected: list[tuple[int, int]] = []
    used_ids: set[int] = set()

    for r, c in eligible:
        idx = int(ids[r, c])
        if idx in used_ids or remaining[idx] <= 1:
            continue
        selected.append((r, c))
        remaining[idx] -= 1
        used_ids.add(idx)
        if len(selected) == target_holes:
            break

    if len(selected) < target_holes:
        for r, c in eligible:
            if (r, c) in selected:
                continue
            idx = int(ids[r, c])
            if remaining[idx] <= 1:
                continue
            selected.append((r, c))
            remaining[idx] -= 1
            if len(selected) == target_holes:
                break

    if len(selected) < 2:
        return None

    repaired_colors = {int(clean[r, c]) for r, c in selected}
    if len(repaired_colors) < 2:
        return None

    holey = clean.copy()
    for r, c in selected:
        holey[r, c] = BLACK
    return holey


def _rectangular_candidate(grid: np.ndarray, rv: int, cw: int) -> bool:
    h, w = grid.shape
    palette = np.full(rv * cw, -1, dtype=int)
    repeat_pairs = 0
    seen = np.zeros(rv * cw, dtype=int)

    rows, cols = np.where(grid != BLACK)
    for r, c in zip(rows.tolist(), cols.tolist()):
        idx = (r % rv) * cw + (c % cw)
        color = int(grid[r, c])
        prev = int(palette[idx])
        if prev == -1:
            palette[idx] = color
        elif prev != color:
            return False
        repeat_pairs += int(seen[idx])
        seen[idx] += 1

    if repeat_pairs == 0:
        return False

    for r in range(-1, h + 1):
        for c in range(-1, w + 1):
            idx = (r % rv) * cw + (c % cw)
            if int(palette[idx]) == -1:
                return False
    return True


def _has_rectangular_shortcut(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for rv in range(1, h):
        for cw in range(1, w):
            if _rectangular_candidate(grid, rv, cw):
                return True
    return False


def _semantic_instance_ok(
    holey: np.ndarray,
    clean: np.ndarray,
    v_period: int,
    h_period: int,
    row_shift: int,
) -> bool:
    if math.gcd(v_period, row_shift) != 1:
        return False
    if _full_horizontal_period(v_period, h_period, row_shift) <= holey.shape[1] + 1:
        return False
    if not np.any(holey == BLACK):
        return False

    clean_candidates = _enumerate_candidates(clean, require_direct_pairs=False)
    if not clean_candidates:
        return False
    clean_candidate = min(clean_candidates, key=lambda c: (c["v_period"], c["h_period"], c["row_shift"]))
    if (
        clean_candidate["v_period"] != v_period
        or clean_candidate["h_period"] != h_period
        or clean_candidate["row_shift"] != row_shift
    ):
        return False

    holey_candidates = _enumerate_candidates(holey, require_direct_pairs=False)
    if not holey_candidates:
        return False
    candidate = min(holey_candidates, key=lambda c: (c["v_period"], c["h_period"], c["row_shift"]))
    if (
        candidate["v_period"] != v_period
        or candidate["h_period"] != h_period
        or candidate["row_shift"] != row_shift
    ):
        return False

    vertical_same, _ = _pair_stats(holey, v_period, 0)
    shift_same, _ = _pair_stats(holey, row_shift, h_period)
    if vertical_same == 0 or shift_same == 0:
        return False

    expected = _render_from_candidate(clean.shape, candidate)
    actual = main(holey)
    if not np.array_equal(actual, expected):
        return False

    if not np.any(actual[1:-1, 1:-1] != holey):
        return False

    ring = np.ones_like(actual, dtype=bool)
    ring[1:-1, 1:-1] = False
    naive = np.pad(clean, 1, mode="edge")
    if not np.any(actual[ring] != naive[ring]):
        return False

    if _has_rectangular_shortcut(holey):
        return False

    return True


def _attempt_instance() -> np.ndarray | None:
    v_period = random.randint(3, 6)
    row_shift_choices = [s for s in range(1, v_period) if math.gcd(s, v_period) == 1]
    if not row_shift_choices:
        return None
    row_shift = random.choice(row_shift_choices)
    h_period = random.randint(2, 5)

    shape = _choose_window_shape(v_period, h_period, row_shift)
    if shape is None:
        return None
    height, width = shape

    tile = _sample_tile(v_period, h_period)
    if tile is None:
        return None

    row0 = random.randint(0, v_period - 1)
    col0 = random.randint(0, _full_horizontal_period(v_period, h_period, row_shift) - 1)
    clean = _make_clean_window(tile, v_period, h_period, row_shift, row0, col0, height, width)

    holey = _try_holes(clean, v_period, h_period, row_shift)
    if holey is None:
        return None

    if not _semantic_instance_ok(holey, clean, v_period, h_period, row_shift):
        return None
    return holey


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shifted-period ARC input")
