from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FOREGROUND_CHOICES = tuple(range(1, 10))
MIN_GAP_BRANCHES = (1, 3)
MAX_ATTEMPTS = 400


def _scale(grid: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor integer upscale."""
    return np.kron(grid, np.ones((int(factor), int(factor)), dtype=int))


def _mirror_pair_columns(width: int, gap: int) -> tuple[int, int]:
    """Return the mirrored marker columns for a given odd width and interior gap."""
    c0 = (int(width) - int(gap) - 2) // 2
    c1 = int(width) - 1 - c0
    return int(c0), int(c1)


def _parse_rows(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Extract mirrored marker pairs as (row, c0, c1, gap)."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 1 and w >= 3 and w % 2 == 1, "expected an odd-width grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]

    pairs: list[tuple[int, int, int, int]] = []
    for r in range(h):
        row = g[r]
        cols = np.flatnonzero(row == color)
        assert len(cols) in (0, 2), "each non-empty row must contain exactly two markers"
        assert np.all((row == BACKGROUND) | (row == color)), "only background and one foreground color are allowed"
        if len(cols) == 0:
            continue
        c0, c1 = map(int, cols)
        assert c0 < c1, "marker columns must be distinct"
        assert c0 + c1 == w - 1, "marker pair must be mirrored across the vertical center"
        gap = int(c1 - c0 - 1)
        assert gap >= 1, "markers must have at least one background cell between them"
        pairs.append((int(r), c0, c1, gap))

    assert pairs, "expected at least one mirrored marker pair"
    return color, pairs


def _connect_min_gap_row(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int]]:
    """Fill the unique minimum-gap mirrored pair row."""
    g = np.asarray(grid, dtype=int)
    color, pairs = _parse_rows(g)
    min_gap = min(gap for _r, _c0, _c1, gap in pairs)
    min_rows = [pair for pair in pairs if pair[3] == min_gap]
    assert len(min_rows) == 1, "expected exactly one row with the unique minimum gap"

    out = g.copy()
    r, c0, c1, gap = min_rows[0]
    out[r, c0 : c1 + 1] = color
    return out, int(gap), min_rows[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the unique closest mirrored marker pair, then upscale the whole grid by gap+1.

    Input structure:
    - The background is black, and all non-black pixels share one foreground color.
    - Each non-empty row contains exactly two foreground marker pixels mirrored across the vertical center.
    - Different marked rows can have different horizontal gaps between their mirrored markers.

    Rule:
    - Compute the number of background cells between the two mirrored markers on every marked row.
    - Find the unique row with the smallest such gap, and fill every cell from its left marker to its right marker with the foreground color.
    - Leave every other row unchanged.
    - Let S be that minimum gap plus one, and upscale the entire post-connection grid by factor S in both dimensions.

    Key insight:
    - The same minimum-gap measurement is used twice: first to choose which row gets connected, then to set the global pixel-upscale factor.
    """
    connected, gap, _pair = _connect_min_gap_row(grid)
    return _scale(connected, gap + 1)


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int, int, int]]:
    """Independent expected transform used by the generator validator."""
    connected, gap, pair = _connect_min_gap_row(grid)
    return _scale(connected, gap + 1), connected, gap, pair


def _sample_dimensions(min_gap: int) -> tuple[int, int]:
    """Choose dimensions that always keep the scaled output within 30x30."""
    if int(min_gap) == 3:
        return 7, 7
    height = random.choice((7, 8))
    width = random.choice((7, 9))
    return int(height), int(width)


def _available_gaps(width: int) -> list[int]:
    """All realizable odd gaps for an odd-width mirrored pair grid."""
    return [int(width - 2 - 2 * c0) for c0 in range(width // 2)]


def _row_plan(height: int, width: int, min_gap: int) -> list[int | None]:
    """Build a randomized per-row gap plan with one unique minimum row."""
    pair_rows = random.randint(4, min(6, int(height)))
    larger_gaps = [gap for gap in _available_gaps(width) if gap > int(min_gap)]
    assert larger_gaps, "generator needs at least one non-minimum gap"

    distinct_non_min = random.randint(1, min(3, len(larger_gaps)))
    active_non_min = sorted(random.sample(larger_gaps, distinct_non_min))
    non_min_rows = [random.choice(active_non_min) for _ in range(pair_rows - 1)]

    if len(active_non_min) > 1 and len(set(non_min_rows)) == 1:
        replace_at = random.randrange(len(non_min_rows))
        alternatives = [gap for gap in active_non_min if gap != non_min_rows[replace_at]]
        non_min_rows[replace_at] = random.choice(alternatives)

    plan: list[int | None] = [int(min_gap), *non_min_rows]
    plan.extend([None] * (int(height) - pair_rows))
    random.shuffle(plan)
    return plan


def _render_input(height: int, width: int, color: int, plan: list[int | None]) -> np.ndarray:
    """Render a grid from a per-row mirrored-gap specification."""
    grid = np.zeros((int(height), int(width)), dtype=int)
    for r, gap in enumerate(plan):
        if gap is None:
            continue
        c0, c1 = _mirror_pair_columns(int(width), int(gap))
        grid[int(r), [int(c0), int(c1)]] = int(color)
    return grid


def _connect_all_rows(grid: np.ndarray) -> np.ndarray:
    """Shortcut baseline used to reject degenerate training cases."""
    g = np.asarray(grid, dtype=int)
    color, pairs = _parse_rows(g)
    out = g.copy()
    for r, c0, c1, _gap in pairs:
        out[r, c0 : c1 + 1] = int(color)
    return out


def _validate_case(grid: np.ndarray) -> bool:
    """Semantic validator: check structure, indispensable steps, and oracle agreement."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    if max(h, w) > 30:
        return False

    try:
        color, pairs = _parse_rows(g)
        if not (4 <= len(pairs) <= 6):
            return False
        if h - len(pairs) > 2:
            return False

        expected, connected, gap, pair = _oracle_output(g)
        if max(expected.shape) > 30:
            return False

        r, c0, c1, _ = pair
        changed = connected != g
        changed_coords = {(int(rr), int(cc)) for rr, cc in np.argwhere(changed)}
        expected_added = {(int(r), int(c)) for c in range(int(c0) + 1, int(c1))}
        if changed_coords != expected_added:
            return False
        if len(changed_coords) != int(gap):
            return False

        for other_r, _, _, other_gap in pairs:
            if int(other_r) == int(r):
                continue
            if other_gap <= gap:
                return False
            if not np.array_equal(connected[int(other_r)], g[int(other_r)]):
                return False

        scale_factor = int(gap) + 1
        if expected.shape != (h * scale_factor, w * scale_factor):
            return False

        # Reject shortcut interpretations.
        if np.array_equal(_scale(g, scale_factor), expected):
            return False
        if np.array_equal(_scale(_connect_all_rows(g), scale_factor), expected):
            return False
        wrong_scale = 4 if scale_factor == 2 else 2
        wrong_scaled = _scale(connected, wrong_scale)
        if wrong_scaled.shape == expected.shape and np.array_equal(wrong_scaled, expected):
            return False

        if not np.array_equal(main(g), expected):
            return False
        return int(color) in FOREGROUND_CHOICES
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    """Sample one candidate input grid."""
    min_gap = random.choice(MIN_GAP_BRANCHES)
    height, width = _sample_dimensions(int(min_gap))
    color = random.choice(FOREGROUND_CHOICES)
    plan = _row_plan(int(height), int(width), int(min_gap))
    grid = _render_input(int(height), int(width), int(color), plan)
    return grid if _validate_case(grid) else None


def generate_input() -> np.ndarray:
    """Generate a valid random training input for this task."""
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
