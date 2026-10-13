from __future__ import annotations

import hashlib
import random
from itertools import product

import numpy as np


BG = 7
SEP = 5
POS_BAR = 8
NEG_BAR = 2
POS_OUT = 1
NEG_OUT = 3

_PATTERNS_BY_RESULT: dict[int, list[tuple[tuple[int, int], ...]]] = {}
_USED_OUTPUT_SIGNATURES: set[str] = set()


def _build_patterns() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_result: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    seen: dict[int, set[tuple[tuple[int, int], ...]]] = {}
    for n_bars in (2, 3):
        for colors in product((POS_BAR, NEG_BAR), repeat=n_bars):
            for heights in product((1, 2, 3), repeat=n_bars):
                result = sum(h if color == POS_BAR else -h for color, h in zip(colors, heights))
                if result == 0 or abs(result) > 4:
                    continue
                pattern = tuple(sorted(zip(colors, heights)))
                seen.setdefault(result, set())
                if pattern in seen[result]:
                    continue
                seen[result].add(pattern)
                by_result.setdefault(result, []).append(pattern)
    return by_result


_PATTERNS_BY_RESULT = _build_patterns()


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    row_candidates = [r for r in range(h) if np.all(grid[r, :] == SEP)]
    col_candidates = [c for c in range(w) if np.all(grid[:, c] == SEP)]
    assert len(row_candidates) == 1, "expected exactly one solid grey separator row"
    assert len(col_candidates) == 1, "expected exactly one solid grey separator column"
    return int(row_candidates[0]), int(col_candidates[0])


def _quadrant_bounds(cross_r: int, cross_c: int, height: int, width: int) -> list[tuple[int, int, int, int]]:
    bounds = [
        (0, cross_r, 0, cross_c),
        (0, cross_r, cross_c + 1, width),
        (cross_r + 1, height, 0, cross_c),
        (cross_r + 1, height, cross_c + 1, width),
    ]
    for r0, r1, c0, c1 in bounds:
        assert r1 > r0 and c1 > c0, "separator cross must leave four non-empty quadrants"
    return bounds


def _parse_bar_column(col: np.ndarray) -> tuple[int, int] | None:
    bar_rows = np.flatnonzero((col == POS_BAR) | (col == NEG_BAR))
    assert not np.any((col != BG) & (col != POS_BAR) & (col != NEG_BAR)), "unexpected color inside quadrant"
    if len(bar_rows) == 0:
        return None
    color_values = {int(v) for v in col[bar_rows]}
    assert len(color_values) == 1, "a bar column must be monochromatic"
    color = color_values.pop()
    start = int(bar_rows[0])
    expected = np.arange(start, len(col))
    assert np.array_equal(bar_rows, expected), "bars must be contiguous and bottom-aligned"
    height = int(len(col) - start)
    assert 1 <= height <= 3, "bar height must be between 1 and 3"
    return color, height


def _quadrant_bars(region: np.ndarray) -> list[dict[str, int]]:
    bars: list[dict[str, int]] = []
    for c in range(region.shape[1]):
        parsed = _parse_bar_column(region[:, c])
        if parsed is None:
            continue
        color, height = parsed
        bars.append({"color": int(color), "col": int(c), "height": int(height)})
    return bars


def _result_from_bars(bars: list[dict[str, int]]) -> int:
    result = 0
    for bar in bars:
        if int(bar["color"]) == POS_BAR:
            result += int(bar["height"])
        else:
            assert int(bar["color"]) == NEG_BAR
            result -= int(bar["height"])
    return int(result)


def _draw_diamond_outline(
    out: np.ndarray,
    bounds: tuple[int, int, int, int],
    center_r: int,
    center_c: int,
    radius: int,
    color: int,
) -> None:
    r0, r1, c0, c1 = bounds
    for rr in range(center_r - radius, center_r + radius + 1):
        if not (r0 <= rr < r1):
            continue
        dc = radius - abs(rr - center_r)
        for cc in {center_c - dc, center_c + dc}:
            if c0 <= cc < c1 and out[rr, cc] == BG:
                out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Convert each quadrant's bar-height arithmetic into a centered colored diamond outline.

    Input structure:
    - An orange background grid contains one solid grey separator row and one solid grey separator column.
    - The separator cross splits the canvas into four rectangular quadrants.
    - Each quadrant contains 2-3 bottom-aligned one-cell-wide bars: teal bars are positive and red bars are negative.

    Rule:
    - Find the grey cross and process each quadrant independently.
    - Compute R = sum(teal bar heights) - sum(red bar heights) inside that quadrant.
    - Erase all bar pixels to orange, then draw the Manhattan-distance outline |r-cr|+|c-cc| = |R| around the quadrant center.
    - Use blue for positive R, green for negative R, and keep the grey separator unchanged.

    Key insight:
    - The bars do not produce another bar: their signed height sum becomes a geometric radius, while the sign becomes the outline color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    cross_r, cross_c = _find_separator_cross(g)

    out = g.copy()
    bounds_by_quad = _quadrant_bounds(cross_r, cross_c, h, w)
    for bounds in bounds_by_quad:
        r0, r1, c0, c1 = bounds
        region = g[r0:r1, c0:c1]
        bars = _quadrant_bars(region)
        result = _result_from_bars(bars)
        assert 2 <= len(bars) <= 3, "each quadrant must contain 2-3 bars"
        assert result != 0, "each quadrant must have a non-zero signed result"
        assert abs(result) <= 4, "quadrant arithmetic result must stay within the supported radius range"

        mask = (out[r0:r1, c0:c1] == POS_BAR) | (out[r0:r1, c0:c1] == NEG_BAR)
        out[r0:r1, c0:c1][mask] = BG

        center_r = r0 + (r1 - r0) // 2
        center_c = c0 + (c1 - c0) // 2
        radius = abs(result)
        color = POS_OUT if result > 0 else NEG_OUT
        _draw_diamond_outline(out, bounds, center_r, center_c, radius, color)

    return out


def _sample_results(max_radius: int) -> list[int]:
    choices = [r for r in range(-max_radius, max_radius + 1) if r != 0]
    for _ in range(200):
        results = [random.choice(choices) for _ in range(4)]
        if not any(r > 0 for r in results):
            continue
        if not any(r < 0 for r in results):
            continue
        if len({abs(r) for r in results}) < 2:
            continue
        if len(set(results)) < 3:
            continue
        if max(abs(r) for r in results) < min(3, max_radius):
            continue
        return results
    raise RuntimeError("failed to sample non-degenerate quadrant results")


def _mixed_pattern(pattern: tuple[tuple[int, int], ...]) -> bool:
    return {color for color, _height in pattern} == {POS_BAR, NEG_BAR}


def _choose_pattern(result: int, require_mixed: bool) -> tuple[tuple[int, int], ...]:
    patterns = _PATTERNS_BY_RESULT[result]
    mixed = [pattern for pattern in patterns if _mixed_pattern(pattern)]
    candidates = mixed if require_mixed and mixed else patterns

    preferred_count = 3 if abs(result) >= 3 else random.choice((2, 3))
    preferred = [pattern for pattern in candidates if len(pattern) == preferred_count]
    if preferred:
        candidates = preferred

    if mixed and not require_mixed and random.random() < 0.7:
        mixed_preferred = [pattern for pattern in candidates if _mixed_pattern(pattern)]
        if mixed_preferred:
            candidates = mixed_preferred

    return tuple(random.choice(candidates))


def _sample_bar_columns(width: int, n_bars: int) -> list[int]:
    candidates = list(range(1, width - 1))
    assert len(candidates) >= n_bars
    for _ in range(100):
        cols = sorted(random.sample(candidates, n_bars))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(len(cols) - 1)):
            return cols
    return sorted(random.sample(candidates, n_bars))


def _quadrant_center(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, r1, c0, c1 = bounds
    return int(r0 + (r1 - r0) // 2), int(c0 + (c1 - c0) // 2)


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_transformation(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and out.ndim == 2
    h, w = inp.shape
    cross_r, cross_c = _find_separator_cross(inp)
    assert np.all(out[cross_r, :] == SEP)
    assert np.all(out[:, cross_c] == SEP)

    results: list[int] = []
    mixed_quadrants = 0
    expected = inp.copy()
    bounds_by_quad = _quadrant_bounds(cross_r, cross_c, h, w)
    for bounds in bounds_by_quad:
        r0, r1, c0, c1 = bounds
        region = inp[r0:r1, c0:c1]
        bars = _quadrant_bars(region)
        assert 2 <= len(bars) <= 3, "generator must place 2-3 bars in each quadrant"
        colors = {int(bar["color"]) for bar in bars}
        if colors == {POS_BAR, NEG_BAR}:
            mixed_quadrants += 1
        result = _result_from_bars(bars)
        results.append(result)

        radius = abs(result)
        center_r, center_c = _quadrant_center(bounds)
        fit_radius = min(center_r - r0, r1 - 1 - center_r, center_c - c0, c1 - 1 - center_c)
        assert 1 <= radius <= min(4, fit_radius), "diamond radius must be visible inside the quadrant"

        cleared = expected[r0:r1, c0:c1]
        cleared[(cleared == POS_BAR) | (cleared == NEG_BAR)] = BG
        color = POS_OUT if result > 0 else NEG_OUT
        _draw_diamond_outline(expected, bounds, center_r, center_c, radius, color)

        out_region = out[r0:r1, c0:c1]
        assert not np.any((out_region == POS_BAR) | (out_region == NEG_BAR)), "bars must be erased in output"

    assert any(r > 0 for r in results) and any(r < 0 for r in results), "both output colors must appear"
    assert len({abs(r) for r in results}) >= 2, "generator must expose at least two radii"
    assert len(set(results)) >= 3, "generator must avoid collapsing all quadrants to the same result"
    assert mixed_quadrants >= 2, "generator must include mixed-sign arithmetic cases"
    assert np.array_equal(expected, out), "output must match the semantic rule exactly"


def _generate_candidate() -> np.ndarray:
    qh = random.randint(9, 13)
    qw = random.randint(9, 13)
    h = 2 * qh + 1
    w = 2 * qw + 1
    grid = np.full((h, w), BG, dtype=int)
    grid[qh, :] = SEP
    grid[:, qw] = SEP

    bounds_by_quad = _quadrant_bounds(qh, qw, h, w)
    max_radius = min(
        4,
        min(
            min(center_r - r0, r1 - 1 - center_r, center_c - c0, c1 - 1 - center_c)
            for (r0, r1, c0, c1), (center_r, center_c) in (
                (bounds, _quadrant_center(bounds)) for bounds in bounds_by_quad
            )
        ),
    )
    assert max_radius >= 4

    results = _sample_results(max_radius)
    must_mixed = set(random.sample(range(4), random.randint(2, 3)))

    for idx, bounds in enumerate(bounds_by_quad):
        r0, r1, c0, c1 = bounds
        pattern = _choose_pattern(results[idx], idx in must_mixed)
        cols = _sample_bar_columns(c1 - c0, len(pattern))
        random.shuffle(cols)
        bars = list(pattern)
        random.shuffle(bars)
        for local_c, (color, height) in zip(cols, bars):
            abs_c = c0 + local_c
            grid[r1 - height : r1, abs_c] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _generate_candidate()
        out = main(inp)
        _validate_transformation(inp, out)

        out_sig = _output_signature(out)
        if out_sig in _USED_OUTPUT_SIGNATURES:
            continue
        _USED_OUTPUT_SIGNATURES.add(out_sig)
        return inp
    raise RuntimeError("failed to generate a non-degenerate valid input")
