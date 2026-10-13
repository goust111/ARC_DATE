"""ARC task 4: infer periodic frame stripes and cancel marker-named lines."""

from __future__ import annotations

from itertools import combinations

import numpy as np


_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_MARKER_COLOR = 2
_MAX_GENERATION_ATTEMPTS = 300


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background must be uniquely dominant"
    return int(winners[0])


def _perimeter_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """Find the uniquely colored complete, one-cell rectangular perimeter."""
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if np.array_equal(grid == color, _perimeter_mask(grid.shape, box)):
            candidates.append((color, box))
    assert len(candidates) == 1, "expected one unambiguous rectangular frame"
    return candidates[0]


def _infer_stripes(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    box: tuple[int, int, int, int],
) -> tuple[str, int, int]:
    """Return (orientation, stripe color, period) from the exact frame interior."""
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inferences: list[tuple[str, int, int]] = []

    for color in map(int, np.unique(grid)):
        if color in (bg, frame_color):
            continue

        full_rows = np.flatnonzero(np.all(interior == color, axis=1))
        if full_rows.size >= 3:
            gaps = np.diff(full_rows)
            if gaps.size and np.all(gaps == gaps[0]) and int(gaps[0]) >= 2:
                expected = np.full_like(interior, bg)
                expected[full_rows, :] = color
                if np.array_equal(interior, expected):
                    inferences.append(("horizontal", color, int(gaps[0])))

        full_cols = np.flatnonzero(np.all(interior == color, axis=0))
        if full_cols.size >= 3:
            gaps = np.diff(full_cols)
            if gaps.size and np.all(gaps == gaps[0]) and int(gaps[0]) >= 2:
                expected = np.full_like(interior, bg)
                expected[:, full_cols] = color
                if np.array_equal(interior, expected):
                    inferences.append(("vertical", color, int(gaps[0])))

    assert len(inferences) == 1, "stripe direction, color, and period must be unique"
    return inferences[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Project the frame's periodic stripes globally, except for marker-cancelled phase lines.

    Input structure:
    - A uniquely dominant background contains one complete one-cell rectangular frame.
    - The frame interior contains at least three equally spaced, full horizontal or vertical stripes of another color.
    - Singleton pixels of one further color lie outside the frame on distinct, not-yet-drawn lines of that same periodic family.

    Rule:
    - Infer stripe direction, period, and phase solely from the complete stripes inside the frame.
    - Extend every line in that periodic family across the whole grid, coloring background cells only and preserving the frame.
    - Omit every whole candidate line containing a marker, and erase all marker pixels to background.

    Key insight:
    - Each external marker names an entire future phase line to cancel, so the global periodic projection must be inferred before the markers can be interpreted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 7, "expected a nontrivial 2D grid"

    bg = _background(g)
    frame_color, box = _find_frame(g, bg)
    orientation, stripe_color, period = _infer_stripes(g, bg, frame_color, box)
    r0, c0, r1, c1 = box

    other_colors = set(map(int, np.unique(g))) - {bg, frame_color, stripe_color}
    assert len(other_colors) == 1, "expected one structurally distinct marker color"
    marker_color = other_colors.pop()
    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape[0] >= 2, "expected several marker pixels"

    if orientation == "horizontal":
        existing = np.flatnonzero(np.all(g[r0 + 1 : r1, c0 + 1 : c1] == stripe_color, axis=1)) + r0 + 1
        phase = int(existing[0])
        axis_values = marker_coords[:, 0]
        frame_lo, frame_hi = r0, r1
        axis_length = g.shape[0]
    else:
        existing = np.flatnonzero(np.all(g[r0 + 1 : r1, c0 + 1 : c1] == stripe_color, axis=0)) + c0 + 1
        phase = int(existing[0])
        axis_values = marker_coords[:, 1]
        frame_lo, frame_hi = c0, c1
        axis_length = g.shape[1]

    assert len(set(map(int, axis_values))) == marker_coords.shape[0], "markers must cancel distinct lines"
    assert np.all((axis_values - phase) % period == 0), "each marker must lie on the inferred phase family"
    assert np.all((axis_values < frame_lo) | (axis_values > frame_hi)), "markers must name new lines outside the frame"
    for r, c in map(tuple, marker_coords):
        assert not (r0 <= r <= r1 and c0 <= c <= c1), "markers must be outside the frame"
        same_neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == marker_color:
                same_neighbors += 1
        assert same_neighbors == 0, "markers must be singleton pixels"

    cancelled = set(map(int, axis_values))
    candidates = [i for i in range(axis_length) if (i - phase) % period == 0]
    assert cancelled < set(candidates), "at least one inferred line must remain drawable"

    out = g.copy()
    out[g == marker_color] = bg
    for axis in candidates:
        if axis in cancelled:
            continue
        line = out[axis, :] if orientation == "horizontal" else out[:, axis]
        line[line == bg] = stripe_color
    return out


def _axis_specs(length: int) -> list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]]:
    """Enumerate unambiguous frame-axis/period/phase configurations."""
    specs: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = []
    for period in (2, 3, 4):
        for lo in range(2, length - 5):
            for hi in range(lo + 6, min(length - 2, lo + 17)):
                for phase in range(period):
                    family = tuple(i for i in range(length) if i % period == phase)
                    if lo in family or hi in family:
                        continue
                    inside = tuple(i for i in family if lo < i < hi)
                    above = tuple(i for i in family if i < lo)
                    below = tuple(i for i in family if i > hi)
                    outside = above + below
                    if not (3 <= len(inside) <= 5):
                        continue
                    if len(above) < 2 or len(below) < 2:
                        continue
                    specs.append((lo, hi, period, inside, outside))
    return specs


def _choose_cancelled(
    outside: tuple[int, ...], lo: int, hi: int
) -> tuple[int, ...] | None:
    """Choose markers on both sides while retaining visible extensions on both sides."""
    choices: list[tuple[int, ...]] = []
    for count in (2, 3):
        for picked in combinations(outside, count):
            picked_set = set(picked)
            if not any(x < lo for x in picked) or not any(x > hi for x in picked):
                continue
            if not any(x < lo and x not in picked_set for x in outside):
                continue
            if not any(x > hi and x not in picked_set for x in outside):
                continue
            choices.append(tuple(map(int, picked)))
    if not choices:
        return None
    return choices[int(_RNG.integers(len(choices)))]


def _independent_expected(
    inp: np.ndarray,
    bg: int,
    stripe_color: int,
    marker_color: int,
    orientation: str,
    phase: int,
    period: int,
    cancelled: tuple[int, ...],
) -> np.ndarray:
    """Render the specified rule without invoking any solver-side detection."""
    out = inp.copy()
    out[out == marker_color] = bg
    axis_length = out.shape[0] if orientation == "horizontal" else out.shape[1]
    cancelled_set = set(cancelled)
    for axis in range(axis_length):
        if (axis - phase) % period or axis in cancelled_set:
            continue
        line = out[axis, :] if orientation == "horizontal" else out[:, axis]
        line[line == bg] = stripe_color
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    bg: int,
    frame_color: int,
    stripe_color: int,
    marker_color: int,
    orientation: str,
    box: tuple[int, int, int, int],
    period: int,
    inside: tuple[int, ...],
    outside: tuple[int, ...],
    cancelled: tuple[int, ...],
) -> None:
    """Check puzzle semantics, including dependencies that solver closure alone misses."""
    r0, c0, r1, c1 = box
    assert set(map(int, np.unique(inp))) == {bg, frame_color, stripe_color, marker_color}
    assert np.array_equal(inp == frame_color, _perimeter_mask(inp.shape, box))
    assert len(inside) >= 3 and np.all(np.diff(inside) == period)
    assert len(cancelled) >= 2 and set(cancelled) < set(outside)
    assert any(x < (r0 if orientation == "horizontal" else c0) for x in cancelled)
    assert any(x > (r1 if orientation == "horizontal" else c1) for x in cancelled)

    marker_coords = np.argwhere(inp == marker_color)
    marker_axes = marker_coords[:, 0] if orientation == "horizontal" else marker_coords[:, 1]
    assert set(map(int, marker_axes)) == set(cancelled)
    assert all(x not in inside for x in marker_axes)

    erase_only = inp.copy()
    erase_only[erase_only == marker_color] = bg
    assert not np.array_equal(erase_only, expected), "periodic extension must materially change the output"

    no_cancel = _independent_expected(
        inp, bg, stripe_color, marker_color, orientation, inside[0], period, ()
    )
    assert not np.array_equal(no_cancel, expected), "marker cancellation must materially change the output"
    for axis in cancelled:
        line = expected[axis, :] if orientation == "horizontal" else expected[:, axis]
        assert np.all(line == bg), "a named external candidate line must be cancelled in full"

    unmarked = sorted(set(outside) - set(cancelled))
    assert len(unmarked) >= 2
    for axis in unmarked:
        line = expected[axis, :] if orientation == "horizontal" else expected[:, axis]
        assert np.all(line == stripe_color), "an unmarked external candidate line must span the canvas"

    # Exact background gaps between adjacent stripes reject the shortcut of using
    # a divisor of the true period or treating the interior as a solid band.
    if orientation == "horizontal":
        interior = inp[r0 + 1 : r1, c0 + 1 : c1]
        stripe_rows = np.array(inside) - (r0 + 1)
        assert np.all(interior[stripe_rows, :] == stripe_color)
        assert np.any(np.all(interior == bg, axis=1))
    else:
        interior = inp[r0 + 1 : r1, c0 + 1 : c1]
        stripe_cols = np.array(inside) - (c0 + 1)
        assert np.all(interior[:, stripe_cols] == stripe_color)
        assert np.any(np.all(interior == bg, axis=0))

    assert np.array_equal(expected[inp == frame_color], inp[inp == frame_color])
    assert expected.shape == inp.shape and max(expected.shape) <= 30


def _generation_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(20, 31))
    width = int(_RNG.integers(20, 31))
    orientation = "horizontal" if int(_RNG.integers(2)) == 0 else "vertical"
    axis_length = height if orientation == "horizontal" else width
    specs = _axis_specs(axis_length)
    if not specs:
        return None

    lo, hi, period, inside, outside = specs[int(_RNG.integers(len(specs)))]
    cancelled = _choose_cancelled(outside, lo, hi)
    if cancelled is None:
        return None

    cross_length = width if orientation == "horizontal" else height
    max_cross_size = min(15, cross_length - 6)
    if max_cross_size < 8:
        return None
    cross_size = int(_RNG.integers(8, max_cross_size + 1))
    cross_lo = int(_RNG.integers(3, cross_length - cross_size - 2))
    cross_hi = cross_lo + cross_size - 1

    if orientation == "horizontal":
        box = (lo, cross_lo, hi, cross_hi)
    else:
        box = (cross_lo, lo, cross_hi, hi)
    r0, c0, r1, c1 = box

    available = [color for color in _PALETTE if color != _MARKER_COLOR]
    bg, frame_color, stripe_color = map(int, _RNG.choice(available, size=3, replace=False))
    inp = np.full((height, width), bg, dtype=int)
    inp[_perimeter_mask(inp.shape, box)] = frame_color
    if orientation == "horizontal":
        for row in inside:
            inp[row, c0 + 1 : c1] = stripe_color
        marker_crosses = (c0 - 2, c1 + 2)
        for rank, row in enumerate(sorted(cancelled)):
            inp[row, marker_crosses[rank % 2]] = _MARKER_COLOR
    else:
        for col in inside:
            inp[r0 + 1 : r1, col] = stripe_color
        marker_crosses = (r0 - 2, r1 + 2)
        for rank, col in enumerate(sorted(cancelled)):
            inp[marker_crosses[rank % 2], col] = _MARKER_COLOR

    if _background(inp) != bg:
        return None
    expected = _independent_expected(
        inp,
        bg,
        stripe_color,
        _MARKER_COLOR,
        orientation,
        inside[0],
        period,
        cancelled,
    )
    try:
        _semantic_checks(
            inp,
            expected,
            bg=bg,
            frame_color=frame_color,
            stripe_color=stripe_color,
            marker_color=_MARKER_COLOR,
            orientation=orientation,
            box=box,
            period=period,
            inside=inside,
            outside=outside,
            cancelled=cancelled,
        )
        if not np.array_equal(main(inp), expected):
            return None
    except (AssertionError, IndexError, ValueError):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance of the puzzle."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic stripe-cancellation input")
