from __future__ import annotations

import random

import numpy as np


BG = 0
MAROON = 9
PERIODS = (3, 4, 5)
INPUT_COLORS = tuple(range(1, 9))
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_SIZE = 30


def _line_signature(line: np.ndarray):
    idx = np.flatnonzero(line != BG)
    if idx.size == 0:
        return None

    start = int(idx[0])
    end = int(idx[-1])
    segment = line[start : end + 1]
    if np.any(segment == BG):
        return False

    colors = np.unique(segment)
    if colors.size != 1:
        return False

    return start, end, int(colors[0])


def _parse_orientation(grid: np.ndarray, axis: int):
    view = grid if axis == 0 else grid.T
    line_info = []
    for i in range(view.shape[0]):
        info = _line_signature(view[i])
        if info is False:
            return None
        line_info.append(info)

    active = [info for info in line_info if info is not None]
    if not active:
        return None

    spans = {(int(start), int(end)) for start, end, _color in active}
    if len(spans) != 1:
        return None
    span = next(iter(spans))

    bands = []
    i = 0
    while i < len(line_info):
        if line_info[i] is None:
            i += 1
            continue

        start = i
        color = int(line_info[i][2])
        while i + 1 < len(line_info) and line_info[i + 1] is not None:
            next_color = int(line_info[i + 1][2])
            if next_color != color:
                break
            i += 1

        bands.append((int(start), int(i), int(color)))
        i += 1

    if len(bands) < 3:
        return None

    thicknesses = {end - start + 1 for start, end, _color in bands}
    if len(thicknesses) != 1:
        return None

    for (_start0, end0, _color0), (start1, _end1, _color1) in zip(bands, bands[1:]):
        if int(start1 - end0 - 1) != 1:
            return None

    return {
        "axis": int(axis),
        "span": span,
        "bands": bands,
        "thickness": int(next(iter(thicknesses))),
    }


def _infer_repair(colors: list[int]):
    n = len(colors)
    candidates = []

    for period in PERIODS:
        residues_in_use = {i % period for i in range(n)}
        for bad_idx in range(n):
            residue_to_color = {}
            ok = True
            for i, color in enumerate(colors):
                if i == bad_idx:
                    continue
                residue = int(i % period)
                if residue not in residue_to_color:
                    residue_to_color[residue] = int(color)
                elif residue_to_color[residue] != int(color):
                    ok = False
                    break
            if not ok:
                continue
            if any(residue not in residue_to_color for residue in residues_in_use):
                continue

            predicted = [int(residue_to_color[i % period]) for i in range(n)]
            mismatches = [i for i, (seen, want) in enumerate(zip(colors, predicted)) if int(seen) != int(want)]
            if mismatches == [bad_idx]:
                candidates.append((int(period), int(bad_idx), int(predicted[bad_idx]), tuple(predicted)))

    assert candidates, "no valid periodic repair found"
    min_period = min(period for period, _bad_idx, _fix, _predicted in candidates)
    candidates = [cand for cand in candidates if cand[0] == min_period]

    signatures = {(bad_idx, fix_color, predicted) for _period, bad_idx, fix_color, predicted in candidates}
    assert len(signatures) == 1, "period repair is ambiguous"
    bad_idx, fix_color, predicted = next(iter(signatures))
    return int(bad_idx), int(fix_color), list(predicted)


def _repair_grid(grid: np.ndarray, parsed: dict) -> np.ndarray:
    axis = int(parsed["axis"])
    span_start, span_end = parsed["span"]
    bands = parsed["bands"]
    colors = [int(color) for _start, _end, color in bands]
    bad_idx, fix_color, _predicted = _infer_repair(colors)
    band_start, band_end, _old_color = bands[bad_idx]

    out = grid.copy()
    if axis == 0:
        out[band_start : band_end + 1, span_start : span_end + 1] = fix_color
        assert band_start > 0 and band_end + 1 < grid.shape[0], "repaired stripe needs markable neighbors"
        assert np.all(grid[band_start - 1, span_start : span_end + 1] == BG)
        assert np.all(grid[band_end + 1, span_start : span_end + 1] == BG)
        out[band_start - 1, span_start : span_end + 1] = MAROON
        out[band_end + 1, span_start : span_end + 1] = MAROON
    else:
        out[span_start : span_end + 1, band_start : band_end + 1] = fix_color
        assert band_start > 0 and band_end + 1 < grid.shape[1], "repaired stripe needs markable neighbors"
        assert np.all(grid[span_start : span_end + 1, band_start - 1] == BG)
        assert np.all(grid[span_start : span_end + 1, band_end + 1] == BG)
        out[span_start : span_end + 1, band_start - 1] = MAROON
        out[span_start : span_end + 1, band_end + 1] = MAROON

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single corrupted stripe in a periodic stripe layout and mark it with maroon borders.

    Input structure:
    - Background is black.
    - The scene contains equal-width horizontal or vertical solid-color stripes.
    - Neighboring stripes are separated by one-pixel black rows or columns.
    - Stripe colors repeat with a hidden period of length 3 to 5, except for one corrupted stripe.

    Rule:
    - Infer the smallest valid period in 3..5.
    - Replace the corrupted stripe with its predicted color.
    - Recolor the two separator lines immediately adjacent to that repaired stripe with maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates = []
    for axis in (0, 1):
        parsed = _parse_orientation(g, axis)
        if parsed is not None:
            candidates.append(parsed)

    assert len(candidates) == 1, "expected a unique stripe orientation"
    return _repair_grid(g, candidates[0])


def _render_stripes(
    stripe_colors: list[int],
    *,
    orientation: str,
    thickness: int,
    length: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
) -> np.ndarray:
    stripe_count = len(stripe_colors)
    if orientation == "horizontal":
        height = top + bottom + stripe_count * thickness + (stripe_count - 1)
        width = left + right + length
        grid = np.full((height, width), BG, dtype=int)
        for i, color in enumerate(stripe_colors):
            r0 = top + i * (thickness + 1)
            grid[r0 : r0 + thickness, left : left + length] = int(color)
        return grid

    height = top + bottom + length
    width = left + right + stripe_count * thickness + (stripe_count - 1)
    grid = np.full((height, width), BG, dtype=int)
    for i, color in enumerate(stripe_colors):
        c0 = left + i * (thickness + 1)
        grid[top : top + length, c0 : c0 + thickness] = int(color)
    return grid


def _render_expected(
    stripe_colors: list[int],
    *,
    orientation: str,
    thickness: int,
    length: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    bad_idx: int,
) -> np.ndarray:
    out = _render_stripes(
        stripe_colors,
        orientation=orientation,
        thickness=thickness,
        length=length,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )
    if orientation == "horizontal":
        r0 = top + bad_idx * (thickness + 1)
        out[r0 - 1, left : left + length] = MAROON
        out[r0 + thickness, left : left + length] = MAROON
    else:
        c0 = left + bad_idx * (thickness + 1)
        out[top : top + length, c0 - 1] = MAROON
        out[top : top + length, c0 + thickness] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIODS)
    orientation = random.choice(("horizontal", "vertical"))

    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)

    min_stripes = 2 * period
    max_stripes = min(12, 2 * period + 3)
    stripe_count = random.randint(min_stripes, max_stripes)

    if orientation == "horizontal":
        max_thickness = (MAX_GRID_SIZE - top - bottom - (stripe_count - 1)) // stripe_count
    else:
        max_thickness = (MAX_GRID_SIZE - left - right - (stripe_count - 1)) // stripe_count
    if max_thickness < 1:
        return None
    thickness = random.randint(1, min(3, max_thickness))

    if orientation == "horizontal":
        max_length = MAX_GRID_SIZE - left - right
    else:
        max_length = MAX_GRID_SIZE - top - bottom
    if max_length < 6:
        return None
    length = random.randint(6, min(24, max_length))

    pattern = random.sample(INPUT_COLORS, period)
    expected_colors = [int(pattern[i % period]) for i in range(stripe_count)]
    bad_idx = random.randint(1, stripe_count - 2)

    wrong_options = [color for color in INPUT_COLORS if color not in pattern]
    if not wrong_options:
        return None
    observed_colors = expected_colors[:]
    observed_colors[bad_idx] = int(random.choice(wrong_options))

    inp = _render_stripes(
        observed_colors,
        orientation=orientation,
        thickness=thickness,
        length=length,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )
    expected = _render_expected(
        expected_colors,
        orientation=orientation,
        thickness=thickness,
        length=length,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        bad_idx=bad_idx,
    )

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stripe-repair input")
