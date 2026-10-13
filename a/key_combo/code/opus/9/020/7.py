from __future__ import annotations

import hashlib
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _arc_utils = _parent / "new" / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.insert(0, str(_arc_utils.parent))
        break
else:
    raise ImportError("could not locate new/solvers/arc_utils.py")

from arc_utils import Color


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        value = int(raw)
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _mask_from_groups(length: int, groups: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in groups:
        mask[start : end + 1] = True
    return mask


def _single_color(values: np.ndarray) -> int:
    unique = np.unique(np.asarray(values, dtype=int))
    assert unique.size == 1
    return int(unique[0])


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(Color.BLACK)

    horizontal_groups = _true_runs(np.all(g != bg, axis=1))
    vertical_groups = _true_runs(np.all(g != bg, axis=0))
    assert horizontal_groups and vertical_groups

    horizontal_mask = _mask_from_groups(h, horizontal_groups)
    vertical_mask = _mask_from_groups(w, vertical_groups)
    free_rows = ~horizontal_mask
    free_cols = ~vertical_mask
    assert bool(free_rows.any()) and bool(free_cols.any())

    horizontal_colors: list[int] = []
    for r0, r1 in horizontal_groups:
        exposed = g[r0 : r1 + 1, free_cols]
        assert exposed.size > 0
        color = _single_color(exposed)
        assert color != bg
        horizontal_colors.append(color)

    vertical_colors: list[int] = []
    for c0, c1 in vertical_groups:
        exposed = g[free_rows, c0 : c1 + 1]
        assert exposed.size > 0
        color = _single_color(exposed)
        assert color != bg
        vertical_colors.append(color)

    assert len(set(horizontal_colors)) == len(horizontal_colors)
    assert len(set(vertical_colors)) == len(vertical_colors)
    assert set(horizontal_colors).isdisjoint(vertical_colors)

    visible = np.empty((len(horizontal_groups), len(vertical_groups)), dtype=int)
    top_is_horizontal = np.zeros_like(visible, dtype=bool)
    for i, (r0, r1) in enumerate(horizontal_groups):
        for j, (c0, c1) in enumerate(vertical_groups):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            color = _single_color(patch)
            h_color = int(horizontal_colors[i])
            v_color = int(vertical_colors[j])
            assert color in (h_color, v_color)
            visible[i, j] = color
            top_is_horizontal[i, j] = color == h_color

    return {
        "horizontal_groups": horizontal_groups,
        "vertical_groups": vertical_groups,
        "horizontal_colors": horizontal_colors,
        "vertical_colors": vertical_colors,
        "visible": visible,
        "top_is_horizontal": top_is_horizontal,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a crossing bar grid into the matrix of hidden colors at each intersection.

    Input structure:
    - Black background surrounds several full-width horizontal color bands and several full-height vertical color bands.
    - Each horizontal band occupies one or two consecutive all-foreground rows, and each vertical band occupies one or two consecutive all-foreground columns.
    - At every horizontal-vertical crossing rectangle, all cells show exactly one of the two band colors, indicating which band is visible on top there.

    Rule:
    - Detect the horizontal bands from top to bottom and the vertical bands from left to right.
    - Recover each band's own color from its exposed non-crossing cells, then inspect every crossing rectangle to determine which band color is visible.
    - Output a compressed matrix whose cell at row i and column j is the color of the band hidden underneath that crossing, not the visible color.

    Key insight:
    - The task combines spatial compression with an information inversion: each large crossing rectangle collapses to one output cell, and that cell encodes the occluded band color rather than the seen surface color.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))

    out = np.empty(info["visible"].shape, dtype=int)
    for i, h_color in enumerate(info["horizontal_colors"]):
        for j, v_color in enumerate(info["vertical_colors"]):
            out[i, j] = int(v_color) if info["visible"][i, j] == int(h_color) else int(h_color)
    return out


def _random_extra_parts(extra: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(extra):
        values[random.randrange(parts)] += 1
    return values


def _sample_axis_groups(count: int, thicknesses: list[int], minimum_extent: int) -> tuple[int, list[tuple[int, int]]]:
    base = sum(thicknesses) + (count + 1)
    length = random.randint(max(base, minimum_extent), min(30, max(base, minimum_extent) + 8))
    gaps = [1] * (count + 1)
    extras = _random_extra_parts(length - base, count + 1)
    gaps = [gap + extra for gap, extra in zip(gaps, extras)]

    groups: list[tuple[int, int]] = []
    pos = gaps[0]
    for idx, thickness in enumerate(thicknesses):
        start = pos
        end = start + thickness - 1
        groups.append((start, end))
        pos = end + 1 + gaps[idx + 1]
    assert pos <= length
    return length, groups


def _sample_visibility(rows: int, cols: int) -> np.ndarray:
    for _ in range(500):
        visible = np.random.randint(0, 2, size=(rows, cols), dtype=int).astype(bool)
        row_sums = visible.sum(axis=1)
        col_sums = visible.sum(axis=0)
        if np.any(row_sums == 0) or np.any(row_sums == cols):
            continue
        if np.any(col_sums == 0) or np.any(col_sums == rows):
            continue
        if len({tuple(map(int, row)) for row in visible}) != rows:
            continue
        if len({tuple(map(int, col)) for col in visible.T}) != cols:
            continue
        return visible
    raise RuntimeError("failed to sample a non-degenerate visibility matrix")


def _render_grid(
    height: int,
    width: int,
    horizontal_groups: list[tuple[int, int]],
    vertical_groups: list[tuple[int, int]],
    horizontal_colors: list[int],
    vertical_colors: list[int],
    top_is_horizontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.full((height, width), int(Color.BLACK), dtype=int)
    for (r0, r1), color in zip(horizontal_groups, horizontal_colors):
        grid[r0 : r1 + 1, :] = int(color)
    for (c0, c1), color in zip(vertical_groups, vertical_colors):
        grid[:, c0 : c1 + 1] = int(color)

    visible = np.empty(top_is_horizontal.shape, dtype=int)
    hidden = np.empty(top_is_horizontal.shape, dtype=int)
    for i, (r0, r1) in enumerate(horizontal_groups):
        for j, (c0, c1) in enumerate(vertical_groups):
            h_color = int(horizontal_colors[i])
            v_color = int(vertical_colors[j])
            if bool(top_is_horizontal[i, j]):
                grid[r0 : r1 + 1, c0 : c1 + 1] = h_color
                visible[i, j] = h_color
                hidden[i, j] = v_color
            else:
                grid[r0 : r1 + 1, c0 : c1 + 1] = v_color
                visible[i, j] = v_color
                hidden[i, j] = h_color
    return grid, visible, hidden


def _array_sig(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    spec: dict,
) -> bool:
    try:
        info = _analyze_grid(grid)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if info["horizontal_groups"] != spec["horizontal_groups"]:
        return False
    if info["vertical_groups"] != spec["vertical_groups"]:
        return False
    if info["horizontal_colors"] != spec["horizontal_colors"]:
        return False
    if info["vertical_colors"] != spec["vertical_colors"]:
        return False
    if not np.array_equal(info["visible"], spec["visible"]):
        return False
    if not np.array_equal(info["top_is_horizontal"], spec["top_is_horizontal"]):
        return False
    if not np.all(info["visible"] != expected):
        return False
    if np.array_equal(info["visible"], expected):
        return False

    top = spec["top_is_horizontal"]
    if np.any(top.sum(axis=1) == 0) or np.any(top.sum(axis=1) == top.shape[1]):
        return False
    if np.any(top.sum(axis=0) == 0) or np.any(top.sum(axis=0) == top.shape[0]):
        return False

    expected_colors = set(map(int, expected.ravel()))
    required_colors = set(spec["horizontal_colors"]) | set(spec["vertical_colors"])
    if expected_colors != required_colors:
        return False
    return True


_RECENT_INPUT_SIGS: deque[str] = deque(maxlen=64)
_RECENT_OUTPUT_SIGS: deque[str] = deque(maxlen=64)
_GEN_RETRIES = 500


def _generate_candidate() -> tuple[np.ndarray, np.ndarray, dict]:
    horizontal_count = random.randint(3, 4)
    vertical_count = random.randint(3, 4)

    horizontal_thicknesses = [random.randint(1, 2) for _ in range(horizontal_count)]
    vertical_thicknesses = [random.randint(1, 2) for _ in range(vertical_count)]

    height, horizontal_groups = _sample_axis_groups(horizontal_count, horizontal_thicknesses, minimum_extent=14)
    width, vertical_groups = _sample_axis_groups(vertical_count, vertical_thicknesses, minimum_extent=14)

    palette = random.sample(list(range(1, 10)), horizontal_count + vertical_count)
    horizontal_colors = [int(color) for color in palette[:horizontal_count]]
    vertical_colors = [int(color) for color in palette[horizontal_count:]]

    top_is_horizontal = _sample_visibility(horizontal_count, vertical_count)
    grid, visible, hidden = _render_grid(
        height=height,
        width=width,
        horizontal_groups=horizontal_groups,
        vertical_groups=vertical_groups,
        horizontal_colors=horizontal_colors,
        vertical_colors=vertical_colors,
        top_is_horizontal=top_is_horizontal,
    )

    spec = {
        "horizontal_groups": horizontal_groups,
        "vertical_groups": vertical_groups,
        "horizontal_colors": horizontal_colors,
        "vertical_colors": vertical_colors,
        "visible": visible,
        "top_is_horizontal": top_is_horizontal,
    }
    return grid, hidden, spec


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid, expected, spec = _generate_candidate()
        in_sig = _array_sig(grid)
        out_sig = _array_sig(expected)
        if in_sig in _RECENT_INPUT_SIGS or out_sig in _RECENT_OUTPUT_SIGS:
            continue
        if not _validate_instance(grid, expected, spec):
            continue

        _RECENT_INPUT_SIGS.append(in_sig)
        _RECENT_OUTPUT_SIGS.append(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid crossing-compression input")
