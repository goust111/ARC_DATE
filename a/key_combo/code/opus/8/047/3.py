from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import py_compile
import random
import shutil

import numpy as np


STEP_LEN = 2
MIN_GRID = 15
MAX_GRID = 24
MAX_GENERATION_ATTEMPTS = 2000
SELF_TEST_TRIALS = 100


@dataclass(frozen=True)
class Case:
    input_grid: np.ndarray
    output_grid: np.ndarray
    signature: tuple[object, ...]


def _segment_cells(
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    length: int,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for k in range(length):
        rr = start_r + dr * k
        cc = start_c + dc * k
        if 0 <= rr < height and 0 <= cc < width:
            cells.add((rr, cc))
    return cells


def _arm_cells(
    anchor: tuple[int, int],
    first_dir: tuple[int, int],
    second_dir: tuple[int, int],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    r, c = anchor
    cells: set[tuple[int, int]] = set()
    while 0 <= r < height and 0 <= c < width:
        cells |= _segment_cells(r, c, first_dir[0], first_dir[1], STEP_LEN, height, width)
        r += first_dir[0] * STEP_LEN
        c += first_dir[1] * STEP_LEN
        cells |= _segment_cells(r, c, second_dir[0], second_dir[1], STEP_LEN, height, width)
        r += second_dir[0] * STEP_LEN
        c += second_dir[1] * STEP_LEN
    return cells


def _staircase_cells(anchor: tuple[int, int], height: int, width: int) -> set[tuple[int, int]]:
    return _arm_cells(anchor, (-1, 0), (0, 1), height, width) | _arm_cells(
        anchor, (1, 0), (0, -1), height, width
    )


def _detect_legend_size(grid: np.ndarray) -> int:
    height, width = grid.shape
    max_size = min(height, width)
    size = 0
    for candidate in range(1, max_size + 1):
        if np.all(grid[:candidate, :candidate] != 0):
            size = candidate
        else:
            break
    assert size >= 3, "expected a non-zero legend square in the top-left corner"
    return size


def _build_lookup(legend: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], int]]:
    size = legend.shape[0]
    row_labels = [int(v) for v in legend[1:size, 0]]
    col_labels = [int(v) for v in legend[0, 1:size]]
    assert len(set(row_labels)) == size - 1, "row labels must be distinct anchor colors"
    assert len(set(col_labels)) == size - 1, "column labels must be distinct anchor colors"
    assert set(row_labels) == set(col_labels), "row/column labels must describe the same anchors"
    lookup: dict[tuple[int, int], int] = {}
    for r_idx, row_color in enumerate(row_labels, start=1):
        for c_idx, col_color in enumerate(col_labels, start=1):
            lookup[(row_color, col_color)] = int(legend[r_idx, c_idx])
    return row_labels, col_labels, lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw staircases from every colored anchor and recolor pairwise intersections via a legend.

    Input structure:
    - The background is black.
    - A solid legend square sits in the top-left corner; its first column and first
      row list the anchor colors, and each interior cell stores the output color for
      that ordered anchor-color pair.
    - Outside the legend there are 2 or 3 one-pixel anchors, all in distinct colors.

    Rule:
    - From each anchor, draw one step-2 staircase that alternates up then right, and
      another that alternates down then left.
    - Pixels covered by only one anchor keep that anchor color.
    - Pixels covered by two different anchors are recolored by the legend, using the
      upper anchor as the row key and the lower anchor as the column key.
    - Anchor pixels and the legend itself stay unchanged.

    Key insight:
    - The geometry comes from mirrored staircases, but the color at each crossing is
      not an overwrite rule; it is a legend lookup keyed by the two meeting anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    legend_size = _detect_legend_size(g)
    legend = g[:legend_size, :legend_size]
    label_colors, _, lookup = _build_lookup(legend)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:legend_size, :legend_size] = True

    anchor_points = np.argwhere((g != 0) & (~legend_mask))
    assert len(anchor_points) == len(label_colors), "expected one anchor per legend color"

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r, c in anchor_points:
        color = int(g[r, c])
        assert color in label_colors, "non-legend foreground must be an anchor color"
        assert color not in anchors_by_color, "anchor colors must be unique"
        anchors_by_color[color] = (int(r), int(c))
    assert set(anchors_by_color) == set(label_colors), "legend labels and anchors must match"

    covers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, anchor in anchors_by_color.items():
        for cell in _staircase_cells(anchor, height, width):
            if not legend_mask[cell]:
                covers[cell].add(color)

    out = np.zeros_like(g)
    out[:legend_size, :legend_size] = legend
    for cell, colors in covers.items():
        if len(colors) == 1:
            out[cell] = next(iter(colors))
            continue
        assert len(colors) == 2, "generator guarantees at most pairwise intersections"
        color_a, color_b = sorted(
            colors,
            key=lambda color: (
                anchors_by_color[color][0],
                anchors_by_color[color][1],
                color,
            ),
        )
        out[cell] = lookup[(color_a, color_b)]

    for color, (r, c) in anchors_by_color.items():
        out[r, c] = color
    return out


def _choose_anchor_positions(
    rng: random.Random,
    height: int,
    width: int,
    legend_size: int,
    anchor_count: int,
) -> list[tuple[int, int]]:
    candidates = [
        (r, c)
        for r in range(legend_size + 2, height - 2)
        for c in range(2, width - 2)
    ]
    for _ in range(1000):
        positions = rng.sample(candidates, anchor_count)
        rows = [r for r, _ in positions]
        if len(set(rows)) != anchor_count:
            continue
        ok = True
        for idx, (r1, c1) in enumerate(positions):
            for r2, c2 in positions[idx + 1 :]:
                if max(abs(r1 - r2), abs(c1 - c2)) < 4:
                    ok = False
                    break
                if abs(r1 - r2) + abs(c1 - c2) < 7:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return positions
    raise RuntimeError("failed to pick separated anchor positions")


def _make_case(rng: random.Random) -> Case:
    palette = list(range(1, 10))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        anchor_count = rng.choice([2, 3])
        legend_size = anchor_count + 1
        height = rng.randint(max(MIN_GRID, legend_size + 8), MAX_GRID)
        width = rng.randint(max(MIN_GRID, legend_size + 8), MAX_GRID)
        anchor_colors = rng.sample(palette, anchor_count)
        corner_color = rng.choice(palette)
        row_labels = rng.sample(anchor_colors, anchor_count)
        col_labels = rng.sample(anchor_colors, anchor_count)
        fill_choices = [color for color in palette if color not in anchor_colors]
        legend = np.zeros((legend_size, legend_size), dtype=int)
        legend[0, 0] = corner_color
        legend[1:, 0] = np.array(row_labels, dtype=int)
        legend[0, 1:] = np.array(col_labels, dtype=int)
        for r_idx in range(anchor_count):
            for c_idx in range(anchor_count):
                legend[r_idx + 1, c_idx + 1] = rng.choice(fill_choices)

        positions = _choose_anchor_positions(rng, height, width, legend_size, anchor_count)
        grid = np.zeros((height, width), dtype=int)
        grid[:legend_size, :legend_size] = legend
        anchor_specs = list(zip(anchor_colors, positions))
        for color, (r, c) in anchor_specs:
            grid[r, c] = color

        legend_cells = {(r, c) for r in range(legend_size) for c in range(legend_size)}
        covers: dict[tuple[int, int], set[int]] = defaultdict(set)
        bad = False
        for color, anchor in anchor_specs:
            cells = _staircase_cells(anchor, height, width)
            if cells & legend_cells:
                bad = True
                break
            for cell in cells:
                covers[cell].add(color)
        if bad:
            continue

        if any(len(colors) > 2 for colors in covers.values()):
            continue
        if any(len(covers[pos]) > 1 for _, pos in anchor_specs):
            continue
        overlap_cells = [cell for cell, colors in covers.items() if len(colors) == 2]
        if not overlap_cells:
            continue

        output_grid = main(grid)
        signature = (
            tuple(grid.shape),
            tuple(sorted((int(color), int(r), int(c)) for color, (r, c) in anchor_specs)),
            tuple(row_labels),
            tuple(col_labels),
            tuple(int(v) for v in legend[1:, 1:].ravel()),
            len(overlap_cells),
        )
        return Case(input_grid=grid, output_grid=output_grid, signature=signature)
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    return _make_case(random.Random(seed)).input_grid


def _validate_random_cases(trials: int = SELF_TEST_TRIALS) -> dict[str, int]:
    variants: set[tuple[object, ...]] = set()
    for trial in range(trials):
        case = _make_case(random.Random(10_000 + trial))
        assert case.input_grid.shape == case.output_grid.shape
        assert case.input_grid.shape[0] <= 30 and case.input_grid.shape[1] <= 30
        predicted = main(case.input_grid)
        assert np.array_equal(predicted, case.output_grid), f"mismatch on trial {trial}"
        variants.add(case.signature)
    assert len(variants) >= 20, f"expected at least 20 variants, got {len(variants)}"
    return {"trials": trials, "variants": len(variants)}


def _compile_self() -> None:
    py_compile.compile(str(Path(__file__)), doraise=True)


def _cleanup_pycache() -> None:
    pycache_dir = Path(__file__).with_name("__pycache__")
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)


if __name__ == "__main__":
    stats = _validate_random_cases()
    _compile_self()
    _cleanup_pycache()
    print(f"validated={stats['trials']} variants={stats['variants']}")
