from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import py_compile
import random

import numpy as np


BACKGROUND = 0
CROSS_COLOR = 2
BAR_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
MIN_QUADRANT = 4
MAX_QUADRANT = 14
MAX_GENERATION_ATTEMPTS = 2000
SELF_TEST_TRIALS = 100


@dataclass(frozen=True)
class BarSpec:
    row: int
    length: int
    color: int


@dataclass(frozen=True)
class Case:
    input_grid: np.ndarray
    output_grid: np.ndarray
    signature: tuple[object, ...]


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height == width and height % 2 == 1, "expected an odd square grid"
    center = height // 2
    cross_color = int(g[center, center])
    assert cross_color != BACKGROUND, "cross color must be non-background"
    assert np.all(g[center, :] == cross_color), "center row must be a full cross arm"
    assert np.all(g[:, center] == cross_color), "center column must be a full cross arm"
    return center, cross_color


def _extract_source_bars(grid: np.ndarray) -> list[tuple[int, int]]:
    center, cross_color = _find_cross(grid)
    upper_left = np.asarray(grid[:center, :center], dtype=int)

    bars: list[tuple[int, int]] = []
    for row_index, row in enumerate(upper_left):
        if np.all(row == BACKGROUND):
            continue
        bar_color = int(row[0])
        assert bar_color not in (BACKGROUND, cross_color), "bars must use non-cross colors"
        length = 0
        while length < center and int(row[length]) == bar_color:
            length += 1
        assert np.all(row[length:] == BACKGROUND), "bars must be contiguous left-anchored runs"
        bars.append((row_index, length))

    assert 2 <= len(bars) <= 3, "expected 2 or 3 source bars"
    return bars


def _fill_periodic_row(
    out: np.ndarray,
    row: int,
    col_start: int,
    col_end: int,
    length: int,
    anchor_left: bool,
    color: int,
) -> None:
    if anchor_left:
        for start in range(col_start, col_end, length):
            out[row, start : min(start + length, col_end)] = color
        return

    end = col_end
    while end > col_start:
        start = max(col_start, end - length)
        out[row, start:end] = color
        end -= length


def _render_from_bars(quadrant_size: int, bars: list[tuple[int, int]], color: int) -> np.ndarray:
    output_size = quadrant_size * 2
    out = np.full((output_size, output_size), BACKGROUND, dtype=int)

    for source_row, length in bars:
        mirrored_row = output_size - 1 - source_row

        _fill_periodic_row(out, source_row, 0, quadrant_size, length, True, color)
        _fill_periodic_row(out, source_row, quadrant_size, output_size, length, False, color)
        _fill_periodic_row(out, mirrored_row, 0, quadrant_size, length, True, color)
        _fill_periodic_row(out, mirrored_row, quadrant_size, output_size, length, False, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove a central cross, mirror upper-left color bars, and tile each mirrored bar by its length.

    Input structure:
    - The input is an odd square grid split into four equal quadrants by a full central row and
      full central column of the same non-black cross color.
    - Only the upper-left quadrant contains foreground content: 2 or 3 horizontal bars, each on
      its own row, left-anchored, contiguous, and otherwise surrounded by black background.
    - The other three quadrants are blank in the input.

    Rule:
    - Delete the central cross row and column, shrinking the grid from (2N+1)x(2N+1) to (2N)x(2N).
    - Recolor every source bar to the cross color and mirror the upper-left pattern into the other
      three quadrants across the removed cross axes.
    - Inside each quadrant, extend every mirrored bar along its row by copying segments whose width
      equals the original bar length; the source bar length is both the mirrored motif and the
      repetition period, and the last partial copy is clipped at the quadrant edge.

    Key insight:
    - The central cross simultaneously defines the removable axes, the four quadrant boundaries, and
      the output color, while each source bar's length controls both where the mirrored seed appears
      and how that row is periodically filled inside every quadrant.
    """
    g = np.asarray(grid, dtype=int)
    center, cross_color = _find_cross(g)
    bars = _extract_source_bars(g)
    return _render_from_bars(center, bars, cross_color)


def _choose_rows(rng: random.Random, quadrant_size: int, count: int) -> list[int]:
    candidates = list(range(quadrant_size))
    for _ in range(1000):
        rows = sorted(rng.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    raise RuntimeError("failed to choose separated bar rows")


def _make_case(rng: random.Random) -> Case:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        quadrant_size = rng.randint(MIN_QUADRANT, MAX_QUADRANT)
        max_length = min(3, quadrant_size // 2)
        bar_count = 3 if max_length >= 3 and rng.random() < 0.6 else 2
        if max_length < bar_count:
            continue

        lengths = rng.sample(list(range(1, max_length + 1)), bar_count)
        rows = _choose_rows(rng, quadrant_size, bar_count)
        colors = rng.sample(BAR_COLORS, bar_count)
        specs = [BarSpec(row=row, length=length, color=color) for row, length, color in zip(rows, lengths, colors)]

        grid_size = quadrant_size * 2 + 1
        center = quadrant_size
        input_grid = np.full((grid_size, grid_size), BACKGROUND, dtype=int)
        input_grid[center, :] = CROSS_COLOR
        input_grid[:, center] = CROSS_COLOR
        for spec in specs:
            input_grid[spec.row, : spec.length] = spec.color

        output_grid = _render_from_bars(
            quadrant_size,
            [(spec.row, spec.length) for spec in sorted(specs, key=lambda spec: spec.row)],
            CROSS_COLOR,
        )
        signature = (
            quadrant_size,
            tuple((spec.row, spec.length, spec.color) for spec in sorted(specs, key=lambda spec: spec.row)),
        )
        return Case(input_grid=input_grid, output_grid=output_grid, signature=signature)

    raise RuntimeError("failed to generate a valid cross-bar puzzle")


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    return _make_case(random.Random(seed)).input_grid


def _validate_random_cases(trials: int = SELF_TEST_TRIALS) -> dict[str, int]:
    variants: set[tuple[object, ...]] = set()
    for trial in range(trials):
        case = _make_case(random.Random(20_000 + trial))
        assert case.input_grid.shape[0] <= 30 and case.input_grid.shape[1] <= 30
        predicted = main(case.input_grid)
        assert predicted.shape == case.output_grid.shape
        assert np.array_equal(predicted, case.output_grid), f"mismatch on trial {trial}"
        variants.add(case.signature)

    assert len(variants) >= 20, f"expected at least 20 variants, got {len(variants)}"
    return {"trials": trials, "variants": len(variants)}


def _compile_self() -> None:
    py_compile.compile(str(Path(__file__)), doraise=True)


def _cleanup_pycache() -> None:
    pycache_dir = Path(__file__).with_name("__pycache__")
    if not pycache_dir.exists():
        return

    stem = Path(__file__).stem
    for pyc_path in pycache_dir.glob(f"{stem}.*.pyc"):
        pyc_path.unlink()

    try:
        next(pycache_dir.iterdir())
    except StopIteration:
        pycache_dir.rmdir()


if __name__ == "__main__":
    stats = _validate_random_cases()
    _compile_self()
    _cleanup_pycache()
    print(f"validated={stats['trials']} variants={stats['variants']}")
