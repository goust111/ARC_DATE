from __future__ import annotations

import hashlib

import numpy as np


RED = 2
GREEN = 3
BLUE = 1
ORANGE = 7
BLACK = 0

_RNG = np.random.default_rng()
_MAX_GEN_ATTEMPTS = 600
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _single_pos(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert pts.shape[0] == 1, f"expected exactly one marker of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _source_rows(grid: np.ndarray, source_col: int) -> list[int]:
    blue = np.argwhere(grid == BLUE)
    assert blue.shape[0] >= 1, "expected source sprites"
    rows = [int(r) for r, c in blue if int(c) == int(source_col)]
    assert len(rows) == int(blue.shape[0]), "all blue sprites must lie in the red lane"
    rows.sort(reverse=True)
    return rows


def _spacing(rows_desc: list[int]) -> int:
    assert len(rows_desc) >= 2
    gaps = [int(rows_desc[i] - rows_desc[i + 1]) for i in range(len(rows_desc) - 1)]
    assert len(set(gaps)) == 1, "source sprites must be evenly spaced"
    gap = int(gaps[0])
    assert gap in (2, 3), "source spacing must be 2 or 3"
    return gap


def _paint_wedge(shape: tuple[int, int], target_col: int, count: int) -> np.ndarray:
    height, width = map(int, shape)
    out = np.zeros((height, width), dtype=int)
    base_row = int(height - 2)
    for layer in range(count):
        row = int(base_row - layer)
        radius = int(layer)
        for col in range(target_col - radius, target_col + radius + 1):
            if 0 <= row < height and 0 <= col < width:
                out[row, col] = BLUE if abs(int(col - target_col)) % 2 == 0 else ORANGE
    return out


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    rs, cs = _single_pos(g, RED)
    rg, ct = _single_pos(g, GREEN)
    del rs, rg

    rows = _source_rows(g, cs)
    count = len(rows)
    assert 2 <= count <= 4, "sprite count must be between 2 and 4"
    _spacing(rows)
    assert cs != ct, "source and target lanes must differ"
    assert count - 1 <= ct <= width - count, "target lane must have side space for the wedge"

    out = g.copy()
    out[g == BLUE] = BLACK
    out[g == RED] = BLACK
    out[g == GREEN] = BLACK

    wedge = _paint_wedge(g.shape, ct, count)
    mask = wedge != BLACK
    out[mask] = wedge[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer blue sprites from the red lane to the green lane and redraw them as an alternating-color wedge.

    Input structure:
    - A black background contains exactly one red marker pixel and one green marker pixel, each naming a vertical lane by its column.
    - The red lane contains 2 to 4 blue single-pixel sprites, all in the same column and evenly spaced by 2 or 3 rows.
    - The green lane is otherwise empty, and it has enough horizontal room on both sides for the final wedge.

    Rule:
    - Read the red marker column as the source lane and the green marker column as the target lane.
    - Count the blue source sprites from bottom to top, then erase all blue sprites and both marker pixels.
    - Starting from the second-to-last row, draw exactly that many rows in the target lane as a centered wedge.
    - The bottom wedge row has width 1, the next has width 3, and in every row cells at even distance from the target column are blue while odd distances are orange.

    Key insight:
    - The source lane contributes only its sprite count; that conserved count becomes the wedge height, while the green marker fixes the wedge axis and distance parity fixes the color pattern.
    """
    return _expected_output(grid)


def _sample_lane_columns(width: int, lane_count: int) -> list[int] | None:
    candidates = np.arange(1, width - 1)
    for _ in range(120):
        cols = sorted(int(x) for x in _RNG.choice(candidates, size=lane_count, replace=False))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _choose_rows(height: int, count: int, spacing: int) -> list[int] | None:
    bottom_margin = int(_RNG.integers(1, 3))
    bottom_row = int(height - 1 - bottom_margin)
    rows = [int(bottom_row - spacing * i) for i in range(count)]
    if rows[-1] < 1:
        return None
    return rows


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30

    _single_pos(inp, RED)
    _, ct = _single_pos(inp, GREEN)
    _, cs = _single_pos(inp, RED)
    rows = _source_rows(inp, cs)
    count = len(rows)
    gap = _spacing(rows)
    assert count in (2, 3, 4)
    assert gap in (2, 3)
    assert np.all(inp[:, ct][:-1] == BLACK), "target lane must be empty above its marker"

    expected = _expected_output(inp)
    assert np.array_equal(out, expected), "main() must match the rule-defined output"
    assert not np.any(out == RED)
    assert not np.any(out == GREEN)
    for row in rows:
        assert int(out[row, cs]) != BLUE, "original source sprite positions must be cleared"

    base_row = inp.shape[0] - 2
    wedge_pixels = 0
    for layer in range(count):
        row = int(base_row - layer)
        radius = int(layer)
        wedge_pixels += 2 * radius + 1
        segment = out[row, ct - radius : ct + radius + 1]
        assert segment.size == 2 * radius + 1
        for idx, value in enumerate(segment):
            dist = abs(idx - radius)
            want = BLUE if dist % 2 == 0 else ORANGE
            assert int(value) == want
    assert int(np.count_nonzero(out)) == wedge_pixels, "output must contain only the wedge"


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(13, 20))
    width = int(_RNG.integers(13, 21))
    lane_count = int(_RNG.integers(3, 6))
    count = int(_RNG.integers(2, 5))
    spacing = int(_RNG.choice([2, 3]))

    lane_cols = _sample_lane_columns(width, lane_count)
    if lane_cols is None:
        return None

    target_candidates = [c for c in lane_cols if count - 1 <= c <= width - count]
    if not target_candidates:
        return None
    target_col = int(_RNG.choice(target_candidates))

    source_candidates = [c for c in lane_cols if c != target_col]
    if not source_candidates:
        return None

    left_sources = [c for c in source_candidates if c < target_col]
    right_sources = [c for c in source_candidates if c > target_col]
    if left_sources and right_sources:
        source_pool = left_sources if bool(_RNG.integers(0, 2)) else right_sources
    else:
        source_pool = source_candidates
    source_col = int(_RNG.choice(source_pool))

    rows = _choose_rows(height, count, spacing)
    if rows is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[height - 1, source_col] = RED
    grid[height - 1, target_col] = GREEN
    for row in rows:
        grid[row, source_col] = BLUE

    out = main(grid)
    if any(int(out[row, source_col]) != BLACK for row in rows):
        return None
    _validate_instance(grid, out)

    input_sig = _sig(grid)
    output_sig = _sig(out)
    if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lane-transfer wedge input")
