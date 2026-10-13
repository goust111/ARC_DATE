from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

STEP_LEN = 3
DIR_BY_COLOR = {
    BLUE: (0, 1),
    RED: (1, 0),
    GREEN: (0, -1),
    YELLOW: (-1, 0),
}

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
PREFIX_LEN_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 500

_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _parse_prefix(grid: np.ndarray) -> list[int]:
    prefix: list[int] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color in DIR_BY_COLOR:
            prefix.append(color)
        else:
            break
    assert PREFIX_LEN_RANGE[0] <= len(prefix) <= PREFIX_LEN_RANGE[1]
    return prefix


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == GREY)
    assert len(starts) == 1, "expected exactly one grey start pixel"
    return int(starts[0, 0]), int(starts[0, 1])


def _execute_program(
    grid: np.ndarray,
    prefix: list[int] | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], bool]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN, YELLOW, GREY}

    dirs = _parse_prefix(g) if prefix is None else list(prefix)
    r, c = _find_start(g)
    out = g.copy()
    segments: list[list[tuple[int, int]]] = []
    stopped_early = False

    for color in dirs:
        dr, dc = DIR_BY_COLOR[int(color)]
        segment: list[tuple[int, int]] = []
        for _ in range(STEP_LEN):
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                stopped_early = True
                segments.append(segment)
                return out, segments, stopped_early
            out[r, c] = ORANGE
            segment.append((r, c))
        segments.append(segment)

    return out, segments, stopped_early


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the left-edge prefix program and draw its fixed-length turning path from the grey start.

    Input structure:
    - The background is black.
    - Column 0 starts with one continuous top-down prefix of 3 to 6 colored pixels.
    - Prefix colors are blue, red, green, and yellow, and a single grey pixel marks the start.

    Rule:
    - Read the prefix from top to bottom and map colors to directions: blue=right, red=down, green=left, yellow=up.
    - Starting from the grey pixel, follow each instruction in order and draw exactly 3 orange cells in that direction.
    - The grey start itself stays unchanged; drawing begins at the first stepped cell.
    - If any step would leave the grid, draw only the in-bounds part of that segment and stop without executing later instructions.

    Key insight:
    - The left column is not decoration or a lookup table; it is a small program whose color order must be executed as a sequence of turn commands.
    """
    out, _segments, _stopped_early = _execute_program(grid)
    return out


def _segment_probe(
    shape: tuple[int, int],
    start: tuple[int, int],
    color: int,
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str]:
    dr, dc = DIR_BY_COLOR[color]
    r, c = start
    cells: list[tuple[int, int]] = []
    for _ in range(STEP_LEN):
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells, "bound"
        if (r, c) in blocked:
            return cells, "blocked"
        cells.append((r, c))
    return cells, "full"


def _coverage_targets(prefix_len: int, mode: int) -> set[int]:
    patterns = (
        (BLUE, RED, GREEN),
        (RED, GREEN, YELLOW),
        (BLUE, GREEN, YELLOW),
        (BLUE, RED, YELLOW),
        (BLUE, RED, GREEN, YELLOW),
        (BLUE, RED, GREEN, YELLOW),
    )
    target = list(patterns[mode % len(patterns)])
    while len(target) > prefix_len:
        target.pop()
    return set(target)


def _weighted_colors(
    rng: np.random.Generator,
    target_missing: set[int],
) -> list[int]:
    weights = np.array(
        [4.0 if color in target_missing else 1.0 for color in DIR_BY_COLOR],
        dtype=float,
    )
    weights /= weights.sum()
    order = rng.choice(np.array(list(DIR_BY_COLOR), dtype=int), size=4, replace=False, p=weights)
    return [int(color) for color in order]


def _build_path(
    shape: tuple[int, int],
    prefix_len: int,
    start: tuple[int, int],
    force_truncate_last: bool,
    target_colors: set[int],
    rng: np.random.Generator,
) -> tuple[list[int], list[tuple[int, int]]] | None:
    prefix_cells = {(r, 0) for r in range(prefix_len)}
    blocked = set(prefix_cells)
    blocked.add(start)
    path_cells: list[tuple[int, int]] = []
    prefix: list[int] = []
    current = start

    for i in range(prefix_len):
        want_trunc = force_truncate_last and i == prefix_len - 1
        missing = target_colors - set(prefix)
        options = []
        for color in _weighted_colors(rng, missing):
            cells, reason = _segment_probe(shape, current, color, blocked)
            if want_trunc:
                if reason == "bound" and 1 <= len(cells) < STEP_LEN:
                    options.append((color, cells))
            else:
                if reason == "full":
                    options.append((color, cells))
        if not options:
            return None

        chosen_color, chosen_cells = options[int(rng.integers(0, len(options)))]
        prefix.append(int(chosen_color))
        path_cells.extend(chosen_cells)
        blocked.update(chosen_cells)
        current = chosen_cells[-1]

    if len(set(prefix)) < min(prefix_len, 3):
        return None
    if target_colors - set(prefix):
        return None
    return prefix, path_cells


def _make_grid(
    shape: tuple[int, int],
    prefix: list[int],
    start: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, color in enumerate(prefix):
        grid[r, 0] = int(color)
    grid[start] = GREY
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    prefix = _parse_prefix(grid)
    out, segments, stopped_early = _execute_program(grid, prefix)
    start = _find_start(grid)

    assert np.array_equal(out[grid != BLACK], np.where(grid[grid != BLACK] == GREY, GREY, grid[grid != BLACK]))
    assert out[start] == GREY
    assert len(segments) == len(prefix) if not stopped_early else len(prefix) >= len(segments)
    assert all(len(segment) == STEP_LEN for segment in segments[:-1])
    if stopped_early:
        assert len(segments) == len(prefix), "truncation must occur on the final instruction"
        assert 1 <= len(segments[-1]) < STEP_LEN
    else:
        assert all(len(segment) == STEP_LEN for segment in segments)

    painted = {(r, c) for r, c in np.argwhere(out == ORANGE)}
    assert painted == {cell for segment in segments for cell in segment}
    assert not any(cell == start for cell in painted)
    assert all(grid[cell] == BLACK for cell in painted)
    assert len(painted) >= STEP_LEN * 2

    for i, original in enumerate(prefix):
        for alt in DIR_BY_COLOR:
            if alt == original:
                continue
            altered = grid.copy()
            altered[i, 0] = alt
            if np.array_equal(main(altered), out):
                return False

    shortened = grid.copy()
    shortened[len(prefix) - 1, 0] = BLACK
    try:
        shortened_out = main(shortened)
    except AssertionError:
        shortened_out = None
    if shortened_out is not None and np.array_equal(shortened_out, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    global _GEN_CALL_COUNT

    rng = np.random.default_rng()
    mode = _GEN_CALL_COUNT % 6
    _GEN_CALL_COUNT += 1

    height = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    prefix_len = int(rng.integers(PREFIX_LEN_RANGE[0], PREFIX_LEN_RANGE[1] + 1))
    force_truncate_last = mode in {1, 4} or bool(rng.integers(0, 4) == 0)
    target_colors = _coverage_targets(prefix_len, mode)

    row_low = max(prefix_len + 1, 3)
    row_high = height - 3
    col_low = 2
    col_high = width - 3
    if row_low > row_high or col_low > col_high:
        return None

    start = (
        int(rng.integers(row_low, row_high + 1)),
        int(rng.integers(col_low, col_high + 1)),
    )
    built = _build_path((height, width), prefix_len, start, force_truncate_last, target_colors, rng)
    if built is None:
        return None

    prefix, _path_cells = built
    grid = _make_grid((height, width), prefix, start)
    out = main(grid)
    if not np.any(out == ORANGE):
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prefix-ray-turning input")


def _self_check(samples: int = 200) -> None:
    for _ in range(samples):
        grid = generate_input()
        out = main(grid)
        assert isinstance(grid, np.ndarray)
        assert isinstance(out, np.ndarray)
        assert grid.ndim == 2 and out.ndim == 2
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
        assert _validate_instance(grid)


if __name__ == "__main__":
    _self_check()
