from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ISOLATED_COLORS = (BLUE, GREEN, YELLOW)

GRID_ROWS = (7, 8)
GRID_COLS = (7, 8)
SEGMENT_LENGTHS = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _scale2x(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _segment_info(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]]:
    red_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    assert 2 <= len(red_coords) <= 4, "expected one red anti-diagonal segment of length 2..4"

    diag_sums = {r + c for r, c in red_coords}
    assert len(diag_sums) == 1, "red cells must lie on one anti-diagonal"
    diag_sum = next(iter(diag_sums))

    cols = sorted(c for _r, c in red_coords)
    assert cols == list(range(cols[0], cols[-1] + 1)), "red segment must be contiguous"

    expected = {(diag_sum - c, c) for c in cols}
    assert expected == set(red_coords), "red cells must form a one-pixel-wide anti-diagonal run"

    top_right = (diag_sum - cols[-1], cols[-1])
    return diag_sum, len(cols), top_right, sorted(red_coords)


def _input_seed_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid != BLACK) & (grid != RED)):
        color = int(grid[r, c])
        seeds.append((int(r), int(c), color))
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the whole picture up, erase the thick red anti-diagonal, and redraw a longer thin one.

    Input structure:
    - The grid has a black background.
    - There is exactly one red one-pixel-wide anti-diagonal segment.
    - There are 1 to 3 isolated non-red single pixels in blue, green, or yellow.

    Rule:
    - First scale the entire grid by a factor of 2 so every input cell becomes a solid 2x2 block.
    - Detect the red anti-diagonal segment, measure its original length L, and erase the red 2x2-block stripe created by scaling it.
    - From the cell one step up-right of that stripe's upper-right end, draw a red thin anti-diagonal of length L+1.
    - Keep the scaled 2x2 blocks of the isolated pixels unchanged.

    Key insight:
    - The whole scene is uniformly enlarged, but the red segment is selectively "unscaled": its thick enlarged stripe is removed and replaced by a thinner longer anti-diagonal whose size still depends on the original segment length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _diag_sum, length, (top_r, top_c), red_coords = _segment_info(g)

    out = _scale2x(g)
    for r, c in red_coords:
        out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = BLACK

    start_r = 2 * top_r - 1
    start_c = 2 * top_c + 2
    for step in range(length + 1):
        rr = start_r - step
        cc = start_c + step
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
        out[rr, cc] = RED

    return out


def _segment_candidates(height: int, width: int, length: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for top_r in range(height):
        for top_c in range(width):
            coords = [(top_r + step, top_c - step) for step in range(length)]
            if any(not (0 <= r < height and 0 <= c < width) for r, c in coords):
                continue

            start = (2 * top_r - 1, 2 * top_c + 2)
            line = [(start[0] - step, start[1] + step) for step in range(length + 1)]
            if any(not (0 <= r < 2 * height and 0 <= c < 2 * width) for r, c in line):
                continue

            candidates.append(
                {
                    "coords": coords,
                    "top_right": (top_r, top_c),
                    "line": line,
                }
            )
    return candidates


def _valid_seed_positions(
    height: int,
    width: int,
    segment_coords: list[tuple[int, int]],
    line: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    segment_set = set(segment_coords)
    line_set = set(line)
    valid: list[tuple[int, int]] = []

    for r in range(height):
        for c in range(width):
            if (r, c) in segment_set:
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in segment_coords):
                continue

            block = {
                (2 * r, 2 * c),
                (2 * r + 1, 2 * c),
                (2 * r, 2 * c + 1),
                (2 * r + 1, 2 * c + 1),
            }
            if block & line_set:
                continue
            valid.append((r, c))

    return valid


def _pick_separated_cells(
    cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    order = cells[:]
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []

    def search(start: int) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen.copy()
        needed = count - len(chosen)
        if len(order) - start < needed:
            return None

        for idx in range(start, len(order)):
            r, c = order[idx]
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            found = search(idx + 1)
            if found is not None:
                return found
            chosen.pop()
        return None

    return search(0)


def _validate_input(grid: np.ndarray) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GREEN, YELLOW})

    info = _segment_info(g)
    _diag_sum, _length, _top_right, red_coords = info
    seeds = _input_seed_cells(g)

    assert 1 <= len(seeds) <= 3, "expected 1..3 isolated pixels"
    seed_colors = [color for _r, _c, color in seeds]
    assert len(seed_colors) == len(set(seed_colors)), "isolated pixel colors must be distinct"
    assert all(color in ISOLATED_COLORS for color in seed_colors)

    for i, (r, c, _color) in enumerate(seeds):
        assert all(max(abs(r - sr), abs(c - sc)) > 1 for sr, sc in red_coords)
        for rr, cc, _other in seeds[i + 1 :]:
            assert max(abs(r - rr), abs(c - cc)) > 1

    return info


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    _diag_sum, length, (top_r, top_c), red_coords = _validate_input(inp)
    assert out.shape == (inp.shape[0] * 2, inp.shape[1] * 2)
    assert max(out.shape) <= 30

    expected = np.zeros_like(out)
    for r, c, color in _input_seed_cells(inp):
        expected[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = color

    line_points = [(2 * top_r - 1 - step, 2 * top_c + 2 + step) for step in range(length + 1)]
    for r, c in line_points:
        expected[r, c] = RED

    for r, c in red_coords:
        assert np.all(out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] == BLACK)

    assert np.array_equal(out, expected), "output must match the described scale/erase/redraw rule"
    assert int(np.count_nonzero(out == RED)) == length + 1


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    length = random.choice(SEGMENT_LENGTHS)
    seed_count = random.randint(1, 3)

    candidates = _segment_candidates(height, width, length)
    random.shuffle(candidates)
    if not candidates:
        return None

    colors = random.sample(ISOLATED_COLORS, seed_count)
    for candidate in candidates:
        segment_coords = candidate["coords"]
        line = candidate["line"]
        valid_cells = _valid_seed_positions(height, width, segment_coords, line)
        chosen_cells = _pick_separated_cells(valid_cells, seed_count)
        if chosen_cells is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for r, c in segment_coords:
            grid[r, c] = RED
        for (r, c), color in zip(chosen_cells, colors):
            grid[r, c] = color

        out = main(grid)
        _validate_output_semantics(grid, out)
        if np.array_equal(out, _scale2x(grid)):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
