import random
from collections import Counter

import numpy as np


BG = 0
GEN_RETRIES = 400
MIN_SIZE = 8
MAX_SIZE = 15  # output doubles, so every side must stay <= 30
MIN_BRANCHES = 4

_SEEN_INPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()
_GEN_CALL_COUNT = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the uniquely rare interrupting color, repair the broken diagonal stripe, and extend it to a doubled canvas.

    Input structure:
    - The background is black (0).
    - Non-background cells lie on one width-1 down-right diagonal path with fixed offset `c - r`.
    - Two main foreground colors alternate by row phase, while 2-3 path cells have been replaced by a third color.

    Rule:
    - Count non-background colors and erase the uniquely least frequent one to black.
    - From the cleaned diagonal path, recover the stripe's fixed offset and the smallest row period that assigns one color to each row phase.
    - Repaint the complete stripe and then draw the same offset/period rule on a `2H x 2W` black canvas, placing row `r` at column `(r + offset) mod (2W)`.

    Key insight:
    - The rare color is not an output color; it is an interruption embedded in the stripe path, so removing it is necessary before the diagonal period can be inferred and extended.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    minority = _unique_minority_color(g)
    cleaned = g.copy()
    cleaned[cleaned == minority] = BG

    offset, period, phase_to_color = _infer_repaired_rule(cleaned)

    height, width = map(int, g.shape)
    out = np.full((2 * height, 2 * width), BG, dtype=int)
    for row in range(2 * height):
        col = (row + offset) % (2 * width)
        out[row, col] = phase_to_color[row % period]
    return out


def _unique_minority_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BG)
    assert len(counts) >= 3, "expected two stripe colors plus one interrupting color"
    ordered = counts.most_common()
    minority_color, minority_count = ordered[-1]
    second_minority_count = ordered[-2][1]
    assert minority_count < second_minority_count, "least frequent color must be unique"
    return int(minority_color)


def _path_rows(height: int, width: int, offset: int) -> list[int]:
    return [r for r in range(height) if 0 <= r + offset < width]


def _infer_repaired_rule(cleaned: np.ndarray) -> tuple[int, int, dict[int, int]]:
    coords = np.argwhere(cleaned != BG)
    assert coords.size > 0, "expected remaining stripe pixels after erasing the rare color"

    offsets = {int(c - r) for r, c in coords}
    assert len(offsets) == 1, "remaining stripe must lie on a single diagonal offset"
    offset = offsets.pop()

    rows_seen = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        assert cc - rr == offset
        color = int(cleaned[rr, cc])
        assert rr not in rows_seen, "stripe width must be one pixel per row"
        rows_seen[rr] = color

    candidate_rows = _path_rows(*cleaned.shape, offset)
    assert candidate_rows, "expected at least one valid stripe row"
    observed = sorted(rows_seen.items())
    observed_colors = {color for _row, color in observed}
    assert len(observed_colors) >= 2, "the repaired stripe should expose at least two phases"

    period, phase_to_color = _learn_minimal_period(observed, candidate_rows)
    assert period >= 2, "expected a nontrivial periodic stripe"
    return offset, period, phase_to_color


def _learn_minimal_period(
    observed: list[tuple[int, int]],
    candidate_rows: list[int],
) -> tuple[int, dict[int, int]]:
    max_period = len(candidate_rows)
    for period in range(1, max_period + 1):
        phase_to_color: dict[int, int] = {}
        ok = True
        for row, color in observed:
            phase = row % period
            prev = phase_to_color.get(phase)
            if prev is None:
                phase_to_color[phase] = color
            elif prev != color:
                ok = False
                break
        if not ok:
            continue

        active_phases = {row % period for row in candidate_rows}
        if any(phase not in phase_to_color for phase in active_phases):
            continue
        return period, phase_to_color

    raise AssertionError("could not infer a consistent row period")


def _full_band(shape: tuple[int, int], offset: int, phase_to_color: dict[int, int], period: int) -> np.ndarray:
    height, width = map(int, shape)
    band = np.full((height, width), BG, dtype=int)
    for row in _path_rows(height, width, offset):
        band[row, row + offset] = phase_to_color[row % period]
    return band


def _input_period_with_noise(grid: np.ndarray, offset: int) -> int:
    coords = np.argwhere(grid != BG)
    observed = sorted((int(r), int(grid[int(r), int(c)])) for r, c in coords if int(c - r) == offset)
    candidate_rows = _path_rows(*grid.shape, offset)
    period, _phase_to_color = _learn_minimal_period(observed, candidate_rows)
    return period


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=np.int8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _expected_output(height: int, width: int, offset: int, phase_to_color: dict[int, int], period: int) -> np.ndarray:
    out = np.full((2 * height, 2 * width), BG, dtype=int)
    for row in range(2 * height):
        out[row, (row + offset) % (2 * width)] = phase_to_color[row % period]
    return out


def _choose_dimensions(branch: int) -> tuple[int, int, int]:
    if branch == 3:
        height = random.randint(8, 13)
        width = random.randint(height + 1, MAX_SIZE)
        offset = random.randint(1, width - height)
    else:
        height = random.randint(MIN_SIZE, MAX_SIZE)
        width = random.randint(height, MAX_SIZE)
        max_offset = width - height
        if branch == 2 and max_offset > 0 and random.random() < 0.7:
            offset = random.randint(1, max_offset)
        else:
            offset = random.randint(0, max_offset)
    return height, width, offset


def _choose_noise_rows(height: int, branch: int) -> list[int] | None:
    rows = list(range(height))
    interior = list(range(1, height - 1))

    for _ in range(120):
        if branch == 0:
            k = random.choice((2, 3))
            picked = sorted(random.sample(interior, k))
            if any(b - a <= 1 for a, b in zip(picked, picked[1:])):
                continue
        elif branch == 1:
            start = random.randint(1, height - 3)
            picked = [start, start + 1]
            if random.random() < 0.75:
                extra = [r for r in interior if abs(r - start) > 1 and abs(r - (start + 1)) > 1]
                if not extra:
                    continue
                picked.append(random.choice(extra))
            picked = sorted(picked)
        elif branch == 2:
            edge = random.choice((0, height - 1))
            k = random.choice((2, 3))
            rest_pool = [r for r in rows if r != edge]
            picked = [edge] + random.sample(rest_pool, k - 1)
            picked = sorted(set(picked))
            if len(picked) != k:
                continue
        else:
            k = 3
            picked = sorted(random.sample(rows, k))
            if len({r % 2 for r in picked}) < 2:
                continue

        if len({r % 2 for r in picked}) < 2:
            continue
        return picked
    return None


def _design_is_valid(height: int, width: int, offset: int, phase_to_color: dict[int, int], noise_rows: list[int], noise_color: int) -> bool:
    input_grid = _full_band((height, width), offset, phase_to_color, 2)
    for row in noise_rows:
        input_grid[row, row + offset] = noise_color

    counts = Counter(int(v) for v in input_grid.ravel() if int(v) != BG)
    ordered = counts.most_common()
    if len(ordered) != 3:
        return False
    if ordered[-1][0] != noise_color or ordered[-1][1] != len(noise_rows):
        return False
    if not (ordered[-1][1] < ordered[-2][1] < ordered[0][1] or ordered[-1][1] < ordered[-2][1] == ordered[0][1]):
        return False

    cleaned = input_grid.copy()
    cleaned[cleaned == noise_color] = BG
    try:
        inferred_offset, inferred_period, inferred_phases = _infer_repaired_rule(cleaned)
    except AssertionError:
        return False
    if inferred_offset != offset or inferred_period != 2:
        return False
    if inferred_phases != phase_to_color:
        return False

    raw_period = _input_period_with_noise(input_grid, offset)
    if raw_period == 2:
        return False

    for row in noise_rows:
        if row + offset < 0 or row + offset >= width:
            return False
        if cleaned[row, row + offset] != BG:
            return False

    repaired = _full_band((height, width), offset, phase_to_color, 2)
    if np.array_equal(repaired, input_grid):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(GEN_RETRIES):
        branch = _GEN_CALL_COUNT % MIN_BRANCHES
        _GEN_CALL_COUNT += 1

        height, width, offset = _choose_dimensions(branch)
        colors = random.sample(range(1, 10), 3)
        phase_to_color = {0: colors[0], 1: colors[1]}
        noise_color = colors[2]

        noise_rows = _choose_noise_rows(height, branch)
        if noise_rows is None:
            continue
        if not _design_is_valid(height, width, offset, phase_to_color, noise_rows, noise_color):
            continue

        grid = _full_band((height, width), offset, phase_to_color, 2)
        for row in noise_rows:
            grid[row, row + offset] = noise_color

        expected = _expected_output(height, width, offset, phase_to_color, 2)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        input_sig = _signature(grid)
        output_sig = _signature(expected)
        if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
            continue

        _SEEN_INPUT_SIGS.add(input_sig)
        _SEEN_OUTPUT_SIGS.add(output_sig)
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
