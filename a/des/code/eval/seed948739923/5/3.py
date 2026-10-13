from __future__ import annotations

import numpy as np


BLACK = 0
ORANGE = 7
AZURE = 8

_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_run_shifts(line: np.ndarray, color: int) -> np.ndarray:
    shifts = np.zeros(int(line.shape[0]), dtype=int)
    idx = 0
    while idx < int(line.shape[0]):
        value = int(line[idx])
        if value == int(color):
            end = idx
            while end + 1 < int(line.shape[0]) and int(line[end + 1]) == int(color):
                end += 1
            run_len = int(end - idx + 1)
            shifts[idx : end + 1] = run_len
            idx = end + 1
            continue
        assert value == BLACK, "instruction strips may only contain background or the marker color"
        idx += 1
    return shifts


def _roll_columns(patch: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = patch.copy()
    for col, shift in enumerate(map(int, shifts)):
        if shift:
            out[:, col] = np.roll(patch[:, col], shift)
    return out


def _roll_rows(patch: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = patch.copy()
    for row, shift in enumerate(map(int, shifts)):
        if shift:
            out[row, :] = np.roll(patch[row, :], shift)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Warp the central rectangle by decoding row and column cyclic shifts from edge strips.

    Input structure
    - Background is black.
    - A single dense non-black rectangle appears away from the top and left edges.
    - The top edge contains orange run markers aligned to the rectangle's columns.
    - The left edge contains azure run markers aligned to the rectangle's rows.

    Rule
    - Each horizontal orange run of length L means every covered rectangle column
      shifts downward by L with wraparound.
    - Each vertical azure run of length L means every covered rectangle row shifts
      rightward by L with wraparound.
    - Apply column shifts first, then row shifts.
    - Return only the transformed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a 2D grid"

    inner = g[1:, 1:]
    pattern_mask = inner != BLACK
    assert bool(pattern_mask.any()), "expected a central non-black rectangle"

    rr0, cc0, rr1, cc1 = _bbox(pattern_mask)
    assert bool(np.all(pattern_mask[rr0 : rr1 + 1, cc0 : cc1 + 1])), "pattern must be a dense rectangle"

    r0, c0 = int(rr0 + 1), int(cc0 + 1)
    r1, c1 = int(rr1 + 1), int(cc1 + 1)
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    col_shifts = _decode_run_shifts(g[0, c0 : c1 + 1], ORANGE)
    row_shifts = _decode_run_shifts(g[r0 : r1 + 1, 0], AZURE)

    after_cols = _roll_columns(patch, col_shifts)
    return _roll_rows(after_cols, row_shifts)


PATTERN_SIZE_RANGE = (6, 10)
MARGIN_RANGE = (1, 3)
PALETTE_SIZE_RANGE = (3, 5)
BRANCH_MODES = ("both", "columns_only", "rows_only")
GENERATION_RETRY_LIMIT = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample_colors(k: int) -> list[int]:
    colors = _RNG.choice(np.arange(1, 10), size=int(k), replace=False)
    return [int(color) for color in colors]


def _line_signatures(arr: np.ndarray, axis: int) -> int:
    if axis == 0:
        return len({tuple(map(int, row)) for row in arr})
    return len({tuple(map(int, col)) for col in arr.T})


def _pattern_is_good(patch: np.ndarray) -> bool:
    if len(np.unique(patch)) < 2:
        return False
    if _line_signatures(patch, axis=0) < min(3, int(patch.shape[0])):
        return False
    if _line_signatures(patch, axis=1) < min(3, int(patch.shape[1])):
        return False
    return True


def _build_pattern(height: int, width: int) -> np.ndarray:
    rr, cc = np.indices((int(height), int(width)))
    for _ in range(200):
        palette = np.array(_sample_colors(_randint(*PALETTE_SIZE_RANGE)), dtype=int)
        k = int(palette.shape[0])
        mode = _choice(("linear", "bands", "rings", "diagonal", "mixed"))

        if mode == "linear":
            a = _randint(1, k)
            b = _randint(1, k)
            idx = (a * rr + b * cc + _randint(0, k - 1)) % k
        elif mode == "bands":
            row_band = _randint(1, max(1, height // 3))
            col_band = _randint(1, max(1, width // 3))
            idx = ((rr // row_band) + 2 * (cc // col_band) + _randint(0, k - 1)) % k
        elif mode == "rings":
            center_r = _randint(0, height - 1)
            center_c = _randint(0, width - 1)
            idx = (np.abs(rr - center_r) + 2 * np.abs(cc - center_c) + _randint(0, k - 1)) % k
        elif mode == "diagonal":
            row_step = _randint(1, max(1, height // 4))
            idx = ((rr - cc) + (rr // row_step) + _randint(0, k - 1)) % k
        else:
            p = _randint(2, 4)
            q = _randint(2, 4)
            idx = (((rr % p) * (cc % q)) + rr + 2 * cc + _randint(0, k - 1)) % k

        patch = palette[idx]
        if _RNG.random() < 0.5:
            patch = np.flipud(patch)
        if _RNG.random() < 0.5:
            patch = np.fliplr(patch)
        if _pattern_is_good(patch):
            return np.array(patch, dtype=int)

    raise RuntimeError("failed to build a sufficiently varied pattern")


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(_RNG.choice(np.arange(total + parts - 1), size=int(parts - 1), replace=False))
    values = []
    prev = -1
    for cut in list(map(int, cuts)) + [int(total + parts - 1)]:
        values.append(int(cut - prev - 1))
        prev = cut
    return values


def _sample_instruction(span: int, max_len: int, active: bool) -> tuple[np.ndarray, np.ndarray]:
    line = np.zeros(int(span), dtype=int)
    shifts = np.zeros(int(span), dtype=int)
    if not active:
        return line, shifts

    max_runs = min(3, int(span))
    for _ in range(200):
        n_runs = _randint(1, max_runs)
        lengths = [_randint(1, max_len) for _ in range(n_runs)]
        min_total = int(sum(lengths) + (n_runs - 1))
        if min_total >= span:
            continue

        extras = _random_composition(int(span - min_total), int(n_runs + 1))
        gaps = [int(extras[0])] + [int(1 + extra) for extra in extras[1:-1]] + [int(extras[-1])]

        pos = int(gaps[0])
        for idx, run_len in enumerate(lengths):
            line[pos : pos + run_len] = 1
            shifts[pos : pos + run_len] = int(run_len)
            pos += int(run_len)
            if idx < n_runs - 1:
                pos += int(gaps[idx + 1])

        if not bool(line.any()) or bool(np.all(line == 1)):
            line.fill(0)
            shifts.fill(0)
            continue
        return line, shifts

    raise RuntimeError("failed to sample an instruction strip")


def _attempt_generate() -> np.ndarray | None:
    height = _randint(*PATTERN_SIZE_RANGE)
    width = _randint(*PATTERN_SIZE_RANGE)
    top_gap = _randint(*MARGIN_RANGE)
    left_gap = _randint(*MARGIN_RANGE)
    bottom_gap = _randint(*MARGIN_RANGE)
    right_gap = _randint(*MARGIN_RANGE)

    grid_h = int(1 + top_gap + height + bottom_gap)
    grid_w = int(1 + left_gap + width + right_gap)
    r0 = int(1 + top_gap)
    c0 = int(1 + left_gap)

    branch = _choice(BRANCH_MODES)
    use_cols = branch in {"both", "columns_only"}
    use_rows = branch in {"both", "rows_only"}

    pattern = _build_pattern(height, width)
    col_line, col_shifts = _sample_instruction(width, min(4, width, height - 1), use_cols)
    row_line, row_shifts = _sample_instruction(height, min(4, height, width - 1), use_rows)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[r0 : r0 + height, c0 : c0 + width] = pattern
    grid[0, c0 : c0 + width][col_line == 1] = ORANGE
    grid[r0 : r0 + height, 0][row_line == 1] = AZURE

    expected = _roll_rows(_roll_columns(pattern, col_shifts), row_shifts)
    if np.array_equal(expected, pattern):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRY_LIMIT} attempts")
