from collections import Counter

import numpy as np
import random


def _mode_count(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a nonempty line"
    return max(counts.values())


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected a nonempty line"
    best_count = max(counts.values())
    best_colors = [color for color, count in counts.items() if count == best_count]
    assert len(best_colors) == 1, "expected each selected stripe line to have a unique dominant color"
    return int(best_colors[0])


def _orientation_score(g: np.ndarray, *, by_rows: bool) -> int:
    if by_rows:
        return sum(_mode_count(g[row, :]) for row in range(g.shape[0]))
    return sum(_mode_count(g[:, col]) for col in range(g.shape[1]))


def _repaint_lines(g: np.ndarray, *, by_rows: bool) -> np.ndarray:
    out = np.empty_like(g)
    if by_rows:
        for row in range(g.shape[0]):
            out[row, :] = _unique_mode(g[row, :])
    else:
        for col in range(g.shape[1]):
            out[:, col] = _unique_mode(g[:, col])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a noisy striped grid by repainting each stripe line with its dominant color.

    Input structure:
    - The grid is a full canvas of either horizontal row stripes or vertical column stripes.
    - Sparse noise pixels may use stripe colors or extra colors, but each true stripe line
      still has a unique dominant color.

    Rule:
    - Score the horizontal-stripe interpretation by summing the dominant-color counts of
      every row, and score the vertical-stripe interpretation the same way over columns.
    - Choose the uniquely better orientation.
    - Repaint each row or column in that orientation entirely with its unique dominant color.

    Key insight:
    - The correct stripe direction is the one whose lines are much closer to monochrome;
      the output is the clean striped canvas hidden under the noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_score = _orientation_score(g, by_rows=True)
    col_score = _orientation_score(g, by_rows=False)
    assert row_score != col_score, "expected a unique stripe orientation"

    return _repaint_lines(g, by_rows=row_score > col_score)

# --------------- generator (o5x) ---------------

_GEN_COLORS = tuple(range(1, 10))  # Palette; every generated input contains all colors.

_GEN_HEIGHT_RANGE = (13, 15)  # Canvas dimensions sampled uniformly, inclusive.
_GEN_WIDTH_RANGE = (14, 17)

_GEN_STRIPE_COUNT_RANGE = (3, 5)  # Stripe bands sampled along the selected orientation.
_GEN_STRIPE_WIDTH_RANGE = (2, 6)

_GEN_NOISE_MIN_DENOM = 10  # Noise density and per-line cap keep strict modes.
_GEN_NOISE_MAX_DENOM = 5
_GEN_LINE_NOISE_DENOM = 2

_GEN_ATTEMPTS = 300  # Rejection sampling retry budget.


def _gen_partition(total: int, parts: int) -> tuple[int, ...] | None:
    """Return a random valid width partition, or None if none exists."""
    lo, hi = _GEN_STRIPE_WIDTH_RANGE
    options: list[tuple[int, ...]] = []

    def _gen_extend(remaining: int, slots: int, prefix: tuple[int, ...]) -> None:
        if slots == 0:
            if remaining == 0:
                options.append(prefix)
            return
        min_w = max(lo, remaining - (slots - 1) * hi)
        max_w = min(hi, remaining - (slots - 1) * lo)
        for width in range(min_w, max_w + 1):
            _gen_extend(remaining - width, slots - 1, prefix + (width,))

    _gen_extend(total, parts, ())
    if not options:
        return None
    return random.choice(options)


def _gen_make_output(height, width, by_rows, stripe_widths, stripe_colors) -> tuple[np.ndarray, list[int]]:
    """Return the clean striped grid and the true color for each row/column."""
    out = np.empty((height, width), dtype=int)
    line_colors: list[int] = []
    offset = 0
    for band_width, color in zip(stripe_widths, stripe_colors):
        if by_rows:
            out[offset : offset + band_width, :] = color
        else:
            out[:, offset : offset + band_width] = color
        line_colors.extend([color] * band_width)
        offset += band_width
    return out, line_colors


def _gen_line_to_rc(line: int, offset: int, by_rows: bool) -> tuple[int, int]:
    return (line, offset) if by_rows else (offset, line)


def _gen_add_noise(inp: np.ndarray, by_rows: bool, line_colors: list[int], noise_count: int) -> None:
    """Mutate inp with noise while preserving strict line modes and all-color coverage."""
    line_len = inp.shape[1] if by_rows else inp.shape[0]
    max_per_line = (line_len - 1) // _GEN_LINE_NOISE_DENOM
    slots_by_line = [list(range(line_len)) for _ in line_colors]
    for slots in slots_by_line:
        random.shuffle(slots)
    used_by_line = [0] * len(line_colors)

    def _gen_place_noise(color: int | None = None) -> bool:
        feasible = [
            line
            for line, slots in enumerate(slots_by_line)
            if slots and used_by_line[line] < max_per_line and color != line_colors[line]
        ]
        if not feasible:
            return False
        line = random.choice(feasible)
        offset = slots_by_line[line].pop()
        if color is None:
            color = random.choice([candidate for candidate in _GEN_COLORS if candidate != line_colors[line]])
        r, c = _gen_line_to_rc(line, offset, by_rows)
        inp[r, c] = color
        used_by_line[line] += 1
        return True

    present = set(map(int, inp.reshape(-1)))
    missing_colors = [color for color in _GEN_COLORS if color not in present]
    for missing_color in random.sample(missing_colors, len(missing_colors)):
        if not _gen_place_noise(missing_color):
            return

    remaining = noise_count - sum(used_by_line)
    for _ in range(remaining):
        if not _gen_place_noise():
            return


def _gen_sample_output() -> tuple[np.ndarray, bool, list[int]] | None:
    """Sample geometry, stripe colors, and the corresponding clean output."""
    height = random.randint(*_GEN_HEIGHT_RANGE)
    width = random.randint(*_GEN_WIDTH_RANGE)
    by_rows = bool(random.getrandbits(1))
    stripe_dim = height if by_rows else width
    num_stripes = random.randint(*_GEN_STRIPE_COUNT_RANGE)
    stripe_widths = _gen_partition(stripe_dim, num_stripes)
    if stripe_widths is None:
        return None

    stripe_colors = random.sample(_GEN_COLORS, num_stripes)
    out, line_colors = _gen_make_output(height, width, by_rows, stripe_widths, stripe_colors)
    return out, by_rows, line_colors


def _gen_solves_to(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    sample = _gen_sample_output()
    if sample is None:
        return None
    out, by_rows, line_colors = sample
    inp = out.copy()
    min_noise = (inp.size + _GEN_NOISE_MIN_DENOM - 1) // _GEN_NOISE_MIN_DENOM
    max_noise = inp.size // _GEN_NOISE_MAX_DENOM
    _gen_add_noise(inp, by_rows, line_colors, random.randint(min_noise, max_noise))
    if _gen_solves_to(inp, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e26a3af2 input")
