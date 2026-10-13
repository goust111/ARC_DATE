import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
GEN_PERIOD_RANGE = (2, 6)
GEN_REPEATS_RANGE = (3, 5)
GEN_LINE_LENGTH_RANGE = (9, 18)


def _mode_support(line: np.ndarray) -> int:
    counts = Counter(map(int, line))
    return max(counts.values())


def _unique_mode(line: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, line))
    color, support = counts.most_common(1)[0]
    if sum(1 for value in counts.values() if value == support) != 1:
        raise AssertionError("expected a unique line majority")
    return int(color), int(support)


def _orientation_score(grid: np.ndarray, by_rows: bool) -> int:
    lines = grid if by_rows else grid.T
    return int(sum(_mode_support(line) for line in lines))


def _majority_sequence(grid: np.ndarray, by_rows: bool) -> list[int]:
    lines = grid if by_rows else grid.T
    return [_unique_mode(line)[0] for line in lines]


def _render_lines(line_colors: list[int], by_rows: bool, other_dim: int) -> np.ndarray:
    if by_rows:
        out = np.empty((len(line_colors), other_dim), dtype=int)
        for r, color in enumerate(line_colors):
            out[r, :] = color
    else:
        out = np.empty((other_dim, len(line_colors)), dtype=int)
        for c, color in enumerate(line_colors):
            out[:, c] = color
    return out


def _period_candidates(seq: list[int]) -> list[tuple[int, int, int, list[int]]]:
    n = len(seq)
    candidates: list[tuple[int, int, int, list[int]]] = []
    for period in range(1, n):
        template: list[int | None] = [None] * period
        liar_index: int | None = None
        liar_true_color: int | None = None
        valid = True

        for residue in range(period):
            indices = list(range(residue, n, period))
            colors = [seq[idx] for idx in indices]
            counts = Counter(colors)
            if len(counts) == 1:
                template[residue] = colors[0]
                continue
            if len(counts) != 2 or liar_index is not None:
                valid = False
                break

            dominant_color, dominant_count = counts.most_common(1)[0]
            if dominant_count != len(indices) - 1:
                valid = False
                break
            if len(indices) - 1 < 2:
                valid = False
                break

            mismatches = [idx for idx in indices if seq[idx] != dominant_color]
            if len(mismatches) != 1:
                valid = False
                break

            liar_index = mismatches[0]
            liar_true_color = int(dominant_color)
            template[residue] = int(dominant_color)

        if not valid or liar_index is None or liar_true_color is None:
            continue

        repaired = [int(template[idx % period]) for idx in range(n)]
        if sum(a != b for a, b in zip(seq, repaired)) != 1:
            continue
        candidates.append((period, liar_index, liar_true_color, repaired))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a noisy striped canvas by majority-repainting each stripe line, then overturn the one lying line via the global period.

    Input structure:
    - The whole canvas is made of either horizontal row-stripes or vertical column-stripes, and each true stripe line is monochrome.
    - Sparse noise is scattered across many lines, while exactly one line is polluted so heavily that even its majority color is false.
    - After taking the majority color of each line in the correct orientation, the resulting color sequence differs from one strict periodic repetition at exactly one line.

    Rule:
    - Compare the row-wise and column-wise stripe interpretations by total line-majority support, and keep the uniquely better orientation.
    - Repaint every line in that orientation to its own majority color, yielding a mostly-correct stripe-color sequence.
    - Find the unique period length for which exactly one line violates the repetition, and recolor that full line to the color agreed on by its periodic companions.

    Key insight:
    - Local majority voting is only the first layer: one line lies, and its true color is recovered only by the higher-level periodic agreement across matching stripe positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_score = _orientation_score(g, by_rows=True)
    col_score = _orientation_score(g, by_rows=False)
    assert row_score != col_score, "expected a unique stripe orientation"
    by_rows = row_score > col_score

    majority_seq = _majority_sequence(g, by_rows)
    candidates = _period_candidates(majority_seq)
    assert len(candidates) == 1, "expected a unique periodic repair"

    _period, _liar_index, _liar_true_color, repaired_seq = candidates[0]
    other_dim = g.shape[1] if by_rows else g.shape[0]
    return _render_lines(repaired_seq, by_rows, other_dim)


def _sample_period() -> tuple[int, int]:
    periods = []
    for period in range(GEN_PERIOD_RANGE[0], GEN_PERIOD_RANGE[1] + 1):
        repeats = [
            rep
            for rep in range(GEN_REPEATS_RANGE[0], GEN_REPEATS_RANGE[1] + 1)
            if 9 <= period * rep <= 24
        ]
        if repeats:
            periods.append((period, random.choice(repeats)))
    return random.choice(periods)


def _make_true_line_colors(period: int, repeats: int) -> list[int]:
    pattern = random.sample(PALETTE, period)
    return [pattern[idx % period] for idx in range(period * repeats)]


def _make_truthful_line(color: int, line_len: int) -> tuple[np.ndarray, int]:
    line = np.full(line_len, color, dtype=int)
    max_noise = max(1, (line_len - 1) // 3)
    noise_count = random.randint(0, max_noise)
    if noise_count == 0:
        return line, 0

    positions = random.sample(range(line_len), noise_count)
    for pos in positions:
        line[pos] = random.choice([candidate for candidate in PALETTE if candidate != color])
    return line, noise_count


def _make_liar_line(true_color: int, wrong_color: int, line_len: int) -> np.ndarray:
    line = np.full(line_len, true_color, dtype=int)
    positions = list(range(line_len))
    random.shuffle(positions)

    wrong_count = random.randint(line_len // 2 + 1, line_len - 2)
    wrong_positions = positions[:wrong_count]
    for pos in wrong_positions:
        line[pos] = wrong_color

    remaining = positions[wrong_count:]
    extra_noise = random.randint(0, max(0, len(remaining) - 1))
    for pos in remaining[:extra_noise]:
        line[pos] = random.choice([candidate for candidate in PALETTE if candidate not in (true_color, wrong_color)])
    return line


def _build_input(
    true_line_colors: list[int],
    by_rows: bool,
    other_dim: int,
    liar_index: int,
    wrong_color: int,
) -> tuple[np.ndarray, int]:
    n_lines = len(true_line_colors)
    grid = np.empty((n_lines, other_dim), dtype=int) if by_rows else np.empty((other_dim, n_lines), dtype=int)
    truthful_noisy_lines = 0

    for idx, color in enumerate(true_line_colors):
        if idx == liar_index:
            line = _make_liar_line(color, wrong_color, other_dim)
        else:
            line, noise_count = _make_truthful_line(color, other_dim)
            truthful_noisy_lines += int(noise_count > 0)

        if by_rows:
            grid[idx, :] = line
        else:
            grid[:, idx] = line

    return grid, truthful_noisy_lines


def _expected_output(true_line_colors: list[int], by_rows: bool, other_dim: int) -> np.ndarray:
    return _render_lines(true_line_colors, by_rows, other_dim)


def _validate_scene(
    inp: np.ndarray,
    out: np.ndarray,
    true_line_colors: list[int],
    by_rows: bool,
    period: int,
    liar_index: int,
    truthful_noisy_lines: int,
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if truthful_noisy_lines < 2 or truthful_noisy_lines >= len(true_line_colors) - 1:
        return False
    if np.array_equal(inp, out):
        return False

    row_score = _orientation_score(inp, True)
    col_score = _orientation_score(inp, False)
    if row_score == col_score:
        return False
    if (row_score > col_score) != by_rows:
        return False

    try:
        seq = _majority_sequence(inp, by_rows)
    except AssertionError:
        return False
    if sum(a != b for a, b in zip(seq, true_line_colors)) != 1:
        return False
    if seq[liar_index] == true_line_colors[liar_index]:
        return False

    candidates = _period_candidates(seq)
    if len(candidates) != 1:
        return False
    cand_period, cand_liar_index, cand_color, repaired = candidates[0]
    if cand_period != period or cand_liar_index != liar_index:
        return False
    if cand_color != true_line_colors[liar_index]:
        return False
    if repaired != true_line_colors:
        return False

    if not np.array_equal(out, _expected_output(true_line_colors, by_rows, inp.shape[1] if by_rows else inp.shape[0])):
        return False
    if int(np.unique(out).size) < 2:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, out))


def _gen_attempt() -> np.ndarray | None:
    by_rows = bool(random.getrandbits(1))
    period, repeats = _sample_period()
    line_len = random.randint(*GEN_LINE_LENGTH_RANGE)
    true_line_colors = _make_true_line_colors(period, repeats)
    liar_index = random.randrange(len(true_line_colors))
    wrong_color = random.choice([color for color in PALETTE if color not in set(true_line_colors)])

    inp, truthful_noisy_lines = _build_input(true_line_colors, by_rows, line_len, liar_index, wrong_color)
    out = _expected_output(true_line_colors, by_rows, line_len)
    if _validate_scene(inp, out, true_line_colors, by_rows, period, liar_index, truthful_noisy_lines):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lying-stripes input")
