from __future__ import annotations

from collections import Counter
import random

import numpy as np


ALL_COLORS = tuple(range(10))
GEN_MAX_ATTEMPTS = 600
HEADER_MIN_LEN = 7
HEADER_MAX_LEN = 22
PERIOD_CHOICES = (1, 2, 3, 4, 5)
PERIOD_WEIGHTS = (1, 3, 3, 2, 2)


def _strict_majority(values: list[int]) -> int | None:
    counts = Counter(values)
    color, freq = counts.most_common(1)[0]
    if freq * 2 <= len(values):
        return None
    return int(color)


def _repair_candidates(seq: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    values = [int(x) for x in np.asarray(seq, dtype=int)]
    n = len(values)
    max_period = n // 3
    candidates: list[tuple[int, int, np.ndarray]] = []

    for period in range(1, max_period + 1):
        residue_colors: list[int] = []
        valid = True
        for residue in range(period):
            color = _strict_majority(values[residue:n:period])
            if color is None:
                valid = False
                break
            residue_colors.append(color)
        if not valid:
            continue
        repaired = np.array([residue_colors[i % period] for i in range(n)], dtype=int)
        mismatches = int(np.count_nonzero(repaired != seq))
        candidates.append((mismatches, period, repaired))

    assert candidates, "header sequence must admit at least one periodic repair"
    return candidates


def _repair_periodic_sequence(seq: np.ndarray) -> np.ndarray:
    candidates = sorted(_repair_candidates(seq), key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _render_table(
    bg: int,
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    corner: int,
) -> np.ndarray:
    row_labels = np.asarray(row_labels, dtype=int)
    col_labels = np.asarray(col_labels, dtype=int)
    out = np.full((len(row_labels) + 1, len(col_labels) + 1), int(bg), dtype=int)

    for r, color in enumerate(row_labels):
        out[r, :-1] = np.where(col_labels == int(color), int(color), int(bg))

    out[:-1, -1] = row_labels
    out[-1, :-1] = col_labels
    out[-1, -1] = int(corner)
    return out


def _fundamental_period(pattern: list[int]) -> int:
    n = len(pattern)
    for period in range(1, n + 1):
        if all(pattern[i] == pattern[i % period] for i in range(n)):
            return period
    return n


def _sample_length(period: int) -> int:
    minimum = max(HEADER_MIN_LEN, 3 * period)
    maximum = HEADER_MAX_LEN
    assert minimum <= maximum
    choices = list(range(minimum, maximum + 1))
    non_multiples = [n for n in choices if n % period != 0]
    if non_multiples and random.random() < 0.6:
        return random.choice(non_multiples)
    return random.choice(choices)


def _sample_base_period(period: int, palette: list[int]) -> np.ndarray:
    for _ in range(200):
        base = [random.choice(palette) for _ in range(period)]
        if _fundamental_period(base) == period:
            return np.array(base, dtype=int)
    raise RuntimeError("failed to sample a primitive period")


def _make_clean_sequence(length: int, period: int, palette: list[int]) -> np.ndarray:
    base = _sample_base_period(period, palette)
    return np.array([int(base[i % period]) for i in range(length)], dtype=int)


def _pick_noise_color(original: int, bg: int, clean_palette: set[int]) -> int:
    foreign = [c for c in ALL_COLORS if c != bg and c != original and c not in clean_palette]
    local = [c for c in clean_palette if c != original]
    if foreign and random.random() < 0.6:
        return int(random.choice(foreign))
    pool = local if local else foreign
    assert pool
    return int(random.choice(pool))


def _inject_noise(clean: np.ndarray, period: int, bg: int, clean_palette: set[int]) -> np.ndarray:
    noisy = np.array(clean, dtype=int).copy()
    per_residue: list[tuple[list[int], int]] = []
    total_budget = 0
    for residue in range(period):
        positions = list(range(residue, len(clean), period))
        max_corrupt = (len(positions) - 1) // 2
        if max_corrupt > 0:
            per_residue.append((positions, max_corrupt))
            total_budget += max_corrupt

    assert per_residue and total_budget > 0

    chosen_positions: set[int] = set()
    random.shuffle(per_residue)
    for positions, max_corrupt in per_residue:
        if random.random() < 0.65:
            corruptions = 1
            if max_corrupt >= 2 and random.random() < 0.25:
                corruptions = 2
            for pos in random.sample(positions, min(corruptions, max_corrupt)):
                chosen_positions.add(int(pos))

    if not chosen_positions:
        positions, _max_corrupt = random.choice(per_residue)
        chosen_positions.add(int(random.choice(positions)))

    for pos in chosen_positions:
        noisy[pos] = _pick_noise_color(int(clean[pos]), bg, clean_palette)

    assert np.count_nonzero(noisy != clean) >= 1
    return noisy


def _best_repairs_are_unique(seq: np.ndarray, expected: np.ndarray) -> bool:
    candidates = _repair_candidates(seq)
    best_mismatches = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_mismatches]
    repaired_signatures = {tuple(map(int, item[2])) for item in best}
    return len(repaired_signatures) == 1 and np.array_equal(best[0][2], expected)


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the noisy periodic headers, then fill the interior where repaired row and column labels are equal.

    Input structure:
    - The interior of the grid is a uniform background.
    - The rightmost column is a row-label sequence and the bottom row is a
      column-label sequence.
    - Each header is a short periodic sequence repeated across the edge, with a
      few corrupted cells; the bottom-right corner is not part of either header.

    Rule:
    - Infer each true header by testing short periods and taking the majority
      color at every residue class.
    - Replace the noisy right and bottom headers with these repaired sequences.
    - For every interior cell, compare its repaired row label with its repaired
      column label: paint the shared color on equality, otherwise leave the
      background.

    Key insight:
    - The visible headers are unreliable, so the equality table is correct only
      after both periodic label sequences are denoised first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = int(Counter(map(int, g[:-1, :-1].ravel())).most_common(1)[0][0])
    row_labels = _repair_periodic_sequence(g[:-1, -1])
    col_labels = _repair_periodic_sequence(g[-1, :-1])
    return _render_table(bg, row_labels, col_labels, int(g[-1, -1]))


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(random.choice(ALL_COLORS))
    usable_colors = [c for c in ALL_COLORS if c != bg]
    palette_size = random.randint(2, min(4, len(usable_colors)))
    label_palette = random.sample(usable_colors, palette_size)

    row_period = random.choices(PERIOD_CHOICES, weights=PERIOD_WEIGHTS, k=1)[0]
    col_period = random.choices(PERIOD_CHOICES, weights=PERIOD_WEIGHTS, k=1)[0]
    row_len = _sample_length(row_period)
    col_len = _sample_length(col_period)

    clean_row = _make_clean_sequence(row_len, row_period, label_palette)
    clean_col = _make_clean_sequence(col_len, col_period, label_palette)
    row_colors = {int(x) for x in clean_row}
    col_colors = {int(x) for x in clean_col}
    if not (row_colors & col_colors):
        return None

    correct_fill = int(np.count_nonzero(clean_row[:, None] == clean_col[None, :]))
    area = row_len * col_len
    if correct_fill < max(6, area // 10) or correct_fill > area - max(6, area // 10):
        return None

    noisy_row = _inject_noise(clean_row, row_period, bg, row_colors)
    noisy_col = _inject_noise(clean_col, col_period, bg, col_colors)

    if not _best_repairs_are_unique(noisy_row, clean_row):
        return None
    if not _best_repairs_are_unique(noisy_col, clean_col):
        return None

    corner = int(random.choice(ALL_COLORS))
    inp = np.full((row_len + 1, col_len + 1), bg, dtype=int)
    inp[:-1, -1] = noisy_row
    inp[-1, :-1] = noisy_col
    inp[-1, -1] = corner

    expected = _render_table(bg, clean_row, clean_col, corner)
    if np.array_equal(expected, _render_table(bg, noisy_row, noisy_col, corner)):
        return None
    if np.array_equal(expected, _render_table(bg, clean_row, noisy_col, corner)):
        return None
    if np.array_equal(expected, _render_table(bg, noisy_row, clean_col, corner)):
        return None
    if np.array_equal(inp, expected):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid noisy periodic header table")
