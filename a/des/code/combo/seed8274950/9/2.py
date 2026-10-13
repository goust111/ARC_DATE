from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _read_seed(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    row = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(row != bg)
    assert cols.size > 0, "expected a non-empty seed row"
    start = int(cols[0])
    end = int(cols[-1])
    assert np.all(row[start : end + 1] != bg), "seed row must be a contiguous colored run"
    return start, [int(v) for v in row[start : end + 1]]


def _read_rules(grid: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    rule_in = np.asarray(grid[1], dtype=int)
    rule_out = np.asarray(grid[2], dtype=int)

    rules: dict[tuple[int, int], int] = {}
    for c in range(grid.shape[1] - 1):
        if int(rule_out[c]) == bg:
            continue
        left = int(rule_in[c])
        right = int(rule_in[c + 1])
        assert left != bg and right != bg, "each visible rule output must sit under a visible input pair"
        pair = (left, right)
        out_color = int(rule_out[c])
        if pair in rules:
            assert rules[pair] == out_color, "duplicate input pairs must map consistently"
        rules[pair] = out_color

    assert rules, "expected at least one local replacement rule"
    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a local replacement table from rows 1-2, then repeatedly shrink the seed row.

    Input structure:
    - Row 0 contains a contiguous seed sequence of non-background pixels.
    - Row 1 encodes rule inputs: each adjacent pair is one key.
    - Row 2 encodes rule outputs: the pixel under the left element of a pair is the result.
    - Remaining cells are background.

    Rule:
    - For each visible rule at column c, map (row1[c], row1[c+1]) -> row2[c].
    - Copy the seed row to output row 0.
    - Output row r+1 is formed from adjacent pairs in output row r.
    - Each new row is one cell shorter and stays left-aligned with the seed start.
    - Stop after reaching one pixel or the bottom of the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3, "expected seed and two rule rows"

    bg = _get_bg(g)
    seed_start, current = _read_seed(g, bg)
    rules = _read_rules(g, bg)

    out = np.full_like(g, bg)
    row = 0
    while row < g.shape[0] and current:
        out[row, seed_start : seed_start + len(current)] = np.array(current, dtype=int)
        if len(current) == 1:
            break

        next_row: list[int] = []
        for i in range(len(current) - 1):
            pair = (int(current[i]), int(current[i + 1]))
            assert pair in rules, f"missing rule for pair {pair}"
            next_row.append(int(rules[pair]))
        current = next_row
        row += 1

    return out


def _de_bruijn(k: int, n: int) -> list[int]:
    a = [0] * (k * n)
    sequence: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                sequence.extend(a[1 : p + 1])
            return

        a[t] = a[t - p]
        db(t + 1, p)
        for j in range(a[t - p] + 1, k):
            a[t] = j
            db(t + 1, t)

    db(1, 1)
    return sequence


def _rule_sequence(symbols: list[int]) -> list[int]:
    base = _de_bruijn(len(symbols), 2)
    if random.random() < 0.5:
        base = list(reversed(base))
    shift = random.randrange(len(base))
    cycle = base[shift:] + base[:shift]
    return [symbols[idx] for idx in cycle + cycle[:1]]


def _sample_mapping(pairs: list[tuple[int, int]], symbols: list[int]) -> dict[tuple[int, int], int]:
    while True:
        mapping = {pair: random.choice(symbols) for pair in pairs}
        if len(set(mapping.values())) >= 2:
            return mapping


def _sample_seed(symbols: list[int], length: int) -> list[int]:
    while True:
        seed = [random.choice(symbols) for _ in range(length)]
        if len(set(seed)) < 2:
            continue
        if len({(seed[i], seed[i + 1]) for i in range(length - 1)}) < 2:
            continue
        return seed


def _simulate_rows(seed: list[int], rules: dict[tuple[int, int], int], max_rows: int) -> list[list[int]]:
    rows = [list(seed)]
    current = list(seed)
    while len(rows) < max_rows and len(current) > 1:
        current = [int(rules[(current[i], current[i + 1])]) for i in range(len(current) - 1)]
        rows.append(list(current))
    return rows


def _build_grid(
    height: int,
    width: int,
    bg: int,
    seed_start: int,
    seed: list[int],
    rule_start: int,
    rule_seq: list[int],
    mapping: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    grid[0, seed_start : seed_start + len(seed)] = np.array(seed, dtype=int)
    grid[1, rule_start : rule_start + len(rule_seq)] = np.array(rule_seq, dtype=int)
    rule_outputs = [mapping[(rule_seq[i], rule_seq[i + 1])] for i in range(len(rule_seq) - 1)]
    grid[2, rule_start : rule_start + len(rule_outputs)] = np.array(rule_outputs, dtype=int)
    return grid


def _valid_attempt(grid: np.ndarray, mode: str, seed_len: int) -> bool:
    out = main(grid)
    bg = _get_bg(grid)

    used_rows = int(np.count_nonzero(np.any(out != bg, axis=1)))
    if used_rows != min(seed_len, grid.shape[0]):
        return False

    occupied = np.flatnonzero(out[used_rows - 1] != bg)
    if occupied.size == 0:
        return False
    last_len = int(occupied.size)

    if mode == "full" and last_len != 1:
        return False
    if mode == "truncated" and last_len <= 1:
        return False

    if len(set(np.unique(out[out != bg]).tolist())) < 2:
        return False

    first_two_rows = tuple(tuple(int(v) for v in row) for row in out[: min(2, used_rows)])
    if len(first_two_rows) == 2 and first_two_rows[0] == first_two_rows[1]:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n_symbols = random.choice([2, 3])
    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    symbols = random.sample(palette, n_symbols)

    rule_seq = _rule_sequence(symbols)
    pairs = [(int(rule_seq[i]), int(rule_seq[i + 1])) for i in range(len(rule_seq) - 1)]
    assert len(set(pairs)) == n_symbols * n_symbols
    mapping = _sample_mapping(pairs, symbols)

    seed_len = random.randint(6, 12)
    seed = _sample_seed(symbols, seed_len)
    mode = random.choice(["full", "truncated"])

    if mode == "full":
        height = random.randint(seed_len, seed_len + 4)
    else:
        height = random.randint(4, seed_len - 1)

    simulated = _simulate_rows(seed, mapping, height)
    if len(simulated) < 3:
        return None
    if mode == "full" and len(simulated[-1]) != 1:
        return None
    if mode == "truncated" and len(simulated[-1]) == 1:
        return None

    min_width = max(len(rule_seq), seed_len) + 2
    width = random.randint(min_width, min_width + 8)
    seed_start = random.randint(0, width - seed_len)
    rule_start = random.randint(0, width - len(rule_seq))

    grid = _build_grid(height, width, bg, seed_start, seed, rule_start, rule_seq, mapping)
    if not _valid_attempt(grid, mode, seed_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
