from __future__ import annotations

import random

import numpy as np


BLACK = 0
GEN_RETRIES = 400
BODY_MODES = ("single_segment", "inverted_segment", "double_segment", "cyclic_band")
_RNG = random.Random()


def _reorder_columns(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    row_strip = [int(v) for v in g[1:, 0]]
    col_strip = [int(v) for v in g[0, 1:]]
    assert len(row_strip) == len(col_strip), "row and column strips must use the same number of colors"

    order = {color: idx for idx, color in enumerate(row_strip)}
    assert len(order) == len(row_strip), "row strip colors must be unique"
    assert set(order) == set(col_strip), "top and left strips must use the same color set"

    body = g[1:, 1:]
    assert body.shape == (len(row_strip), len(col_strip))

    reordered = np.empty_like(body)
    for src_col, color in enumerate(col_strip):
        reordered[:, order[int(color)]] = body[:, src_col]

    out = g.copy()
    out[0, 1:] = np.array(row_strip, dtype=int)
    out[1:, 1:] = reordered
    return out


def _non_identity_permutation(n: int, rng: random.Random) -> list[int]:
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if any(idx != val for idx, val in enumerate(perm)):
            return perm[:]


def _single_segment_candidates(n: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for start in range(n):
        for end in range(start + 1, n + 1):
            if start == 0 and end == n:
                continue
            col = np.zeros(n, dtype=int)
            col[start:end] = 1
            candidates.append(col)
    return candidates


def _double_segment_candidates(n: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for top_len in range(n + 1):
        for bottom_start in range(top_len + 1, n + 1):
            col = np.zeros(n, dtype=int)
            col[:top_len] = 1
            col[bottom_start:] = 1
            if col.sum() == 0 or col.sum() == n:
                continue
            candidates.append(col)
    return candidates


def _sample_unique_columns(candidates: list[np.ndarray], n: int, rng: random.Random) -> list[np.ndarray]:
    idxs = list(range(len(candidates)))
    rng.shuffle(idxs)
    picked = [candidates[idx].copy() for idx in idxs[:n]]
    picked.sort(key=lambda col: (int(np.argmax(col)), int(col.sum()), tuple(int(v) for v in col.tolist())))
    return picked


def _cyclic_band_columns(n: int, rng: random.Random) -> list[np.ndarray]:
    width = rng.randint(1, n - 1)
    columns: list[np.ndarray] = []
    for shift in range(n):
        col = np.zeros(n, dtype=int)
        for offset in range(width):
            col[(shift + offset) % n] = 1
        columns.append(col)
    return columns


def _build_ordered_body(n: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray:
    mode = rng.choice(BODY_MODES)
    if mode == "single_segment":
        columns = _sample_unique_columns(_single_segment_candidates(n), n, rng)
        invert = False
    elif mode == "inverted_segment":
        columns = _sample_unique_columns(_single_segment_candidates(n), n, rng)
        invert = True
    elif mode == "double_segment":
        columns = _sample_unique_columns(_double_segment_candidates(n), n, rng)
        invert = False
    else:
        columns = _cyclic_band_columns(n, rng)
        invert = rng.choice([False, True])

    body = np.full((n, n), color_b, dtype=int)
    for col_idx, column in enumerate(columns):
        bits = 1 - column if invert else column
        body[bits.astype(bool), col_idx] = color_a

    assert len({tuple(int(v) for v in body[:, c].tolist()) for c in range(n)}) == n
    assert len(np.unique(body)) == 2
    return body


def _make_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    n = rng.randint(3, 7)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    body_color_a = palette[0]
    body_color_b = palette[1]
    strip_colors = palette[2 : 2 + n]

    row_strip = strip_colors[:]
    rng.shuffle(row_strip)

    ordered_body = _build_ordered_body(n, body_color_a, body_color_b, rng)
    perm = _non_identity_permutation(n, rng)
    col_strip = [row_strip[idx] for idx in perm]
    shuffled_body = ordered_body[:, perm]

    inp = np.empty((n + 1, n + 1), dtype=int)
    inp[0, 0] = BLACK
    inp[0, 1:] = np.array(col_strip, dtype=int)
    inp[1:, 0] = np.array(row_strip, dtype=int)
    inp[1:, 1:] = shuffled_body

    out = np.empty_like(inp)
    out[0, 0] = BLACK
    out[0, 1:] = np.array(row_strip, dtype=int)
    out[1:, 0] = np.array(row_strip, dtype=int)
    out[1:, 1:] = ordered_body

    assert not np.array_equal(inp, out)
    assert np.array_equal(_reorder_columns(inp), out)
    return inp, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the interior columns so the top color strip matches the left color strip order.

    Input structure:
    - The grid has shape (N+1) x (N+1) with a black top-left corner cell.
    - The first column below the corner is a row strip: N distinct colors listed in the canonical top-to-bottom order.
    - The first row to the right of the corner is a column strip: the same N colors, but permuted.
    - The remaining NxN body is a two-color board, and each body column travels together with its top strip color.

    Rule:
    - Read the left strip as a color-to-position lookup table.
    - For each column in the body, inspect the strip color directly above it in the first row and find that color's position in the left strip.
    - Move that top strip color and the whole body column into the matching target column.
    - Keep the left strip and the top-left corner unchanged.

    Key insight:
    - The edge strips are control markers, not decoration: the left strip defines the standard order, and the top strip tells which output slot each body column belongs to.
    """
    return _reorder_columns(grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, out = _make_case(_RNG)
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid non-degenerate example")
