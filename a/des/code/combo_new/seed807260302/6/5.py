import random

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIZE = 30
GEN_ATTEMPTS = 300


def _find_separator_row(g: np.ndarray) -> int:
    grey_rows = [r for r in range(g.shape[0]) if np.any(g[r] == GREY)]
    assert len(grey_rows) == 1, "expected exactly one horizontal grey separator"
    sep_r = int(grey_rows[0])

    row = g[sep_r]
    assert np.all((row == BG) | (row == GREY)), "separator row may contain only black and grey"
    cols = np.flatnonzero(row == GREY)
    assert cols.size > 0, "separator must contain grey pixels"
    assert np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1)), (
        "grey separator must be one contiguous horizontal line"
    )
    return sep_r


def _strip_sequences(g: np.ndarray, sep_r: int) -> list[np.ndarray]:
    strips: list[np.ndarray] = []
    for r in range(sep_r):
        row = g[r]
        cols = np.flatnonzero((row != BG) & (row != GREY))
        if cols.size == 0:
            continue

        c0 = int(cols[0])
        c1 = int(cols[-1])
        segment = row[c0 : c1 + 1]
        assert np.all((segment != BG) & (segment != GREY)), "colored strip cannot contain black or grey gaps"
        assert np.array_equal(cols, np.arange(c0, c1 + 1)), "colored strip must be contiguous"
        strips.append(segment.astype(int, copy=True))

    assert strips, "expected at least one colored strip above the separator"
    return strips


def _fold_strip(seq: np.ndarray, width: int) -> np.ndarray:
    assert len(seq) % width == 0, "strip length must be an integer multiple of the fold width"
    rows = []
    for i in range(len(seq) // width):
        row = seq[i * width : (i + 1) * width].copy()
        if i % 2 == 1:
            row = row[::-1]
        rows.append(row)
    return np.vstack(rows).astype(int, copy=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replay each colored recording strip as snake-folded rows and concatenate the decoded canvas.

    Input structure:
    - Black is the background and a single horizontal grey line separates the grid into an upper recording area and a lower blank area.
    - Above the separator are one or more one-pixel-high colored strips, each a contiguous non-black, non-grey sequence possibly padded by black space.
    - The lower area is entirely black; its drawing width is the length of the shortest colored strip.

    Rule:
    - Read the colored strips from top to bottom and use the shortest strip length as the output width W.
    - Each strip length is an integer multiple of W; split a strip into consecutive chunks of W pixels.
    - Place chunk 1 left-to-right, chunk 2 right-to-left, chunk 3 left-to-right, and so on, making a snake-folded block for that strip.
    - Stack all decoded strip blocks vertically in the same top-to-bottom order and discard the original strips, separator, and blank area.

    Key insight:
    - The shortest recording strip fixes the fold width, and alternating row parity reveals the boustrophedon path used to replay every longer strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_r = _find_separator_row(g)
    assert sep_r < g.shape[0] - 1, "separator must have a black area below it"
    assert np.all(g[sep_r + 1 :] == BG), "lower area must be completely black"

    strips = _strip_sequences(g, sep_r)
    width = min(len(seq) for seq in strips)
    assert width > 0

    decoded = [_fold_strip(seq, width) for seq in strips]
    out = np.vstack(decoded).astype(int, copy=False)
    assert max(out.shape) <= MAX_SIZE, "decoded output must fit ARC size limits"
    return out


def _encoded_strip(decoded_rows: np.ndarray) -> np.ndarray:
    chunks = []
    for i, row in enumerate(decoded_rows):
        chunks.append(row[::-1] if i % 2 == 1 else row)
    return np.concatenate(chunks).astype(int, copy=False)


def _random_decoded_rows(rng: random.Random, row_count: int, width: int, palette: list[int]) -> np.ndarray:
    rows = np.array(
        [[rng.choice(palette) for _ in range(width)] for _ in range(row_count)],
        dtype=int,
    )
    if len(set(map(int, rows.flat))) == 1:
        rows[0, 0] = rng.choice([c for c in palette if c != int(rows[0, 0])])
    return rows


def _choose_multipliers(rng: random.Random, strip_count: int, width: int) -> list[int] | None:
    max_k = min(5, MAX_SIZE // width)
    if max_k < 2:
        return None

    for _ in range(100):
        ks = [rng.randint(1, max_k) for _ in range(strip_count)]
        ks[rng.randrange(strip_count)] = 1
        if not any(k > 1 for k in ks):
            ks[rng.randrange(strip_count)] = rng.randint(2, max_k)
        if sum(ks) <= 18 and len(set(ks)) >= 2:
            return ks
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    strip_count = rng.randint(2, 5)
    width = rng.randint(2, 8)
    ks = _choose_multipliers(rng, strip_count, width)
    if ks is None:
        return None

    max_len = width * max(ks)
    left_pad = rng.randint(0, min(3, MAX_SIZE - max_len))
    right_limit = MAX_SIZE - max_len - left_pad
    if right_limit < 0:
        return None
    right_pad = rng.randint(0, min(4, right_limit))
    grid_w = left_pad + max_len + right_pad
    if grid_w > MAX_SIZE:
        return None

    top_margin = rng.randint(0, 2)
    gaps = [rng.randint(0, 2) for _ in range(strip_count - 1)]
    top_h = top_margin + strip_count + sum(gaps)
    lower_h = sum(ks)
    grid_h = top_h + 1 + lower_h
    if grid_h > MAX_SIZE or lower_h > MAX_SIZE:
        return None

    palette_size = rng.randint(3, min(6, len(COLORS)))
    palette = rng.sample(COLORS, palette_size)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    r = top_margin
    for i, k in enumerate(ks):
        decoded_rows = _random_decoded_rows(rng, k, width, palette)
        seq = _encoded_strip(decoded_rows)
        if len(set(map(int, seq))) < 2:
            return None
        grid[r, left_pad : left_pad + len(seq)] = seq
        if i < len(gaps):
            r += 1 + gaps[i]

    sep_r = top_h
    grid[sep_r, :] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != (lower_h, width):
        return None
    if max(grid.shape) > MAX_SIZE or max(out.shape) > MAX_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-recording input")
