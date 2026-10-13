import random

import numpy as np


BG_COLOR = 0


def _half_spans(length: int) -> tuple[int, int]:
    return (length - 1) // 2, length // 2


def _pair_geometry(kind: str, mid_r: int, mid_c: int, length: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    left, right = _half_spans(length)
    if kind == "H":
        anchors = [(mid_r, mid_c - left), (mid_r, mid_c + right)]
        mirror = [(mid_r + dr, mid_c) for dr in range(-left, right + 1)]
    else:
        anchors = [(mid_r - left, mid_c), (mid_r + right, mid_c)]
        mirror = [(mid_r, mid_c + dc) for dc in range(-left, right + 1)]
    return anchors, mirror


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each aligned color pair and draw an equal-length perpendicular segment through the pair's midpoint.

    Input structure:
    - The background is black.
    - Each non-black color appears exactly twice as two anchor pixels.
    - Every same-color pair is aligned either on one row or on one column, never diagonally.

    Rule:
    - For each color pair, measure the inclusive length L of the hidden segment between the anchors.
    - Keep the anchor pixels unchanged and do not draw that hidden segment itself.
    - Instead, draw a segment of the same color and the same length L through the hidden segment's midpoint, with orientation rotated by 90 degrees.
    - When L is even, use the midpoint cell closer to the grid origin, which makes the new segment extend `floor((L-1)/2)` cells toward the origin side and `floor(L/2)` cells toward the opposite side.

    Key insight:
    - The two same-color pixels are not independent markers; together they encode an unseen bar whose length and midpoint determine a perpendicular copy in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for color in map(int, np.unique(g)):
        if color == BG_COLOR:
            continue

        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = (tuple(map(int, rc)) for rc in coords)

        if r0 == r1:
            c_lo, c_hi = sorted((c0, c1))
            length = c_hi - c_lo + 1
            mid_c = c_lo + (length - 1) // 2
            start_r = r0 - (length - 1) // 2
            end_r = r0 + length // 2
            assert 0 <= start_r <= end_r < h
            for rr in range(start_r, end_r + 1):
                assert int(out[rr, mid_c]) in (BG_COLOR, color)
                out[rr, mid_c] = color
        else:
            assert c0 == c1, "matched anchors must share a row or a column"
            r_lo, r_hi = sorted((r0, r1))
            length = r_hi - r_lo + 1
            mid_r = r_lo + (length - 1) // 2
            start_c = c0 - (length - 1) // 2
            end_c = c0 + length // 2
            assert 0 <= start_c <= end_c < w
            for cc in range(start_c, end_c + 1):
                assert int(out[mid_r, cc]) in (BG_COLOR, color)
                out[mid_r, cc] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
PAIR_COUNT_RANGE = (2, 5)
ROW_BASE_RANGE = (11, 24)
COL_BASE_RANGE = (11, 24)
BUFFER_RADIUS = 1
PLACEMENT_TRIES = 200


def _balanced_binary_labels(n: int, a: str, b: str) -> list[str]:
    count_a = n // 2
    if n % 2 == 1:
        count_a += random.randint(0, 1)
    labels = [a] * count_a + [b] * (n - count_a)
    random.shuffle(labels)
    return labels


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-BUFFER_RADIUS, BUFFER_RADIUS + 1):
            for dc in range(-BUFFER_RADIUS, BUFFER_RADIUS + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _length_candidates(limit: int, parity: str) -> list[int]:
    lo = 3 if parity == "odd" else 4
    return [length for length in range(lo, limit + 1) if length % 2 == (1 if parity == "odd" else 0)]


def _sample_specs(rows: int, cols: int, n_pairs: int) -> list[dict] | None:
    orientations = _balanced_binary_labels(n_pairs, "H", "V")
    parities = _balanced_binary_labels(n_pairs, "odd", "even")
    max_length = min(rows, cols, max(6, 12 - n_pairs))
    specs = []
    for kind, parity in zip(orientations, parities):
        candidates = _length_candidates(max_length, parity)
        if not candidates:
            return None
        length = random.choice(candidates)
        specs.append({"kind": kind, "length": length})
    random.shuffle(specs)
    specs.sort(key=lambda item: item["length"], reverse=True)
    return specs


def _place_specs(rows: int, cols: int, specs: list[dict]) -> list[dict] | None:
    forbidden: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for spec in specs:
        kind = spec["kind"]
        length = spec["length"]
        left, right = _half_spans(length)
        candidates = [(r, c) for r in range(left, rows - right) for c in range(left, cols - right)]
        random.shuffle(candidates)

        chosen = None
        for mid_r, mid_c in candidates[:PLACEMENT_TRIES]:
            anchors, mirror = _pair_geometry(kind, mid_r, mid_c, length)
            footprint = set(anchors) | set(mirror)
            if footprint & forbidden:
                continue
            chosen = {
                "kind": kind,
                "length": length,
                "mid_r": mid_r,
                "mid_c": mid_c,
                "anchors": anchors,
                "mirror": mirror,
            }
            forbidden |= _dilate(footprint, (rows, cols))
            break

        if chosen is None:
            return None
        placed.append(chosen)

    return placed


def _render_grids(rows: int, cols: int, placed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    colors = random.sample(range(1, 10), len(placed))
    inp = np.full((rows, cols), BG_COLOR, dtype=int)
    out = np.full((rows, cols), BG_COLOR, dtype=int)

    for pair, color in zip(placed, colors):
        for r, c in pair["anchors"]:
            inp[r, c] = color
            out[r, c] = color
        for r, c in pair["mirror"]:
            assert int(out[r, c]) in (BG_COLOR, color)
            out[r, c] = color

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    min_rows = max(ROW_BASE_RANGE[0], 6 + 3 * n_pairs)
    min_cols = max(COL_BASE_RANGE[0], 6 + 3 * n_pairs)
    rows = random.randint(min_rows, ROW_BASE_RANGE[1])
    cols = random.randint(min_cols, COL_BASE_RANGE[1])

    specs = _sample_specs(rows, cols, n_pairs)
    if specs is None:
        return None

    placed = _place_specs(rows, cols, specs)
    if placed is None:
        return None

    inp, expected = _render_grids(rows, cols, placed)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
