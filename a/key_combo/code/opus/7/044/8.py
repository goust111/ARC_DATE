from collections import Counter
import random

import numpy as np


BLACK = 0
PALETTE_POOL = (1, 2, 3, 4, 5)

MIN_SEPARATORS = 2
MAX_SEPARATORS = 4
CONTENT_WIDTH_RANGE = (7, 12)
BAND_HEIGHT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[int(r), :] == BLACK)]


def _band_spans(height: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "each separator must split off a non-empty band"
        spans.append((int(start), int(sep)))
        start = int(sep) + 1
    assert start < height, "final separator must leave a non-empty bottom band"
    spans.append((int(start), int(height)))
    return spans


def _majority_non_black(block: np.ndarray) -> int:
    values = [int(v) for v in block.ravel() if int(v) != BLACK]
    assert values, "each band must contain non-black pixels"
    return int(Counter(values).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """Use separator-row count as the run length and band majorities as the palette for the rightmost target column.

    Input structure:
    - The grid contains 2-4 full-width horizontal black separator rows.
    - Those separator rows split the non-black area into 3-5 horizontal bands.
    - Each band has one non-black majority color, while the rightmost column is entirely black.

    Rule:
    - Count the full-width black separator rows and call that number K.
    - For each band between separators, compute its most frequent non-black color and read those majorities top-to-bottom as the palette.
    - Fill the rightmost column from top to bottom by cycling through that palette, repeating each palette color for K consecutive rows.
    - Leave every other cell unchanged.

    Key insight:
    - The same separator rows provide both needed signals: their count gives the vertical run length, and the bands they carve out provide the ordered palette through band-majority extraction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected at least one content column and one target column"
    assert np.all(g[:, -1] == BLACK), "expected the rightmost column to be the black target column"

    separators = _separator_rows(g)
    assert MIN_SEPARATORS <= len(separators) <= MAX_SEPARATORS, "expected 2-4 separator rows"
    spans = _band_spans(g.shape[0], separators)
    palette = [_majority_non_black(g[r0:r1, :-1]) for r0, r1 in spans]

    run_len = len(separators)
    out = g.copy()
    for r in range(g.shape[0]):
        out[r, -1] = palette[(r // run_len) % len(palette)]
    return out


def _random_walk_profile(width: int, max_depth: int) -> np.ndarray:
    if max_depth <= 0:
        return np.zeros(width, dtype=int)

    level = random.randint(0, max_depth)
    profile = np.zeros(width, dtype=int)
    touched = False
    for c in range(width):
        if c > 0:
            lo = max(0, level - 1)
            hi = min(max_depth, level + 1)
            level = random.randint(lo, hi)
            if random.random() < 0.2:
                level = random.randint(0, max_depth)
        profile[c] = level
        touched = touched or level > 0

    if not touched:
        profile[random.randrange(width)] = max_depth
    return profile


def _paint_jagged_intrusion(
    grid: np.ndarray,
    r0: int,
    r1: int,
    width: int,
    side: str,
    color: int,
    max_depth: int,
) -> None:
    profile = _random_walk_profile(width, max_depth)
    for c, depth in enumerate(profile):
        depth = int(depth)
        if depth <= 0:
            continue
        if side == "top":
            grid[r0 : r0 + depth, c] = int(color)
        else:
            grid[r1 - depth : r1, c] = int(color)


def _apply_band_noise(
    grid: np.ndarray,
    spans: list[tuple[int, int]],
    band_colors: list[int],
    band_index: int,
    width: int,
) -> None:
    r0, r1 = spans[band_index]
    band_height = r1 - r0
    max_depth = min(2, (band_height - 1) // 2)
    if max_depth <= 0:
        return

    neighbor_colors = []
    if band_index > 0:
        neighbor_colors.append(int(band_colors[band_index - 1]))
    if band_index + 1 < len(band_colors):
        neighbor_colors.append(int(band_colors[band_index + 1]))
    if not neighbor_colors:
        neighbor_colors = [color for color in band_colors if color != band_colors[band_index]]

    sides = ["top", "bottom"]
    random.shuffle(sides)
    side_count = 1 if band_height == BAND_HEIGHT_RANGE[0] or random.random() < 0.65 else 2
    for side in sides[:side_count]:
        intrusion_color = random.choice(neighbor_colors)
        depth = random.randint(1, max_depth)
        _paint_jagged_intrusion(grid, r0, r1, width, side, intrusion_color, depth)


def _build_expected_output(inp: np.ndarray, palette: list[int], run_len: int) -> np.ndarray:
    out = inp.copy()
    for r in range(inp.shape[0]):
        out[r, -1] = int(palette[(r // run_len) % len(palette)])
    return out


def _band_majority_fraction(block: np.ndarray, color: int) -> float:
    non_black = block[block != BLACK]
    assert non_black.size > 0
    return float(np.count_nonzero(non_black == int(color)) / non_black.size)


def _valid_candidate(inp: np.ndarray, palette: list[int], separators: list[int]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not np.all(inp[:, -1] == BLACK):
        return False
    if _separator_rows(inp) != separators:
        return False

    spans = _band_spans(inp.shape[0], separators)
    inferred_palette = []
    for (r0, r1), expected_color in zip(spans, palette):
        band = inp[r0:r1, :-1]
        inferred = _majority_non_black(band)
        if inferred != int(expected_color):
            return False
        if _band_majority_fraction(band, expected_color) <= 0.60:
            return False
        inferred_palette.append(inferred)

    expected = _build_expected_output(inp, inferred_palette, len(separators))
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    separator_count = random.randint(MIN_SEPARATORS, MAX_SEPARATORS)
    band_count = separator_count + 1
    width = random.randint(*CONTENT_WIDTH_RANGE)
    band_heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(band_count)]
    height = sum(band_heights) + separator_count
    if height > 30:
        return None

    band_colors = random.sample(PALETTE_POOL, band_count)
    inp = np.zeros((height, width + 1), dtype=int)

    separators: list[int] = []
    spans: list[tuple[int, int]] = []
    row = 0
    for i, band_height in enumerate(band_heights):
        r0 = row
        r1 = row + band_height
        spans.append((int(r0), int(r1)))
        inp[r0:r1, :width] = int(band_colors[i])
        row = r1
        if i != band_count - 1:
            separators.append(int(row))
            row += 1

    for band_index in range(band_count):
        _apply_band_noise(inp, spans, band_colors, band_index, width)

    if not _valid_candidate(inp, band_colors, separators):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
