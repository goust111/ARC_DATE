from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

SEPARATOR = GREY
STRIPE_PALETTE = (PINK, 7, 8, 9)
LENGTH_TO_FILL = {2: RED, 3: BLUE, 4: YELLOW}
CORNERS = ("tl", "tr", "bl", "br")

GEN_RETRIES = 400
CORNER_BAND_MIN = 4
CORNER_BAND_MAX = 9
MIDDLE_BAND_MIN = 3
MIDDLE_BAND_MAX = 7


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    cuts = sorted(int(x) for x in separators)
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in cuts:
        assert start < cut < size
        bands.append((start, cut))
        start = cut + 1
    assert start < size
    bands.append((start, size))
    assert len(bands) == 3
    return bands


def _collapse(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if not out or value != out[-1]:
            out.append(value)
    return out


def _stripe_sequence(region: np.ndarray) -> list[int]:
    h, w = region.shape
    rows_constant = all(np.all(region[r, :] == region[r, 0]) for r in range(h))
    cols_constant = all(np.all(region[:, c] == region[0, c]) for c in range(w))
    assert rows_constant != cols_constant, "corner region must contain horizontal or vertical full-span stripes"
    if rows_constant:
        return _collapse([int(region[r, 0]) for r in range(h)])
    return _collapse([int(region[0, c]) for c in range(w)])


def _corner_lengths(grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> dict[str, int]:
    regions = {
        "tl": (0, 0),
        "tr": (0, 2),
        "bl": (2, 0),
        "br": (2, 2),
    }
    lengths: dict[str, int] = {}
    for name, (ri, ci) in regions.items():
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        region = grid[r0:r1, c0:c1]
        seq = _stripe_sequence(region)
        assert len(seq) in LENGTH_TO_FILL, "compressed stripe length must be 2, 3, or 4"
        lengths[name] = len(seq)
    return lengths


def _cross_colors(lengths: dict[str, int]) -> dict[str, int]:
    corner_fills = {name: LENGTH_TO_FILL[lengths[name]] for name in CORNERS}

    def merge(a: str, b: str) -> int:
        ca = corner_fills[a]
        cb = corner_fills[b]
        return ca if ca == cb else GREEN

    return {
        "tm": merge("tl", "tr"),
        "ml": merge("tl", "bl"),
        "mr": merge("tr", "br"),
        "bm": merge("bl", "br"),
    }


def _unique_mode(values: list[int]) -> int | None:
    counts = Counter(int(v) for v in values)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    return None


def _fill_region(out: np.ndarray, row_band: tuple[int, int], col_band: tuple[int, int], color: int) -> None:
    r0, r1 = row_band
    c0, c1 = col_band
    out[r0:r1, c0:c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress corner stripes inside a separator-defined 3x3 lattice and propagate the stripe counts into the plus-shaped regions.

    Input structure:
    - The grid has black background with exactly two full-width grey rows and two full-height grey columns that divide it into nine rectangular regions.
    - Each corner region contains either horizontal full-width stripes or vertical full-height stripes with thickness 1-3 and no adjacent equal colors.
    - The four edge-middle regions and the center region start as black.

    Rule:
    - Keep the four corner regions and the grey separators unchanged.
    - In each corner, collapse consecutive duplicate stripe rows or columns to get a deduplicated stripe sequence; length 2 maps to red, length 3 maps to blue, and length 4 maps to yellow.
    - Fill each edge-middle region from its two adjacent corners: if both mapped colors agree use that color, otherwise use green; then fill the center with the unique majority edge-middle color, or pink if no unique majority exists.

    Key insight:
    - The visible stripes are not copied directly: each corner is first locally compressed to a stripe count, and only that count-controlled color is transmitted into neighboring regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEPARATOR)]
    assert len(sep_rows) == 2 and len(sep_cols) == 2

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    lengths = _corner_lengths(g, row_bands, col_bands)
    cross = _cross_colors(lengths)

    center = _unique_mode([cross["tm"], cross["ml"], cross["mr"], cross["bm"]])
    center_color = PINK if center is None else center

    out = g.copy()
    _fill_region(out, row_bands[0], col_bands[1], cross["tm"])
    _fill_region(out, row_bands[1], col_bands[0], cross["ml"])
    _fill_region(out, row_bands[1], col_bands[2], cross["mr"])
    _fill_region(out, row_bands[2], col_bands[1], cross["bm"])
    _fill_region(out, row_bands[1], col_bands[1], center_color)
    return out


def _random_composition(total: int, parts: int, lo: int = 1, hi: int = 3) -> list[int]:
    valid: list[list[int]] = []

    def rec(remaining: int, needed: int, prefix: list[int]) -> None:
        if needed == 1:
            if lo <= remaining <= hi:
                valid.append(prefix + [remaining])
            return
        for value in range(lo, hi + 1):
            rest = remaining - value
            if lo * (needed - 1) <= rest <= hi * (needed - 1):
                rec(rest, needed - 1, prefix + [value])

    rec(int(total), int(parts), [])
    assert valid, f"no stripe thickness split for total={total}, parts={parts}"
    return random.choice(valid)


def _random_stripe_colors(length: int) -> list[int]:
    colors: list[int] = []
    for _ in range(length):
        choices = [c for c in STRIPE_PALETTE if not colors or c != colors[-1]]
        colors.append(int(random.choice(choices)))
    return colors


def _paint_corner(region: np.ndarray, orientation: str, stripe_count: int) -> None:
    h, w = region.shape
    if orientation == "h":
        thicknesses = _random_composition(h, stripe_count)
        colors = _random_stripe_colors(stripe_count)
        start = 0
        for thickness, color in zip(thicknesses, colors):
            region[start : start + thickness, :] = color
            start += thickness
        return

    assert orientation == "v"
    thicknesses = _random_composition(w, stripe_count)
    colors = _random_stripe_colors(stripe_count)
    start = 0
    for thickness, color in zip(thicknesses, colors):
        region[:, start : start + thickness] = color
        start += thickness


def _pick_lengths() -> dict[str, int]:
    branch = random.choices(
        ["three_same", "row_pairs", "col_pairs", "general"],
        weights=[4, 3, 3, 4],
        k=1,
    )[0]
    values = list(LENGTH_TO_FILL)

    if branch == "three_same":
        base = random.choice(values)
        other = random.choice([v for v in values if v != base])
        odd = random.choice(CORNERS)
        lengths = {name: base for name in CORNERS}
        lengths[odd] = other
        return lengths

    if branch == "row_pairs":
        top = random.choice(values)
        bottom = random.choice([v for v in values if v != top])
        return {"tl": top, "tr": top, "bl": bottom, "br": bottom}

    if branch == "col_pairs":
        left = random.choice(values)
        right = random.choice([v for v in values if v != left])
        return {"tl": left, "bl": left, "tr": right, "br": right}

    while True:
        lengths = {name: random.choice(values) for name in CORNERS}
        if len(set(lengths.values())) > 1:
            return lengths


def _pick_band_size(required_lengths: list[int]) -> int | None:
    lo = CORNER_BAND_MIN
    hi = CORNER_BAND_MAX
    for length in required_lengths:
        lo = max(lo, int(length))
        hi = min(hi, 3 * int(length))
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _expected_center_type(cross: dict[str, int]) -> int:
    return PINK if _unique_mode(list(cross.values())) is None else _unique_mode(list(cross.values()))


def _build_input() -> np.ndarray | None:
    lengths = _pick_lengths()
    orientations = {name: random.choice(["h", "v"]) for name in CORNERS}

    top_h = _pick_band_size([lengths[name] for name in ("tl", "tr") if orientations[name] == "h"])
    bot_h = _pick_band_size([lengths[name] for name in ("bl", "br") if orientations[name] == "h"])
    left_w = _pick_band_size([lengths[name] for name in ("tl", "bl") if orientations[name] == "v"])
    right_w = _pick_band_size([lengths[name] for name in ("tr", "br") if orientations[name] == "v"])
    if None in (top_h, bot_h, left_w, right_w):
        return None

    mid_h = random.randint(MIDDLE_BAND_MIN, MIDDLE_BAND_MAX)
    mid_w = random.randint(MIDDLE_BAND_MIN, MIDDLE_BAND_MAX)
    h = top_h + 1 + mid_h + 1 + bot_h
    w = left_w + 1 + mid_w + 1 + right_w
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    sep_rows = [top_h, top_h + 1 + mid_h]
    sep_cols = [left_w, left_w + 1 + mid_w]
    grid[sep_rows, :] = SEPARATOR
    grid[:, sep_cols] = SEPARATOR

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    corner_pos = {
        "tl": (0, 0),
        "tr": (0, 2),
        "bl": (2, 0),
        "br": (2, 2),
    }
    for name, (ri, ci) in corner_pos.items():
        r0, r1 = row_bands[ri]
        c0, c1 = col_bands[ci]
        _paint_corner(grid[r0:r1, c0:c1], orientations[name], lengths[name])

    actual_lengths = _corner_lengths(grid, row_bands, col_bands)
    if actual_lengths != lengths:
        return None

    cross = _cross_colors(lengths)
    center = _expected_center_type(cross)
    has_conflict = any(color == GREEN for color in cross.values())
    has_agreement = any(color in (RED, BLUE, YELLOW) for color in cross.values())
    if not has_conflict or not has_agreement:
        return None
    if len(set(lengths.values())) == 1:
        return None
    if center == PINK and Counter(cross.values()).most_common(1)[0][1] != 2:
        return None

    out = main(grid)
    if not np.array_equal(out[: row_bands[0][1], : col_bands[0][1]], grid[: row_bands[0][1], : col_bands[0][1]]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
