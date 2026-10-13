import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
INDICATOR_COLORS = (1, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 1000
GEN_BAND_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _separator_rows(grid: np.ndarray, bg: int) -> tuple[int | None, list[int]]:
    sep_color = None
    rows: list[int] = []
    for r in range(grid.shape[0]):
        row_color = int(grid[r, 0])
        if row_color == bg:
            continue
        if np.all(grid[r, :] == row_color):
            rows.append(r)
            if sep_color is None:
                sep_color = row_color
            else:
                assert row_color == sep_color, "separator rows must share one color"
    return sep_color, rows


def _band_ranges(height: int, sep_rows: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [r + 1 for r in sep_rows]
    ends = sep_rows + [height]
    return [(start, end) for start, end in zip(starts, ends) if start < end]


def _indicator_color(band: np.ndarray, bg: int, sep_color: int | None) -> int:
    forbidden = {bg}
    if sep_color is not None:
        forbidden.add(sep_color)
    candidates = [
        int(band[r, 0]) for r in range(band.shape[0]) if int(band[r, 0]) not in forbidden
    ]
    assert len(candidates) == 1, "each band must contain exactly one left-edge indicator"
    color = candidates[0]
    assert color != RED, "indicator color must differ from the red object"
    return color


def _axis_score(red_mask: np.ndarray, forbidden: np.ndarray, axis_twice: int) -> tuple[int, int, int]:
    left_excess = 0
    right_excess = 0
    invalid = 0
    for r_raw, c_raw in np.argwhere(red_mask):
        r = int(r_raw)
        c = int(c_raw)
        cc = axis_twice - c
        if cc < 0 or cc >= red_mask.shape[1]:
            invalid += 1
            if 2 * c < axis_twice:
                left_excess += 1
            elif 2 * c > axis_twice:
                right_excess += 1
            continue
        cc = int(cc)
        if not red_mask[r, cc]:
            if forbidden[r, cc]:
                invalid += 1
            if 2 * c < axis_twice:
                left_excess += 1
            elif 2 * c > axis_twice:
                right_excess += 1
            else:
                invalid += 1
    return invalid, min(left_excess, right_excess), left_excess + right_excess


def _rank_axes(red_mask: np.ndarray, forbidden: np.ndarray) -> list[tuple[tuple[int, int, int], int]]:
    scores = []
    for axis_twice in range(2 * red_mask.shape[1] - 1):
        scores.append((_axis_score(red_mask, forbidden, axis_twice), axis_twice))
    scores.sort(key=lambda item: (item[0], item[1]))
    return scores


def _complete_band(band: np.ndarray, bg: int, indicator: int) -> np.ndarray:
    red_mask = band == RED
    components = _components(red_mask)
    assert len(components) == 1, "each band must contain exactly one red object"

    forbidden = (band != bg) & ~red_mask
    best_score, axis_twice = _rank_axes(red_mask, forbidden)[0]
    assert best_score[0] == 0, "best symmetry axis must stay in bounds and avoid markers"

    out = band.copy()
    for r_raw, c_raw in np.argwhere(red_mask):
        r = int(r_raw)
        c = int(c_raw)
        cc = int(axis_twice - c)
        assert 0 <= cc < band.shape[1]
        if out[r, cc] == bg:
            out[r, cc] = indicator
        else:
            assert out[r, cc] == RED, "mirrored completion may only overlap the original red object"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete red object by vertical reflection, coloring only the newly inferred half with the row-band indicator.

    Input structure:
    - The background is black.
    - One or more full-width grey separator rows split the grid into horizontal bands.
    - Each band contains exactly one 1x1 indicator pixel in the leftmost column and exactly one 4-connected red object elsewhere.
    - Each red object is a vertically near-symmetric shape with some pixels missing from only one side of its hidden symmetry axis.

    Rule:
    - Process each horizontal band independently.
    - Find the vertical reflection axis that makes the red object closest to a one-sided incomplete symmetric shape.
    - Reflect the existing red pixels across that axis to infer the missing counterpart pixels.
    - Keep the original red pixels red, but color only the newly added reflected pixels with that band's left-edge indicator color.

    Key insight:
    - The red object reveals the symmetric geometry, but the completion color is not part of the object itself; it is indexed by the separate marker placed on the left edge of the same band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_color, sep_rows = _separator_rows(g, BLACK)
    out = g.copy()
    for start, end in _band_ranges(g.shape[0], sep_rows):
        band = g[start:end, :]
        indicator = _indicator_color(band, BLACK, sep_color)
        out[start:end, :] = _complete_band(band, BLACK, indicator)
    return out


def _sample_band_heights(n_bands: int) -> list[int]:
    return [random.randint(4, 6) for _ in range(n_bands)]


def _sample_profile(obj_h: int, max_half: int) -> list[int]:
    profile = [random.randint(1, max_half)]
    for _ in range(1, obj_h):
        lo = max(1, profile[-1] - 2)
        hi = min(max_half, profile[-1] + 2)
        profile.append(random.randint(lo, hi))
    if len(set(profile)) == 1 and max_half > 1:
        idx = random.randrange(obj_h)
        profile[idx] = random.randint(1, max_half)
    return profile


def _row_intervals(profile: list[int], parity: int, left: int) -> tuple[list[tuple[int, int]], int]:
    max_half = max(profile)
    intervals: list[tuple[int, int]] = []
    if parity == 0:
        center = left + max_half
        axis_twice = 2 * center
        for half in profile:
            intervals.append((center - half, center + half))
    else:
        axis_twice = 2 * left + 2 * max_half - 1
        for half in profile:
            intervals.append((left + max_half - half, left + max_half + half - 1))
    return intervals, axis_twice


def _sample_trims(profile: list[int]) -> list[int]:
    obj_h = len(profile)
    trimmed_rows = set(random.sample(range(obj_h), k=random.randint(1, obj_h - 1)))
    trims = [0] * obj_h
    for idx in trimmed_rows:
        trims[idx] = random.randint(1, min(2, profile[idx]))
    return trims


def _gen_band(width: int, band_h: int, indicator: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(GEN_BAND_ATTEMPTS):
        obj_h = random.randint(3, band_h)
        parity = random.randint(0, 1)
        max_half_limit = min(5, (width - 5) // 2 if parity == 0 else (width - 4) // 2)
        if max_half_limit < 1:
            return None
        profile = _sample_profile(obj_h, max_half_limit)
        full_width = 2 * max(profile) + 1 if parity == 0 else 2 * max(profile)
        if full_width > width - 3:
            continue

        top = random.randint(0, band_h - obj_h)
        left = random.randint(2, width - full_width - 1)
        full_intervals, axis_twice = _row_intervals(profile, parity, left)
        trims = _sample_trims(profile)
        side = random.choice(("left", "right"))

        band = np.full((band_h, width), BLACK, dtype=int)
        expected = np.full((band_h, width), BLACK, dtype=int)
        red_mask = np.zeros((band_h, width), dtype=bool)
        full_mask = np.zeros((band_h, width), dtype=bool)

        for local_r, (base_left, base_right) in enumerate(full_intervals):
            r = top + local_r
            full_mask[r, base_left : base_right + 1] = True
            if side == "left":
                in_left, in_right = base_left + trims[local_r], base_right
            else:
                in_left, in_right = base_left, base_right - trims[local_r]
            if in_left > in_right:
                break
            red_mask[r, in_left : in_right + 1] = True
        else:
            if red_mask.sum() == full_mask.sum():
                continue
            if len(_components(red_mask)) != 1:
                continue

            indicator_row = random.randrange(band_h)
            band[indicator_row, 0] = indicator
            band[red_mask] = RED
            expected[indicator_row, 0] = indicator
            expected[red_mask] = RED
            expected[full_mask & ~red_mask] = indicator

            forbidden = (band != BLACK) & ~red_mask
            ranked = _rank_axes(red_mask, forbidden)
            if ranked[0][0][0] != 0 or ranked[0][1] != axis_twice:
                continue
            if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
                continue
            if not np.array_equal(_complete_band(band, BLACK, indicator), expected):
                continue
            return band, expected
    return None


def _gen_attempt() -> np.ndarray | None:
    n_bands = random.randint(2, 4)
    heights = _sample_band_heights(n_bands)
    total_h = sum(heights) + n_bands - 1
    if total_h > 30:
        return None

    width = random.randint(14, 24)
    indicators = random.sample(INDICATOR_COLORS, k=n_bands)

    grid = np.full((total_h, width), BLACK, dtype=int)
    expected = np.full((total_h, width), BLACK, dtype=int)

    row = 0
    for band_h, indicator in zip(heights, indicators):
        built = _gen_band(width, band_h, indicator)
        if built is None:
            return None
        band, band_expected = built
        grid[row : row + band_h, :] = band
        expected[row : row + band_h, :] = band_expected
        row += band_h
        if row < total_h:
            grid[row, :] = GREY
            expected[row, :] = GREY
            row += 1

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
