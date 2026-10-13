import random
from collections import Counter

import numpy as np


BLACK = 0
DIVIDER = 5
BAND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300

_SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _black_ratio(region: np.ndarray) -> float:
    return float(np.count_nonzero(region == BLACK)) / float(region.size)


def _majority_color(line: np.ndarray) -> tuple[int, float]:
    vals, counts = np.unique(line, return_counts=True)
    idx = int(np.argmax(counts))
    return int(vals[idx]), float(counts[idx]) / float(line.size)


def _compress_colors(seq: list[int]) -> tuple[list[int], list[int]]:
    colors: list[int] = []
    widths: list[int] = []
    for color in seq:
        if not colors or color != colors[-1]:
            colors.append(int(color))
            widths.append(1)
        else:
            widths[-1] += 1
    return colors, widths


def _split_by_separator(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    counts = Counter(map(int, grid.ravel()))
    candidates = []

    def score_regions(a: np.ndarray, b: np.ndarray, line_color: int, line_len: int) -> tuple[int, int, int, int, float]:
        a_black = _black_ratio(a)
        b_black = _black_ratio(b)
        band_like = int((a_black <= 0.02 and b_black >= 0.50) or (b_black <= 0.02 and a_black >= 0.50))
        isolated_color = int(np.count_nonzero(a == int(line_color)) == 0 and np.count_nonzero(b == int(line_color)) == 0)
        unique_line = int(counts[int(line_color)] == int(line_len))
        non_black = int(int(line_color) != BLACK)
        contrast = abs(a_black - b_black)
        return band_like, isolated_color, unique_line, non_black, contrast

    for r in range(1, h - 1):
        if np.all(grid[r] == grid[r, 0]):
            top = grid[:r]
            bottom = grid[r + 1 :]
            candidates.append((score_regions(top, bottom, int(grid[r, 0]), w), "row", r))

    for c in range(1, w - 1):
        if np.all(grid[:, c] == grid[0, c]):
            left = grid[:, :c]
            right = grid[:, c + 1 :]
            candidates.append((score_regions(left, right, int(grid[0, c]), h), "col", c))

    assert candidates, "expected a full separator line"
    _score, axis, idx = max(candidates, key=lambda item: item[0])

    if axis == "row":
        region_a = grid[:idx]
        region_b = grid[idx + 1 :]
    else:
        region_a = grid[:, :idx]
        region_b = grid[:, idx + 1 :]

    if _black_ratio(region_a) >= _black_ratio(region_b):
        motif, band = region_a.copy(), region_b.copy()
    else:
        motif, band = region_b.copy(), region_a.copy()

    assert motif.size > 0 and band.size > 0
    return motif, band


def _band_signature(region: np.ndarray, axis: int) -> dict:
    dominant: list[int] = []
    strengths: list[float] = []
    line_count = region.shape[axis]
    for idx in range(line_count):
        line = region[idx, :] if axis == 0 else region[:, idx]
        color, frac = _majority_color(line)
        dominant.append(color)
        strengths.append(frac)
    colors, widths = _compress_colors(dominant)
    palette = {int(v) for v in np.unique(region)}
    return {
        "axis": axis,
        "colors": colors,
        "widths": widths,
        "palette": palette,
        "mean_strength": float(np.mean(strengths)),
        "covers_palette": set(colors) == palette,
        "unique_once": len(colors) == len(set(colors)),
    }


def _extract_band_key(region: np.ndarray) -> tuple[list[int], list[int]]:
    signatures = [_band_signature(region, 0), _band_signature(region, 1)]

    def key(sig: dict) -> tuple[int, int, float, int]:
        return (
            int(sig["covers_palette"] and sig["unique_once"]),
            len(sig["colors"]),
            sig["mean_strength"],
            -sig["axis"],
        )

    best = max(signatures, key=key)
    assert best["covers_palette"] and best["unique_once"], "failed to recover unique band order"
    return list(best["colors"]), list(best["widths"])


def _build_mapping(colors: list[int], widths: list[int]) -> dict[int, int]:
    k = len(colors)
    assert k >= 2
    mapping: dict[int, int] = {}
    for idx, color in enumerate(colors):
        mapping[int(color)] = int(colors[(idx + widths[idx]) % k])
    return mapping


def _recolor_motif(motif: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = motif.copy()
    for src, dst in mapping.items():
        out[motif == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the motif region using a cyclic key encoded by compressed band widths, then return only the motif crop.

    Input structure:
    - A single full-row or full-column separator splits the grid into exactly two rectangular regions.
    - One region is a black-background motif whose non-black colors all come from the other region's palette.
    - The other region is fully covered by jagged color bands; along one unique axis, each color forms exactly one contiguous dominant band.

    Rule:
    - Identify the motif region and the band region, discarding the separator from the output.
    - Compress the band region along its band axis by taking the dominant color of each row or column, then merge consecutive repeats into a color order c1..ck with band widths w1..wk.
    - Build the cyclic recoloring key ci -> c((i + wi) mod k), apply it to every non-black motif pixel, and keep black unchanged.

    Key insight:
    - The band panel is not copied to the output; it is a spatially written substitution cipher where each band's own thickness is the step size for that band's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    motif, band = _split_by_separator(g)
    band_colors, band_widths = _extract_band_key(band)
    mapping = _build_mapping(band_colors, band_widths)

    motif_colors = {int(v) for v in np.unique(motif) if int(v) != BLACK}
    assert motif_colors <= set(mapping), "motif colors must be encoded by the band palette"
    return _recolor_motif(motif, mapping)


def _sample_widths(min_total: int, max_total: int) -> list[int]:
    for _ in range(200):
        band_count = random.randint(4, 6)
        widths = [random.randint(1, 4) for _ in range(band_count)]
        total = sum(widths)
        if min_total <= total <= max_total and any(w > 1 for w in widths):
            return widths
    raise RuntimeError("failed to sample band widths")


def _random_shape() -> np.ndarray:
    shape = np.array(random.choice(_SHAPES), copy=True)
    for _ in range(random.randrange(4)):
        shape = np.rot90(shape)
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    return shape


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _make_motif_region(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    region = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    assignments = list(colors)
    assignments.extend(random.choices(colors, k=random.randint(1, 3)))
    random.shuffle(assignments)

    for color in assignments:
        placed = False
        for _ in range(120):
            shape = _random_shape()
            sh, sw = shape.shape
            if sh > height or sw > width:
                continue
            r0 = random.randint(0, height - sh)
            c0 = random.randint(0, width - sw)
            rel = np.argwhere(shape > 0)
            cells = [(r0 + int(r), c0 + int(c)) for r, c in rel]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                region[r, c] = int(color)
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            return None

    used = {int(v) for v in np.unique(region) if int(v) != BLACK}
    colored = int(np.count_nonzero(region))
    if used != set(colors):
        return None
    if colored < len(colors) * 2 or colored > int(region.size * 0.45):
        return None
    return region


def _make_band_region(height: int, width: int, axis: int, colors: list[int], widths: list[int]) -> np.ndarray:
    region = np.zeros((height, width), dtype=int)
    cursor = 0

    if axis == 0:
        assert height == sum(widths)
        for color, band_w in zip(colors, widths):
            region[cursor : cursor + band_w, :] = int(color)
            cursor += band_w
        cursor = 0
        for i, band_w in enumerate(widths[:-1]):
            cursor += band_w
            upper = int(colors[i])
            lower = int(colors[i + 1])
            n = random.randint(1, max(1, width // 4))
            cols = random.sample(range(width), k=min(width, 2 * n))
            region[cursor - 1, cols[::2]] = lower
            region[cursor, cols[1::2]] = upper
    else:
        assert width == sum(widths)
        for color, band_w in zip(colors, widths):
            region[:, cursor : cursor + band_w] = int(color)
            cursor += band_w
        cursor = 0
        for i, band_w in enumerate(widths[:-1]):
            cursor += band_w
            left = int(colors[i])
            right = int(colors[i + 1])
            n = random.randint(1, max(1, height // 4))
            rows = random.sample(range(height), k=min(height, 2 * n))
            region[rows[::2], cursor - 1] = right
            region[rows[1::2], cursor] = left

    return region


def _expected_output(motif: np.ndarray, colors: list[int], widths: list[int]) -> np.ndarray:
    return _recolor_motif(motif, _build_mapping(colors, widths))


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    divider_axis = random.choice(("row", "col"))
    band_axis = random.choice((0, 1))
    side_first = random.choice((True, False))

    if divider_axis == "row":
        if band_axis == 0:
            widths = _sample_widths(6, 13)
            band_h = sum(widths)
            band_w = random.randint(9, 15)
        else:
            widths = _sample_widths(9, 15)
            band_w = sum(widths)
            band_h = random.randint(7, 12)
        motif_h = random.randint(8, 12)
        motif_w = band_w
        grid_h = band_h + 1 + motif_h
        grid_w = band_w
    else:
        if band_axis == 0:
            widths = _sample_widths(9, 15)
            band_h = sum(widths)
            band_w = random.randint(7, 12)
        else:
            widths = _sample_widths(6, 13)
            band_w = sum(widths)
            band_h = random.randint(9, 15)
        motif_h = band_h
        motif_w = random.randint(8, 12)
        grid_h = band_h
        grid_w = band_w + 1 + motif_w

    assert max(grid_h, grid_w) <= 30

    colors = random.sample(BAND_COLOR_POOL, len(widths))
    motif = _make_motif_region(motif_h, motif_w, colors)
    if motif is None:
        raise RuntimeError("motif placement failed")

    band = _make_band_region(band_h, band_w, band_axis, colors, widths)
    expected = _expected_output(motif, colors, widths)
    assert not np.array_equal(expected, motif)

    grid = np.full((grid_h, grid_w), DIVIDER, dtype=int)
    if divider_axis == "row":
        if side_first:
            grid[:band_h, :] = band
            grid[band_h + 1 :, :] = motif
        else:
            grid[:motif_h, :] = motif
            grid[motif_h + 1 :, :] = band
    else:
        if side_first:
            grid[:, :band_w] = band
            grid[:, band_w + 1 :] = motif
        else:
            grid[:, :motif_w] = motif
            grid[:, motif_w + 1 :] = band

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue

        if np.array_equal(main(grid), expected):
            motif, band = _split_by_separator(grid)
            box = _bbox(motif != BLACK)
            assert box is not None
            assert _crop_bbox(motif, box).size > 0
            return grid

    raise RuntimeError("failed to generate a valid color-band key puzzle")
