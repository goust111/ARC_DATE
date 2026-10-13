from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
OBJECT_TRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_1d.tolist() + [False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    return runs


def _analyze_bands(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    mid = h // 2
    lower = g[mid:, :]
    col_mask = (lower != BLACK).any(axis=0)
    spans = _runs(col_mask)
    assert 4 <= len(spans) <= 5, "expected 4-5 lower-half bands"

    bands: list[dict[str, int]] = []
    for idx, (c0, c1) in enumerate(spans):
        width = c1 - c0
        assert 2 <= width <= 3, "band width must be 2-3"
        if idx:
            gap = c0 - spans[idx - 1][1]
            assert 1 <= gap <= 2, "separator width must be 1-2"
        sub = lower[:, c0:c1]
        colors = [int(v) for v in sub[sub != BLACK]]
        assert colors, "band must contain foreground cells"
        color, _ = Counter(colors).most_common(1)[0]
        assert set(colors) == {color}, "each band must use exactly one foreground color"
        bands.append({"c0": c0, "c1": c1, "color": int(color)})

    return mid, bands


def _fill_from_counts(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    mid, bands = _analyze_bands(g)
    upper = g[:mid, :]
    counts: dict[int, int] = {}
    for band in bands:
        color = int(band["color"])
        counts[color] = len(_components_mask(upper == color))
    assert all(v > 0 for v in counts.values()), "every band color must appear above as at least one object"
    assert len(set(counts.values())) == len(counts), "object counts must be unique across colors"

    out = g.copy()
    n = len(bands)
    for i, band in enumerate(bands):
        src_color = int(band["color"])
        shift = counts[src_color]
        fill_color = int(bands[(i + shift) % n]["color"])
        c0, c1 = int(band["c0"]), int(band["c1"])
        region = out[mid:, c0:c1]
        region[region == BLACK] = fill_color
    return out, counts, bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill lower vertical bands by borrowing colors from cyclically shifted neighbors whose offsets come from upper-half object counts.

    Input structure:
    - The grid has black background, with 4-5 colored vertical bands confined to the lower half.
    - Each band occupies a contiguous 2-3 column span, bands are ordered left-to-right, and 1-2 black separator columns sit between neighboring bands.
    - The upper half contains scattered small 4-connected monochrome objects whose colors match the band colors.

    Rule:
    - Count, in the upper half only, how many separate 4-connected objects exist for each band color.
    - Order the lower-half bands from left to right as a cyclic list.
    - For each band, shift right by that color's object count within the cyclic band list and take the destination band's color.
    - Replace only the black gap cells inside the source band with that destination color; preserve all existing non-black pixels and everything outside the bands.

    Key insight:
    - The upper objects do not move, but their per-color component counts determine how far each lower band must rotate through the left-to-right cycle before borrowing its fill color.
    """
    out, _counts, _bands = _fill_from_counts(np.asarray(grid, dtype=int))
    return out


_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for rot in range(4):
        turned = np.rot90(shape, rot)
        for flip in (False, True):
            candidate = np.fliplr(turned) if flip else turned
            if not any(candidate.shape == prev.shape and np.array_equal(candidate, prev) for prev in variants):
                variants.append(candidate.copy())
    return variants


_ALL_SHAPES = tuple(tuple(_shape_variants(shape)) for shape in _SHAPES)


def _sample_band_pattern(height: int, width: int) -> np.ndarray:
    if width == 2:
        row_patterns = [(1, 1), (1, 0), (0, 1)]
    else:
        row_patterns = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0)]

    for _ in range(200):
        rows = np.zeros((height, width), dtype=bool)
        for r in range(height):
            rows[r] = np.array(random.choice(row_patterns), dtype=bool)
        if not (~rows).any():
            continue
        if not rows.any(axis=0).all():
            continue
        if any(row.any() for row in rows):
            return rows
    raise RuntimeError("failed to sample band pattern")


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    top_rows: int,
    color: int,
) -> bool:
    h, w = grid.shape
    shape_family = random.choice(_ALL_SHAPES)
    shape = random.choice(shape_family)
    sh, sw = shape.shape
    if top_rows < sh:
        return False

    for _ in range(OBJECT_TRIES):
        r0 = random.randint(0, top_rows - sh)
        c0 = random.randint(0, w - sw)
        cells: list[tuple[int, int]] = []
        ok = True
        for rr, cc in np.argwhere(shape):
            r = r0 + int(rr)
            c = c0 + int(cc)
            if blocked[r, c] or grid[r, c] != BLACK:
                ok = False
                break
            cells.append((r, c))
        if not ok:
            continue
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells)
        return True
    return False


def _generate_one() -> np.ndarray | None:
    band_count = random.choice((4, 5))
    heights = [18, 20, 22, 24, 26, 28]
    h = random.choice(heights)
    mid = h // 2
    band_height = h - mid
    widths = [random.randint(2, 3) for _ in range(band_count)]
    gaps = [random.randint(1, 2) for _ in range(band_count - 1)]
    needed_w = sum(widths) + sum(gaps)
    min_slack = max(0, 18 - needed_w)
    max_slack = max(0, 30 - needed_w)
    if min_slack > max_slack:
        return None
    side_slack = random.randint(min_slack, max_slack)
    left_margin = random.randint(0, side_slack)
    right_margin = side_slack - left_margin
    w = needed_w + left_margin + right_margin
    assert 18 <= w <= 30

    colors = random.sample(PALETTE, band_count)

    while True:
        counts_list = random.sample(range(1, band_count + 1), band_count)
        if counts_list == sorted(counts_list) or counts_list == sorted(counts_list, reverse=True):
            continue
        targets = {(i + counts_list[i]) % band_count for i in range(band_count)}
        if len(targets) < 3:
            continue
        break
    counts_by_color = {color: count for color, count in zip(colors, counts_list)}

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[mid - 1 :, :] = True

    for color in colors:
        for _ in range(counts_by_color[color]):
            if not _place_object(grid, blocked, mid - 1, color):
                return None

    spans: list[tuple[int, int, int]] = []
    c = left_margin
    for idx, (width, color) in enumerate(zip(widths, colors)):
        spans.append((c, c + width, color))
        c += width
        if idx < len(gaps):
            c += gaps[idx]

    for c0, c1, color in spans:
        mask = _sample_band_pattern(band_height, c1 - c0)
        band = np.where(mask, color, BLACK)
        grid[mid:, c0:c1] = band

    if not (grid[:mid - 1] != BLACK).any():
        return None
    if (grid[mid - 1] != BLACK).any():
        return None

    expected = grid.copy()
    for idx, (c0, c1, color) in enumerate(spans):
        fill_color = spans[(idx + counts_by_color[color]) % band_count][2]
        region = expected[mid:, c0:c1]
        region[region == BLACK] = fill_color

    actual, found_counts, found_bands = _fill_from_counts(grid)
    if not np.array_equal(actual, expected):
        return None
    if [band["color"] for band in found_bands] != colors:
        return None
    if [found_counts[color] for color in colors] != counts_list:
        return None
    if np.array_equal(actual, grid):
        return None

    # Counterexamples against fixed-shift shortcuts: at least one self-fill and one shift > 1.
    if band_count not in counts_list or max(counts_list) <= 1:
        return None

    upper_counts = sorted(len(comp) for color in colors for comp in _components_mask(grid[:mid, :] == color))
    if min(upper_counts) < 2 or max(upper_counts) > 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
