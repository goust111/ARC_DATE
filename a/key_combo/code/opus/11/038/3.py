from __future__ import annotations

import math
import random

import numpy as np


BG = 0
SEP = 5
BAND_COLORS = (1, 2, 3)
TRIPLE_COLOR = 4

BIT_CODES = (1, 2, 4)
ALL_REGION_CODES = (7, 3, 5, 6, 1, 2, 4)
PAIR_CODES = (3, 5, 6)
SINGLE_CODES = (1, 2, 4)

CODE_TO_PAIR = {
    3: (0, 1),
    5: (0, 2),
    6: (1, 2),
}
GROW_DIR = {
    7: (0.0, 0.0),
    3: (-1.0, 0.0),
    5: (0.0, -1.0),
    6: (0.0, 1.0),
    1: (1.0, -1.0),
    2: (1.0, 0.0),
    4: (1.0, 1.0),
}

MAX_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_bands(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == SEP)]
    assert len(sep_rows) == 2, "expected exactly two full separator rows"
    s0, s1 = sep_rows
    assert 0 < s0 < s1 < g.shape[0] - 1, "separators must split three non-empty bands"
    h0 = s0
    h1 = s1 - s0 - 1
    h2 = g.shape[0] - s1 - 1
    assert h0 == h1 == h2, "bands must have equal height"
    bands = [g[:s0], g[s0 + 1 : s1], g[s1 + 1 :]]
    assert all(np.all(band != SEP) for band in bands), "separator color must not appear inside bands"
    return bands


def _masks_from_bands(bands: list[np.ndarray]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    shape = bands[0].shape
    for band, color in zip(bands, BAND_COLORS):
        assert band.shape == shape, "all bands must share one shape"
        assert set(map(int, np.unique(band))) <= {BG, color}, "band must be a binary mask in its own color"
        mask = band == color
        assert np.any(mask), "each band needs foreground"
        masks.append(mask)
    return masks


def _winner(i: int, j: int, areas: list[int]) -> int:
    return i if areas[i] >= areas[j] else j


def _expected_output_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    areas = [int(mask.sum()) for mask in masks]
    stacked = np.stack(masks, axis=0)
    count = stacked.sum(axis=0)

    out = np.zeros(masks[0].shape, dtype=int)
    for idx, color in enumerate(BAND_COLORS):
        out[(count == 1) & masks[idx]] = color

    for i, j in ((0, 1), (0, 2), (1, 2)):
        both = (count == 2) & masks[i] & masks[j]
        out[both] = BAND_COLORS[_winner(i, j, areas)]

    out[count == 3] = TRIPLE_COLOR
    return out


def _labels_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros(masks[0].shape, dtype=int)
    for bit, mask in zip(BIT_CODES, masks):
        labels |= bit * mask.astype(int)
    return labels


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_irregular(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return int(mask.sum()) < (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse three separator-delimited masks and recolor double overlaps by global area ranking.

    Input structure:
    - The grid contains three equal-height horizontal bands separated by two full grey rows.
    - Band 1 uses blue foreground on black background, band 2 uses red foreground on black background, and band 3 uses green foreground on black background.
    - Each band is a same-size binary mask, aligned by position across the three bands.

    Rule:
    - Remove the two grey separator rows and align the three bands into one output grid of band size.
    - For each position, count how many bands contain foreground there.
    - Output black for count 0, the unique band color for count 1, yellow for count 3, and for count 2 choose the foreground color of the larger-area competing band, breaking area ties in favor of the higher band in the input.

    Key insight:
    - A double-overlap pixel cannot be resolved locally; it depends on the global foreground areas of the two participating masks.
    """
    bands = _extract_bands(grid)
    masks = _masks_from_bands(bands)
    return _expected_output_from_masks(masks)


def _sample_sizes(area: int, tie_pair: tuple[int, int] | None) -> tuple[dict[int, int], list[int]] | None:
    low = math.ceil(0.30 * area)
    high = math.floor(0.60 * area)
    pair_lo = max(2, area // 14)
    pair_hi = min(9, max(pair_lo + 1, area // 6))
    single_lo = 2
    single_hi = min(10, max(single_lo + 1, area // 5))
    bg_min = max(5, area // 7)

    for _ in range(400):
        sizes = {
            7: random.randint(4, min(6, max(4, area // 8))),
            3: random.randint(pair_lo, pair_hi),
            5: random.randint(pair_lo, pair_hi),
            6: random.randint(pair_lo, pair_hi),
            1: random.randint(single_lo, single_hi),
            2: random.randint(single_lo, single_hi),
            4: random.randint(single_lo, single_hi),
        }
        band_areas = [
            sizes[1] + sizes[3] + sizes[5] + sizes[7],
            sizes[2] + sizes[3] + sizes[6] + sizes[7],
            sizes[4] + sizes[5] + sizes[6] + sizes[7],
        ]
        union = sum(sizes.values())
        if union > area - bg_min:
            continue
        if any(not low <= a <= high for a in band_areas):
            continue
        if tie_pair is None:
            if len(set(band_areas)) != 3:
                continue
            if min(abs(band_areas[i] - band_areas[j]) for i, j in ((0, 1), (0, 2), (1, 2))) < 2:
                continue
        else:
            i, j = tie_pair
            k = ({0, 1, 2} - {i, j}).pop()
            if band_areas[i] != band_areas[j]:
                continue
            if band_areas[k] == band_areas[i] or abs(band_areas[k] - band_areas[i]) < 2:
                continue

        winners = {code: _winner(*CODE_TO_PAIR[code], band_areas) for code in PAIR_CODES}
        if len(set(winners.values())) < 2:
            continue

        lower_win = any(winners[code] == CODE_TO_PAIR[code][1] for code in PAIR_CODES)
        upper_win = any(winners[code] == CODE_TO_PAIR[code][0] for code in PAIR_CODES)
        if not (lower_win and upper_win):
            continue

        return sizes, band_areas
    return None


def _frontier_score(code: int, r: int, c: int, center: tuple[float, float]) -> float:
    dr = r - center[0]
    dc = c - center[1]
    vr, vc = GROW_DIR[code]
    if code == 7:
        return -(abs(dr) + abs(dc)) + random.random()

    proj = dr * vr + dc * vc
    penalty = 0.45 * (abs(dr) + abs(dc))
    if vr != 0 and dr * vr < 0:
        penalty += 2.0
    if vc != 0 and dc * vc < 0:
        penalty += 2.0
    return 3.0 * proj - penalty + random.random()


def _grow_labels(h: int, w: int, sizes: dict[int, int]) -> np.ndarray | None:
    labels = np.zeros((h, w), dtype=int)
    cr = h // 2 - 1
    cc = w // 2 - 1
    core = [(cr, cc), (cr, cc + 1), (cr + 1, cc), (cr + 1, cc + 1)]
    anchors = {
        3: (cr - 1, cc),
        5: (cr, cc - 1),
        6: (cr, cc + 2),
        1: (cr + 1, cc - 1),
        2: (cr + 2, cc),
        4: (cr + 2, cc + 1),
    }
    for r, c in core + list(anchors.values()):
        if not (0 <= r < h and 0 <= c < w):
            return None
    if len(set(core + list(anchors.values()))) != 10:
        return None

    for r, c in core:
        labels[r, c] = 7
    for code, (r, c) in anchors.items():
        labels[r, c] = code

    counts = {7: 4, 3: 1, 5: 1, 6: 1, 1: 1, 2: 1, 4: 1}
    center = (cr + 0.5, cc + 0.5)
    for _ in range(h * w * 4):
        if all(counts[code] == sizes[code] for code in ALL_REGION_CODES):
            break
        order = list(ALL_REGION_CODES)
        random.shuffle(order)
        order.sort(key=lambda code: (sizes[code] - counts[code], random.random()), reverse=True)
        progressed = False
        for code in order:
            if counts[code] >= sizes[code]:
                continue
            frontier: set[tuple[int, int]] = set()
            pts = np.argwhere(labels == code)
            for r, c in pts:
                rr = int(r)
                cc2 = int(c)
                for nr, nc in _neighbors4(rr, cc2):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == 0:
                        frontier.add((nr, nc))
            if not frontier:
                return None
            scored = [(_frontier_score(code, r, c, center), r, c) for r, c in frontier]
            best_score = max(score for score, _r, _c in scored)
            candidates = [
                (r, c)
                for score, r, c in scored
                if score >= best_score - 0.35
            ]
            nr, nc = random.choice(candidates)
            labels[nr, nc] = code
            counts[code] += 1
            progressed = True
        if not progressed:
            return None
    else:
        return None

    if not all(counts[code] == sizes[code] for code in ALL_REGION_CODES):
        return None
    if random.random() < 0.5:
        labels = np.flipud(labels).copy()
    if random.random() < 0.5:
        labels = np.fliplr(labels).copy()
    return labels


def _masks_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    return [((labels & bit) != 0) for bit in BIT_CODES]


def _assemble_input(masks: list[np.ndarray]) -> np.ndarray:
    bands = [np.where(mask, color, BG) for mask, color in zip(masks, BAND_COLORS)]
    sep = np.full((1, masks[0].shape[1]), SEP, dtype=int)
    return np.vstack([bands[0], sep, bands[1], sep, bands[2]])


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bands = _extract_bands(grid)
        masks = _masks_from_bands(bands)
    except AssertionError:
        return False

    area = int(masks[0].size)
    low = math.ceil(0.30 * area)
    high = math.floor(0.60 * area)
    areas = [int(mask.sum()) for mask in masks]
    if any(not low <= count <= high for count in areas):
        return False
    if any(not _is_connected(mask) for mask in masks):
        return False
    if any(not _is_irregular(mask) for mask in masks):
        return False
    if any(np.array_equal(masks[i], masks[j]) for i, j in ((0, 1), (0, 2), (1, 2))):
        return False

    labels = _labels_from_masks(masks)
    present = set(map(int, np.unique(labels)))
    if present != {0, 1, 2, 3, 4, 5, 6, 7}:
        return False

    winners = {code: _winner(*CODE_TO_PAIR[code], areas) for code in PAIR_CODES}
    if len(set(winners.values())) < 2:
        return False
    if not any(winners[code] == CODE_TO_PAIR[code][0] for code in PAIR_CODES):
        return False
    if not any(winners[code] == CODE_TO_PAIR[code][1] for code in PAIR_CODES):
        return False

    expected = _expected_output_from_masks(masks)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if set(map(int, np.unique(actual))) != {0, 1, 2, 3, 4}:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(9, 13)
    tie_pair = random.choice([None, None, (0, 1), (0, 2), (1, 2)])
    sampled = _sample_sizes(h * w, tie_pair)
    if sampled is None:
        return None
    sizes, _areas = sampled
    labels = _grow_labels(h, w, sizes)
    if labels is None:
        return None
    masks = _masks_from_labels(labels)
    grid = _assemble_input(masks)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 3")
