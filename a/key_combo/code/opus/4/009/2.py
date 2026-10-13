from __future__ import annotations

import itertools
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
colors_present = _ARC_UTILS.colors_present
paste = _ARC_UTILS.paste


BAND_COUNT = 4
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NATURAL_ORDER = (0, 1, 2, 3)
FIXED_3D31_ORDER = (0, 1, 3, 2)
GEN_RETRIES = 400
ORDER_POOL = [
    order
    for order in itertools.permutations(range(BAND_COUNT))
    if order[0] != 0 and order[1] != 0
]
_GEN_CALL_COUNT = 0


def _split_bands(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    assert g.ndim == 2, "expected a 2D grid"
    assert height % BAND_COUNT == 0, "expected four equal-height horizontal bands"
    band_height = height // BAND_COUNT
    assert band_height > 0 and width > 0
    return [g[i * band_height : (i + 1) * band_height, :] for i in range(BAND_COUNT)]


def _overlay_by_order(bands: list[np.ndarray], order: tuple[int, ...]) -> np.ndarray:
    out = np.full_like(bands[0], Color.BLACK)
    for band_index in reversed(order):
        paste(out, bands[band_index], 0, 0, transparent=Color.BLACK)
    return out


def _overlay_with_producer(
    bands: list[np.ndarray], order: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full_like(bands[0], Color.BLACK)
    producer = np.full(bands[0].shape, -1, dtype=int)
    for band_index in reversed(order):
        mask = bands[band_index] != Color.BLACK
        out[mask] = bands[band_index][mask]
        producer[mask] = int(band_index)
    return out, producer


def _priority_from_counts(bands: list[np.ndarray]) -> tuple[int, ...]:
    counts = [int(np.count_nonzero(band)) for band in bands]
    assert len(set(counts)) == BAND_COUNT, "expected distinct foreground counts per band"
    return tuple(sorted(range(BAND_COUNT), key=lambda idx: (-counts[idx], idx)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four stacked monochrome mask bands into one band using pixel-density priority.

    Input structure:
    - The grid is split into four equal-height horizontal bands on a black background.
    - Each band is a same-size mask that uses exactly one non-black foreground color.
    - The four band colors are all distinct, and the four foreground pixel counts are all distinct.

    Rule:
    - Count the non-black pixels in each band.
    - Sort the bands from largest count to smallest count; this sorted order is the overlay priority.
    - For each local coordinate, output the first non-black pixel found when checking the bands in that priority order; if all four are black there, output black.

    Key insight:
    - The bands are transparent layers in one shared coordinate system, but unlike fixed-priority collapse tasks, the winner at each overlap is determined by global per-band density.
    """
    bands = _split_bands(grid)
    bg = get_bg(np.asarray(grid, dtype=int))
    assert bg == Color.BLACK, "expected black background"

    seen_colors: list[int] = []
    for band in bands:
        colors = colors_present(band, exclude={bg})
        assert len(colors) == 1, "expected one foreground color per band"
        color = int(colors[0])
        assert np.all((band == bg) | (band == color)), "expected monochrome mask bands"
        seen_colors.append(color)
    assert len(set(seen_colors)) == BAND_COUNT, "expected distinct band colors"

    order = _priority_from_counts(bands)
    return _overlay_by_order(bands, order)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _weighted_choice(coords: list[tuple[int, int]], h: int, w: int) -> tuple[int, int]:
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    weights = []
    for r, c in coords:
        dist = abs(r - center_r) + abs(c - center_c)
        weights.append(1.0 / (1.0 + dist))
    total = float(sum(weights))
    pick = float(np.random.random() * total)
    acc = 0.0
    for coord, weight in zip(coords, weights):
        acc += weight
        if pick <= acc:
            return coord
    return coords[-1]


def _grow_mask(
    h: int,
    w: int,
    target: int,
    required: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> np.ndarray | None:
    if target < len(required):
        return None

    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        for r, c in required:
            mask[r, c] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in forbidden:
                        frontier.add((nr, nc))

            if frontier and np.random.random() < 0.9:
                coord = _weighted_choice(sorted(frontier), h, w)
            else:
                empties = [
                    (r, c)
                    for r in range(h)
                    for c in range(w)
                    if not mask[r, c] and (r, c) not in forbidden
                ]
                if not empties:
                    return None
                coord = _weighted_choice(empties, h, w)

            mask[coord] = True

        if np.random.random() < 0.5:
            mask = np.flipud(mask)
        if np.random.random() < 0.5:
            mask = np.fliplr(mask)

        if not all(mask[r, c] for r, c in required):
            continue
        if any(mask[r, c] for r, c in forbidden):
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        return mask
    return None


def _distinct_counts(area: int) -> list[int] | None:
    low = max(4, area // 6)
    high = min(area - 1, max(low + 6, int(round(area * 0.58))))
    values = list(range(low, high + 1))
    if len(values) < BAND_COUNT:
        return None

    for _ in range(200):
        counts = sorted(np.random.choice(values, size=BAND_COUNT, replace=False).tolist(), reverse=True)
        if all(counts[i] - counts[i + 1] >= 2 for i in range(BAND_COUNT - 1)):
            return [int(v) for v in counts]
    return None


def _choose_order() -> tuple[int, ...]:
    global _GEN_CALL_COUNT
    order = ORDER_POOL[_GEN_CALL_COUNT % len(ORDER_POOL)]
    _GEN_CALL_COUNT += 1
    return tuple(int(v) for v in order)


def _build_instance() -> np.ndarray | None:
    band_height = int(np.random.randint(4, 7))
    width = int(np.random.randint(5, 8))
    area = band_height * width
    counts_desc = _distinct_counts(area)
    if counts_desc is None:
        return None

    order = _choose_order()
    colors = [int(v) for v in np.random.choice(PALETTE, size=BAND_COUNT, replace=False)]

    witness_cells = [(0, 0), (0, width - 1), (band_height - 1, 0), (band_height - 1, width - 1)]
    np.random.shuffle(witness_cells)
    witness_by_band = {band: witness_cells[band] for band in range(BAND_COUNT)}

    shared_top = (band_height // 2, width // 2)
    shared_mid = (band_height // 2, 1)
    shared_tail = (band_height // 2, width - 2)

    reserved = set(witness_cells) | {shared_top, shared_mid}
    use_tail = order[3] == 0
    if use_tail:
        reserved.add(shared_tail)

    masks: list[np.ndarray | None] = [None] * BAND_COUNT
    for rank, band in enumerate(order):
        required = {witness_by_band[band], shared_top}
        if band in order[1:]:
            required.add(shared_mid)
        if use_tail and band in order[2:]:
            required.add(shared_tail)

        forbidden = reserved - required
        mask = _grow_mask(band_height, width, counts_desc[rank], required, forbidden)
        if mask is None:
            return None
        masks[band] = mask

    typed_masks = [mask for mask in masks if mask is not None]
    if len(typed_masks) != BAND_COUNT:
        return None

    bands = []
    for band_index, mask in enumerate(typed_masks):
        band = np.zeros((band_height, width), dtype=int)
        band[mask] = colors[band_index]
        bands.append(band)

    out, producer = _overlay_with_producer(bands, order)
    contributors = {int(v) for v in np.unique(producer) if int(v) >= 0}
    if contributors != set(range(BAND_COUNT)):
        return None

    if np.array_equal(out, _overlay_by_order(bands, NATURAL_ORDER)):
        return None
    if np.array_equal(out, _overlay_by_order(bands, FIXED_3D31_ORDER)):
        return None

    top_correct_rest_natural = (order[0],) + tuple(idx for idx in NATURAL_ORDER if idx != order[0])
    if np.array_equal(out, _overlay_by_order(bands, top_correct_rest_natural)):
        return None

    if use_tail:
        top_two_correct_rest_natural = (
            order[0],
            order[1],
        ) + tuple(idx for idx in NATURAL_ORDER if idx not in {order[0], order[1]})
        if np.array_equal(out, _overlay_by_order(bands, top_two_correct_rest_natural)):
            return None

    for band in bands:
        if np.array_equal(out, band):
            return None

    return np.vstack(bands)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid density-priority overlay input")
