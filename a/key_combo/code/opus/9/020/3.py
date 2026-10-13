from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
RETRY_LIMIT = 400
BAN_PRIORITY_ORDERS = {
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (3, 2, 1, 0),
    (3, 2, 0, 1),
}
CONTEST_TEMPLATES = (
    ((0, 3), (1, 2), (0, 1, 3)),
    ((0, 2), (1, 3), (0, 1, 2)),
    ((0, 3), (0, 2), (1, 3)),
    ((0, 1, 2, 3), (0, 3), (1, 2)),
)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _band_slices(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 4 == 0, "expected four equal-height horizontal bands"
    band_height = height // 4
    assert band_height > 0 and width > 0
    return [g[i * band_height : (i + 1) * band_height, :] for i in range(4)]


def _band_color(band: np.ndarray) -> int:
    colors = sorted(int(v) for v in np.unique(band) if int(v) != BACKGROUND)
    assert len(colors) == 1, "each band must contain exactly one foreground color"
    color = int(colors[0])
    assert np.all((band == BACKGROUND) | (band == color)), "each band must be a monochrome mask"
    return color


def _priority_from_counts(counts: list[int]) -> tuple[int, int, int, int]:
    assert len(counts) == 4
    assert len(set(map(int, counts))) == 4, "foreground counts must be pairwise distinct"
    return tuple(sorted(range(4), key=lambda idx: int(counts[idx])))


def _composite(bands: list[np.ndarray], priority_order: tuple[int, int, int, int]) -> np.ndarray:
    band_height, width = map(int, bands[0].shape)
    out = np.full((band_height, width), BACKGROUND, dtype=int)
    for band_idx in reversed(priority_order):
        band = bands[int(band_idx)]
        mask = band != BACKGROUND
        out[mask] = band[mask]
    return out


def _winner_map(bands: list[np.ndarray], priority_order: tuple[int, int, int, int]) -> np.ndarray:
    winners = np.full(bands[0].shape, -1, dtype=int)
    for band_idx in reversed(priority_order):
        winners[bands[band_idx] != BACKGROUND] = int(band_idx)
    return winners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four stacked color-mask bands by compositing them in foreground-density order.

    Input structure:
    - The grid has black background and is split into four equal-height horizontal bands.
    - Each band is a same-size monochrome mask using one non-black color on black.
    - The four band colors are distinct, and the masks share the same local coordinates.

    Rule:
    - Count the non-black pixels in each band.
    - Sort the bands from sparsest to densest; fewer pixels means higher overlay priority.
    - For each local cell, output the first non-black pixel encountered in that priority order.
      If all four corresponding cells are black, output black.

    Key insight:
    - The overlay priority is not tied to a fixed band position or color; it is determined anew
      in each example by each band's foreground-pixel count.
    """
    bands = _band_slices(grid)
    colors = [_band_color(band) for band in bands]
    assert len(set(colors)) == 4, "expected four distinct band colors"

    counts = [int(np.count_nonzero(band)) for band in bands]
    priority_order = _priority_from_counts(counts)
    return _composite(bands, priority_order)


def _all_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [(r, c) for r in range(h) for c in range(w)]


def _pick_from_region(
    rng: random.Random,
    rows: range,
    cols: range,
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in rows for c in cols if (r, c) not in forbidden]
    if not candidates:
        return None
    return rng.choice(candidates)


def _exclusive_positions(
    shape: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    top = range(0, max(1, h // 2))
    bottom = range(max(1, h // 2), h)
    left = range(0, max(1, w // 2))
    right = range(max(1, w // 2), w)
    regions = [(top, left), (top, right), (bottom, left), (bottom, right)]

    used: set[tuple[int, int]] = set()
    picks: list[tuple[int, int]] = []
    for rows, cols in regions:
        cell = _pick_from_region(rng, rows, cols, used)
        if cell is None:
            return None
        used.add(cell)
        picks.append(cell)
    rng.shuffle(picks)
    return picks


def _contest_positions(
    shape: tuple[int, int],
    rng: random.Random,
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    if h >= 5:
        row_range = range(1, h - 1)
    else:
        row_range = range(h)
    if w >= 5:
        col_range = range(1, w - 1)
    else:
        col_range = range(w)
    candidates = [(r, c) for r in row_range for c in col_range if (r, c) not in forbidden]
    if len(candidates) < count:
        candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    if len(candidates) < count:
        return None
    return rng.sample(candidates, count)


def _choose_priority_order(rng: random.Random) -> tuple[int, int, int, int]:
    orders = [perm for perm in itertools.permutations(range(4)) if perm not in BAN_PRIORITY_ORDERS]
    return tuple(rng.choice(orders))


def _sample_counts(
    area: int,
    min_required: list[int],
    rng: random.Random,
) -> list[int] | None:
    count_min = max(5, max(min_required))
    count_max = min(max(count_min + 3, area // 2 + 3), area - 2, 15)
    if count_max - count_min + 1 < 4:
        return None
    counts = sorted(rng.sample(list(range(count_min, count_max + 1)), 4))
    return counts


def _score_candidate(
    pos: tuple[int, int],
    anchor: tuple[int, int],
    occupancy: dict[tuple[int, int], int],
    preferred_overlap: set[tuple[int, int]],
    rng: random.Random,
) -> float:
    r, c = pos
    ar, ac = anchor
    distance = abs(r - ar) + abs(c - ac)
    overlap_penalty = 1.75 * occupancy.get(pos, 0)
    if pos in preferred_overlap:
        overlap_penalty -= 1.25
    return distance + overlap_penalty + rng.random() * 0.25


def _grow_mask(
    shape: tuple[int, int],
    target_count: int,
    required: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    occupancy: dict[tuple[int, int], int],
    preferred_overlap: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    h, w = map(int, shape)
    if len(required) > target_count:
        return None

    cells = set(required)
    valid_cells = [cell for cell in _all_cells(shape) if cell not in forbidden]
    if len(valid_cells) < target_count:
        return None

    if cells:
        anchor = rng.choice(list(cells))
    else:
        anchor = rng.choice(valid_cells)

    while len(cells) < target_count:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                nxt = (rr, cc)
                if 0 <= rr < h and 0 <= cc < w and nxt not in cells and nxt not in forbidden:
                    frontier.add(nxt)
        if frontier and rng.random() < 0.88:
            candidates = list(frontier)
        else:
            candidates = [cell for cell in valid_cells if cell not in cells]
        if not candidates:
            return None

        sample_size = min(10, len(candidates))
        sampled = rng.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
        best = min(sampled, key=lambda pos: _score_candidate(pos, anchor, occupancy, preferred_overlap, rng))
        cells.add(best)
        if rng.random() < 0.22:
            anchor = best

    mask = np.zeros(shape, dtype=int)
    for r, c in cells:
        mask[r, c] = 1
    return mask


def _build_masks(
    shape: tuple[int, int],
    colors: list[int],
    counts: list[int],
    priority_order: tuple[int, int, int, int],
    template: tuple[tuple[int, ...], ...],
    rng: random.Random,
) -> list[np.ndarray] | None:
    exclusive_positions = _exclusive_positions(shape, rng)
    if exclusive_positions is None:
        return None

    contest_positions = _contest_positions(shape, rng, set(exclusive_positions), len(template))
    if contest_positions is None:
        return None

    required_by_band = [set([exclusive_positions[idx]]) for idx in range(4)]
    preferred_overlap_by_band = [set() for _ in range(4)]
    for pos, rank_group in zip(contest_positions, template):
        for rank in rank_group:
            band_idx = priority_order[rank]
            required_by_band[band_idx].add(pos)
            preferred_overlap_by_band[band_idx].add(pos)

    sorted_counts = sorted(int(v) for v in counts)
    target_counts = [0] * 4
    for rank, band_idx in enumerate(priority_order):
        target_counts[band_idx] = sorted_counts[rank]

    occupancy: dict[tuple[int, int], int] = {}
    masks: list[np.ndarray] = []
    all_exclusives = set(exclusive_positions)
    for band_idx in range(4):
        forbidden = all_exclusives - {exclusive_positions[band_idx]}
        mask = _grow_mask(
            shape,
            target_counts[band_idx],
            required_by_band[band_idx],
            forbidden,
            occupancy,
            preferred_overlap_by_band[band_idx],
            rng,
        )
        if mask is None:
            return None
        mask *= int(colors[band_idx])
        masks.append(mask)
        for r, c in np.argwhere(mask != BACKGROUND):
            pos = (int(r), int(c))
            occupancy[pos] = occupancy.get(pos, 0) + 1

    return masks


def _validate_generated(grid: np.ndarray) -> bool:
    bands = _band_slices(grid)
    colors = [_band_color(band) for band in bands]
    if len(set(colors)) != 4:
        return False

    counts = [int(np.count_nonzero(band)) for band in bands]
    if len(set(counts)) != 4:
        return False
    priority_order = _priority_from_counts(counts)
    if priority_order in BAN_PRIORITY_ORDERS:
        return False

    occupancy = np.zeros_like(bands[0], dtype=int)
    for band in bands:
        occupancy += (band != BACKGROUND).astype(int)
    if int(np.count_nonzero(occupancy > 1)) < 2:
        return False
    if int(np.count_nonzero(occupancy == 0)) < 2:
        return False

    out = _composite(bands, priority_order)
    winners = _winner_map(bands, priority_order)
    visible_bands = {int(idx) for idx in winners[winners >= 0]}
    if visible_bands != {0, 1, 2, 3}:
        return False

    shortcut_orders = [
        (0, 1, 2, 3),
        (0, 1, 3, 2),
        (3, 2, 1, 0),
        (1, 0, 2, 3),
    ]
    for order in shortcut_orders:
        if tuple(order) == priority_order:
            return False
        alt = _composite(bands, tuple(order))
        if np.array_equal(out, alt):
            return False

    differing_orders = 0
    for order in itertools.permutations(range(4)):
        if tuple(order) == priority_order:
            continue
        if not np.array_equal(out, _composite(bands, tuple(order))):
            differing_orders += 1
    if differing_orders < 12:
        return False

    if not np.array_equal(out, main(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    band_height = rng.randint(4, 7)
    width = rng.randint(5, 8)
    area = band_height * width

    priority_order = _choose_priority_order(rng)
    colors = rng.sample(list(COLOR_POOL), 4)

    template = rng.choice(CONTEST_TEMPLATES)
    min_required = [1] * 4
    for rank_group in template:
        for rank in rank_group:
            min_required[priority_order[rank]] += 1

    counts = _sample_counts(area, min_required, rng)
    if counts is None:
        return None

    masks = _build_masks((band_height, width), colors, counts, priority_order, template, rng)
    if masks is None:
        return None

    grid = np.vstack(masks)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid density-sorted overlay input")
