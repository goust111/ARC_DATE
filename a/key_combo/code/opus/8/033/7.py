from __future__ import annotations

from collections import deque

import numpy as np


MARKER_COLOR = 2
BACKGROUND_COLOR = 0
LAYER_HEIGHT = 3
MARKER_SIZE = 2
NON_MARKER_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
RING_CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
RING_SIDES = ((0, 1), (1, 2), (2, 1), (1, 0))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _marker_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    comps = _components(grid == MARKER_COLOR)
    assert len(comps) >= 6 and len(comps) % 2 == 0

    blocks = []
    for coords in comps:
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert coords.shape[0] == 4
        assert r1 == r0 + 1 and c1 == c0 + 1
        blocks.append(box)

    by_top: dict[int, list[tuple[int, int, int, int]]] = {}
    for box in blocks:
        by_top.setdefault(box[0], []).append(box)

    pairs = []
    gap_widths = set()
    for top in sorted(by_top):
        row_blocks = sorted(by_top[top], key=lambda b: b[1])
        assert len(row_blocks) == 2
        left, right = row_blocks
        assert left[0] == right[0] and left[2] == right[2]
        gap_width = right[1] - left[3] - 1
        assert gap_width > 0
        assert top + LAYER_HEIGHT <= grid.shape[0]
        pairs.append((left, right))
        gap_widths.add(gap_width)

    assert len(pairs) in (3, 4)
    assert len(gap_widths) == 1
    return pairs


def _is_ring_patch(patch: np.ndarray) -> bool:
    if patch.shape != (3, 3):
        return False
    if int(patch[1, 1]) != BACKGROUND_COLOR:
        return False
    ring_cells = [int(patch[r, c]) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    return all(color != BACKGROUND_COLOR for color in ring_cells)


def _rotate_ring_patch(patch: np.ndarray, steps: int) -> np.ndarray:
    assert _is_ring_patch(patch)
    k = int(steps) % 4
    out = patch.copy()
    corner_vals = np.array([int(patch[r, c]) for r, c in RING_CORNERS], dtype=int)
    side_vals = np.array([int(patch[r, c]) for r, c in RING_SIDES], dtype=int)
    corner_vals = np.roll(corner_vals, k)
    side_vals = np.roll(side_vals, -k)
    for (r, c), value in zip(RING_CORNERS, corner_vals):
        out[r, c] = int(value)
    for (r, c), value in zip(RING_SIDES, side_vals):
        out[r, c] = int(value)
    out[1, 1] = BACKGROUND_COLOR
    return out


def _ring_columns(band: np.ndarray) -> list[int]:
    return [c for c in range(band.shape[1] - 2) if _is_ring_patch(band[:, c : c + 3])]


def _transform_stacked_layers(stripes: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(stripes)
    out = stacked.copy()
    for layer_index, stripe in enumerate(stripes, start=1):
        transformed = stripe.copy()
        for c in _ring_columns(stripe):
            transformed[:, c : c + 3] = _rotate_ring_patch(stripe[:, c : c + 3], layer_index)
        r0 = (layer_index - 1) * LAYER_HEIGHT
        out[r0 : r0 + LAYER_HEIGHT] = transformed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract 3-row bands between paired red markers, stack them, and rotate each 3x3 ring by its source-layer index.

    Input structure
    - The background is black.
    - Red 2x2 marker blocks occur as left/right pairs; in each layer the two markers share the same top row and occupy the first two rows of a 3-row extraction band.
    - The rectangle strictly between each paired marker blocks contains one or two 3x3 colored rings, plus optional non-ring pixels; stray non-red noise may appear elsewhere.

    Rule
    - Pair the red 2x2 blocks by common top row, sort the pairs from top to bottom, and treat them as layers 1, 2, 3, ...
    - For each layer, crop the full 3-row rectangle strictly between the two markers, then stack all cropped rectangles vertically in layer order.
    - In the stacked result, every 3x3 ring inherited from layer k is recolored by k opposite cycles: corners move clockwise and side-centers move counterclockwise once per step.
    - Keep every non-ring pixel of the stacked rectangles unchanged and discard the markers and all outside noise.

    Key insight
    - After stacking, the rings no longer visibly remember which marker pair they came from, so the correct cycle count can only be recovered by tracking the extraction layer order.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _marker_pairs(g)

    stripes = []
    widths = set()
    for left, right in pairs:
        top = left[0]
        stripe = g[top : top + LAYER_HEIGHT, left[3] + 1 : right[1]].copy()
        assert stripe.shape[0] == LAYER_HEIGHT and stripe.shape[1] > 0
        stripes.append(stripe)
        widths.add(stripe.shape[1])

    assert len(widths) == 1
    return _transform_stacked_layers(stripes)


def _random_ring(rng: np.random.Generator, required_steps: tuple[int, ...] = ()) -> np.ndarray:
    while True:
        patch = np.zeros((3, 3), dtype=int)
        color_count = int(rng.integers(2, 5))
        palette = rng.choice(NON_MARKER_COLORS, size=color_count, replace=False)
        values = rng.choice(palette, size=8, replace=True)
        patch[1, 1] = BACKGROUND_COLOR
        for (r, c), value in zip(((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)), values):
            patch[r, c] = int(value)
        if len(set(map(int, values))) < 2:
            continue
        transformed = {_rotate_ring_patch(patch, step).tobytes() for step in required_steps}
        if len(transformed) == len(required_steps):
            return patch


def _position_options(width: int, ring_count: int) -> list[tuple[int, ...]]:
    starts = list(range(width - 2))
    if ring_count == 1:
        return [(c,) for c in starts]
    return [(a, b) for a in starts for b in starts if b >= a + 4]


def _band_with_layout(
    rng: np.random.Generator,
    width: int,
    ring_patches: list[np.ndarray],
    starts: tuple[int, ...],
) -> np.ndarray:
    band = np.zeros((LAYER_HEIGHT, width), dtype=int)
    occupied = np.zeros((LAYER_HEIGHT, width), dtype=bool)
    for patch, start in zip(ring_patches, starts):
        band[:, start : start + 3] = patch
        occupied[:, start : start + 3] = True

    free_cols = [
        c
        for c in range(width)
        if not occupied[:, c].any()
        and all(not occupied[:, nc].any() for nc in (c - 1, c + 1) if 0 <= nc < width)
    ]
    filler_count = min(len(free_cols), int(rng.integers(0, 3)))
    if filler_count:
        chosen_cols = rng.choice(free_cols, size=filler_count, replace=False)
        for col in np.atleast_1d(chosen_cols):
            row = int(rng.integers(0, LAYER_HEIGHT))
            band[row, int(col)] = int(rng.choice(NON_MARKER_COLORS))
    return band


def _build_stripes(rng: np.random.Generator, num_layers: int, width: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    ring_counts = np.ones(num_layers, dtype=int)
    if rng.random() < 0.75:
        two_ring_layers = int(rng.integers(1, min(num_layers, 2) + 1))
        chosen = rng.choice(num_layers, size=two_ring_layers, replace=False)
        ring_counts[chosen] = 2

    shared_layers = tuple(sorted(map(int, rng.choice(num_layers, size=2, replace=False) + 1)))
    shared_ring = _random_ring(rng, required_steps=shared_layers)

    stripes: list[np.ndarray] = []
    for layer_index in range(1, num_layers + 1):
        ring_count = int(ring_counts[layer_index - 1])
        options = _position_options(width, ring_count)
        starts = options[int(rng.integers(0, len(options)))]
        patches = []
        for ring_slot in range(ring_count):
            if ring_slot == 0 and layer_index in shared_layers:
                patches.append(shared_ring.copy())
            else:
                patches.append(_random_ring(rng))
        band = _band_with_layout(rng, width, patches, starts)
        if len(_ring_columns(band)) != ring_count:
            return _build_stripes(rng, num_layers, width)
        stripes.append(band)

    return stripes, shared_layers


def _render_input(
    rng: np.random.Generator,
    stripes: list[np.ndarray],
    width: int,
) -> np.ndarray:
    num_layers = len(stripes)
    top_margin = int(rng.integers(0, 3))
    inter_gaps = [int(rng.integers(1, 4)) for _ in range(num_layers - 1)]
    tops = [top_margin]
    for gap in inter_gaps:
        tops.append(tops[-1] + LAYER_HEIGHT + gap)
    height = tops[-1] + LAYER_HEIGHT + int(rng.integers(0, 3))

    grid_width_min = width + MARKER_SIZE * 2 + num_layers
    grid_width = int(rng.integers(grid_width_min, min(22, grid_width_min + 5)))
    left_options = np.arange(0, grid_width - width - MARKER_SIZE * 2 + 1)
    left_cols = rng.choice(left_options, size=num_layers, replace=False)

    grid = np.zeros((height, grid_width), dtype=int)
    protected = np.zeros_like(grid, dtype=bool)
    for stripe, top, left in zip(stripes, tops, left_cols):
        left = int(left)
        right = left + MARKER_SIZE + width
        grid[top : top + MARKER_SIZE, left : left + MARKER_SIZE] = MARKER_COLOR
        grid[top : top + MARKER_SIZE, right : right + MARKER_SIZE] = MARKER_COLOR
        grid[top : top + LAYER_HEIGHT, left + MARKER_SIZE : right] = stripe
        protected[top : top + LAYER_HEIGHT, left : right + MARKER_SIZE] = True

    noise_budget = int(rng.integers(0, max(1, (height * grid_width) // 12)))
    candidates = np.argwhere(~protected)
    if noise_budget and len(candidates):
        chosen = candidates[rng.choice(len(candidates), size=min(noise_budget, len(candidates)), replace=False)]
        for r, c in chosen:
            grid[int(r), int(c)] = int(rng.choice(NON_MARKER_COLORS))
    return grid


def _expected_from_stripes(stripes: list[np.ndarray]) -> np.ndarray:
    return _transform_stacked_layers([stripe.copy() for stripe in stripes])


def _validate_instance(grid: np.ndarray, stripes: list[np.ndarray], shared_layers: tuple[int, int]) -> bool:
    if max(grid.shape) > 30:
        return False

    expected = _expected_from_stripes(stripes)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if actual.shape != (len(stripes) * LAYER_HEIGHT, stripes[0].shape[1]):
        return False

    total_rings = 0
    for stripe in stripes:
        cols = _ring_columns(stripe)
        if len(cols) not in (1, 2):
            return False
        total_rings += len(cols)
    if total_rings < len(stripes):
        return False

    pair_a, pair_b = shared_layers
    ring_a = stripes[pair_a - 1][:, _ring_columns(stripes[pair_a - 1])[0] : _ring_columns(stripes[pair_a - 1])[0] + 3]
    ring_b = stripes[pair_b - 1][:, _ring_columns(stripes[pair_b - 1])[0] : _ring_columns(stripes[pair_b - 1])[0] + 3]
    if not np.array_equal(ring_a, ring_b):
        return False
    if np.array_equal(_rotate_ring_patch(ring_a, pair_a), _rotate_ring_patch(ring_b, pair_b)):
        return False

    pairs = _marker_pairs(grid)
    if len(pairs) != len(stripes):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        num_layers = int(rng.integers(3, 5))
        width = int(rng.integers(8, 11))
        stripes, shared_layers = _build_stripes(rng, num_layers, width)
        grid = _render_input(rng, stripes, width)
        if _validate_instance(grid, stripes, shared_layers):
            return grid
    raise RuntimeError("failed to generate a valid layer-index ring-cycling input")
