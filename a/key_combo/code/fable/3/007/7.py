from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg


MARKER_SIZE = 2
BACKGROUND_COLOR = 0
ARC_COLORS = np.arange(1, 10, dtype=int)
MAX_GENERATION_ATTEMPTS = 500


def _is_2x2_block(component: dict) -> bool:
    box = component["bbox"]
    return (
        int(component["area"]) == MARKER_SIZE * MARKER_SIZE
        and bbox_h(box) == MARKER_SIZE
        and bbox_w(box) == MARKER_SIZE
    )


def _paired_marker_blocks(
    grid: np.ndarray, bg: int
) -> dict[int, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    pairs = {}
    for color in colors_present(grid, exclude={bg}):
        blocks = [
            comp["bbox"]
            for comp in find_components(grid, bg=bg, colors=[color], connectivity=4)
            if _is_2x2_block(comp)
        ]
        if len(blocks) != 2:
            continue

        left, right = sorted(blocks, key=lambda b: b[1])
        if left[0] != right[0] or left[2] != right[2]:
            continue
        if right[1] - left[3] - 1 <= 0:
            continue
        pairs[int(color)] = (left, right)

    assert pairs, "expected at least one colored pair of row-aligned 2x2 marker blocks"
    return pairs


def _single_pixel_component_at(
    grid: np.ndarray, bg: int, color: int, r: int, c: int
) -> bool:
    for comp in find_components(grid, bg=bg, colors=[color], connectivity=4):
        if int(comp["area"]) != 1:
            continue
        rr, cc = map(int, comp["coords"][0])
        if rr == int(r) and cc == int(c):
            return True
    return False


def _edge_key_order(grid: np.ndarray, bg: int, pair_colors: set[int]) -> list[int]:
    h, w = grid.shape
    candidates: list[list[int]] = []

    for c in (0, w - 1):
        rows = [int(r) for r in np.flatnonzero(grid[:, c] != bg)]
        if not rows:
            continue

        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        if not set(colors).issubset(pair_colors):
            continue

        inward_c = 1 if c == 0 else w - 2
        ok = True
        for r, color in zip(rows, colors):
            if int(grid[r, inward_c]) != bg:
                ok = False
                break
            if not _single_pixel_component_at(grid, bg, color, r, c):
                ok = False
                break
        if ok:
            candidates.append(colors)

    assert len(candidates) == 1, "expected one edge column of isolated color keys"
    return candidates[0]


def _extract_between(
    grid: np.ndarray,
    pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> np.ndarray:
    left, right = pair
    assert left[0] == right[0] and left[2] == right[2]
    stripe = grid[left[0] : left[2] + 1, left[3] + 1 : right[1]].copy()
    assert stripe.shape[0] == MARKER_SIZE and stripe.shape[1] > 0
    return stripe


def main(grid: np.ndarray) -> np.ndarray:
    """Stack the regions between colored 2x2 marker pairs in edge-key order.

    Input structure:
    - The background is the dominant color, with scattered noise away from the key and marker spans.
    - Several colors each form exactly one left/right pair of 2x2 marker blocks sharing the same two-row span.
    - One vertical edge column contains isolated one-pixel color keys, read from top to bottom; key colors are a subset of the marker-pair colors.
    - Some marker-pair colors may be missing from the edge key column.

    Rule:
    - For every same-color marker pair, copy the rectangle strictly between the two 2x2 blocks.
    - Read the edge key column from top to bottom and stack only the copied rectangles whose colors appear there, in that key order.
    - Marker blocks, unkeyed copied rectangles, the key pixels, and all other noise are discarded.

    Key insight:
    - The vertical position of a marker pair is not the ordering rule; the edge color key is both the order list and the filter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    pairs = _paired_marker_blocks(g, bg)
    key_order = _edge_key_order(g, bg, set(pairs))

    stripes = []
    widths = set()
    for color in key_order:
        assert color in pairs, "key color must name a marker pair"
        stripe = _extract_between(g, pairs[color])
        stripes.append(stripe)
        widths.add(int(stripe.shape[1]))

    assert stripes and len(widths) == 1, "all keyed extracts must have a common width"
    return np.vstack(stripes)


# --------------- generator ---------------


def _row_starts(rng: np.random.Generator, pair_count: int, height: int) -> list[int] | None:
    gaps = rng.integers(1, 4, size=pair_count - 1)
    used = MARKER_SIZE * pair_count + int(gaps.sum())
    extra = height - used
    if extra < 0:
        return None

    row = int(rng.integers(0, extra + 1))
    starts = [row]
    for gap in gaps:
        row += MARKER_SIZE + int(gap)
        starts.append(row)
    return starts


def _key_rows(rng: np.random.Generator, height: int, count: int) -> list[int] | None:
    span = 2 * count - 1
    if span > height:
        return None
    top = int(rng.integers(0, height - span + 1))
    return [top + 2 * i for i in range(count)]


def _sample_stripe(
    rng: np.random.Generator, gap_width: int, payload_colors: np.ndarray
) -> np.ndarray:
    stripe = np.zeros((MARKER_SIZE, gap_width), dtype=int)
    area = int(stripe.size)
    lo = max(2, (area + 3) // 4)
    hi = max(lo, (3 * area) // 4)
    count = int(rng.integers(lo, hi + 1))
    cells = rng.choice(area, size=count, replace=False)
    stripe.flat[cells] = rng.choice(payload_colors, size=count, replace=True)
    return stripe


def _sample_unique_stripes(
    rng: np.random.Generator, pair_count: int, gap_width: int, payload_colors: np.ndarray
) -> list[np.ndarray] | None:
    stripes: list[np.ndarray] = []
    seen = set()
    for _ in range(pair_count):
        for _attempt in range(80):
            stripe = _sample_stripe(rng, gap_width, payload_colors)
            key = tuple(map(int, stripe.ravel()))
            if key not in seen:
                seen.add(key)
                stripes.append(stripe)
                break
        else:
            return None
    return stripes


def _sample_key_order(
    rng: np.random.Generator, pair_count: int
) -> tuple[list[int], list[int]] | None:
    keyed_count = int(rng.integers(2, pair_count))
    keyed = sorted(map(int, rng.choice(pair_count, size=keyed_count, replace=False)))

    for _ in range(40):
        order = list(map(int, rng.permutation(keyed)))
        if order != keyed:
            return keyed, order
    return keyed, list(reversed(keyed)) if len(keyed) > 1 else None


def _render_scene(
    height: int,
    width: int,
    key_edge: int,
    marker_colors: np.ndarray,
    row_starts: list[int],
    left_cols: list[int],
    gap_width: int,
    stripes: list[np.ndarray],
    key_order: list[int],
    key_rows: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    protected = np.zeros((height, width), dtype=bool)

    for i, (row, col) in enumerate(zip(row_starts, left_cols)):
        color = int(marker_colors[i])
        right_col = col + MARKER_SIZE + gap_width
        grid[row : row + MARKER_SIZE, col : col + MARKER_SIZE] = color
        grid[row : row + MARKER_SIZE, col + MARKER_SIZE : right_col] = stripes[i]
        grid[row : row + MARKER_SIZE, right_col : right_col + MARKER_SIZE] = color
        protected[row : row + MARKER_SIZE, col : right_col + MARKER_SIZE] = True

    key_col = 0 if key_edge == 0 else width - 1
    inward_col = 1 if key_edge == 0 else width - 2
    protected[:, key_col] = True
    protected[:, inward_col] = True
    for r, pair_index in zip(key_rows, key_order):
        grid[r, key_col] = int(marker_colors[pair_index])

    return grid, protected


def _add_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    protected: np.ndarray,
    noise_colors: np.ndarray,
) -> None:
    free = np.argwhere(~protected)
    if len(free) == 0:
        return

    max_noise = min(len(free), max(4, grid.size // 7))
    min_noise = min(max_noise, max(3, grid.size // 16))
    count = int(rng.integers(min_noise, max_noise + 1))
    chosen = free[rng.choice(len(free), size=count, replace=False)]
    colors = rng.choice(noise_colors, size=count, replace=True)
    grid[chosen[:, 0], chosen[:, 1]] = colors


def _oracle(stripes: list[np.ndarray], key_order: list[int]) -> np.ndarray:
    return np.vstack([stripes[i] for i in key_order])


def _passes_semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    stripes: list[np.ndarray],
    keyed_spatial: list[int],
    key_order: list[int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if expected.ndim != 2 or max(expected.shape) > 30:
        return False

    unkeyed = [i for i in range(len(stripes)) if i not in set(key_order)]
    if not unkeyed or any(np.count_nonzero(stripes[i]) == 0 for i in unkeyed):
        return False

    spatial_stack = np.vstack([stripes[i] for i in keyed_spatial])
    if np.array_equal(spatial_stack, expected):
        return False

    all_spatial = np.vstack(stripes)
    if all_spatial.shape == expected.shape and np.array_equal(all_spatial, expected):
        return False

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[BACKGROUND_COLOR] <= max(counts[1:]):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    pair_count = int(rng.integers(3, 6))
    gap_width = int(rng.integers(3, 8))
    keyed = _sample_key_order(rng, pair_count)
    if keyed is None:
        return None
    keyed_spatial, key_order = keyed

    min_height = max(12, MARKER_SIZE * pair_count + pair_count - 1, 2 * len(key_order) - 1)
    height = int(rng.integers(min_height, 27))
    row_starts = _row_starts(rng, pair_count, height)
    key_positions = _key_rows(rng, height, len(key_order))
    if row_starts is None or key_positions is None:
        return None

    min_width = max(14, gap_width + 12)
    width = int(rng.integers(min_width, 31))
    body_margin = 3
    span_width = 2 * MARKER_SIZE + gap_width
    left_min = body_margin
    left_high = width - body_margin - span_width + 1
    if left_high <= left_min:
        return None
    left_cols = [int(rng.integers(left_min, left_high)) for _ in range(pair_count)]

    marker_colors = rng.choice(ARC_COLORS, size=pair_count, replace=False)
    payload_colors = np.array([c for c in ARC_COLORS if c not in set(marker_colors)], dtype=int)
    if len(payload_colors) < 2:
        return None

    stripes = _sample_unique_stripes(rng, pair_count, gap_width, payload_colors)
    if stripes is None:
        return None

    expected = _oracle(stripes, key_order)
    key_edge = int(rng.integers(2))
    grid, protected = _render_scene(
        height,
        width,
        key_edge,
        marker_colors,
        row_starts,
        left_cols,
        gap_width,
        stripes,
        key_order,
        key_positions,
    )
    _add_noise(rng, grid, protected, payload_colors)

    if not _passes_semantic_checks(grid, expected, stripes, keyed_spatial, key_order):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid order-key stacking input")
