from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
RED_COLOR = 2
MARKER_CHOICES = [1, 3, 4, 5, 6, 7, 8, 9]

PAIR_COUNT_RANGE = (2, 5)
STRIPE_WIDTH_RANGE = (4, 10)
ROW_GAP_RANGE = (1, 3)
EXTRA_HEIGHT_MAX = 6
EXTRA_WIDTH_MAX = 8
NOISE_RATE_RANGE = (0.04, 0.10)
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_marker_pairs(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    candidates: list[list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if len(comps) < 2 or len(comps) % 2 != 0:
            continue

        boxes = []
        ok = True
        for comp in comps:
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            if len(comp) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
                ok = False
                break
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                ok = False
                break
            boxes.append(box)
        if not ok:
            continue

        by_rows: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for box in boxes:
            by_rows.setdefault((box[0], box[2]), []).append(box)
        if not by_rows or any(len(group) != 2 for group in by_rows.values()):
            continue

        pair_rows = []
        stripe_width = None
        for key in sorted(by_rows):
            left, right = sorted(by_rows[key], key=lambda box: box[1])
            gap = right[1] - left[3] - 1
            if gap <= 0:
                ok = False
                break
            if stripe_width is None:
                stripe_width = gap
            elif gap != stripe_width:
                ok = False
                break
            pair_rows.append((left, right))
        if ok:
            candidates.append(pair_rows)

    assert len(candidates) == 1, "expected one unique marker color made of paired 2x2 blocks"
    return candidates[0]


def _extract_stripes(grid: np.ndarray) -> list[np.ndarray]:
    stripes = []
    for left, right in _find_marker_pairs(grid):
        stripe = grid[left[0] : left[2] + 1, left[3] + 1 : right[1]].copy()
        red_positions = np.argwhere(stripe == RED_COLOR)
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        assert len(red_positions) == 1, "each stripe must contain exactly one red anchor"
        stripes.append(stripe)
    assert stripes, "expected at least one stripe"
    return stripes


def _align_and_stack(stripes: list[np.ndarray]) -> np.ndarray:
    aligned = [stripes[0].copy()]
    prev_red_col = int(np.argwhere(aligned[0] == RED_COLOR)[0, 1])
    for stripe in stripes[1:]:
        red_col = int(np.argwhere(stripe == RED_COLOR)[0, 1])
        rolled = np.roll(stripe, shift=prev_red_col - red_col, axis=1)
        new_red_col = int(np.argwhere(rolled == RED_COLOR)[0, 1])
        assert new_red_col == prev_red_col
        aligned.append(rolled)
    return np.vstack(aligned)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the stripe between each paired 2x2 marker block and stack the stripes after red-anchor phase alignment.

    Input structure:
    - The grid contains a dominant background, sparse distractor noise, and one marker color whose connected components are all solid 2x2 blocks.
    - Marker blocks appear as left/right pairs on the same two rows, with one 2-row stripe strictly between each pair.
    - Every stripe contains exactly one red pixel and otherwise consists of non-marker pattern colors.

    Rule:
    - Detect the unique marker color made only of row-aligned 2x2 pairs and extract the rectangle strictly between each left/right pair.
    - Keep the first stripe unchanged; for every later stripe, apply a cyclic horizontal roll so its red pixel lands in the same column as the previous aligned stripe's red pixel.
    - Stack the aligned stripes from top to bottom and discard the markers and all outside noise.

    Key insight:
    - The red pixel is a phase anchor inside each stripe, so the task is not plain extraction: the stripes must be cyclically normalized before stacking.
    """
    g = np.asarray(grid, dtype=int)
    return _align_and_stack(_extract_stripes(g))


def _make_stripe(width: int, palette: list[int], red_col: int, rng: random.Random) -> np.ndarray | None:
    stripe = np.empty((2, width), dtype=int)
    prev_pair: tuple[int, int] | None = None
    for c in range(width):
        for _ in range(40):
            pair = (rng.choice(palette), rng.choice(palette))
            if pair == prev_pair:
                continue
            stripe[0, c], stripe[1, c] = pair
            prev_pair = pair
            break
        else:
            return None
    stripe[rng.randrange(2), red_col] = RED_COLOR
    return stripe


def _sample_row_starts(num_pairs: int, rng: random.Random) -> tuple[int, list[int]]:
    gaps = [rng.randint(ROW_GAP_RANGE[0], ROW_GAP_RANGE[1]) for _ in range(num_pairs - 1)]
    used_height = 2 * num_pairs + sum(gaps)
    extra = rng.randint(0, EXTRA_HEIGHT_MAX)
    height = used_height + extra
    top = rng.randint(0, height - used_height)

    row_starts = []
    row = top
    for i in range(num_pairs):
        row_starts.append(row)
        if i < num_pairs - 1:
            row += 2 + gaps[i]
    return height, row_starts


def _noise_target(free_cells: int, rng: random.Random) -> int:
    low = max(1, int(round(free_cells * NOISE_RATE_RANGE[0])))
    high = max(low, int(round(free_cells * NOISE_RATE_RANGE[1])))
    return rng.randint(low, high)


def _scatter_noise(grid: np.ndarray, protected: np.ndarray, colors: list[int], rng: random.Random) -> int:
    h, w = grid.shape
    coords = [(r, c) for r in range(h) for c in range(w) if not protected[r, c]]
    rng.shuffle(coords)
    occupied = np.zeros_like(grid, dtype=bool)
    target = _noise_target(len(coords), rng)
    placed = 0

    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            continue
        grid[r, c] = rng.choice(colors)
        occupied[r, c] = True
        placed += 1
        if placed >= target:
            break
    return placed


def _expected_output(stripes: list[np.ndarray]) -> np.ndarray:
    return _align_and_stack([stripe.copy() for stripe in stripes])


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    marker_color = rng.choice(MARKER_CHOICES)
    palette = [c for c in range(1, 10) if c not in (marker_color, RED_COLOR)]
    noise_palette = palette[:]
    rng.shuffle(palette)
    palette = palette[: rng.randint(3, min(5, len(palette)))]

    num_pairs = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    width = rng.randint(STRIPE_WIDTH_RANGE[0], STRIPE_WIDTH_RANGE[1])

    raw_red_cols = [rng.randrange(width)]
    for i in range(1, num_pairs):
        if i == 1 and width > 1:
            choices = [c for c in range(width) if c != raw_red_cols[0]]
            raw_red_cols.append(rng.choice(choices))
        else:
            raw_red_cols.append(rng.randrange(width))

    stripes = []
    for red_col in raw_red_cols:
        stripe = _make_stripe(width, palette, red_col, rng)
        if stripe is None:
            return None
        stripes.append(stripe)

    height, row_starts = _sample_row_starts(num_pairs, rng)
    min_width = width + 4
    max_width = min(30, min_width + EXTRA_WIDTH_MAX)
    grid_width = rng.randint(min_width, max_width)
    if height > 30:
        return None

    grid = np.full((height, grid_width), BG_COLOR, dtype=int)
    protected = np.zeros_like(grid, dtype=bool)

    for row, stripe in zip(row_starts, stripes):
        left_col = rng.randint(0, grid_width - width - 4)
        right_col = left_col + width + 2
        grid[row : row + 2, left_col : left_col + 2] = marker_color
        grid[row : row + 2, left_col + 2 : right_col] = stripe
        grid[row : row + 2, right_col : right_col + 2] = marker_color

        r0 = max(0, row - 1)
        r1 = min(height, row + 3)
        c0 = max(0, left_col - 1)
        c1 = min(grid_width, right_col + 3)
        protected[r0:r1, c0:c1] = True

    noise_count = _scatter_noise(grid, protected, noise_palette, rng)
    if noise_count == 0:
        return None

    expected = _expected_output(stripes)
    if np.array_equal(expected, np.vstack(stripes)):
        return None

    bg_count = int(np.count_nonzero(grid == BG_COLOR))
    fg_count = grid.size - bg_count
    if bg_count <= fg_count:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
