import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

MAX_ATTEMPTS = 400


def _find_separator_columns(grid: np.ndarray) -> tuple[int, int]:
    sep_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert sep_cols.size == 2, "expected exactly two full grey separator columns"
    left_sep, right_sep = map(int, sep_cols)
    assert 0 < left_sep < right_sep - 1 < grid.shape[1] - 1, "separators must split three nonempty panels"
    return left_sep, right_sep


def _sample_widths(rng: np.random.Generator) -> tuple[int, int, int]:
    for _ in range(MAX_ATTEMPTS):
        left_w = int(rng.integers(2, 7))
        mid_w = int(rng.integers(left_w + 1, min(left_w + 6, 10)))
        right_low = mid_w + 1
        right_high = min(mid_w + 6, 12)
        if right_low >= right_high:
            continue
        right_w = int(rng.integers(right_low, right_high))
        if left_w + mid_w + right_w + 2 <= 30:
            return left_w, mid_w, right_w
    raise RuntimeError("failed to sample panel widths")


def _random_rectangle(mask: np.ndarray, rng: np.random.Generator, value: bool) -> None:
    h, w = mask.shape
    r0 = int(rng.integers(0, h - 1))
    r1 = int(rng.integers(r0 + 1, h + 1))
    c0 = int(rng.integers(0, w - 1))
    c1 = int(rng.integers(c0 + 1, w + 1))
    mask[r0:r1, c0:c1] = value


def _sample_target_mask(
    height: int,
    right_w: int,
    left_w: int,
    mid_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    zone_a = slice(0, mid_w - left_w)
    zone_b = slice(mid_w - left_w, mid_w)
    zone_c = slice(mid_w, right_w)

    for _ in range(MAX_ATTEMPTS):
        row_sel = rng.random(height) < rng.uniform(0.35, 0.7)
        col_sel = rng.random(right_w) < rng.uniform(0.35, 0.7)
        mask = np.outer(row_sel, col_sel)

        for _ in range(int(rng.integers(1, 4))):
            _random_rectangle(mask, rng, True)

        if rng.random() < 0.7:
            _random_rectangle(mask, rng, False)

        if rng.random() < 0.5:
            band_r0 = int(rng.integers(0, height - 1))
            band_r1 = int(rng.integers(band_r0 + 1, height + 1))
            mask[band_r0:band_r1, :] |= (rng.random(right_w) < rng.uniform(0.25, 0.65))

        if rng.random() < 0.5:
            band_c0 = int(rng.integers(0, right_w - 1))
            band_c1 = int(rng.integers(band_c0 + 1, right_w + 1))
            mask[:, band_c0:band_c1] |= (rng.random((height, 1)) < rng.uniform(0.25, 0.65))

        density = float(mask.mean())
        if not 0.2 <= density <= 0.75:
            continue

        a = mask[:, zone_a]
        b = mask[:, zone_b]
        c = mask[:, zone_c]
        if a.sum() < 1 or a.size - a.sum() < 1:
            continue
        if b.sum() < 1 or b.size - b.sum() < 2:
            continue
        if c.sum() < 1 or c.size - c.sum() < 3:
            continue
        return mask.astype(bool)

    raise RuntimeError("failed to build a target mask")


def _enforce_zero_patterns(
    right: np.ndarray,
    mid_folded: np.ndarray,
    left: np.ndarray,
    target: np.ndarray,
    left_w: int,
    mid_w: int,
    rng: np.random.Generator,
) -> None:
    shift = mid_w - left_w
    zone2_zeros = np.argwhere(~target[:, :shift])
    zone3_zeros = np.argwhere(~target[:, shift:mid_w])

    if zone2_zeros.shape[0] >= 2:
        rng.shuffle(zone2_zeros)
        r0, c0 = map(int, zone2_zeros[0])
        r1, c1 = map(int, zone2_zeros[1])
        right[r0, c0], mid_folded[r0, c0] = True, False
        right[r1, c1], mid_folded[r1, c1] = False, True

    if zone3_zeros.shape[0] >= 3:
        rng.shuffle(zone3_zeros)
        for values, (rr, cc) in zip(
            ((True, True, False), (True, False, True), (False, True, True)),
            zone3_zeros[:3],
        ):
            r, c = map(int, (rr, cc))
            right[r, shift + c], mid_folded[r, shift + c], left[r, c] = values


def _sample_layers(
    target: np.ndarray,
    left_w: int,
    mid_w: int,
    right_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = target.shape[0]
    shift = mid_w - left_w

    right = np.zeros((height, right_w), dtype=bool)
    mid_folded = np.zeros((height, mid_w), dtype=bool)
    left = np.zeros((height, left_w), dtype=bool)

    right[:, mid_w:] = target[:, mid_w:]

    zone2_zero_choices = np.array(
        [
            [False, False],
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )
    zone2_probs = np.array([0.15, 0.425, 0.425])

    zone3_zero_choices = np.array(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    zone3_probs = np.array([0.08, 0.12, 0.12, 0.12, 0.19, 0.19, 0.18])

    for r in range(height):
        for c in range(mid_w):
            if target[r, c]:
                right[r, c] = True
                mid_folded[r, c] = True
                if c >= shift:
                    left[r, c - shift] = True
                continue

            if c < shift:
                choice = zone2_zero_choices[int(rng.choice(len(zone2_zero_choices), p=zone2_probs))]
                right[r, c], mid_folded[r, c] = choice
            else:
                choice = zone3_zero_choices[int(rng.choice(len(zone3_zero_choices), p=zone3_probs))]
                right[r, c], mid_folded[r, c], left[r, c - shift] = choice

    _enforce_zero_patterns(right, mid_folded, left, target, left_w, mid_w, rng)

    middle = np.fliplr(mid_folded)
    return left, middle, right


def _build_grid(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = left.shape[0]
    left_w = left.shape[1]
    mid_w = middle.shape[1]
    right_w = right.shape[1]
    grid = np.full((height, left_w + mid_w + right_w + 2), BLACK, dtype=int)
    grid[:, :left_w] = np.where(left, BLUE, BLACK)
    grid[:, left_w] = GREY
    grid[:, left_w + 1 : left_w + 1 + mid_w] = np.where(middle, BLUE, BLACK)
    grid[:, left_w + 1 + mid_w] = GREY
    grid[:, left_w + mid_w + 2 :] = np.where(right, BLUE, BLACK)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the left and middle panels successively onto the right panel and keep only cells where every covered layer is blue.

    Input structure:
    - The grid is split by two full grey separator columns into left, middle, and right panels.
    - Each panel contains only blue and black cells, and panel widths may differ.
    - The construction guarantees that both folds land completely inside the right panel.

    Rule:
    - Fold the left panel rightward across the first grey column onto the middle panel.
    - Fold the middle panel, together with the already folded left panel, rightward across the second grey column onto the right panel.
    - Output a grid with the right panel's size: color a cell red exactly when the right-panel cell itself is blue and every other folded layer landing there is also blue; otherwise color it black.

    Key insight:
    - The left panel is mirrored twice, so it reaches the right panel as a translation, while the middle panel reaches it as a single mirror image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    left_sep, right_sep = _find_separator_columns(g)
    left = g[:, :left_sep] == BLUE
    middle = g[:, left_sep + 1 : right_sep] == BLUE
    right = g[:, right_sep + 1 :] == BLUE

    left_w = left.shape[1]
    mid_w = middle.shape[1]
    right_w = right.shape[1]
    assert left_w <= mid_w <= right_w, "panel widths must allow both folds to land in the right panel"

    out_mask = right.copy()
    out_mask[:, :mid_w] &= np.fliplr(middle)
    start = mid_w - left_w
    out_mask[:, start : start + left_w] &= left

    out = np.full(right.shape, BLACK, dtype=int)
    out[out_mask] = RED
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        height = int(rng.integers(6, 13))
        left_w, mid_w, right_w = _sample_widths(rng)
        target = _sample_target_mask(height, right_w, left_w, mid_w, rng)
        left, middle, right = _sample_layers(target, left_w, mid_w, right_w, rng)
        grid = _build_grid(left, middle, right)
        if np.array_equal(main(grid) == RED, target):
            return grid
    raise RuntimeError("failed to generate a valid folding input")
