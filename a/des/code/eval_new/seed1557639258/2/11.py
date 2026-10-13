from __future__ import annotations

import random

import numpy as np


BG = 0
CREASE = 5
STATIONARY_COLORS = (1, 2)
FOLDED_COLORS = (3, 4)
MIX_FROM = {
    (1, 3): 6,
    (1, 4): 7,
    (2, 3): 8,
    (2, 4): 9,
}
INVERSE_MIX = {mix: pair for pair, mix in MIX_FROM.items()}
PAYLOAD_COLORS = set(STATIONARY_COLORS) | set(FOLDED_COLORS) | set(INVERSE_MIX)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_non_bg(arr: np.ndarray) -> np.ndarray:
    return _crop_to_bbox(arr, arr != BG)


def _extract_normalized_half(grid: np.ndarray) -> tuple[str, str, np.ndarray]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == CREASE)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == CREASE)]
    assert len(row_lines) + len(col_lines) == 1, "expected exactly one crease line"

    if col_lines:
        crease_col = int(col_lines[0])
        left_mask = np.isin(grid[:, :crease_col], list(PAYLOAD_COLORS))
        right_mask = np.isin(grid[:, crease_col + 1 :], list(PAYLOAD_COLORS))
        left_count = int(left_mask.sum())
        right_count = int(right_mask.sum())
        assert (left_count > 0) != (right_count > 0), "payload must lie on one side of the crease"

        if right_count > 0:
            side = "right"
            half = grid[:, crease_col + 1 :]
            region = _crop_to_bbox(half, right_mask)
            assert np.any(region[:, 0] != BG), "payload must touch the crease edge"
        else:
            side = "left"
            half = grid[:, :crease_col]
            region = _crop_to_bbox(half, left_mask)
            assert np.any(region[:, -1] != BG), "payload must touch the crease edge"
            region = np.fliplr(region)

        return "vertical", side, region

    crease_row = int(row_lines[0])
    top_mask = np.isin(grid[:crease_row, :], list(PAYLOAD_COLORS))
    bottom_mask = np.isin(grid[crease_row + 1 :, :], list(PAYLOAD_COLORS))
    top_count = int(top_mask.sum())
    bottom_count = int(bottom_mask.sum())
    assert (top_count > 0) != (bottom_count > 0), "payload must lie on one side of the crease"

    if bottom_count > 0:
        side = "bottom"
        half = grid[crease_row + 1 :, :]
        region = _crop_to_bbox(half, bottom_mask)
        assert np.any(region[0, :] != BG), "payload must touch the crease edge"
    else:
        side = "top"
        half = grid[:crease_row, :]
        region = _crop_to_bbox(half, top_mask)
        assert np.any(region[-1, :] != BG), "payload must touch the crease edge"
        region = np.flipud(region)

    return "horizontal", side, region


def _decode_layers(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stationary = np.full(region.shape, BG, dtype=int)
    folded = np.full(region.shape, BG, dtype=int)

    for color in np.unique(region):
        value = int(color)
        cells = region == value
        if value == BG:
            continue
        if value in STATIONARY_COLORS:
            stationary[cells] = value
        elif value in FOLDED_COLORS:
            folded[cells] = value
        elif value in INVERSE_MIX:
            base, flap = INVERSE_MIX[value]
            stationary[cells] = base
            folded[cells] = flap
        else:
            raise AssertionError(f"unexpected payload color {value}")

    return stationary, folded


def _compose_visible(stationary: np.ndarray, folded: np.ndarray) -> np.ndarray:
    visible = np.full(stationary.shape, BG, dtype=int)
    stationary_mask = stationary != BG
    folded_mask = folded != BG
    both = stationary_mask & folded_mask

    visible[stationary_mask & ~folded_mask] = stationary[stationary_mask & ~folded_mask]
    visible[folded_mask & ~stationary_mask] = folded[folded_mask & ~stationary_mask]

    for pair, mix in MIX_FROM.items():
        base, flap = pair
        visible[both & (stationary == base) & (folded == flap)] = mix

    return visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold a stacked half-pattern across the grey crease and separate mixed overlap colors into their two layers.

    Input structure:
    - The background is black, and one full grey row or column marks the crease line.
    - All non-grey foreground lies on exactly one side of the crease and forms the folded view of a larger picture.
    - Colors 1/2 belong to the stationary half, colors 3/4 belong to the folded half, and colors 6/7/8/9 are visible overlap colors.

    Rule:
    - Detect whether the grey crease is horizontal or vertical, and which side contains the stacked visible half.
    - Normalize that half so coordinates run outward from the crease, then decode each pixel: stationary-only colors stay on the visible side, folded-only colors belong to the mirrored hidden side, and each mix color is inverted to one fixed stationary/folded color pair.
    - Remove the crease, mirror the recovered folded layer back across it, and place the stationary layer on the side where it originally remained after folding.

    Key insight:
    - The overlap colors are not independent paints: each one is a reversible code for a specific stationary color covered by a specific folded color, so the visible half can be split back into two aligned layers before mirroring.
    """
    g = np.asarray(grid, dtype=int)
    orientation, side, region = _extract_normalized_half(g)
    stationary, folded = _decode_layers(region)

    if orientation == "vertical":
        if side == "right":
            out = np.concatenate([np.fliplr(folded), stationary], axis=1)
        else:
            out = np.concatenate([np.fliplr(stationary), folded], axis=1)
    else:
        if side == "bottom":
            out = np.concatenate([np.flipud(folded), stationary], axis=0)
        else:
            out = np.concatenate([np.flipud(stationary), folded], axis=0)

    return _crop_non_bg(out)


MODE_CHOICES = ("left", "right", "top", "bottom")
HALF_HEIGHT_RANGE = (5, 12)
HALF_WIDTH_RANGE = (5, 12)
RECT_COUNT_RANGE = (2, 4)
MARGIN_RANGE = (0, 3)
MAX_ATTEMPTS = 500


def _random_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(random.randint(*RECT_COUNT_RANGE)):
        if random.random() < 0.5:
            rect_h = random.randint(max(1, height // 4), height)
            rect_w = random.randint(1, max(1, width // 2))
        else:
            rect_h = random.randint(1, max(1, height // 2))
            rect_w = random.randint(max(1, width // 4), width)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    if random.random() < 0.5:
        if random.random() < 0.5:
            row = random.randrange(height)
            c0 = random.randrange(width)
            c1 = random.randrange(c0, width)
            mask[row, c0 : c1 + 1] = True
        else:
            col = random.randrange(width)
            r0 = random.randrange(height)
            r1 = random.randrange(r0, height)
            mask[r0 : r1 + 1, col] = True

    return mask


def _make_layer_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(100):
        stationary = _random_mask(height, width)
        folded = _random_mask(height, width)
        union = stationary | folded
        if not union.any():
            continue

        stationary = _crop_to_bbox(stationary.astype(int), union).astype(bool)
        folded = _crop_to_bbox(folded.astype(int), union).astype(bool)
        union = stationary | folded
        overlap = stationary & folded
        stationary_only = stationary & ~folded
        folded_only = folded & ~stationary

        area = int(union.sum())
        h, w = stationary.shape
        if overlap.sum() < 2 or stationary_only.sum() < 2 or folded_only.sum() < 2:
            continue
        if area < 12 or area > int(0.8 * h * w):
            continue
        if h < 4 or w < 4 or h > 15 or w > 15:
            continue

        return stationary, folded

    return None


def _paint_layer(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(mask.shape)
    for _ in range(40):
        phase = random.randint(0, 1)
        scheme = random.choice(("rows", "cols", "diag", "blocks"))
        if scheme == "rows":
            band = random.randint(1, max(1, mask.shape[0] // 3))
            pattern = ((rr // band) + phase) % 2
        elif scheme == "cols":
            band = random.randint(1, max(1, mask.shape[1] // 3))
            pattern = ((cc // band) + phase) % 2
        elif scheme == "diag":
            pattern = (rr + cc + phase) % 2
        else:
            band_r = random.randint(1, max(1, mask.shape[0] // 2))
            band_c = random.randint(1, max(1, mask.shape[1] // 2))
            pattern = ((rr // band_r) + (cc // band_c) + phase) % 2

        painted = np.full(mask.shape, BG, dtype=int)
        painted[mask] = np.where(pattern[mask] == 0, colors[0], colors[1])
        if len(np.unique(painted[mask])) == 2:
            return painted

    painted = np.full(mask.shape, BG, dtype=int)
    split = max(1, mask.shape[1] // 2)
    left = mask.copy()
    left[:, split:] = False
    painted[mask] = colors[1]
    painted[left] = colors[0]
    return painted


def _expected_output(mode: str, stationary: np.ndarray, folded: np.ndarray) -> np.ndarray:
    if mode == "left":
        out = np.concatenate([np.fliplr(folded), stationary], axis=1)
    elif mode == "right":
        out = np.concatenate([np.fliplr(stationary), folded], axis=1)
    elif mode == "top":
        out = np.concatenate([np.flipud(folded), stationary], axis=0)
    else:
        out = np.concatenate([np.flipud(stationary), folded], axis=0)
    return _crop_non_bg(out)


def _render_input(mode: str, visible: np.ndarray) -> np.ndarray | None:
    height, width = visible.shape
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    if mode in {"left", "right"}:
        grid_h = top + height + bottom
        grid_w = left + width + 1 + right
        if grid_h > 30 or grid_w > 30:
            return None
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        if mode == "left":
            crease_col = left
            grid[:, crease_col] = CREASE
            grid[top : top + height, crease_col + 1 : crease_col + 1 + width] = visible
        else:
            crease_col = left + width
            grid[:, crease_col] = CREASE
            grid[top : top + height, left : left + width] = np.fliplr(visible)
        return grid

    grid_h = top + height + 1 + bottom
    grid_w = left + width + right
    if grid_h > 30 or grid_w > 30:
        return None
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    if mode == "top":
        crease_row = top
        grid[crease_row, :] = CREASE
        grid[crease_row + 1 : crease_row + 1 + height, left : left + width] = visible
    else:
        crease_row = top + height
        grid[crease_row, :] = CREASE
        grid[top : top + height, left : left + width] = np.flipud(visible)
    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODE_CHOICES)
    raw_height = random.randint(*HALF_HEIGHT_RANGE)
    raw_width = random.randint(*HALF_WIDTH_RANGE)
    masks = _make_layer_masks(raw_height, raw_width)
    if masks is None:
        return None

    stationary_mask, folded_mask = masks
    stationary = _paint_layer(stationary_mask, STATIONARY_COLORS)
    folded = _paint_layer(folded_mask, FOLDED_COLORS)
    visible = _compose_visible(stationary, folded)

    mix_count = int(np.isin(visible, list(INVERSE_MIX)).sum())
    if mix_count < 2:
        return None

    expected = _expected_output(mode, stationary, folded)
    if max(expected.shape) > 30:
        return None

    grid = _render_input(mode, visible)
    if grid is None or max(grid.shape) > 30:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-pattern input")
