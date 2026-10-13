from __future__ import annotations

import random

import numpy as np


BG = 0
FOREGROUND_COLORS = list(range(1, 10))
BAND_COUNT = 4
GEN_ATTEMPTS = 500

# Forced local coverage patterns. These guarantee that:
# - counts 0/1/2/3/4 all occur
# - all four band colors can be visible
# - even positive overlap cancels to black
# - odd visibility and topmost color both matter
ANCHOR_PATTERNS = (
    frozenset(),
    frozenset({0}),
    frozenset({1}),
    frozenset({2}),
    frozenset({3}),
    frozenset({0, 1}),
    frozenset({0, 2, 3}),
    frozenset({1, 2, 3}),
    frozenset({0, 1, 2, 3}),
)


def _band_colors_and_masks(grid: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % BAND_COUNT == 0, "expected four equal-height horizontal bands"

    band_height = height // BAND_COUNT
    assert band_height > 0 and width > 0, "expected non-empty bands"

    colors: list[int] = []
    masks: list[np.ndarray] = []
    for band_index in range(BAND_COUNT):
        band = g[band_index * band_height : (band_index + 1) * band_height]
        present = [int(value) for value in np.unique(band) if int(value) != BG]
        assert len(present) == 1, "each band must be a monochrome mask"
        color = present[0]
        assert np.all((band == BG) | (band == color)), "band may only contain black and its own color"
        colors.append(color)
        masks.append(band == color)

    assert len(set(colors)) == BAND_COUNT, "band colors must be distinct"
    return colors, masks


def _composite_from_masks(masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    cover = np.stack(masks, axis=0)
    counts = cover.sum(axis=0)
    visible = (counts % 2) == 1

    out = np.zeros(cover.shape[1:], dtype=int)
    top_index = np.argmax(cover, axis=0)
    for band_index, color in enumerate(colors):
        mask = visible & cover[band_index] & (top_index == band_index)
        out[mask] = int(color)
    return out


def _simple_priority_overlay(masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    out = np.zeros(masks[0].shape, dtype=int)
    for band_index in range(BAND_COUNT - 1, -1, -1):
        out[masks[band_index]] = int(colors[band_index])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse four stacked monochrome bands into one image using odd-coverage transparency.

    Input structure:
    - The grid is vertically split into four equal-height horizontal bands.
    - Each band is a black-background mask with exactly one distinct foreground color.
    - All four bands share the same local coordinate system and are read top to bottom.

    Rule:
    - Treat the first band as the frontmost transparent sheet and the fourth band as the backmost.
    - At each local position, count how many bands contain foreground at that coordinate.
    - If the count is odd, output the color of the frontmost covering band; if the count is even, output black.

    Key insight:
    - Brightness and hue come from different causes: parity decides whether a cell survives, while layer order decides which color is seen when it does survive.
    """
    colors, masks = _band_colors_and_masks(np.asarray(grid, dtype=int))
    return _composite_from_masks(masks, colors)


def _paint_random_footprint(mask: np.ndarray) -> None:
    h, w = mask.shape
    kind = random.choice(["rect", "hbar", "vbar", "plus", "corner"])

    if kind == "rect":
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        r1 = random.randrange(r0, h)
        c1 = random.randrange(c0, w)
        if (r1 - r0 + 1) * (c1 - c0 + 1) > max(2, (h * w) // 2):
            r1 = min(h - 1, r0 + max(0, h // 2))
            c1 = min(w - 1, c0 + max(0, w // 2))
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        return

    if kind == "hbar":
        r = random.randrange(h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        mask[r, c0 : c1 + 1] = True
        return

    if kind == "vbar":
        c = random.randrange(w)
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        mask[r0 : r1 + 1, c] = True
        return

    if kind == "plus":
        r = random.randrange(h)
        c = random.randrange(w)
        arm_r = random.randint(0, max(1, h // 2))
        arm_c = random.randint(0, max(1, w // 2))
        mask[max(0, r - arm_r) : min(h, r + arm_r + 1), c] = True
        mask[r, max(0, c - arm_c) : min(w, c + arm_c + 1)] = True
        return

    r = random.randrange(h)
    c = random.randrange(w)
    r1 = min(h, r + random.randint(1, max(1, h // 2)))
    c1 = min(w, c + random.randint(1, max(1, w // 2)))
    mask[r:r1, c] = True
    mask[r, c:c1] = True


def _random_band_mask(h: int, w: int) -> np.ndarray:
    min_area = max(3, (h * w) // 5)
    max_area = max(min_area + 2, int(np.ceil(h * w * 0.6)))

    for _ in range(100):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(2, 4)):
            _paint_random_footprint(mask)

        if random.random() < 0.45 and int(mask.sum()) > min_area + 1:
            painted = list(zip(*np.where(mask)))
            random.shuffle(painted)
            for r, c in painted[: random.randint(1, min(2, len(painted)))]:
                mask[r, c] = False

        area = int(mask.sum())
        if min_area <= area <= max_area and area < h * w:
            return mask

    raise RuntimeError("failed to sample a band mask")


def _pick_anchor_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    return cells[: len(ANCHOR_PATTERNS)]


def _apply_anchor_patterns(masks: list[np.ndarray]) -> None:
    h, w = masks[0].shape
    anchor_cells = _pick_anchor_cells(h, w)
    for (r, c), subset in zip(anchor_cells, ANCHOR_PATTERNS):
        for band_index, mask in enumerate(masks):
            mask[r, c] = band_index in subset


def _render_input(masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    h, w = masks[0].shape
    grid = np.zeros((h * BAND_COUNT, w), dtype=int)
    for band_index, (mask, color) in enumerate(zip(masks, colors)):
        band = grid[band_index * h : (band_index + 1) * h]
        band[mask] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        colors, masks = _band_colors_and_masks(grid)
        out = main(grid)
    except Exception:
        return False

    expected = _composite_from_masks(masks, colors)
    if not np.array_equal(out, expected):
        return False

    cover = np.stack(masks, axis=0)
    counts = cover.sum(axis=0)
    if any(not np.any(counts == value) for value in range(5)):
        return False

    simple = _simple_priority_overlay(masks, colors)
    if np.array_equal(simple, out):
        return False

    if not np.any((counts > 0) & (out == BG)):
        return False

    visible_top_layers = {
        int(np.argmax(cover[:, r, c]))
        for r, c in np.argwhere(out != BG)
    }
    if visible_top_layers != {0, 1, 2, 3}:
        return False

    unique_singletons = [np.any(mask & (counts == 1)) for mask in masks]
    if not all(unique_singletons):
        return False

    if len({int(value) for value in np.unique(out) if int(value) != BG}) != 4:
        return False

    if float(counts.mean()) > 2.2:
        return False

    if np.array_equal(out, masks[0].astype(int) * colors[0]):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    band_height = random.randint(3, 6)
    width = random.randint(6, 10)
    colors = random.sample(FOREGROUND_COLORS, BAND_COUNT)

    masks = [_random_band_mask(band_height, width) for _ in range(BAND_COUNT)]
    _apply_anchor_patterns(masks)

    if any(int(mask.sum()) == 0 or int(mask.sum()) == band_height * width for mask in masks):
        return None

    grid = _render_input(masks, colors)
    if _semantic_checks(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid odd-parity overlay input")
