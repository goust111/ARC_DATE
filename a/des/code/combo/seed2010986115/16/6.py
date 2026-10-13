from __future__ import annotations

import random

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 300
LAYER_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (3, 7)
INNER_WIDTH_RANGE = (3, 8)
MARGIN_RANGE = (1, 5)
CENTER_OFFSET_CHOICES = (-1, 0, 1)


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "each ring color must appear"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_direction(grid: np.ndarray, color: int) -> tuple[tuple[int, int, int, int], int]:
    r0, c0, r1, c1 = _bbox_of_color(grid, color)
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True

    assert not np.any(mask & ~expected), "ring pixels must lie on the bounding rectangle border"
    missing = np.argwhere(expected & ~mask)
    assert missing.shape[0] == 1, "each ring must have exactly one missing border pixel"

    mr, mc = map(int, missing[0])
    h, w = mask.shape
    assert h >= 3 and w >= 3, "rings must have an interior and a non-corner gap position"

    if mr == 0 and 0 < mc < w - 1:
        direction = 0
    elif mc == w - 1 and 0 < mr < h - 1:
        direction = 1
    elif mr == h - 1 and 0 < mc < w - 1:
        direction = 2
    elif mc == 0 and 0 < mr < h - 1:
        direction = 3
    else:
        raise AssertionError("gap must sit on exactly one non-corner edge")

    expected_area = int(expected.sum()) - 1
    assert int(mask.sum()) == expected_area, "ring must be a complete rectangle border except for the gap"
    return (r0, c0, r1, c1), direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular rings and encode each ring's missing-edge direction.

    The output has one row per ring from outermost to innermost and four columns:
    top, right, bottom, left. In each row, exactly one colored pixel marks the gap
    direction of that ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert 2 <= len(colors) <= 4, "expected between two and four ring colors"

    rings: list[tuple[tuple[int, int, int, int], int, int]] = []
    for color in colors:
        bbox, direction = _ring_direction(g, color)
        rings.append((bbox, direction, color))

    rings.sort(
        key=lambda item: (
            (item[0][2] - item[0][0] + 1) * (item[0][3] - item[0][1] + 1),
            item[0][2] - item[0][0],
            item[0][3] - item[0][1],
        ),
        reverse=True,
    )

    out = np.zeros((len(rings), 4), dtype=int)
    for row, (_bbox, direction, color) in enumerate(rings):
        out[row, direction] = color
    return out


def _paint_ring(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    direction: int,
    gap_index: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == 0:
        grid[r0, gap_index] = BG
    elif direction == 1:
        grid[gap_index, c1] = BG
    elif direction == 2:
        grid[r1, gap_index] = BG
    elif direction == 3:
        grid[gap_index, c0] = BG
    else:
        raise ValueError(f"invalid direction: {direction}")


def _sample_gap_index(rng: random.Random, bbox: tuple[int, int, int, int], direction: int) -> int:
    r0, c0, r1, c1 = bbox
    if direction in (0, 2):
        return rng.randint(c0 + 1, c1 - 1)
    return rng.randint(r0 + 1, r1 - 1)


def _build_expected(colors: list[int], directions: list[int]) -> np.ndarray:
    out = np.zeros((len(colors), 4), dtype=int)
    for row, (color, direction) in enumerate(zip(colors, directions, strict=True)):
        out[row, direction] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    layer_count = rng.randint(LAYER_RANGE[0], LAYER_RANGE[1])
    inner_h = rng.randint(INNER_HEIGHT_RANGE[0], INNER_HEIGHT_RANGE[1])
    inner_w = rng.randint(INNER_WIDTH_RANGE[0], INNER_WIDTH_RANGE[1])
    outer_h = inner_h + 4 * (layer_count - 1)
    outer_w = inner_w + 4 * (layer_count - 1)

    top_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
    bottom_margin = max(1, top_margin + rng.choice(CENTER_OFFSET_CHOICES))
    left_margin = rng.randint(MARGIN_RANGE[0], MARGIN_RANGE[1])
    right_margin = max(1, left_margin + rng.choice(CENTER_OFFSET_CHOICES))
    grid = np.zeros((outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin), dtype=int)

    palette = rng.sample(list(range(1, 10)), layer_count)
    directions = [rng.randrange(4) for _ in range(layer_count)]

    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_h - 1,
        left_margin + outer_w - 1,
    )
    for layer in range(layer_count):
        inset = 2 * layer
        bbox = (
            outer_bbox[0] + inset,
            outer_bbox[1] + inset,
            outer_bbox[2] - inset,
            outer_bbox[3] - inset,
        )
        direction = directions[layer]
        gap_index = _sample_gap_index(rng, bbox, direction)
        _paint_ring(grid, bbox, palette[layer], direction, gap_index)

    expected = _build_expected(palette, directions)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
