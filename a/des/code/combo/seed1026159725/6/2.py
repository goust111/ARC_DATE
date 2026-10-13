import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8

# Bit order: red=1, blue=2, green=4.
COLOR_BY_BITS = np.array([BLACK, RED, BLUE, MAGENTA, GREEN, YELLOW, AZURE, GREY], dtype=int)
BITS_BY_COLOR = np.zeros(10, dtype=int)
BITS_BY_COLOR[BLACK] = 0
BITS_BY_COLOR[RED] = 1
BITS_BY_COLOR[BLUE] = 2
BITS_BY_COLOR[GREEN] = 4
BITS_BY_COLOR[MAGENTA] = 3
BITS_BY_COLOR[YELLOW] = 5
BITS_BY_COLOR[AZURE] = 6
BITS_BY_COLOR[GREY] = 7
ALLOWED_COLORS = np.array([BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, AZURE], dtype=int)

MODES = ("disjoint", "one_pair", "chain", "triple")
GEN_RETRIES = 200


def _decode_bits(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g < 10))
    assert np.all(np.isin(g, ALLOWED_COLORS))
    return BITS_BY_COLOR[g]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose the mixed overlay colors into the original red, blue, and green rectangle masks.

    Output three panels of the same size as the input, separated by a one-column black gap:
    red mask, blue mask, green mask.
    """
    bits = _decode_bits(grid)
    h, _w = bits.shape
    sep = np.zeros((h, 1), dtype=int)

    red_panel = np.where(bits & 1, RED, BLACK)
    blue_panel = np.where(bits & 2, BLUE, BLACK)
    green_panel = np.where(bits & 4, GREEN, BLACK)
    return np.concatenate([red_panel, sep, blue_panel, sep, green_panel], axis=1)


def _rect_mask(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _compose_grid(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    bits = np.zeros(shape, dtype=int)
    for i, rect in enumerate(rects):
        bits += _rect_mask(rect, shape).astype(int) * (1 << i)
    return COLOR_BY_BITS[bits]


def _relation_stats(
    rects: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> tuple[list[int], dict[tuple[int, int], int], dict[tuple[int, int], int], int]:
    masks = [_rect_mask(rect, shape) for rect in rects]
    bits = np.zeros(shape, dtype=int)
    for i, mask in enumerate(masks):
        bits += mask.astype(int) * (1 << i)

    exclusives = [int(np.sum(bits == (1 << i))) for i in range(3)]
    pair_only = {
        (0, 1): int(np.sum(bits == 3)),
        (0, 2): int(np.sum(bits == 5)),
        (1, 2): int(np.sum(bits == 6)),
    }
    pair_any = {
        (0, 1): int(np.sum(masks[0] & masks[1])),
        (0, 2): int(np.sum(masks[0] & masks[2])),
        (1, 2): int(np.sum(masks[1] & masks[2])),
    }
    triple = int(np.sum(bits == 7))
    return exclusives, pair_only, pair_any, triple


def _validate_mode(mode: str, rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> None:
    exclusives, pair_only, pair_any, triple = _relation_stats(rects, shape)
    assert all(v > 0 for v in exclusives)

    pair_only_hits = sum(v > 0 for v in pair_only.values())
    pair_any_hits = sum(v > 0 for v in pair_any.values())
    if mode == "disjoint":
        assert pair_any_hits == 0 and triple == 0
    elif mode == "one_pair":
        assert pair_only_hits == 1 and pair_any_hits == 1 and triple == 0
    elif mode == "chain":
        assert pair_only_hits == 2 and pair_any_hits == 2 and triple == 0
    elif mode == "triple":
        assert pair_any_hits == 3 and triple > 0
    else:
        raise AssertionError(mode)


def _apply_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    turns = random.randrange(4)
    g = np.rot90(g, turns)
    if random.random() < 0.5:
        g = np.fliplr(g)
    return g.copy()


def _build_disjoint() -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    h1, w1 = random.randint(2, 5), random.randint(2, 5)
    h2, w2 = random.randint(2, 5), random.randint(2, 5)
    h3, w3 = random.randint(2, 5), random.randint(2, 5)
    gap_x = random.randint(1, 3)
    gap_y = random.randint(1, 3)
    row2 = random.randint(0, 2)

    rect1 = (0, 0, h1 - 1, w1 - 1)
    col2 = w1 + gap_x
    rect2 = (row2, col2, row2 + h2 - 1, col2 + w2 - 1)

    top_h = max(h1, row2 + h2)
    row3 = top_h + gap_y
    max_col3 = max(col2 + w2, w1)
    col3 = random.randint(0, max(0, max_col3 - w3))
    rect3 = (row3, col3, row3 + h3 - 1, col3 + w3 - 1)

    shape = (row3 + h3, max(max_col3, col3 + w3))
    return [rect1, rect2, rect3], shape


def _build_one_pair() -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    top_only = random.randint(1, 3)
    common_h = random.randint(2, 4)
    bottom_only = random.randint(1, 3)
    left_only = random.randint(1, 3)
    common_w = random.randint(2, 4)
    right_only = random.randint(1, 3)

    rect1 = (0, 0, top_only + common_h - 1, left_only + common_w - 1)
    rect2 = (
        top_only,
        left_only,
        top_only + common_h + bottom_only - 1,
        left_only + common_w + right_only - 1,
    )

    pair_h = max(rect1[2] + 1, rect2[2] + 1)
    pair_w = max(rect1[3] + 1, rect2[3] + 1)
    h3, w3 = random.randint(2, 5), random.randint(2, 5)
    gap = random.randint(1, 3)

    if random.random() < 0.5:
        row3 = random.randint(0, max(0, pair_h - h3))
        col3 = pair_w + gap
        shape = (max(pair_h, row3 + h3), col3 + w3)
    else:
        row3 = pair_h + gap
        col3 = random.randint(0, max(0, pair_w - w3))
        shape = (row3 + h3, max(pair_w, col3 + w3))

    rect3 = (row3, col3, row3 + h3 - 1, col3 + w3 - 1)
    return [rect1, rect2, rect3], shape


def _build_chain() -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    height = random.randint(2, 5)
    left_only = random.randint(1, 3)
    overlap1 = random.randint(1, 3)
    middle_only = random.randint(1, 3)
    overlap2 = random.randint(1, 3)
    right_only = random.randint(1, 3)

    rect1 = (0, 0, height - 1, left_only + overlap1 - 1)
    rect2 = (0, left_only, height - 1, left_only + overlap1 + middle_only + overlap2 - 1)
    rect3 = (
        0,
        left_only + overlap1 + middle_only,
        height - 1,
        left_only + overlap1 + middle_only + overlap2 + right_only - 1,
    )
    shape = (height, rect3[3] + 1)
    return [rect1, rect2, rect3], shape


def _build_triple() -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    top_only = random.randint(1, 3)
    common_h = random.randint(2, 4)
    blue_left_only = random.randint(1, 3)
    red_blue_only = random.randint(1, 3)
    common_w = random.randint(2, 4)
    green_right_only = random.randint(1, 3)

    rect1 = (
        0,
        blue_left_only,
        top_only + common_h - 1,
        blue_left_only + red_blue_only + common_w - 1,
    )
    rect2 = (
        top_only,
        0,
        top_only + common_h - 1,
        blue_left_only + red_blue_only + common_w - 1,
    )
    rect3 = (
        top_only,
        blue_left_only + red_blue_only,
        top_only + common_h - 1,
        blue_left_only + red_blue_only + common_w + green_right_only - 1,
    )
    shape = (top_only + common_h, rect3[3] + 1)
    return [rect1, rect2, rect3], shape


def _build_rectangles(mode: str) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    if mode == "disjoint":
        return _build_disjoint()
    if mode == "one_pair":
        return _build_one_pair()
    if mode == "chain":
        return _build_chain()
    if mode == "triple":
        return _build_triple()
    raise AssertionError(mode)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        mode = random.choice(MODES)
        rects, shape = _build_rectangles(mode)
        _validate_mode(mode, rects, shape)

        grid = _compose_grid(rects, shape)
        grid = _apply_symmetry(grid)

        pad_top = random.randint(1, 4)
        pad_bottom = random.randint(1, 4)
        pad_left = random.randint(1, 4)
        pad_right = random.randint(1, 4)
        grid = np.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=BLACK)

        bits = _decode_bits(grid)
        assert np.any(bits == 1) or np.any(bits == 3) or np.any(bits == 5) or np.any(bits == 7)
        assert np.any(bits == 2) or np.any(bits == 3) or np.any(bits == 6) or np.any(bits == 7)
        assert np.any(bits == 4) or np.any(bits == 5) or np.any(bits == 6) or np.any(bits == 7)
        return grid

    raise RuntimeError("failed to generate a valid overlay puzzle instance")
