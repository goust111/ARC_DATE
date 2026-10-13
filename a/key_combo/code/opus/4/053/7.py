import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


FOREGROUND_PAIRS = [
    (Color.BLUE, Color.RED),
    (Color.GREEN, Color.YELLOW),
    (Color.PINK, Color.ORANGE),
    (Color.TEAL, Color.MAROON),
    (Color.BLUE, Color.ORANGE),
    (Color.RED, Color.TEAL),
]

SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]

GRID_HEIGHT_RANGE = (8, 12)
HALF_WIDTH_RANGE = (5, 7)
SHAPE_COUNT_RANGE = (2, 4)
MIN_OCCUPIED = 6
GENERATION_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    """Return the most common grid color."""
    return int(Counter(map(int, np.asarray(grid).flatten())).most_common(1)[0][0])


def _swap_color(value: int, color_a: int, color_b: int) -> int:
    if int(value) == int(color_a):
        return int(color_b)
    if int(value) == int(color_b):
        return int(color_a)
    raise AssertionError("swap requested for a non-foreground color")


def _separator_col(grid: np.ndarray) -> int:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == Color.GREY)]
    assert len(sep_cols) == 1, "expected exactly one full-height grey separator"
    return int(sep_cols[0])


def _transform_left(
    left: np.ndarray,
    mirrored_right: np.ndarray,
    color_a: int,
    color_b: int,
    bg: int,
) -> np.ndarray:
    out = left.copy()
    h, w = left.shape
    for r in range(h):
        for c in range(w):
            left_value = int(left[r, c])
            right_value = int(mirrored_right[r, c])
            left_fg = left_value != int(bg)
            right_fg = right_value != int(bg)
            if left_fg and right_fg and left_value != right_value:
                out[r, c] = _swap_color(left_value, color_a, color_b)
            elif left_fg != right_fg:
                out[r, c] = left_value if left_fg else right_value
            else:
                out[r, c] = left_value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two folded halves and swap the left-half foreground colors only at disagreement cells.

    Input structure:
    - The grid has a black background and one full-height grey separator column.
    - The separator splits the canvas into equal-width left and right regions.
    - Apart from the separator, exactly two foreground colors appear in the two regions.

    Rule:
    - Mirror the right region horizontally so it aligns with the left region.
    - At cells where both aligned pixels are foreground but use different colors, swap the left pixel between the two foreground colors.
    - At cells where only one side is foreground, keep that visible foreground color; otherwise preserve the left pixel and output only the left region.

    Key insight:
    - The fold is a comparison tool: it reveals which left-half cells must be recolored, while unchanged matches and foreground-only mismatches stay as they are.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.shape[1] > 0

    fg_colors = sorted(
        int(color)
        for color in np.unique(g)
        if int(color) not in {bg, Color.GREY}
    )
    assert len(fg_colors) == 2, "expected exactly two non-separator foreground colors"
    color_a, color_b = fg_colors

    mirrored_right = np.fliplr(right)
    return _transform_left(left, mirrored_right, color_a, color_b, bg)


def _shape_variants(shape: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for base in (shape, np.fliplr(shape)):
        for k in range(4):
            rotated = np.rot90(base, k)
            key = rotated.tobytes() + bytes(rotated.shape)
            if key not in seen:
                seen.add(key)
                variants.append(rotated.copy())
    return variants


def _can_place(mask: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> bool:
    h, w = shape.shape
    if r0 < 0 or c0 < 0 or r0 + h > mask.shape[0] or c0 + w > mask.shape[1]:
        return False
    cells = np.argwhere(shape)
    for dr, dc in cells:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(mask.shape[0], rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(mask.shape[1], cc + 2)
        if np.any(mask[r1:r2, c1:c2]):
            return False
    return True


def _stamp(mask: np.ndarray, shape: np.ndarray, r0: int, c0: int) -> None:
    h, w = shape.shape
    view = mask[r0 : r0 + h, c0 : c0 + w]
    view[shape] = True


def _build_mask(height: int, width: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    for _ in range(shape_count):
        shape = random.choice(_ALL_SHAPE_VARIANTS)
        placed = False
        for _ in range(80):
            r0 = random.randint(0, height - shape.shape[0])
            c0 = random.randint(0, width - shape.shape[1])
            if _can_place(mask, shape, r0, c0):
                _stamp(mask, shape, r0, c0)
                placed = True
                break
        if not placed:
            return None

    if int(mask.sum()) < MIN_OCCUPIED:
        return None
    if int(mask.sum()) > (height * width) // 2:
        return None
    return mask


def _colorize_left(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray | None:
    left = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < MIN_OCCUPIED:
        return None

    random.shuffle(coords)
    split = random.randint(1, len(coords) - 1)
    first_group = coords[:split]
    second_group = coords[split:]
    if not first_group or not second_group:
        return None

    for r, c in first_group:
        left[r, c] = color_a
    for r, c in second_group:
        left[r, c] = color_b

    # Add local variation so colors are not separated into simple big blocks.
    for r, c in coords:
        if random.random() < 0.25:
            left[r, c] = _swap_color(left[r, c], color_a, color_b)

    if not np.any(left == color_a) or not np.any(left == color_b):
        return None
    return left


def _sample_modifications(
    left: np.ndarray,
    color_a: int,
    color_b: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    occ_coords = [tuple(map(int, rc)) for rc in np.argwhere(left != Color.BLACK)]
    by_color_a = [coord for coord in occ_coords if int(left[coord]) == int(color_a)]
    by_color_b = [coord for coord in occ_coords if int(left[coord]) == int(color_b)]
    bg_coords = [tuple(map(int, rc)) for rc in np.argwhere(left == Color.BLACK)]
    if not by_color_a or not by_color_b or len(bg_coords) < 1:
        return None

    diff_coords = [random.choice(by_color_a), random.choice(by_color_b)]
    remaining_occ = [coord for coord in occ_coords if coord not in set(diff_coords)]
    if not remaining_occ:
        return None

    extra_diff_max = min(2, len(remaining_occ) - 1)
    extra_diff_count = random.randint(0, max(0, extra_diff_max))
    if extra_diff_count:
        diff_coords.extend(random.sample(remaining_occ, extra_diff_count))

    remaining_occ = [coord for coord in occ_coords if coord not in set(diff_coords)]
    if not remaining_occ:
        return None

    left_only_count = 1 + int(len(remaining_occ) >= 2 and random.random() < 0.35)
    if len(remaining_occ) < left_only_count:
        return None
    left_only_coords = random.sample(remaining_occ, left_only_count)

    remaining_bg = bg_coords[:]
    right_only_count = 0
    if remaining_bg and random.random() < 0.5:
        right_only_count = 1 + int(len(remaining_bg) >= 2 and random.random() < 0.25)
    if len(remaining_bg) < right_only_count:
        return None
    right_only_coords = random.sample(remaining_bg, right_only_count)
    return diff_coords, left_only_coords, right_only_coords


def _compose_grid(
    left: np.ndarray,
    color_a: int,
    color_b: int,
    diff_coords: list[tuple[int, int]],
    left_only_coords: list[tuple[int, int]],
    right_only_coords: list[tuple[int, int]],
) -> np.ndarray:
    height, width = left.shape
    right = np.fliplr(left.copy())

    for r, c in diff_coords:
        right[r, width - 1 - c] = _swap_color(int(left[r, c]), color_a, color_b)
    for r, c in left_only_coords:
        right[r, width - 1 - c] = Color.BLACK
    for r, c in right_only_coords:
        right[r, width - 1 - c] = random.choice((color_a, color_b))

    full = np.full((height, 2 * width + 1), Color.BLACK, dtype=int)
    full[:, :width] = left
    full[:, width] = Color.GREY
    full[:, width + 1 :] = right
    return full


def _semantic_signature(grid: np.ndarray) -> tuple[int, int, int]:
    sep = _separator_col(grid)
    left = grid[:, :sep]
    mirrored_right = np.fliplr(grid[:, sep + 1 :])
    bg = Color.BLACK

    overlap_diff = 0
    left_only = 0
    right_only = 0
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            left_fg = int(left[r, c]) != bg
            right_fg = int(mirrored_right[r, c]) != bg
            if left_fg and right_fg and int(left[r, c]) != int(mirrored_right[r, c]):
                overlap_diff += 1
            elif left_fg and not right_fg:
                left_only += 1
            elif right_fg and not left_fg:
                right_only += 1
    return overlap_diff, left_only, right_only


def _validate_generated(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert get_bg(g) == Color.BLACK
    sep = _separator_col(g)
    left = g[:, :sep]
    mirrored_right = np.fliplr(g[:, sep + 1 :])
    fg_colors = sorted(
        int(color)
        for color in np.unique(g)
        if int(color) not in {Color.BLACK, Color.GREY}
    )
    assert len(fg_colors) == 2

    overlap_diff, left_only, _right_only = _semantic_signature(g)
    assert 2 <= overlap_diff <= 4
    assert left_only >= 1

    expected = _transform_left(left, mirrored_right, fg_colors[0], fg_colors[1], Color.BLACK)
    actual = main(g)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, left), "generator must rule out plain left-half copying"
    assert not np.array_equal(actual, mirrored_right), "generator must rule out plain mirrored-right copying"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*HALF_WIDTH_RANGE)
    color_a, color_b = random.choice(FOREGROUND_PAIRS)

    mask = _build_mask(height, width)
    if mask is None:
        return None

    left = _colorize_left(mask, color_a, color_b)
    if left is None:
        return None

    sampled = _sample_modifications(left, color_a, color_b)
    if sampled is None:
        return None
    diff_coords, left_only_coords, right_only_coords = sampled

    grid = _compose_grid(
        left,
        color_a,
        color_b,
        diff_coords,
        left_only_coords,
        right_only_coords,
    )
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")


_ALL_SHAPE_VARIANTS = [
    variant
    for shape in SHAPES
    for variant in _shape_variants(shape)
]
