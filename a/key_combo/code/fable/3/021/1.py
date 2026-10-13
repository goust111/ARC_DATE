from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

ACTIVE_COLORS = (RED, 5, 6, 7, 8, 9)
GENERATOR_RETRIES = 500

_GEN_CALL_COUNT = 0


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    assert w % 2 == 1, "expected equal-width panels around one separator column"
    sep_c = w // 2

    col = grid[:, sep_c]
    marker_rows = np.argwhere(col == YELLOW).flatten()
    assert marker_rows.size == 1, "expected exactly one yellow marker on the separator"
    marker_r = int(marker_rows[0])
    assert np.all((col == BLUE) | (col == YELLOW)), "separator must be blue except for the marker"

    other_yellow = np.argwhere(grid == YELLOW)
    assert other_yellow.shape[0] == 1, "yellow must appear only as the separator marker"
    assert int(other_yellow[0, 0]) == marker_r and int(other_yellow[0, 1]) == sep_c

    return sep_c, marker_r


def _active_color(left: np.ndarray, right: np.ndarray) -> int:
    panel_colors = set(map(int, np.unique(left))) | set(map(int, np.unique(right)))
    panel_colors.discard(BLACK)
    assert len(panel_colors) == 1, "the two panels must share one active mask color"
    color = int(next(iter(panel_colors)))
    assert color not in (BLUE, GREEN, YELLOW), "active color must be distinct from marker and output colors"
    assert np.any(left == color) and np.any(right == color), "both masks must contain active cells"
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the row marker on the separator, roll the right mask, and output the aligned XOR.

    Input structure:
    - A black-background grid is split by one full-height blue vertical separator column.
    - The separator contains exactly one yellow marker pixel; its row is the encoded shift.
    - The left and right panels have the same shape and use one shared active color plus black.

    Rule:
    - Let r be the zero-based row of the yellow marker on the separator.
    - Cyclically roll the right panel downward by r rows in the panel coordinate frame.
    - Compare the left active mask with the rolled right active mask cell by cell.
    - Output green where exactly one aligned mask is active, and black everywhere else.

    Key insight:
    - The separator is also a key: the logical combination is self-consistent only after using
      the marker row to align the right mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_c, marker_r = _find_separator(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    assert left.shape == right.shape and left.size > 0, "separator must create equal non-empty panels"

    active = _active_color(left, right)
    assert np.all((left == BLACK) | (left == active)), "left panel must be a binary mask"
    assert np.all((right == BLACK) | (right == active)), "right panel must be a binary mask"

    left_mask = left == active
    right_mask = right == active
    aligned_right = np.roll(right_mask, shift=marker_r, axis=0)

    out = np.full(left.shape, BLACK, dtype=int)
    out[left_mask ^ aligned_right] = GREEN
    return out


_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
)


def _rotated_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = list(shape)
    for _ in range(random.randrange(4)):
        max_c = max(c for _r, c in cells)
        cells = [(c, max_c - r) for r, c in cells]
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _can_place(mask: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> bool:
    h, w = mask.shape
    for dr, dc in shape:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(mask[r, c]):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                    return False
    return True


def _random_mask(h: int, w: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    target_components = random.randint(2, min(6, max(2, (h * w) // 18 + 2)))

    for _ in range(target_components):
        placed = False
        for _attempt in range(100):
            shape = _rotated_shape(random.choice(_SHAPES))
            max_r = max(r for r, _c in shape)
            max_c = max(c for _r, c in shape)
            r0 = random.randint(0, h - max_r - 1)
            c0 = random.randint(0, w - max_c - 1)
            if not _can_place(mask, shape, r0, c0):
                continue
            for dr, dc in shape:
                mask[r0 + dr, c0 + dc] = True
            placed = True
            break
        if not placed:
            return None

    active = int(np.count_nonzero(mask))
    if active < 4 or active > (h * w) // 2:
        return None
    return mask


def _ensure_relation_witnesses(left: np.ndarray, right_aligned: np.ndarray) -> np.ndarray | None:
    adjusted = left.copy()

    overlap = np.argwhere(right_aligned)
    empty_against_right = np.argwhere(~right_aligned)
    if overlap.size == 0 or empty_against_right.size == 0:
        return None

    if not np.any(adjusted & right_aligned):
        r, c = map(int, random.choice(overlap))
        adjusted[r, c] = True

    if not np.any((~adjusted) & right_aligned):
        candidates = np.argwhere(right_aligned)
        random.shuffle(candidates)
        for r, c in candidates:
            if np.count_nonzero(adjusted) > 4:
                adjusted[int(r), int(c)] = False
                break

    if not np.any(adjusted & (~right_aligned)):
        r, c = map(int, random.choice(empty_against_right))
        adjusted[r, c] = True

    return adjusted


def _valid_masks(left: np.ndarray, right: np.ndarray, marker_r: int) -> bool:
    h, w = left.shape
    area = h * w
    right_aligned = np.roll(right, shift=marker_r, axis=0)
    out = left ^ right_aligned

    left_count = int(np.count_nonzero(left))
    right_count = int(np.count_nonzero(right))
    out_count = int(np.count_nonzero(out))
    if min(left_count, right_count) < 4:
        return False
    if max(left_count, right_count) > int(area * 0.55):
        return False
    if not (max(4, area // 10) <= out_count <= int(area * 0.68)):
        return False

    if not np.any(left & right_aligned):
        return False
    if not np.any(left & ~right_aligned):
        return False
    if not np.any((~left) & right_aligned):
        return False
    if not np.any((~left) & (~right_aligned)):
        return False

    matching_shifts = [
        s for s in range(h) if np.array_equal(left ^ np.roll(right, shift=s, axis=0), out)
    ]
    if matching_shifts != [int(marker_r)]:
        return False

    if marker_r != 0:
        direct = left ^ right
        diff = int(np.count_nonzero(direct != out))
        if diff < max(2, area // 10):
            return False

    return True


def _compose_input(left: np.ndarray, right: np.ndarray, marker_r: int, active: int) -> np.ndarray:
    h, w = left.shape
    grid = np.full((h, 2 * w + 1), BLACK, dtype=int)
    grid[:, :w][left] = active
    grid[:, w] = BLUE
    grid[int(marker_r), w] = YELLOW
    grid[:, w + 1 :][right] = active
    return grid


def _marker_mode() -> str:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    phase = _GEN_CALL_COUNT % 3
    if phase == 1:
        return "zero"
    if phase == 2:
        return "nonzero"
    return "random"


def _sample_marker_row(h: int, mode: str) -> int:
    if mode == "zero":
        return 0
    if mode == "nonzero":
        return random.randint(1, h - 1)
    return random.randint(0, h - 1)


def _gen_attempt(mode: str) -> np.ndarray | None:
    h = random.randint(5, 14)
    w = random.randint(4, 14)
    marker_r = _sample_marker_row(h, mode)

    right = _random_mask(h, w)
    left = _random_mask(h, w)
    if right is None or left is None:
        return None

    right_aligned = np.roll(right, shift=marker_r, axis=0)
    left = _ensure_relation_witnesses(left, right_aligned)
    if left is None or not _valid_masks(left, right, marker_r):
        return None

    active = random.choice(ACTIVE_COLORS)
    grid = _compose_input(left, right, marker_r, active)

    expected = np.full((h, w), BLACK, dtype=int)
    expected[left ^ right_aligned] = GREEN
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _marker_mode()
    for _ in range(GENERATOR_RETRIES):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid aligned-key XOR input")
