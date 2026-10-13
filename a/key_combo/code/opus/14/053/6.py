from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(cols) != 1:
        raise ValueError("expected exactly one full-height grey separator column")
    return cols[0]


def _paint_ring(out: np.ndarray, center_r: int, center_c: int, radius: int, color: int) -> None:
    r0, r1 = center_r - radius, center_r + radius
    c0, c1 = center_c - radius, center_c + radius
    for c in range(c0, c1 + 1):
        if 0 <= r0 < out.shape[0] and 0 <= c < out.shape[1]:
            out[r0, c] = color
        if 0 <= r1 < out.shape[0] and 0 <= c < out.shape[1]:
            out[r1, c] = color
    for r in range(r0, r1 + 1):
        if 0 <= r < out.shape[0] and 0 <= c0 < out.shape[1]:
            out[r, c0] = color
        if 0 <= r < out.shape[0] and 0 <= c1 < out.shape[1]:
            out[r, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify symmetric overlap between two blue bitmasks after reflecting the left one across the separator.

    Input structure:
    - The grid has a black background with one full-height gray separator column.
    - The columns left of the separator and right of the separator form two equal-sized halves.
    - Each half contains a blue/black bitmask, with blue marking active pixels.

    Rule:
    - Reflect the left-half bitmask horizontally into the right-half coordinate system.
    - Compare the reflected left mask with the original right mask cell by cell.
    - Output only the right-half-sized comparison grid: overlap becomes red, left-only becomes green,
      right-only becomes yellow, and double-background stays black.

    Key insight:
    - Reflection is the alignment step: the two masks become comparable only after the left half is mirrored
      into the right half's coordinates, and the three output colors separate shared versus one-sided structure.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    if left.shape != right.shape:
        raise ValueError("left and right halves must have the same shape")

    left_mask = left == BLUE
    right_mask = right == BLUE
    reflected_left = np.fliplr(left_mask)

    out = np.zeros(right.shape, dtype=int)
    out[reflected_left & right_mask] = RED
    out[reflected_left & ~right_mask] = GREEN
    out[~reflected_left & right_mask] = YELLOW
    return out


def _grow_connected_mask(height: int, width: int, size: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    while int(mask.sum()) < size:
        cells = np.argwhere(mask)
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _grow_attached_cells(base: np.ndarray, extra_size: int) -> np.ndarray | None:
    height, width = base.shape
    occupied = base.copy()
    extra = np.zeros_like(base, dtype=bool)
    while int(extra.sum()) < extra_size:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(occupied):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not occupied[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        occupied[nr, nc] = True
        extra[nr, nc] = True
    return extra


def _mask_bbox(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)


def _render_input(reflected_left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, half_width = right.shape
    grid = np.zeros((height, half_width * 2 + 1), dtype=int)
    grid[:, half_width] = GREY
    grid[:, :half_width][np.fliplr(reflected_left)] = BLUE
    grid[:, half_width + 1 :][right] = BLUE
    return grid


def _expected_output(reflected_left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros(reflected_left.shape, dtype=int)
    out[reflected_left & right] = RED
    out[reflected_left & ~right] = GREEN
    out[~reflected_left & right] = YELLOW
    return out


def _wrong_unreflected_output(grid: np.ndarray) -> np.ndarray:
    sep = _find_separator_col(grid)
    left_mask = grid[:, :sep] == BLUE
    right_mask = grid[:, sep + 1 :] == BLUE
    out = np.zeros(right_mask.shape, dtype=int)
    out[left_mask & right_mask] = RED
    out[left_mask & ~right_mask] = GREEN
    out[~left_mask & right_mask] = YELLOW
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if not np.array_equal(out, expected):
        return False

    if out.shape != expected.shape or inp.ndim != 2 or out.ndim != 2:
        return False

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    if set(np.unique(inp)) - {BLACK, BLUE, GREY}:
        return False
    if set(np.unique(out)) - {BLACK, RED, GREEN, YELLOW}:
        return False

    if np.count_nonzero(out == RED) < 3:
        return False
    if np.count_nonzero(out == GREEN) < 2:
        return False
    if np.count_nonzero(out == YELLOW) < 2:
        return False

    mirrored_left = np.isin(out, (RED, GREEN))
    right = np.isin(out, (RED, YELLOW))
    if not np.array_equal(_render_input(mirrored_left, right), inp):
        return False

    wrong = _wrong_unreflected_output(inp)
    if np.array_equal(wrong, out):
        return False
    if np.count_nonzero(wrong != out) < 4:
        return False

    sep = _find_separator_col(inp)
    left_mask = inp[:, :sep] == BLUE
    if np.count_nonzero(left_mask != mirrored_left) < 4:
        return False

    bbox = _mask_bbox(mirrored_left | right)
    if bbox is None:
        return False
    if bbox[0] < 2 or bbox[1] < 3:
        return False

    return True


def _generate_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_shared = min(10, max(4, (height * width) // 6))
    shared_size = random.randint(4, max_shared)
    shared = _grow_connected_mask(height, width, shared_size)
    if shared is None:
        return None

    left_extra_size = random.randint(2, min(5, height * width - shared_size))
    right_extra_size = random.randint(2, min(5, height * width - shared_size))
    left_extra = _grow_attached_cells(shared, left_extra_size)
    right_extra = _grow_attached_cells(shared, right_extra_size)
    if left_extra is None or right_extra is None:
        return None

    reflected_left = shared | left_extra
    right = shared | right_extra

    if np.array_equal(left_extra, right_extra):
        return None
    if np.count_nonzero(left_extra ^ right_extra) < 3:
        return None

    return reflected_left, right


def _attempt_generate_input() -> np.ndarray | None:
    height = random.randint(8, 14)
    half_width = random.randint(7, 12)

    masks = _generate_masks(height, half_width)
    if masks is None:
        return None
    reflected_left, right = masks

    inp = _render_input(reflected_left, right)
    expected = _expected_output(reflected_left, right)
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _attempt_generate_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-bitmask example")

