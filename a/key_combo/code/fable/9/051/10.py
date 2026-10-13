from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREY = 5
INPUT_FOREGROUNDS = (1, 3, 4, 6, 7, 8, 9)

HALF_HEIGHT_RANGE = (7, 12)
HALF_WIDTH_RANGE = (7, 13)
SHIFT_ZERO_PROB = 0.30
MAX_SHIFT = 3
CORE_SIZE_RANGE = (6, 16)
EXTRA_SIZE_RANGE = (2, 6)
GENERATION_ATTEMPTS = 500


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a boolean mask by (dr, dc) inside the same canvas, dropping overflow."""
    h, w = mask.shape
    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return np.zeros_like(mask, dtype=bool)

    dst_r0 = max(0, dr)
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = max(0, dc)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    out = np.zeros_like(mask, dtype=bool)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _overlap_mask(top_mask: np.ndarray, bottom_mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return top_mask & _shift_mask(bottom_mask, dr, dc)


def _all_translations(shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            yield dr, dc


def _best_translation(top_mask: np.ndarray, bottom_mask: np.ndarray) -> tuple[tuple[int, int], int, dict[tuple[int, int], int]]:
    assert top_mask.shape == bottom_mask.shape
    scores: dict[tuple[int, int], int] = {}
    for dr, dc in _all_translations(top_mask.shape):
        scores[(dr, dc)] = int(np.count_nonzero(_overlap_mask(top_mask, bottom_mask, dr, dc)))

    best_score = max(scores.values())
    best_shifts = [shift for shift, score in scores.items() if score == best_score]
    assert best_score > 0, "expected some foreground overlap under at least one translation"
    assert len(best_shifts) == 1, "expected a unique overlap-maximizing translation"
    return best_shifts[0], best_score, scores


def _extract_masks(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    total_h, _w = g.shape
    assert total_h % 2 == 1, "expected two equal halves separated by one row"

    sep = total_h // 2
    sep_row = g[sep, :]
    assert np.all(sep_row == sep_row[0]), "expected a uniform separator row"
    assert int(sep_row[0]) != BLACK, "separator row must be non-black"

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    assert BLACK in top_colors and BLACK in bottom_colors
    assert len(top_colors - {BLACK}) == 1, "top half must have one foreground color"
    assert len(bottom_colors - {BLACK}) == 1, "bottom half must have one foreground color"
    assert RED not in top_colors | bottom_colors, "red is reserved for the output"

    return top != BLACK, bottom != BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Register the lower mask to the upper mask at maximum overlap, then reveal their common foreground.

    Input structure:
    - The grid is split into two equal black-background halves by one full gray separator row.
    - The upper half contains exactly one non-black foreground color on black.
    - The lower half contains exactly one other non-black foreground color on black.

    Rule:
    - Treat each half as a foreground mask and test every integer translation of the lower mask.
    - Count overlap only on cells that remain inside the half after the translation, and choose the unique shift with the largest foreground-on-foreground overlap.
    - Output one half-sized grid: paint red where the upper mask and the best-shifted lower mask are both foreground, and black elsewhere.

    Key insight:
    - The visible pattern is a misregistered double exposure: direct cellwise intersection is a lure, and the common shape only appears after recovering the unique best translation.
    """
    top_mask, bottom_mask = _extract_masks(grid)
    (dr, dc), _score, _scores = _best_translation(top_mask, bottom_mask)
    overlap = _overlap_mask(top_mask, bottom_mask, dr, dc)

    out = np.zeros(top_mask.shape, dtype=int)
    out[overlap] = RED
    return out


def _random_shift(h: int, w: int) -> tuple[int, int]:
    if random.random() < SHIFT_ZERO_PROB:
        return 0, 0

    max_dr = min(MAX_SHIFT, h - 2)
    max_dc = min(MAX_SHIFT, w - 2)
    choices = [
        (dr, dc)
        for dr in range(-max_dr, max_dr + 1)
        for dc in range(-max_dc, max_dc + 1)
        if (dr, dc) != (0, 0) and max(abs(dr), abs(dc)) >= 1
    ]
    return random.choice(choices)


def _safe_rect_for_inverse_shift(h: int, w: int, dr: int, dc: int) -> tuple[int, int, int, int]:
    r0 = max(0, dr)
    r1 = min(h - 1, h - 1 + dr)
    c0 = max(0, dc)
    c1 = min(w - 1, w - 1 + dc)
    return r0, c0, r1, c1


def _allowed_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _grow_connected_mask(
    h: int,
    w: int,
    size_range: tuple[int, int],
    allowed: set[tuple[int, int]],
    forbidden: set[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    forbidden = forbidden or set()
    available = list(allowed - forbidden)
    if not available:
        return None

    max_size = min(size_range[1], len(available))
    if max_size < size_range[0]:
        return None
    target = random.randint(size_range[0], max_size)

    start = random.choice(available)
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in allowed and (nr, nc) not in cells and (nr, nc) not in forbidden:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_asymmetric(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    variants = (
        np.flipud(crop),
        np.fliplr(crop),
        np.rot90(crop),
        np.rot90(crop, 2),
        np.rot90(crop, 3),
    )
    return not any(np.array_equal(crop, variant) for variant in variants)


def _mask_to_colored(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _second_best_score(scores: dict[tuple[int, int], int], best_shift: tuple[int, int]) -> int:
    return max(score for shift, score in scores.items() if shift != best_shift)


def _validate_semantics(
    grid: np.ndarray,
    expected_shift: tuple[int, int],
    expected_overlap: np.ndarray,
) -> bool:
    try:
        top_mask, bottom_mask = _extract_masks(grid)
        best_shift, best_score, scores = _best_translation(top_mask, bottom_mask)
    except AssertionError:
        return False
    if best_shift != expected_shift:
        return False

    actual_overlap = _overlap_mask(top_mask, bottom_mask, *best_shift)
    if not np.array_equal(actual_overlap, expected_overlap):
        return False

    if best_score != int(np.count_nonzero(expected_overlap)):
        return False
    if best_score < 4:
        return False

    if np.array_equal(actual_overlap, top_mask):
        return False
    if np.array_equal(actual_overlap, _shift_mask(bottom_mask, *best_shift)):
        return False

    zero_score = scores[(0, 0)]
    alt_score = _second_best_score(scores, best_shift)
    if expected_shift != (0, 0):
        if not (0 < zero_score < best_score):
            return False
        if alt_score <= 0:
            return False
    else:
        if alt_score <= 0 or alt_score >= best_score:
            return False

    if not _is_asymmetric(actual_overlap):
        return False
    if RED in set(map(int, np.unique(grid))):
        return False
    return np.array_equal(main(grid), _mask_to_colored(expected_overlap, RED))


def _attempt_case() -> np.ndarray | None:
    h = random.randint(*HALF_HEIGHT_RANGE)
    w = random.randint(*HALF_WIDTH_RANGE)
    shift = _random_shift(h, w)
    safe_rect = _safe_rect_for_inverse_shift(h, w, *shift)
    safe_allowed = set(_allowed_cells(safe_rect))

    core = _grow_connected_mask(h, w, CORE_SIZE_RANGE, safe_allowed)
    if core is None or not _is_asymmetric(core):
        return None

    top_extra = _grow_connected_mask(
        h,
        w,
        EXTRA_SIZE_RANGE,
        set((r, c) for r in range(h) for c in range(w)),
        forbidden=set(map(tuple, np.argwhere(core))),
    )
    if top_extra is None:
        return None

    top_mask = core | top_extra
    bottom_extra = _grow_connected_mask(
        h,
        w,
        EXTRA_SIZE_RANGE,
        safe_allowed,
        forbidden=set(map(tuple, np.argwhere(top_mask))),
    )
    if bottom_extra is None:
        return None

    aligned_bottom = core | bottom_extra
    bottom_input = _shift_mask(aligned_bottom, -shift[0], -shift[1])

    if np.count_nonzero(bottom_input) != np.count_nonzero(aligned_bottom):
        return None

    if np.array_equal(top_mask, aligned_bottom):
        return None

    overlap = top_mask & aligned_bottom
    if not np.array_equal(overlap, core):
        return None

    top_color, bottom_color = random.sample(INPUT_FOREGROUNDS, 2)
    top_grid = _mask_to_colored(top_mask, top_color)
    bottom_grid = _mask_to_colored(bottom_input, bottom_color)
    separator = np.full((1, w), GREY, dtype=int)
    grid = np.vstack([top_grid, separator, bottom_grid])

    if not _validate_semantics(grid, shift, core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
