from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5
INPUT_BG_CHOICES = [0, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_separator_row(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    candidates = []
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if not np.all(grid[r, :] == color):
            continue
        if int(np.count_nonzero(grid == color)) != w:
            continue
        if 2 * r + 1 != h:
            continue
        candidates.append(r)
    assert len(candidates) == 1, "expected one unique full-width separator row"
    return candidates[0]


def _unique_anchor(half: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(half == RED)
    assert coords.shape == (1, 2), "each half must contain exactly one red anchor"
    return tuple(map(int, coords[0]))


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r + dr)
        cc = int(c + dc)
        assert 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1], "shifted mask left canvas"
        out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the lower grey mask to the upper red anchor, then color symmetric difference green and overlap blue.

    Input structure:
    - A single full-width separator row uses a color that appears nowhere else and splits
      the canvas into equal-height top and bottom halves with the same background color.
    - Each half contains exactly one red anchor pixel and a grey foreground mask.
    - The lower half is an offset copy in anchor coordinates: after translating by the
      anchor offset, its grey mask lies fully inside the half-canvas.

    Rule:
    - Remove the separator row and treat the upper half as the output canvas.
    - Translate the lower half so its red anchor coincides with the upper red anchor.
    - Color cells that are grey in exactly one aligned mask green, cells grey in both masks
      blue, leave everything else as background, and color the shared anchor red.

    Key insight:
    - The two halves must first be registered by their red anchors; only after that anchor-based
      translation does the intended overlap partition into blue intersection and green xor regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    sep_r = _find_separator_row(g)
    top = g[:sep_r, :]
    bottom = g[sep_r + 1 :, :]
    assert top.shape == bottom.shape, "separator row must split equal halves"

    bg_top = _most_common_color(top)
    bg_bottom = _most_common_color(bottom)
    assert bg_top == bg_bottom, "top and bottom halves must share a background color"

    top_anchor = _unique_anchor(top)
    bottom_anchor = _unique_anchor(bottom)
    top_mask = top == GREY
    bottom_mask = bottom == GREY

    dr = int(top_anchor[0] - bottom_anchor[0])
    dc = int(top_anchor[1] - bottom_anchor[1])
    aligned_bottom = _shift_mask(bottom_mask, dr, dc)

    out = np.full(top.shape, bg_top, dtype=int)
    overlap = top_mask & aligned_bottom
    xor_mask = top_mask ^ aligned_bottom
    out[xor_mask] = GREEN
    out[overlap] = BLUE
    out[top_anchor] = RED
    return out


def _grow_connected(
    allowed: set[tuple[int, int]],
    start: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    cells = set(start)
    if len(cells) > target_size:
        return None
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) in allowed and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(sorted(frontier)))
    return cells


def _grow_extension(
    allowed: set[tuple[int, int]],
    base: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    extra: set[tuple[int, int]] = set()
    while len(extra) < target_size:
        frontier = set()
        source = base | extra
        for r, c in source:
            for rr, cc in _neighbors4(r, c):
                cell = (rr, cc)
                if cell not in allowed or cell in source or cell in blocked:
                    continue
                frontier.add(cell)
        if not frontier:
            return None
        extra.add(random.choice(sorted(frontier)))
    return extra


def _build_connected_masks(
    half_h: int,
    half_w: int,
    anchor: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    ar, ac = anchor
    up = random.randint(1, min(4, ar))
    down = random.randint(1, min(4, half_h - 1 - ar))
    left = random.randint(1, min(4, ac))
    right = random.randint(1, min(4, half_w - 1 - ac))
    allowed = {
        (r, c)
        for r in range(ar - up, ar + down + 1)
        for c in range(ac - left, ac + right + 1)
        if (r, c) != anchor
    }

    anchor_neighbors = [cell for cell in _neighbors4(ar, ac) if cell in allowed]
    if not anchor_neighbors:
        return None
    start_cell = random.choice(anchor_neighbors)

    max_shared = min(len(allowed) - 2, 8)
    if max_shared < 2:
        return None
    shared_size = random.randint(2, max_shared)
    shared = _grow_connected(allowed, {start_cell}, shared_size)
    if shared is None:
        return None

    remaining = len(allowed) - len(shared)
    if remaining < 2:
        return None
    top_extra_size = random.randint(1, min(4, remaining - 1))
    bottom_extra_max = min(4, len(allowed) - len(shared) - top_extra_size)
    if bottom_extra_max < 1:
        return None
    bottom_extra_size = random.randint(1, bottom_extra_max)

    top_extra = _grow_extension(allowed, shared, set(), top_extra_size)
    if top_extra is None:
        return None
    bottom_extra = _grow_extension(allowed, shared, top_extra, bottom_extra_size)
    if bottom_extra is None:
        return None

    top_mask = shared | top_extra
    bottom_mask = shared | bottom_extra
    if not (top_mask & bottom_mask):
        return None
    if not (top_mask - bottom_mask):
        return None
    if not (bottom_mask - top_mask):
        return None
    if not any(abs(r - ar) + abs(c - ac) == 1 for r, c in top_mask):
        return None
    if not any(abs(r - ar) + abs(c - ac) == 1 for r, c in bottom_mask):
        return None
    return top_mask, bottom_mask


def _translation_candidates(
    half_h: int,
    half_w: int,
    anchor: tuple[int, int],
    bottom_mask: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    ar, ac = anchor
    candidates = []
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) < 2:
                continue
            if not (0 <= ar + dr < half_h and 0 <= ac + dc < half_w):
                continue
            shifted = {(r + dr, c + dc) for r, c in bottom_mask}
            if all(0 <= r < half_h and 0 <= c < half_w for r, c in shifted):
                candidates.append((dr, dc))
    return candidates


def _render_half(
    half_h: int,
    half_w: int,
    bg: int,
    mask: set[tuple[int, int]],
    anchor: tuple[int, int],
) -> np.ndarray:
    half = np.full((half_h, half_w), bg, dtype=int)
    for r, c in mask:
        half[r, c] = GREY
    half[anchor] = RED
    return half


def _expected_output(
    half_h: int,
    half_w: int,
    bg: int,
    top_mask: set[tuple[int, int]],
    bottom_mask: set[tuple[int, int]],
    anchor: tuple[int, int],
) -> np.ndarray:
    out = np.full((half_h, half_w), bg, dtype=int)
    for r, c in top_mask ^ bottom_mask:
        out[r, c] = GREEN
    for r, c in top_mask & bottom_mask:
        out[r, c] = BLUE
    out[anchor] = RED
    return out


def _separator_color(bg: int) -> int:
    pool = [color for color in range(10) if color not in {bg, RED, GREY}]
    return random.choice(pool)


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(6, 12)
    half_w = random.randint(7, 15)
    bg = random.choice(INPUT_BG_CHOICES)
    sep = _separator_color(bg)

    margin_r = 1 if half_h >= 3 else 0
    margin_c = 1 if half_w >= 3 else 0
    anchor = (
        random.randint(margin_r, half_h - 1 - margin_r),
        random.randint(margin_c, half_w - 1 - margin_c),
    )

    masks = _build_connected_masks(half_h, half_w, anchor)
    if masks is None:
        return None
    top_mask, bottom_mask = masks

    if len(top_mask | bottom_mask) > (half_h * half_w) // 3:
        return None
    if len(top_mask & bottom_mask) > len(top_mask | bottom_mask) // 2:
        return None

    translations = _translation_candidates(half_h, half_w, anchor, bottom_mask)
    if not translations:
        return None
    shift_r, shift_c = random.choice(translations)

    shifted_anchor = (anchor[0] + shift_r, anchor[1] + shift_c)
    shifted_bottom_mask = {(r + shift_r, c + shift_c) for r, c in bottom_mask}

    top_half = _render_half(half_h, half_w, bg, top_mask, anchor)
    bottom_half = _render_half(half_h, half_w, bg, shifted_bottom_mask, shifted_anchor)
    separator_row = np.full((1, half_w), sep, dtype=int)
    grid = np.vstack([top_half, separator_row, bottom_half])

    expected = _expected_output(half_h, half_w, bg, top_mask, bottom_mask, anchor)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    color_counts = Counter(map(int, grid.flatten()))
    if color_counts[sep] != half_w:
        return None
    if np.any(np.all(top_half == sep, axis=1)) or np.any(np.all(bottom_half == sep, axis=1)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-alignment overlay input")
