from __future__ import annotations

import random

import numpy as np


BLACK = 0
DEFAULT_GREY = 5
FRAME_SIZES = (5, 7)
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _palette_starts(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    starts = {(0, 0), (0, w - 4), (h - 2, 0), (h - 2, w - 4)}
    return [(r, c) for r, c in starts if 0 <= r <= h - 2 and 0 <= c <= w - 4]


def _find_palette(grid: np.ndarray) -> tuple[list[int], int, tuple[int, int]]:
    candidates: list[tuple[list[int], int, tuple[int, int]]] = []
    for r0, c0 in _palette_starts(grid.shape):
        block = grid[r0 : r0 + 2, c0 : c0 + 4]
        top = [int(v) for v in block[0].tolist()]
        bottom = [int(v) for v in block[1].tolist()]
        if len(set(top)) != 4 or BLACK in top:
            continue
        if len(set(bottom)) != 1 or int(bottom[0]) == BLACK:
            continue
        grey = int(bottom[0])
        if grey in top:
            continue
        candidates.append((top, grey, (int(r0), int(c0))))
    assert len(candidates) == 1, "expected one 2x4 corner palette"
    return candidates[0]


def _border_is_grey(block: np.ndarray, grey: int) -> bool:
    return bool(
        np.all(block[0, :] == grey)
        and np.all(block[-1, :] == grey)
        and np.all(block[:, 0] == grey)
        and np.all(block[:, -1] == grey)
    )


def _quadrant_slices(inner_size: int) -> list[tuple[slice, slice]]:
    k = inner_size // 2
    return [
        (slice(0, k), slice(0, k)),
        (slice(0, k), slice(k + 1, inner_size)),
        (slice(k + 1, inner_size), slice(0, k)),
        (slice(k + 1, inner_size), slice(k + 1, inner_size)),
    ]


def _normalize_quadrant(mask: np.ndarray, quadrant_index: int) -> np.ndarray:
    if quadrant_index == 0:
        return mask.copy()
    if quadrant_index == 1:
        return np.fliplr(mask)
    if quadrant_index == 2:
        return np.flipud(mask)
    return np.flipud(np.fliplr(mask))


def _denormalize_quadrant(mask: np.ndarray, quadrant_index: int) -> np.ndarray:
    if quadrant_index == 0:
        return mask.copy()
    if quadrant_index == 1:
        return np.fliplr(mask)
    if quadrant_index == 2:
        return np.flipud(mask)
    return np.flipud(np.fliplr(mask))


def _find_frames(grid: np.ndarray, grey: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    frames: list[tuple[int, int, int, int]] = []
    for size in FRAME_SIZES:
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                block = grid[r0 : r0 + size, c0 : c0 + size]
                if not _border_is_grey(block, grey):
                    continue
                inner = block[1:-1, 1:-1]
                colors = {int(v) for v in np.unique(inner)}
                if grey in colors:
                    continue
                payload = [color for color in colors if color != BLACK]
                if len(payload) != 1:
                    continue
                frames.append((int(r0), int(c0), int(size), int(payload[0])))
    assert frames, "expected at least one grey frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each missing frame quadrant with the palette color encoded by its symmetry-defect direction.

    Input structure:
    - The background is black.
    - Several grey square frames of size 5x5 or 7x7 are scattered on the grid.
    - Each frame contains a one-color pattern that is horizontally and vertically symmetric except that one quadrant is missing and replaced by black.
    - One grid corner contains a 2x4 palette: four colors on the top row and a grey base row beneath them.

    Rule:
    - Locate the corner palette and read its four top-row colors in left-to-right order as the labels for missing quadrants: upper-left, upper-right, lower-left, lower-right.
    - For each grey frame, compare its four interior quadrants after mirroring them into a common orientation.
    - The three matching quadrants reveal the intended motif, and the mismatching all-black quadrant is the missing one.
    - Reconstruct that quadrant by reflecting the shared motif back into place and draw only those restored pixels in the palette color assigned to that quadrant.

    Key insight:
    - The missing shape comes from symmetry, but the fill color comes from the palette's positional code rather than from the frame's original pattern color.
    """
    g = np.asarray(grid, dtype=int)
    palette, grey, _palette_pos = _find_palette(g)
    out = g.copy()

    for r0, c0, size, color in _find_frames(g, grey):
        inner = g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        occupancy = inner == color
        inner_size = int(inner.shape[0])
        quads = []
        for idx, (rs, cs) in enumerate(_quadrant_slices(inner_size)):
            quad = occupancy[rs, cs]
            quads.append(_normalize_quadrant(quad, idx))

        best_template = None
        best_count = -1
        for candidate in quads:
            count = sum(int(np.array_equal(candidate, other)) for other in quads)
            if count > best_count:
                best_template = candidate
                best_count = count

        assert best_template is not None and best_count == 3, "expected one missing quadrant"
        assert bool(np.any(best_template)), "the symmetric quadrant motif must be non-empty"
        missing = [idx for idx, quad in enumerate(quads) if not np.array_equal(quad, best_template)]
        assert len(missing) == 1, "missing quadrant must be unique"
        missing_idx = int(missing[0])

        restored = _denormalize_quadrant(best_template, missing_idx)
        rs, cs = _quadrant_slices(inner_size)[missing_idx]
        target = out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        current = target[rs, cs]
        assert bool(np.all(current[restored] == BLACK)), "restored pixels must replace black background"
        current[restored] = int(palette[missing_idx])

    return out


def _reserve(mask: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c0 + width + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _can_place(mask: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + size + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _random_base_mask(k: int) -> np.ndarray:
    if k == 1:
        return np.ones((1, 1), dtype=bool)

    while True:
        bits = np.zeros((k, k), dtype=bool)
        for r in range(k):
            for c in range(k):
                bits[r, c] = random.random() < 0.6
        if bool(bits.any()):
            return bits


def _make_full_pattern(inner_size: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    pattern = np.zeros((inner_size, inner_size), dtype=int)
    k = inner_size // 2
    base = _random_base_mask(k)

    quadrants = _quadrant_slices(inner_size)
    pattern[quadrants[0]][base] = color
    pattern[quadrants[1]][np.fliplr(base)] = color
    pattern[quadrants[2]][np.flipud(base)] = color
    pattern[quadrants[3]][np.flipud(np.fliplr(base))] = color

    center = inner_size // 2
    if random.random() < 0.5:
        pattern[center, center] = color

    axis_top = [random.random() < 0.5 for _ in range(k)]
    axis_left = [random.random() < 0.5 for _ in range(k)]
    for idx, flag in enumerate(axis_top):
        if flag:
            pattern[idx, center] = color
            pattern[inner_size - 1 - idx, center] = color
    for idx, flag in enumerate(axis_left):
        if flag:
            pattern[center, idx] = color
            pattern[center, inner_size - 1 - idx] = color

    return pattern, base


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, grey: int, inner: np.ndarray) -> None:
    grid[r0 : r0 + size, c0 : c0 + size] = BLACK
    grid[r0, c0 : c0 + size] = grey
    grid[r0 + size - 1, c0 : c0 + size] = grey
    grid[r0 : r0 + size, c0] = grey
    grid[r0 : r0 + size, c0 + size - 1] = grey
    grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = inner


def _palette_anchor(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 4
    if corner == "bl":
        return h - 2, 0
    return h - 2, w - 4


def _place_palette(grid: np.ndarray, colors: list[int], grey: int, corner: str) -> tuple[int, int]:
    r0, c0 = _palette_anchor(grid.shape, corner)
    grid[r0, c0 : c0 + 4] = np.array(colors, dtype=int)
    grid[r0 + 1, c0 : c0 + 4] = grey
    return r0, c0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grey = DEFAULT_GREY
    palette_colors = random.sample(list(PALETTE_POOL), 4)
    frame_colors = [color for color in PALETTE_POOL if color not in palette_colors]
    frame_count = random.randint(1, 4)
    corner = random.choice(["tl", "tr", "bl", "br"])

    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pr0, pc0 = _place_palette(inp, palette_colors, grey, corner)
    _place_palette(out, palette_colors, grey, corner)
    _reserve(occupied, pr0, pc0, 2, 4, pad=1)

    for _ in range(frame_count):
        placed = False
        for _attempt in range(200):
            size = random.choice(FRAME_SIZES)
            r0 = random.randint(0, h - size)
            c0 = random.randint(0, w - size)
            if not _can_place(occupied, r0, c0, size):
                continue

            color = random.choice(frame_colors)
            inner_size = size - 2
            full_pattern, base = _make_full_pattern(inner_size, color)
            missing_idx = random.randrange(4)
            input_inner = full_pattern.copy()
            rs, cs = _quadrant_slices(inner_size)[missing_idx]
            input_inner[rs, cs] = BLACK

            solved_inner = input_inner.copy()
            restored = _denormalize_quadrant(base, missing_idx)
            solved_inner[rs, cs][restored] = palette_colors[missing_idx]

            _draw_frame(inp, r0, c0, size, grey, input_inner)
            _draw_frame(out, r0, c0, size, grey, solved_inner)
            _reserve(occupied, r0, c0, size, size, pad=1)
            placed = True
            break
        if not placed:
            return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
