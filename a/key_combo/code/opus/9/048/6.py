from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WAVE_COLOR = 5
PALETTE_CHOICES = [1, 2, 3, 4, 6, 7, 8, 9]
LAYER_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _column_interval(mask: np.ndarray, c: int) -> tuple[int, int]:
    rows = np.flatnonzero(mask[:, c])
    assert rows.size > 0, "expected every wave column to be non-empty"
    r0, r1 = int(rows[0]), int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1)), "expected contiguous vertical spans"
    return r0, r1


def _parse(grid: np.ndarray) -> tuple[int, list[int], int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _h, w = g.shape
    bg = _get_bg(g)

    c = 0
    while c < w and np.all(g[:, c] == bg):
        c += 1
    palette_start = c
    palette_cols: list[int] = []
    palette: list[int] = []
    while c < w:
        col = g[:, c]
        color = int(col[0])
        if color != bg and np.all(col == color):
            palette_cols.append(c)
            palette.append(color)
            c += 1
        else:
            break

    assert 3 <= len(palette_cols) <= 5, "expected 3-5 adjacent palette columns"
    assert palette_cols == list(range(palette_start, palette_start + len(palette_cols)))
    assert len(set(palette)) == len(palette), "palette colors must be distinct"

    mask = g != bg
    mask[:, palette_cols] = False
    assert bool(mask.any()), "expected a target wave besides the palette"
    other_colors = {int(v) for v in g[mask]}
    assert len(other_colors) == 1, "expected one non-palette drawing color"

    r0, c0, r1, c1 = _bbox(mask)
    wave = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert c1 - c0 + 1 == len(palette), "palette width must match wave width"
    assert bool(np.all(wave.any(axis=0))), "every wave column must be populated"

    intervals = [_column_interval(wave, j) for j in range(wave.shape[1])]
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert max(a0, b0) <= min(a1, b1), "adjacent wave columns must overlap by at least one row"

    return bg, palette, r0, c0, wave


def _render_from_wave(
    bg: int,
    width: int,
    top: int,
    left: int,
    wave: np.ndarray,
    palette: list[int],
    *,
    flip_geometry: bool,
    reverse_on_flip: bool,
    overlap_step: int,
) -> np.ndarray:
    wave_h = wave.shape[0]
    out_h = top + overlap_step * (LAYER_COUNT - 1) + wave_h
    out = np.full((out_h, width), bg, dtype=int)
    flipped = np.flipud(wave)

    for layer in range(LAYER_COUNT):
        is_flipped = layer % 2 == 1
        patch = flipped if (flip_geometry and is_flipped) else wave
        colors = palette[::-1] if (reverse_on_flip and is_flipped) else palette
        r_base = top + layer * overlap_step
        for j, color in enumerate(colors):
            rows = np.flatnonzero(patch[:, j])
            if rows.size:
                out[r_base + rows, left + j] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an alternating vertically flipped wave stack using a palette whose read direction flips with the geometry.

    Input structure:
    - The background is black, with 3-5 adjacent full-height palette columns on the left and one grey wave on the right.
    - The palette columns are distinct single-color columns read left-to-right.
    - The wave occupies one tight bounding box; each wave column contains one contiguous vertical span, and neighboring columns overlap in at least one row.

    Rule:
    - Read the palette colors from left to right and extract the grey wave's bounding box as the seed shape.
    - Keep the wave's original top row and column placement, but expand the canvas vertically to hold four copies with one-row overlaps.
    - Stack four layers: original, vertically flipped, original, vertically flipped, using step = wave_height - 1.
    - Recolor each layer column-by-column: even layers use the palette in forward order, odd layers use the palette in reverse order.
    - Remove the original palette columns so they become background; later layers overwrite earlier ones on overlap rows.

    Key insight:
    - The odd layers couple two changes at once: the wave is flipped vertically, and the palette is read in the opposite direction, so geometry and color order must be inferred together.
    """
    bg, palette, top, left, wave = _parse(grid)
    return _render_from_wave(
        bg,
        grid.shape[1],
        top,
        left,
        wave,
        palette,
        flip_geometry=True,
        reverse_on_flip=True,
        overlap_step=wave.shape[0] - 1,
    )


def _all_intervals(height: int) -> list[tuple[int, int]]:
    return [
        (r0, r1)
        for r0 in range(height)
        for r1 in range(r0 + 1, height)
        if r1 - r0 + 1 >= 2
    ]


def _intervals_to_mask(intervals: list[tuple[int, int]], height: int) -> np.ndarray:
    mask = np.zeros((height, len(intervals)), dtype=bool)
    for c, (r0, r1) in enumerate(intervals):
        mask[r0 : r1 + 1, c] = True
    return mask


def _generate_wave(height: int, width: int) -> np.ndarray | None:
    intervals_all = _all_intervals(height)

    for _ in range(200):
        intervals = [random.choice(intervals_all)]
        for _c in range(1, width):
            prev0, prev1 = intervals[-1]
            candidates = [
                cur
                for cur in intervals_all
                if max(prev0, cur[0]) <= min(prev1, cur[1])
                and abs(cur[0] - prev0) <= 2
                and abs(cur[1] - prev1) <= 2
            ]
            if not candidates:
                break
            intervals.append(random.choice(candidates))
        if len(intervals) != width:
            continue

        starts = [r0 for r0, _ in intervals]
        ends = [r1 for _, r1 in intervals]
        if min(starts) != 0 or max(ends) != height - 1:
            continue
        if len(set(intervals)) == 1:
            continue
        if len(set(starts)) == 1 and len(set(ends)) == 1:
            continue
        if not any(r0 == 0 and j != width - 1 - j for j, (r0, _r1) in enumerate(intervals)):
            continue
        if not any(r1 == height - 1 and j != width - 1 - j for j, (_r0, r1) in enumerate(intervals)):
            continue

        wave = _intervals_to_mask(intervals, height)
        if np.array_equal(wave, np.flipud(wave)):
            continue
        return wave

    return None


def _check_overlap_conflicts(out: np.ndarray, wave: np.ndarray, palette: list[int], top: int, left: int) -> bool:
    flipped = np.flipud(wave)
    conflicts = 0
    for layer in range(LAYER_COUNT - 1):
        lower = wave if layer % 2 == 0 else flipped
        upper = flipped if layer % 2 == 0 else wave
        lower_colors = palette if layer % 2 == 0 else palette[::-1]
        upper_colors = palette[::-1] if layer % 2 == 0 else palette
        overlap_row = top + (layer + 1) * (wave.shape[0] - 1)
        for j in range(wave.shape[1]):
            if lower[-1, j] and upper[0, j] and lower_colors[j] != upper_colors[j]:
                if int(out[overlap_row, left + j]) != int(upper_colors[j]):
                    return False
                conflicts += 1
    return conflicts >= 1


def _is_valid_example(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, palette, top, left, wave = _parse(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    wrong_direction = _render_from_wave(
        bg,
        inp.shape[1],
        top,
        left,
        wave,
        palette,
        flip_geometry=True,
        reverse_on_flip=False,
        overlap_step=wave.shape[0] - 1,
    )
    wrong_geometry = _render_from_wave(
        bg,
        inp.shape[1],
        top,
        left,
        wave,
        palette,
        flip_geometry=False,
        reverse_on_flip=True,
        overlap_step=wave.shape[0] - 1,
    )
    no_overlap = _render_from_wave(
        bg,
        inp.shape[1],
        top,
        left,
        wave,
        palette,
        flip_geometry=True,
        reverse_on_flip=True,
        overlap_step=wave.shape[0],
    )
    if np.array_equal(expected, wrong_direction):
        return False
    if np.array_equal(expected, wrong_geometry):
        return False
    if np.array_equal(expected, no_overlap):
        return False
    if not _check_overlap_conflicts(expected, wave, palette, top, left):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    palette_count = random.randint(3, 5)
    wave_h = random.randint(4, 6)
    wave = _generate_wave(wave_h, palette_count)
    if wave is None:
        return None

    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    gap = random.randint(1, 3)
    right_pad = random.randint(0, 4)
    palette = random.sample(PALETTE_CHOICES, palette_count)

    in_h = top_pad + wave_h + bottom_pad
    in_w = left_pad + palette_count + gap + palette_count + right_pad
    if max(in_h, in_w) > 30:
        return None

    wave_left = left_pad + palette_count + gap
    inp = np.full((in_h, in_w), BG, dtype=int)
    for i, color in enumerate(palette):
        inp[:, left_pad + i] = int(color)
    for j in range(palette_count):
        rows = np.flatnonzero(wave[:, j])
        inp[top_pad + rows, wave_left + j] = WAVE_COLOR

    expected = _render_from_wave(
        BG,
        in_w,
        top_pad,
        wave_left,
        wave,
        palette,
        flip_geometry=True,
        reverse_on_flip=True,
        overlap_step=wave_h - 1,
    )
    if max(expected.shape) > 30:
        return None
    if not _is_valid_example(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wave-palette stacking example")
