from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg


BBox = tuple[int, int, int, int]


def _intervals(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    coords = np.flatnonzero(mask_1d)
    if coords.size == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = int(coords[0])
    prev = start
    for raw in coords[1:]:
        cur = int(raw)
        if cur != prev + 1:
            spans.append((start, prev))
            start = cur
        prev = cur
    spans.append((start, prev))
    return spans


def _border_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _box_inside(shape: tuple[int, int], box: BBox) -> bool:
    height, width = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width


def _ring_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    height, width = shape
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)

    clip_c0 = max(0, c0)
    clip_c1 = min(width - 1, c1)
    if clip_c0 <= clip_c1:
        if 0 <= r0 < height:
            mask[r0, clip_c0 : clip_c1 + 1] = True
        if 0 <= r1 < height:
            mask[r1, clip_c0 : clip_c1 + 1] = True

    clip_r0 = max(0, r0)
    clip_r1 = min(height - 1, r1)
    if clip_r0 <= clip_r1:
        if 0 <= c0 < width:
            mask[clip_r0 : clip_r1 + 1, c0] = True
        if 0 <= c1 < width:
            mask[clip_r0 : clip_r1 + 1, c1] = True
    return mask


def _outside_distance(r: int, c: int, box: BBox) -> int:
    r0, c0, r1, c1 = box
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return max(dr, dc)


def _seed_lengths(grid: np.ndarray, bg: int, r0: int, c0: int, c1: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        length = 0
        r = r0 - 1
        while r >= 0 and int(grid[r, c]) != bg:
            length += 1
            r -= 1
        if length > 0:
            out.append((c, length))
    return out


def _valid_frame_candidate(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    box: BBox,
) -> tuple[int, list[int]] | None:
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    if r0 == 0:
        return None

    border = _border_mask(grid.shape, box)
    if not np.all(grid[border] == frame_color):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    seeds = _seed_lengths(grid, bg, r0, c0, c1)
    if len(seeds) != 1:
        return None
    seed_c, seed_len = seeds[0]

    expected_fg = border.copy()
    expected_fg[r0 - seed_len : r0, seed_c] = True
    if not np.array_equal(grid != bg, expected_fg):
        return None

    palette = [int(grid[r0 - d, seed_c]) for d in range(1, seed_len + 1)]
    return seed_c, palette


def _infer_structure(grid: np.ndarray) -> tuple[int, int, BBox, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    assert len(find_components(g, bg=bg, connectivity=4, monochromatic=False)) == 1, (
        "expected one connected foreground structure"
    )

    candidates: list[tuple[int, BBox, int, list[int]]] = []
    for color in colors_present(g, exclude={bg}):
        rows_by_span: dict[tuple[int, int], list[int]] = {}
        for r in range(g.shape[0]):
            for c0, c1 in _intervals(g[r] == color):
                if c1 - c0 + 1 >= 3:
                    rows_by_span.setdefault((c0, c1), []).append(r)

        for (c0, c1), rows in rows_by_span.items():
            if len(rows) < 2:
                continue
            for i, r0 in enumerate(rows):
                for r1 in rows[i + 1 :]:
                    box = (int(r0), int(c0), int(r1), int(c1))
                    info = _valid_frame_candidate(g, bg, int(color), box)
                    if info is None:
                        continue
                    seed_c, palette = info
                    candidates.append((int(color), box, seed_c, palette))

    assert len(candidates) == 1, "expected a unique hollow rectangle and seed column"
    frame_color, box, seed_c, palette = candidates[0]
    assert len(palette) >= 1, "expected a non-empty seed column"
    return bg, frame_color, box, seed_c, palette


def _render_output(grid: np.ndarray, box: BBox, palette: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = box
    period = len(palette)
    assert period >= 1

    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            d = _outside_distance(r, c, box)
            out[r, c] = int(palette[(d - 1) % period])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the clipped family of concentric rectangle rings from a hollow rectangle and its upward radial seed column.

    Input structure:
    - One dominant background contains a one-cell-thick hollow rectangle frame.
    - A single non-background vertical seed column rises from one top-edge column of the frame.
    - Its d-th seed pixel lies exactly Chebyshev distance d outside the frame, and there are no other foreground cells.

    Rule:
    - Keep the original frame and its interior unchanged.
    - For every outside cell, read its Chebyshev distance d from the frame's bbox and color it with the seed column's d-th ring color, cycling by seed-column length.
    - This is equivalent to drawing every outward concentric rectangle ring, clipped by the canvas, until the next ring is fully outside the grid.

    Key insight:
    - The short color column is not an object to move; it is a radial sample of an infinite concentric ring family, so every visible cell outside the frame is determined by distance-from-frame and the cyclic seed order.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_color, box, _seed_c, palette = _infer_structure(g)
    assert int(np.count_nonzero(g == bg)) > 0, "expected background cells"
    return _render_output(g, box, palette)


GEN_ATTEMPTS = 400
SEED_LENGTH_CHOICES = (2, 3, 4)
FRAME_H_RANGE = (3, 8)
FRAME_W_RANGE = (3, 9)
MARGIN_MAX_EXTRA = 7


def _max_visible_distance(shape: tuple[int, int], box: BBox) -> int:
    height, width = shape
    r0, c0, r1, c1 = box
    return max(r0, c0, height - 1 - r1, width - 1 - c1)


def _visible_clipped_ring_exists(shape: tuple[int, int], box: BBox) -> bool:
    max_d = _max_visible_distance(shape, box)
    r0, c0, r1, c1 = box
    for d in range(1, max_d + 1):
        expanded = (r0 - d, c0 - d, r1 + d, c1 + d)
        if _ring_mask(shape, expanded).any() and not _box_inside(shape, expanded):
            return True
    return False


def _render_input(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    box: BBox,
    seed_c: int,
    palette: list[int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    grid[_border_mask(shape, box)] = frame_color
    r0 = box[0]
    for d, color in enumerate(palette, start=1):
        grid[r0 - d, seed_c] = int(color)
    return grid


def _pick_colors(seed_len: int) -> tuple[int, int, list[int]]:
    all_colors = list(Color.ALL_COLORS)
    bg = random.choice(all_colors)
    remaining = [color for color in all_colors if color != bg]
    frame_color = random.choice(remaining)
    palette_pool = [color for color in remaining if color != frame_color]
    palette = random.sample(palette_pool, seed_len)
    return int(bg), int(frame_color), [int(color) for color in palette]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    seed_len = random.choice(SEED_LENGTH_CHOICES)
    frame_h = random.randint(*FRAME_H_RANGE)
    frame_w = random.randint(*FRAME_W_RANGE)

    top = random.randint(seed_len, seed_len + MARGIN_MAX_EXTRA)
    bottom = random.randint(0, seed_len + MARGIN_MAX_EXTRA)
    left = random.randint(0, seed_len + MARGIN_MAX_EXTRA)
    right = random.randint(0, seed_len + MARGIN_MAX_EXTRA)

    height = top + frame_h + bottom
    width = left + frame_w + right
    if max(height, width) > 30:
        raise AssertionError("generated shape exceeds ARC bounds")

    box = (top, left, top + frame_h - 1, left + frame_w - 1)
    margins = (top, left, bottom, right)
    max_visible = max(margins)
    if max_visible <= seed_len:
        raise AssertionError("need visible rings beyond the seed column")
    if min(margins) == max_visible:
        raise AssertionError("need at least one clipped visible ring")

    bg, frame_color, palette = _pick_colors(seed_len)
    seed_c = random.randint(box[1], box[3])

    inp = _render_input((height, width), bg, frame_color, box, seed_c, palette)
    out = _render_output(inp, box, palette)
    return inp, out


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, frame_color, box, seed_c, palette = _infer_structure(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return False
    if len(set(palette)) != len(palette) or len(palette) < 2:
        return False
    if frame_color in palette:
        return False
    if len(find_components(inp, bg=bg, connectivity=4, monochromatic=False)) != 1:
        return False

    r0, c0, r1, c1 = box
    max_visible = _max_visible_distance(inp.shape, box)
    if max_visible <= len(palette):
        return False
    if not _visible_clipped_ring_exists(inp.shape, box):
        return False
    if not any(_ring_mask(inp.shape, (r0 - d, c0 - d, r1 + d, c1 + d)).any() for d in range(len(palette) + 1, max_visible + 1)):
        return False

    truncated = inp.copy()
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            d = _outside_distance(r, c, box)
            if d <= len(palette):
                truncated[r, c] = palette[d - 1]
    if np.array_equal(truncated, expected):
        return False

    outside_colors = {
        int(expected[r, c])
        for r in range(expected.shape[0])
        for c in range(expected.shape[1])
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    }
    if len(outside_colors) < 2:
        return False

    seed_pixels = [int(inp[r0 - d, seed_c]) for d in range(1, len(palette) + 1)]
    return seed_pixels == palette


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, out = _generate_case()
        except AssertionError:
            continue
        if _validate_case(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid single-ring-generator sample")
