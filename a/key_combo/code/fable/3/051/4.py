from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _color_components(grid, FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        if len(coords) != 2 * (hh + ww) - 4:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if not np.all(patch[border] == FRAME_COLOR):
            continue
        if np.any(patch[~border] == FRAME_COLOR):
            continue
        if not np.all(patch[1:-1, 1:-1] == BG):
            continue
        candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one grey rectangular frame"
    return candidates[0]


def _collect_segments(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    segments: list[tuple[int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, FRAME_COLOR):
            continue
        comps = _color_components(grid, color)
        assert len(comps) == 1, "each segment color must appear in one component"
        coords = comps[0]
        sr0, sc0, sr1, sc1 = _bbox(coords)
        length = sc1 - sc0 + 1
        assert sr0 == sr1, "each object must be a horizontal segment"
        assert len(coords) == length, "segment must be contiguous"
        assert not (r0 <= sr0 <= r1 and c0 <= sc0 and sc1 <= c1), "segments must lie outside the frame"
        segments.append((length, color))

    assert segments, "expected at least one colored segment"
    lengths = [length for length, _color in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be unique"
    return sorted(segments)


def _diagonal_band_colors(segments: list[tuple[int, int]], diag_count: int) -> np.ndarray:
    colors = np.empty(diag_count, dtype=int)
    idx = 0
    filled = 0
    while filled < diag_count:
        length, color = segments[idx]
        end = min(diag_count, filled + length)
        colors[filled:end] = color
        filled = end
        idx = (idx + 1) % len(segments)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered horizontal segments by length and weave their colors into cyclic diagonal bands inside the grey frame.

    Input structure:
    - The grid has a black background, exactly one grey rectangular frame, and a black empty canvas inside that frame.
    - Outside the frame sit one or more monochrome horizontal segments, each in a distinct color and with a distinct length.
    - The segments do not touch the frame and their positions do not encode the final order.

    Rule:
    - Detect the frame and read the lengths and colors of all horizontal segments.
    - Sort the segments from shortest to longest; this sorted order gives the repeating color sequence, and each segment length also gives that color's diagonal-band width.
    - Fill the frame interior by increasing local r+c: color each anti-diagonal with the current color for as many consecutive diagonals as that segment length, then advance to the next color cyclically until the interior is full.

    Key insight:
    - Each segment length is used twice: first to rank the colors, then again to decide how many consecutive anti-diagonals that color occupies inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background"

    frame_r0, frame_c0, frame_r1, frame_c1 = _find_frame_bbox(g)
    segments = _collect_segments(g, (frame_r0, frame_c0, frame_r1, frame_c1))

    inner_h = frame_r1 - frame_r0 - 1
    inner_w = frame_c1 - frame_c0 - 1
    diag_colors = _diagonal_band_colors(segments, inner_h + inner_w - 1)

    out = g.copy()
    for rr in range(inner_h):
        for cc in range(inner_w):
            out[frame_r0 + 1 + rr, frame_c0 + 1 + cc] = int(diag_colors[rr + cc])
    return out


def _sample_spaced_indices(span: int, count: int) -> list[int] | None:
    for _ in range(100):
        picks = sorted(random.sample(range(span), count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_lengths(diag_count: int) -> list[int] | None:
    pool = list(range(2, min(8, diag_count - 1) + 1))
    feasible_counts = [
        count
        for count in range(2, min(4, len(pool)) + 1)
        if sum(range(2, 2 + count)) < diag_count
    ]
    if not feasible_counts:
        return None

    for _ in range(100):
        count = random.choice(feasible_counts)
        lengths = sorted(random.sample(pool, count))
        if sum(lengths) < diag_count:
            return lengths
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _place_segments(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    lengths: list[int],
    colors: list[int],
) -> None:
    zr0, zc0, zr1, zc1 = zone
    rows = _sample_spaced_indices(zr1 - zr0 + 1, len(lengths))
    assert rows is not None, "segment zone must fit all rows"

    pairs = list(zip(lengths, colors))
    random.shuffle(pairs)
    for local_row, (length, color) in zip(rows, pairs):
        row = zr0 + local_row
        start = random.randint(zc0, zc1 - length + 1)
        grid[row, start : start + length] = color


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(5, 9)
    inner_w = random.randint(5, 9)
    diag_count = inner_h + inner_w - 1
    lengths = _sample_lengths(diag_count)
    if lengths is None:
        return None

    colors = random.sample(list(SEGMENT_COLORS), len(lengths))
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    side = random.choice(("top", "bottom", "left", "right"))

    outer_top = random.randint(1, 2)
    outer_bottom = random.randint(1, 2)
    outer_left = random.randint(1, 2)
    outer_right = random.randint(1, 2)
    gap = random.randint(2, 3)
    row_need = 2 * len(lengths) - 1 + random.randint(0, 1)
    max_length = max(lengths)

    if side in ("top", "bottom"):
        strip_h = row_need
        content_w = max(frame_w, max_length) + random.randint(1, 3)
        height = outer_top + strip_h + gap + frame_h + outer_bottom
        width = outer_left + content_w + outer_right
        if height > 30 or width > 30:
            return None
        if side == "top":
            frame_r0 = outer_top + strip_h + gap
            zone = (outer_top, outer_left, outer_top + strip_h - 1, width - outer_right - 1)
        else:
            frame_r0 = outer_top
            zone = (
                outer_top + frame_h + gap,
                outer_left,
                outer_top + frame_h + gap + strip_h - 1,
                width - outer_right - 1,
            )
        frame_c0 = random.randint(outer_left, width - outer_right - frame_w)
    else:
        strip_w = max_length + random.randint(1, 3)
        content_h = max(frame_h, row_need) + random.randint(0, 2)
        height = outer_top + content_h + outer_bottom
        width = outer_left + strip_w + gap + frame_w + outer_right
        if height > 30 or width > 30:
            return None
        frame_r0 = random.randint(outer_top, height - outer_bottom - frame_h)
        if side == "left":
            frame_c0 = outer_left + strip_w + gap
            zone = (outer_top, outer_left, height - outer_bottom - 1, outer_left + strip_w - 1)
        else:
            frame_c0 = outer_left
            zone = (
                outer_top,
                outer_left + frame_w + gap,
                height - outer_bottom - 1,
                outer_left + frame_w + gap + strip_w - 1,
            )

    grid = np.full((height, width), BG, dtype=int)
    _draw_frame(grid, frame_r0, frame_c0, inner_h, inner_w)
    _place_segments(grid, zone, lengths, colors)

    out = main(grid)
    interior = out[frame_r0 + 1 : frame_r0 + 1 + inner_h, frame_c0 + 1 : frame_c0 + 1 + inner_w]
    for color in colors:
        assert np.any(interior == color), "every input color should appear in the woven canvas"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
