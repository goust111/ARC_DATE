from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_RETRIES = 400

# Every pattern keeps the four corners as key-color cells.
PATTERN_LIBRARY = (
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 1, 0), (0, 1, 0)),
    ((0, 0, 0), (1, 1, 1), (0, 0, 0)),
    ((0, 1, 0), (1, 1, 1), (0, 0, 0)),
    ((0, 0, 0), (1, 1, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 1, 0), (0, 0, 0)),
    ((0, 1, 0), (0, 1, 1), (0, 0, 0)),
    ((0, 0, 0), (1, 1, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 1, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 0, 1), (0, 1, 0)),
    ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 0)),
)


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for r in range(int(grid.shape[0])):
        row = grid[r]
        if int(row[0]) != int(bg) and bool((row == row[0]).all()):
            candidates.append(int(r))
    assert candidates, "expected a solid separator row"
    sep_row = min(r for r in candidates if r >= 3)
    return int(sep_row)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = int(i)
        elif not flag and start is not None:
            spans.append((int(start), int(i)))
            start = None
    if start is not None:
        spans.append((int(start), int(mask.size)))
    return spans


def _extract_mapping(grid: np.ndarray, bg: int) -> tuple[int, dict[int, np.ndarray]]:
    sep_row = _find_separator_row(grid, bg)
    template = grid[sep_row - 3 : sep_row, :]
    fg_cols = np.any(template != int(bg), axis=0)
    spans = _segments(fg_cols)
    assert spans, "expected at least one template block"

    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in spans:
        assert int(c1 - c0) == 3, "template blocks must be 3 columns wide"
        block = template[:, c0:c1]
        corners = [int(block[0, 0]), int(block[0, 2]), int(block[2, 0]), int(block[2, 2])]
        key = int(corners[0])
        assert key != int(bg) and all(v == key for v in corners), "corners must hold the key color"

        value_mask = block != int(key)
        assert bool(value_mask.any()), "template pattern must contain at least one value pixel"
        value_colors = {int(v) for v in block[value_mask].tolist()}
        assert len(value_colors) == 1, "template block must use exactly one value color"
        value = int(next(iter(value_colors)))
        assert value != int(bg), "value color must be visible"
        assert key not in mapping, "key colors must be unique"

        stamp = np.full((3, 3), int(bg), dtype=int)
        stamp[value_mask] = int(value)
        mapping[int(key)] = stamp

    return int(sep_row), mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each data pixel using a 3x3 template legend above the separator row.

    Input structure:
    - A 3-row template area sits immediately above a full-width separator row.
    - The template area contains several 3x3 blocks separated by background.
      Within each block, the corner color is the lookup key and the other color
      marks the value-pattern cells.
    - Below the separator, isolated non-background pixels use the template key
      colors.

    Rule:
    - Build a key -> 3x3 stamp lookup from the template row.
    - For each non-background data pixel, stamp the mapped 3x3 pattern centered
      on that pixel. Template-key cells become background in the stamped result.
    - Later pixels overwrite earlier ones; the template area and separator stay
      unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_row, mapping = _extract_mapping(g, bg)

    out = g.copy()
    for r in range(int(sep_row + 1), int(g.shape[0])):
        for c in range(int(g.shape[1])):
            key = int(g[r, c])
            if key == int(bg):
                continue
            assert key in mapping, "every data pixel must match a template key"
            stamp = mapping[int(key)]
            for dr in range(-1, 2):
                rr = int(r + dr)
                if not (sep_row + 1 <= rr < g.shape[0]):
                    continue
                for dc in range(-1, 2):
                    cc = int(c + dc)
                    if 0 <= cc < g.shape[1]:
                        out[rr, cc] = int(stamp[dr + 1, dc + 1])
    return out


def _stamp_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) <= 2 and abs(int(a[1]) - int(b[1])) <= 2


def _choose_centers(
    height: int,
    width: int,
    count: int,
    force_overlap: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(5, int(height - 1)) for c in range(1, int(width - 1))]
    rng.shuffle(candidates)
    if len(candidates) < count:
        return None

    if not force_overlap:
        centers: list[tuple[int, int]] = []
        for pos in candidates:
            if all(not _stamp_overlap(pos, other) for other in centers):
                centers.append(pos)
                if len(centers) == count:
                    return centers
        return None

    for upper in candidates:
        nearby = [
            lower
            for lower in candidates
            if lower != upper and int(lower[0]) > int(upper[0]) and _stamp_overlap(upper, lower)
        ]
        rng.shuffle(nearby)
        for lower in nearby:
            centers = [upper, lower]
            for pos in candidates:
                if pos == upper or pos == lower:
                    continue
                if all(not _stamp_overlap(pos, other) for other in centers):
                    centers.append(pos)
                    if len(centers) == count:
                        return centers
    return None


def _pattern_array(pattern: tuple[tuple[int, ...], ...]) -> np.ndarray:
    arr = np.array(pattern, dtype=bool)
    assert arr.shape == (3, 3)
    assert not bool(arr[0, 0]) and not bool(arr[0, 2]) and not bool(arr[2, 0]) and not bool(arr[2, 2])
    assert bool(arr.any())
    return arr


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    num_templates = rng.randint(2, 4)
    chosen_colors = rng.sample(PALETTE, 2 * num_templates)
    key_colors = chosen_colors[:num_templates]
    value_colors = chosen_colors[num_templates:]
    patterns = [_pattern_array(p) for p in rng.sample(PATTERN_LIBRARY, num_templates)]

    required_width = 3 * num_templates + (num_templates - 1)
    width = rng.randint(required_width + 1, required_width + 8)
    height = rng.randint(12, 19)
    left_margin = rng.randint(0, width - required_width)

    grid = np.full((height, width), BG, dtype=int)
    grid[3, :] = SEP

    for idx, (key, value, mask) in enumerate(zip(key_colors, value_colors, patterns, strict=True)):
        c0 = int(left_margin + 4 * idx)
        block = np.full((3, 3), int(key), dtype=int)
        block[mask] = int(value)
        grid[0:3, c0 : c0 + 3] = block

    min_pixels = max(4, num_templates)
    max_pixels = min(8, (height - 6) * max(1, width - 2))
    if min_pixels > max_pixels:
        return None
    pixel_count = rng.randint(min_pixels, max_pixels)
    force_overlap = bool(rng.randint(0, 1))
    centers = _choose_centers(height, width, pixel_count, force_overlap, rng)
    if centers is None:
        return None

    data_keys = key_colors.copy()
    while len(data_keys) < pixel_count:
        data_keys.append(rng.choice(key_colors))
    rng.shuffle(data_keys)

    for (r, c), key in zip(centers, data_keys, strict=True):
        grid[r, c] = int(key)

    out = main(grid)
    if bool(np.array_equal(out, grid)):
        return None
    if not bool(np.array_equal(out[:4, :], grid[:4, :])):
        return None
    if {int(grid[r, c]) for r, c in centers} != set(map(int, key_colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
