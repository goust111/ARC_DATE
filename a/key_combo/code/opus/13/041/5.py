from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
CORNER_PALETTE = [4, 6, 7, 8]
PATTERN_PALETTE = [1, 2, 3, 5, 9]
GRID_SIDE_RANGE = (5, 10)
SUPPORT_AREA_RANGE = (7, 16)
MAX_GENERATION_ATTEMPTS = 400
FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _corner_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = map(int, grid.shape)
    return (
        int(grid[0, 0]),
        int(grid[0, w - 1]),
        int(grid[h - 1, 0]),
        int(grid[h - 1, w - 1]),
    )


def _clear_corners(grid: np.ndarray) -> np.ndarray:
    cleaned = np.array(grid, dtype=int, copy=True)
    h, w = map(int, cleaned.shape)
    cleaned[0, 0] = BACKGROUND
    cleaned[0, w - 1] = BACKGROUND
    cleaned[h - 1, 0] = BACKGROUND
    cleaned[h - 1, w - 1] = BACKGROUND
    return cleaned


def _dominant_color(pattern: np.ndarray) -> int:
    counts = Counter(int(v) for v in pattern[pattern != BACKGROUND])
    assert counts, "expected at least one non-background pattern color"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _mirror_tile(pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        pattern,
        np.fliplr(pattern),
        np.flipud(pattern),
        np.flipud(np.fliplr(pattern)),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-tile the source pattern into four quadrants and recolor the dominant pattern color per corner legend.

    Input structure:
    - The grid has black background with four distinct colored corner pixels.
    - Aside from those corner markers, the remaining cells form a multicolor source
      pattern using 2-3 non-background colors.
    - One of those non-background pattern colors appears strictly more often than
      any other and is therefore the shared replaceable color.

    Rule:
    - Read the four corner pixels as the legend colors for the upper-left,
      upper-right, lower-left, and lower-right output quadrants.
    - Replace the four corner pixels with black to obtain the source pattern P.
    - Identify the unique most frequent non-background color in P.
    - Build a 2H x 2W output by placing P, its left-right mirror, its top-bottom
      mirror, and its 180-degree rotation into the four quadrants.
    - In each quadrant, recolor every pixel of the replaceable color to that
      quadrant's corner legend color; keep all other colors unchanged.

    Key insight:
    - The corners do not belong to the mirrored motif; they are a quadrant legend
      that only changes the dominant pattern color after the geometric mirror tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "grid must have four corners"

    legends = _corner_colors(g)
    assert len(set(legends)) == 4, "expected four distinct corner colors"

    pattern = _clear_corners(g)
    replaceable = _dominant_color(pattern)

    quadrants = list(_mirror_tile(pattern))
    out = np.zeros((2 * h, 2 * w), dtype=int)
    placements = [
        (0, 0),
        (0, w),
        (h, 0),
        (h, w),
    ]
    for (r0, c0), quad, legend in zip(placements, quadrants, legends):
        block = quad.copy()
        block[block == replaceable] = int(legend)
        out[r0 : r0 + h, c0 : c0 + w] = block
    return out


def _non_corner_cells(h: int, w: int) -> list[tuple[int, int]]:
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return [(r, c) for r in range(h) for c in range(w) if (r, c) not in corners]


def _support_mask(h: int, w: int, area: int) -> np.ndarray | None:
    cells = _non_corner_cells(h, w)
    if len(cells) < area:
        return None

    seed_pool = [(r, c) for r, c in cells if 0 < r < h - 1 and 0 < c < w - 1]
    if not seed_pool:
        seed_pool = cells
    seed = random.choice(seed_pool)

    chosen = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in FOUR_NEIGHBORS
        if (seed[0] + dr, seed[1] + dc) in cells
    }

    while len(chosen) < area:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        chosen.add((r, c))
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 3 or len(cols) < 3:
        return None
    return mask


def _paint_pattern(mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    support = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    support_area = len(support)
    color_count = random.choice([2, 3])
    colors = random.sample(PATTERN_PALETTE, color_count)
    dominant = colors[0]
    accents = colors[1:]

    max_accent_total = min(5, support_area - 3)
    if max_accent_total < len(accents):
        return None

    accent_total = random.randint(len(accents), max_accent_total)
    accent_counts = [1] * len(accents)
    for _ in range(accent_total - len(accents)):
        accent_counts[random.randrange(len(accents))] += 1

    dominant_count = support_area - sum(accent_counts)
    if dominant_count <= max(accent_counts):
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in support:
        grid[r, c] = dominant

    random.shuffle(support)
    idx = 0
    for accent, count in zip(accents, accent_counts):
        for r, c in support[idx : idx + count]:
            grid[r, c] = int(accent)
        idx += count

    if np.array_equal(grid, np.fliplr(grid)) or np.array_equal(grid, np.flipud(grid)):
        return None
    if len(set(int(v) for v in grid[grid != BACKGROUND])) != color_count:
        return None
    return grid


def _recolor_signature_changes(pattern: np.ndarray, legends: tuple[int, int, int, int]) -> bool:
    replaceable = _dominant_color(pattern)
    plain = np.block(
        [
            [pattern, np.fliplr(pattern)],
            [np.flipud(pattern), np.flipud(np.fliplr(pattern))],
        ]
    )
    colored = main(_embed_with_corners(pattern, legends))
    if np.array_equal(plain, colored):
        return False

    for i in range(4):
        changed = list(legends)
        changed[i], changed[(i + 1) % 4] = changed[(i + 1) % 4], changed[i]
        if np.array_equal(colored, main(_embed_with_corners(pattern, tuple(changed)))):
            return False

    if int(np.count_nonzero(pattern == replaceable)) == 0:
        return False
    return True


def _embed_with_corners(pattern: np.ndarray, legends: tuple[int, int, int, int]) -> np.ndarray:
    h, w = map(int, pattern.shape)
    grid = np.array(pattern, dtype=int, copy=True)
    grid[0, 0] = int(legends[0])
    grid[0, w - 1] = int(legends[1])
    grid[h - 1, 0] = int(legends[2])
    grid[h - 1, w - 1] = int(legends[3])
    return grid


def _validate_example(inp: np.ndarray) -> bool:
    h, w = map(int, inp.shape)
    if max(h, w) > 15:
        return False

    legends = _corner_colors(inp)
    if sorted(legends) != sorted(CORNER_PALETTE):
        return False

    pattern = _clear_corners(inp)
    colors = {int(v) for v in pattern[pattern != BACKGROUND]}
    if not (2 <= len(colors) <= 3):
        return False
    if colors & set(CORNER_PALETTE):
        return False

    replaceable = _dominant_color(pattern)
    counts = Counter(int(v) for v in pattern[pattern != BACKGROUND])
    if counts[replaceable] < 3:
        return False
    if np.array_equal(pattern, np.fliplr(pattern)) or np.array_equal(pattern, np.flipud(pattern)):
        return False

    out = main(inp)
    if out.shape != (2 * h, 2 * w):
        return False
    if max(out.shape) > 30:
        return False
    if not _recolor_signature_changes(pattern, legends):
        return False

    quadrants = [
        out[:h, :w],
        out[:h, w:],
        out[h:, :w],
        out[h:, w:],
    ]
    for legend, quad in zip(legends, quadrants):
        if int(np.count_nonzero(quad == legend)) == 0:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    if max(2 * h, 2 * w) > 30:
        return None

    max_area = min(SUPPORT_AREA_RANGE[1], h * w - 4)
    min_area = min(SUPPORT_AREA_RANGE[0], max_area)
    if max_area < 7:
        return None

    area = random.randint(min_area, max_area)
    mask = _support_mask(h, w, area)
    if mask is None:
        return None

    pattern = _paint_pattern(mask)
    if pattern is None:
        return None

    legends = tuple(random.sample(CORNER_PALETTE, 4))
    inp = _embed_with_corners(pattern, legends)
    if not _validate_example(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
