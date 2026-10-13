from __future__ import annotations

import random
from typing import Iterable

import numpy as np


COLOR_POOL = tuple(range(10))
LEGEND_COUNT_CHOICES = (2, 3, 4)
CELL_WIDTH_CHOICES = (3, 4, 5)
LEGEND_HEIGHT_CHOICES = (4, 5, 6)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120
PATTERN_ATTEMPTS = 80


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _legend_boundaries(legend: np.ndarray, grey: int) -> list[int] | None:
    h, w = legend.shape
    if h < 2 or w < 5:
        return None
    boundaries = [c for c in range(w) if np.all(legend[:, c] == grey)]
    if len(boundaries) < 3 or boundaries[0] != 0 or boundaries[-1] != w - 1:
        return None
    steps = [b1 - b0 for b0, b1 in zip(boundaries, boundaries[1:], strict=False)]
    if not steps or len(set(steps)) != 1 or steps[0] < 3:
        return None
    cell_w = steps[0] - 1
    if cell_w < 2:
        return None
    return boundaries


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, _ = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 2):
        row = grid[r]
        if not np.all(row == row[0]):
            continue
        grey = int(row[0])
        if grey == bg:
            continue
        if _legend_boundaries(grid[r + 1 :], grey) is None:
            continue
        candidates.append((r, grey))
    assert candidates, "expected a full-width legend separator row"
    return max(candidates, key=lambda item: item[0])


def _extract_legend(grid: np.ndarray, sep_row: int, bg: int, grey: int) -> dict[int, np.ndarray]:
    legend = grid[sep_row + 1 :]
    boundaries = _legend_boundaries(legend, grey)
    assert boundaries is not None, "legend boundaries must be detectable"

    mapping: dict[int, np.ndarray] = {}
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        cell = legend[:, left + 1 : right]
        assert cell.size > 0, "legend cell must have positive size"
        key = int(cell[0, 0])
        assert key not in {bg, grey}, "legend key colors must be distinct from bg and grey"
        pattern = (cell == key)
        pattern[0, 0] = False
        assert np.any(pattern), "legend pattern must contain at least one foreground pixel"
        assert key not in mapping, "legend key colors must be unique"
        mapping[key] = pattern
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle by tiling the legend pattern keyed by its marker color.

    Input structure:
    - The most common color is the background.
    - The main area contains several grey rectangular frames; each frame encloses a solid rectangular interior with exactly one non-background, non-grey marker pixel.
    - The bottom legend area is separated from the main area by a full-width grey row, and the legend is split into equal-width cells by vertical grey lines.
    - In each legend cell, the top-left pixel is an index color and the remaining pixels form a binary pattern drawn in that same index color on the background.

    Rule:
    - For each enclosed rectangle, read its marker color.
    - Find the legend cell whose index color matches that marker color, and treat that cell's binary pattern as a repeating tile.
    - Starting from the interior's top-left corner, tile that pattern across the rectangle interior and recolor only the interior background pixels where the tile is foreground.
    - Keep the grey borders, the marker pixels, and the legend unchanged.

    Key insight:
    - The marker color is a lookup key into the bottom legend; the actual fill is a spatial pattern, not a flat color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    sep_row, grey = _find_separator_row(g, bg)
    mapping = _extract_legend(g, sep_row, bg, grey)

    out = g.copy()
    main_area = g[:sep_row, :]
    for comp in _components(main_area != grey):
        if any(r == 0 or c == 0 or r == main_area.shape[0] - 1 or c == main_area.shape[1] - 1 for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        marker_colors = sorted({int(v) for v in np.unique(patch) if int(v) != bg})
        if len(marker_colors) != 1:
            continue
        marker_color = marker_colors[0]
        if marker_color not in mapping:
            continue
        if np.count_nonzero(patch == marker_color) != 1:
            continue

        pattern = mapping[marker_color]
        ph, pw = pattern.shape
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                if out[r0 + rr, c0 + cc] != bg:
                    continue
                if pattern[rr % ph, cc % pw]:
                    out[r0 + rr, c0 + cc] = marker_color
    return out


def _random_pattern(height: int, width: int) -> np.ndarray | None:
    rr, cc = np.indices((height, width))
    for _ in range(PATTERN_ATTEMPTS):
        mode = random.choice(("checker", "vstripe", "hstripe", "diag", "border", "random"))
        if mode == "checker":
            pattern = (rr + cc + random.randint(0, 1)) % 2 == 0
        elif mode == "vstripe":
            pattern = (cc + random.randint(0, 1)) % 2 == 0
        elif mode == "hstripe":
            pattern = (rr + random.randint(0, 1)) % 2 == 0
        elif mode == "diag":
            pattern = (rr - cc) % random.choice((2, 3)) == 0
        elif mode == "border":
            pattern = np.zeros((height, width), dtype=bool)
            pattern[0, :] = True
            pattern[-1, :] = True
            pattern[:, 0] = True
            pattern[:, -1] = True
            if height > 2 and width > 2 and random.random() < 0.5:
                pattern[1:-1, 1:-1] = (rr[1:-1, 1:-1] + cc[1:-1, 1:-1]) % 2 == 0
        else:
            pattern = np.zeros((height, width), dtype=bool)
            density = random.uniform(0.25, 0.65)
            for r in range(height):
                for c in range(width):
                    pattern[r, c] = random.random() < density

        if random.random() < 0.5:
            pattern = np.flipud(pattern)
        if random.random() < 0.5:
            pattern = np.fliplr(pattern)

        pattern = pattern.copy()
        pattern[0, 0] = False
        total = int(pattern.sum())
        if 2 <= total < height * width - 1:
            return pattern
    return None


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _sample_region_colors(legend_colors: list[int], count: int) -> list[int]:
    if count == 1:
        return [random.choice(legend_colors)]
    if count <= len(legend_colors) and random.random() < 0.5:
        colors = random.sample(legend_colors, count)
    else:
        colors = [random.choice(legend_colors) for _ in range(count)]
        if len(set(colors)) == 1:
            colors[0] = legend_colors[0]
            colors[-1] = legend_colors[-1]
    random.shuffle(colors)
    return colors


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], grey: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = grey
    grid[r1, c0 : c1 + 1] = grey
    grid[r0 : r1 + 1, c0] = grey
    grid[r0 : r1 + 1, c1] = grey


def _place_regions(
    grid: np.ndarray,
    main_h: int,
    grey: int,
    legend_colors: list[int],
) -> bool:
    height, width = grid.shape
    del height

    max_inner_h = min(7, main_h - 4)
    max_inner_w = min(7, width - 4)
    if max_inner_h < 3 or max_inner_w < 3:
        return False

    capacity = max(1, min(4, (main_h * width) // 42))
    region_count = random.randint(1, capacity)
    region_colors = _sample_region_colors(legend_colors, region_count)

    placed: list[tuple[int, int, int, int]] = []
    for color in region_colors:
        placed_one = False
        for _ in range(PLACEMENT_ATTEMPTS):
            inner_h = random.randint(3, max_inner_h)
            inner_w = random.randint(3, max_inner_w)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = main_h - outer_h - 1
            max_c0 = width - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                return False
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(bbox, other) for other in placed):
                continue

            _draw_frame(grid, bbox, grey)
            mr = random.randint(r0 + 1, bbox[2] - 1)
            mc = random.randint(c0 + 1, bbox[3] - 1)
            grid[mr, mc] = color
            placed.append(bbox)
            placed_one = True
            break
        if not placed_one:
            return False
    return True


def _build_legend(
    grid: np.ndarray,
    sep_row: int,
    grey: int,
    bg: int,
    legend_colors: list[int],
    patterns: dict[int, np.ndarray],
) -> None:
    legend_h = grid.shape[0] - sep_row - 1
    cell_w = next(iter(patterns.values())).shape[1]
    grid[sep_row, :] = grey
    legend = grid[sep_row + 1 :, :]

    boundaries = [i * (cell_w + 1) for i in range(len(legend_colors) + 1)]
    for c in boundaries:
        legend[:, c] = grey
    for idx, color in enumerate(legend_colors):
        left = boundaries[idx]
        cell = legend[:, left + 1 : left + 1 + cell_w]
        cell[:, :] = bg
        cell[0, 0] = color
        cell[patterns[color]] = color
    assert legend.shape[0] == legend_h


def _gen_attempt() -> np.ndarray | None:
    bg, grey = random.sample(COLOR_POOL, 2)
    legend_count = random.choice(LEGEND_COUNT_CHOICES)
    cell_w = random.choice(CELL_WIDTH_CHOICES)
    legend_h = random.choice(LEGEND_HEIGHT_CHOICES)
    width = legend_count * cell_w + legend_count + 1
    if width > 30:
        return None

    max_main_h = 30 - legend_h - 1
    if max_main_h < 8:
        return None
    main_h = random.randint(8, max_main_h)
    total_h = main_h + 1 + legend_h

    available = [c for c in COLOR_POOL if c not in {bg, grey}]
    if len(available) < legend_count:
        return None
    legend_colors = random.sample(available, legend_count)

    patterns: dict[int, np.ndarray] = {}
    for color in legend_colors:
        pattern = _random_pattern(legend_h, cell_w)
        if pattern is None:
            return None
        patterns[color] = pattern

    grid = np.full((total_h, width), bg, dtype=int)
    _build_legend(grid, main_h, grey, bg, legend_colors, patterns)
    if not _place_regions(grid, main_h, grey, legend_colors):
        return None

    if _mode_color(grid) != bg:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the legend-pattern tiling task")
