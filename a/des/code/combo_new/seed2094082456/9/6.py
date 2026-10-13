from __future__ import annotations

import random

import numpy as np


ALL_COLORS = tuple(range(10))
PATH_COLOR = 5

# Generation bounds.
MIN_WIDTH = 8
MAX_WIDTH = 28
MIN_BANDS = 2
MAX_BANDS = 6
MIN_TOP_PAD = 1
MAX_TOP_PAD = 5
MIN_BOTTOM_PAD = 1
MAX_BOTTOM_PAD = 5
MIN_GAP = 1
MAX_GAP = 3
MAX_PATHS = 6
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height > 0 and width > 0

    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    candidates: list[dict[str, object]] = []

    for path_color in colors:
        band_rows: list[int] = []
        band_colors: list[int] = []
        path_patterns: list[tuple[int, ...]] = []
        valid = True

        for r in range(height):
            row = g[r]
            non_bg_colors = set(map(int, np.unique(row))) - {bg}
            if not non_bg_colors:
                valid = False
                break

            if non_bg_colors <= {path_color}:
                path_patterns.append(tuple(map(int, np.flatnonzero(row == path_color))))
                continue

            remaining = row[row != path_color]
            remaining_colors = set(map(int, np.unique(remaining)))
            if len(remaining_colors) == 1 and bg not in remaining_colors:
                band_rows.append(r)
                band_colors.append(int(next(iter(remaining_colors))))
                continue

            valid = False
            break

        if not valid or not band_rows:
            continue
        if len(set(band_colors)) != len(band_colors):
            continue
        if any(b - a < 2 for a, b in zip(band_rows, band_rows[1:])):
            continue

        if not path_patterns or not all(path_patterns):
            continue
        first_pattern = path_patterns[0]
        if any(pattern != first_pattern for pattern in path_patterns[1:]):
            continue

        path_cols = list(first_pattern)
        non_path_mask = np.ones(width, dtype=bool)
        non_path_mask[path_cols] = False
        band_row_set = set(band_rows)

        for r, band_color in zip(band_rows, band_colors):
            if np.any(g[r, non_path_mask] != band_color):
                valid = False
                break
            if not np.all(np.isin(g[r, path_cols], [band_color, path_color])):
                valid = False
                break
        if not valid:
            continue

        for r in range(height):
            if r in band_row_set:
                continue
            if np.any(g[r, non_path_mask] != bg):
                valid = False
                break
            if not np.all(g[r, path_cols] == path_color):
                valid = False
                break
        if not valid:
            continue

        candidates.append(
            {
                "bg": bg,
                "path_color": path_color,
                "path_cols": path_cols,
                "band_rows": band_rows,
                "band_colors": band_colors,
            }
        )

    assert len(candidates) == 1, f"expected one valid band/path interpretation, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each vertical gray path with the color of the first horizontal band it crosses.

    Input structure:
    - A dominant background fills the canvas.
    - Several one-row horizontal color bands span the full width; each band uses a different color and neighboring bands are separated by background rows.
    - One or more one-pixel-wide vertical gray paths run from the top edge to the bottom edge, crossing every band.

    Rule:
    - Identify the path columns and the ordered list of horizontal band rows.
    - Read the bands from top to bottom and take the first band's color as the instruction color for every path.
    - Repaint each full vertical path, including every crossing cell, with that first-band color.
    - Leave all non-path cells unchanged.

    Key insight:
    - The gray columns and the colored rows play different roles: the row order encodes the repaint color, and the topmost band is the decisive signal.
    """
    g = np.asarray(grid, dtype=int)
    structure = _detect_structure(g)
    path_cols = list(structure["path_cols"])
    band_colors = list(structure["band_colors"])
    assert band_colors, "expected at least one horizontal band"

    out = g.copy()
    out[:, path_cols] = band_colors[0]
    return out


def _choose_non_adjacent_columns(width: int, count: int) -> list[int] | None:
    for _ in range(80):
        cols = sorted(random.sample(range(width), count))
        if all(b - a > 1 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_band_rows(band_count: int) -> list[int] | None:
    top_pad = random.randint(MIN_TOP_PAD, MAX_TOP_PAD)
    bottom_pad = random.randint(MIN_BOTTOM_PAD, MAX_BOTTOM_PAD)
    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(band_count - 1)]

    height = top_pad + bottom_pad + band_count + sum(gaps)
    if height > 30:
        return None

    rows: list[int] = []
    r = top_pad
    for i in range(band_count):
        rows.append(r)
        if i < band_count - 1:
            r += 1 + gaps[i]
    return rows


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    band_count = random.randint(MIN_BANDS, MAX_BANDS)
    band_rows = _build_band_rows(band_count)
    if band_rows is None:
        return None

    height = band_rows[-1] + random.randint(MIN_BOTTOM_PAD, MAX_BOTTOM_PAD) + 1
    if height > 30:
        return None

    max_paths = min(MAX_PATHS, (width + 1) // 2, max(1, width // 3))
    path_count = random.randint(1, max_paths)
    path_cols = _choose_non_adjacent_columns(width, path_count)
    if path_cols is None:
        return None

    bg_choices = [color for color in ALL_COLORS if color != PATH_COLOR]
    bg = random.choice(bg_choices)
    band_palette = [color for color in ALL_COLORS if color not in (bg, PATH_COLOR)]
    if len(band_palette) < band_count:
        return None
    band_colors = random.sample(band_palette, band_count)

    show_gray_crossings = random.choice([False, True])
    grid = np.full((height, width), bg, dtype=int)
    for row, color in zip(band_rows, band_colors):
        grid[row, :] = color

    band_row_set = set(band_rows)
    for col in path_cols:
        if show_gray_crossings:
            grid[:, col] = PATH_COLOR
        else:
            for row in range(height):
                if row not in band_row_set:
                    grid[row, col] = PATH_COLOR

    if _most_common_color(grid) != bg:
        return None

    try:
        structure = _detect_structure(grid)
        out = main(grid)
    except AssertionError:
        return None

    if int(structure["bg"]) != bg:
        return None
    if int(structure["path_color"]) != PATH_COLOR:
        return None
    if list(structure["path_cols"]) != path_cols:
        return None
    if list(structure["band_rows"]) != band_rows:
        return None
    if list(structure["band_colors"]) != band_colors:
        return None

    expected = grid.copy()
    expected[:, path_cols] = band_colors[0]
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
