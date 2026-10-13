import random

import numpy as np


BG = 0
SEPARATOR = 9
PANEL_COUNT = 4
MAX_GENERATION_ATTEMPTS = 300
LAYOUTS = [(1, 4), (2, 2), (4, 1)]
PANEL_SIZE_RANGE = (4, 7)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]

def _separator_color(grid: np.ndarray) -> int:
    panel_region = grid[:-1]
    best_color = None
    best_score = -1
    for color in map(int, np.unique(panel_region)):
        full_rows = int(np.all(panel_region == color, axis=1).sum())
        full_cols = int(np.all(panel_region == color, axis=0).sum())
        score = full_rows + full_cols
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score > 0, "failed to identify separator color"
    return int(best_color)


def _segments_from_separators(indices: list[int], length: int) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted(set(indices)) + [length]:
        start = prev + 1
        end = idx
        if end > start:
            segments.append((start, end))
        prev = idx
    assert segments, "expected at least one panel segment"
    return segments


def _palette_order(bottom_row: np.ndarray, bg: int, separator: int) -> list[int]:
    non_special = [i for i, color in enumerate(bottom_row) if int(color) not in {bg, separator}]
    assert non_special, "expected a palette row"

    runs: list[tuple[int, int]] = []
    start = prev = non_special[0]
    for idx in non_special[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))

    run_start, run_end = max(runs, key=lambda rc: rc[1] - rc[0])
    palette = [int(color) for color in bottom_row[run_start : run_end + 1]]
    assert len(palette) == len(set(palette)), "palette colors must be unique"
    return palette


def _panel_segments(grid: np.ndarray, separator: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    panel_region = grid[:-1]
    row_lines = [r for r in range(panel_region.shape[0]) if bool(np.all(panel_region[r, :] == separator))]
    col_lines = [c for c in range(panel_region.shape[1]) if bool(np.all(panel_region[:, c] == separator))]

    row_segments = _segments_from_separators(row_lines, panel_region.shape[0])
    col_segments = _segments_from_separators(col_lines, panel_region.shape[1])

    heights = {r1 - r0 for r0, r1 in row_segments}
    widths = {c1 - c0 for c0, c1 in col_segments}
    assert len(heights) == 1 and len(widths) == 1, "panels must be equal-sized"
    assert heights == widths, "panels must be square"
    return row_segments, col_segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each panel's pattern according to the order of its seed color in the bottom palette row.

    Input structure
    - A single-pixel-wide separator color partitions the upper region into equal square panels.
    - Each panel contains one shared-color pattern plus a unique seed pixel at the panel's top-left cell.
    - The bottom row contains a contiguous palette strip listing the seed colors from left to right.

    Rule
    - Map palette position 0, 1, 2, 3 to clockwise rotations 0, 90, 180, 270 degrees.
    - For each panel, remove its seed pixel, rotate the remaining panel content by the seed's mapped angle,
      and write the rotated pattern back into the same panel.
    - Keep the separator grid and the palette row unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "expected a 2D grid with a palette row"

    bg = BG
    separator = _separator_color(g)
    palette = _palette_order(g[-1], bg, separator)
    palette_index = {color: idx for idx, color in enumerate(palette)}

    row_segments, col_segments = _panel_segments(g, separator)
    assert len(row_segments) * len(col_segments) == len(palette), "panel count must match palette size"

    out = g.copy()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            panel = g[r0:r1, c0:c1].copy()
            seed_color = int(panel[0, 0])
            assert seed_color in palette_index, "panel seed must appear in the palette row"
            turns = palette_index[seed_color] % 4
            panel[0, 0] = bg
            out[r0:r1, c0:c1] = np.rot90(panel, -turns)

    return out


def _neighbors(size: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size and (nr, nc) != (0, 0):
            out.append((nr, nc))
    return out


def _mask_variants(mask: np.ndarray) -> set[bytes]:
    return {np.rot90(mask, k).tobytes() for k in range(4)}


def _random_mask(size: int) -> np.ndarray | None:
    min_area = max(3, size - 1)
    max_area = min(size * size - 2, size + 4)
    target_area = random.randint(min_area, max_area)

    choices = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    start = random.choice(choices)
    cells = {start}
    frontier = set(_neighbors(size, start))

    while len(cells) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for neighbor in _neighbors(size, cell):
            if neighbor not in cells:
                frontier.add(neighbor)

    if len(cells) != target_area:
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if len(_mask_variants(mask)) != 4:
        return None
    return mask


def _generate_pattern(size: int) -> np.ndarray:
    for _ in range(100):
        mask = _random_mask(size)
        if mask is not None:
            return mask
    raise RuntimeError("failed to generate a non-symmetric panel pattern")


def _panel_starts(count: int, panel_size: int) -> list[int]:
    return [1 + idx * (panel_size + 1) for idx in range(count)]


def _render_grid(
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    pattern_mask: np.ndarray,
    pattern_color: int,
    panel_seeds: list[int],
    palette: list[int],
) -> np.ndarray:
    body_h = panel_rows * panel_size + panel_rows + 1
    body_w = panel_cols * panel_size + panel_cols + 1
    grid = np.full((body_h + 1, body_w), BG, dtype=int)

    for r in range(0, body_h, panel_size + 1):
        grid[r, :] = SEPARATOR
    for c in range(0, body_w, panel_size + 1):
        grid[:body_h, c] = SEPARATOR
    grid[body_h - 1, :] = SEPARATOR

    seed_iter = iter(panel_seeds)
    row_starts = _panel_starts(panel_rows, panel_size)
    col_starts = _panel_starts(panel_cols, panel_size)
    for r0 in row_starts:
        for c0 in col_starts:
            panel = grid[r0 : r0 + panel_size, c0 : c0 + panel_size]
            panel[pattern_mask] = pattern_color
            panel[0, 0] = next(seed_iter)

    palette_start = random.randint(0, body_w - len(palette))
    grid[-1, palette_start : palette_start + len(palette)] = palette
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = random.choice(LAYOUTS)
    panel_size = random.randint(*PANEL_SIZE_RANGE)
    pattern_mask = _generate_pattern(panel_size)

    pattern_color = random.choice(COLOR_POOL)
    available = [color for color in COLOR_POOL if color != pattern_color]
    seed_colors = random.sample(available, PANEL_COUNT)

    palette = random.sample(seed_colors, len(seed_colors))
    panel_seeds = random.sample(seed_colors, len(seed_colors))
    grid = _render_grid(panel_rows, panel_cols, panel_size, pattern_mask, pattern_color, panel_seeds, palette)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
