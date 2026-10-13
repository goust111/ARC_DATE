from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND_COLOR = 0
SEPARATOR_COLOR = 9
POINT_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)

PANEL_MIN_SIZE = 5
PANEL_MAX_SIZE = 14
MAX_PANEL_POINT_COLORS = 3
MAX_PANEL_POINTS = 8
GENERATION_RETRIES = 400

PANEL_KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)

    row_candidates: list[tuple[int, int]] = []
    for row in range(height):
        color = int(grid[row, 0])
        if bool(np.all(grid[row, :] == color)):
            row_candidates.append((row, color))

    col_candidates: list[tuple[int, int]] = []
    for col in range(width):
        color = int(grid[0, col])
        if bool(np.all(grid[:, col] == color)):
            col_candidates.append((col, color))

    matches = [(row, col, row_color) for row, row_color in row_candidates for col, col_color in col_candidates if row_color == col_color]
    assert len(matches) == 1, "expected one full-width and one full-height separator line of the same color"
    return matches[0]


def _panel_slices(shape: tuple[int, int], sep_row: int, sep_col: int) -> dict[tuple[int, int], tuple[slice, slice]]:
    height, width = map(int, shape)
    panels = {
        (0, 0): (slice(0, sep_row), slice(0, sep_col)),
        (0, 1): (slice(0, sep_row), slice(sep_col + 1, width)),
        (1, 0): (slice(sep_row + 1, height), slice(0, sep_col)),
        (1, 1): (slice(sep_row + 1, height), slice(sep_col + 1, width)),
    }
    for row_slice, col_slice in panels.values():
        assert row_slice.stop > row_slice.start and col_slice.stop > col_slice.start, "each panel must be non-empty"
    return panels


def _background_color(grid: np.ndarray, separator_color: int) -> int:
    remaining = grid[grid != separator_color]
    assert remaining.size > 0, "separator cannot cover the entire grid"
    return int(Counter(map(int, remaining.reshape(-1))).most_common(1)[0][0])


def _winner_color(region: np.ndarray, background_color: int, separator_color: int) -> int:
    points = region[(region != background_color) & (region != separator_color)]
    assert points.size > 0, "each panel must contain at least one vote pixel"
    counts = Counter(map(int, points.reshape(-1)))
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "each panel must have a strict majority color"
    return int(ranked[0][0])


def _stripe_fill(shape: tuple[int, int], top_color: int, alt_color: int) -> np.ndarray:
    height, width = map(int, shape)
    striped = np.empty((height, width), dtype=int)
    for row in range(height):
        striped[row, :] = top_color if row % 2 == 0 else alt_color
    return striped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the four cross-separated panels using panel-majority votes and the unique source panel.

    Input structure:
    - A uniform background is split by a one-cell-thick full-width/full-height separator cross.
    - The separator creates four rectangular corner panels: top-left, top-right, bottom-left, and bottom-right.
    - Each panel contains scattered single-pixel vote markers in up to three non-background, non-separator colors.
    - Inside every panel, one vote color appears strictly more often than any other vote color.

    Rule:
    - Detect the separator cross and treat the four corner rectangles as independent panels.
    - In each panel, count only the scattered vote pixels; the most frequent vote color is that panel's winner.
    - Exactly one panel has a winner color that appears in no other panel; this is the source panel.
    - Fill the source panel with its winner, fill the two edge-adjacent panels with their own winners, and fill the diagonal panel with horizontal stripes alternating source-winner / diagonal-winner from the top row downward.
    - Keep the separator cross unchanged.

    Key insight:
    - The scattered pixels are not output directly: first they vote for a panel winner, then the unique winner identifies the source panel whose spatial relationship to the others determines whether a panel becomes solid or striped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row, sep_col, separator_color = _find_separator_cross(g)
    background_color = _background_color(g, separator_color)
    panels = _panel_slices(g.shape, sep_row, sep_col)

    winners: dict[tuple[int, int], int] = {}
    for key, (row_slice, col_slice) in panels.items():
        winners[key] = _winner_color(g[row_slice, col_slice], background_color, separator_color)

    winner_counts = Counter(winners.values())
    source_keys = [key for key, color in winners.items() if winner_counts[color] == 1]
    assert len(source_keys) == 1, "expected exactly one panel with a unique winner color"
    source_key = source_keys[0]
    source_color = winners[source_key]

    out = g.copy()
    for key, (row_slice, col_slice) in panels.items():
        panel_shape = out[row_slice, col_slice].shape
        if key == source_key or key[0] == source_key[0] or key[1] == source_key[1]:
            out[row_slice, col_slice] = winners[key]
        else:
            out[row_slice, col_slice] = _stripe_fill(panel_shape, source_color, winners[key])

    return out


def _max_isolated_points(height: int, width: int) -> int:
    return ((height + 1) // 2) * ((width + 1) // 2)


def _sample_counts(max_slots: int, winner_color: int, palette: list[int], require_minor: bool, force_full_palette: bool) -> dict[int, int] | None:
    max_points = min(MAX_PANEL_POINTS, max_slots)
    other_colors = [color for color in palette if color != winner_color]
    if require_minor and not other_colors:
        return None

    for _ in range(80):
        if force_full_palette:
            used_colors = [winner_color] + other_colors
        else:
            min_used = 2 if require_minor else 1
            max_used = min(len(palette), MAX_PANEL_POINT_COLORS)
            used_count = int(np.random.randint(min_used, max_used + 1))
            used_colors = [winner_color]
            if used_count > 1:
                extras = list(np.random.choice(other_colors, size=used_count - 1, replace=False))
                used_colors.extend(int(color) for color in extras)

        winner_min = 3 if len(used_colors) == 1 else 2
        winner_max = min(6, max_points - (len(used_colors) - 1))
        if winner_max < winner_min:
            continue

        winner_count = int(np.random.randint(winner_min, winner_max + 1))
        counts = {winner_color: winner_count}
        remaining = max_points - winner_count

        ok = True
        extras = used_colors[1:]
        for index, color in enumerate(extras):
            extras_left = len(extras) - index
            max_here = min(winner_count - 1, 3, remaining - (extras_left - 1))
            if max_here < 1:
                ok = False
                break
            count = int(np.random.randint(1, max_here + 1))
            counts[int(color)] = count
            remaining -= count
        if not ok:
            continue

        bonus = int(np.random.randint(0, min(remaining, 2) + 1))
        counts[winner_color] += bonus
        return counts

    return None


def _sample_isolated_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    all_coords = [(row, col) for row in range(height) for col in range(width)]
    for _ in range(60):
        blocked = np.zeros((height, width), dtype=bool)
        chosen: list[tuple[int, int]] = []
        for index in np.random.permutation(len(all_coords)):
            row, col = all_coords[int(index)]
            if blocked[row, col]:
                continue
            chosen.append((row, col))
            blocked[max(0, row - 1) : min(height, row + 2), max(0, col - 1) : min(width, col + 2)] = True
            if len(chosen) == count:
                return chosen
    return None


def _make_panel(height: int, width: int, winner_color: int, palette: list[int], require_minor: bool, force_full_palette: bool) -> np.ndarray | None:
    max_slots = _max_isolated_points(height, width)
    counts = _sample_counts(max_slots, winner_color, palette, require_minor=require_minor, force_full_palette=force_full_palette)
    if counts is None:
        return None

    total_points = sum(counts.values())
    positions = _sample_isolated_positions(height, width, total_points)
    if positions is None:
        return None

    colors: list[int] = []
    for color, count in counts.items():
        colors.extend([int(color)] * int(count))
    np.random.shuffle(colors)

    panel = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for (row, col), color in zip(positions, colors):
        panel[row, col] = int(color)
    return panel


def _gen_attempt() -> np.ndarray | None:
    top = int(np.random.randint(PANEL_MIN_SIZE, PANEL_MAX_SIZE + 1))
    bottom = int(np.random.randint(PANEL_MIN_SIZE, PANEL_MAX_SIZE + 1))
    left = int(np.random.randint(PANEL_MIN_SIZE, PANEL_MAX_SIZE + 1))
    right = int(np.random.randint(PANEL_MIN_SIZE, PANEL_MAX_SIZE + 1))

    height = top + 1 + bottom
    width = left + 1 + right
    if max(height, width) > 30:
        return None

    palette_size = 2 if bool(np.random.randint(0, 2) == 0) else 3
    palette = [int(color) for color in np.random.choice(POINT_COLOR_POOL, size=palette_size, replace=False)]
    np.random.shuffle(palette)
    source_color = int(palette[0])
    common_color = int(palette[1])

    source_index = int(np.random.randint(0, 4))
    source_key = PANEL_KEYS[source_index]
    winners = {key: (source_color if key == source_key else common_color) for key in PANEL_KEYS}

    need_third_color = palette_size == 3
    forced_full_palette_key = PANEL_KEYS[int(np.random.randint(0, 4))] if need_third_color else None
    forced_minor_keys = set()
    for key in PANEL_KEYS:
        if bool(np.random.randint(0, 100) < 65):
            forced_minor_keys.add(key)
    if not forced_minor_keys:
        forced_minor_keys.add(PANEL_KEYS[int(np.random.randint(0, 4))])

    panels: dict[tuple[int, int], np.ndarray] = {}
    panel_shapes = {
        (0, 0): (top, left),
        (0, 1): (top, right),
        (1, 0): (bottom, left),
        (1, 1): (bottom, right),
    }
    for key in PANEL_KEYS:
        force_full = key == forced_full_palette_key
        require_minor = key in forced_minor_keys or force_full
        panel = _make_panel(
            panel_shapes[key][0],
            panel_shapes[key][1],
            winners[key],
            palette,
            require_minor=require_minor,
            force_full_palette=force_full,
        )
        if panel is None:
            return None
        panels[key] = panel

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    sep_row = top
    sep_col = left
    grid[sep_row, :] = SEPARATOR_COLOR
    grid[:, sep_col] = SEPARATOR_COLOR

    slices = _panel_slices(grid.shape, sep_row, sep_col)
    for key, (row_slice, col_slice) in slices.items():
        grid[row_slice, col_slice] = panels[key]

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
