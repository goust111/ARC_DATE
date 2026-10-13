from collections import Counter
import random

import numpy as np


PANEL_SIZE = 9
STRIDE = 10
BG = 0
SLOTS = (
    (1, 4, 1, 4),
    (1, 4, 5, 8),
    (5, 8, 1, 4),
    (5, 8, 5, 8),
)
PANEL_KINDS = ("row", "col", "none")
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _is_full_non_bg_line(line: np.ndarray, bg: int) -> bool:
    return bool(np.all(line == line[0]) and int(line[0]) != int(bg))


def _panel_intervals(separator_indices: list[int], size: int) -> list[tuple[int, int]]:
    if not separator_indices:
        assert size == PANEL_SIZE, "single-panel grids must be exactly 9 cells wide/tall"
        return [(0, size)]

    intervals = []
    start = 0
    for idx in separator_indices:
        intervals.append((start, idx))
        start = idx + 1
    intervals.append((start, size))
    assert all(end - start == PANEL_SIZE for start, end in intervals), "panels must be 9x9"
    return intervals


def _slot_color(panel: np.ndarray, bounds: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = bounds
    block = panel[r0:r1, c0:c1]
    assert np.all(block == block[0, 0]), "slot blocks must be monochrome"
    return int(block[0, 0])


def _slot_colors(panel: np.ndarray) -> list[int]:
    return [_slot_color(panel, bounds) for bounds in SLOTS]


def _fill_slot(panel: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bounds
    panel[r0:r1, c0:c1] = color


def _transform_panel(panel: np.ndarray) -> np.ndarray:
    row_axis = _is_full_non_bg_line(panel[4, :], BG)
    col_axis = _is_full_non_bg_line(panel[:, 4], BG)
    assert not (row_axis and col_axis), "a panel cannot contain both guide axes"

    if col_axis:
        return np.fliplr(panel)
    if row_axis:
        return np.flipud(panel)

    assert np.all(panel[4, :] == BG) and np.all(panel[:, 4] == BG), "unguided panels must have an empty cross"
    counts = Counter(_slot_colors(panel))
    assert sorted(counts.values()) == [1, 3], "unguided panels need one minority color"
    fill_color = next(color for color, count in counts.items() if count == 1)
    out = panel.copy()
    for bounds in SLOTS:
        _fill_slot(out, bounds, fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_separators = [r for r in range(h) if _is_full_non_bg_line(g[r, :], BG)]
    col_separators = [c for c in range(w) if _is_full_non_bg_line(g[:, c], BG)]
    row_intervals = _panel_intervals(row_separators, h)
    col_intervals = _panel_intervals(col_separators, w)

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (PANEL_SIZE, PANEL_SIZE), "every panel must be 9x9"
            out[r0:r1, c0:c1] = _transform_panel(panel)
    return out


def _sample_separator_color() -> int:
    return random.choice(PALETTE)


def _colors_excluding(*blocked: int) -> list[int]:
    blocked_set = set(blocked)
    return [color for color in PALETTE if color not in blocked_set]


def _make_guided_panel(kind: str, guide_color: int) -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    colors = random.sample(_colors_excluding(guide_color), 4)
    random.shuffle(colors)
    for bounds, color in zip(SLOTS, colors, strict=False):
        _fill_slot(panel, bounds, color)
    if kind == "row":
        panel[4, :] = guide_color
    elif kind == "col":
        panel[:, 4] = guide_color
    else:
        raise ValueError(f"unknown guided panel kind: {kind}")
    return panel


def _make_unguided_panel() -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BG, dtype=int)
    major, minor = random.sample(PALETTE, 2)
    colors = [major, major, major, major]
    colors[random.randrange(4)] = minor
    random.shuffle(colors)
    for bounds, color in zip(SLOTS, colors, strict=False):
        _fill_slot(panel, bounds, color)
    return panel


def _make_panel(kind: str, separator_color: int) -> np.ndarray:
    if kind == "none":
        return _make_unguided_panel()
    guide_color = random.choice(_colors_excluding(separator_color))
    return _make_guided_panel(kind, guide_color)


def _balanced_panel_kinds(count: int) -> list[str]:
    kinds = list(PANEL_KINDS)
    while len(kinds) < count:
        kinds.extend(random.sample(PANEL_KINDS, len(PANEL_KINDS)))
    random.shuffle(kinds)
    return kinds[:count]


def _assemble_grid(n_rows: int, n_cols: int, separator_color: int, panels: list[np.ndarray]) -> np.ndarray:
    height = n_rows * PANEL_SIZE + (n_rows - 1)
    width = n_cols * PANEL_SIZE + (n_cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    if n_rows > 1:
        for r in range(PANEL_SIZE, height, STRIDE):
            grid[r, :] = separator_color
    if n_cols > 1:
        for c in range(PANEL_SIZE, width, STRIDE):
            grid[:, c] = separator_color

    idx = 0
    for pr in range(n_rows):
        for pc in range(n_cols):
            r0 = pr * STRIDE
            c0 = pc * STRIDE
            grid[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = panels[idx]
            idx += 1
    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    panel_count = n_rows * n_cols
    separator_color = _sample_separator_color()
    kinds = _balanced_panel_kinds(panel_count)
    panels = [_make_panel(kind, separator_color) for kind in kinds]
    grid = _assemble_grid(n_rows, n_cols, separator_color, panels)
    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
