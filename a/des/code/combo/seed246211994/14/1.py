from __future__ import annotations

from collections import Counter
import random

import numpy as np


PANEL_SIZE = 9
SEPARATOR_WIDTH = 1
PANEL_STRIDE = PANEL_SIZE + SEPARATOR_WIDTH
CENTER = 4
SLOTS = (
    (1, 4, 1, 4),
    (1, 4, 5, 8),
    (5, 8, 1, 4),
    (5, 8, 5, 8),
)
COL_AXIS = "col_axis"
ROW_AXIS = "row_axis"
NO_GUIDE = "no_guide"
PANEL_KINDS = (COL_AXIS, ROW_AXIS, NO_GUIDE)
ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _panel_background(panel: np.ndarray) -> int:
    mask = np.ones((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r0, r1, c0, c1 in SLOTS:
        mask[r0:r1, c0:c1] = False
    mask[CENTER, :] = False
    mask[:, CENTER] = False
    background_cells = panel[mask]
    assert background_cells.size > 0
    assert np.all(background_cells == background_cells[0]), "panel background must be uniform"
    return int(background_cells[0])


def _is_full_non_bg_line(line: np.ndarray, bg: int) -> bool:
    return bool(np.all(line == line[0]) and int(line[0]) != int(bg))


def _slot_colors(panel: np.ndarray) -> list[int]:
    colors: list[int] = []
    for r0, r1, c0, c1 in SLOTS:
        block = panel[r0:r1, c0:c1]
        assert np.all(block == block[0, 0]), "each corner block must be monochrome"
        colors.append(int(block[0, 0]))
    return colors


def _separator_positions(length: int) -> list[int]:
    assert (length + 1) % PANEL_STRIDE == 0, "grid shape must fit 9x9 panels separated every 10 cells"
    panel_count = (length + 1) // PANEL_STRIDE
    return [panel for panel in range(PANEL_SIZE, length, PANEL_STRIDE)][: max(0, panel_count - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each 9x9 panel according to its center guide line.

    Panels with a full non-background center column are mirrored left-right.
    Panels with a full non-background center row are mirrored up-down.
    Panels without a guide line recolor all four corner 3x3 blocks to the
    color that appears exactly once among those four blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    row_separators = _separator_positions(h)
    col_separators = _separator_positions(w)
    for r in row_separators:
        assert np.all(g[r, :] == g[r, 0]), "separator rows must be uniform"
    for c in col_separators:
        assert np.all(g[:, c] == g[0, c]), "separator columns must be uniform"

    out = g.copy()
    for panel_r0 in range(0, h, PANEL_STRIDE):
        for panel_c0 in range(0, w, PANEL_STRIDE):
            panel = g[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE]
            assert panel.shape == (PANEL_SIZE, PANEL_SIZE), "incomplete panel"

            bg = _panel_background(panel)
            has_row_axis = _is_full_non_bg_line(panel[CENTER, :], bg)
            has_col_axis = _is_full_non_bg_line(panel[:, CENTER], bg)
            assert not (has_row_axis and has_col_axis), "panel cannot contain both guide axes"

            if has_col_axis:
                out[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE] = np.fliplr(panel)
                continue

            if has_row_axis:
                out[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE] = np.flipud(panel)
                continue

            assert np.all(panel[CENTER, :] == bg), "unguided panel must have a background center row"
            assert np.all(panel[:, CENTER] == bg), "unguided panel must have a background center column"

            counts = Counter(_slot_colors(panel))
            assert sorted(counts.values()) == [1, 3], "unguided panel must have one rare slot color"
            rare_color = next(color for color, count in counts.items() if count == 1)

            transformed = panel.copy()
            for r0, r1, c0, c1 in SLOTS:
                transformed[r0:r1, c0:c1] = rare_color
            out[panel_r0 : panel_r0 + PANEL_SIZE, panel_c0 : panel_c0 + PANEL_SIZE] = transformed

    return out


def _choose_color(exclude: set[int]) -> int:
    choices = [color for color in ALL_COLORS if color not in exclude]
    assert choices, "no colors available"
    return random.choice(choices)


def _fill_slots(panel: np.ndarray, colors: list[int]) -> None:
    for color, (r0, r1, c0, c1) in zip(colors, SLOTS, strict=True):
        panel[r0:r1, c0:c1] = int(color)


def _build_guided_panel(kind: str, bg: int, forbidden: set[int]) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    guide_color = _choose_color(forbidden | {bg})
    slot_pool = [color for color in ALL_COLORS if color not in forbidden | {bg, guide_color}]
    random.shuffle(slot_pool)
    assert len(slot_pool) >= 4
    slot_colors = slot_pool[:4]

    if kind == COL_AXIS:
        assert slot_colors[0] != slot_colors[1] or slot_colors[2] != slot_colors[3]
        panel[:, CENTER] = guide_color
        _fill_slots(panel, slot_colors)
        return panel, np.fliplr(panel)

    assert kind == ROW_AXIS
    assert slot_colors[0] != slot_colors[2] or slot_colors[1] != slot_colors[3]
    panel[CENTER, :] = guide_color
    _fill_slots(panel, slot_colors)
    return panel, np.flipud(panel)


def _build_unguided_panel(bg: int, forbidden: set[int]) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    majority = _choose_color(forbidden | {bg})
    rare = _choose_color(forbidden | {bg, majority})
    rare_index = random.randrange(len(SLOTS))
    slot_colors = [majority] * len(SLOTS)
    slot_colors[rare_index] = rare
    _fill_slots(panel, slot_colors)

    out = panel.copy()
    _fill_slots(out, [rare] * len(SLOTS))
    return panel, out


def _build_panel(kind: str, bg: int, separator_color: int) -> tuple[np.ndarray, np.ndarray]:
    forbidden = {separator_color}
    if kind == NO_GUIDE:
        return _build_unguided_panel(bg, forbidden)
    return _build_guided_panel(kind, bg, forbidden)


def _draw_grid_shell(panel_rows: int, panel_cols: int, bg: int, separator_color: int) -> tuple[np.ndarray, np.ndarray]:
    height = panel_rows * PANEL_STRIDE - SEPARATOR_WIDTH
    width = panel_cols * PANEL_STRIDE - SEPARATOR_WIDTH
    grid = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)

    for canvas in (grid, expected):
        for r in range(PANEL_SIZE, height, PANEL_STRIDE):
            canvas[r, :] = separator_color
        for c in range(PANEL_SIZE, width, PANEL_STRIDE):
            canvas[:, c] = separator_color

    return grid, expected


def _panel_kind_layout(count: int) -> list[str]:
    if count >= len(PANEL_KINDS):
        kinds = list(PANEL_KINDS)
        while len(kinds) < count:
            kinds.append(random.choice(PANEL_KINDS))
    else:
        kinds = random.sample(PANEL_KINDS, count)
    random.shuffle(kinds)
    return kinds


def _matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(grid), expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.randint(1, 4)
    panel_cols = random.randint(1, 4)
    panel_count = panel_rows * panel_cols

    bg = random.choice(ALL_COLORS)
    separator_color = _choose_color({bg})
    grid, expected = _draw_grid_shell(panel_rows, panel_cols, bg, separator_color)
    kinds = _panel_kind_layout(panel_count)

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            panel, out_panel = _build_panel(kinds[idx], bg, separator_color)
            idx += 1
            r0 = pr * PANEL_STRIDE
            c0 = pc * PANEL_STRIDE
            grid[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = panel
            expected[r0 : r0 + PANEL_SIZE, c0 : c0 + PANEL_SIZE] = out_panel

    if np.array_equal(grid, expected):
        return None
    return grid if _matches_solver(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
