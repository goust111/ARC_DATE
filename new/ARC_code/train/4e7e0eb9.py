from collections import Counter

import numpy as np

from arc_utils import get_bg, paste
import random


SLOTS = (
    (1, 4, 1, 4),
    (1, 4, 5, 8),
    (5, 8, 1, 4),
    (5, 8, 5, 8),
)


def _is_full_non_bg_line(line: np.ndarray, bg: int) -> bool:
    return bool(np.all(line == line[0]) and int(line[0]) != int(bg))


def _slot_colors(panel: np.ndarray) -> list[int]:
    colors = []
    for r0, r1, c0, c1 in SLOTS:
        block = panel[r0:r1, c0:c1]
        assert np.all(block == block[0, 0]), "each 3x3 slot must be monochrome"
        colors.append(int(block[0, 0]))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

    Input structure
    The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
    panel has four monochrome 3x3 slots at the quadrant positions, with either a
    full non-background center row, a full non-background center column, or no
    guide line.

    Rule
    A panel with a full center column is reflected left-to-right, and a panel
    with a full center row is reflected top-to-bottom. A panel with no guide line
    has exactly one slot color that appears once and one slot color that appears
    three times; recolor all four slots to the once-occurring slot color.

    Key insight
    The center guide line encodes the mirror axis, while panels without a guide
    line are completed by copying the structurally unique slot color.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert (H + 1) % 10 == 0 and (W + 1) % 10 == 0, "expected 9x9 panels separated every 10 cells"

    out = g.copy()
    for panel_r0 in range(0, H, 10):
        for panel_c0 in range(0, W, 10):
            panel = g[panel_r0 : panel_r0 + 9, panel_c0 : panel_c0 + 9]
            assert panel.shape == (9, 9), "incomplete 9x9 panel"
            bg = get_bg(panel)

            has_row_axis = _is_full_non_bg_line(panel[4, :], bg)
            has_col_axis = _is_full_non_bg_line(panel[:, 4], bg)
            assert not (has_row_axis and has_col_axis), "panel cannot have both mirror axes"

            if has_col_axis:
                out[panel_r0 : panel_r0 + 9, panel_c0 : panel_c0 + 9] = np.fliplr(panel)
            elif has_row_axis:
                out[panel_r0 : panel_r0 + 9, panel_c0 : panel_c0 + 9] = np.flipud(panel)
            else:
                assert np.all(panel[4, :] == bg) and np.all(panel[:, 4] == bg), "unguided panel has empty center cross"
                counts = Counter(_slot_colors(panel))
                assert sorted(counts.values()) == [1, 3], "unguided panel must have one exemplar slot"
                fill_colors = [color for color, count in counts.items() if count == 1]
                assert len(fill_colors) == 1
                fill_color = fill_colors[0]
                for r0, r1, c0, c1 in SLOTS:
                    out[panel_r0 + r0 : panel_r0 + r1, panel_c0 + c0 : panel_c0 + c1] = fill_color

    return out

# --------------- generator (att2, o5x) ---------------
# Canvas and panel geometry.
PANEL_SIZE = 9
SEPARATOR_WIDTH = 1
PANEL_STRIDE = PANEL_SIZE + SEPARATOR_WIDTH
MIN_PANEL_COUNT, MAX_PANEL_COUNT = 1, 3
CENTER_INDEX = 4
SLOT_BOUNDS = (
    (1, 4, 1, 4),
    (1, 4, 5, 8),
    (5, 8, 1, 4),
    (5, 8, 5, 8),
)

# Color choices.
BACKGROUND_COLOR, GUIDE_COLOR, NO_GUIDE_BASE_COLOR = 0, 4, 1
GUIDED_PALETTE = tuple(range(1, 10))
UNIQUE_SLOT_COLORS = tuple(range(2, 10))
SEPARATOR_COLORS = (2, 3, 5, 6, 7, 8, 9)

# Panel type and retry sampling.
COL_AXIS, ROW_AXIS, NO_GUIDE = PANEL_KINDS = ("col_axis", "row_axis", "no_guide")
MAX_GENERATION_ATTEMPTS = 300


def _guided_colors() -> list[int]:
    first = random.choice(GUIDED_PALETTE)
    second = random.choice([color for color in GUIDED_PALETTE if color != first])
    colors = [first, second, random.choice(GUIDED_PALETTE), random.choice(GUIDED_PALETTE)]
    random.shuffle(colors)
    return colors


def _fill_slots(panel: np.ndarray, colors: list[int]) -> None:
    for color, (r0, r1, c0, c1) in zip(colors, SLOT_BOUNDS):
        panel[r0:r1, c0:c1] = color


def _make_panel(kind: str) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND_COLOR, dtype=int)

    if kind == NO_GUIDE:
        unique_pos = random.randrange(len(SLOT_BOUNDS))
        unique_color = random.choice(UNIQUE_SLOT_COLORS)
        colors = [NO_GUIDE_BASE_COLOR] * len(SLOT_BOUNDS)
        colors[unique_pos] = unique_color
        _fill_slots(panel, colors)

        out = panel.copy()
        _fill_slots(out, [unique_color] * len(SLOT_BOUNDS))
        return panel, out

    colors = _guided_colors()
    _fill_slots(panel, colors)
    if kind == COL_AXIS:
        panel[:, CENTER_INDEX] = GUIDE_COLOR
        return panel, np.fliplr(panel)
    if kind == ROW_AXIS:
        panel[CENTER_INDEX, :] = GUIDE_COLOR
        return panel, np.flipud(panel)
    raise ValueError(f"unknown panel kind: {kind}")


def _make_grid_pair(nr: int, nc: int, sep_color: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = nr * PANEL_STRIDE - SEPARATOR_WIDTH, nc * PANEL_STRIDE - SEPARATOR_WIDTH
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    expected = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    for canvas in (grid, expected):
        if nr > MIN_PANEL_COUNT:
            canvas[PANEL_SIZE::PANEL_STRIDE, :] = sep_color
        if nc > MIN_PANEL_COUNT:
            canvas[:, PANEL_SIZE::PANEL_STRIDE] = sep_color
    return grid, expected


def _matches_solver(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(grid), expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    nr = random.randint(MIN_PANEL_COUNT, MAX_PANEL_COUNT)
    nc = random.randint(MIN_PANEL_COUNT, MAX_PANEL_COUNT)
    sep_color = random.choice(SEPARATOR_COLORS)
    grid, expected = _make_grid_pair(nr, nc, sep_color)

    for pr in range(nr):
        for pc in range(nc):
            panel, out_panel = _make_panel(random.choice(PANEL_KINDS))
            paste(grid, panel, pr * PANEL_STRIDE, pc * PANEL_STRIDE)
            paste(expected, out_panel, pr * PANEL_STRIDE, pc * PANEL_STRIDE)

    return grid if _matches_solver(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 4e7e0eb9 input")
