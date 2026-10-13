from __future__ import annotations

from collections import Counter
import random

import numpy as np


PANEL_SIZE = 9
SEP_WIDTH = 1
STRIDE = PANEL_SIZE + SEP_WIDTH
CENTER = 4
SLOTS = (
    (1, 4, 1, 4),
    (1, 4, 5, 8),
    (5, 8, 1, 4),
    (5, 8, 5, 8),
)

PANEL_KINDS = ("col_axis", "row_axis", "no_guide")
MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _is_full_non_bg_line(line: np.ndarray, bg: int) -> bool:
    return bool(np.all(line == line[0]) and int(line[0]) != int(bg))


def _slot_colors(panel: np.ndarray) -> list[int]:
    colors: list[int] = []
    for r0, r1, c0, c1 in SLOTS:
        block = panel[r0:r1, c0:c1]
        assert np.all(block == block[0, 0]), "slot must be monochrome"
        colors.append(int(block[0, 0]))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each 9x9 panel by mirroring across its center guide line or
    filling all four slots with the unique singleton slot color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert (h + SEP_WIDTH) % STRIDE == 0 and (w + SEP_WIDTH) % STRIDE == 0, "invalid panel lattice shape"

    out = g.copy()
    for pr in range(0, h, STRIDE):
        for pc in range(0, w, STRIDE):
            panel = g[pr : pr + PANEL_SIZE, pc : pc + PANEL_SIZE]
            assert panel.shape == (PANEL_SIZE, PANEL_SIZE), "incomplete panel"
            bg = _get_bg(panel)

            has_col_axis = _is_full_non_bg_line(panel[:, CENTER], bg)
            has_row_axis = _is_full_non_bg_line(panel[CENTER, :], bg)
            assert not (has_col_axis and has_row_axis), "panel cannot have both axes"

            if has_col_axis:
                out[pr : pr + PANEL_SIZE, pc : pc + PANEL_SIZE] = np.fliplr(panel)
                continue

            if has_row_axis:
                out[pr : pr + PANEL_SIZE, pc : pc + PANEL_SIZE] = np.flipud(panel)
                continue

            assert np.all(panel[:, CENTER] == bg) and np.all(panel[CENTER, :] == bg), "unguided panel must have empty center cross"
            counts = Counter(_slot_colors(panel))
            assert sorted(counts.values()) == [1, 3], "unguided panel must have a 1-vs-3 slot color split"
            fill_color = next(color for color, count in counts.items() if count == 1)
            for r0, r1, c0, c1 in SLOTS:
                out[pr + r0 : pr + r1, pc + c0 : pc + c1] = fill_color

    return out


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _sample_colors(k: int, exclude: set[int]) -> list[int]:
    pool = [c for c in range(10) if c not in exclude]
    assert len(pool) >= k
    random.shuffle(pool)
    return pool[:k]


def _fill_slots(panel: np.ndarray, colors: list[int]) -> None:
    for color, (r0, r1, c0, c1) in zip(colors, SLOTS, strict=False):
        panel[r0:r1, c0:c1] = color


def _make_guided_panel(kind: str, bg: int) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    slot_colors = _sample_colors(4, {bg})
    guide_color = _sample_colors(1, {bg, *slot_colors})[0]
    _fill_slots(panel, slot_colors)

    if kind == "col_axis":
        panel[:, CENTER] = guide_color
        return panel, np.fliplr(panel)
    if kind == "row_axis":
        panel[CENTER, :] = guide_color
        return panel, np.flipud(panel)
    raise ValueError(f"unknown guided panel kind: {kind}")


def _make_no_guide_panel(bg: int) -> tuple[np.ndarray, np.ndarray]:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), bg, dtype=int)
    base_color, unique_color = _sample_colors(2, {bg})
    colors = [base_color] * 4
    colors[random.randrange(4)] = unique_color
    _fill_slots(panel, colors)

    out = panel.copy()
    _fill_slots(out, [unique_color] * 4)
    return panel, out


def _make_panel(kind: str, bg: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "no_guide":
        return _make_no_guide_panel(bg)
    return _make_guided_panel(kind, bg)


def _grid_shape(nr: int, nc: int) -> tuple[int, int]:
    return nr * STRIDE - SEP_WIDTH, nc * STRIDE - SEP_WIDTH


def _balanced_panel_kinds(count: int) -> list[str]:
    reps = (count + len(PANEL_KINDS) - 1) // len(PANEL_KINDS)
    kinds = list(PANEL_KINDS) * reps
    random.shuffle(kinds)
    return kinds[:count]


def _expected_grid(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    nr = random.randint(1, 4)
    nc = random.randint(1, 4)
    total = nr * nc
    bg = random.randint(0, 9)
    sep_color = _sample_colors(1, {bg})[0]
    h, w = _grid_shape(nr, nc)

    grid = np.full((h, w), bg, dtype=int)
    if nr > 1:
        grid[PANEL_SIZE::STRIDE, :] = sep_color
    if nc > 1:
        grid[:, PANEL_SIZE::STRIDE] = sep_color

    panel_kinds = _balanced_panel_kinds(total)
    if total >= 3:
        panel_kinds[0] = "col_axis"
        panel_kinds[1] = "row_axis"
        panel_kinds[2] = "no_guide"
        random.shuffle(panel_kinds)

    idx = 0
    for pr in range(nr):
        for pc in range(nc):
            panel, _ = _make_panel(panel_kinds[idx], bg)
            _paste(grid, panel, pr * STRIDE, pc * STRIDE)
            idx += 1

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    expected = _expected_grid(grid)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
