import random
from typing import Iterable

import numpy as np


BACKGROUND = 5  # grey
SEPARATOR = 1  # blue
MARKER = 2  # red
PANEL_SIZES = (3, 5, 7)
PANEL_COUNT_RANGE = (2, 4)
PALETTE = (0, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        mask = np.all(grid == color, axis=1)
    else:
        mask = np.all(grid == color, axis=0)
    return [int(i) for i in np.flatnonzero(mask)]


def _partition_from_lines(length: int, lines: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [idx + 1 for idx in lines]
    ends = lines + [length]
    spans = [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]
    assert spans
    sizes = {b - a for a, b in spans}
    assert len(sizes) == 1
    size = sizes.pop()
    assert size > 0 and size % 2 == 1
    if len(lines) > 1:
        steps = {b - a for a, b in zip(lines, lines[1:], strict=False)}
        assert steps == {size + 1}
    return spans


def _panel_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        row_lines = _full_line_indices(grid, color, axis=0)
        col_lines = _full_line_indices(grid, color, axis=1)
        if not row_lines or not col_lines:
            continue
        try:
            rows = _partition_from_lines(grid.shape[0], row_lines)
            cols = _partition_from_lines(grid.shape[1], col_lines)
        except AssertionError:
            continue
        side = rows[0][1] - rows[0][0]
        if all((r1 - r0) == side for r0, r1 in rows) and all((c1 - c0) == side for c0, c1 in cols):
            candidates.append((len(row_lines) + len(col_lines), color, rows, cols))
    assert candidates
    candidates.sort(reverse=True)
    _score, sep, rows, cols = candidates[0]
    return rows, cols, int(sep)


def _is_fourfold_symmetric(panel: np.ndarray) -> bool:
    return bool(
        np.array_equal(panel, np.rot90(panel, 1))
        and np.array_equal(panel, np.rot90(panel, 2))
        and np.array_equal(panel, np.rot90(panel, 3))
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, sep = _panel_spans(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            assert panel.shape[0] == panel.shape[1]
            assert panel.shape[0] % 2 == 1
            assert not np.any(panel == sep)
            if _is_fourfold_symmetric(panel):
                center = panel.shape[0] // 2
                out[r0 + center, c0 + center] = MARKER

    return out


def _orbit_cells(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        seen.add((rr, cc))
        rr, cc = cc, side - 1 - rr
    return tuple(sorted(seen))


def _representatives(side: int) -> Iterable[tuple[int, int]]:
    center = side // 2
    for r in range(side):
        for c in range(side):
            orbit = _orbit_cells(side, r, c)
            if orbit[0] == (r, c):
                yield r, c
    assert center >= 1


def _make_symmetric_panel(side: int) -> np.ndarray:
    panel = np.full((side, side), BACKGROUND, dtype=int)
    used_color = False
    for r, c in _representatives(side):
        orbit = _orbit_cells(side, r, c)
        if len(orbit) == 1:
            color = random.choice(PALETTE) if random.random() < 0.65 else BACKGROUND
        else:
            color = random.choice(PALETTE) if random.random() < 0.55 else BACKGROUND
        if color != BACKGROUND:
            used_color = True
        for rr, cc in orbit:
            panel[rr, cc] = color
    if not used_color:
        center = side // 2
        panel[center, center] = random.choice(PALETTE)
    return panel


def _make_asymmetric_panel(side: int) -> np.ndarray:
    for _ in range(200):
        panel = np.full((side, side), BACKGROUND, dtype=int)
        fill = random.randint(max(2, side), max(3, side * side // 2))
        cells = random.sample([(r, c) for r in range(side) for c in range(side)], fill)
        color_count = random.randint(1, min(4, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        for r, c in cells:
            panel[r, c] = random.choice(colors)
        if not _is_fourfold_symmetric(panel):
            return panel
    raise RuntimeError("failed to build asymmetric panel")


def _balanced_flags(count: int) -> list[bool]:
    sym_count = count // 2
    if count % 2 == 1:
        sym_count += random.randint(0, 1)
    flags = [True] * sym_count + [False] * (count - sym_count)
    random.shuffle(flags)
    return flags


def _gen_attempt() -> np.ndarray:
    side = random.choice(PANEL_SIZES)
    panel_rows = random.randint(*PANEL_COUNT_RANGE)
    panel_cols = random.randint(*PANEL_COUNT_RANGE)
    h = panel_rows * side + (panel_rows - 1)
    w = panel_cols * side + (panel_cols - 1)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for r in range(1, panel_rows):
        grid[r * side + (r - 1), :] = SEPARATOR
    for c in range(1, panel_cols):
        grid[:, c * side + (c - 1)] = SEPARATOR

    flags = _balanced_flags(panel_rows * panel_cols)
    if panel_rows * panel_cols > 1 and (all(flags) or not any(flags)):
        flags[0] = True
        flags[-1] = False
        random.shuffle(flags)

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (side + 1)
            c0 = pc * (side + 1)
            panel = _make_symmetric_panel(side) if flags[idx] else _make_asymmetric_panel(side)
            grid[r0 : r0 + side, c0 : c0 + side] = panel
            idx += 1

    out = main(grid)
    changed = int(np.count_nonzero(out != grid))
    if panel_rows * panel_cols > 1 and changed == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
