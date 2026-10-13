from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
SEP = 5
MAX_GENERATION_ATTEMPTS = 300


def _contiguous_spans(indices: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(limit):
        active = idx in indices
        if active and start is None:
            start = idx
        if not active and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, limit))
    return spans


def _infer_separator_color(grid: np.ndarray) -> int:
    line_colors: list[int] = []
    for row in grid:
        if np.all(row == row[0]):
            line_colors.append(int(row[0]))
    for col in grid.T:
        if np.all(col == col[0]):
            line_colors.append(int(col[0]))
    assert line_colors, "expected separator lines"
    return Counter(line_colors).most_common(1)[0][0]


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    sep = _infer_separator_color(g)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]

    row_spans = _contiguous_spans([r for r in range(g.shape[0]) if r not in sep_rows], g.shape[0])
    col_spans = _contiguous_spans([c for c in range(g.shape[1]) if c not in sep_cols], g.shape[1])
    assert row_spans and col_spans, "expected at least one panel span on each axis"

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "panels must have equal size"

    panels: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panels.append(g[r0:r1, c0:c1].copy())

    shape = (next(iter(heights)), next(iter(widths)))
    return panels, shape, sep


def _consensus_panel(panels: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(panels, axis=0)
    n, h, w = stack.shape
    consensus = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            counts = Counter(map(int, stack[:, r, c]))
            ranked = counts.most_common()
            assert len(ranked) >= 1
            if len(ranked) > 1:
                assert ranked[0][1] > ranked[1][1], "majority color must be unique"
            consensus[r, c] = ranked[0][0]
    assert n >= 3
    return consensus


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare equal panels by local coordinates, isolate each panel's one anomaly,
    and collect those anomaly colors into a single panel-sized output.
    """
    panels, (h, w), sep = _split_panels(grid)
    assert 3 <= len(panels) <= 5, "expected 3 to 5 panels"

    consensus = _consensus_panel(panels)
    out = np.full((h, w), BG, dtype=int)
    for panel in panels:
        diff = panel != consensus
        assert int(diff.sum()) == 1, "each panel must have exactly one anomaly"
        r, c = map(int, np.argwhere(diff)[0])
        color = int(panel[r, c])
        assert int(out[r, c]) in (BG, color), "anomaly positions must not conflict"
        out[r, c] = color
    return out


def _paint_rect(panel: np.ndarray, color: int) -> None:
    h, w = panel.shape
    rh = int(np.random.randint(1, h + 1))
    rw = int(np.random.randint(1, w + 1))
    r0 = int(np.random.randint(0, h - rh + 1))
    c0 = int(np.random.randint(0, w - rw + 1))
    panel[r0 : r0 + rh, c0 : c0 + rw] = color


def _paint_line(panel: np.ndarray, color: int) -> None:
    h, w = panel.shape
    if bool(np.random.randint(0, 2)):
        r = int(np.random.randint(0, h))
        c0 = int(np.random.randint(0, w))
        c1 = int(np.random.randint(c0, w))
        panel[r, c0 : c1 + 1] = color
    else:
        c = int(np.random.randint(0, w))
        r0 = int(np.random.randint(0, h))
        r1 = int(np.random.randint(r0, h))
        panel[r0 : r1 + 1, c] = color


def _paint_checker(panel: np.ndarray, color_a: int, color_b: int) -> None:
    h, w = panel.shape
    rh = int(np.random.randint(2, h + 1))
    rw = int(np.random.randint(2, w + 1))
    r0 = int(np.random.randint(0, h - rh + 1))
    c0 = int(np.random.randint(0, w - rw + 1))
    for r in range(r0, r0 + rh):
        for c in range(c0, c0 + rw):
            panel[r, c] = color_a if (r + c) % 2 == 0 else color_b


def _make_template(h: int, w: int) -> np.ndarray:
    panel = np.full((h, w), BG, dtype=int)
    palette = [int(c) for c in np.random.permutation([1, 2, 3, 4, 6, 7, 8, 9])]
    color_count = int(np.random.randint(2, 5))
    colors = palette[:color_count]

    op_count = int(np.random.randint(3, 7))
    for _ in range(op_count):
        mode = int(np.random.randint(0, 3))
        if mode == 0:
            _paint_rect(panel, int(colors[int(np.random.randint(0, len(colors)))]))
        elif mode == 1:
            _paint_line(panel, int(colors[int(np.random.randint(0, len(colors)))]))
        else:
            a, b = np.random.choice(colors, size=2, replace=len(colors) == 1)
            _paint_checker(panel, int(a), int(b))

    if np.all(panel == BG):
        panel[int(np.random.randint(0, h)), int(np.random.randint(0, w))] = colors[0]
    if len(np.unique(panel)) < 2:
        _paint_rect(panel, int(colors[-1]))
    return panel


def _assemble_panels(panels: list[np.ndarray], vertical: bool) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    count = len(panels)
    if vertical:
        grid = np.full((count * panel_h + count + 1, panel_w + 2), SEP, dtype=int)
        for i, panel in enumerate(panels):
            r0 = 1 + i * (panel_h + 1)
            grid[r0 : r0 + panel_h, 1 : 1 + panel_w] = panel
    else:
        grid = np.full((panel_h + 2, count * panel_w + count + 1), SEP, dtype=int)
        for i, panel in enumerate(panels):
            c0 = 1 + i * (panel_w + 1)
            grid[1 : 1 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(np.random.randint(3, 6))
    vertical = bool(np.random.randint(0, 2))
    panel_h = int(np.random.randint(4, 8))
    panel_w = int(np.random.randint(4, 8))
    if panel_h * panel_w < panel_count:
        return None

    template = _make_template(panel_h, panel_w)
    positions = np.random.choice(panel_h * panel_w, size=panel_count, replace=False)
    anomaly_positions = [tuple(map(int, np.unravel_index(int(pos), (panel_h, panel_w)))) for pos in positions]

    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    panels: list[np.ndarray] = []
    for r, c in anomaly_positions:
        panel = template.copy()
        choices = [color for color in colors if color != int(template[r, c])]
        panel[r, c] = int(choices[int(np.random.randint(0, len(choices)))])
        panels.append(panel)

    grid = _assemble_panels(panels, vertical=vertical)

    # Validate the generated example against the intended rule.
    split_panels, shape, sep = _split_panels(grid)
    if len(split_panels) != panel_count or shape != (panel_h, panel_w) or sep != SEP:
        return None
    consensus = _consensus_panel(split_panels)
    seen_positions: set[tuple[int, int]] = set()
    for panel in split_panels:
        diff = np.argwhere(panel != consensus)
        if len(diff) != 1:
            return None
        pos = tuple(map(int, diff[0]))
        if pos in seen_positions:
            return None
        seen_positions.add(pos)

    out = main(grid)
    if out.shape != (panel_h, panel_w):
        return None
    if int(np.count_nonzero(out != BG)) != panel_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
