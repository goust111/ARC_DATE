import random

import numpy as np


BG = 0
SEP = 5
RED = 2


def _intervals_from_separator_mask(is_sep: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(is_sep.tolist()):
        if not flag and start is None:
            start = idx
        elif flag and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(is_sep)))
    return intervals


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    mask[rr, cc] = True
    return mask


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    main_diag = bool(mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T))
    anti_diag = bool(
        mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.fliplr(mask).T)
    )
    return horizontal, vertical, main_diag, anti_diag


def _unique_axis(mask: np.ndarray) -> str | None:
    horizontal, vertical, main_diag, anti_diag = _symmetry_flags(mask)
    if main_diag or anti_diag:
        return None
    if horizontal ^ vertical:
        return "H" if horizontal else "V"
    return None


def _panel_intervals(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_intervals = _intervals_from_separator_mask(np.all(grid == SEP, axis=1))
    col_intervals = _intervals_from_separator_mask(np.all(grid == SEP, axis=0))
    assert row_intervals and col_intervals, "grid must contain at least one panel"
    assert len({r1 - r0 for r0, r1 in row_intervals}) == 1, "panel heights must match"
    assert len({c1 - c0 for c0, c1 in col_intervals}) == 1, "panel widths must match"
    return row_intervals, col_intervals


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_intervals, col_intervals = _panel_intervals(g)
    out = g.copy()

    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            panel = g[r0:r1, c0:c1]
            assert not np.any(panel == SEP), "separator color cannot appear inside panels"
            colors = [int(color) for color in np.unique(panel) if int(color) != BG]
            assert len(colors) == 1, "each panel must contain exactly one foreground color"
            obj_color = colors[0]
            mask = panel == obj_color
            comps = _components(mask)
            assert len(comps) == 1, "panel object must be a single 4-connected component"

            axis = _unique_axis(_coords_to_local_mask(comps[0]))
            assert axis is not None, "object must have exactly one horizontal or vertical mirror axis"

            reflected = np.flipud(mask) if axis == "H" else np.fliplr(mask)
            panel_out = panel.copy()
            panel_out[reflected & ~mask] = RED
            out[r0:r1, c0:c1] = panel_out

    return out


# --------------- generator ---------------

PANEL_ROWS_RANGE = (1, 3)
PANEL_COLS_RANGE = (1, 4)
PANEL_HEIGHT_RANGE = (7, 11)
PANEL_WIDTH_RANGE = (7, 11)
OBJECT_COLORS = [1, 3, 4, 6, 7, 8, 9]
AXES = ("H", "V")

SHAPE_RETRIES = 80
PANEL_RETRIES = 80
GEN_RETRIES = 400


def _random_ext_array(length: int, max_ext: int) -> np.ndarray:
    ext = np.zeros(length, dtype=int)
    active = random.randint(1, max(1, min(length, 4)))
    for idx in random.sample(range(length), active):
        ext[idx] = random.randint(1, max_ext)
    return ext


def _make_h_mask(max_h: int, max_w: int) -> np.ndarray | None:
    heights = [h for h in range(3, max_h + 1, 2)]
    widths = [w for w in range(3, max_w + 1)]
    if not heights or not widths:
        return None

    for _ in range(SHAPE_RETRIES):
        h = random.choice(heights)
        w = random.choice(widths)
        center = h // 2
        ext = _random_ext_array(w, center)
        if np.array_equal(ext, ext[::-1]):
            asym_cols = [c for c in range(w) if c != w - 1 - c]
            if not asym_cols:
                continue
            col = random.choice(asym_cols)
            choices = [value for value in range(center + 1) if value != ext[w - 1 - col]]
            if not choices:
                continue
            ext[col] = random.choice(choices)
            if not ext.any():
                ext[col] = random.randint(1, center)

        mask = np.zeros((h, w), dtype=bool)
        mask[center, :] = True
        for c, value in enumerate(ext.tolist()):
            if value:
                mask[center - value : center + value + 1, c] = True
        if _unique_axis(mask) == "H":
            return mask
    return None


def _make_v_mask(max_h: int, max_w: int) -> np.ndarray | None:
    heights = [h for h in range(3, max_h + 1)]
    widths = [w for w in range(3, max_w + 1, 2)]
    if not heights or not widths:
        return None

    for _ in range(SHAPE_RETRIES):
        h = random.choice(heights)
        w = random.choice(widths)
        center = w // 2
        ext = _random_ext_array(h, center)
        if np.array_equal(ext, ext[::-1]):
            asym_rows = [r for r in range(h) if r != h - 1 - r]
            if not asym_rows:
                continue
            row = random.choice(asym_rows)
            choices = [value for value in range(center + 1) if value != ext[h - 1 - row]]
            if not choices:
                continue
            ext[row] = random.choice(choices)
            if not ext.any():
                ext[row] = random.randint(1, center)

        mask = np.zeros((h, w), dtype=bool)
        mask[:, center] = True
        for r, value in enumerate(ext.tolist()):
            if value:
                mask[r, center - value : center + value + 1] = True
        if _unique_axis(mask) == "V":
            return mask
    return None


def _place_mask(panel_h: int, panel_w: int, mask: np.ndarray, axis: str) -> np.ndarray | None:
    h, w = mask.shape
    row_choices = list(range(panel_h - h + 1))
    col_choices = list(range(panel_w - w + 1))
    if axis == "H":
        row_choices = [r for r in row_choices if 2 * r != panel_h - h]
    else:
        col_choices = [c for c in col_choices if 2 * c != panel_w - w]
    if not row_choices or not col_choices:
        return None

    r0 = random.choice(row_choices)
    c0 = random.choice(col_choices)
    full = np.zeros((panel_h, panel_w), dtype=bool)
    full[r0 : r0 + h, c0 : c0 + w] = mask
    reflected = np.flipud(full) if axis == "H" else np.fliplr(full)
    if not np.any(reflected & ~full):
        return None
    return full


def _build_panel(panel_h: int, panel_w: int, axis: str, color: int) -> np.ndarray | None:
    max_h = panel_h - 1
    max_w = panel_w - 1
    for _ in range(PANEL_RETRIES):
        mask = _make_h_mask(max_h, max_w) if axis == "H" else _make_v_mask(max_h, max_w)
        if mask is None:
            return None
        placed = _place_mask(panel_h, panel_w, mask, axis)
        if placed is None:
            continue
        panel = np.full((panel_h, panel_w), BG, dtype=int)
        panel[placed] = color
        return panel
    return None


def _sample_grid_shape() -> tuple[int, int]:
    while True:
        nr = random.randint(*PANEL_ROWS_RANGE)
        nc = random.randint(*PANEL_COLS_RANGE)
        if nr * nc >= 2:
            return nr, nc


def _sample_axes(count: int) -> list[str]:
    axes = [random.choice(AXES) for _ in range(count)]
    if count > 1 and len(set(axes)) == 1:
        axes[random.randrange(count)] = "H" if axes[0] == "V" else "V"
    return axes


def _assemble_grid(
    panel_rows: int, panel_cols: int, panel_h: int, panel_w: int, panels: list[np.ndarray]
) -> np.ndarray:
    stride_h = panel_h + 1
    stride_w = panel_w + 1
    grid_h = panel_rows * panel_h + (panel_rows - 1)
    grid_w = panel_cols * panel_w + (panel_cols - 1)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    if panel_rows > 1:
        grid[panel_h::stride_h, :] = SEP
    if panel_cols > 1:
        grid[:, panel_w::stride_w] = SEP

    idx = 0
    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * stride_h
            c0 = pc * stride_w
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panels[idx]
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = _sample_grid_shape()
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    axes = _sample_axes(panel_rows * panel_cols)

    panels: list[np.ndarray] = []
    for axis in axes:
        color = random.choice(OBJECT_COLORS)
        panel = _build_panel(panel_h, panel_w, axis, color)
        if panel is None:
            return None
        panels.append(panel)

    grid = _assemble_grid(panel_rows, panel_cols, panel_h, panel_w, panels)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
