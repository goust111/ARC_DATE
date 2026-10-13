from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _panel_ranges(separator_lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(separator_lines) >= 2, "expected at least two separator lines"
    assert separator_lines[0] == 0 and separator_lines[-1] == limit - 1
    ranges = []
    for a, b in zip(separator_lines, separator_lines[1:]):
        assert b > a + 1, "adjacent separator lines would create an empty panel row"
        ranges.append((a + 1, b))
    return ranges


def _column_ranges(separator_cols: list[int], width: int) -> list[tuple[int, int]]:
    assert separator_cols and separator_cols[-1] == width - 1
    assert 0 not in separator_cols, "leftmost column must contain row labels"
    ranges = []
    start = 1
    for c in separator_cols:
        assert c > start, "adjacent separator columns would create an empty panel"
        ranges.append((start, c))
        start = c + 1
    assert start == width, "separator columns must end at the right edge"
    return ranges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine each column's repeated template mask with its single solid color swatch.

    Input structure:
    - The grid is a table of framed panels separated by background rows/columns.
    - The leftmost column gives one solid row-label color for each panel row.
    - In each panel column, exactly one panel has a solid non-background interior,
      while the other panels share the same mask drawn in one template color on
      the background.

    Rule:
    - For every panel column, find the row containing the solid interior swatch.
    - Rebuild one output panel using that row's left label as the frame color.
    - Fill the repeated template mask with the swatch color and leave the mask
      background cells as background; separate rebuilt panels with one
      background column.

    Key insight:
    The table encodes a colorized stencil per column: the solid panel supplies
    the paint color and frame label, while the duplicate template panels supply
    the shape mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == bg)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == bg)]
    row_ranges = _panel_ranges(sep_rows, h)
    col_ranges = _column_ranges(sep_cols, w)

    panel_heights = {r1 - r0 for r0, r1 in row_ranges}
    panel_widths = {c1 - c0 for c0, c1 in col_ranges}
    assert len(panel_heights) == 1 and len(panel_widths) == 1
    panel_h = panel_heights.pop()
    panel_w = panel_widths.pop()
    assert panel_h >= 3 and panel_w >= 3, "panels must have a frame and interior"

    labels: list[int] = []
    panels: list[list[tuple[str, int, np.ndarray | None]]] = []
    border_colors: set[int] = set()
    mask_colors: set[int] = set()

    for r0, r1 in row_ranges:
        label_cells = g[r0:r1, 0]
        assert np.all(label_cells == label_cells[0]), "row label must be constant"
        label = int(label_cells[0])
        assert label != bg, "row label must be distinct from the background"
        labels.append(label)

        row_panels = []
        for c0, c1 in col_ranges:
            panel = g[r0:r1, c0:c1]
            assert panel.shape == (panel_h, panel_w)

            border = np.concatenate(
                [panel[0, :], panel[-1, :], panel[1:-1, 0], panel[1:-1, -1]]
            )
            assert np.all(border == border[0]), "each input panel frame must be solid"
            border_colors.add(int(border[0]))

            interior = panel[1:-1, 1:-1]
            vals = set(map(int, np.unique(interior)))
            if len(vals) == 1:
                color = next(iter(vals))
                assert color != bg, "solid swatch cannot be background"
                row_panels.append(("solid", color, None))
            else:
                assert len(vals) == 2 and bg in vals, "template panel must use bg plus one color"
                color = next(v for v in vals if v != bg)
                mask_colors.add(color)
                row_panels.append(("mask", color, interior == color))
        panels.append(row_panels)

    assert len(border_colors) == 1, "all input panels must use one frame color"
    assert len(mask_colors) == 1, "all template panels must use one mask color"

    out_w = len(col_ranges) * panel_w + len(col_ranges) - 1
    out = np.full((panel_h, out_w), bg, dtype=int)

    for ci in range(len(col_ranges)):
        solid_entries: list[tuple[int, int]] = []
        masks: list[np.ndarray] = []
        for ri in range(len(row_ranges)):
            kind, color, mask = panels[ri][ci]
            if kind == "solid":
                solid_entries.append((ri, color))
            else:
                assert mask is not None
                masks.append(mask)

        assert len(solid_entries) == 1, "each panel column must have one solid swatch"
        assert len(masks) == len(row_ranges) - 1 and masks, "remaining panels must be masks"
        base_mask = masks[0]
        for mask in masks[1:]:
            assert np.array_equal(mask, base_mask), "mask panels in a column must match"

        swatch_row, swatch_color = solid_entries[0]
        frame_color = labels[swatch_row]
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        panel[0, :] = frame_color
        panel[-1, :] = frame_color
        panel[:, 0] = frame_color
        panel[:, -1] = frame_color
        panel[1:-1, 1:-1][base_mask] = swatch_color

        out_c = ci * (panel_w + 1)
        out[:, out_c : out_c + panel_w] = panel

    return out

# --------------- generator (c) ---------------

# --- Configuration ---
_PANEL_ROWS = 3                  # number of panel rows (R)
_PANEL_COLS_CHOICES = [3, 4]     # number of panel columns (C)
_PANEL_SIZE_CHOICES = [5, 6, 7]  # panel side length including frame (N)
_FILL_DENSITY_LO = 0.3           # minimum mask fill fraction
_FILL_DENSITY_HI = 0.7           # maximum mask fill fraction
_MASK_RETRIES = 100              # attempts per column to find a unique mask
_GEN_RETRIES = 300               # outer generation attempts


def _make_framed_panel(N: int, frame_color: int, bg: int,
                       interior: np.ndarray | None = None) -> np.ndarray:
    """Return an NxN panel with a 1-cell frame; interior defaults to bg."""
    panel = np.full((N, N), bg, dtype=int)
    panel[0, :] = panel[-1, :] = frame_color
    panel[:, 0] = panel[:, -1] = frame_color
    if interior is not None:
        panel[1:-1, 1:-1] = interior
    return panel


def _generate_masks(C: int, interior_size: int) -> list[np.ndarray] | None:
    """Generate C distinct binary masks on (interior_size x interior_size) with fill in [30-70%]."""
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    total = interior_size * interior_size
    for _ in range(C):
        for _ in range(_MASK_RETRIES):
            density = random.uniform(_FILL_DENSITY_LO, _FILL_DENSITY_HI)
            m = np.random.random((interior_size, interior_size)) < density
            fill = m.sum()
            if fill < _FILL_DENSITY_LO * total or fill > _FILL_DENSITY_HI * total:
                continue
            if fill == 0:
                continue
            key = m.tobytes()
            if key not in seen:
                seen.add(key)
                masks.append(m)
                break
        else:
            return None
    return masks


def _gen_attempt() -> np.ndarray | None:
    R = _PANEL_ROWS
    C = random.choice(_PANEL_COLS_CHOICES)
    N = random.choice(_PANEL_SIZE_CHOICES)

    # Assign distinct colors: bg, border, mask, R row-labels, C swatches
    total_colors = 3 + R + C
    if total_colors > 10:
        return None
    chosen = random.sample(range(10), total_colors)
    bg, border, mask_color = chosen[0], chosen[1], chosen[2]
    labels = chosen[3:3 + R]
    swatches = chosen[3 + R:]

    masks = _generate_masks(C, N - 2)
    if masks is None:
        return None

    # For each column, pick which row holds the solid swatch
    swatch_rows = [random.randint(0, R - 1) for _ in range(C)]

    # Build input grid: bg separators + label column + R x C framed panels
    grid_h = 1 + R * (N + 1)
    grid_w = 1 + C * (N + 1)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    for ri in range(R):
        r0 = 1 + ri * (N + 1)
        grid[r0:r0 + N, 0] = labels[ri]
        for ci in range(C):
            c0 = 1 + ci * (N + 1)
            if ri == swatch_rows[ci]:
                interior = np.full((N - 2, N - 2), swatches[ci], dtype=int)
            else:
                interior = np.where(masks[ci], mask_color, bg)
            grid[r0:r0 + N, c0:c0 + N] = _make_framed_panel(N, border, bg, interior)

    # Verify with solver
    try:
        out = main(grid)
    except Exception:
        return None

    # Build expected output and cross-check
    out_w = C * N + C - 1
    expected = np.full((N, out_w), bg, dtype=int)
    for ci in range(C):
        frame_color = labels[swatch_rows[ci]]
        interior = np.where(masks[ci], swatches[ci], bg)
        expected[:, ci * (N + 1):ci * (N + 1) + N] = _make_framed_panel(N, frame_color, bg, interior)

    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after %d attempts" % _GEN_RETRIES)
