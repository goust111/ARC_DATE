from __future__ import annotations

import random

import numpy as np


ARC_GRID_MAX = 30
GENERATION_ATTEMPTS = 400
PANEL_ATTEMPTS = 120
SEPARATOR_COLOR = 5


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _count_components(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _axis_layout_candidates(length: int, separators: list[int]) -> list[tuple[int, int, bool]]:
    separators = sorted(separators)
    count = len(separators)
    candidates: list[tuple[int, int, bool]] = []

    panel_count = count + 1
    if panel_count >= 2:
        numer = length - count
        if numer > 0 and numer % panel_count == 0:
            panel_size = numer // panel_count
            expected = [panel_size + i * (panel_size + 1) for i in range(count)]
            if panel_size >= 1 and expected == separators:
                candidates.append((panel_size, panel_count, False))

    if count >= 3:
        panel_count = count - 1
        numer = length - count
        if panel_count >= 2 and numer > 0 and numer % panel_count == 0:
            panel_size = numer // panel_count
            expected = [i * (panel_size + 1) for i in range(count)]
            if panel_size >= 1 and expected == separators:
                candidates.append((panel_size, panel_count, True))

    return candidates


def _panel_slices(panel_size: int, panel_count: int, bordered: bool) -> list[slice]:
    start = 1 if bordered else 0
    return [
        slice(start + i * (panel_size + 1), start + i * (panel_size + 1) + panel_size)
        for i in range(panel_count)
    ]


def _find_panel_layout(grid: np.ndarray) -> tuple[int, list[slice], list[slice]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    matches: list[tuple[int, list[slice], list[slice]]] = []
    for color in map(int, np.unique(g)):
        full_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not full_rows or not full_cols:
            continue

        row_layouts = _axis_layout_candidates(h, full_rows)
        col_layouts = _axis_layout_candidates(w, full_cols)
        for row_size, row_count, row_bordered in row_layouts:
            for col_size, col_count, col_bordered in col_layouts:
                if row_size != col_size:
                    continue
                panel_size = row_size
                row_slices = _panel_slices(panel_size, row_count, row_bordered)
                col_slices = _panel_slices(panel_size, col_count, col_bordered)
                matches.append((color, row_slices, col_slices))

    deduped: list[tuple[int, list[slice], list[slice]]] = []
    seen = set()
    for color, row_slices, col_slices in matches:
        key = (
            color,
            tuple((s.start, s.stop) for s in row_slices),
            tuple((s.start, s.stop) for s in col_slices),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((color, row_slices, col_slices))

    assert len(deduped) == 1, "expected exactly one separator layout"
    return deduped[0]


def _panel_colors(panel: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(panel, return_counts=True)
    order = np.argsort(-counts)
    values = values[order]
    counts = counts[order]
    assert len(values) == 2, "each panel must contain one background and one foreground color"
    assert counts[0] > counts[1], "background must be uniquely dominant inside each panel"
    bg = int(values[0])
    fg = int(values[1])
    return bg, fg


def _stripe_rows(size: int, count: int) -> np.ndarray:
    assert 1 <= count <= size
    rows = np.floor((np.arange(count) + 0.5) * size / count).astype(int)
    assert len(np.unique(rows)) == count
    return rows


def _render_panel(size: int, bg: int, fg: int, component_count: int) -> np.ndarray:
    panel = np.full((size, size), bg, dtype=int)
    if component_count == 1:
        panel[:, :] = fg
        return panel

    for row in _stripe_rows(size, component_count):
        panel[int(row), :] = fg
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each panel by a connectivity summary.

    The separator grid is preserved. A panel whose foreground has one 4-connected
    component becomes a solid block; otherwise it becomes horizontal stripes,
    with one stripe per 4-connected component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator, row_slices, col_slices = _find_panel_layout(g)
    panel_size = row_slices[0].stop - row_slices[0].start
    out = np.full_like(g, separator)

    for rs in row_slices:
        for cs in col_slices:
            panel = g[rs, cs]
            assert panel.shape == (panel_size, panel_size)
            bg, fg = _panel_colors(panel)
            component_count = _count_components(panel == fg)
            assert component_count >= 1
            rendered = _render_panel(panel_size, bg, fg, component_count)
            out[rs, cs] = rendered

    return out


def _grow_connected_mask(height: int, width: int, max_area: int) -> np.ndarray:
    area = random.randint(1, max_area)
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = [start]

    while int(mask.sum()) < area:
        cr, cc = random.choice(cells)
        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(cr, cc, height, width)
            if not mask[nr, nc]
        ]
        if not neighbors:
            frontier = []
            for cell in cells:
                frontier.extend(
                    (nr, nc)
                    for nr, nc in _neighbors4(cell[0], cell[1], height, width)
                    if not mask[nr, nc]
                )
            if not frontier:
                break
            nr, nc = random.choice(frontier)
        else:
            nr, nc = random.choice(neighbors)
        mask[nr, nc] = True
        cells.append((nr, nc))

    return mask


def _make_single_component_panel(size: int, bg: int, fg: int) -> np.ndarray | None:
    area_limit = (size * size) // 2
    for _ in range(PANEL_ATTEMPTS):
        box_h = random.randint(1, size)
        box_w = random.randint(1, size)
        r0 = random.randint(0, size - box_h)
        c0 = random.randint(0, size - box_w)
        max_area = min(box_h * box_w, area_limit)
        if max_area < 1:
            continue

        panel = np.full((size, size), bg, dtype=int)
        mask = _grow_connected_mask(box_h, box_w, max_area)
        panel[r0 : r0 + box_h, c0 : c0 + box_w][mask] = fg

        fg_area = int((panel == fg).sum())
        if not (1 <= fg_area <= area_limit):
            continue
        if _count_components(panel == fg) == 1:
            return panel
    return None


def _make_boxed_multi_panel(size: int, bg: int, fg: int, count: int) -> np.ndarray | None:
    area_limit = (size * size) // 2
    for _ in range(PANEL_ATTEMPTS):
        split_r = random.randint(1, size - 2)
        split_c = random.randint(1, size - 2)
        slots = [
            (0, split_r, 0, split_c),
            (0, split_r, split_c + 1, size),
            (split_r + 1, size, 0, split_c),
            (split_r + 1, size, split_c + 1, size),
        ]
        slots = [slot for slot in slots if slot[1] > slot[0] and slot[3] > slot[2]]
        if len(slots) < count:
            continue

        panel = np.full((size, size), bg, dtype=int)
        total_area = 0
        for r0, r1, c0, c1 in random.sample(slots, count):
            height = r1 - r0
            width = c1 - c0
            max_area = min(height * width, max(1, size // 2 + 1))
            mask = _grow_connected_mask(height, width, max_area)
            total_area += int(mask.sum())
            panel[r0:r1, c0:c1][mask] = fg

        if total_area > area_limit:
            continue
        if _count_components(panel == fg) == count:
            return panel
    return None


def _make_diagonal_multi_panel(size: int, bg: int, fg: int, count: int) -> np.ndarray | None:
    if count > size:
        return None
    panel = np.full((size, size), bg, dtype=int)
    if random.choice([True, False]):
        r0 = random.randint(0, size - count)
        c0 = random.randint(0, size - count)
        for i in range(count):
            panel[r0 + i, c0 + i] = fg
    else:
        r0 = random.randint(0, size - count)
        c0 = random.randint(count - 1, size - 1)
        for i in range(count):
            panel[r0 + i, c0 - i] = fg
    if _count_components(panel == fg) == count:
        return panel
    return None


def _make_multi_component_panel(size: int, bg: int, fg: int, count: int) -> np.ndarray | None:
    builders = [_make_boxed_multi_panel]
    if count >= 2:
        builders.append(_make_diagonal_multi_panel)
    for _ in range(PANEL_ATTEMPTS):
        builder = random.choice(builders)
        panel = builder(size, bg, fg, count)
        if panel is None:
            continue
        fg_area = int((panel == fg).sum())
        if not (count <= fg_area <= (size * size) // 2):
            continue
        if _count_components(panel == fg) == count:
            return panel
    return None


def _grid_shape(
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    border_rows: bool,
    border_cols: bool,
) -> tuple[int, int]:
    row_separators = panel_rows + 1 if border_rows else panel_rows - 1
    col_separators = panel_cols + 1 if border_cols else panel_cols - 1
    return panel_rows * panel_size + row_separators, panel_cols * panel_size + col_separators


def _assemble_grid(
    panels: list[np.ndarray],
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    border_rows: bool,
    border_cols: bool,
    separator: int,
) -> np.ndarray:
    height, width = _grid_shape(panel_rows, panel_cols, panel_size, border_rows, border_cols)
    grid = np.full((height, width), separator, dtype=int)
    row_slices = _panel_slices(panel_size, panel_rows, border_rows)
    col_slices = _panel_slices(panel_size, panel_cols, border_cols)

    idx = 0
    for rs in row_slices:
        for cs in col_slices:
            grid[rs, cs] = panels[idx]
            idx += 1
    return grid


def _build_expected_output(
    panel_rows: int,
    panel_cols: int,
    panel_size: int,
    border_rows: bool,
    border_cols: bool,
    separator: int,
    bg: int,
    colors: list[int],
    counts: list[int],
) -> np.ndarray:
    panels = [
        _render_panel(panel_size, bg, color, count)
        for color, count in zip(colors, counts, strict=True)
    ]
    return _assemble_grid(
        panels,
        panel_rows,
        panel_cols,
        panel_size,
        border_rows,
        border_cols,
        separator,
    )


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.randint(2, 4)
    panel_cols = random.randint(2, 4)
    border_rows = random.choice([False, True])
    border_cols = random.choice([False, True])

    row_separators = panel_rows + 1 if border_rows else panel_rows - 1
    col_separators = panel_cols + 1 if border_cols else panel_cols - 1
    max_size = min(
        8,
        (ARC_GRID_MAX - row_separators) // panel_rows,
        (ARC_GRID_MAX - col_separators) // panel_cols,
    )
    if max_size < 5:
        return None

    panel_size = random.randint(5, max_size)
    max_components = min(4, panel_size - 1)
    if max_components < 2:
        return None

    palette = [color for color in range(10) if color != SEPARATOR_COLOR]
    bg = random.choice(palette)
    fg_palette = [color for color in palette if color != bg]

    total_panels = panel_rows * panel_cols
    counts = []
    for _ in range(total_panels):
        if random.choice([False, True]):
            counts.append(1)
        else:
            counts.append(random.randint(2, max_components))
    if 1 not in counts:
        counts[random.randrange(total_panels)] = 1
    if not any(count >= 2 for count in counts):
        replace_idx = random.randrange(total_panels)
        counts[replace_idx] = random.randint(2, max_components)

    colors = [random.choice(fg_palette) for _ in range(total_panels)]
    panels: list[np.ndarray] = []
    for color, count in zip(colors, counts, strict=True):
        if count == 1:
            panel = _make_single_component_panel(panel_size, bg, color)
        else:
            panel = _make_multi_component_panel(panel_size, bg, color, count)
        if panel is None:
            return None
        panels.append(panel)

    inp = _assemble_grid(
        panels,
        panel_rows,
        panel_cols,
        panel_size,
        border_rows,
        border_cols,
        SEPARATOR_COLOR,
    )
    expected = _build_expected_output(
        panel_rows,
        panel_cols,
        panel_size,
        border_rows,
        border_cols,
        SEPARATOR_COLOR,
        bg,
        colors,
        counts,
    )

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
