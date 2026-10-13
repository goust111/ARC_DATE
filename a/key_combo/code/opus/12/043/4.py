from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

PANEL_SIZE_OPTIONS = (4, 5)
ROW_COL_OPTIONS = ((3, 2), (3, 3), (4, 2))  # (panel_rows, data_cols)
GEN_RETRIES = 400
MASK_RETRIES = 120


def _split_ranges(separator_lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(separator_lines) >= 2
    assert separator_lines[0] == 0
    assert separator_lines[-1] == limit - 1
    ranges: list[tuple[int, int]] = []
    for a, b in zip(separator_lines, separator_lines[1:]):
        assert b > a + 1
        ranges.append((a + 1, b))
    return ranges


def _panel_edges(panel: np.ndarray) -> np.ndarray:
    return np.concatenate([panel[0, :], panel[-1, :], panel[1:-1, 0], panel[1:-1, -1]])


def _build_output_panel(
    mask: np.ndarray, swatch_color: int, frame_color: int, band_height: int
) -> np.ndarray:
    interior_h, interior_w = mask.shape
    panel = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    panel[0, :] = frame_color
    panel[-1, :] = frame_color
    panel[:, 0] = frame_color
    panel[:, -1] = frame_color
    for r in range(interior_h):
        band_color = swatch_color if (r // band_height) % 2 == 0 else BLACK
        row = panel[1 + r, 1:-1]
        row[mask[r]] = band_color
    return panel


def _marker_height(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MAROON)
    assert len(coords) > 0, "expected a maroon marker segment"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, "marker must be one column wide"
    marker_col = int(cols[0])
    rows = np.sort(coords[:, 0])
    expected = np.arange(len(rows))
    assert np.array_equal(rows, expected), "marker must be top-aligned and contiguous"
    assert np.all(grid[len(rows) :, marker_col] == GREY), "marker column below the segment must be grey"
    return marker_col, int(len(rows))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each data column as a single bordered panel whose template mask reveals a cyclic swatch/black stripe fill.

    Input structure:
    - Grey separator rows and columns partition the left table into equal rectangular panels.
    - The first panel column is a row-label column: each label panel is uniformly filled with one label color.
    - In every remaining panel column, exactly one panel is a uniform solid swatch color, while the other panels show the same template color mask on a black panel interior.
    - One maroon vertical segment sits in the extra rightmost column, top-aligned on a grey background; its height is the stripe run length.

    Rule:
    - Read the maroon segment height H.
    - For each data column, find its unique solid swatch panel, read its swatch color C, and take the row label color F from the same row in the label column.
    - Read the shared template mask M from the other panels in that column.
    - Build one output panel with frame color F; inside the frame, alternate H rows of color C, then H rows of black, repeating downward.
    - Keep those stripe colors only where M is foreground, turn the remaining interior cells black, and output one rebuilt panel per data column with one grey separator column between panels.

    Key insight:
    The result is determined jointly by three independent controls: the column's template mask, the solid swatch color, and the maroon marker height that sets the stripe bandwidth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_col, band_height = _marker_height(g)
    table = g[:, :marker_col]
    assert table.shape[1] >= 3

    separator_rows = [r for r in range(table.shape[0]) if np.all(table[r, :] == GREY)]
    separator_cols = [c for c in range(table.shape[1]) if np.all(table[:, c] == GREY)]
    row_ranges = _split_ranges(separator_rows, table.shape[0])
    col_ranges = _split_ranges(separator_cols, table.shape[1])
    assert len(row_ranges) >= 3
    assert len(col_ranges) >= 3, "need one label column plus at least two data columns"

    row_heights = {r1 - r0 for r0, r1 in row_ranges}
    col_widths = {c1 - c0 for c0, c1 in col_ranges}
    assert len(row_heights) == 1 and len(col_widths) == 1
    panel_h = row_heights.pop()
    panel_w = col_widths.pop()
    assert panel_h == panel_w and panel_h >= 4

    labels: list[int] = []
    template_colors: set[int] = set()
    parsed_columns: list[list[tuple[str, int, np.ndarray | None]]] = []

    for r0, r1 in row_ranges:
        label_panel = table[r0:r1, col_ranges[0][0] : col_ranges[0][1]]
        label_values = set(map(int, np.unique(label_panel)))
        assert len(label_values) == 1
        label = next(iter(label_values))
        assert label not in {BLACK, GREY, MAROON}
        labels.append(label)

        row_entries: list[tuple[str, int, np.ndarray | None]] = []
        for c0, c1 in col_ranges[1:]:
            panel = table[r0:r1, c0:c1]
            values = set(map(int, np.unique(panel)))
            if len(values) == 1:
                solid_color = next(iter(values))
                assert solid_color not in {BLACK, GREY, MAROON}
                row_entries.append(("solid", solid_color, None))
                continue

            assert GREY not in values and MAROON not in values
            assert BLACK in values and len(values) == 2
            template_color = next(v for v in values if v != BLACK)
            template_colors.add(template_color)
            edges = _panel_edges(panel)
            assert np.all(edges == BLACK), "template panel should keep its border area black"
            mask = panel[1:-1, 1:-1] == template_color
            assert np.any(mask), "template mask cannot be empty"
            row_entries.append(("mask", template_color, mask))
        parsed_columns.append(row_entries)

    assert len(template_colors) == 1

    data_cols = len(col_ranges) - 1
    out = np.full((panel_h, data_cols * panel_w + (data_cols - 1)), GREY, dtype=int)

    for ci in range(data_cols):
        solid_entries: list[tuple[int, int]] = []
        masks: list[np.ndarray] = []
        for ri in range(len(row_ranges)):
            kind, color, mask = parsed_columns[ri][ci]
            if kind == "solid":
                solid_entries.append((ri, color))
            else:
                assert mask is not None
                masks.append(mask)

        assert len(solid_entries) == 1
        assert len(masks) >= 2
        base_mask = masks[0]
        for mask in masks[1:]:
            assert np.array_equal(mask, base_mask), "template masks within a column must match"

        swatch_row, swatch_color = solid_entries[0]
        frame_color = labels[swatch_row]
        panel = _build_output_panel(base_mask, swatch_color, frame_color, band_height)
        c0 = ci * (panel_w + 1)
        out[:, c0 : c0 + panel_w] = panel

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _connected_mask(shape: tuple[int, int], size: int) -> np.ndarray | None:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = {start}
    while len(cells) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c, h, w))
        frontier -= cells
        if not frontier:
            return None
        cell = random.choice(list(frontier))
        mask[cell] = True
        cells.add(cell)
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _valid_mask(mask: np.ndarray, band_height: int) -> bool:
    if mask.sum() < 3:
        return False
    occupied_rows = np.where(mask.any(axis=1))[0]
    if len(occupied_rows) < 2:
        return False
    band_types = {(int(r) // band_height) % 2 for r in occupied_rows}
    if 0 not in band_types:
        return False
    if mask.shape[0] > band_height and len(band_types) < 2:
        return False
    return True


def _make_template_panel(mask: np.ndarray, template_color: int) -> np.ndarray:
    panel = np.full((mask.shape[0] + 2, mask.shape[1] + 2), BLACK, dtype=int)
    panel[1:-1, 1:-1][mask] = template_color
    return panel


def _semantic_checks(
    expected_panels: list[np.ndarray],
    masks: list[np.ndarray],
    labels: list[int],
    swatches: list[int],
    swatch_rows: list[int],
    band_height: int,
) -> bool:
    for ci, panel in enumerate(expected_panels):
        frame_color = labels[swatch_rows[ci]]
        if not np.all(panel[0, :] == frame_color):
            return False
        if not np.any(panel[1:-1, 1:-1] == swatches[ci]):
            return False

        alt_label = labels[(swatch_rows[ci] + 1) % len(labels)]
        if alt_label != frame_color:
            if np.array_equal(
                _build_output_panel(masks[ci], swatches[ci], alt_label, band_height), panel
            ):
                return False

        alt_swatch = swatches[(ci + 1) % len(swatches)]
        if alt_swatch != swatches[ci]:
            if np.array_equal(
                _build_output_panel(masks[ci], alt_swatch, frame_color, band_height), panel
            ):
                return False

        alt_mask = masks[(ci + 1) % len(masks)]
        if not np.array_equal(alt_mask, masks[ci]):
            if np.array_equal(
                _build_output_panel(alt_mask, swatches[ci], frame_color, band_height), panel
            ):
                return False

    interior_h = masks[0].shape[0]
    alternatives = [h for h in range(1, min(3, interior_h) + 1) if h != band_height]
    if not alternatives:
        return False
    if not any(
        any(
            not np.array_equal(
                _build_output_panel(masks[ci], swatches[ci], labels[swatch_rows[ci]], alt_h),
                expected_panels[ci],
            )
            for ci in range(len(expected_panels))
        )
        for alt_h in alternatives
    ):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    panel_rows, data_cols = random.choice(ROW_COL_OPTIONS)
    panel_size = random.choice(PANEL_SIZE_OPTIONS)
    interior_size = panel_size - 2
    band_height = random.randint(1, min(3, interior_size))

    available = [1, 2, 3, 4, 6, 7, 8]
    random.shuffle(available)
    template_color = available[0]
    label_colors = available[1 : 1 + panel_rows]
    swatch_colors = available[1 + panel_rows : 1 + panel_rows + data_cols]
    assert len(label_colors) == panel_rows and len(swatch_colors) == data_cols

    swatch_rows = random.sample(range(panel_rows), data_cols)
    masks: list[np.ndarray] = []
    seen_masks: set[tuple[tuple[int, int], bytes]] = set()
    max_cells = min(8, interior_size * interior_size)
    for _ in range(data_cols):
        built = None
        for _ in range(MASK_RETRIES):
            size = random.randint(3, max_cells)
            mask = _connected_mask((interior_size, interior_size), size)
            if mask is None or not _valid_mask(mask, band_height):
                continue
            sig = _mask_signature(mask)
            if sig in seen_masks:
                continue
            seen_masks.add(sig)
            built = mask
            break
        if built is None:
            return None
        masks.append(built)

    expected_panels = [
        _build_output_panel(masks[ci], swatch_colors[ci], label_colors[swatch_rows[ci]], band_height)
        for ci in range(data_cols)
    ]
    if not _semantic_checks(
        expected_panels, masks, label_colors, swatch_colors, swatch_rows, band_height
    ):
        return None

    table_h = panel_rows * panel_size + (panel_rows + 1)
    table_w = (data_cols + 1) * panel_size + (data_cols + 2)
    grid = np.full((table_h, table_w + 1), GREY, dtype=int)

    for ri in range(panel_rows):
        r0 = 1 + ri * (panel_size + 1)
        grid[r0 : r0 + panel_size, 1 : 1 + panel_size] = label_colors[ri]
        for ci in range(data_cols):
            c0 = 1 + (ci + 1) * (panel_size + 1)
            if ri == swatch_rows[ci]:
                panel = np.full((panel_size, panel_size), swatch_colors[ci], dtype=int)
            else:
                panel = _make_template_panel(masks[ci], template_color)
            grid[r0 : r0 + panel_size, c0 : c0 + panel_size] = panel

    grid[:band_height, -1] = MAROON

    expected = np.full(
        (panel_size, data_cols * panel_size + (data_cols - 1)), GREY, dtype=int
    )
    for ci, panel in enumerate(expected_panels):
        c0 = ci * (panel_size + 1)
        expected[:, c0 : c0 + panel_size] = panel

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
