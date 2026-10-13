from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = [2, 3, 4, 6, 7, 8, 9]
LAYOUTS = [(2, 2), (2, 3), (3, 2)]
MAX_GENERATION_ATTEMPTS = 300

BASE_SHAPES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(rot):
        out = {(c, -r) for r, c in out}
    out = _normalize(out)
    if flip:
        max_c = max(c for _, c in out)
        out = {(r, max_c - c) for r, c in out}
    return _normalize(out)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _count_components(mask: np.ndarray) -> int:
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _separator_axes(g: np.ndarray) -> tuple[list[int], list[int], int]:
    sep_rows = [r for r in range(g.shape[0]) if np.all((g[r, :] != BLACK) & (g[r, :] != BLUE))]
    sep_cols = [c for c in range(g.shape[1]) if np.all((g[:, c] != BLACK) & (g[:, c] != BLUE))]
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3, "expected a bordered separator lattice"
    assert sep_rows[0] == 0 and sep_rows[-1] == g.shape[0] - 1
    assert sep_cols[0] == 0 and sep_cols[-1] == g.shape[1] - 1

    row_steps = [b - a for a, b in zip(sep_rows, sep_rows[1:])]
    col_steps = [b - a for a, b in zip(sep_cols, sep_cols[1:])]
    assert len(set(row_steps)) == 1 and row_steps[0] >= 2, "panel row spacing must be constant"
    assert len(set(col_steps)) == 1 and col_steps[0] >= 2, "panel col spacing must be constant"

    boundary_mask = np.zeros_like(g, dtype=bool)
    boundary_mask[sep_rows, :] = True
    boundary_mask[:, sep_cols] = True
    values, counts = np.unique(g[boundary_mask], return_counts=True)
    grey = int(values[np.argmax(counts)])
    assert grey not in (BLACK, BLUE), "separator color must be distinct from black and blue"
    return sep_rows, sep_cols, grey


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each panel by its border marker, then overlay the panels from most objects to fewest.

    Input structure:
    - Black cells form the transparent background and blue cells form the panel objects.
    - Grey separator rows and columns, together with a grey outer frame, split the grid into equal rectangular panels.
    - Each panel has exactly one non-grey marker pixel on one of its border segments.

    Rule:
    - For every panel, count its blue 4-connected objects and recolor every blue cell to that panel's border-marker color.
    - Create an output grid with the size of one panel interior, removing all separator lines.
    - Stack the recolored panels so the panel with the most objects is placed first and the panel with the fewest objects is placed last; black cells stay transparent and do not overwrite lower layers.

    Key insight:
    - The marker determines the per-panel recoloring, while the number of disconnected blue objects determines the visibility order in the final composite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows, sep_cols, grey = _separator_axes(g)
    panel_h = sep_rows[1] - sep_rows[0] - 1
    panel_w = sep_cols[1] - sep_cols[0] - 1
    panels = []

    for ri, (r0, r1) in enumerate(zip(sep_rows, sep_rows[1:])):
        for ci, (c0, c1) in enumerate(zip(sep_cols, sep_cols[1:])):
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            assert interior.shape == (panel_h, panel_w), "all panels must share one size"
            assert np.isin(interior, [BLACK, BLUE]).all(), "panel interiors may only contain black or blue"

            border_cells = np.concatenate(
                [
                    g[r0, c0 + 1 : c1],
                    g[r1, c0 + 1 : c1],
                    g[r0 + 1 : r1, c0],
                    g[r0 + 1 : r1, c1],
                ]
            )
            marker_values = [int(v) for v in border_cells if int(v) != grey]
            assert len(marker_values) == 1, f"panel {(ri, ci)} must have exactly one marker"
            marker = marker_values[0]
            assert marker not in (BLACK, BLUE), "marker must be a visible recolor"

            component_count = _count_components(interior == BLUE)
            assert component_count >= 1, "each panel must contain at least one blue object"

            recolored = np.where(interior == BLUE, marker, BLACK)
            panels.append({"count": component_count, "grid": recolored})

    counts = [panel["count"] for panel in panels]
    assert len(set(counts)) == len(counts), "panel object counts must be unique for an unambiguous stack"

    out = np.full((panel_h, panel_w), BLACK, dtype=int)
    for panel in sorted(panels, key=lambda item: item["count"], reverse=True):
        mask = panel["grid"] != BLACK
        out[mask] = panel["grid"][mask]
    return out


def _slot_shape(slot_h: int, slot_w: int) -> set[tuple[int, int]]:
    candidates = []
    for base in BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                shape = _transform_shape(base, rot, flip)
                h, w = _shape_dims(shape)
                if h <= slot_h and w <= slot_w:
                    candidates.append(shape)
    assert candidates
    shape = set(random.choice(candidates))
    h, w = _shape_dims(shape)
    row_offset = random.randint(0, slot_h - h)
    col_offset = random.randint(0, slot_w - w)
    return {(r + row_offset, c + col_offset) for r, c in shape}


def _choose_object_grid(num_objects: int) -> tuple[int, int]:
    if num_objects == 4:
        return 2, 2
    return random.choice([(2, 3), (3, 2)])


def _panel_size_bounds(rows: int, cols: int) -> tuple[int, int]:
    return (30 - (rows + 1)) // rows, (30 - (cols + 1)) // cols


def _build_object_library(num_objects: int, rows: int, cols: int) -> tuple[int, int, list[set[tuple[int, int]]]] | None:
    slot_rows, slot_cols = _choose_object_grid(num_objects)
    max_panel_h, max_panel_w = _panel_size_bounds(rows, cols)

    if slot_rows == 2 and slot_cols == 3:
        slot_h = random.choice([2, 3])
        slot_w = 2
    elif slot_rows == 3 and slot_cols == 2:
        slot_h = 2
        slot_w = random.choice([2, 3])
    else:
        slot_h = random.choice([2, 3])
        slot_w = random.choice([2, 3])

    row_gaps = [random.choice([1, 2]) for _ in range(slot_rows - 1)]
    col_gaps = [random.choice([1, 2]) for _ in range(slot_cols - 1)]
    min_h = slot_rows * slot_h + sum(row_gaps)
    min_w = slot_cols * slot_w + sum(col_gaps)
    if min_h > max_panel_h or min_w > max_panel_w:
        return None

    panel_h = random.randint(min_h, min(max_panel_h, min_h + 2))
    panel_w = random.randint(min_w, min(max_panel_w, min_w + 2))
    top_margin = random.randint(0, panel_h - min_h)
    left_margin = random.randint(0, panel_w - min_w)

    row_starts = [top_margin]
    for gap in row_gaps:
        row_starts.append(row_starts[-1] + slot_h + gap)
    col_starts = [left_margin]
    for gap in col_gaps:
        col_starts.append(col_starts[-1] + slot_w + gap)

    slots = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    random.shuffle(slots)
    slots = slots[:num_objects]

    objects = []
    for r0, c0 in slots:
        shape = _slot_shape(slot_h, slot_w)
        objects.append({(r0 + r, c0 + c) for r, c in shape})
    return panel_h, panel_w, objects


def _available_outer_sides(panel_r: int, panel_c: int, rows: int, cols: int) -> list[str]:
    sides = []
    if panel_r == 0:
        sides.append("top")
    if panel_r == rows - 1:
        sides.append("bottom")
    if panel_c == 0:
        sides.append("left")
    if panel_c == cols - 1:
        sides.append("right")
    return sides


def _place_marker(grid: np.ndarray, panel_r: int, panel_c: int, rows: int, cols: int, panel_h: int, panel_w: int, color: int) -> None:
    r0 = panel_r * (panel_h + 1)
    c0 = panel_c * (panel_w + 1)
    r1 = r0 + panel_h + 1
    c1 = c0 + panel_w + 1
    side = random.choice(_available_outer_sides(panel_r, panel_c, rows, cols))

    if side == "top":
        grid[r0, random.randint(c0 + 1, c1 - 1)] = color
    elif side == "bottom":
        grid[r1, random.randint(c0 + 1, c1 - 1)] = color
    elif side == "left":
        grid[random.randint(r0 + 1, r1 - 1), c0] = color
    else:
        grid[random.randint(r0 + 1, r1 - 1), c1] = color


def _panel_from_objects(panel_h: int, panel_w: int, objects: list[set[tuple[int, int]]], count: int) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    for shape in objects[:count]:
        for r, c in shape:
            panel[r, c] = BLUE
    return panel


def _expected_output(panel_h: int, panel_w: int, objects: list[set[tuple[int, int]]], count_by_panel: list[int], color_by_panel: list[int]) -> np.ndarray:
    out = np.full((panel_h, panel_w), BLACK, dtype=int)
    ordered = sorted(zip(count_by_panel, color_by_panel), key=lambda item: item[0], reverse=True)
    for count, color in ordered:
        panel = _panel_from_objects(panel_h, panel_w, objects, count)
        out[panel == BLUE] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    num_panels = rows * cols
    colors = random.sample(MARKER_COLORS, num_panels)
    count_by_panel = random.sample(list(range(1, num_panels + 1)), num_panels)

    library = _build_object_library(num_panels, rows, cols)
    if library is None:
        return None
    panel_h, panel_w, objects = library

    grid_h = rows * panel_h + rows + 1
    grid_w = cols * panel_w + cols + 1
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), GREY, dtype=int)
    for panel_index, count in enumerate(count_by_panel):
        pr, pc = divmod(panel_index, cols)
        r0 = pr * (panel_h + 1) + 1
        c0 = pc * (panel_w + 1) + 1
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = _panel_from_objects(panel_h, panel_w, objects, count)
        _place_marker(grid, pr, pc, rows, cols, panel_h, panel_w, colors[panel_index])

    expected = _expected_output(panel_h, panel_w, objects, count_by_panel, colors)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
