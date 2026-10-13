from __future__ import annotations

import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, find_components, get_bg, paste


def _regular_lattice(
    components: list[dict],
) -> tuple[list[int], list[int], dict[tuple[int, int], dict], int, int] | None:
    by_anchor: dict[tuple[int, int], dict] = {}
    heights: set[int] = set()
    widths: set[int] = set()

    for component in components:
        b = component["bbox"]
        heights.add(bbox_h(b))
        widths.add(bbox_w(b))
        by_anchor[(b[0], b[1])] = component

    if len(heights) != 1 or len(widths) != 1:
        return None

    height = heights.pop()
    width = widths.pop()
    rows = sorted({r for r, _c in by_anchor})
    cols = sorted({c for _r, c in by_anchor})

    if len(rows) * len(cols) != len(components):
        return None
    if any((r, c) not in by_anchor for r in rows for c in cols):
        return None
    if any(rows[i + 1] - rows[i] != height + 1 for i in range(len(rows) - 1)):
        return None
    if any(cols[i + 1] - cols[i] != width + 1 for i in range(len(cols) - 1)):
        return None

    return rows, cols, by_anchor, height, width


def _find_template_lattice(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int], dict[tuple[int, int], dict], tuple[int, int, int, int]]:
    candidates: list[tuple[int, list[int], list[int], dict[tuple[int, int], dict], tuple[int, int, int, int]]] = []

    for color in colors_present(g, exclude={bg}):
        components = find_components(g, bg=bg, colors=[color], connectivity=4)
        if len(components) <= 1:
            continue

        lattice = _regular_lattice(components)
        if lattice is None:
            continue

        rows, cols, by_anchor, height, width = lattice
        if (height, width) != (3, 3):
            continue

        control_mask = (g != bg) & (g != color)
        control_bbox = bbox(control_mask)
        if control_bbox is None:
            continue

        r0, c0, r1, c1 = control_bbox
        if (r1 - r0 + 1, c1 - c0 + 1) != (len(rows), len(cols)):
            continue
        if not np.all(control_mask[r0 : r1 + 1, c0 : c1 + 1]):
            continue

        candidates.append((color, rows, cols, by_anchor, control_bbox))

    assert len(candidates) == 1, "expected one template lattice with a matching control matrix"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a colored control matrix to recolor a matching lattice of gray 3x3 template masks.

    Input structure
    - The background contains a rectangular lattice of same-color 3x3 template objects, spaced by one background row and column.
    - The remaining foreground pixels form a dense rectangular control matrix whose height and width match the template lattice.
    - Each template object may have holes, but all objects in the lattice share the same 3x3 bbox geometry.

    Rule
    - Match each control-matrix pixel to the template object at the same row and column in the lattice.
    - Recolor only that template object's foreground pixels to the matched control color.
    - Leave background holes, the control matrix, and all other pixels unchanged.

    Key insight
    The colored matrix is a legend indexed by lattice position; it does not draw new shapes, it recolors the existing template masks in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    template_color, rows, cols, by_anchor, control_bbox = _find_template_lattice(g, bg)
    r0, c0, _r1, _c1 = control_bbox

    out = g.copy()
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            paint_color = int(g[r0 + i, c0 + j])
            assert paint_color != bg and paint_color != template_color
            component = by_anchor[(row, col)]
            coords = component["coords"]
            out[coords[:, 0], coords[:, 1]] = paint_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_GRID_SIZE = 23
_GEN_BACKGROUND_COLOR = 0
_GEN_TEMPLATE_COLOR = 5
_GEN_CONTROL_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
# Lattice geometry and sampled dimensions.
_GEN_TEMPLATE_SIZE = 3
_GEN_CELL_GAP = 1
_GEN_CELL_STRIDE = _GEN_TEMPLATE_SIZE + _GEN_CELL_GAP
_GEN_TEMPLATE_AREA = _GEN_TEMPLATE_SIZE * _GEN_TEMPLATE_SIZE
_GEN_VALID_DIMS = [(3, 5), (4, 4), (4, 5), (5, 3), (5, 4)]
# Layout, palette, and retry bounds.
_GEN_BORDER = 1
_GEN_LAYOUT_GAP = 1
_GEN_HOLE_COUNT_RANGE = (1, 2)
_GEN_COLOR_COUNT_RANGE = (2, 5)
_GEN_RETRY_LIMIT = 300


def _gen_template_mask() -> np.ndarray:
    mask = np.ones((_GEN_TEMPLATE_SIZE, _GEN_TEMPLATE_SIZE), dtype=bool)
    hole_count = int(np.random.randint(_GEN_HOLE_COUNT_RANGE[0], _GEN_HOLE_COUNT_RANGE[1] + 1))
    mask.flat[np.random.choice(_GEN_TEMPLATE_AREA, size=hole_count, replace=False)] = False
    return mask


def _gen_control_matrix(rows: int, cols: int) -> np.ndarray:
    color_count = int(np.random.randint(_GEN_COLOR_COUNT_RANGE[0], _GEN_COLOR_COUNT_RANGE[1] + 1))
    palette = np.random.choice(_GEN_CONTROL_COLORS, size=color_count, replace=False)
    flat = np.empty(rows * cols, dtype=int)
    flat[:color_count] = palette
    flat[color_count:] = np.random.choice(palette, size=rows * cols - color_count)
    np.random.shuffle(flat)
    return flat.reshape(rows, cols)


def _gen_layout_choices(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    lattice_h = rows * _GEN_CELL_STRIDE - _GEN_CELL_GAP
    lattice_w = cols * _GEN_CELL_STRIDE - _GEN_CELL_GAP
    choices: list[tuple[int, int, int, int]] = []
    for lat_r in range(_GEN_BORDER, _GEN_GRID_SIZE - lattice_h):
        for lat_c in range(_GEN_BORDER, _GEN_GRID_SIZE - lattice_w):
            for ctl_r in range(_GEN_BORDER, _GEN_GRID_SIZE - rows):
                for ctl_c in range(_GEN_BORDER, lat_c - cols - _GEN_LAYOUT_GAP + 1):
                    choices.append((lat_r, lat_c, ctl_r, ctl_c))
            for ctl_r in range(lat_r + lattice_h + _GEN_LAYOUT_GAP, _GEN_GRID_SIZE - rows):
                for ctl_c in range(_GEN_BORDER, _GEN_GRID_SIZE - cols):
                    choices.append((lat_r, lat_c, ctl_r, ctl_c))
    return choices


def _gen_scene(template: np.ndarray, control: np.ndarray, layout: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return paired input/output grids for one sampled template, control matrix, and layout."""
    lat_r, lat_c, ctl_r, ctl_c = layout
    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    inp = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    input_cell = np.where(template, _GEN_TEMPLATE_COLOR, _GEN_BACKGROUND_COLOR)
    rows, cols = control.shape
    for i in range(rows):
        for j in range(cols):
            rr = lat_r + i * _GEN_CELL_STRIDE
            cc = lat_c + j * _GEN_CELL_STRIDE
            paste(out, np.where(template, control[i, j], _GEN_BACKGROUND_COLOR), rr, cc)
            paste(inp, input_cell, rr, cc)
    paste(out, control, ctl_r, ctl_c)
    paste(inp, control, ctl_r, ctl_c)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _GEN_VALID_DIMS[int(np.random.randint(len(_GEN_VALID_DIMS)))]
    template = _gen_template_mask()
    control = _gen_control_matrix(rows, cols)
    choices = _gen_layout_choices(rows, cols)
    if not choices:
        return None

    layout = choices[int(np.random.randint(len(choices)))]
    inp, out = _gen_scene(template, control, layout)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
