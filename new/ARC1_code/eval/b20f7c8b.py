from __future__ import annotations

import numpy as np

from arc_utils import (
    bbox_coords,
    bbox_h,
    bbox_w,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
)


def _normalize(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]):
    coords = np.asarray(list(coords), dtype=int)
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    shifted = coords - np.array([r0, c0])
    shape = frozenset((int(r), int(c)) for r, c in shifted)
    return shape, (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _d4_variants(shape: frozenset[tuple[int, int]], dims: tuple[int, int]):
    h, w = dims
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (c, h - 1 - r),
        lambda r, c: (h - 1 - r, w - 1 - c),
        lambda r, c: (w - 1 - c, r),
        lambda r, c: (r, w - 1 - c),
        lambda r, c: (h - 1 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (w - 1 - c, h - 1 - r),
    )
    variants = set()
    for transform in transforms:
        variants.add(_normalize([transform(r, c) for r, c in shape]))
    return variants


def _panel(g: np.ndarray, bg: int):
    h, _w = g.shape
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    candidates = [
        comp
        for comp in comps
        if bbox_h(comp["bbox"]) == h and bbox_w(comp["bbox"]) >= 5 and comp["area"] > 25
    ]
    assert len(candidates) == 1
    color = int(candidates[0]["color"])
    box = candidates[0]["bbox"]
    r0, c0, r1, c1 = box
    region = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(region != bg)
    return color, box


def _legend_shapes(g: np.ndarray, panel_color: int, panel_box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = panel_box
    panel_mask = np.zeros(g.shape, dtype=bool)
    panel_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    panel_region = g[r0 : r1 + 1, c0 : c1 + 1]

    legend = {}
    for color in colors_present(panel_region, exclude={panel_color}):
        comps = components_mask((g == color) & panel_mask, connectivity=8)
        assert len(comps) == 1
        shape, dims = _normalize(comps[0])
        assert dims[0] <= 5 and dims[1] <= 5
        legend[int(color)] = (shape, dims)
    assert legend
    return legend, panel_mask


def _block_boxes(g: np.ndarray, bg: int, panel_mask: np.ndarray):
    work = g.copy()
    work[panel_mask] = bg
    comps = find_components(work, bg=bg, connectivity=4, monochromatic=False)
    boxes = []
    for comp in comps:
        box = comp["bbox"]
        assert bbox_h(box) == 5 and bbox_w(box) == 5 and comp["area"] == 25
        boxes.append(box)
    assert boxes
    return sorted(boxes)


def _border_values(block: np.ndarray):
    return np.concatenate((block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]))


def _pattern_colors(g: np.ndarray, block_boxes: list[tuple[int, int, int, int]]):
    pairs = []
    for box in block_boxes:
        block = crop_bbox(g, box)
        colors = set(map(int, block.ravel()))
        if len(colors) != 2:
            continue
        border_colors = set(map(int, _border_values(block)))
        assert len(border_colors) == 1
        base_color = next(iter(border_colors))
        marker_colors = colors - {base_color}
        assert len(marker_colors) == 1
        pairs.append((base_color, next(iter(marker_colors))))
    assert pairs
    assert len(set(pairs)) == 1
    return pairs[0]


def _encode_pattern(
    shape: frozenset[tuple[int, int]],
    dims: tuple[int, int],
    base_color: int,
    marker_color: int,
):
    h, w = dims
    assert (5 - h) % 2 == 0 and (5 - w) % 2 == 0
    r_off = (5 - h) // 2
    c_off = (5 - w) // 2
    patch = np.full((5, 5), int(base_color), dtype=int)
    for r, c in shape:
        patch[r_off + r, c_off + c] = int(marker_color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Translate between colored legend shapes and 5x5 encoded pattern blocks.

    Input structure
    - A large teal-colored panel contains one 8-connected shape for each legend color.
    - Outside the panel are 5x5 non-background blocks, either solid color blocks or a shared two-color pattern.

    Rule
    - A solid color block is replaced by the common two-color pattern whose centered marker cells are that color's legend shape.
    - A two-color pattern block has its marker shape matched to a legend shape up to rotation/reflection, then the whole block is filled with the matched legend color.

    Key insight
    The panel is a color/shape dictionary; the 5x5 blocks are either encoded dictionary entries or decoded color swatches.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    panel_color, panel_box = _panel(g, bg)
    legend, panel_mask = _legend_shapes(g, panel_color, panel_box)
    block_boxes = _block_boxes(g, bg, panel_mask)
    base_color, marker_color = _pattern_colors(g, block_boxes)

    out = g.copy()
    for r0, c0, r1, c1 in block_boxes:
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, block.ravel()))
        if len(colors) == 1:
            solid_color = next(iter(colors))
            assert solid_color in legend
            shape, dims = legend[solid_color]
            out[r0 : r1 + 1, c0 : c1 + 1] = _encode_pattern(
                shape, dims, base_color, marker_color
            )
            continue

        assert colors == {base_color, marker_color}
        assert set(map(int, _border_values(block))) == {base_color}
        marker_shape, marker_dims = _normalize(np.argwhere(block == marker_color))
        matches = {
            color
            for color, (shape, dims) in legend.items()
            if (marker_shape, marker_dims) in _d4_variants(shape, dims)
        }
        assert len(matches) == 1
        out[r0 : r1 + 1, c0 : c1 + 1] = next(iter(matches))

    return out

# --------------- generator (o5x) ---------------
# Canvas and fixed colors.
GRID_SHAPE = (18, 22)
PANEL_COLOR, BASE_COLOR, MARKER_COLOR = 8, 2, 1
LEGEND_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 9])

# Legend shape and panel slot sampling.
LEGEND_COUNT, LEGEND_BBOX_SIZE = 4, 3
LEGEND_CELL_MIN, LEGEND_CELL_MAX_EXCLUSIVE = 4, 8
LEGEND_GAP_MIN, LEGEND_GAP_MAX = 1, 3
PANEL_WIDTH, PANEL_LEFT_C0, PANEL_RIGHT_C0 = 6, 0, 16
PANEL_LEGEND_COL_MAX_EXCLUSIVE = PANEL_WIDTH - LEGEND_BBOX_SIZE + 1
LEGEND_GAP_OPTIONS = [
    (a, b, c) for a in range(LEGEND_GAP_MIN, LEGEND_GAP_MAX + 1)
    for b in range(LEGEND_GAP_MIN, LEGEND_GAP_MAX + 1)
    for c in range(LEGEND_GAP_MIN, LEGEND_GAP_MAX + 1)
    if LEGEND_COUNT * LEGEND_BBOX_SIZE + a + b + c <= GRID_SHAPE[0]
]

# Block layout and input/output encoding mix.
BLOCK_SIZE, BLOCK_COUNT = 5, 4
BLOCK_ROWS = (1, 8)
BLOCK_COLS_PANEL_LEFT = (8, 15)
BLOCK_COLS_PANEL_RIGHT = (2, 9)
BLOCK_INDICES = np.arange(BLOCK_COUNT)
INPUT_SOLID_COUNT_MIN, INPUT_SOLID_COUNT_MAX_EXCLUSIVE = 0, 3

# Retry budget.
GENERATION_RETRY_LIMIT = 300

_SHAPE_ORBITS_BY_COUNT = None

def _gen_shape_orbits_by_count():
    import itertools

    global _SHAPE_ORBITS_BY_COUNT
    if _SHAPE_ORBITS_BY_COUNT is not None:
        return _SHAPE_ORBITS_BY_COUNT

    by_count = {n: [] for n in range(LEGEND_CELL_MIN, LEGEND_CELL_MAX_EXCLUSIVE)}
    for n in range(LEGEND_CELL_MIN, LEGEND_CELL_MAX_EXCLUSIVE):
        seen = set()
        for cells in itertools.combinations(range(LEGEND_BBOX_SIZE**2), n):
            mask = np.zeros((LEGEND_BBOX_SIZE, LEGEND_BBOX_SIZE), dtype=bool)
            mask.flat[list(cells)] = True
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if len(components_mask(mask, connectivity=8)) != 1:
                continue
            shape, dims = _normalize(np.argwhere(mask))
            variants = _d4_variants(shape, dims)
            key = min((v_dims, tuple(sorted(v_shape))) for v_shape, v_dims in variants)
            if key in seen:
                continue
            seen.add(key)
            by_count[n].append((shape, dims, key))
    _SHAPE_ORBITS_BY_COUNT = by_count
    return by_count

def _gen_random_variant(shape, dims, rng):
    variants = list(_d4_variants(shape, dims))
    return variants[int(rng.integers(len(variants)))]

def _gen_pick_shapes(rng):
    by_count = _gen_shape_orbits_by_count()
    chosen_keys = set()
    shapes = []
    for cell_count in rng.integers(
        LEGEND_CELL_MIN, LEGEND_CELL_MAX_EXCLUSIVE, size=LEGEND_COUNT
    ):
        candidates = list(by_count[int(cell_count)])
        rng.shuffle(candidates)
        for shape, dims, key in candidates:
            if key in chosen_keys:
                continue
            chosen_keys.add(key)
            shapes.append(_gen_random_variant(shape, dims, rng))
            break
        else:
            return None
    return shapes

def _gen_slot_rows(rng):
    gaps = LEGEND_GAP_OPTIONS[int(rng.integers(len(LEGEND_GAP_OPTIONS)))]
    spare = GRID_SHAPE[0] - LEGEND_COUNT * LEGEND_BBOX_SIZE - sum(gaps)
    top = int(rng.integers(spare + 1))
    rows = [top]
    for gap in gaps:
        rows.append(rows[-1] + LEGEND_BBOX_SIZE + gap)
    return rows

def _gen_place_panel(grid, panel_c0, colors, shapes, rng):
    grid[:, panel_c0 : panel_c0 + PANEL_WIDTH] = PANEL_COLOR
    for row, color, (shape, _dims) in zip(_gen_slot_rows(rng), colors, shapes):
        col = panel_c0 + int(rng.integers(0, PANEL_LEGEND_COL_MAX_EXCLUSIVE))
        for dr, dc in shape:
            grid[row + dr, col + dc] = int(color)

def _gen_block_boxes(panel_left):
    cols = BLOCK_COLS_PANEL_LEFT if panel_left else BLOCK_COLS_PANEL_RIGHT
    return [(r, c, r + BLOCK_SIZE - 1, c + BLOCK_SIZE - 1) for r in BLOCK_ROWS for c in cols]

def _gen_write_block(grid, box, block):
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = block

def _gen_attempt():
    rng = np.random.default_rng()
    shapes = _gen_pick_shapes(rng)
    if shapes is None:
        return None

    colors = rng.choice(LEGEND_COLOR_POOL, size=LEGEND_COUNT, replace=False)
    legend = {int(color): shape for color, shape in zip(colors, shapes)}

    panel_left = bool(rng.integers(2))
    panel_c0 = PANEL_LEFT_C0 if panel_left else PANEL_RIGHT_C0
    boxes = _gen_block_boxes(panel_left)
    block_colors = [int(c) for c in rng.permutation(colors)]
    input_solid_count = int(
        rng.integers(INPUT_SOLID_COUNT_MIN, INPUT_SOLID_COUNT_MAX_EXCLUSIVE)
    )
    input_solid = set(map(int, rng.choice(BLOCK_INDICES, size=input_solid_count, replace=False)))

    out = np.zeros(GRID_SHAPE, dtype=int)
    _gen_place_panel(out, panel_c0, colors, shapes, rng)

    for i, (box, color) in enumerate(zip(boxes, block_colors)):
        shape, dims = legend[color]
        if i in input_solid:
            block = _encode_pattern(shape, dims, BASE_COLOR, MARKER_COLOR)
        else:
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), color, dtype=int)
        _gen_write_block(out, box, block)

    inp = out.copy()
    for i, (box, color) in enumerate(zip(boxes, block_colors)):
        shape, dims = legend[color]
        if i in input_solid:
            block = np.full((BLOCK_SIZE, BLOCK_SIZE), color, dtype=int)
        else:
            variant_shape, variant_dims = _gen_random_variant(shape, dims, rng)
            block = _encode_pattern(variant_shape, variant_dims, BASE_COLOR, MARKER_COLOR)
        _gen_write_block(inp, box, block)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b20f7c8b input")
