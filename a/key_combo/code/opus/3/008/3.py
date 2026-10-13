from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

CONTROL_COLORS = (BLUE, RED, ORANGE, MAROON)
STRIPE_PATTERN = (YELLOW, YELLOW, GREEN)
TEMPLATE_SIZE = 3
CELL_GAP = 1
CELL_STRIDE = TEMPLATE_SIZE + CELL_GAP
LATTICE_DIM = 4
CONTROL_DIM = 4
RETRY_LIMIT = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bool_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        coords_list = []
        while queue:
            rr, cc = queue.popleft()
            coords_list.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        coords = np.array(coords_list, dtype=int)
        box = bbox_coords(coords)
        rel = np.zeros((bbox_h(box), bbox_w(box)), dtype=bool)
        rel[coords[:, 0] - box[0], coords[:, 1] - box[1]] = True
        out.append({"coords": coords, "bbox": box, "mask": rel, "area": int(coords.shape[0])})
    return out


def _regular_lattice(components: list[dict]) -> tuple[list[int], list[int], dict[tuple[int, int], dict]] | None:
    by_anchor: dict[tuple[int, int], dict] = {}
    heights = {bbox_h(comp["bbox"]) for comp in components}
    widths = {bbox_w(comp["bbox"]) for comp in components}
    if len(heights) != 1 or len(widths) != 1:
        return None
    rows = sorted({int(comp["bbox"][0]) for comp in components})
    cols = sorted({int(comp["bbox"][1]) for comp in components})
    if len(rows) * len(cols) != len(components):
        return None
    for comp in components:
        key = (int(comp["bbox"][0]), int(comp["bbox"][1]))
        by_anchor[key] = comp
    if any((r, c) not in by_anchor for r in rows for c in cols):
        return None
    if any(rows[i + 1] - rows[i] != CELL_STRIDE for i in range(len(rows) - 1)):
        return None
    if any(cols[i + 1] - cols[i] != CELL_STRIDE for i in range(len(cols) - 1)):
        return None
    return rows, cols, by_anchor


def _find_top_segment(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(grid[0] == YELLOW)
    assert 2 <= len(cols) <= 4, "expected a top yellow segment of width 2..4"
    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert np.array_equal(cols, np.arange(c0, c1 + 1)), "top yellow segment must be contiguous"
    return c0, c1


def _find_control_matrix(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.isin(grid, CONTROL_COLORS)
    box = bbox(mask)
    assert box is not None, "expected a dense 4x4 control matrix"
    r0, c0, r1, c1 = box
    assert (bbox_h(box), bbox_w(box)) == (CONTROL_DIM, CONTROL_DIM), "control matrix must be 4x4"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "control matrix must be dense"
    return box


def _find_template_lattice(grid: np.ndarray) -> tuple[list[int], list[int], dict[tuple[int, int], dict]]:
    components = _components_of_color(grid, GREY)
    assert len(components) == LATTICE_DIM * LATTICE_DIM, "expected 16 gray template objects"
    lattice = _regular_lattice(components)
    assert lattice is not None, "gray objects must form a regular 4x4 lattice"
    rows, cols, by_anchor = lattice
    assert len(rows) == len(cols) == LATTICE_DIM
    ref_mask = None
    for comp in components:
        box = comp["bbox"]
        assert (bbox_h(box), bbox_w(box)) == (TEMPLATE_SIZE, TEMPLATE_SIZE), "templates must use 3x3 bounding boxes"
        assert 3 <= comp["area"] <= 6, "template area must be 3..6"
        assert _bool_connected(comp["mask"]), "template must be 4-connected"
        if ref_mask is None:
            ref_mask = comp["mask"]
        else:
            assert np.array_equal(comp["mask"], ref_mask), "all templates must share one geometry"
    return rows, cols, by_anchor


def _stripe_color_at_cell(row: int, col: int, seg_c0: int, seg_c1: int) -> int:
    if row % 2 == 1:
        return BG
    base = STRIPE_PATTERN[(row // 2) % len(STRIPE_PATTERN)]
    if base == GREEN:
        return GREEN
    return YELLOW if seg_c0 <= col <= seg_c1 else BG


def _component_state(box: tuple[int, int, int, int], seg_c0: int, seg_c1: int) -> int:
    center_row = int(box[0] + 1)
    if center_row % 2 == 1:
        return BG
    stripe = STRIPE_PATTERN[(center_row // 2) % len(STRIPE_PATTERN)]
    if stripe == GREEN:
        return GREEN
    return YELLOW if not (box[3] < seg_c0 or box[1] > seg_c1) else BG


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit inside 30x30"
    assert get_bg(g) == BG, "background must be black"

    seg_c0, seg_c1 = _find_top_segment(g)
    control_bbox = _find_control_matrix(g)
    rows, cols, by_anchor = _find_template_lattice(g)

    ctl_r0, ctl_c0, ctl_r1, ctl_c1 = control_bbox
    assert ctl_c0 < cols[0] and ctl_r1 > rows[-1], "control matrix must sit in the lower-left area"

    protected = np.zeros_like(g, dtype=bool)
    protected[ctl_r0 : ctl_r1 + 1, ctl_c0 : ctl_c1 + 1] = True
    for comp in by_anchor.values():
        coords = comp["coords"]
        protected[coords[:, 0], coords[:, 1]] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if protected[r, c]:
                continue
            expected = _stripe_color_at_cell(r, c, seg_c0, seg_c1)
            assert int(g[r, c]) == expected, "visible stripe pixels must match the top-segment propagation rule"

    control = g[ctl_r0 : ctl_r1 + 1, ctl_c0 : ctl_c1 + 1].copy()
    assert control.shape == (CONTROL_DIM, CONTROL_DIM)

    return {
        "grid": g,
        "seg_c0": seg_c0,
        "seg_c1": seg_c1,
        "rows": rows,
        "cols": cols,
        "by_anchor": by_anchor,
        "control_bbox": control_bbox,
        "control": control,
    }


def _build_output(
    grid: np.ndarray,
    info: dict,
    *,
    yellow_uses_lookup: bool = True,
    green_uses_green: bool = True,
    yellow_needs_overlap: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    seg_c0 = int(info["seg_c0"])
    seg_c1 = int(info["seg_c1"])
    control = np.asarray(info["control"], dtype=int)
    rows = list(info["rows"])
    cols = list(info["cols"])
    by_anchor = dict(info["by_anchor"])

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            comp = by_anchor[(row, col)]
            box = comp["bbox"]
            center_row = int(box[0] + 1)
            row_stripe = BG if center_row % 2 else STRIPE_PATTERN[(center_row // 2) % len(STRIPE_PATTERN)]
            overlaps_yellow = not (box[3] < seg_c0 or box[1] > seg_c1)

            paint = None
            if row_stripe == GREEN:
                paint = GREEN if green_uses_green else int(control[i, j])
            elif row_stripe == YELLOW and (overlaps_yellow or not yellow_needs_overlap):
                paint = int(control[i, j]) if yellow_uses_lookup else GREEN

            if paint is not None:
                coords = comp["coords"]
                out[coords[:, 0], coords[:, 1]] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Route each gray template through lookup-recolor or uniform-green based on propagated stripe type.

    Input structure:
    - The background is black and already shows the stripe field implied by one yellow top-edge segment.
    - A 4x4 lattice of identical gray 3x3 template objects sits away from a dense 4x4 control matrix in the lower-left area.
    - Yellow stripes occupy only the segment's columns, while green stripes occupy full rows; both are visible only on non-template, non-control cells.

    Rule:
    - Read the unique yellow segment on the top edge and extend its stripe schedule downward with period yellow, yellow, green on every other row.
    - Determine each template object's stripe state from its bbox center row: green rows affect every column, yellow rows affect only templates whose bbox overlaps the segment's column span.
    - Recolor yellow-state templates from the same-position control-matrix cell, recolor green-state templates uniformly green, and leave no-stripe templates gray.

    Key insight:
    - Stripe color is not the final paint itself; it selects between two different recoloring mechanisms, so the control matrix matters only on yellow-selected templates.
    """
    info = _parse_scene(grid)
    return _build_output(np.asarray(grid, dtype=int), info)


def _all_valid_template_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (TEMPLATE_SIZE * TEMPLATE_SIZE)):
        area = int(bits.bit_count())
        if area < 3 or area > 6:
            continue
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        for idx in range(TEMPLATE_SIZE * TEMPLATE_SIZE):
            if (bits >> idx) & 1:
                mask[idx // TEMPLATE_SIZE, idx % TEMPLATE_SIZE] = True
        box = bbox(mask)
        if box != (0, 0, TEMPLATE_SIZE - 1, TEMPLATE_SIZE - 1):
            continue
        if not _bool_connected(mask):
            continue
        masks.append(mask)
    assert masks, "expected non-empty template mask catalogue"
    return masks


VALID_TEMPLATE_MASKS = _all_valid_template_masks()


def _yellow_rows_for_lattice(lat_r: int) -> list[int]:
    rows = []
    for i in range(LATTICE_DIM):
        center_row = lat_r + i * CELL_STRIDE + 1
        if _component_state((center_row - 1, 0, center_row + 1, 2), 0, 2) == YELLOW:
            rows.append(i)
    return rows


def _green_rows_for_lattice(lat_r: int) -> list[int]:
    rows = []
    for i in range(LATTICE_DIM):
        center_row = lat_r + i * CELL_STRIDE + 1
        if _component_state((center_row - 1, 0, center_row + 1, 2), 0, 2) == GREEN:
            rows.append(i)
    return rows


def _sample_control_matrix(rng: np.random.Generator, lat_r: int, selected_col: int) -> np.ndarray:
    control = rng.choice(np.array(CONTROL_COLORS, dtype=int), size=(CONTROL_DIM, CONTROL_DIM), replace=True)
    yellow_rows = _yellow_rows_for_lattice(lat_r)
    green_rows = _green_rows_for_lattice(lat_r)
    assert yellow_rows and green_rows
    while len({int(control[r, selected_col]) for r in yellow_rows}) < 2:
        for r in yellow_rows:
            control[r, selected_col] = int(rng.choice(CONTROL_COLORS))
    return control.astype(int)


def _segment_bounds(rng: np.random.Generator, template_col_start: int, width: int) -> tuple[int, int]:
    if width == 4:
        c0 = template_col_start - 1
    elif width == 3:
        c0 = template_col_start
    else:
        c0 = int(rng.integers(template_col_start, template_col_start + 2))
    return c0, c0 + width - 1


def _draw_input(
    shape: tuple[int, int],
    template_mask: np.ndarray,
    lat_r: int,
    lat_c: int,
    control: np.ndarray,
    ctl_r: int,
    ctl_c: int,
    seg_c0: int,
    seg_c1: int,
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = _stripe_color_at_cell(r, c, seg_c0, seg_c1)
    grid[ctl_r : ctl_r + CONTROL_DIM, ctl_c : ctl_c + CONTROL_DIM] = control
    for i in range(LATTICE_DIM):
        for j in range(LATTICE_DIM):
            r0 = lat_r + i * CELL_STRIDE
            c0 = lat_c + j * CELL_STRIDE
            patch = grid[r0 : r0 + TEMPLATE_SIZE, c0 : c0 + TEMPLATE_SIZE]
            patch[template_mask] = GREY
    return grid


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        info = _parse_scene(inp)
    except AssertionError:
        return False

    seg_c0 = int(info["seg_c0"])
    seg_c1 = int(info["seg_c1"])
    states = []
    yellow_lookup_colors = []
    for i, row in enumerate(info["rows"]):
        for j, col in enumerate(info["cols"]):
            comp = info["by_anchor"][(row, col)]
            state = _component_state(comp["bbox"], seg_c0, seg_c1)
            states.append(state)
            if state == YELLOW:
                yellow_lookup_colors.append(int(info["control"][i, j]))

    if not ({BG, YELLOW, GREEN} <= set(states)):
        return False
    if len(set(yellow_lookup_colors)) < 2:
        return False

    actual = main(inp)
    if np.array_equal(actual, inp):
        return False

    wrong_uniform_yellow = _build_output(inp, info, yellow_uses_lookup=False)
    wrong_green_lookup = _build_output(inp, info, green_uses_green=False)
    wrong_full_yellow_rows = _build_output(inp, info, yellow_needs_overlap=False)
    if np.array_equal(actual, wrong_uniform_yellow):
        return False
    if np.array_equal(actual, wrong_green_lookup):
        return False
    if np.array_equal(actual, wrong_full_yellow_rows):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(22, 29))
    width = int(rng.integers(20, 29))
    ctl_r = height - CONTROL_DIM
    ctl_c = 0

    lat_r_choices = [r for r in range(1, height - 19, 2)]
    lat_c_choices = [c for c in range(5, width - 14)]
    if not lat_r_choices or not lat_c_choices:
        return None

    lat_r = int(rng.choice(lat_r_choices))
    lat_c = int(rng.choice(lat_c_choices))
    template_mask = VALID_TEMPLATE_MASKS[int(rng.integers(len(VALID_TEMPLATE_MASKS)))]
    selected_col = int(rng.integers(0, LATTICE_DIM))
    segment_width = int(rng.integers(2, 5))
    seg_c0, seg_c1 = _segment_bounds(rng, lat_c + selected_col * CELL_STRIDE, segment_width)
    if seg_c0 < 0 or seg_c1 >= width:
        return None

    control = _sample_control_matrix(rng, lat_r, selected_col)
    inp = _draw_input((height, width), template_mask, lat_r, lat_c, control, ctl_r, ctl_c, seg_c0, seg_c1)

    try:
        expected = _build_output(inp, _parse_scene(inp))
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_check(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
