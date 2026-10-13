import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_INSIDE_PAIR_BASE_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
]

_INSIDE_CENTER_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
]

_GENERAL_TEMPLATES = [
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_col(left: int, right: int, c: int) -> int:
    return int(left + right - c)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_mask = g == GREY
    box = _bbox(marker_mask)
    assert box is not None, "expected grey marker pixels"
    r0, c0, r1, c1 = box

    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r, c] != ORANGE:
                continue
            mc = _mirror_col(c0, c1, c)
            if c0 <= mc <= c1 and g[r, mc] == ORANGE:
                out[r, c] = GREEN
            else:
                out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor orange pixels inside the grey-marker bounding box by whether they have a horizontal mirror partner.

    Input structure:
    - The grid has a black background plus orange(7) pattern pixels and a sparse grey(5) marker mask.
    - Orange pixels form scattered 4-connected mini-components of size 1-4 across the canvas.
    - The grey pixels are much sparser and their axis-aligned bounding box defines the active window.

    Rule:
    - Compute the minimal bounding box of all grey pixels and use its left/right columns as the mirror frame.
    - For each orange pixel inside that box, reflect it horizontally across the box's vertical midline.
    - If the reflected same-row cell inside the box is also orange, recolor the pixel green(3); otherwise recolor it red(2).
    - Leave all grey pixels, all black pixels, and all orange pixels outside the grey bounding box unchanged.

    Key insight:
    - The sparse grey mask does double duty: it selects which orange pixels are eligible and it implicitly defines the symmetry axis used to split them into paired and unpaired cases.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert not np.any(g == RED), "red should be absent from the input"
    assert not np.any(g == GREEN), "green should be absent from the input"
    assert np.any(g == GREY), "expected at least one grey marker"
    return _transform(g)


def _cells_in_box(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def _pair_masks(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = box
    paired = np.zeros_like(g, dtype=bool)
    unpaired = np.zeros_like(g, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if g[r, c] != ORANGE:
                continue
            mc = _mirror_col(c0, c1, c)
            if c0 <= mc <= c1 and g[r, mc] == ORANGE:
                paired[r, c] = True
            else:
                unpaired[r, c] = True
    return paired, unpaired


def _alt_all_inside_green(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    box = _bbox(g == GREY)
    assert box is not None
    r0, c0, r1, c1 = box
    out = g.copy()
    window = out[r0 : r1 + 1, c0 : c1 + 1]
    window[window == ORANGE] = GREEN
    return out


def _alt_global_axis(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    box = _bbox(g == GREY)
    assert box is not None
    r0, c0, r1, c1 = box
    out = g.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if g[r, c] != ORANGE:
                continue
            mc = w - 1 - c
            if 0 <= mc < w and g[r, mc] == ORANGE:
                out[r, c] = GREEN
            else:
                out[r, c] = RED
    return out


def _alt_global_scope(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    box = _bbox(g == GREY)
    assert box is not None
    _r0, c0, _r1, c1 = box
    out = g.copy()
    for r, c in np.argwhere(g == ORANGE):
        mc = _mirror_col(c0, c1, int(c))
        if 0 <= mc < g.shape[1] and g[int(r), mc] == ORANGE:
            out[int(r), int(c)] = GREEN
        else:
            out[int(r), int(c)] = RED
    return out


def _global_mirror_cols(w: int, cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, w - 1 - c) for r, c in cells}


def _can_place_group(
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for r, c in cell_set:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in orange or (r, c) in blocked or (r, c) in protected_empty:
            return False
    for r, c in cell_set:
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc) and (nr, nc) in orange and (nr, nc) not in cell_set:
                return False
    return True


def _add_group(orange: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    orange.update((int(r), int(c)) for r, c in cells)


def _template_dims(template: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in template)
    w = 1 + max(c for _, c in template)
    return int(h), int(w)


def _place_required_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r0, c0, _r1, _c1 = box
    local_blocked = {(r - r0, c - c0) for r, c in blocked if r0 <= r <= box[2] and c0 <= c <= box[3]}
    bh = box[2] - r0 + 1
    bw = box[3] - c0 + 1
    left_cols = [c for c in range(bw) if 2 * c < bw - 1]
    rng.shuffle(left_cols)
    rows = list(range(bh))
    rng.shuffle(rows)
    for lr in rows:
        for lc in left_cols:
            rc = bw - 1 - lc
            cells = [(r0 + lr, c0 + lc), (r0 + lr, c0 + rc)]
            if not _can_place_group((h, w), cells, orange, blocked, protected_empty):
                continue
            global_mirrors = _global_mirror_cols(w, cells)
            if any(pos in orange or pos in blocked for pos in global_mirrors):
                continue
            _add_group(orange, cells)
            protected_empty.update(global_mirrors - set(cells))
            return True
    return False


def _place_required_unpaired(
    rng: np.random.Generator,
    h: int,
    w: int,
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r0, c0, _r1, _c1 = box
    bh = box[2] - r0 + 1
    bw = box[3] - c0 + 1
    candidates = [(lr, lc) for lr in range(bh) for lc in range(bw)]
    rng.shuffle(candidates)
    for lr, lc in candidates:
        mc = bw - 1 - lc
        if lc == mc:
            continue
        cell = (r0 + lr, c0 + lc)
        mirror = (r0 + lr, c0 + mc)
        if not _can_place_group((h, w), [cell], orange, blocked, protected_empty):
            continue
        if mirror in orange:
            continue
        _add_group(orange, [cell])
        protected_empty.add(mirror)
        return True
    return False


def _try_place_inside_pair_component(
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    templates = list(_INSIDE_PAIR_BASE_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        th, tw = _template_dims(template)
        row_order = list(range(bh - th + 1))
        col_order = list(range(bw - tw + 1))
        rng.shuffle(row_order)
        rng.shuffle(col_order)
        for ar in row_order:
            for ac in col_order:
                base = [(r0 + ar + dr, c0 + ac + dc) for dr, dc in template]
                mirrored = []
                valid = True
                for gr, gc in base:
                    lc = gc - c0
                    mc = bw - 1 - lc
                    if lc >= mc:
                        valid = False
                        break
                    mirrored.append((gr, c0 + mc))
                if not valid:
                    continue
                group = base + mirrored
                if len(set(group)) != len(group) or len(group) > 4:
                    continue
                if not _can_place_group((r1 + 1, c1 + 1), group, orange, blocked, protected_empty):
                    continue
                _add_group(orange, group)
                return True
    return False


def _try_place_center_component(
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    bw = c1 - c0 + 1
    if bw % 2 == 0:
        return False
    center_c = c0 + bw // 2
    bh = r1 - r0 + 1
    templates = list(_INSIDE_CENTER_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        th, _tw = _template_dims(template)
        row_order = list(range(bh - th + 1))
        rng.shuffle(row_order)
        for ar in row_order:
            group = [(r0 + ar + dr, center_c) for dr, _dc in template]
            if not _can_place_group((r1 + 1, c1 + 1), group, orange, blocked, protected_empty):
                continue
            _add_group(orange, group)
            return True
    return False


def _try_place_unpaired_component(
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    templates = list(_GENERAL_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        th, tw = _template_dims(template)
        row_order = list(range(bh - th + 1))
        col_order = list(range(bw - tw + 1))
        rng.shuffle(row_order)
        rng.shuffle(col_order)
        for ar in row_order:
            for ac in col_order:
                group = [(r0 + ar + dr, c0 + ac + dc) for dr, dc in template]
                mirrors = []
                valid = True
                group_set = set(group)
                for gr, gc in group:
                    mc = _mirror_col(c0, c1, gc)
                    if mc == gc or (gr, mc) in group_set or (gr, mc) in orange:
                        valid = False
                        break
                    mirrors.append((gr, mc))
                if not valid:
                    continue
                if not _can_place_group((r1 + 1, c1 + 1), group, orange, blocked, protected_empty):
                    continue
                _add_group(orange, group)
                protected_empty.update(mirrors)
                return True
    return False


def _try_place_outside_component(
    rng: np.random.Generator,
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    orange: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    protected_empty: set[tuple[int, int]],
) -> bool:
    h, w = shape
    box_cells = _cells_in_box(box)
    templates = list(_GENERAL_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        th, tw = _template_dims(template)
        row_order = list(range(h - th + 1))
        col_order = list(range(w - tw + 1))
        rng.shuffle(row_order)
        rng.shuffle(col_order)
        for ar in row_order:
            for ac in col_order:
                group = [(ar + dr, ac + dc) for dr, dc in template]
                if any(cell in box_cells for cell in group):
                    continue
                if not _can_place_group(shape, group, orange, blocked, protected_empty):
                    continue
                _add_group(orange, group)
                return True
    return False


def _marker_layout(rng: np.random.Generator, bh: int, bw: int) -> list[tuple[int, int]]:
    mode = int(rng.choice([2, 3, 4]))
    if mode == 2:
        if rng.random() < 0.5:
            return [(0, 0), (bh - 1, bw - 1)]
        return [(0, bw - 1), (bh - 1, 0)]
    if mode == 3:
        corner = int(rng.integers(4))
        if corner == 0:
            return [(0, 0), (bh - 1, int(rng.integers(1, bw))), (int(rng.integers(1, bh)), bw - 1)]
        if corner == 1:
            return [(0, bw - 1), (bh - 1, int(rng.integers(0, bw - 1))), (int(rng.integers(1, bh)), 0)]
        if corner == 2:
            return [(bh - 1, 0), (0, int(rng.integers(0, bw - 1))), (int(rng.integers(0, bh - 1)), bw - 1)]
        return [(bh - 1, bw - 1), (0, int(rng.integers(1, bw))), (int(rng.integers(0, bh - 1)), 0)]
    return [
        (0, int(rng.integers(1, bw - 1))),
        (bh - 1, int(rng.integers(1, bw - 1))),
        (int(rng.integers(1, bh - 1)), 0),
        (int(rng.integers(1, bh - 1)), bw - 1),
    ]


def _sample_bbox(rng: np.random.Generator, h: int, w: int) -> tuple[int, int, int, int]:
    bbox_h = int(rng.integers(4, min(8, h - 2) + 1))
    width_options = [x for x in range(5, min(9, w - 2) + 1)]
    want_odd = bool(rng.integers(2))
    parity_options = [x for x in width_options if (x % 2 == 1) == want_odd]
    bbox_w = int(rng.choice(parity_options if parity_options else width_options))
    row_options = list(range(1, h - bbox_h))
    col_options = list(range(1, w - bbox_w))
    rng.shuffle(row_options)
    rng.shuffle(col_options)
    for r0 in row_options:
        for c0 in col_options:
            c1 = c0 + bbox_w - 1
            if c0 + c1 != w - 1:
                return int(r0), int(c0), int(r0 + bbox_h - 1), int(c1)
    raise RuntimeError("failed to sample a bbox with a non-global symmetry axis")


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert 12 <= g.shape[0] <= 15 and 12 <= g.shape[1] <= 15
    assert not np.any(g == RED)
    assert not np.any(g == GREEN)
    assert np.any(g == GREY)
    assert np.any(g == ORANGE)

    grey_count = int(np.count_nonzero(g == GREY))
    orange_count = int(np.count_nonzero(g == ORANGE))
    assert 2 <= grey_count <= 4
    assert grey_count < orange_count

    orange_sizes = _component_sizes(g == ORANGE)
    assert orange_sizes and max(orange_sizes) <= 4

    box = _bbox(g == GREY)
    assert box is not None
    r0, c0, r1, c1 = box
    assert c0 + c1 != g.shape[1] - 1

    paired, unpaired = _pair_masks(g, box)
    assert int(paired.sum()) >= 2
    assert int(unpaired.sum()) >= 1

    in_box = np.zeros_like(g, dtype=bool)
    in_box[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.any((g == ORANGE) & ~in_box)

    out = _transform(g)
    assert np.any(out == GREEN)
    assert np.any(out == RED)
    assert np.array_equal(out[g == GREY], g[g == GREY])
    assert np.any(out[(g == ORANGE) & ~in_box] == ORANGE)

    assert not np.array_equal(out, _alt_all_inside_green(g))
    assert not np.array_equal(out, _alt_global_axis(g))
    assert not np.array_equal(out, _alt_global_scope(g))


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 16))
    w = int(rng.integers(12, 16))
    box = _sample_bbox(rng, h, w)
    r0, c0, r1, c1 = box
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1

    marker_cells_local = _marker_layout(rng, bh, bw)
    marker_cells = {(r0 + lr, c0 + lc) for lr, lc in marker_cells_local}
    orange: set[tuple[int, int]] = set()
    protected_empty: set[tuple[int, int]] = set()

    if not _place_required_pair(rng, h, w, box, orange, marker_cells, protected_empty):
        return None
    if not _place_required_unpaired(rng, h, w, box, orange, marker_cells, protected_empty):
        return None

    extra_pairs = int(rng.integers(0, 3))
    for _ in range(extra_pairs):
        if not _try_place_inside_pair_component(rng, box, orange, marker_cells, protected_empty):
            break

    if bw % 2 == 1 and rng.random() < 0.6:
        _try_place_center_component(rng, box, orange, marker_cells, protected_empty)

    extra_unpaired = int(rng.integers(0, 3))
    for _ in range(extra_unpaired):
        if not _try_place_unpaired_component(rng, box, orange, marker_cells, protected_empty):
            break

    outside_count = int(rng.integers(2, 6))
    placed_outside = 0
    for _ in range(outside_count):
        if _try_place_outside_component(rng, (h, w), box, orange, marker_cells, protected_empty):
            placed_outside += 1
    if placed_outside == 0:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r, c in marker_cells:
        grid[r, c] = GREY
    for r, c in orange:
        grid[r, c] = ORANGE

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 048/1")
