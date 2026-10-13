import numpy as np


BLACK = 0
GREY = 5
CLOCK_SIZE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_POINTERS = (
    ((1, 2), (0, 2)),  # up
    ((2, 3), (2, 4)),  # right
    ((3, 2), (4, 2)),  # down
    ((2, 1), (2, 0)),  # left
)
GREY_CELLS = ((0, 0), (0, 4), (2, 2), (4, 0), (4, 4))

RNG = np.random.default_rng()
LAYOUT_CHOICES = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
ROW_GAP_RANGE = (2, 6)
COL_GAP_RANGE = (2, 6)
GRID_H_RANGE = (18, 33)
GRID_W_RANGE = (18, 33)
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for sr, sc in starts:
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _crop_component(color: int, coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return out


def _match_clock(patch: np.ndarray) -> tuple[int, int] | None:
    if patch.shape != (CLOCK_SIZE, CLOCK_SIZE):
        return None
    for r, c in GREY_CELLS:
        if int(patch[r, c]) != GREY:
            return None
    if int(np.count_nonzero(patch != BLACK)) != 7:
        return None

    grey_set = set(GREY_CELLS)
    for rotation, pointer_cells in enumerate(DIR_POINTERS):
        color = int(patch[pointer_cells[0]])
        if color in (BLACK, GREY):
            continue
        if int(patch[pointer_cells[1]]) != color:
            continue
        pointer_set = set(pointer_cells)
        ok = True
        for r in range(CLOCK_SIZE):
            for c in range(CLOCK_SIZE):
                val = int(patch[r, c])
                if (r, c) in grey_set:
                    ok = ok and val == GREY
                elif (r, c) in pointer_set:
                    ok = ok and val == color
                else:
                    ok = ok and val == BLACK
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return color, rotation
    return None


def _find_clocks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    clocks: list[dict] = []
    for r0 in range(h - CLOCK_SIZE + 1):
        for c0 in range(w - CLOCK_SIZE + 1):
            found = _match_clock(g[r0 : r0 + CLOCK_SIZE, c0 : c0 + CLOCK_SIZE])
            if found is None:
                continue
            color, rotation = found
            clocks.append({"r0": int(r0), "c0": int(c0), "color": int(color), "rotation": int(rotation)})

    assert clocks, "no clocks found"
    colors = [clock["color"] for clock in clocks]
    assert len(colors) == len(set(colors)), "clock pointer colors must be unique"

    occupied = np.zeros(g.shape, dtype=bool)
    for clock in clocks:
        r0 = clock["r0"]
        c0 = clock["c0"]
        assert not occupied[r0 : r0 + CLOCK_SIZE, c0 : c0 + CLOCK_SIZE].any(), "clocks must not overlap"
        occupied[r0 : r0 + CLOCK_SIZE, c0 : c0 + CLOCK_SIZE] = True
    return clocks


def _extract_shapes(grid: np.ndarray, clocks: list[dict]) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    clock_mask = np.zeros(g.shape, dtype=bool)
    clock_colors = {int(clock["color"]) for clock in clocks}
    for clock in clocks:
        r0 = clock["r0"]
        c0 = clock["c0"]
        clock_mask[r0 : r0 + CLOCK_SIZE, c0 : c0 + CLOCK_SIZE] = True

    extra_colors = set(map(int, np.unique(g[~clock_mask]))) - {BLACK, GREY}
    assert extra_colors == clock_colors, "every non-clock color must belong to exactly one shape"

    shapes: dict[int, np.ndarray] = {}
    for color in sorted(clock_colors):
        comps = _components((g == color) & ~clock_mask)
        assert len(comps) == 1, f"expected exactly one shape for color {color}"
        shapes[color] = _crop_component(color, comps[0])
    return shapes


def _assemble_output(shape_grid: list[list[np.ndarray]]) -> np.ndarray:
    rows = len(shape_grid)
    cols = len(shape_grid[0])
    row_heights = [max(int(shape_grid[r][c].shape[0]) for c in range(cols)) for r in range(rows)]
    col_widths = [max(int(shape_grid[r][c].shape[1]) for r in range(rows)) for c in range(cols)]

    out_h = int(sum(row_heights) + max(0, rows - 1))
    out_w = int(sum(col_widths) + max(0, cols - 1))
    out = np.full((out_h, out_w), BLACK, dtype=int)

    row_starts: list[int] = []
    cursor = 0
    for idx, height in enumerate(row_heights):
        row_starts.append(int(cursor))
        cursor += int(height)
        if idx != rows - 1:
            out[cursor, :] = GREY
            cursor += 1

    col_starts: list[int] = []
    cursor = 0
    for idx, width in enumerate(col_widths):
        col_starts.append(int(cursor))
        cursor += int(width)
        if idx != cols - 1:
            out[:, cursor] = GREY
            cursor += 1

    for r in range(rows):
        for c in range(cols):
            patch = shape_grid[r][c]
            h, w = patch.shape
            rr = row_starts[r]
            cc = col_starts[c]
            mask = patch != BLACK
            out[rr : rr + h, cc : cc + w][mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    clocks = _find_clocks(g)
    shapes = _extract_shapes(g, clocks)

    row_values = sorted({int(clock["r0"]) for clock in clocks})
    col_values = sorted({int(clock["c0"]) for clock in clocks})
    assert len(row_values) * len(col_values) == len(clocks), "clocks must form a complete matrix"

    row_index = {value: idx for idx, value in enumerate(row_values)}
    col_index = {value: idx for idx, value in enumerate(col_values)}
    shape_grid: list[list[np.ndarray | None]] = [
        [None for _ in range(len(col_values))] for _ in range(len(row_values))
    ]

    for clock in clocks:
        rr = row_index[int(clock["r0"])]
        cc = col_index[int(clock["c0"])]
        assert shape_grid[rr][cc] is None, "duplicate clock matrix slot"
        shape = shapes[int(clock["color"])]
        shape_grid[rr][cc] = np.rot90(shape, -int(clock["rotation"]))

    assert all(all(cell is not None for cell in row) for row in shape_grid)
    return _assemble_output(shape_grid)  # type: ignore[arg-type]


def _render_clock(color: int, rotation: int) -> np.ndarray:
    patch = np.zeros((CLOCK_SIZE, CLOCK_SIZE), dtype=int)
    for r, c in GREY_CELLS:
        patch[r, c] = GREY
    for r, c in DIR_POINTERS[int(rotation)]:
        patch[r, c] = int(color)
    return patch


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rotation_sensitive(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rotated = np.rot90(mask, k)
        if rotated.shape == mask.shape and np.array_equal(rotated, mask):
            return False
    return True


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(3, 6))
        w = int(RNG.integers(3, 6))
        area_min = max(4, (h * w) // 2)
        area = int(RNG.integers(area_min, h * w))

        cells = {(int(RNG.integers(h)), int(RNG.integers(w)))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            idx = int(RNG.integers(len(frontier)))
            cells.add(frontier[idx])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_bool(mask)
        mh, mw = mask.shape
        if mh < 3 or mw < 3 or mh > 5 or mw > 5:
            continue
        if int(mask.sum()) == int(mh * mw):
            continue
        if not _rotation_sensitive(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a rotation-sensitive shape")


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, gap: int) -> bool:
    bh, bw = blocked.shape
    if top < 0 or left < 0 or top + h > bh or left + w > bw:
        return False
    r0 = max(0, top - gap)
    r1 = min(bh, top + h + gap)
    c0 = max(0, left - gap)
    c1 = min(bw, left + w + gap)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, gap: int) -> None:
    bh, bw = blocked.shape
    r0 = max(0, top - gap)
    r1 = min(bh, top + h + gap)
    c0 = max(0, left - gap)
    c1 = min(bw, left + w + gap)
    blocked[r0:r1, c0:c1] = True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _sample_rotations(count: int) -> list[int]:
    if count <= 4:
        choices = np.arange(4)
        RNG.shuffle(choices)
        return [int(v) for v in choices[:count]]
    base = [0, 1, 2, 3]
    extra = [int(v) for v in RNG.integers(0, 4, size=count - 4)]
    all_rots = base + extra
    RNG.shuffle(all_rots)
    return [int(v) for v in all_rots]


def _gen_attempt() -> np.ndarray | None:
    rows, cols = LAYOUT_CHOICES[int(RNG.integers(len(LAYOUT_CHOICES)))]
    count = rows * cols
    row_gap = int(RNG.integers(*ROW_GAP_RANGE))
    col_gap = int(RNG.integers(*COL_GAP_RANGE))
    clock_h = rows * CLOCK_SIZE + (rows - 1) * row_gap
    clock_w = cols * CLOCK_SIZE + (cols - 1) * col_gap

    colors = [int(c) for c in RNG.choice(np.array(COLOR_POOL), size=count, replace=False)]
    rotations = _sample_rotations(count)
    base_shapes = {color: _crop_component(color, np.argwhere(_random_connected_mask())) for color in colors}

    shape_grid: list[list[np.ndarray]] = []
    idx = 0
    for r in range(rows):
        row_shapes: list[np.ndarray] = []
        for c in range(cols):
            color = colors[idx]
            row_shapes.append(np.rot90(base_shapes[color], -rotations[idx]))
            idx += 1
        shape_grid.append(row_shapes)
    expected = _assemble_output(shape_grid)

    shape_boxes = [base_shapes[color].shape for color in colors]
    buffered_area = sum((int(h) + 2) * (int(w) + 2) for h, w in shape_boxes)

    min_h = max(GRID_H_RANGE[0], clock_h + 2)
    min_w = max(GRID_W_RANGE[0], clock_w + 2)
    max_h = max(min_h + 2, GRID_H_RANGE[1] - 1)
    max_w = max(min_w + 2, GRID_W_RANGE[1] - 1)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(40):
        grid_h = int(RNG.integers(min_h, max_h + 1))
        grid_w = int(RNG.integers(min_w, max_w + 1))
        if grid_h * grid_w < clock_h * clock_w + buffered_area + 30:
            continue

        grid = np.full((grid_h, grid_w), BLACK, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)

        clock_top = int(RNG.integers(0, grid_h - clock_h + 1))
        clock_left = int(RNG.integers(0, grid_w - clock_w + 1))
        _mark_blocked(blocked, clock_top, clock_left, clock_h, clock_w, gap=1)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                top = clock_top + r * (CLOCK_SIZE + row_gap)
                left = clock_left + c * (CLOCK_SIZE + col_gap)
                _paint_patch(grid, _render_clock(colors[idx], rotations[idx]), top, left)
                idx += 1

        order = sorted(range(count), key=lambda i: int(base_shapes[colors[i]].size), reverse=True)
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for i in order:
            patch = base_shapes[colors[i]]
            ph, pw = patch.shape
            placed = False
            for _ in range(300):
                top = int(RNG.integers(0, grid_h - ph + 1))
                left = int(RNG.integers(0, grid_w - pw + 1))
                if not _can_place(blocked, top, left, ph, pw, gap=1):
                    continue
                placements[i] = (top, left)
                _mark_blocked(blocked, top, left, ph, pw, gap=1)
                _paint_patch(grid, patch, top, left)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
