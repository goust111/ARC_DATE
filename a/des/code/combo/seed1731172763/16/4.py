import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400

# Arrow code: the occupied corner that touches both legs is the direction marker.
ARROW_SHAPES = {
    0: {(0, 0), (0, 1), (1, 0)},  # up    -> rotate 90 deg clockwise
    1: {(0, 0), (0, 1), (1, 1)},  # right -> horizontal flip
    2: {(0, 1), (1, 0), (1, 1)},  # down  -> rotate 90 deg counterclockwise
    3: {(0, 0), (1, 0), (1, 1)},  # left  -> vertical flip
}
ASYMMETRIC_TEMPLATE = np.arange(1, 10, dtype=int).reshape(3, 3)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _panel_slices(grid: np.ndarray) -> list[slice]:
    is_sep = np.all(grid == SEPARATOR, axis=0)
    spans: list[slice] = []
    c = 0
    while c < grid.shape[1]:
        while c < grid.shape[1] and is_sep[c]:
            c += 1
        if c >= grid.shape[1]:
            break
        start = c
        while c < grid.shape[1] and not is_sep[c]:
            c += 1
        spans.append(slice(start, c))
    assert spans, "expected at least one panel"
    widths = {s.stop - s.start for s in spans}
    assert len(widths) == 1, "panels must have equal widths"
    return spans


def _decode_arrow(block: np.ndarray) -> int:
    assert block.shape == (2, 2)
    coords = {tuple(map(int, rc)) for rc in np.argwhere(block != BG)}
    assert len(coords) == 3
    for code, shape in ARROW_SHAPES.items():
        if coords == shape:
            return code
    raise AssertionError("unrecognized arrow marker")


def _apply_op(grid: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return np.rot90(grid, -1)
    if op == 1:
        return np.fliplr(grid)
    if op == 2:
        return np.rot90(grid, 1)
    if op == 3:
        return np.flipud(grid)
    raise AssertionError(f"unknown op {op}")


def _apply_ops(grid: np.ndarray, ops: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for op in ops:
        out = _apply_op(out, op)
    return out


def _symmetry_variants(grid: np.ndarray) -> list[np.ndarray]:
    return [
        grid,
        np.rot90(grid, -1),
        np.rot90(grid, 2),
        np.rot90(grid, 1),
        np.fliplr(grid),
        np.flipud(grid),
        grid.T,
        np.rot90(grid, 2).T,
    ]


def _is_fully_asymmetric(grid: np.ndarray) -> bool:
    variants = _symmetry_variants(grid)
    return all(not np.array_equal(variants[0], variant) for variant in variants[1:])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the input into equal-width panels, decode each bottom-row arrow sequence,
    and apply the encoded transforms in left-to-right order to the panel's upper
    square before concatenating the transformed upper regions without separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == SEPARATOR), "expected grey separator columns"

    outputs: list[np.ndarray] = []
    for panel_slice in _panel_slices(g):
        panel = g[:, panel_slice]
        comps = _components4(panel != BG)
        arrow_comps = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 3 and (r1 - r0, c1 - c0) == (2, 2) and r1 == panel.shape[0]:
                arrow_comps.append((c0, comp))

        assert 1 <= len(arrow_comps) <= 3, "each panel must have one to three arrows"
        arrow_row0 = min(_bbox(comp)[0] for _, comp in arrow_comps)
        upper = panel[:arrow_row0, :].copy()
        assert upper.shape[0] == upper.shape[1], "generated puzzles use square upper regions"
        assert np.any(upper != BG), "upper region must contain a pattern"

        ops = []
        for _, comp in sorted(arrow_comps, key=lambda item: item[0]):
            r0, c0, r1, c1 = _bbox(comp)
            ops.append(_decode_arrow(panel[r0:r1, c0:c1]))
        outputs.append(_apply_ops(upper, ops))

    heights = {arr.shape[0] for arr in outputs}
    assert len(heights) == 1
    return np.hstack(outputs)


def _random_connected_cells(h: int, w: int, target_area: int) -> set[tuple[int, int]] | None:
    start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
    cells = {start}
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        frontier_list = list(frontier)
        cells.add(frontier_list[int(RNG.integers(0, len(frontier_list)))])
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None
    return cells


def _colorize_shape(h: int, w: int, cells: set[tuple[int, int]]) -> np.ndarray:
    local = np.zeros((h, w), dtype=int)
    coords = np.array(sorted(cells), dtype=int)
    color_count = int(RNG.integers(2, 4))
    colors = [int(c) for c in RNG.choice(PALETTE, size=color_count, replace=False)]
    key_mode = int(RNG.integers(0, 4))
    if key_mode == 0:
        order = np.lexsort((coords[:, 1], coords[:, 0]))
    elif key_mode == 1:
        order = np.lexsort((coords[:, 0], coords[:, 1]))
    elif key_mode == 2:
        order = np.lexsort((coords[:, 1], coords[:, 0] + coords[:, 1]))
    else:
        order = np.lexsort((coords[:, 0], coords[:, 0] - coords[:, 1]))
    ordered = coords[order]

    split_points = sorted(
        int(x)
        for x in RNG.choice(np.arange(1, len(ordered)), size=color_count - 1, replace=False)
    )
    starts = [0] + split_points
    stops = split_points + [len(ordered)]
    for color, start, stop in zip(colors, starts, stops):
        for r, c in ordered[start:stop]:
            local[int(r), int(c)] = color
    return local


def _generate_upper(size: int) -> np.ndarray | None:
    for _ in range(120):
        box_h = int(RNG.integers(3, size - 1))
        box_w = int(RNG.integers(3, size - 1))
        area_min = max(7, (box_h * box_w) // 3)
        area_max = max(area_min + 1, (2 * box_h * box_w) // 3 + 1)
        target_area = int(RNG.integers(area_min, area_max))
        cells = _random_connected_cells(box_h, box_w, target_area)
        if cells is None:
            continue
        zone = np.zeros((size, size), dtype=int)
        top = int(RNG.integers(1, size - box_h))
        left = int(RNG.integers(1, size - box_w))
        zone[top : top + box_h, left : left + box_w] = _colorize_shape(box_h, box_w, cells)
        if np.count_nonzero(zone) < 7:
            continue
        if _is_fully_asymmetric(zone):
            return zone
    return None


def _sequence_is_non_identity(ops: list[int]) -> bool:
    return not np.array_equal(_apply_ops(ASYMMETRIC_TEMPLATE, ops), ASYMMETRIC_TEMPLATE)


def _sample_sequences(panel_count: int) -> list[list[int]] | None:
    for _ in range(160):
        counts = [int(RNG.integers(1, 4)) for _ in range(panel_count)]
        if sum(counts) < 4:
            continue

        pool = [0, 1, 2, 3]
        if sum(counts) > 4:
            pool.extend(int(x) for x in RNG.integers(0, 4, size=sum(counts) - 4))
        RNG.shuffle(pool)

        sequences = []
        idx = 0
        valid = True
        for count in counts:
            ops = [int(x) for x in pool[idx : idx + count]]
            idx += count
            if not _sequence_is_non_identity(ops):
                valid = False
                break
            sequences.append(ops)
        if not valid:
            continue
        if len({tuple(seq) for seq in sequences}) < 2:
            continue
        return sequences
    return None


def _render_panel(upper: np.ndarray, ops: list[int]) -> np.ndarray:
    size = upper.shape[0]
    panel = np.zeros((size + 2, size), dtype=int)
    panel[:size, :] = upper

    total_arrow_width = 2 * len(ops) + (len(ops) - 1)
    left = int(RNG.integers(0, size - total_arrow_width + 1))
    available = [int(c) for c in PALETTE if c not in set(map(int, upper.ravel()))]
    arrow_color = available[int(RNG.integers(0, len(available)))] if available else int(PALETTE[0])
    for i, op in enumerate(ops):
        c0 = left + 3 * i
        for dr, dc in ARROW_SHAPES[op]:
            panel[size + dr, c0 + dc] = arrow_color
    return panel


def _gen_attempt() -> np.ndarray | None:
    panel_count = int(RNG.integers(2, 5))
    panel_size = int(RNG.integers(8, 11))
    sep_width = int(RNG.integers(1, 3))
    sequences = _sample_sequences(panel_count)
    if sequences is None:
        return None

    uppers = []
    for ops in sequences:
        upper = _generate_upper(panel_size)
        if upper is None:
            return None
        if np.array_equal(_apply_ops(upper, ops), upper):
            return None
        uppers.append(upper)

    panel_h = panel_size + 2
    total_w = panel_count * panel_size + (panel_count - 1) * sep_width
    grid = np.zeros((panel_h, total_w), dtype=int)
    col = 0
    for i, (upper, ops) in enumerate(zip(uppers, sequences)):
        grid[:, col : col + panel_size] = _render_panel(upper, ops)
        col += panel_size
        if i + 1 < panel_count:
            grid[:, col : col + sep_width] = SEPARATOR
            col += sep_width

    expected = np.hstack([_apply_ops(upper, ops) for upper, ops in zip(uppers, sequences)])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate side-by-side square panels separated by grey columns.
    Each panel has a colored upper pattern and a bottom row band of one to three
    arrow markers whose directions cover all four operation types across the grid.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
