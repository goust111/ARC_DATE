import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _boundary_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_bounds = [
        r
        for r in range(h)
        if np.all((grid[r, :] != BLACK) & (grid[r, :] != MAROON)) and np.any(grid[r, :] == GREY)
    ]
    col_bounds = [
        c
        for c in range(w)
        if np.all((grid[:, c] != BLACK) & (grid[:, c] != MAROON)) and np.any(grid[:, c] == GREY)
    ]
    assert row_bounds and row_bounds[0] == 0 and row_bounds[-1] == h - 1
    assert col_bounds and col_bounds[0] == 0 and col_bounds[-1] == w - 1
    return row_bounds, col_bounds


def _collect_mapping(
    grid: np.ndarray, r0: int, r1: int, c0: int, c1: int
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    mapping: dict[int, int] = {}
    targets: set[tuple[int, int]] = set()
    seen = 0

    for c in range(c0 + 1, c1):
        source = int(grid[r0, c])
        if source != GREY:
            target = int(grid[r0 + 1, c])
            if source in mapping:
                assert mapping[source] == target
            mapping[source] = target
            targets.add((r0 + 1, c))
            seen += 1

        source = int(grid[r1, c])
        if source != GREY:
            target = int(grid[r1 - 1, c])
            if source in mapping:
                assert mapping[source] == target
            mapping[source] = target
            targets.add((r1 - 1, c))
            seen += 1

    for r in range(r0 + 1, r1):
        source = int(grid[r, c0])
        if source != GREY:
            target = int(grid[r, c0 + 1])
            if source in mapping:
                assert mapping[source] == target
            mapping[source] = target
            targets.add((r, c0 + 1))
            seen += 1

        source = int(grid[r, c1])
        if source != GREY:
            target = int(grid[r, c1 - 1])
            if source in mapping:
                assert mapping[source] == target
            mapping[source] = target
            targets.add((r, c1 - 1))
            seen += 1

    assert seen > 0
    return mapping, targets


def _mirror_axis(room: np.ndarray) -> tuple[str, int]:
    coords = np.argwhere(room == MAROON)
    assert coords.size > 0
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        return "h", int(rows[0])
    if len(cols) == 1:
        return "v", int(cols[0])
    raise AssertionError("mirror must be a straight maroon segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each room's pattern across its maroon mirror, then recolor only the
    reflected copy using the room's boundary color-pair legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    row_bounds, col_bounds = _boundary_indices(g)
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i + 1]
            c0, c1 = col_bounds[j], col_bounds[j + 1]
            assert r1 - r0 >= 3 and c1 - c0 >= 3

            mapping, marker_targets = _collect_mapping(g, r0, r1, c0, c1)
            room = g[r0 + 1 : r1, c0 + 1 : c1]
            axis, local_idx = _mirror_axis(room)

            pattern: list[tuple[int, int, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if (r, c) in marker_targets:
                        continue
                    value = int(g[r, c])
                    if value != BLACK and value != MAROON:
                        pattern.append((r, c, value))
            assert pattern

            if axis == "h":
                mirror_r = r0 + 1 + local_idx
                sides = {1 if r > mirror_r else -1 for r, _c, _v in pattern}
                assert len(sides) == 1
                for r, c, value in pattern:
                    rr, cc = 2 * mirror_r - r, c
                    assert r0 < rr < r1 and c0 < cc < c1
                    new_value = mapping.get(value, value)
                    assert out[rr, cc] in (BLACK, new_value)
                    out[rr, cc] = new_value
            else:
                mirror_c = c0 + 1 + local_idx
                sides = {1 if c > mirror_c else -1 for _r, c, _v in pattern}
                assert len(sides) == 1
                for r, c, value in pattern:
                    rr, cc = r, 2 * mirror_c - c
                    assert r0 < rr < r1 and c0 < cc < c1
                    new_value = mapping.get(value, value)
                    assert out[rr, cc] in (BLACK, new_value)
                    out[rr, cc] = new_value

    return out


# --------------- generator ---------------

GRID_SIDE_MAX = 30
ROOM_SIZE_MIN = 7
ROOM_SIZE_MAX = 10
PATTERN_MIN_CELLS = 4
PATTERN_MAX_CELLS = 8
MAX_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 100


def _pick_spaced_positions(candidates: list[int], count: int) -> list[int] | None:
    pool = list(candidates)
    random.shuffle(pool)
    chosen: list[int] = []
    for pos in pool:
        if all(abs(pos - other) >= 2 for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _grow_shape(allowed: np.ndarray) -> set[tuple[int, int]] | None:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    if len(cells) < PATTERN_MIN_CELLS:
        return None

    max_cells = min(PATTERN_MAX_CELLS, len(cells))
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(PATTERN_MIN_CELLS, max_cells)
        seed = random.choice(cells)
        shape = {seed}
        frontier = [pt for pt in _neighbors4(*seed) if 0 <= pt[0] < allowed.shape[0] and 0 <= pt[1] < allowed.shape[1] and allowed[pt]]
        while frontier and len(shape) < target:
            idx = random.randrange(len(frontier))
            pt = frontier.pop(idx)
            if pt in shape or not allowed[pt]:
                continue
            shape.add(pt)
            for nbr in _neighbors4(*pt):
                nr, nc = nbr
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1] and allowed[nbr] and nbr not in shape:
                    frontier.append(nbr)
        if len(shape) != target:
            continue
        rows = {r for r, _c in shape}
        cols = {c for _r, c in shape}
        if len(rows) == 1 and len(cols) == 1:
            continue
        return shape
    return None


def _assign_pattern_colors(shape: set[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], list[int]]:
    count = random.randint(2, min(4, len(shape)))
    colors = random.sample(PALETTE, count)
    coords = list(shape)
    random.shuffle(coords)
    payload = colors[:]
    while len(payload) < len(coords):
        payload.append(random.choice(colors))
    random.shuffle(payload)
    return {coord: color for coord, color in zip(coords, payload)}, colors


def _sample_room_spec(
    room_h: int, room_w: int, orient: str, room_r: int, room_c: int, room_rows: int, room_cols: int
):
    if orient == "v":
        mirror = random.randint(2, room_w - 3)
        depth = min(mirror, room_w - mirror - 1)
        if depth < 2:
            return None
        source_side = random.choice(("left", "right"))
        if room_rows == 1:
            map_side = random.choice(("top", "bottom"))
        else:
            map_side = "top" if room_r == 0 else "bottom"
        row_range = range(1, room_h) if map_side == "top" else range(0, room_h - 1)
        if source_side == "left":
            col_range = range(mirror - depth, mirror)
        else:
            col_range = range(mirror + 1, mirror + depth + 1)
        allowed = np.zeros((room_h, room_w), dtype=bool)
        allowed[np.ix_(list(row_range), list(col_range))] = True
        shape = _grow_shape(allowed)
        if shape is None:
            return None
        pattern, pattern_colors = _assign_pattern_colors(shape)
        map_count = random.randint(1, min(3, len(pattern_colors)))
        sources = random.sample(pattern_colors, map_count)
        targets = random.sample([c for c in PALETTE if c not in pattern_colors], map_count)
        slots = _pick_spaced_positions([c for c in range(1, room_w - 1) if c != mirror], map_count)
        if slots is None:
            return None
    else:
        mirror = random.randint(2, room_h - 3)
        depth = min(mirror, room_h - mirror - 1)
        if depth < 2:
            return None
        source_side = random.choice(("top", "bottom"))
        if room_cols == 1:
            map_side = random.choice(("left", "right"))
        else:
            map_side = "left" if room_c == 0 else "right"
        col_range = range(1, room_w) if map_side == "left" else range(0, room_w - 1)
        if source_side == "top":
            row_range = range(mirror - depth, mirror)
        else:
            row_range = range(mirror + 1, mirror + depth + 1)
        allowed = np.zeros((room_h, room_w), dtype=bool)
        allowed[np.ix_(list(row_range), list(col_range))] = True
        shape = _grow_shape(allowed)
        if shape is None:
            return None
        pattern, pattern_colors = _assign_pattern_colors(shape)
        map_count = random.randint(1, min(3, len(pattern_colors)))
        sources = random.sample(pattern_colors, map_count)
        targets = random.sample([c for c in PALETTE if c not in pattern_colors], map_count)
        slots = _pick_spaced_positions([r for r in range(1, room_h - 1) if r != mirror], map_count)
        if slots is None:
            return None

    return {
        "orient": orient,
        "mirror": mirror,
        "map_side": map_side,
        "pattern": pattern,
        "mapping": dict(zip(sources, targets)),
        "markers": list(zip(slots, sources, targets)),
    }


def _render_pair(
    room_heights: list[int], room_widths: list[int], specs: list[list[dict]]
) -> tuple[np.ndarray, np.ndarray]:
    grid_h = sum(room_heights) + len(room_heights) + 1
    grid_w = sum(room_widths) + len(room_widths) + 1
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = np.full((grid_h, grid_w), BLACK, dtype=int)

    row_bounds = [0]
    for size in room_heights:
        row_bounds.append(row_bounds[-1] + size + 1)
    col_bounds = [0]
    for size in room_widths:
        col_bounds.append(col_bounds[-1] + size + 1)

    for r in row_bounds:
        inp[r, :] = GREY
        out[r, :] = GREY
    for c in col_bounds:
        inp[:, c] = GREY
        out[:, c] = GREY

    for i, room_h in enumerate(room_heights):
        for j, room_w in enumerate(room_widths):
            spec = specs[i][j]
            top = row_bounds[i] + 1
            left = col_bounds[j] + 1

            if spec["orient"] == "v":
                mirror_c = left + spec["mirror"]
                inp[top : top + room_h, mirror_c] = MAROON
                out[top : top + room_h, mirror_c] = MAROON
            else:
                mirror_r = top + spec["mirror"]
                inp[mirror_r, left : left + room_w] = MAROON
                out[mirror_r, left : left + room_w] = MAROON

            for (lr, lc), color in spec["pattern"].items():
                inp[top + lr, left + lc] = color
                out[top + lr, left + lc] = color

            for slot, source, target in spec["markers"]:
                side = spec["map_side"]
                if side == "top":
                    inp[top - 1, left + slot] = source
                    out[top - 1, left + slot] = source
                    inp[top, left + slot] = target
                    out[top, left + slot] = target
                elif side == "bottom":
                    inp[top + room_h, left + slot] = source
                    out[top + room_h, left + slot] = source
                    inp[top + room_h - 1, left + slot] = target
                    out[top + room_h - 1, left + slot] = target
                elif side == "left":
                    inp[top + slot, left - 1] = source
                    out[top + slot, left - 1] = source
                    inp[top + slot, left] = target
                    out[top + slot, left] = target
                else:
                    inp[top + slot, left + room_w] = source
                    out[top + slot, left + room_w] = source
                    inp[top + slot, left + room_w - 1] = target
                    out[top + slot, left + room_w - 1] = target

            for (lr, lc), color in spec["pattern"].items():
                if spec["orient"] == "v":
                    rr, cc = lr, 2 * spec["mirror"] - lc
                else:
                    rr, cc = 2 * spec["mirror"] - lr, lc
                out[top + rr, left + cc] = spec["mapping"].get(color, color)

    return inp, out


def _sample_layout() -> tuple[list[int], list[int]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        room_rows = random.choice((1, 2))
        room_cols = random.choice((1, 2))
        room_heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_rows)]
        room_widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_cols)]
        grid_h = sum(room_heights) + room_rows + 1
        grid_w = sum(room_widths) + room_cols + 1
        if grid_h <= GRID_SIDE_MAX and grid_w <= GRID_SIDE_MAX:
            return room_heights, room_widths
    return None


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    room_heights, room_widths = layout
    room_rows = len(room_heights)
    room_cols = len(room_widths)
    room_count = room_rows * room_cols

    orientations = [random.choice(("v", "h")) for _ in range(room_count)]
    if room_count >= 2 and len(set(orientations)) == 1:
        flip = random.randrange(room_count)
        orientations[flip] = "h" if orientations[flip] == "v" else "v"

    specs: list[list[dict]] = []
    idx = 0
    for i, room_h in enumerate(room_heights):
        row_specs: list[dict] = []
        for j, room_w in enumerate(room_widths):
            spec = _sample_room_spec(room_h, room_w, orientations[idx], i, j, room_rows, room_cols)
            if spec is None:
                return None
            row_specs.append(spec)
            idx += 1
        specs.append(row_specs)

    inp, expected = _render_pair(room_heights, room_widths, specs)
    if np.array_equal(inp, expected):
        return None
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid mirrored-room sample")
