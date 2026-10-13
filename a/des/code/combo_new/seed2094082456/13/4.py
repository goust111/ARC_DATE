import numpy as np


GATE_COLORS = (2, 4, 7)
PATH_COLORS = (1, 3, 5, 6, 8, 9)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

GRID_MIN = 14
GRID_MAX_EXCLUSIVE = 31
PATH_COUNT_MIN = 1
PATH_COUNT_MAX_EXCLUSIVE = 5
OUTER_ATTEMPTS = 500
LOCAL_ATTEMPTS = 250


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _walk_open(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]
) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    curr = start
    while True:
        order.append(curr)
        nxt = [cell for cell in neighbors[curr] if cell != prev]
        if not nxt:
            return order
        assert len(nxt) == 1, "open path must continue uniquely"
        prev, curr = curr, nxt[0]


def _walk_cycle(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    first: tuple[int, int],
    size: int,
) -> list[tuple[int, int]]:
    order = [start]
    prev = start
    curr = first
    while curr != start:
        order.append(curr)
        nxt = [cell for cell in neighbors[curr] if cell != prev]
        assert len(nxt) == 1, "cycle must continue uniquely"
        prev, curr = curr, nxt[0]
        assert len(order) <= size, "cycle walk exceeded component size"
    assert len(order) == size, "cycle walk must visit every cell exactly once"
    return order


def _cycle_orientation(order: list[tuple[int, int]]) -> int:
    total = 0
    for i, (r0, c0) in enumerate(order):
        r1, c1 = order[(i + 1) % len(order)]
        total += c0 * r1 - c1 * r0
    return int(total)


def _ordered_component(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], bool]:
    cells = set(coords)
    neighbors = {
        cell: [nb for nb in _neighbors4(cell) if nb in cells]
        for cell in cells
    }
    if len(cells) == 1:
        return [next(iter(cells))], False

    degrees = {cell: len(nbs) for cell, nbs in neighbors.items()}
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    if len(endpoints) == 2 and all(degree in (1, 2) for degree in degrees.values()):
        return _walk_open(neighbors, endpoints[0]), False

    assert all(degree == 2 for degree in degrees.values()), "component must be a simple chain or cycle"
    start = min(cells)
    a, b = neighbors[start]
    order_a = _walk_cycle(neighbors, start, a, len(cells))
    order_b = _walk_cycle(neighbors, start, b, len(cells))
    if _cycle_orientation(order_a) > 0:
        return order_a, True
    return order_b, True


def _recolor_component(
    grid: np.ndarray, order: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    base_colors = {
        int(grid[r, c])
        for r, c in order
        if int(grid[r, c]) not in GATE_COLORS
    }
    assert len(base_colors) == 1, "each path must have one visible non-gate base color"
    current = next(iter(base_colors))
    painted: dict[tuple[int, int], int] = {}
    for r, c in order:
        painted[(r, c)] = int(current)
        color = int(grid[r, c])
        if color in GATE_COLORS:
            current = int(color)
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint each thin path in segments whose colors switch after gate pixels.

    Input structure:
    - The background is black.
    - Several disconnected one-pixel-wide orthogonal paths appear in non-black colors.
    - Some path cells are overwritten by red, yellow, or orange gate pixels.

    Rule:
    - Treat every non-background connected component as one simple path object.
    - For open paths, traverse from the row-major earlier endpoint; for loops, start at the row-major earliest pixel and follow the clockwise cyclic order.
    - Keep the current path color until a gate is reached; the gate pixel itself is drawn with the incoming color, and the color switches to the gate's color from the next step onward.

    Key insight:
    - Gate pixels are markers embedded inside paths rather than separate objects, so solving requires ordering each path first and only then propagating color changes along that order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = np.full_like(g, int(bg))
    for component in _find_components(g != bg):
        order, _is_cycle = _ordered_component(component)
        painted = _recolor_component(g, order)
        for (r, c), color in painted.items():
            out[r, c] = int(color)
    return out


def _normalize_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in order)
    min_c = min(c for _r, c in order)
    return [(r - min_r, c - min_c) for r, c in order]


def _transform_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in order]
    if bool(RNG.integers(0, 2)):
        coords = [(r, -c) for r, c in coords]
    turns = int(RNG.integers(0, 4))
    for _ in range(turns):
        coords = [(c, -r) for r, c in coords]
    return _normalize_order(coords)


def _shape_of(order: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in order) + 1, max(c for _r, c in order) + 1


def _trace_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for i, (r0, c0) in enumerate(vertices):
        r1, c1 = vertices[(i + 1) % len(vertices)]
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        steps = max(abs(r1 - r0), abs(c1 - c0))
        for step in range(steps):
            cell = (r0 + dr * step, c0 + dc * step)
            if not order or cell != order[-1]:
                order.append(cell)
    return order


def _generate_open_path() -> list[tuple[int, int]] | None:
    target_len = int(RNG.integers(6, 19))
    require_turn = bool(RNG.integers(0, 2))
    for _ in range(LOCAL_ATTEMPTS):
        order = [(0, 0)]
        occupied = {(0, 0)}
        current = (0, 0)
        prev_dir: tuple[int, int] | None = None
        turns = 0
        while len(order) < target_len:
            if prev_dir is None:
                dirs = CARDINALS[:]
                RNG.shuffle(dirs)
            else:
                left = (-prev_dir[1], prev_dir[0])
                right = (prev_dir[1], -prev_dir[0])
                if RNG.random() < 0.6:
                    dirs = [prev_dir, left, right]
                else:
                    dirs = [left, right, prev_dir]
                if bool(RNG.integers(0, 2)):
                    dirs[1], dirs[2] = dirs[2], dirs[1]
            moved = False
            for dr, dc in dirs:
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in occupied:
                    continue
                if sum(nb in occupied for nb in _neighbors4(nxt)) != 1:
                    continue
                allowed8 = {current}
                if len(order) >= 2:
                    allowed8.add(order[-2])
                if any(nb in occupied and nb not in allowed8 for nb in _neighbors8(nxt)):
                    continue
                occupied.add(nxt)
                order.append(nxt)
                if prev_dir is not None and (dr, dc) != prev_dir:
                    turns += 1
                prev_dir = (dr, dc)
                current = nxt
                moved = True
                break
            if not moved:
                break
        if len(order) != target_len:
            continue
        if require_turn and turns == 0:
            continue
        order = _transform_order(order)
        if max(_shape_of(order)) > 11:
            continue
        return order
    return None


def _generate_loop_path() -> list[tuple[int, int]] | None:
    for _ in range(LOCAL_ATTEMPTS):
        if RNG.random() < 0.5:
            h = int(RNG.integers(3, 9))
            w = int(RNG.integers(3, 9))
            vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
        else:
            h = int(RNG.integers(6, 11))
            w = int(RNG.integers(5, 11))
            indent = int(RNG.integers(1, w - 2))
            top_cut = int(RNG.integers(1, h - 3))
            lower_bound = top_cut + 2
            upper_bound = h - 2
            if lower_bound >= upper_bound:
                continue
            lower_row = int(RNG.integers(lower_bound, upper_bound))
            inner_c = w - 1 - indent
            vertices = [
                (0, 0),
                (0, w - 1),
                (top_cut, w - 1),
                (top_cut, inner_c),
                (lower_row, inner_c),
                (lower_row, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
            ]
        order = _transform_order(_trace_cycle(vertices))
        if max(_shape_of(order)) > 11:
            continue
        return order
    return None


def _max_gate_count(length: int, is_cycle: bool) -> int:
    if is_cycle:
        return min(3, length // 2)
    return min(3, (length - 1) // 2)


def _valid_gate_spacing(
    index: int, chosen: list[int], length: int, is_cycle: bool
) -> bool:
    for other in chosen:
        dist = abs(index - other)
        if is_cycle:
            dist = min(dist, length - dist)
        if dist <= 1:
            return False
    return True


def _choose_gate_positions(length: int, is_cycle: bool, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(length)) if is_cycle else list(range(1, length - 1))
    for _ in range(80):
        RNG.shuffle(candidates)
        chosen: list[int] = []
        for index in candidates:
            if _valid_gate_spacing(index, chosen, length, is_cycle):
                chosen.append(int(index))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _choose_gate_colors(count: int, is_cycle: bool) -> list[int]:
    if count == 0:
        return []
    colors: list[int] = []
    previous = None
    for i in range(count):
        pool = [color for color in GATE_COLORS if color != previous]
        if is_cycle and i == count - 1 and count > 1 and colors[0] in pool and len(pool) > 1:
            pool = [color for color in pool if color != colors[0]]
        color = int(pool[int(RNG.integers(0, len(pool)))])
        colors.append(color)
        previous = color
    return colors


def _mark_forbidden(forbidden: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        forbidden[r0:r1, c0:c1] = True


def _place_paths(
    shape: tuple[int, int],
    paths: list[dict[str, object]],
) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros(shape, dtype=int)
    forbidden = np.zeros(shape, dtype=bool)
    order_paths = sorted(
        paths,
        key=lambda item: _shape_of(item["order"])[0] * _shape_of(item["order"])[1],
        reverse=True,
    )

    for item in order_paths:
        order = item["order"]
        ph, pw = _shape_of(order)
        if ph > h or pw > w:
            return None
        positions = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(w - pw + 1)]
        RNG.shuffle(positions)
        gate_map = {
            int(pos): int(color)
            for pos, color in zip(item["gate_positions"], item["gate_colors"])
        }
        base_color = int(item["base_color"])
        placed = False
        for r0, c0 in positions:
            abs_coords = [(r0 + r, c0 + c) for r, c in order]
            if any(bool(forbidden[r, c]) for r, c in abs_coords):
                continue
            for idx, (r, c) in enumerate(abs_coords):
                grid[r, c] = gate_map.get(idx, base_color)
            _mark_forbidden(forbidden, abs_coords)
            placed = True
            break
        if not placed:
            return None
    return grid


def _sample_path_modes(n_paths: int) -> list[bool]:
    mode = int(RNG.integers(0, 3))
    if mode == 0:
        return [False] * n_paths
    if mode == 1:
        return [True] * n_paths
    if n_paths == 1:
        return [bool(RNG.integers(0, 2))]
    modes = [False, True]
    while len(modes) < n_paths:
        modes.append(bool(RNG.integers(0, 2)))
    RNG.shuffle(modes)
    return modes


def _sample_gate_counts(lengths: list[int], cycle_flags: list[bool]) -> list[int] | None:
    gate_mode = int(RNG.integers(0, 3))
    max_counts = [_max_gate_count(length, is_cycle) for length, is_cycle in zip(lengths, cycle_flags)]
    if any(max_count == 0 for max_count in max_counts):
        return None

    if gate_mode == 0:
        return [0] * len(lengths)

    counts = []
    if gate_mode == 1:
        for max_count in max_counts:
            counts.append(int(RNG.integers(1, max_count + 1)))
        return counts

    if len(lengths) == 1:
        return [int(RNG.integers(0, max_counts[0] + 1))]

    gated_indices = set(RNG.choice(len(lengths), size=int(RNG.integers(1, len(lengths))), replace=False).tolist())
    for i, max_count in enumerate(max_counts):
        if i in gated_indices:
            counts.append(int(RNG.integers(1, max_count + 1)))
        else:
            counts.append(0)
    if all(count == 0 for count in counts):
        counts[0] = 1
    if all(count > 0 for count in counts):
        counts[0] = 0
    return counts


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    width = int(RNG.integers(GRID_MIN, GRID_MAX_EXCLUSIVE))
    n_paths = int(RNG.integers(PATH_COUNT_MIN, PATH_COUNT_MAX_EXCLUSIVE))
    cycle_flags = _sample_path_modes(n_paths)
    gate_counts = None

    for _ in range(40):
        raw_orders: list[list[tuple[int, int]]] = []
        for is_cycle in cycle_flags:
            order = _generate_loop_path() if is_cycle else _generate_open_path()
            if order is None:
                break
            raw_orders.append(order)
        if len(raw_orders) != n_paths:
            continue
        lengths = [len(order) for order in raw_orders]
        gate_counts = _sample_gate_counts(lengths, cycle_flags)
        if gate_counts is not None:
            break
    if gate_counts is None:
        return None

    base_colors = RNG.choice(PATH_COLORS, size=n_paths, replace=False)
    paths: list[dict[str, object]] = []
    for order, is_cycle, gate_count, base_color in zip(raw_orders, cycle_flags, gate_counts, base_colors):
        positions = _choose_gate_positions(len(order), is_cycle, int(gate_count))
        if positions is None:
            return None
        paths.append(
            {
                "order": order,
                "base_color": int(base_color),
                "gate_positions": positions,
                "gate_colors": _choose_gate_colors(len(positions), is_cycle),
            }
        )

    inp = _place_paths((height, width), paths)
    if inp is None:
        return None
    if _most_common_color(inp) != 0:
        return None

    try:
        main(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gated-path puzzle")
