import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BLOCK = 2
EDGE_MARGIN = 1
VISIT_CYCLE = (BLUE, YELLOW, GREEN)

GEN_ATTEMPTS = 400
GEN_ROW_RANGE = (20, 29)
GEN_COL_RANGE = (20, 29)
GEN_EXTRA_PATH_MAX = 3
GEN_ISOLATED_MAX = 2


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_bboxes(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(cells) == 4 and r1 == r0 + 1 and c1 == c0 + 1, "expected monochrome 2x2 blocks"
            blocks.append((r0, c0))
    return blocks


def _parse_blocks(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    assert bg == BLACK, "expected a black background"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREY}, "expected only black, red, and grey in the input"

    starts = _component_bboxes(g, RED)
    targets = _component_bboxes(g, GREY)
    assert len(starts) == 1, "expected exactly one red 2x2 start block"
    assert targets, "expected at least one grey target block"
    return starts[0], sorted(targets)


def _move(pos: tuple[int, int], axis: int, sign: int, dist: int) -> tuple[int, int]:
    r, c = pos
    if axis == 0:
        return r + sign * dist, c
    return r, c + sign * dist


def _top_left_distance(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    if a[0] == b[0]:
        return abs(a[1] - b[1])
    if a[1] == b[1]:
        return abs(a[0] - b[0])
    return None


def _alignment_axis(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    if a[0] == b[0]:
        return 1
    if a[1] == b[1]:
        return 0
    return None


def _visible(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    if a[0] == b[0]:
        c0 = min(a[1], b[1]) + BLOCK
        c1 = max(a[1], b[1])
        return np.all(grid[a[0] : a[0] + BLOCK, c0:c1] == BLACK)
    if a[1] == b[1]:
        r0 = min(a[0], b[0]) + BLOCK
        r1 = max(a[0], b[0])
        return np.all(grid[r0:r1, a[1] : a[1] + BLOCK] == BLACK)
    return False


def _walk_debug(grid: np.ndarray):
    start, targets = _parse_blocks(grid)
    unvisited = list(targets)
    current = start
    order: list[tuple[int, int]] = []
    steps = []

    while True:
        aligned = []
        visible = []
        for block in unvisited:
            axis = _alignment_axis(current, block)
            if axis is None:
                continue
            distance = _top_left_distance(current, block)
            assert distance is not None
            item = {"block": block, "axis": axis, "distance": distance}
            aligned.append(item)
            if _visible(grid, current, block):
                visible.append(item)

        if not visible:
            steps.append({"current": current, "chosen": None, "visible": [], "aligned": aligned})
            break

        min_distance = min(item["distance"] for item in visible)
        nearest = [item for item in visible if item["distance"] == min_distance]
        assert len(nearest) == 1, "expected a unique nearest visible target"

        chosen = nearest[0]
        steps.append({"current": current, "chosen": chosen, "visible": visible, "aligned": aligned})
        order.append(chosen["block"])
        unvisited.remove(chosen["block"])
        current = chosen["block"]

    return start, targets, order, steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the grey blocks reached by a nearest-visible walk from the red block with a blue-yellow-green cycle.

    Input structure:
    - The grid has a black background, one red 2x2 start block, and several grey 2x2 target blocks.
    - Two blocks interact only when they share the same two-row band or the same two-column band.
    - Any colored pixels sitting between aligned blocks in that band block line of sight.

    Rule:
    - Start at the red block.
    - Repeatedly choose the unique nearest unvisited grey block that is aligned with the current block and visible through an all-black corridor in that shared band.
    - Continue the walk from the chosen grey block until no such grey block exists.
    - Do not draw the walk itself; instead recolor the visited grey blocks in visit order as blue, then yellow, then green, repeating cyclically.

    Key insight:
    - The output only encodes visit order, so the solver must mentally replay the whole greedy traversal while respecting both sight blocking and nearest-target priority.
    """
    g = np.asarray(grid, dtype=int)
    _start, targets, order, _steps = _walk_debug(g)
    out = g.copy()
    visited = set(order)
    for index, block in enumerate(order):
        color = VISIT_CYCLE[index % len(VISIT_CYCLE)]
        r, c = block
        out[r : r + BLOCK, c : c + BLOCK] = color
    for block in targets:
        if block in visited:
            continue
        r, c = block
        out[r : r + BLOCK, c : c + BLOCK] = GREY
    return out


def _in_bounds(pos: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = shape
    return EDGE_MARGIN <= r <= h - BLOCK - EDGE_MARGIN and EDGE_MARGIN <= c <= w - BLOCK - EDGE_MARGIN


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + BLOCK - 1, ac0 + BLOCK - 1
    br1, bc1 = br0 + BLOCK - 1, bc0 + BLOCK - 1
    row_separated = ar1 + 1 < br0 or br1 + 1 < ar0
    col_separated = ac1 + 1 < bc0 or bc1 + 1 < ac0
    return not (row_separated or col_separated)


def _can_place(pos: tuple[int, int], occupied: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    if not _in_bounds(pos, shape):
        return False
    return all(not _too_close(pos, other) for other in occupied)


def _shares_band_with_any(pos: tuple[int, int], blocks: list[tuple[int, int]], *, allow: set[tuple[int, int]] | None = None) -> bool:
    allow = allow or set()
    for block in blocks:
        if block in allow:
            continue
        if pos[0] == block[0] or pos[1] == block[1]:
            return True
    return False


def _render_grid(shape: tuple[int, int], start: tuple[int, int], targets: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    sr, sc = start
    grid[sr : sr + BLOCK, sc : sc + BLOCK] = RED
    for r, c in targets:
        grid[r : r + BLOCK, c : c + BLOCK] = GREY
    return grid


def _extend_path(
    rng: np.random.Generator,
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    extras: list[tuple[int, int]],
    last_axis: int,
) -> list[tuple[int, int]] | None:
    occupied = path + extras
    count = int(rng.integers(0, GEN_EXTRA_PATH_MAX + 1))
    for _ in range(count):
        current = path[-1]
        axis = 1 - last_axis
        signs = [int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))]
        dists = rng.permutation(np.arange(3, 8))
        placed = None
        for sign in signs:
            for dist in dists:
                candidate = _move(current, axis, sign, int(dist))
                if not _can_place(candidate, occupied, shape):
                    continue
                if _shares_band_with_any(candidate, occupied, allow={current}):
                    continue
                placed = candidate
                break
            if placed is not None:
                break
        if placed is None:
            return None
        path.append(placed)
        occupied.append(placed)
        last_axis = axis
    return path


def _add_isolated_targets(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occupied = list(blocks)
    isolated = []
    count = int(rng.integers(0, GEN_ISOLATED_MAX + 1))
    for _ in range(count):
        placed = None
        for _attempt in range(80):
            candidate = (int(rng.integers(EDGE_MARGIN, shape[0] - BLOCK - EDGE_MARGIN + 1)),
                         int(rng.integers(EDGE_MARGIN, shape[1] - BLOCK - EDGE_MARGIN + 1)))
            if not _can_place(candidate, occupied, shape):
                continue
            if _shares_band_with_any(candidate, occupied):
                continue
            placed = candidate
            break
        if placed is None:
            return None
        isolated.append(placed)
        occupied.append(placed)
    return isolated


def _has_open_choice(steps) -> bool:
    for step in steps:
        visible = step["visible"]
        chosen = step["chosen"]
        if chosen is None or len(visible) < 2:
            continue
        if sum(item["distance"] == chosen["distance"] for item in visible) == 1:
            return True
    return False


def _has_blocked_closer(steps) -> bool:
    for step in steps:
        chosen = step["chosen"]
        if chosen is None:
            continue
        visible_blocks = {item["block"] for item in step["visible"]}
        for item in step["aligned"]:
            if item["block"] in visible_blocks:
                continue
            if item["distance"] < chosen["distance"]:
                return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    start, targets, order, steps = _walk_debug(grid)
    if len(order) < 2:
        return False
    if len(order) >= len(targets):
        return False
    if not _has_open_choice(steps):
        return False
    if not _has_blocked_closer(steps):
        return False

    out = main(grid)
    sr, sc = start
    if not np.all(out[sr : sr + BLOCK, sc : sc + BLOCK] == RED):
        return False

    visited = set(order)
    for index, block in enumerate(order):
        r, c = block
        if not np.all(out[r : r + BLOCK, c : c + BLOCK] == VISIT_CYCLE[index % len(VISIT_CYCLE)]):
            return False
    for block in targets:
        if block in visited:
            continue
        r, c = block
        if not np.all(out[r : r + BLOCK, c : c + BLOCK] == GREY):
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shape = (
        int(rng.integers(GEN_ROW_RANGE[0], GEN_ROW_RANGE[1])),
        int(rng.integers(GEN_COL_RANGE[0], GEN_COL_RANGE[1])),
    )

    axis_in = int(rng.choice((0, 1)))
    axis_out = 1 - axis_in
    sign_in = int(rng.choice((-1, 1)))
    sign_out = int(rng.choice((-1, 1)))

    dist_prev = 3
    dist_hidden = 4
    dist_next = int(rng.choice((8, 9)))
    dist_open = dist_next + int(rng.choice((2, 3)))

    h, w = shape
    if axis_in == 1:
        min_c = EDGE_MARGIN + (dist_prev + dist_hidden if sign_in == 1 else dist_open)
        max_c = w - BLOCK - EDGE_MARGIN - (dist_open if sign_in == 1 else dist_prev + dist_hidden)
        min_r = EDGE_MARGIN + (dist_next if axis_out == 0 and sign_out == 1 else 0)
        max_r = h - BLOCK - EDGE_MARGIN - (dist_next if axis_out == 0 and sign_out == -1 else 0)
    else:
        min_r = EDGE_MARGIN + (dist_prev + dist_hidden if sign_in == 1 else dist_open)
        max_r = h - BLOCK - EDGE_MARGIN - (dist_open if sign_in == 1 else dist_prev + dist_hidden)
        min_c = EDGE_MARGIN + (dist_next if axis_out == 1 and sign_out == 1 else 0)
        max_c = w - BLOCK - EDGE_MARGIN - (dist_next if axis_out == 1 and sign_out == -1 else 0)

    if min_r > max_r or min_c > max_c:
        return None

    a = (int(rng.integers(min_r, max_r + 1)), int(rng.integers(min_c, max_c + 1)))
    start = _move(a, axis_in, -sign_in, dist_prev)
    hidden = _move(start, axis_in, -sign_in, dist_hidden)
    open_far = _move(a, axis_in, sign_in, dist_open)
    nxt = _move(a, axis_out, sign_out, dist_next)

    core = [start, a, nxt]
    extras = [hidden, open_far]
    all_blocks = core + extras
    if len(set(all_blocks)) != len(all_blocks):
        return None
    if any(not _can_place(pos, [other for other in all_blocks if other != pos], shape) for pos in all_blocks):
        return None

    path = _extend_path(rng, shape, core[:], extras[:], axis_out)
    if path is None:
        return None

    occupied = path + extras
    isolated = _add_isolated_targets(rng, shape, occupied)
    if isolated is None:
        return None

    grid = _render_grid(shape, start, [block for block in path[1:]] + extras + isolated)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
