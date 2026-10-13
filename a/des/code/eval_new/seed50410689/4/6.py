import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
SOURCE_COLORS = [1, 2, 3, 6, 7, 8, 9]

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
TURN_OPTIONS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
SIDES = ("top", "bottom", "left", "right")

MAX_GENERATION_ATTEMPTS = 400


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_boundary(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _line_to_boundary(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    *,
    include_start: bool,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = start
    if include_start:
        out.append(cur)
    while True:
        cur = _add(cur, direction)
        if not _in_bounds(cur, shape):
            break
        out.append(cur)
    return out


def _boundary_positions(side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    assert side == "right"
    return [(r, w - 1) for r in range(1, h - 1)]


def _initial_direction(cell: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return DOWN
    if r == h - 1 and 0 < c < w - 1:
        return UP
    if c == 0 and 0 < r < h - 1:
        return RIGHT
    if c == w - 1 and 0 < r < h - 1:
        return LEFT
    raise AssertionError("source must be on a non-corner boundary cell")


def _turn_layout(
    pos: tuple[int, int],
    direction: tuple[int, int],
    turned: tuple[int, int],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    hit = _add(pos, direction)
    if direction in (UP, DOWN):
        if turned == RIGHT:
            pair = (hit, (hit[0], hit[1] + 1))
            q = (hit[0], hit[1] + 2)
        else:
            assert turned == LEFT
            pair = ((hit[0], hit[1] - 1), hit)
            q = (hit[0], hit[1] - 2)
    else:
        if turned == DOWN:
            pair = (hit, (hit[0] + 1, hit[1]))
            q = (hit[0] + 2, hit[1])
        else:
            assert turned == UP
            pair = ((hit[0] - 1, hit[1]), hit)
            q = (hit[0] - 2, hit[1])
    pair = tuple(sorted(pair))  # stable ordering for storage
    return pair, q


def _mirror_pair_ok(
    pair: tuple[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    a, b = pair
    if not (_in_bounds(a, shape) and _in_bounds(b, shape)):
        return False
    if _is_boundary(a, shape) or _is_boundary(b, shape):
        return False
    if a in mirror_cells or b in mirror_cells or a in blocked or b in blocked:
        return False
    pair_set = {a, b}
    for r, c in pair:
        for dr, dc in (UP, DOWN, LEFT, RIGHT):
            nei = (r + dr, c + dc)
            if nei in mirror_cells and nei not in pair_set:
                return False
    return True


def _path_cells_ok(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
    allowed_overlap: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        if not _in_bounds(cell, shape):
            return False
        if cell in mirror_cells or cell in forbidden:
            return False
        if cell in used_path and cell not in allowed_overlap:
            return False
    return True


def _build_shape0(
    source: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    tail = _line_to_boundary(source, direction, shape, include_start=False)
    if not tail:
        return None
    if not _path_cells_ok(tail, shape, mirror_cells, forbidden, used_path, set()):
        return None
    cells = [source] + tail
    return cells, []


def _build_shape1(
    source: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    first_leg = _line_to_boundary(source, direction, shape, include_start=False)
    if len(first_leg) < 2:
        return None
    turn_dirs = list(TURN_OPTIONS[direction])
    random.shuffle(turn_dirs)
    first_indices = list(range(len(first_leg)))
    random.shuffle(first_indices)
    blocked = forbidden | used_path
    for turned in turn_dirs:
        for idx in first_indices:
            seg1 = first_leg[: idx + 1]
            pair, q = _turn_layout(seg1[-1], direction, turned)
            if not _mirror_pair_ok(pair, shape, mirror_cells, blocked):
                continue
            tail = _line_to_boundary(q, turned, shape, include_start=True)
            if not tail:
                continue
            full = [source] + seg1 + tail
            if len(set(full)) != len(full):
                continue
            if not _path_cells_ok(seg1 + tail, shape, mirror_cells, forbidden, used_path, set()):
                continue
            return full, [pair]
    return None


def _build_shape2(
    source: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    first_leg = _line_to_boundary(source, direction, shape, include_start=False)
    if len(first_leg) < 2:
        return None
    first_indices = list(range(len(first_leg)))
    random.shuffle(first_indices)
    first_turns = list(TURN_OPTIONS[direction])
    random.shuffle(first_turns)
    blocked = forbidden | used_path
    for turned1 in first_turns:
        for idx1 in first_indices:
            seg1 = first_leg[: idx1 + 1]
            pair1, q1 = _turn_layout(seg1[-1], direction, turned1)
            if not _mirror_pair_ok(pair1, shape, mirror_cells, blocked):
                continue
            second_leg = _line_to_boundary(q1, turned1, shape, include_start=True)
            if len(second_leg) < 2:
                continue
            second_indices = list(range(len(second_leg)))
            random.shuffle(second_indices)
            second_turns = list(TURN_OPTIONS[turned1])
            random.shuffle(second_turns)
            for turned2 in second_turns:
                for idx2 in second_indices:
                    seg2 = second_leg[: idx2 + 1]
                    pair2, q2 = _turn_layout(seg2[-1], turned1, turned2)
                    local_mirrors = mirror_cells | {pair1[0], pair1[1]}
                    local_blocked = blocked | {pair1[0], pair1[1]}
                    if not _mirror_pair_ok(pair2, shape, local_mirrors, local_blocked):
                        continue
                    tail = _line_to_boundary(q2, turned2, shape, include_start=True)
                    if not tail:
                        continue
                    full = [source] + seg1 + seg2 + tail
                    if len(set(full)) != len(full):
                        continue
                    if not _path_cells_ok(seg1 + seg2 + tail, shape, mirror_cells, forbidden, used_path, set()):
                        continue
                    if any(cell in pair1 or cell in pair2 for cell in full):
                        continue
                    return full, [pair1, pair2]
    return None


def _build_noncross_ray(
    source: tuple[int, int],
    direction: tuple[int, int],
    shape_kind: int,
    shape: tuple[int, int],
    mirror_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    used_path: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    if shape_kind == 0:
        return _build_shape0(source, direction, shape, mirror_cells, forbidden, used_path)
    if shape_kind == 1:
        return _build_shape1(source, direction, shape, mirror_cells, forbidden, used_path)
    return _build_shape2(source, direction, shape, mirror_cells, forbidden, used_path)


def _render_expected(
    shape: tuple[int, int],
    mirrors: list[tuple[tuple[int, int], tuple[int, int]]],
    rays: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for a, b in mirrors:
        out[a] = GREY
        out[b] = GREY

    owner: dict[tuple[int, int], int] = {}
    mixed: set[tuple[int, int]] = set()
    for color, cells in rays.items():
        for cell in cells:
            prev = owner.get(cell)
            if prev is None:
                owner[cell] = color
            elif prev != color:
                mixed.add(cell)

    for cell, color in owner.items():
        out[cell] = YELLOW if cell in mixed else color
    return out


def _build_input(
    shape: tuple[int, int],
    mirrors: list[tuple[tuple[int, int], tuple[int, int]]],
    rays: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    for a, b in mirrors:
        inp[a] = GREY
        inp[b] = GREY
    for color, cells in rays.items():
        inp[cells[0]] = color
    return inp


def _extract_mirrors(g: np.ndarray) -> dict[tuple[int, int], tuple[str, tuple[tuple[int, int], tuple[int, int]]]]:
    h, w = g.shape
    seen: set[tuple[int, int]] = set()
    out: dict[tuple[int, int], tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = {}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != GREY or (r, c) in seen:
                continue
            comp = []
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in (UP, DOWN, LEFT, RIGHT):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            assert len(comp) == 2, "each mirror must be exactly two grey cells"
            comp = sorted(comp)
            (r0, c0), (r1, c1) = comp
            if r0 == r1 and c1 == c0 + 1:
                orient = "h"
            else:
                assert c0 == c1 and r1 == r0 + 1, "mirror must be horizontal or vertical"
                orient = "v"
            pair = (comp[0], comp[1])
            out[pair[0]] = (orient, pair)
            out[pair[1]] = (orient, pair)
    return out


def _paint_path(
    out: np.ndarray,
    owners: dict[tuple[int, int], int],
    mixed: set[tuple[int, int]],
    cell: tuple[int, int],
    color: int,
) -> None:
    prev = owners.get(cell)
    if prev is None:
        owners[cell] = color
        out[cell] = color
    elif prev == color:
        if cell in mixed:
            out[cell] = YELLOW
        else:
            out[cell] = color
    else:
        mixed.add(cell)
        out[cell] = YELLOW


def _reflect(
    hit: tuple[int, int],
    orient: str,
    pair: tuple[tuple[int, int], tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if orient == "h":
        assert direction in (UP, DOWN), "horizontal mirror only reflects vertical rays"
        left, right = sorted(pair, key=lambda x: x[1])
        if hit == left:
            return RIGHT, right
        assert hit == right
        return LEFT, left
    assert orient == "v", "unexpected mirror orientation"
    assert direction in (LEFT, RIGHT), "vertical mirror only reflects horizontal rays"
    top, bottom = sorted(pair, key=lambda x: x[0])
    if hit == top:
        return DOWN, bottom
    assert hit == bottom
    return UP, top


def _trace_ray(
    out: np.ndarray,
    grid: np.ndarray,
    source: tuple[int, int],
    color: int,
    mirrors: dict[tuple[int, int], tuple[str, tuple[tuple[int, int], tuple[int, int]]]],
    owners: dict[tuple[int, int], int],
    mixed: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    direction = _initial_direction(source, (h, w))
    pos = source
    _paint_path(out, owners, mixed, pos, color)
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for _ in range(h * w * 8):
        state = (pos, direction)
        if state in seen_states:
            return
        seen_states.add(state)

        nxt = _add(pos, direction)
        if not _in_bounds(nxt, (h, w)):
            return
        if int(grid[nxt]) == GREY:
            orient, pair = mirrors[nxt]
            direction, pos = _reflect(nxt, orient, pair, direction)
            continue
        pos = nxt
        _paint_path(out, owners, mixed, pos, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored edge light sources through grey two-cell mirrors and paint their reflected trajectories.

    Input structure:
    - The background is black. Non-black, non-grey pixels on the grid boundary are
      colored light sources, and each source launches inward perpendicular to its edge.
    - Inside the grid, every mirror is exactly two grey pixels arranged either
      horizontally or vertically.
    - Aside from boundary sources and grey mirrors, all other pixels are background.

    Rule:
    - Each source paints its own cell and then advances one cell at a time in its
      current direction through background cells.
    - When a vertical-moving ray reaches a horizontal mirror, or a horizontal-moving
      ray reaches a vertical mirror, it turns toward the mirror end farther along
      the mirror's span from the hit cell; mirror pixels themselves stay grey.
    - A ray keeps bouncing until it would leave the grid. If different source colors
      paint the same non-mirror cell, that cell becomes yellow.

    Key insight:
    - The two-cell mirror does not paint over itself: the hit cell chooses which end
      of the mirror the ray exits from, so solving the task is repeated ray tracing
      with orientation-dependent reflections and yellow overlap mixing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape
    assert 1 <= h <= 30 and 1 <= w <= 30

    mirrors = _extract_mirrors(g)

    sources: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val in (BG, GREY):
                continue
            assert _is_boundary((r, c), (h, w)), "all non-grey colored pixels must be boundary sources"
            direction = _initial_direction((r, c), (h, w))
            del direction  # structure check only
            sources.append((r, c, val))

    assert sources, "expected at least one source"

    out = g.copy()
    owners: dict[tuple[int, int], int] = {}
    mixed: set[tuple[int, int]] = set()
    for r, c, color in sources:
        _trace_ray(out, g, (r, c), color, mirrors, owners, mixed)
    return out


def _gen_crossing(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    side = random.choice(SIDES)
    sources = _boundary_positions(side, shape)
    if len(sources) < 2:
        return None

    s1 = random.choice(sources)
    d0 = _initial_direction(s1, shape)
    path1_builder = _build_shape1(s1, d0, shape, set(), set(), set())
    if path1_builder is None:
        return None
    path1, mirrors1 = path1_builder

    orth_cells = []
    if side in ("top", "bottom"):
        src_col = s1[1]
        for cell in path1:
            if cell[0] != s1[0] and cell[1] != src_col and not _is_boundary(cell, shape):
                orth_cells.append(cell)
        coord_get = lambda cell: (0, cell[1]) if side == "top" else (h - 1, cell[1])
    else:
        src_row = s1[0]
        for cell in path1:
            if cell[1] != s1[1] and cell[0] != src_row and not _is_boundary(cell, shape):
                orth_cells.append(cell)
        coord_get = lambda cell: (cell[0], 0) if side == "left" else (cell[0], w - 1)

    orth_cells = [cell for cell in orth_cells if coord_get(cell) != s1]
    if not orth_cells:
        return None
    cross = random.choice(orth_cells)
    s2 = coord_get(cross)
    if s2 not in sources or s2 == s1:
        return None

    d2 = _initial_direction(s2, shape)
    path2 = [s2] + _line_to_boundary(s2, d2, shape, include_start=False)
    used1 = set(path1)
    if cross not in path2:
        return None
    if set(path2) & {s1, s2} != {s2}:
        return None
    if set(path2) & {cell for pair in mirrors1 for cell in pair}:
        return None
    if set(path2) & used1 != {cross}:
        return None

    colors = random.sample(SOURCE_COLORS, 2)
    rays = {colors[0]: path1, colors[1]: path2}
    inp = _build_input(shape, mirrors1, rays)
    expected = _render_expected(shape, mirrors1, rays)

    if random.random() < 0.45:
        available = [cell for cell in sources if cell not in {s1, s2} and cell not in used1 and cell not in set(path2)]
        random.shuffle(available)
        used_path = set(path1) | set(path2)
        mirror_cells = {cell for pair in mirrors1 for cell in pair}
        for s3 in available:
            shape_kind = random.choice([0, 1, 2])
            built = _build_noncross_ray(s3, _initial_direction(s3, shape), shape_kind, shape, mirror_cells, {s1, s2}, used_path)
            if built is None:
                continue
            path3, mirrors3 = built
            color3 = random.choice([c for c in SOURCE_COLORS if c not in colors])
            mirrors_all = mirrors1 + mirrors3
            rays[color3] = path3
            inp = _build_input(shape, mirrors_all, rays)
            expected = _render_expected(shape, mirrors_all, rays)
            break

    return inp, expected


def _gen_noncross(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice(SIDES)
    candidates = _boundary_positions(side, shape)
    random.shuffle(candidates)
    n_sources = random.randint(1, min(3, len(candidates)))
    chosen: list[tuple[int, int]] = []
    rays: dict[int, list[tuple[int, int]]] = {}
    mirrors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    mirror_cells: set[tuple[int, int]] = set()
    used_path: set[tuple[int, int]] = set()
    colors = random.sample(SOURCE_COLORS, n_sources)

    shapes = [random.choice([1, 2])] + [random.choice([0, 1, 2]) for _ in range(n_sources - 1)]
    for color, shape_kind in zip(colors, shapes):
        placed = False
        for source in candidates:
            if source in chosen or source in used_path:
                continue
            built = _build_noncross_ray(
                source,
                _initial_direction(source, shape),
                shape_kind,
                shape,
                mirror_cells,
                set(chosen),
                used_path,
            )
            if built is None:
                continue
            path, new_mirrors = built
            placed = True
            chosen.append(source)
            rays[color] = path
            mirrors.extend(new_mirrors)
            for pair in new_mirrors:
                mirror_cells.add(pair[0])
                mirror_cells.add(pair[1])
            used_path.update(path)
            break
        if not placed:
            return None

    inp = _build_input(shape, mirrors, rays)
    expected = _render_expected(shape, mirrors, rays)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 20)
    w = random.randint(10, 20)
    shape = (h, w)
    scenario = _gen_crossing if random.random() < 0.5 else _gen_noncross
    built = scenario(shape)
    if built is None:
        return None
    inp, expected = built
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if np.count_nonzero(inp == GREY) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray input")
