from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 7, 8, 9)
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_pattern_color(color: int) -> bool:
    return color not in {BLACK, RED, GREY}


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _best_supported_translations(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    pattern = (g != BLACK) & (g != RED) & (g != GREY)
    h, w = g.shape
    best_support = -1
    best: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            a = g[r0:r1, c0:c1]
            b = g[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = pattern[r0:r1, c0:c1] & pattern[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue
            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))

    assert best_support > 0, "expected a supported nonzero translation"
    return best


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _infer_missing_colors(grid: np.ndarray, shifts: Iterable[tuple[int, int]]) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    uf = _UnionFind(h * w)

    for dr, dc in shifts:
        r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
        for r in range(r0, r1):
            rr = r + dr
            for c in range(c0, c1):
                uf.union(r * w + c, rr * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if not _is_pattern_color(color):
                continue
            root = uf.find(r * w + c)
            colors_by_root.setdefault(root, set()).add(color)

    out: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(g == BLACK):
        root = uf.find(int(r) * w + int(c))
        colors = colors_by_root.get(root, set())
        assert len(colors) == 1, "every black gap must have a unique periodic color"
        out[(int(r), int(c))] = next(iter(colors))
    return out


def _marker_rays(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    rays = []
    if r == 0:
        rays.append((1, 0))
    if r == h - 1:
        rays.append((-1, 0))
    if c == 0:
        rays.append((0, 1))
    if c == w - 1:
        rays.append((0, -1))
    assert rays, "markers must lie on the border"
    return rays


def _visible_black_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visible: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == RED):
        rr = int(r)
        cc = int(c)
        for dr, dc in _marker_rays(rr, cc, h, w):
            cr = rr + dr
            cc2 = cc + dc
            while _in_bounds(g.shape, cr, cc2):
                color = int(g[cr, cc2])
                if color == BLACK:
                    visible.add((cr, cc2))
                elif color == GREY or _is_pattern_color(color):
                    break
                cr += dr
                cc2 += dc
    return visible


def main(grid: np.ndarray) -> np.ndarray:
    """Fill only the line-of-sight-visible black gaps of a periodic color field.

    Input structure:
    - A roughly 14x14 grid contains a repeated non-black pattern with some black missing cells.
    - Grey wall pixels sit inside the pattern and red marker pixels sit on the outer border.
    - Each border marker looks straight inward along its row or column.

    Rule:
    - Ignore red markers and grey walls, then find the nonzero translation whose overlapping visible pattern pixels have maximum support with no color conflict.
    - Use that translation and its inverse to infer the unique periodic color of each black gap from its translation-equivalence class.
    - Fill a black gap only if at least one border marker can see it along a straight inward ray with no grey wall and no visible pattern pixel in between.

    Key insight:
    - Periodicity decides what color a gap should become, but border line-of-sight decides whether that otherwise-determined fill is allowed to happen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    best = _best_supported_translations(g)
    assert len(best) == 2 and best[0] == (-best[1][0], -best[1][1]), "expected one bidirectional best translation"

    fill_colors = _infer_missing_colors(g, best)
    visible = _visible_black_cells(g)

    out = g.copy()
    for cell in visible:
        out[cell] = fill_colors[cell]
    return out


def _sequence_has_shift_symmetry(seq: list[int]) -> bool:
    n = len(seq)
    for shift in range(1, n):
        if all(seq[i] == seq[(i + shift) % n] for i in range(n)):
            return True
    return False


def _sample_sequence(rng: np.random.Generator, period: int) -> list[int]:
    while True:
        n_colors = 3 if period == 3 or rng.random() < 0.6 else 2
        palette = rng.choice(PATTERN_COLORS, size=n_colors, replace=False).tolist()
        seq = [int(palette[int(rng.integers(0, len(palette)))]) for _ in range(period)]
        if len(set(seq)) < 2 or len(set(seq)) < len(palette):
            continue
        if _sequence_has_shift_symmetry(seq):
            continue
        return seq


def _build_periodic_field(
    rng: np.random.Generator, h: int, w: int
) -> tuple[np.ndarray, tuple[int, int], str]:
    for _ in range(200):
        orientation = str(rng.choice(["h", "v"]))
        period = int(rng.choice([3, 4]))
        seq = _sample_sequence(rng, period)
        if orientation == "h":
            offsets = rng.integers(0, period, size=h)
            if len(set(map(int, offsets))) < 2:
                continue
            full = np.zeros((h, w), dtype=int)
            for r in range(h):
                for c in range(w):
                    full[r, c] = seq[(c + int(offsets[r])) % period]
            vector = (0, period)
        else:
            offsets = rng.integers(0, period, size=w)
            if len(set(map(int, offsets))) < 2:
                continue
            full = np.zeros((h, w), dtype=int)
            for c in range(w):
                for r in range(h):
                    full[r, c] = seq[(r + int(offsets[c])) % period]
            vector = (period, 0)
        if set(_best_supported_translations(full)) == {vector, (-vector[0], -vector[1])}:
            return full, vector, orientation
    raise RuntimeError("failed to build a uniquely translated periodic field")


def _marker_position(side: str, line: int, h: int, w: int) -> tuple[int, int]:
    if side == "L":
        return line, 0
    if side == "R":
        return line, w - 1
    if side == "T":
        return 0, line
    if side == "B":
        return h - 1, line
    raise ValueError(side)


def _ray_cell(side: str, line: int, depth: int, h: int, w: int) -> tuple[int, int]:
    if side == "L":
        return line, depth
    if side == "R":
        return line, w - 1 - depth
    if side == "T":
        return depth, line
    if side == "B":
        return h - 1 - depth, line
    raise ValueError(side)


def _axis_length(side: str, h: int, w: int) -> int:
    return w if side in {"L", "R"} else h


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _carve_wall_corridor(
    grid: np.ndarray,
    side: str,
    line: int,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    max_depth = _axis_length(side, h, w) - 2
    visible_len = _randint(rng, 2, min(4, max_depth - 3))
    wall_len = _randint(rng, 1, min(3, max_depth - visible_len - 1))
    blocked_len = _randint(rng, 1, min(3, max_depth - visible_len - wall_len))

    visible = {
        _ray_cell(side, line, depth, h, w)
        for depth in range(1, visible_len + 1)
    }
    wall = {
        _ray_cell(side, line, depth, h, w)
        for depth in range(visible_len + 1, visible_len + wall_len + 1)
    }
    blocked = {
        _ray_cell(side, line, depth, h, w)
        for depth in range(visible_len + wall_len + 1, visible_len + wall_len + blocked_len + 1)
    }
    for r, c in visible | blocked:
        grid[r, c] = BLACK
    for r, c in wall:
        grid[r, c] = GREY
    return visible, blocked, wall


def _carve_clear_then_hidden_corridor(
    grid: np.ndarray,
    side: str,
    line: int,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid.shape
    max_depth = _axis_length(side, h, w) - 2
    visible_len = _randint(rng, 1, min(4, max_depth - 3))
    hidden_len = _randint(rng, 1, min(2, max_depth - visible_len - 2))
    blocker_depth = visible_len + 1

    visible = {
        _ray_cell(side, line, depth, h, w)
        for depth in range(1, visible_len + 1)
    }
    hidden = {
        _ray_cell(side, line, depth, h, w)
        for depth in range(blocker_depth + 1, blocker_depth + hidden_len + 1)
    }
    for r, c in visible | hidden:
        grid[r, c] = BLACK
    return visible, hidden


def _place_extra_walls(
    grid: np.ndarray,
    forbidden_lines: set[int],
    orientation: str,
    count: int,
    rng: np.random.Generator,
) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]
    placed: list[set[tuple[int, int]]] = []

    for _ in range(count):
        success = False
        for _ in range(100):
            shape = shapes[int(rng.integers(0, len(shapes)))]
            max_r = max(dr for dr, _ in shape)
            max_c = max(dc for _, dc in shape)
            r0 = _randint(rng, 1, h - 2 - max_r)
            c0 = _randint(rng, 1, w - 2 - max_c)
            cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if orientation == "h" and any(r in forbidden_lines for r, _ in cells):
                continue
            if orientation == "v" and any(c in forbidden_lines for _, c in cells):
                continue
            if any(grid[r, c] in {BLACK, RED, GREY} for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = GREY
            placed.append(cells)
            success = True
            break
        if not success:
            raise RuntimeError("failed to place extra wall")
    return placed


def _place_extra_hidden_holes(
    grid: np.ndarray,
    forbidden_lines: set[int],
    orientation: str,
    count: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    h, w = grid.shape
    hidden: set[tuple[int, int]] = set()
    for _ in range(count):
        success = False
        for _ in range(100):
            r = _randint(rng, 1, h - 2)
            c = _randint(rng, 1, w - 2)
            if orientation == "h" and r in forbidden_lines:
                continue
            if orientation == "v" and c in forbidden_lines:
                continue
            if grid[r, c] in {BLACK, RED, GREY}:
                continue
            grid[r, c] = BLACK
            hidden.add((r, c))
            success = True
            break
        if not success:
            raise RuntimeError("failed to place extra hidden hole")
    return hidden


def _class_has_visible_pattern(grid: np.ndarray, cell: tuple[int, int], vector: tuple[int, int]) -> bool:
    h, w = grid.shape
    r0, c0 = cell
    for sign in (-1, 1):
        r = r0
        c = c0
        while True:
            r += sign * vector[0]
            c += sign * vector[1]
            if not (0 <= r < h and 0 <= c < w):
                break
            if _is_pattern_color(int(grid[r, c])):
                return True
    return False


def _marker_specs(markers: list[tuple[str, int]], h: int, w: int) -> list[tuple[int, int, int, int]]:
    specs = []
    for side, line in markers:
        r, c = _marker_position(side, line, h, w)
        if side == "L":
            specs.append((r, c, 0, 1))
        elif side == "R":
            specs.append((r, c, 0, -1))
        elif side == "T":
            specs.append((r, c, 1, 0))
        else:
            specs.append((r, c, -1, 0))
    return specs


def _visible_by_direct_check(
    grid: np.ndarray,
    holes: set[tuple[int, int]],
    markers: list[tuple[int, int, int, int]],
) -> set[tuple[int, int]]:
    visible: set[tuple[int, int]] = set()
    for hr, hc in holes:
        for mr, mc, dr, dc in markers:
            if dr != 0:
                if hc != mc:
                    continue
                delta = hr - mr
                if delta * dr <= 0:
                    continue
                steps = abs(delta)
            else:
                if hr != mr:
                    continue
                delta = hc - mc
                if delta * dc <= 0:
                    continue
                steps = abs(delta)
            blocked = False
            for step in range(1, steps):
                rr = mr + step * dr
                cc = mc + step * dc
                color = int(grid[rr, cc])
                if color == GREY or _is_pattern_color(color):
                    blocked = True
                    break
            if not blocked:
                visible.add((hr, hc))
                break
    return visible


def _build_expected_output(
    inp: np.ndarray,
    truth: np.ndarray,
    visible_holes: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in visible_holes:
        out[r, c] = int(truth[r, c])
    return out


def _generate_case() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = _randint(rng, 13, 17)
    w = _randint(rng, 13, 17)
    truth, vector, orientation = _build_periodic_field(rng, h, w)
    inp = truth.copy()

    marker_count = _randint(rng, 2, 4)
    primary_sides = ("L", "R") if orientation == "h" else ("T", "B")
    line_pool = list(range(1, h - 1)) if orientation == "h" else list(range(1, w - 1))
    chosen_lines = rng.choice(line_pool, size=marker_count, replace=False).tolist()
    markers = [(str(rng.choice(primary_sides)), int(line)) for line in chosen_lines]

    visible_holes: set[tuple[int, int]] = set()
    blocked_holes: set[tuple[int, int]] = set()
    pattern_blocked: set[tuple[int, int]] = set()

    v1, b1, _wall1 = _carve_wall_corridor(inp, markers[0][0], markers[0][1], rng)
    v2, b2 = _carve_clear_then_hidden_corridor(inp, markers[1][0], markers[1][1], rng)
    visible_holes |= v1 | v2
    blocked_holes |= b1 | b2
    pattern_blocked |= b2

    extra_hidden = _place_extra_hidden_holes(
        inp,
        {line for _side, line in markers},
        orientation,
        _randint(rng, 0, 2),
        rng,
    )
    blocked_holes |= extra_hidden

    extra_wall_count = _randint(rng, 1, 3)
    _place_extra_walls(inp, {line for _side, line in markers}, orientation, extra_wall_count, rng)

    for side, line in markers:
        r, c = _marker_position(side, line, h, w)
        inp[r, c] = RED

    holes = visible_holes | blocked_holes
    if not visible_holes or not blocked_holes or not pattern_blocked:
        return None
    if not all(_class_has_visible_pattern(inp, hole, vector) for hole in holes):
        return None

    visible_check = _visible_by_direct_check(inp, holes, _marker_specs(markers, h, w))
    if visible_check != visible_holes:
        return None

    best = set(_best_supported_translations(inp))
    if best != {vector, (-vector[0], -vector[1])}:
        return None

    expected = _build_expected_output(inp, truth, visible_holes)
    if np.array_equal(expected, inp):
        return None
    if not np.any(expected == BLACK):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic line-of-sight ARC input")
