from __future__ import annotations

import random

import numpy as np


_PALETTE = (1, 2, 3, 4)


class _DSU:
    def __init__(self, n: int) -> None:
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


def _infer_period(grid: np.ndarray) -> int:
    h, w = grid.shape
    del h
    nonzero = np.argwhere(grid != 0)
    assert len(nonzero) > 0, "expected visible clues"

    for p in range(1, w):
        ok = True
        for r, c in nonzero:
            r = int(r)
            c = int(c)
            if c + p < w and grid[r, c + p] != 0 and grid[r, c + p] != grid[r, c]:
                ok = False
                break
        if ok:
            return p
    raise AssertionError("no valid period found")


def _solve_with_constraints(
    grid: np.ndarray,
    period: int,
    *,
    use_periodic: bool,
    use_mirror: bool,
    require_resolved: bool,
) -> tuple[np.ndarray, int]:
    h, w = grid.shape
    dsu = _DSU(h * w)

    def idx(r: int, c: int) -> int:
        return r * w + c

    for r in range(h):
        if use_periodic:
            for c in range(w - period):
                dsu.union(idx(r, c), idx(r, c + period))
        if use_mirror:
            for c in range(w):
                dsu.union(idx(r, c), idx(r, w - 1 - c))

    root_cells: dict[int, list[tuple[int, int]]] = {}
    root_colors: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            root = dsu.find(idx(r, c))
            root_cells.setdefault(root, []).append((r, c))
            if grid[r, c] != 0:
                root_colors.setdefault(root, set()).add(int(grid[r, c]))

    out = np.zeros_like(grid)
    unresolved = 0
    for root, cells in root_cells.items():
        colors = root_colors.get(root, set())
        assert len(colors) <= 1, "conflicting clues inside one equivalence class"
        if not colors:
            unresolved += len(cells)
            continue
        color = next(iter(colors))
        for r, c in cells:
            out[r, c] = color

    if require_resolved:
        assert unresolved == 0, "every orbit must be color-constrained"
    return out, unresolved


def _row_pattern(period: int, width: int, left_color: int, inner_color: int) -> np.ndarray:
    if period == 3:
        tile = np.array([left_color, inner_color, left_color], dtype=int)
    else:
        tile = np.array([left_color, inner_color, inner_color, left_color], dtype=int)
    return tile[np.arange(width) % period]


def _orbit_columns(width: int, period: int) -> tuple[list[int], list[int]]:
    orbit_a = [c for c in range(width) if c % period in {0, period - 1}]
    if period == 3:
        orbit_b = [c for c in range(width) if c % period == 1]
    else:
        orbit_b = [c for c in range(width) if c % period in {1, 2}]
    return orbit_a, orbit_b


def _witness_columns(width: int, period: int) -> list[int]:
    if period == 3:
        starts = [s for s in range(0, width - 3, period)]
        start = random.choice(starts)
        return [start, start + 1, start + 3]
    starts = [s for s in range(0, width - 4, period)]
    start = random.choice(starts)
    return [start, start + 1, start + 2, start + 4]


def _render_output(period: int, height: int, width: int, row_pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r, (left_color, inner_color) in enumerate(row_pairs):
        out[r] = _row_pattern(period, width, left_color, inner_color)
    return out


def _build_input(output: np.ndarray, period: int, witness_row: int) -> np.ndarray:
    height, width = output.shape
    inp = np.zeros_like(output)
    orbit_a, orbit_b = _orbit_columns(width, period)
    witness_cols = _witness_columns(width, period)

    for r in range(height):
        if r == witness_row:
            cols = witness_cols
        else:
            cols = [random.choice(orbit_a), random.choice(orbit_b)]
        for c in cols:
            inp[r, c] = int(output[r, c])
    return inp


def _row_pairs(height: int, colors: tuple[int, ...]) -> list[tuple[int, int]]:
    ordered = [(a, b) for a in colors for b in colors if a != b]
    random.shuffle(ordered)

    pairs: list[tuple[int, int]] = []
    while len(pairs) < height:
        pairs.extend(ordered)
        random.shuffle(ordered)
    pairs = pairs[:height]

    used = {color for pair in pairs for color in pair}
    missing = [color for color in colors if color not in used]
    for i, color in enumerate(missing):
        other = random.choice([c for c in colors if c != color])
        pairs[i] = (color, other) if random.random() < 0.5 else (other, color)
    return pairs


def _minimal_period_of_output(grid: np.ndarray) -> int:
    h, w = grid.shape
    del h
    for p in range(1, w):
        cols = np.arange(w)
        if np.array_equal(grid, grid[:, cols % p]):
            return p
    return w


def _validate_instance(inp: np.ndarray, out: np.ndarray, period: int) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if inp.shape != out.shape:
        return False
    h, w = inp.shape
    if h > 30 or w > 30 or not (w > h):
        return False

    clue_count = int(np.count_nonzero(inp))
    if not (8 <= clue_count <= 15):
        return False

    input_colors = sorted(int(c) for c in np.unique(inp) if c != 0)
    output_colors = sorted(int(c) for c in np.unique(out) if c != 0)
    if not (2 <= len(input_colors) <= 3 and input_colors == output_colors):
        return False
    if np.any(out == 0):
        return False
    if _infer_period(inp) != period:
        return False
    if _minimal_period_of_output(out) != period:
        return False
    if len({tuple(map(int, row)) for row in out}) < 2:
        return False

    solved, unresolved = _solve_with_constraints(
        inp,
        period,
        use_periodic=True,
        use_mirror=True,
        require_resolved=True,
    )
    if unresolved != 0 or not np.array_equal(solved, out):
        return False

    periodic_only, periodic_unresolved = _solve_with_constraints(
        inp,
        period,
        use_periodic=True,
        use_mirror=False,
        require_resolved=False,
    )
    if periodic_unresolved == 0 or np.array_equal(periodic_only, out):
        return False

    mirror_only, mirror_unresolved = _solve_with_constraints(
        inp,
        period,
        use_periodic=False,
        use_mirror=True,
        require_resolved=False,
    )
    if mirror_unresolved == 0 or np.array_equal(mirror_only, out):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every blank cell with the unique color forced by horizontal periodicity and vertical mirror symmetry.

    Input structure
    - The grid is wider than tall, with black background and a sparse set of non-black clue pixels.
    - Every row belongs to a horizontally periodic pattern with an unknown period, and the finished row is also symmetric across the grid's vertical midline.
    - Clues are consistent inside the joint equivalence classes generated by period shifts and left-right reflection.

    Rule
    - Infer the smallest positive horizontal period whose visible clue pairs never disagree at that offset.
    - In each row, connect cells that are period-shift equivalents and cells that are mirror counterparts across the vertical axis.
    - For every resulting transitive-closure class, copy its unique visible clue color to all cells in that class, leaving original clues unchanged.

    Key insight
    - Neither periodic repetition nor mirror symmetry is sufficient on its own; the missing colors are determined only by the closure of both constraints together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    period = _infer_period(g)
    out, _unresolved = _solve_with_constraints(
        g,
        period,
        use_periodic=True,
        use_mirror=True,
        require_resolved=True,
    )
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        period = random.choice((3, 4))
        width = random.choice((period * 2, period * 3))
        max_height = min(6, width - 1)
        min_height = 4
        if max_height < min_height:
            continue
        height = random.randint(min_height, max_height)

        color_count = random.choice((2, 3))
        colors = tuple(sorted(random.sample(_PALETTE, color_count)))
        row_pairs = _row_pairs(height, colors)
        witness_row = random.randrange(height)

        out = _render_output(period, height, width, row_pairs)
        inp = _build_input(out, period, witness_row)

        if _validate_instance(inp, out, period):
            return inp

    raise RuntimeError("failed to generate a valid mirror-period puzzle")
