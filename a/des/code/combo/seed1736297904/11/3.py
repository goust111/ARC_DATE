from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Candidate:
    r0: int
    c0: int
    above: tuple[int, ...]


def _background_color(grid: np.ndarray) -> int:
    if bool((grid == 0).any()):
        return 0
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return np.pad(pref, ((1, 0), (1, 0)))


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(pref[r1, c1] - pref[r0, c1] - pref[r1, c0] + pref[r0, c0])


def _infer_size(grid: np.ndarray, colors: list[int]) -> int:
    spans = []
    for color in colors:
        pts = np.argwhere(grid == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        spans.append(int(max(r1 - r0 + 1, c1 - c0 + 1)))
    return max(spans)


def _enumerate_candidates(
    grid: np.ndarray,
    bg: int,
    color: int,
    size: int,
    bg_pref: np.ndarray,
) -> list[Candidate]:
    h, w = grid.shape
    pts = np.argwhere(grid == color)
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)

    row_start = max(0, int(rmax) - size + 1)
    row_stop = min(int(rmin), h - size)
    col_start = max(0, int(cmax) - size + 1)
    col_stop = min(int(cmin), w - size)

    out: list[Candidate] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(row_start, row_stop + 1):
        for c0 in range(col_start, col_stop + 1):
            if _rect_sum(bg_pref, r0, c0, r0 + size, c0 + size) != 0:
                continue
            key = (r0, c0)
            if key in seen:
                continue
            seen.add(key)
            patch = grid[r0 : r0 + size, c0 : c0 + size]
            above = tuple(
                sorted(int(v) for v in np.unique(patch) if int(v) not in (bg, color))
            )
            out.append(Candidate(r0=r0, c0=c0, above=above))
    return out


def _partial_acyclic(assigned: dict[int, Candidate]) -> bool:
    state: dict[int, int] = {}

    def dfs(color: int) -> bool:
        cur = state.get(color, 0)
        if cur == 1:
            return False
        if cur == 2:
            return True
        state[color] = 1
        for higher in assigned[color].above:
            if higher in assigned and not dfs(higher):
                return False
        state[color] = 2
        return True

    return all(dfs(color) for color in assigned)


def _partial_visible_ok(
    grid: np.ndarray,
    size: int,
    assigned: dict[int, Candidate],
    visible_cells: dict[int, np.ndarray],
) -> bool:
    for lower, cand in assigned.items():
        pts = visible_cells[lower]
        for higher in cand.above:
            if higher not in assigned:
                continue
            other = assigned[higher]
            rr = pts[:, 0]
            cc = pts[:, 1]
            covered = (
                (other.r0 <= rr)
                & (rr < other.r0 + size)
                & (other.c0 <= cc)
                & (cc < other.c0 + size)
            )
            if bool(covered.any()):
                return False
    return True


def _topological_order(assigned: dict[int, Candidate]) -> list[int] | None:
    lowers: dict[int, set[int]] = {color: set() for color in assigned}
    indeg = {color: 0 for color in assigned}
    for lower, cand in assigned.items():
        for higher in cand.above:
            if higher not in assigned:
                return None
            if lower not in lowers[higher]:
                lowers[higher].add(lower)
                indeg[lower] += 1

    ready = [color for color, deg in indeg.items() if deg == 0]
    ready.sort(key=lambda color: (assigned[color].r0, assigned[color].c0))
    order: list[int] = []
    while ready:
        color = ready.pop(0)
        order.append(color)
        for lower in sorted(lowers[color], key=lambda x: (assigned[x].r0, assigned[x].c0)):
            indeg[lower] -= 1
            if indeg[lower] == 0:
                ready.append(lower)
        ready.sort(key=lambda x: (assigned[x].r0, assigned[x].c0))

    if len(order) != len(assigned):
        return None
    return order


def _render_input(
    shape: tuple[int, int],
    bg: int,
    size: int,
    assigned: dict[int, Candidate],
    top_to_bottom: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in reversed(top_to_bottom):
        cand = assigned[color]
        out[cand.r0 : cand.r0 + size, cand.c0 : cand.c0 + size] = color
    return out


def _solve_scene(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors, "expected at least one non-background square"

    size = _infer_size(g, colors)
    bg_pref = _prefix_sum(g == bg)
    candidates_by_color = {
        color: _enumerate_candidates(g, bg, color, size, bg_pref)
        for color in colors
    }
    assert all(candidates_by_color[color] for color in colors), "failed to place a square"

    visible_cells = {color: np.argwhere(g == color) for color in colors}
    search_order = sorted(colors, key=lambda color: (len(candidates_by_color[color]), color))
    assigned: dict[int, Candidate] = {}
    solved: tuple[int, list[int]] | None = None

    def dfs(idx: int) -> None:
        nonlocal solved
        if solved is not None:
            return
        if idx == len(search_order):
            topo = _topological_order(assigned)
            if topo is None:
                return
            rendered = _render_input(g.shape, bg, size, assigned, topo)
            if np.array_equal(rendered, g):
                solved = (size, topo)
            return

        color = search_order[idx]
        for cand in candidates_by_color[color]:
            assigned[color] = cand
            if _partial_acyclic(assigned) and _partial_visible_ok(g, size, assigned, visible_cells):
                dfs(idx + 1)
            assigned.pop(color)

    dfs(0)
    assert solved is not None, "no valid square reconstruction found"
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped equal-sized filled squares, sort them from top to bottom,
    and lay the complete squares out from left to right.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    size, order = _solve_scene(g)
    out = np.full((size, size * len(order)), bg, dtype=int)
    for i, color in enumerate(order):
        out[:, i * size : (i + 1) * size] = color
    return out


_BG = 0
_COLORS = tuple(range(1, 10))
_SIZE_RANGE = (3, 6)
_COUNT_RANGE = (2, 5)
_PAD_RANGE = (1, 4)
_MAX_ATTEMPTS = 400
_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _build_chain_positions(
    rng: np.random.Generator, size: int, count: int
) -> list[tuple[int, int]]:
    direction = _DIAGONALS[int(rng.integers(len(_DIAGONALS)))]
    positions = [(0, 0)]
    for _ in range(1, count):
        step_r = int(rng.integers(1, size))
        step_c = int(rng.integers(1, size))
        prev_r, prev_c = positions[-1]
        positions.append(
            (prev_r + direction[0] * step_r, prev_c + direction[1] * step_c)
        )
    return positions


def _shift_positions(
    rng: np.random.Generator, positions: list[tuple[int, int]], size: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)

    pad_top = int(rng.integers(_PAD_RANGE[0], _PAD_RANGE[1] + 1))
    pad_bottom = int(rng.integers(_PAD_RANGE[0], _PAD_RANGE[1] + 1))
    pad_left = int(rng.integers(_PAD_RANGE[0], _PAD_RANGE[1] + 1))
    pad_right = int(rng.integers(_PAD_RANGE[0], _PAD_RANGE[1] + 1))

    dr = pad_top - min_r
    dc = pad_left - min_c
    shifted = [(r + dr, c + dc) for r, c in positions]
    height = max_r - min_r + size + pad_top + pad_bottom
    width = max_c - min_c + size + pad_left + pad_right
    return shifted, (height, width)


def _render_chain(
    size: int,
    colors_bottom_to_top: list[int],
    positions_bottom_to_top: list[tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for color, (r0, c0) in zip(colors_bottom_to_top, positions_bottom_to_top):
        grid[r0 : r0 + size, c0 : c0 + size] = color
    return grid


def _expected_output(size: int, colors_bottom_to_top: list[int]) -> np.ndarray:
    top_to_bottom = list(reversed(colors_bottom_to_top))
    out = np.full((size, size * len(top_to_bottom)), _BG, dtype=int)
    for i, color in enumerate(top_to_bottom):
        out[:, i * size : (i + 1) * size] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.integers(_SIZE_RANGE[0], _SIZE_RANGE[1] + 1))
    count = int(rng.integers(_COUNT_RANGE[0], _COUNT_RANGE[1] + 1))
    positions = _build_chain_positions(rng, size, count)
    shifted, shape = _shift_positions(rng, positions, size)
    colors = list(map(int, rng.choice(_COLORS, size=count, replace=False)))

    grid = _render_chain(size, colors, shifted, shape)
    expected = _expected_output(size, colors)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-square scene")
