from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SOURCE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_orientations(grid: np.ndarray) -> tuple[np.ndarray, int]:
    orient = np.full(grid.shape, -1, dtype=np.int8)
    count = 0
    for comp in _components4(grid == GREY):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        assert len(comp) >= 2
        assert len(rows) == 1 or len(cols) == 1
        value = 0 if len(rows) == 1 else 1
        for r, c in comp:
            orient[r, c] = value
        count += 1
    return orient, count


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    orient, mirror_count = _mirror_orientations(g)
    coverage = np.zeros((h, w), dtype=np.int32)

    sources = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BLACK) & (g != GREY) & (g != YELLOW))]
    stats = {
        "blocked_side_hits": 0,
        "colored_cells": 0,
        "mirror_count": mirror_count,
        "passes": 0,
        "reflections": 0,
        "source_count": len(sources),
        "yellow_cells": 0,
    }

    q = deque()
    for r, c, color in sources:
        for dr, dc in DIRS:
            q.append((r, c, dr, dc, 1, color))

    while q:
        r, c, dr, dc, can_reflect, color = q.popleft()
        cur_r = int(r)
        cur_c = int(c)

        while True:
            nr = cur_r + dr
            nc = cur_c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break

            value = int(g[nr, nc])
            if value == GREY:
                axis = int(orient[nr, nc])
                parallel = (axis == 0 and dc != 0) or (axis == 1 and dr != 0)
                if parallel:
                    stats["passes"] += 1
                    rr = nr
                    cc = nc
                    while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == GREY:
                        rr += dr
                        cc += dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    cur_r = rr - dr
                    cur_c = cc - dc
                    continue

                if can_reflect:
                    stats["reflections"] += 1
                    q.append((cur_r, cur_c, dc, -dr, 0, color))
                    q.append((cur_r, cur_c, -dc, dr, 0, color))
                else:
                    stats["blocked_side_hits"] += 1
                break

            cur_r = nr
            cur_c = nc
            if value == BLACK:
                coverage[cur_r, cur_c] |= 1 << color

    nonzero = coverage != 0
    yellow = nonzero & ((coverage & (coverage - 1)) != 0)
    stats["colored_cells"] = int(np.count_nonzero(nonzero))
    stats["yellow_cells"] = int(np.count_nonzero(yellow))
    return coverage, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast four orthogonal rays from each source, reflect once on mirror sides,
    pass through mirror ends, and color multi-source overlaps yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    coverage, _stats = _trace_rays(g)
    out = g.copy()

    for r, c in np.argwhere((g == BLACK) & (coverage != 0)):
        bits = int(coverage[r, c])
        if bits & (bits - 1):
            out[r, c] = YELLOW
        else:
            out[r, c] = bits.bit_length() - 1

    return out


def _pick(rng: np.random.Generator, seq: list[object] | tuple[object, ...]) -> object:
    return seq[int(rng.integers(0, len(seq)))]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _can_place(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c] or int(grid[r, c]) != BLACK or (r, c) in reserved_clear:
            return False
    return True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
    _mark_blocked(blocked, cells)


def _line_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0 + step, c1, step):
            cells.append((r0, c))
    elif c0 == c1:
        step = 1 if r1 > r0 else -1
        for r in range(r0 + step, r1, step):
            cells.append((r, c0))
    return cells


def _contact_cell(source: tuple[int, int], cells: list[tuple[int, int]], axis: int) -> tuple[int, int]:
    sr, sc = source
    if axis == 0:
        rows = sorted(r for r, _ in cells)
        if rows[0] > sr:
            return rows[0], sc
        return rows[-1], sc
    cols = sorted(c for _, c in cells)
    if cols[0] > sc:
        return sr, cols[0]
    return sr, cols[-1]


def _reflect_candidates(r: int, c: int, axis: int, h: int, w: int) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if axis == 0:
        for sign in (-1, 1):
            for dist in range(2, 5):
                mr = r + sign * dist
                if not (0 <= mr < h):
                    continue
                max_len = min(6, w)
                for length in range(2, max_len + 1):
                    start_lo = max(0, c - length + 1)
                    start_hi = min(c, w - length)
                    for start in range(start_lo, start_hi + 1):
                        candidates.append([(mr, cc) for cc in range(start, start + length)])
    else:
        for sign in (-1, 1):
            for dist in range(2, 5):
                mc = c + sign * dist
                if not (0 <= mc < w):
                    continue
                max_len = min(6, h)
                for length in range(2, max_len + 1):
                    start_lo = max(0, r - length + 1)
                    start_hi = min(r, h - length)
                    for start in range(start_lo, start_hi + 1):
                        candidates.append([(rr, mc) for rr in range(start, start + length)])
    return candidates


def _pass_candidates(
    r: int,
    c: int,
    axis: int,
    h: int,
    w: int,
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    if axis == 0:
        for sign in (-1, 1):
            for dist in range(2, 5):
                if sign < 0:
                    bottom = r - dist
                    max_len = min(5, bottom)
                    for length in range(2, max_len + 1):
                        top = bottom - length + 1
                        cells = [(rr, c) for rr in range(top, bottom + 1)]
                        candidates.append((cells, (top - 1, c)))
                else:
                    top = r + dist
                    max_len = min(5, h - 1 - top)
                    for length in range(2, max_len + 1):
                        bottom = top + length - 1
                        cells = [(rr, c) for rr in range(top, bottom + 1)]
                        candidates.append((cells, (bottom + 1, c)))
    else:
        for sign in (-1, 1):
            for dist in range(2, 5):
                if sign < 0:
                    right = c - dist
                    max_len = min(5, right)
                    for length in range(2, max_len + 1):
                        left = right - length + 1
                        cells = [(r, cc) for cc in range(left, right + 1)]
                        candidates.append((cells, (r, left - 1)))
                else:
                    left = c + dist
                    max_len = min(5, w - 1 - left)
                    for length in range(2, max_len + 1):
                        right = left + length - 1
                        cells = [(r, cc) for cc in range(left, right + 1)]
                        candidates.append((cells, (r, right + 1)))
    return candidates


def _choose_cells(
    rng: np.random.Generator,
    candidates: list[list[tuple[int, int]]],
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved_clear: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if not candidates:
        return None
    order = rng.permutation(len(candidates))
    for idx in order:
        cells = candidates[int(idx)]
        if _can_place(grid, blocked, cells, reserved_clear):
            return cells
    return None


def _choose_pass_cells(
    rng: np.random.Generator,
    candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]],
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved_clear: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    if not candidates:
        return None
    order = rng.permutation(len(candidates))
    h, w = grid.shape
    for idx in order:
        cells, beyond = candidates[int(idx)]
        br, bc = beyond
        if not (0 <= br < h and 0 <= bc < w):
            continue
        if (br, bc) in reserved_clear or int(grid[br, bc]) != BLACK:
            continue
        if _can_place(grid, blocked, cells, reserved_clear):
            return cells, beyond
    return None


def _pair_positions(rng: np.random.Generator, h: int, w: int, axis: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(100):
        if axis == 0:
            r = int(rng.integers(1, h - 1))
            c0 = int(rng.integers(1, w - 5))
            c1 = int(rng.integers(c0 + 4, w - 1))
            return (r, c0), (r, c1)
        c = int(rng.integers(1, w - 1))
        r0 = int(rng.integers(1, h - 5))
        r1 = int(rng.integers(r0 + 4, h - 1))
        return (r0, c), (r1, c)
    return None


def _place_extra_source(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved_clear: set[tuple[int, int]],
    color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        cells = [(r, c)]
        if _can_place(grid, blocked, cells, reserved_clear):
            _place_object(grid, blocked, cells, color)
            return True
    return False


def _place_extra_mirror(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    reserved_clear: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(150):
        axis = int(rng.integers(0, 2))
        if axis == 0:
            length = int(rng.integers(2, min(7, w) + 1))
            r = int(rng.integers(0, h))
            c0 = int(rng.integers(0, w - length + 1))
            cells = [(r, c) for c in range(c0, c0 + length)]
        else:
            length = int(rng.integers(2, min(7, h) + 1))
            c = int(rng.integers(0, w))
            r0 = int(rng.integers(0, h - length + 1))
            cells = [(r, c) for r in range(r0, r0 + length)]
        if _can_place(grid, blocked, cells, reserved_clear):
            _place_object(grid, blocked, cells, GREY)
            return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 25))
    w = int(rng.integers(12, 25))
    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    reserved_clear: set[tuple[int, int]] = set()

    pair_axis = int(rng.integers(0, 2))
    pair = _pair_positions(rng, h, w, pair_axis)
    if pair is None:
        return None
    src_a, src_b = pair

    extra_sources = int(rng.integers(0, 3))
    extra_mirrors = int(rng.integers(0, 3))
    total_sources = 2 + extra_sources
    colors = rng.choice(np.array(SOURCE_COLORS, dtype=int), size=total_sources, replace=False)
    colors = [int(v) for v in colors]

    if not _can_place(grid, blocked, [src_a], reserved_clear):
        return None
    _place_object(grid, blocked, [src_a], colors[0])
    if not _can_place(grid, blocked, [src_b], reserved_clear):
        return None
    _place_object(grid, blocked, [src_b], colors[1])

    for cell in _line_between(src_a, src_b):
        reserved_clear.add(cell)

    ray_axis = 1 - pair_axis
    if bool(rng.integers(0, 2)):
        reflect_src, pass_src = src_a, src_b
    else:
        reflect_src, pass_src = src_b, src_a

    reflect_cells = _choose_cells(
        rng,
        _reflect_candidates(reflect_src[0], reflect_src[1], ray_axis, h, w),
        grid,
        blocked,
        reserved_clear,
    )
    if reflect_cells is None:
        return None
    for cell in _line_between(reflect_src, _contact_cell(reflect_src, reflect_cells, ray_axis)):
        reserved_clear.add(cell)
    _place_object(grid, blocked, reflect_cells, GREY)

    pass_pick = _choose_pass_cells(
        rng,
        _pass_candidates(pass_src[0], pass_src[1], ray_axis, h, w),
        grid,
        blocked,
        reserved_clear,
    )
    if pass_pick is None:
        return None
    pass_cells, beyond = pass_pick
    for cell in _line_between(pass_src, _contact_cell(pass_src, pass_cells, ray_axis)):
        reserved_clear.add(cell)
    reserved_clear.add(beyond)
    _place_object(grid, blocked, pass_cells, GREY)

    for color in colors[2:]:
        if not _place_extra_source(rng, grid, blocked, reserved_clear, color):
            return None

    for _ in range(extra_mirrors):
        if not _place_extra_mirror(rng, grid, blocked, reserved_clear):
            return None

    coverage, stats = _trace_rays(grid)
    if stats["reflections"] < 1 or stats["passes"] < 1 or stats["yellow_cells"] < 2:
        return None
    if stats["colored_cells"] < 10 or stats["mirror_count"] < 2 or stats["source_count"] < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    non_black = np.argwhere(grid != BLACK)
    for r, c in non_black:
        if int(grid[r, c]) == GREY:
            continue
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                if (rr, cc) == (int(r), int(c)):
                    continue
                if int(grid[rr, cc]) == GREY:
                    return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 7")
