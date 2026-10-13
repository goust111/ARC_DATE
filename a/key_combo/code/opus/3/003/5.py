import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

_LINE_SPECS = ((3, 3), (3, 5), (5, 3))  # (square_size, square_count)
_ISOLATED_SIZES = (3, 5, 7)
_GRID_RANGE = (19, 30)  # exclusive upper bound
_GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    ps = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
    return np.pad(ps, ((1, 0), (1, 0)), constant_values=0)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _make_square(r0: int, c0: int, size: int) -> dict:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "size": size,
        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
    }


def _square_key(square: dict) -> tuple[int, int, int]:
    return int(square["r0"]), int(square["c0"]), int(square["size"])


def _rectangles_overlap(a: dict, b: dict) -> bool:
    return not (
        a["r1"] < b["r0"]
        or b["r1"] < a["r0"]
        or a["c1"] < b["c0"]
        or b["c1"] < a["c0"]
    )


def _adjacent(a: dict, b: dict) -> bool:
    row_gap = max(0, b["r0"] - a["r1"], a["r0"] - b["r1"])
    col_gap = max(0, b["c0"] - a["c1"], a["c0"] - b["c1"])
    return max(row_gap, col_gap) <= 1


def _all_black_squares(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    black_ps = _prefix_sum(grid == BLACK)
    candidates: list[dict] = []
    for size in range(min(h, w), 1, -1):
        area = size * size
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                if _rect_sum(black_ps, r0, c0, r0 + size - 1, c0 + size - 1) == area:
                    candidates.append(_make_square(r0, c0, size))
    return candidates


def _greedy_select(candidates: list[dict], shape: tuple[int, int]) -> list[dict]:
    occupied = np.zeros(shape, dtype=bool)
    selected: list[dict] = []
    for sq in candidates:
        r0, c0, r1, c1 = sq["r0"], sq["c0"], sq["r1"], sq["c1"]
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        selected.append(sq)
    return selected


def _neighbor_data(selected: list[dict]) -> tuple[list[set[int]], list[int]]:
    neighbors = [set() for _ in selected]
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if _adjacent(selected[i], selected[j]):
                neighbors[i].add(j)
                neighbors[j].add(i)
    degrees = [len(nbrs) for nbrs in neighbors]
    return neighbors, degrees


def _corner_infos(square: dict) -> list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]]:
    r0, c0, r1, c1 = square["r0"], square["c0"], square["r1"], square["c1"]
    return [
        ((r0, c0), ((r0 - 1, c0), (r0, c0 - 1))),
        ((r0, c1), ((r0 - 1, c1), (r0, c1 + 1))),
        ((r1, c0), ((r1 + 1, c0), (r1, c0 - 1))),
        ((r1, c1), ((r1 + 1, c1), (r1, c1 + 1))),
    ]


def _closest_corner_indices(square: dict, neighbor: dict) -> tuple[list[int], list[float]]:
    nr, nc = neighbor["center"]
    scored = []
    dists = []
    for idx, (corner, _mark_positions) in enumerate(_corner_infos(square)):
        cr, cc = corner
        dist = math.hypot(cr - nr, cc - nc)
        scored.append((dist, idx))
        dists.append(dist)
    scored.sort()
    return [scored[0][1], scored[1][1]], sorted(dists)


def _base_after_blocks(grid: np.ndarray, selected: list[dict], degrees: list[int]) -> np.ndarray:
    out = grid.copy()
    for idx, sq in enumerate(selected):
        color = YELLOW if degrees[idx] >= 2 else BLUE
        out[sq["r0"] : sq["r1"] + 1, sq["c0"] : sq["c1"] + 1] = color
    return out


def _mark_count(base: np.ndarray, square: dict, corner_indices: list[int]) -> int:
    count = 0
    infos = _corner_infos(square)
    for idx in corner_indices:
        for rr, cc in infos[idx][1]:
            if _in_bounds(base.shape, rr, cc) and int(base[rr, cc]) == BLACK:
                count += 1
    return count


def _render_output(grid: np.ndarray, selected: list[dict], neighbors: list[set[int]], degrees: list[int]) -> np.ndarray:
    out = _base_after_blocks(grid, selected, degrees)
    base = out.copy()
    for idx, sq in enumerate(selected):
        if degrees[idx] >= 2:
            continue
        if degrees[idx] == 0:
            corner_indices = [0, 1, 2, 3]
        else:
            only_neighbor = selected[next(iter(neighbors[idx]))]
            corner_indices, _ = _closest_corner_indices(sq, only_neighbor)
        infos = _corner_infos(sq)
        for corner_idx in corner_indices:
            for rr, cc in infos[corner_idx][1]:
                if _in_bounds(out.shape, rr, cc) and int(base[rr, cc]) == BLACK:
                    out[rr, cc] = RED
    return out


def _analyze(grid: np.ndarray) -> dict:
    candidates = _all_black_squares(grid)
    selected = _greedy_select(candidates, grid.shape)
    neighbors, degrees = _neighbor_data(selected)
    return {
        "candidates": candidates,
        "selected": selected,
        "neighbors": neighbors,
        "degrees": degrees,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Greedily select non-overlapping black squares, then recolor and corner-mark them by adjacency degree.

    Input structure:
    - A black background contains grey obstacle pixels that block candidate square regions.
    - Any all-black axis-aligned square of side >= 2 is a candidate; candidates may overlap.
    - The same open black region can therefore support several competing square candidates of different sizes or offsets.

    Rule:
    - Enumerate every all-black square, sort by side length descending, then by top-left row and column ascending.
    - Traverse that order greedily and keep a square iff it has no pixel overlap with any already selected square.
    - Fill every selected square blue, compute which selected squares are adjacent by boundary 8-neighborhood, and use that degree:
      degree 0 keeps blue and gets red outside-corner marks on all four corners;
      degree 1 keeps blue and gets red marks only on the two corners closest to its unique neighbor's center;
      degree >= 2 is recolored yellow and gets no marks.
    - Red marks are painted only onto in-bounds black background pixels; grey obstacles remain unchanged.

    Key insight:
    - The output depends on a global greedy square packing: obstacle placement changes which squares survive, which changes the adjacency graph, which then changes both recoloring and corner-mark placement.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    return _render_output(g, info["selected"], info["neighbors"], info["degrees"])


def _checker_base(h: int, w: int, off_r: int, off_c: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    grid[((rr - off_r) % 2 == 0) & ((cc - off_c) % 2 == 0)] = GREY
    return grid


def _rects_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _random_aligned_rect(
    h: int,
    w: int,
    rect_h: int,
    rect_w: int,
    row_parity: int,
    col_parity: int,
    blocked: list[tuple[int, int, int, int]],
    margin: int,
) -> tuple[int, int, int, int] | None:
    starts = [
        (r0, c0)
        for r0 in range(1, h - rect_h)
        for c0 in range(1, w - rect_w)
        if r0 % 2 == row_parity and c0 % 2 == col_parity
    ]
    random.shuffle(starts)
    for r0, c0 in starts:
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_rects_close(rect, other, margin) for other in blocked):
            continue
        return rect
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(*_GRID_RANGE)
    w = random.randrange(*_GRID_RANGE)
    off_r = random.randrange(2)
    off_c = random.randrange(2)
    free_row_parity = 1 - off_r
    free_col_parity = 1 - off_c

    line_options = []
    for square_size, square_count in _LINE_SPECS:
        for horizontal in (True, False):
            rect_h = square_size if horizontal else square_size * square_count
            rect_w = square_size * square_count if horizontal else square_size
            if rect_h <= h - 3 and rect_w <= w - 3:
                line_options.append((square_size, square_count, horizontal, rect_h, rect_w))
    if not line_options:
        return None
    square_size, square_count, horizontal, line_h, line_w = random.choice(line_options)

    isolated_options = [
        size
        for size in _ISOLATED_SIZES
        if size <= h - 3 and size <= w - 3
    ]
    if not isolated_options:
        return None
    isolated_size = random.choices(isolated_options, weights=[4 if s == 3 else 3 if s == 5 else 1 for s in isolated_options])[0]

    grid = _checker_base(h, w, off_r, off_c)
    blocked: list[tuple[int, int, int, int]] = []

    line_rect = _random_aligned_rect(
        h,
        w,
        line_h,
        line_w,
        free_row_parity,
        free_col_parity,
        blocked,
        margin=2,
    )
    if line_rect is None:
        return None
    blocked.append(line_rect)
    lr0, lc0, lr1, lc1 = line_rect
    grid[lr0 : lr1 + 1, lc0 : lc1 + 1] = BLACK

    isolated_rect = _random_aligned_rect(
        h,
        w,
        isolated_size,
        isolated_size,
        free_row_parity,
        free_col_parity,
        blocked,
        margin=2,
    )
    if isolated_rect is None:
        return None
    ir0, ic0, ir1, ic1 = isolated_rect
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = BLACK

    # Add a few extra irrelevant obstacles to vary the background without creating new squares.
    noise_trials = random.randint(0, 8)
    for _ in range(noise_trials):
        r = random.randrange(h)
        c = random.randrange(w)
        if any(rect[0] - 1 <= r <= rect[2] + 1 and rect[1] - 1 <= c <= rect[3] + 1 for rect in blocked):
            continue
        grid[r, c] = GREY

    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    info = _analyze(grid)
    selected = info["selected"]
    neighbors = info["neighbors"]
    degrees = info["degrees"]
    assert len(selected) >= 4
    assert any(deg == 0 for deg in degrees)
    assert any(deg == 1 for deg in degrees)
    assert any(deg >= 2 for deg in degrees)

    selected_keys = {_square_key(sq) for sq in selected}
    assert any(
        _square_key(cand) not in selected_keys and any(_rectangles_overlap(cand, sq) for sq in selected)
        for cand in info["candidates"]
    ), "need overlapping rejected candidates so greedy order matters"

    base = _base_after_blocks(grid, selected, degrees)
    isolated_mark_counts = []
    single_mark_counts = []
    for idx, sq in enumerate(selected):
        if degrees[idx] == 0:
            isolated_mark_counts.append(_mark_count(base, sq, [0, 1, 2, 3]))
        elif degrees[idx] == 1:
            neighbor_idx = next(iter(neighbors[idx]))
            chosen, dists = _closest_corner_indices(sq, selected[neighbor_idx])
            assert dists[1] < dists[2] - 1e-9, "degree-1 corner choice must be unique"
            single_mark_counts.append(_mark_count(base, sq, chosen))
    assert isolated_mark_counts and max(isolated_mark_counts) == 8
    assert single_mark_counts and max(single_mark_counts) >= 2

    out = main(grid)
    assert isinstance(out, np.ndarray)
    assert out.shape == grid.shape
    assert np.all(out[grid == GREY] == GREY)
    assert np.any(out == BLUE)
    assert np.any(out == RED)
    assert np.any(out == YELLOW)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            _validate_generated_grid(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid greedy-square adjacency puzzle input")
