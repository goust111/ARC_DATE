from __future__ import annotations

from collections import Counter

import numpy as np


ENDPOINT_COLOR = 6
COLOR_POOL = [1, 2, 3, 4, 5, 7, 8, 9]
RING_SHAPES = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2), (3, 3)],
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
MIN_SEG_LEN = 2
MAX_SEG_LEN = 6

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _clockwise_perimeter(r0: int, c0: int, hh: int, ww: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c0 + ww)]
    coords += [(r, c0 + ww - 1) for r in range(r0 + 1, r0 + hh - 1)]
    if hh > 1:
        coords += [(r0 + hh - 1, c) for c in range(c0 + ww - 1, c0 - 1, -1)]
    if ww > 1:
        coords += [(r, c0) for r in range(r0 + hh - 2, r0, -1)]
    return coords


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _in_corner_window(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> bool:
    return (
        (r1 < 4 and c1 < 4)
        or (r1 < 4 and c0 >= w - 4)
        or (r0 >= h - 4 and c1 < 4)
        or (r0 >= h - 4 and c0 >= w - 4)
    )


def _find_ring(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, list[int]]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if not (2 <= hh <= 4 and 2 <= ww <= 4):
            continue
        if not _in_corner_window(r0, c0, r1, c1, h, w):
            continue
        seq = _clockwise_perimeter(r0, c0, hh, ww)
        if set(seq) != set(comp):
            continue
        colors = [int(grid[r, c]) for r, c in seq]
        if len(set(colors)) != len(colors):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        candidates.append((mask, colors))
    assert len(candidates) == 1, "expected exactly one corner color ring"
    return candidates[0]


def _segment_info(grid: np.ndarray, ring_mask: np.ndarray, color: int) -> dict[str, object]:
    pts = [(int(r), int(c)) for r, c in np.argwhere((grid == color) & (~ring_mask))]
    assert len(pts) >= 2, f"missing segment for color {color}"
    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(rows) == 1:
        row = next(iter(rows))
        cols_sorted = sorted(c for _, c in pts)
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
        cells = [(row, c) for c in cols_sorted]
    else:
        assert len(cols) == 1, f"color {color} is not a straight line"
        col = next(iter(cols))
        rows_sorted = sorted(r for r, _ in pts)
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
        cells = [(r, col) for r in rows_sorted]
    return {"cells": cells, "endpoints": (cells[0], cells[-1])}


def _gap_cell(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    if a[0] == b[0] and abs(a[1] - b[1]) == 2:
        return (a[0], (a[1] + b[1]) // 2)
    if a[1] == b[1] and abs(a[0] - b[0]) == 2:
        return ((a[0] + b[0]) // 2, a[1])
    return None


def _choose_connections(
    all_candidates: list[list[tuple[int, int, tuple[int, int]]]]
) -> list[tuple[int, int, tuple[int, int]]]:
    def dfs(edge_idx: int, used_on_left: int | None) -> list[tuple[int, int, tuple[int, int]]] | None:
        if edge_idx == len(all_candidates):
            return []
        for cand in all_candidates[edge_idx]:
            left_idx, right_idx, mid = cand
            if used_on_left is not None and left_idx == used_on_left:
                continue
            rest = dfs(edge_idx + 1, right_idx)
            if rest is not None:
                return [cand] + rest
        return None

    chosen = dfs(0, None)
    assert chosen is not None, "could not assign endpoints consistently"
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect colored straight segments in the order given by a corner color ring,
    repaint the whole path with the ring's first color, and mark the two exposed
    endpoints with a reserved special color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert ENDPOINT_COLOR not in set(map(int, np.unique(g))), "endpoint color must be absent from input"

    ring_mask, order = _find_ring(g, bg)
    assert len(order) >= 2

    segments = [_segment_info(g, ring_mask, color) for color in order]

    all_candidates: list[list[tuple[int, int, tuple[int, int]]]] = []
    for i in range(len(segments) - 1):
        cur_eps = segments[i]["endpoints"]
        nxt_eps = segments[i + 1]["endpoints"]
        edge_candidates: list[tuple[int, int, tuple[int, int]]] = []
        for a_idx, a in enumerate(cur_eps):
            for b_idx, b in enumerate(nxt_eps):
                mid = _gap_cell(a, b)
                if mid is None:
                    continue
                if int(g[mid]) != bg:
                    continue
                edge_candidates.append((a_idx, b_idx, mid))
        assert edge_candidates, f"no valid connection between colors {order[i]} and {order[i + 1]}"
        all_candidates.append(edge_candidates)

    chosen = _choose_connections(all_candidates)

    out = np.full_like(g, bg)
    path_color = order[0]
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = path_color
    for _, _, mid in chosen:
        out[mid] = path_color

    start_used = chosen[0][0]
    end_used = chosen[-1][1]
    start_ep = segments[0]["endpoints"][1 - start_used]
    end_ep = segments[-1]["endpoints"][1 - end_used]
    out[start_ep] = ENDPOINT_COLOR
    out[end_ep] = ENDPOINT_COLOR
    return out


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _shuffle(items: list, rng: np.random.Generator) -> list:
    if len(items) < 2:
        return items
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def _chebyshev_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in cell_set:
                    continue
                if (nr, nc) in occupied:
                    return False
    return True


def _is_gap_relation(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return _gap_cell(a, b) is not None


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(length)]


def _build_chain(
    h: int,
    w: int,
    blocked: np.ndarray,
    segment_count: int,
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]] | None:
    def valid_cells(
        cells: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        gap_cells: set[tuple[int, int]],
    ) -> bool:
        for r, c in cells:
            if not _in_bounds(h, w, r, c):
                return False
            if blocked[r, c] or (r, c) in occupied or (r, c) in gap_cells:
                return False
        return _chebyshev_clear(cells, occupied)

    def endpoint_clash(
        endpoint: tuple[int, int],
        endpoints: list[tuple[int, int]],
        allowed_partner: tuple[int, int] | None = None,
    ) -> bool:
        for other in endpoints:
            if allowed_partner is not None and other == allowed_partner:
                continue
            if _is_gap_relation(endpoint, other):
                return True
        return False

    def start_candidates() -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
        cands: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if blocked[r, c]:
                    continue
                for direction in DIRS4:
                    for length in range(MIN_SEG_LEN, MAX_SEG_LEN + 1):
                        cells = _segment_cells((r, c), direction, length)
                        if not valid_cells(cells, set(), set()):
                            continue
                        cands.append((cells, cells[-1]))
        return _shuffle(cands, rng)

    def next_candidates(
        current_conn: tuple[int, int],
        start_endpoint: tuple[int, int],
        occupied: set[tuple[int, int]],
        gap_cells: set[tuple[int, int]],
        endpoints: list[tuple[int, int]],
    ) -> list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]]:
        cands: list[tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]] = []
        for gap_dir in _shuffle(DIRS4.copy(), rng):
            mid = (current_conn[0] + gap_dir[0], current_conn[1] + gap_dir[1])
            near = (current_conn[0] + 2 * gap_dir[0], current_conn[1] + 2 * gap_dir[1])
            if not _in_bounds(h, w, *mid) or not _in_bounds(h, w, *near):
                continue
            if blocked[mid] or blocked[near] or mid in occupied or near in occupied or mid in gap_cells:
                continue
            if endpoint_clash(near, endpoints, allowed_partner=current_conn):
                continue
            dirs = [d for d in DIRS4 if d != (-gap_dir[0], -gap_dir[1])]
            for direction in _shuffle(dirs, rng):
                for length in _shuffle(list(range(MIN_SEG_LEN, MAX_SEG_LEN + 1)), rng):
                    cells = _segment_cells(near, direction, int(length))
                    far = cells[-1]
                    if not valid_cells(cells, occupied, gap_cells):
                        continue
                    if endpoint_clash(far, endpoints):
                        continue
                    if _is_gap_relation(far, start_endpoint):
                        continue
                    cands.append((cells, far, mid))
        return cands

    def dfs(
        idx: int,
        occupied: set[tuple[int, int]],
        gap_cells: set[tuple[int, int]],
        endpoints: list[tuple[int, int]],
        start_endpoint: tuple[int, int],
        current_conn: tuple[int, int],
        segments: list[list[tuple[int, int]]],
    ) -> list[list[tuple[int, int]]] | None:
        if idx == segment_count:
            return segments
        for cells, far, mid in next_candidates(current_conn, start_endpoint, occupied, gap_cells, endpoints):
            new_occupied = occupied | set(cells)
            new_gap_cells = set(gap_cells)
            new_gap_cells.add(mid)
            new_endpoints = endpoints + [cells[0], far]
            result = dfs(
                idx + 1,
                new_occupied,
                new_gap_cells,
                new_endpoints,
                start_endpoint,
                far,
                segments + [cells],
            )
            if result is not None:
                return result
        return None

    for first_cells, first_far in start_candidates():
        start_endpoint = first_cells[0]
        result = dfs(
            1,
            set(first_cells),
            set(),
            [start_endpoint, first_far],
            start_endpoint,
            first_far,
            [first_cells],
        )
        if result is not None:
            return result
    return None


def _ring_bbox(
    corner: str,
    ring_h: int,
    ring_w: int,
    h: int,
    w: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if corner in ("tl", "tr"):
        base_r = 0
    else:
        base_r = h - 4
    if corner in ("tl", "bl"):
        base_c = 0
    else:
        base_c = w - 4
    off_r = int(rng.integers(0, 4 - ring_h + 1))
    off_c = int(rng.integers(0, 4 - ring_w + 1))
    return base_r + off_r, base_c + off_c


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    segment_count = int(rng.choice([4, 6, 8]))
    ring_h, ring_w = _shuffle(RING_SHAPES[segment_count].copy(), rng)[0]

    h = int(rng.integers(16, 25))
    w = int(rng.integers(16, 25))
    grid = np.zeros((h, w), dtype=int)

    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    r0, c0 = _ring_bbox(corner, ring_h, ring_w, h, w, rng)
    ring_coords = _clockwise_perimeter(r0, c0, ring_h, ring_w)

    blocked = np.zeros((h, w), dtype=bool)
    win_r0 = 0 if corner in ("tl", "tr") else h - 4
    win_c0 = 0 if corner in ("tl", "bl") else w - 4
    _mark_rect(blocked, win_r0, win_c0, win_r0 + 3, win_c0 + 3, pad=1)

    segments = _build_chain(h, w, blocked, segment_count, rng)
    if segments is None:
        return None

    colors = list(rng.choice(COLOR_POOL, size=segment_count, replace=False))
    for color, cells in zip(colors, segments):
        for r, c in cells:
            grid[r, c] = int(color)
    for color, (r, c) in zip(colors, ring_coords):
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if int(np.sum(grid == ENDPOINT_COLOR)) != 0:
        return None
    if int(np.sum(out == ENDPOINT_COLOR)) != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-path input")
