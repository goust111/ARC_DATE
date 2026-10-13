from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_GREY = 5
_BG = 0
_STAMP_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_MAX_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


def _turn_left(d: int) -> int:
    return (d - 1) % 4


def _turn_right(d: int) -> int:
    return (d + 1) % 4


def _extract_quadrant_stamp(
    grid: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    bg: int,
) -> np.ndarray:
    region = grid[row_lo:row_hi, col_lo:col_hi]
    box = _bbox(region != bg)
    assert box is not None, "each legend quadrant must contain a stamp"
    r0, c0, r1, c1 = box
    patch = region[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, _ = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    c, color = candidates[0]
    assert 0 < c < grid.shape[1] - 1
    assert np.all(grid[:, c] == color)
    return int(c), int(color)


def _ordered_path(mask: np.ndarray, sep_c: int) -> list[tuple[int, int]]:
    points = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert points, "path must be non-empty"
    point_set = set(points)

    degree: dict[tuple[int, int], int] = {}
    for p in points:
        degree[p] = sum((q in point_set) for q in _neighbors4(p))
        assert degree[p] in (1, 2), "path must be a simple polyline"

    endpoints = [p for p in points if degree[p] == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    h, w = mask.shape
    edge_endpoints = [
        p
        for p in endpoints
        if p[0] in (0, h - 1) or p[1] == w - 1
    ]
    assert len(edge_endpoints) == 1, "path must have a unique outer-edge start"
    start = edge_endpoints[0]

    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [q for q in _neighbors4(cur) if q in point_set and q != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    assert len(ordered) == len(points), "path traversal must cover all path pixels"
    return ordered


def _path_segments(path: list[tuple[int, int]]) -> list[tuple[int, list[tuple[int, int]]]]:
    assert len(path) >= 2
    dirs: list[int] = []
    for a, b in zip(path, path[1:], strict=False):
        dr = int(b[0] - a[0])
        dc = int(b[1] - a[1])
        d = _DIRS.index((dr, dc))
        dirs.append(d)

    segments: list[tuple[int, list[tuple[int, int]]]] = []
    seg_start = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            segments.append((dirs[seg_start], path[seg_start : i + 1]))
            seg_start = i
    segments.append((dirs[seg_start], path[seg_start:]))
    return segments


def _stamp_on_output(
    out: np.ndarray,
    stamp: np.ndarray,
    center: tuple[int, int],
    bg: int,
    sep_c: int,
) -> None:
    k = stamp.shape[0]
    assert stamp.shape == (k, k)
    rad = k // 2
    cr, cc = center
    for sr in range(k):
        for sc in range(k):
            val = int(stamp[sr, sc])
            if val == bg:
                continue
            tr = cr + sr - rad
            tc = cc + sc - rad
            if 0 <= tr < out.shape[0] and sep_c < tc < out.shape[1]:
                out[tr, tc] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grid is split into a left stamp legend and a right grey polyline path.
    The four legend stamps are the same motif in 0/90/180/270 degree rotations,
    laid out as UL/UR/LR/LL. Traverse the right-hand path from its unique
    boundary endpoint, split it into straight segments, and stamp the segment
    midpoint with the rotation keyed by the segment direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, _ = g.shape
    bg = _most_common_color(g)
    sep_c, sep_color = _find_separator(g, bg)

    left_w = sep_c
    mid_r = h // 2
    mid_c = left_w // 2
    assert mid_r > 0 and mid_c > 0

    stamp_up = _extract_quadrant_stamp(g, 0, mid_r, 0, mid_c, bg)
    stamp_right = _extract_quadrant_stamp(g, 0, mid_r, mid_c, left_w, bg)
    stamp_left = _extract_quadrant_stamp(g, mid_r, h, 0, mid_c, bg)
    stamp_down = _extract_quadrant_stamp(g, mid_r, h, mid_c, left_w, bg)

    k = stamp_up.shape[0]
    assert k % 2 == 1, "stamp size must be odd"
    assert stamp_up.shape == stamp_right.shape == stamp_left.shape == stamp_down.shape
    assert stamp_up.shape[0] == stamp_up.shape[1]
    assert np.array_equal(np.rot90(stamp_up, 3), stamp_right)
    assert np.array_equal(np.rot90(stamp_up, 2), stamp_down)
    assert np.array_equal(np.rot90(stamp_up, 1), stamp_left)

    right_mask = g[:, sep_c + 1 :] == sep_color
    path = _ordered_path(right_mask, sep_c)
    path = [(r, c + sep_c + 1) for r, c in path]
    segments = _path_segments(path)

    dir_to_stamp = {
        0: stamp_up,
        1: stamp_right,
        2: stamp_down,
        3: stamp_left,
    }

    out = g.copy()
    for d, seg in segments:
        assert len(seg) >= k
        center = seg[len(seg) // 2]
        _stamp_on_output(out, dir_to_stamp[d], center, bg, sep_c)
    return out


def _connect_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    if _RNG.integers(0, 2) == 0:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _grow_stamp_mask(k: int) -> np.ndarray | None:
    target = int(_RNG.integers(max(k + 2, (k * k) // 3), (2 * k * k) // 3 + 1))
    center = (k // 2, k // 2)
    edge_targets = [
        (0, int(_RNG.integers(0, k))),
        (k - 1, int(_RNG.integers(0, k))),
        (int(_RNG.integers(0, k)), 0),
        (int(_RNG.integers(0, k)), k - 1),
    ]
    mask = np.zeros((k, k), dtype=bool)
    mask[center] = True
    for end in edge_targets:
        _connect_path(mask, center, end)
    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in _DIRS:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < k and 0 <= nc < k and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = frontier[int(_RNG.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_stamp(k: int, bg: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _grow_stamp_mask(k)
        if mask is None:
            continue
        box = _bbox(mask)
        if box != (0, 0, k - 1, k - 1):
            continue

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        n_colors = int(_RNG.integers(2, min(5, len(coords)) + 1))
        palette = _RNG.choice(_STAMP_COLORS, size=n_colors, replace=False)
        cell_colors = _RNG.choice(palette, size=len(coords), replace=True)
        cell_colors[:n_colors] = palette
        cell_colors = cell_colors[_RNG.permutation(len(coords))]

        stamp = np.full((k, k), bg, dtype=int)
        for (r, c), color in zip(coords, cell_colors, strict=False):
            stamp[r, c] = int(color)

        rots = [np.rot90(stamp, i) for i in range(4)]
        distinct = {
            (rot.shape, rot.tobytes())
            for rot in rots
        }
        if len(distinct) != 4:
            continue
        return stamp
    return None


def _new_cells_valid(
    new_cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    origin_prev: tuple[int, int] | None,
) -> bool:
    seen = set(new_cells)
    for i, cell in enumerate(new_cells):
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return False
        if c == 0:
            return False
        if r in (0, h - 1) or c == w - 1:
            return False
        if cell in occupied:
            return False
        prev = new_cells[i - 1] if i > 0 else origin_prev
        nxt = new_cells[i + 1] if i + 1 < len(new_cells) else None
        for nbr in _neighbors4(cell):
            if nbr in occupied and nbr != prev:
                return False
            if nbr in seen and nbr != prev and nbr != nxt:
                return False
    return True


def _can_extend_min(
    start: tuple[int, int],
    d: int,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    min_pix: int,
) -> bool:
    dr, dc = _DIRS[d]
    r, c = start
    new_cells: list[tuple[int, int]] = []
    for _ in range(min_pix - 1):
        r += dr
        c += dc
        new_cells.append((r, c))
    return _new_cells_valid(new_cells, occupied, h, w, start)


def _candidate_lengths(
    start: tuple[int, int],
    d: int,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    min_pix: int,
    is_last: bool,
) -> list[int]:
    dr, dc = _DIRS[d]
    r, c = start
    prefix: list[tuple[int, int]] = []
    candidates: list[int] = []
    max_steps = max(h, w) + 2
    for step in range(1, max_steps):
        r += dr
        c += dc
        prefix.append((r, c))
        if not _new_cells_valid(prefix, occupied, h, w, start):
            break
        pix_len = step + 1
        if pix_len < min_pix or pix_len % 2 == 0:
            continue
        if is_last:
            candidates.append(pix_len)
            continue
        occ2 = occupied | set(prefix)
        turn_dirs = [_turn_left(d), _turn_right(d)]
        feasible = any(_can_extend_min(prefix[-1], nd, occ2, h, w, min_pix) for nd in turn_dirs)
        if feasible:
            candidates.append(pix_len)
    return candidates


def _build_path(
    h: int,
    w: int,
    k: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    min_pix = 2 * k + 1
    seg_choices = [2, 3, 4] if k == 5 else [2, 3, 4, 5]
    n_segments = int(seg_choices[int(_RNG.integers(0, len(seg_choices)))])

    side = ["top", "bottom", "right"][int(_RNG.integers(0, 3))]
    if side == "top":
        start = (0, int(_RNG.integers(1, w - 1)))
        first_dir = 2
    elif side == "bottom":
        start = (h - 1, int(_RNG.integers(1, w - 1)))
        first_dir = 0
    else:
        start = (int(_RNG.integers(1, h - 1)), w - 1)
        first_dir = 3

    def rec(
        seg_idx: int,
        cur: tuple[int, int],
        d: int,
        occupied: set[tuple[int, int]],
        points: list[tuple[int, int]],
        dirs: list[int],
        lens: list[int],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
        is_last = seg_idx == n_segments - 1
        lengths = _candidate_lengths(cur, d, occupied, h, w, min_pix, is_last)
        if not lengths:
            return None
        order = _RNG.permutation(len(lengths))
        for idx in order:
            pix_len = int(lengths[int(idx)])
            dr, dc = _DIRS[d]
            r, c = cur
            new_cells: list[tuple[int, int]] = []
            for _ in range(pix_len - 1):
                r += dr
                c += dc
                new_cells.append((r, c))
            occ2 = occupied | set(new_cells)
            points2 = points + new_cells
            dirs2 = dirs + [d]
            lens2 = lens + [pix_len]
            if is_last:
                end = new_cells[-1]
                if end[0] in (0, h - 1) or end[1] in (0, w - 1):
                    continue
                return points2, dirs2, lens2

            turn_dirs = [_turn_left(d), _turn_right(d)]
            turn_dirs = [turn_dirs[i] for i in _RNG.permutation(2)]
            for nd in turn_dirs:
                if not _can_extend_min(new_cells[-1], nd, occ2, h, w, min_pix):
                    continue
                got = rec(seg_idx + 1, new_cells[-1], nd, occ2, points2, dirs2, lens2)
                if got is not None:
                    return got
        return None

    return rec(0, start, first_dir, {start}, [start], [], [])


def _segments_from_metadata(
    points: list[tuple[int, int]],
    dirs: list[int],
    lens: list[int],
) -> list[tuple[int, list[tuple[int, int]]]]:
    segments: list[tuple[int, list[tuple[int, int]]]] = []
    idx = 0
    for d, pix_len in zip(dirs, lens, strict=False):
        seg = points[idx : idx + pix_len]
        segments.append((d, seg))
        idx += pix_len - 1
    assert idx == len(points) - 1
    return segments


def _stamp_expected(
    inp: np.ndarray,
    base_stamp: np.ndarray,
    sep_c: int,
    segments: list[tuple[int, list[tuple[int, int]]]],
) -> np.ndarray:
    out = inp.copy()
    bg = _most_common_color(inp)
    stamps = {
        0: base_stamp,
        1: np.rot90(base_stamp, 3),
        2: np.rot90(base_stamp, 2),
        3: np.rot90(base_stamp, 1),
    }
    for d, seg in segments:
        center = seg[len(seg) // 2]
        _stamp_on_output(out, stamps[d], center, bg, sep_c)
    return out


def _place_stamp(out: np.ndarray, stamp: np.ndarray, r0: int, c0: int) -> None:
    h, w = stamp.shape
    out[r0 : r0 + h, c0 : c0 + w] = stamp


def _gen_attempt() -> np.ndarray | None:
    k = 3 if _RNG.integers(0, 2) == 0 else 5
    stamp = _make_stamp(k, _BG)
    if stamp is None:
        return None

    height = int(_RNG.integers(max(2 * k + 5, 17), max(2 * k + 14, 29)))
    left_w = int(_RNG.integers(2 * k + 5, 2 * k + 12))
    right_w = int(_RNG.integers(max(2 * k + 5, 13), max(2 * k + 12, 24)))

    mid_r = height // 2
    mid_c = left_w // 2
    if mid_r <= k + 1 or mid_c <= k + 1 or height - mid_r <= k + 1 or left_w - mid_c <= k + 1:
        return None

    path_data = _build_path(height, right_w, k)
    if path_data is None:
        return None
    local_points, seg_dirs, seg_lens = path_data
    segments_local = _segments_from_metadata(local_points, seg_dirs, seg_lens)

    width = left_w + 1 + right_w
    sep_c = left_w
    inp = np.full((height, width), _BG, dtype=int)
    inp[:, sep_c] = _GREY

    ul = stamp
    ur = np.rot90(stamp, 3)
    ll = np.rot90(stamp, 1)
    lr = np.rot90(stamp, 2)

    top_r = int(_RNG.integers(1, mid_r - k))
    bottom_r = int(_RNG.integers(mid_r + 1, height - k))
    left_c = int(_RNG.integers(1, mid_c - k))
    right_c = int(_RNG.integers(mid_c + 1, left_w - k))
    _place_stamp(inp, ul, top_r, left_c)
    _place_stamp(inp, ur, top_r, right_c)
    _place_stamp(inp, ll, bottom_r, left_c)
    _place_stamp(inp, lr, bottom_r, right_c)

    for r, c in local_points:
        inp[r, sep_c + 1 + c] = _GREY

    segments_global = [
        (d, [(r, sep_c + 1 + c) for r, c in seg])
        for d, seg in segments_local
    ]
    expected = _stamp_expected(inp, stamp, sep_c, segments_global)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
