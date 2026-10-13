from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
ORANGE = 7
MARKER_POOL = [1, 2, 3, 4, 6, 8]

MIN_SIZE = 12
MAX_SIZE = 18
MAX_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _append_segment(path: list[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and cells[0] == path[-1]:
        cells = cells[1:]
    path.extend(cells)


def _trace_pipe(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    assert cells, "pipe must be non-empty"

    deg: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, h, w))
    assert all(v in (1, 2) for v in deg.values()), "pipe must have no branches"

    endpoints = sorted(cell for cell, v in deg.items() if v == 1)
    assert len(endpoints) == 2, "pipe must have exactly two endpoints"
    start = endpoints[0]

    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nb for nb in _neighbors4(cur[0], cur[1], h, w) if nb in cells and nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "pipe traversal must be unambiguous"
        prev, cur = cur, nxts[0]
        order.append(cur)

    assert len(order) == len(cells), "pipe must be a single simple path"
    return order


def _ordered_markers(grid: np.ndarray, path: list[tuple[int, int]]) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = grid.shape
    bg = _bg_color(grid)
    path_index = {cell: idx for idx, cell in enumerate(path)}
    markers: list[tuple[int, int, tuple[int, int]]] = []

    for r, c in np.argwhere((grid != bg) & (grid != GREY) & (grid != MAROON)):
        rr = int(r)
        cc = int(c)
        hits = [path_index[(nr, nc)] for nr, nc in _neighbors4(rr, cc, h, w) if (nr, nc) in path_index]
        assert len(hits) == 1, "each marker must touch exactly one pipe cell"
        markers.append((int(hits[0]), int(grid[rr, cc]), (rr, cc)))

    markers.sort(key=lambda x: x[0])
    assert len(markers) >= 2, "need at least two markers"

    idxs = [idx for idx, _color, _cell in markers]
    cols = [color for _idx, color, _cell in markers]
    assert len(set(idxs)) == len(idxs), "marker anchor cells must be unique"
    assert len(set(cols)) == len(cols), "marker colors must be distinct"
    return markers


def _visible_colors(path_len: int, markers: list[tuple[int, int, tuple[int, int]]]) -> list[int]:
    marker_by_idx = {idx: color for idx, color, _cell in markers}
    current = int(markers[0][1])
    colors: list[int] = []
    for idx in range(path_len):
        colors.append(current)
        if idx in marker_by_idx:
            current = int(marker_by_idx[idx])
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a single grey pipe by the traversal order of off-pipe color markers.

    Rule:
    - Traverse the grey/maroon pipe from the lexicographically smaller endpoint.
    - When walking, the current marker color is applied to visible grey pipe cells.
      Passing a marker updates the color used for subsequent cells.
    - Each contiguous maroon run is then inferred from its neighboring visible
      colored segments: same color -> that color, different colors -> orange,
      endpoint run -> the only visible side's color.
    - Marker pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    mask = (g == GREY) | (g == MAROON)
    path = _trace_pipe(mask)
    markers = _ordered_markers(g, path)

    out = g.copy()
    marker_by_idx = {idx: color for idx, color, _cell in markers}
    current = int(markers[0][1])

    for idx, (r, c) in enumerate(path):
        if int(g[r, c]) == GREY:
            out[r, c] = current
        if idx in marker_by_idx:
            current = int(marker_by_idx[idx])

    idx = 0
    while idx < len(path):
        r, c = path[idx]
        if int(g[r, c]) != MAROON:
            idx += 1
            continue

        j = idx
        while j + 1 < len(path):
            rr, cc = path[j + 1]
            if int(g[rr, cc]) != MAROON:
                break
            j += 1

        left_color = None if idx == 0 else int(out[path[idx - 1]])
        right_color = None if j + 1 == len(path) else int(out[path[j + 1]])

        if left_color is None:
            fill = int(right_color)
        elif right_color is None:
            fill = int(left_color)
        elif left_color == right_color:
            fill = int(left_color)
        else:
            fill = ORANGE

        for k in range(idx, j + 1):
            rr, cc = path[k]
            out[rr, cc] = fill
        idx = j + 1

    return out


def _random_spaced_values(low: int, high: int, count: int, min_gap: int) -> list[int]:
    vals = list(range(low, high + 1))
    for _ in range(200):
        chosen = sorted(random.sample(vals, count))
        if all(chosen[i + 1] - chosen[i] >= min_gap for i in range(len(chosen) - 1)):
            return chosen
    raise RuntimeError("failed to sample spaced values")


def _build_path() -> tuple[tuple[int, int], list[tuple[int, int]]]:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        max_rows = min(5, ((h - 3) // 2) + 1, ((w - 5) // 2) + 2)
        band_count = random.randint(3, max_rows)
        rows = _random_spaced_values(1, h - 2, band_count, 2)
        random.shuffle(rows)
        bends_x = _random_spaced_values(2, w - 3, band_count - 1, 2)

        path: list[tuple[int, int]] = [(rows[0], 0)]
        cur = path[0]
        for i in range(band_count - 1):
            nxt = (rows[i], bends_x[i])
            _append_segment(path, cur, nxt)
            cur = nxt
            nxt = (rows[i + 1], bends_x[i])
            _append_segment(path, cur, nxt)
            cur = nxt
        _append_segment(path, cur, (rows[-1], w - 1))
    else:
        max_cols = min(5, ((w - 3) // 2) + 1, ((h - 5) // 2) + 2)
        band_count = random.randint(3, max_cols)
        cols = _random_spaced_values(1, w - 2, band_count, 2)
        random.shuffle(cols)
        bends_y = _random_spaced_values(2, h - 3, band_count - 1, 2)

        path = [(0, cols[0])]
        cur = path[0]
        for i in range(band_count - 1):
            nxt = (bends_y[i], cols[i])
            _append_segment(path, cur, nxt)
            cur = nxt
            nxt = (bends_y[i], cols[i + 1])
            _append_segment(path, cur, nxt)
            cur = nxt
        _append_segment(path, cur, (h - 1, cols[-1]))

    assert len(path) == len(set(path)), "generated path must not self-intersect"
    return (h, w), path


def _bend_infos(path: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, list[tuple[int, int]]]]:
    path_set = set(path)
    infos: list[tuple[int, list[tuple[int, int]]]] = []
    for idx in range(1, len(path) - 1):
        pr, pc = path[idx - 1]
        r, c = path[idx]
        nr, nc = path[idx + 1]
        d1 = (r - pr, c - pc)
        d2 = (nr - r, nc - c)
        if d1 == d2:
            continue

        candidates: list[tuple[int, int]] = []
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) in path_set:
                continue
            hits = [nb for nb in _neighbors4(rr, cc, h, w) if nb in path_set]
            if hits == [(r, c)] or (len(hits) == 1 and hits[0] == (r, c)):
                candidates.append((rr, cc))
        if candidates:
            infos.append((idx, candidates))
    return infos


def _interval_conflicts(interval: tuple[int, int], used: list[tuple[int, int, str]]) -> bool:
    a, b = interval
    for ua, ub, _kind in used:
        if not (b < ua - 1 or a > ub + 1):
            return True
    return False


def _same_candidates(colors: list[int], used: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    n = len(colors)
    out: list[tuple[int, int]] = []
    for length in (1, 2, 3):
        for a in range(1, n - length):
            b = a + length - 1
            if b >= n - 1 or _interval_conflicts((a, b), used):
                continue
            window = colors[a - 1 : b + 2]
            if len(set(window)) == 1:
                out.append((a, b))
    return out


def _mixed_candidates(colors: list[int], used: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    n = len(colors)
    out: list[tuple[int, int]] = []
    for length in (1, 2, 3):
        for a in range(1, n - length):
            b = a + length - 1
            if b >= n - 1 or _interval_conflicts((a, b), used):
                continue
            if colors[a - 1] != colors[b + 1]:
                out.append((a, b))
    return out


def _endpoint_candidates(colors: list[int], used: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    n = len(colors)
    out: list[tuple[int, int]] = []
    for length in (1, 2, 3):
        if length >= n:
            continue
        left = (0, length - 1)
        right = (n - length, n - 1)
        if not _interval_conflicts(left, used):
            out.append(left)
        if not _interval_conflicts(right, used):
            out.append(right)
    return out


def _marker_layout(
    path: list[tuple[int, int]], h: int, w: int, profile: list[str]
) -> tuple[list[tuple[int, int, tuple[int, int]]], list[tuple[int, int, str]]] | None:
    bends = _bend_infos(path, h, w)
    if len(bends) < 2:
        return None

    path_len = len(path)
    max_markers = min(5, len(bends), len(MARKER_POOL))

    for _ in range(250):
        marker_count = min(max_markers, random.choice([2, 3, 3, 4]))
        marker_count = max(2, marker_count)
        chosen = sorted(random.sample(bends, marker_count), key=lambda x: x[0])
        marker_idxs = [idx for idx, _cands in chosen]
        if any(marker_idxs[i + 1] - marker_idxs[i] < 3 for i in range(len(marker_idxs) - 1)):
            continue

        marker_cells: list[tuple[int, int]] = []
        used_cells: set[tuple[int, int]] = set()
        ok = True
        for _idx, candidates in chosen:
            options = [cell for cell in candidates if cell not in used_cells]
            if not options:
                ok = False
                break
            cell = random.choice(options)
            used_cells.add(cell)
            marker_cells.append(cell)
        if not ok:
            continue

        colors = random.sample(MARKER_POOL, marker_count)
        markers = [(marker_idxs[i], colors[i], marker_cells[i]) for i in range(marker_count)]
        visible = _visible_colors(path_len, markers)

        blocks: list[tuple[int, int, str]] = []
        branches = profile[:]
        random.shuffle(branches)
        for branch in branches:
            if branch == "same":
                candidates = _same_candidates(visible, blocks)
            elif branch == "mixed":
                candidates = _mixed_candidates(visible, blocks)
            else:
                candidates = _endpoint_candidates(visible, blocks)
            if not candidates:
                ok = False
                break
            a, b = random.choice(candidates)
            blocks.append((a, b, branch))

        if not ok:
            continue

        blocks.sort(key=lambda x: x[0])
        if any(b - a + 1 > 3 for a, b, _kind in blocks):
            continue
        if any(a2 <= b1 + 1 for (_a1, b1, _), (a2, _b2, _) in zip(blocks, blocks[1:])):
            continue
        if not any(kind == "endpoint" for _a, _b, kind in blocks) and random.random() < 0.35:
            extra = _endpoint_candidates(visible, blocks)
            if extra:
                a, b = random.choice(extra)
                blocks.append((a, b, "endpoint"))
                blocks.sort(key=lambda x: x[0])

        # Keep the path readable by leaving some visible grey on both sides of
        # non-endpoint blocks.
        readable = True
        for a, b, kind in blocks:
            if kind != "endpoint" and (a == 0 or b == path_len - 1):
                readable = False
                break
        if not readable:
            continue

        return markers, blocks

    return None


def _validate_input(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        path = _trace_pipe((g == GREY) | (g == MAROON))
        markers = _ordered_markers(g, path)
        h, w = g.shape
        bends = {idx for idx, _cands in _bend_infos(path, h, w)}
        marker_idxs = {idx for idx, _color, _cell in markers}
        if not marker_idxs <= bends:
            return False

        maroon_runs = 0
        idx = 0
        while idx < len(path):
            r, c = path[idx]
            if int(g[r, c]) != MAROON:
                idx += 1
                continue
            j = idx
            while j + 1 < len(path):
                rr, cc = path[j + 1]
                if int(g[rr, cc]) != MAROON:
                    break
                j += 1
            if j - idx + 1 > 3:
                return False
            maroon_runs += 1
            idx = j + 1
        if maroon_runs < 2:
            return False

        out = main(g)
        if np.any(out == GREY) or np.any(out == MAROON):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    profiles = [
        ["same", "mixed"],
        ["same", "endpoint"],
        ["mixed", "endpoint"],
        ["same", "mixed", "endpoint"],
    ]
    profile = random.choice(profiles)

    try:
        (h, w), path = _build_path()
    except RuntimeError:
        return None

    layout = _marker_layout(path, h, w, profile)
    if layout is None:
        return None
    markers, blocks = layout

    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in path:
        grid[r, c] = GREY
    for _idx, color, (r, c) in markers:
        grid[r, c] = color
    for a, b, _kind in blocks:
        for idx in range(a, b + 1):
            r, c = path[idx]
            grid[r, c] = MAROON

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
