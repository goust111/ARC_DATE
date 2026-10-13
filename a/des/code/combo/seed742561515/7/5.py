import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MARKER_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
SEGMENT_COUNT_RANGE = (5, 9)
SEGMENT_LEN_RANGE = (2, 5)

PATH_ATTEMPTS = 120
GEN_ATTEMPTS = 500

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    assert cells, "path must be non-empty"

    degree = {}
    for cell in cells:
        r, c = cell
        degree[cell] = sum((rr, cc) in cells for rr, cc in _neighbors4(r, c, h, w))

    endpoints = sorted(cell for cell, deg in degree.items() if deg == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg <= 2 for deg in degree.values()), "path must be a simple chain"

    start = endpoints[0]
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        r, c = cur
        for rr, cc in _neighbors4(r, c, h, w):
            if (rr, cc) not in cells or (rr, cc) == prev:
                continue
            nxts.append((rr, cc))
        if not nxts:
            break
        assert len(nxts) == 1, "chain traversal must be unique"
        prev, cur = cur, nxts[0]
        path.append(cur)

    assert len(path) == len(cells), "path must visit every cell exactly once"
    return path


def _infer_grey(grid: np.ndarray, bg: int) -> int:
    colors = [int(x) for x in np.unique(grid) if int(x) != bg]
    assert colors, "grid must contain a path"
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    grey = max(colors, key=lambda color: counts[color])
    return int(grey)


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(b[0] - a[0]), int(b[1] - a[1])


def _corner_indices(path: list[tuple[int, int]]) -> list[int]:
    corners = []
    for i in range(1, len(path) - 1):
        if _step_dir(path[i - 1], path[i]) != _step_dir(path[i], path[i + 1]):
            corners.append(i)
    return corners


def _analyze(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    grey = _infer_grey(g, bg)
    path = _trace_path(g != bg)

    marker_indices = [i for i, (r, c) in enumerate(path) if int(g[r, c]) != grey]
    assert marker_indices[0] == 0 and marker_indices[-1] == len(path) - 1
    assert len(marker_indices) >= 4 and len(marker_indices) % 2 == 0

    colored_vals = [int(x) for x in np.unique(g) if int(x) not in (bg, grey)]
    for color in colored_vals:
        assert int(np.count_nonzero(g == color)) == 2, "marker colors must appear in pairs"

    corner_set = set(_corner_indices(path))
    for idx in marker_indices[1:-1]:
        assert idx in corner_set, "interior markers must sit on corners"

    segments = []
    for left, right in zip(marker_indices, marker_indices[1:]):
        start = path[left]
        end = path[right]
        color = int(g[start])
        ext_dir = None
        if 0 < right < len(path) - 1:
            dir_out = _step_dir(path[right], path[right + 1])
            ext_dir = (-dir_out[0], -dir_out[1])
        segments.append(
            {
                "coords": path[left : right + 1],
                "start": start,
                "end": end,
                "color": color,
                "length": right - left + 1,
                "ext_dir": ext_dir,
            }
        )

    return bg, grey, path, marker_indices, segments


def _apply_rule(grid: np.ndarray, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    bg, _grey, _path, _markers, segments = _analyze(g)

    out = g.copy()
    for seg in segments:
        for r, c in seg["coords"]:
            out[r, c] = seg["color"]

    h, w = out.shape
    success = 0
    skipped = 0
    for seg in segments:
        if seg["ext_dir"] is None:
            continue
        er, ec = seg["end"]
        dr, dc = seg["ext_dir"]
        targets = []
        ok = True
        for step in range(1, seg["length"] + 1):
            rr = er + dr * step
            cc = ec + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            if int(out[rr, cc]) != bg:
                ok = False
                break
            targets.append((rr, cc))
        if not ok:
            skipped += 1
            continue
        for rr, cc in targets:
            out[rr, cc] = seg["color"]
        success += 1

    if want_stats:
        return out, {"success": success, "skip": skipped, "extendable": len(segments) - 1}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marker-delimited path segment by its start-marker color, then
    extend an equal-length line from each colored corner along the outward side
    of the post-turn direction. The path is traced from the top-left endpoint.
    """
    return _apply_rule(grid)


def _perpendicular_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _simulate_segment(
    tail: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    length: int,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    prev = tail
    cur_occ = set(occupied)
    cells = []
    for _ in range(length):
        nr = prev[0] + direction[0]
        nc = prev[1] + direction[1]
        cell = (nr, nc)
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if cell in cur_occ:
            return None
        for rr, cc in _neighbors4(nr, nc, h, w):
            if (rr, cc) in cur_occ and (rr, cc) != prev:
                return None
        cells.append(cell)
        cur_occ.add(cell)
        prev = cell
    return cells


def _has_future_segment(
    tail: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    good = []
    for nxt_dir in _perpendicular_dirs(direction):
        for length in range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1):
            if _simulate_segment(tail, nxt_dir, occupied, length, h, w) is not None:
                good.append(nxt_dir)
                break
    return good


def _build_path(h: int, w: int) -> list[tuple[int, int]] | None:
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    for _ in range(PATH_ATTEMPTS):
        start = (random.randrange(1, h - 1), random.randrange(1, w - 1))
        direction = random.choice(DIRS4)
        path = [start]
        occupied = {start}
        ok = True

        for seg_idx in range(segment_count):
            lengths = list(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
            random.shuffle(lengths)
            candidates = []
            for length in lengths:
                cells = _simulate_segment(path[-1], direction, occupied, length, h, w)
                if cells is None:
                    continue
                new_occupied = occupied | set(cells)
                tail = cells[-1]
                next_dirs = []
                if seg_idx < segment_count - 1:
                    next_dirs = _has_future_segment(tail, direction, new_occupied, h, w)
                    if not next_dirs:
                        continue
                candidates.append((cells, new_occupied, next_dirs))
            if not candidates:
                ok = False
                break

            cells, occupied, next_dirs = random.choice(candidates)
            path.extend(cells)
            if seg_idx < segment_count - 1:
                direction = random.choice(next_dirs)

        if not ok:
            continue
        if len(_corner_indices(path)) < 2:
            continue
        return path
    return None


def _marker_sequence(n_markers: int) -> list[int] | None:
    assert n_markers % 2 == 0
    pair_count = n_markers // 2
    if pair_count > len(MARKER_PALETTE):
        return None
    chosen = random.sample(MARKER_PALETTE, pair_count)
    bag = chosen + chosen
    for _ in range(100):
        random.shuffle(bag)
        if any(bag[i] == bag[i + 1] for i in range(len(bag) - 1)):
            continue
        return bag[:]
    return None


def _paint_input(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    marker_indices: list[int],
    marker_colors: list[int],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in path:
        grid[r, c] = GREY
    for idx, color in zip(marker_indices, marker_colors):
        r, c = path[idx]
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    path = _build_path(h, w)
    if path is None:
        return None
    if path[0] > path[-1]:
        path = list(reversed(path))

    corners = _corner_indices(path)
    valid_counts = [count for count in range(2, len(corners) + 1) if count % 2 == 0]
    if not valid_counts:
        return None

    # Prefer partial marking over coloring every corner.
    partial_counts = [count for count in valid_counts if count < len(corners)]
    if partial_counts and random.random() < 0.7:
        colored_corner_count = random.choice(partial_counts)
    else:
        colored_corner_count = random.choice(valid_counts)

    colored_corners = sorted(random.sample(corners, colored_corner_count))
    marker_indices = [0] + colored_corners + [len(path) - 1]
    marker_colors = _marker_sequence(len(marker_indices))
    if marker_colors is None:
        return None

    grid = _paint_input(h, w, path, marker_indices, marker_colors)
    grey_count = int(np.count_nonzero(grid == GREY))
    if grey_count <= 2:
        return None

    try:
        _out, stats = _apply_rule(grid, want_stats=True)
    except AssertionError:
        return None

    if stats["success"] == 0:
        return None
    if random.random() < 0.55 and stats["skip"] == 0:
        return None
    if int(np.count_nonzero(grid != BG)) > (h * w * 3) // 10:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
