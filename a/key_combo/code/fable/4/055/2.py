from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import binary_fill_holes, label


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    lab, n = label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _turn_counts(path: list[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross != 0:
            counts[1 if cross > 0 else -1] += 1
    return counts


def _concave_count(coords: np.ndarray) -> int:
    counts = _turn_counts(_trace_cycle(coords))
    assert counts
    assert abs(counts[1] - counts[-1]) == 4, counts
    return int(min(counts[1], counts[-1]))


def _component_interior(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    interior = binary_fill_holes(local) & ~local
    return (r0, c0, r1, c1), interior


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey loop interior with the outside color whose scattered pixel count matches the loop's concave-corner count.

    Input structure:
    - The background is black.
    - Grey foreground pixels form one or more one-cell-thick 4-connected closed orthogonal loops, and every loop pixel has degree 2.
    - All other foreground pixels are isolated scattered markers grouped by color outside the loops.

    Rule:
    - Trace every grey loop in order and count its bend directions; the minority turn direction gives that loop's concave-corner count.
    - Match each loop to the unique outside marker color whose total pixel count equals that concave count.
    - Erase all scattered markers, keep the grey loops, and fill every enclosed background cell of each loop with its matched color.

    Key insight:
    - The pairing key is not a loop's size or position but its turning structure: a simple orthogonal cycle always has four more convex corners than concave corners, so the minority turn count identifies the right color group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    loop_mask = g == GREY
    loop_components = _components_mask(loop_mask)
    assert loop_components

    source_counts: dict[int, int] = {}
    for color in sorted(set(map(int, g.ravel())) - {bg, GREY}):
        count = int(np.sum(g == color))
        if count:
            assert count not in source_counts
            source_counts[count] = color

    concave_counts = [_concave_count(coords) for coords in loop_components]
    assert set(concave_counts) == set(source_counts), (concave_counts, source_counts)

    out = g.copy()
    out[(g != bg) & (g != GREY)] = bg
    for coords in loop_components:
        count = _concave_count(coords)
        color = source_counts[count]
        (r0, c0, _r1, _c1), interior = _component_interior(coords)
        view = out[r0 : r0 + interior.shape[0], c0 : c0 + interior.shape[1]]
        view[interior] = color
    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _sample_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    candidates = []
    for start in range(2, width - 3):
        for length in (1, 2, 3):
            end = start + length - 1
            if end <= width - 3:
                candidates.append((start, end))
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for start, end in candidates:
        if all(end + 2 < prev_start or prev_end + 2 < start for prev_start, prev_end in chosen):
            chosen.append((start, end))
            if len(chosen) == count:
                chosen.sort()
                return chosen
    return None


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _make_loop_patch(concave_count: int) -> tuple[np.ndarray, np.ndarray]:
    assert concave_count > 0 and concave_count % 2 == 0
    notches = concave_count // 2
    for _ in range(300):
        width_min = max(8, 3 * notches + 4)
        width = random.randint(width_min, min(19, width_min + 4))
        height = random.randint(6, 13)
        top_count = random.randint(0, notches)
        bottom_count = notches - top_count
        top_intervals = _sample_intervals(width, top_count)
        bottom_intervals = _sample_intervals(width, bottom_count)
        if top_intervals is None or bottom_intervals is None:
            continue

        top = [0] * width
        bottom = [height - 1] * width
        for start, end in top_intervals:
            top_depth = random.randint(1, max(1, height - 4))
            for c in range(start, end + 1):
                top[c] = top_depth
        for start, end in bottom_intervals:
            bottom_depth = random.randint(1, max(1, height - 4))
            for c in range(start, end + 1):
                bottom[c] = height - 1 - bottom_depth
        if min(bottom[c] - top[c] for c in range(width)) < 2:
            continue

        path = _path_from_profiles(top, bottom)
        rows = [r for r, _c in path]
        cols = [c for _r, c in path]
        mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
        for r, c in path:
            mask[r, c] = True

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randint(0, 3))
        mask = _normalize_mask(mask)

        coords = np.argwhere(mask)
        if len(_components_mask(mask)) != 1:
            continue
        try:
            actual_concave = _concave_count(coords)
        except AssertionError:
            continue
        if actual_concave != concave_count:
            continue
        interior = binary_fill_holes(mask) & ~mask
        if int(np.count_nonzero(interior)) == 0:
            continue
        return mask, interior
    raise RuntimeError(f"failed to generate loop with {concave_count} concave corners")


def _can_place_box(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(occupied.shape[0], r + h + pad)
    c1 = min(occupied.shape[1], c + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_loops(shape: tuple[int, int], specs: list[dict]) -> list[tuple[dict, int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placed: list[tuple[dict, int, int]] = []
    order = sorted(specs, key=lambda spec: int(spec["patch"].shape[0]) * int(spec["patch"].shape[1]), reverse=True)
    for spec in order:
        patch = spec["patch"]
        h, w = patch.shape
        candidates = []
        for r in range(1, shape[0] - h):
            for c in range(1, shape[1] - w):
                if _can_place_box(occupied, r, c, h, w, pad=1):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        placed.append((spec, r, c))
        occupied[max(0, r - 1) : min(shape[0], r + h + 1), max(0, c - 1) : min(shape[1], c + w + 1)] = True
    return placed


def _blocked_for_markers(shape: tuple[int, int], placed: list[tuple[dict, int, int]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for spec, r, c in placed:
        h, w = spec["patch"].shape
        blocked[max(0, r - 1) : min(shape[0], r + h + 1), max(0, c - 1) : min(shape[1], c + w + 1)] = True
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    return blocked


def _scatter_markers(grid: np.ndarray, blocked: np.ndarray, jobs: list[int]) -> bool:
    for color in jobs:
        choices = np.argwhere(~blocked)
        if choices.size == 0:
            return False
        r, c = map(int, choices[random.randrange(len(choices))])
        grid[r, c] = color
        blocked[max(0, r - 1) : min(grid.shape[0], r + 2), max(0, c - 1) : min(grid.shape[1], c + 2)] = True
    return True


def _build_expected(inp: np.ndarray, placed: list[tuple[dict, int, int]]) -> np.ndarray:
    out = inp.copy()
    out[(inp != BLACK) & (inp != GREY)] = BLACK
    for spec, r, c in placed:
        interior = spec["interior"]
        h, w = interior.shape
        view = out[r : r + h, c : c + w]
        view[interior] = int(spec["fill_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_loops = random.choice((2, 2, 3))
    concave_counts = sorted(random.sample((2, 4, 6, 8), n_loops))
    colors = random.sample(MARKER_COLORS, n_loops)

    specs = []
    for concave_count, fill_color in zip(concave_counts, colors):
        patch, interior = _make_loop_patch(concave_count)
        specs.append(
            {
                "patch": patch,
                "interior": interior,
                "concave_count": concave_count,
                "fill_color": fill_color,
            }
        )

    total_markers = sum(concave_counts)
    min_side = max(18, int(np.ceil(np.sqrt(total_markers * 10))))
    for _ in range(120):
        height = random.randint(min_side, 30)
        width = random.randint(min_side, 30)
        placed = _place_loops((height, width), specs)
        if placed is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        for spec, r, c in placed:
            patch = np.where(spec["patch"], GREY, BLACK)
            grid[r : r + patch.shape[0], c : c + patch.shape[1]][spec["patch"]] = GREY

        jobs: list[int] = []
        for spec in placed:
            loop_spec = spec[0]
            jobs.extend([int(loop_spec["fill_color"])] * int(loop_spec["concave_count"]))
        random.shuffle(jobs)

        blocked = _blocked_for_markers(grid.shape, placed)
        if not _scatter_markers(grid, blocked, jobs):
            continue

        if int(np.count_nonzero(grid == GREY)) == 0:
            continue
        if _get_bg(grid) != BLACK:
            continue

        expected = _build_expected(grid, placed)
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
