from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    dr = int(r1 - r0)
    dc = int(c1 - c0)
    assert not (dr == 0 and dc == 0), "line endpoints must be distinct"
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "endpoints must share row, column, or 45-degree diagonal"

    step_r = _sign(dr)
    step_c = _sign(dc)
    steps = max(abs(dr), abs(dc))
    return [(int(r0 + i * step_r), int(c0 + i * step_c)) for i in range(steps + 1)]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.pop()
            comp.append((int(r), int(c)))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        components.append(comp)
    return components


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != bg
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []

    for comp in _mask_components(mask):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 != h - 1 or c1 != w - 1:
            continue
        box = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not bool(np.all(box)):
            continue
        candidates.append((len(comp), (int(r0), int(c0), int(r1), int(c1))))

    assert len(candidates) == 1, "expected one filled bottom-right legend block"
    _area, (r0, c0, r1, c1) = candidates[0]
    legend = g[r0 : r1 + 1, c0 : c1 + 1]

    row_colors: list[int] = []
    for row in legend:
        colors = {int(v) for v in row.tolist()}
        assert len(colors) == 1, "legend rows must be solid-color strips"
        row_colors.append(int(next(iter(colors))))

    priority: list[int] = []
    for color in row_colors:
        if not priority or color != priority[-1]:
            priority.append(int(color))

    assert len(priority) >= 2, "legend must contain at least two priority colors"
    assert len(set(priority)) == len(priority), "legend colors must be unique"

    legend_mask = np.zeros_like(mask, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return priority, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored point pairs with straight segments and resolve crossings by the legend's layer priority.

    Input structure:
    - The background is a dominant gray-like color.
    - Outside the legend, each non-background color appears as exactly two isolated pixels.
    - The two pixels of each color are aligned horizontally, vertically, or on a 45-degree diagonal.
    - A filled rectangular legend block sits in the bottom-right corner, stacking the line colors from top to bottom by priority.

    Rule:
    - Find the bottom-right legend and read its colors from top to bottom; higher rows have higher drawing priority.
    - For every legend color, locate its two non-legend pixels and draw the full straight segment connecting them, including both endpoints.
    - When segments overlap, keep the color with the higher legend priority at that cell.
    - Leave the legend area itself unchanged.

    Key insight:
    - The puzzle combines geometric pairing with an explicit but easy-to-miss z-order key: the bottom-right legend tells which color should stay visible wherever multiple valid line segments cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    priority, legend_mask = _legend_info(g, bg)

    outside_colors = {int(v) for v in g[~legend_mask].tolist() if int(v) != bg}
    assert outside_colors == set(priority), "legend colors must match the point-pair colors"

    out = g.copy()
    for color in reversed(priority):
        pts = np.argwhere((g == int(color)) & (~legend_mask))
        assert pts.shape == (2, 2), "each line color must have exactly two non-legend endpoints"
        cells = _line_cells((int(pts[0, 0]), int(pts[0, 1])), (int(pts[1, 0]), int(pts[1, 1])))
        for r, c in cells:
            if not legend_mask[r, c]:
                out[r, c] = int(color)
    return out


# --------------- generator ---------------

_BG = 5
_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_ORIENTATIONS = ("h", "v", "d1", "d2")
_CROSSING_PAIRS = (("h", "v"), ("h", "d1"), ("h", "d2"), ("v", "d1"), ("v", "d2"), ("d1", "d2"))
_DIRS = {"h": (0, 1), "v": (1, 0), "d1": (1, 1), "d2": (1, -1)}

_GRID_MIN = 12
_GRID_MAX = 26
_MIN_COLORS = 3
_MAX_COLORS = 6
_MAX_LINE_LEN = 10
_MAX_TRIES = 300
_SEGMENT_TRIES = 200
_CROSS_TRIES = 120


def _segment(end_a: tuple[int, int], end_b: tuple[int, int], orientation: str) -> dict:
    cells = _line_cells(end_a, end_b)
    return {
        "orientation": orientation,
        "endpoints": (tuple(map(int, end_a)), tuple(map(int, end_b))),
        "cells": cells,
        "cell_set": set(cells),
    }


def _sample_orientations(count: int, used: list[str]) -> list[str]:
    if count <= 0:
        return []
    unused = [ori for ori in _ORIENTATIONS if ori not in used]
    picked = random.sample(unused, min(count, len(unused)))
    while len(picked) < count:
        picked.append(random.choice(_ORIENTATIONS))
    random.shuffle(picked)
    return picked


def _random_segment(height: int, width: int, valid_mask: np.ndarray, orientation: str) -> dict | None:
    max_len = min(_MAX_LINE_LEN, max(height, width))
    if orientation == "h":
        if width < 3:
            return None
        length = random.randint(3, min(max_len, width))
        r = random.randrange(height)
        c0 = random.randint(0, width - length)
        c1 = c0 + length - 1
        cand = _segment((r, c0), (r, c1), orientation)
    elif orientation == "v":
        if height < 3:
            return None
        length = random.randint(3, min(max_len, height))
        r0 = random.randint(0, height - length)
        c = random.randrange(width)
        cand = _segment((r0, c), (r0 + length - 1, c), orientation)
    elif orientation == "d1":
        if min(height, width) < 3:
            return None
        length = random.randint(3, min(max_len, height, width))
        r0 = random.randint(0, height - length)
        c0 = random.randint(0, width - length)
        cand = _segment((r0, c0), (r0 + length - 1, c0 + length - 1), orientation)
    else:
        if min(height, width) < 3:
            return None
        length = random.randint(3, min(max_len, height, width))
        r0 = random.randint(0, height - length)
        c0 = random.randint(length - 1, width - 1)
        cand = _segment((r0, c0), (r0 + length - 1, c0 - length + 1), orientation)

    if all(valid_mask[r, c] for r, c in cand["cells"]):
        return cand
    return None


def _can_add_segment(
    candidate: dict,
    segments: list[dict],
    used_crossings: set[tuple[int, int]],
    allow_crossings: bool,
) -> tuple[bool, list[tuple[int, int]]]:
    end_set = set(candidate["endpoints"])
    confirmed_hits: list[tuple[int, int]] = []
    for seg in segments:
        seg_end_set = set(seg["endpoints"])
        if end_set & seg["cell_set"]:
            return False, []
        if seg_end_set & candidate["cell_set"]:
            return False, []
        inter = candidate["cell_set"] & seg["cell_set"]
        if len(inter) > 1:
            return False, []
        if len(inter) == 1:
            cell = next(iter(inter))
            if cell in end_set or cell in seg_end_set:
                return False, []
            confirmed_hits.append(tuple(map(int, cell)))

    if not allow_crossings and confirmed_hits:
        return False, []
    if len(confirmed_hits) > 1:
        return False, []
    if any(cell in used_crossings for cell in confirmed_hits):
        return False, []
    return True, confirmed_hits


def _axis_limits(center: tuple[int, int], step: tuple[int, int], valid_mask: np.ndarray) -> tuple[int, int]:
    h, w = valid_mask.shape
    r, c = map(int, center)
    dr, dc = map(int, step)

    back = 0
    nr, nc = r - dr, c - dc
    while 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc]:
        back += 1
        nr -= dr
        nc -= dc

    forward = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc]:
        forward += 1
        nr += dr
        nc += dc
    return int(back), int(forward)


def _cross_segment(center: tuple[int, int], orientation: str, valid_mask: np.ndarray) -> dict | None:
    step = _DIRS[orientation]
    back, forward = _axis_limits(center, step, valid_mask)
    if back < 1 or forward < 1:
        return None
    left = random.randint(1, min(back, 4))
    right = random.randint(1, min(forward, 4))
    r, c = map(int, center)
    dr, dc = step
    a = (int(r - left * dr), int(c - left * dc))
    b = (int(r + right * dr), int(c + right * dc))
    return _segment(a, b, orientation)


def _make_crossing_pair(height: int, width: int, valid_mask: np.ndarray) -> list[dict] | None:
    pairs = list(_CROSSING_PAIRS)
    random.shuffle(pairs)
    centers = [tuple(map(int, rc)) for rc in np.argwhere(valid_mask)]
    random.shuffle(centers)
    for ori_a, ori_b in pairs:
        for center in centers[:_CROSS_TRIES]:
            seg_a = _cross_segment(center, ori_a, valid_mask)
            seg_b = _cross_segment(center, ori_b, valid_mask)
            if seg_a is None or seg_b is None:
                continue
            inter = seg_a["cell_set"] & seg_b["cell_set"]
            if inter == {center}:
                return [seg_a, seg_b]
    return None


def _count_crossings(segments: list[dict]) -> int:
    count = 0
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            inter = a["cell_set"] & b["cell_set"]
            if len(inter) == 1:
                count += 1
    return count


def _render_from_spec(
    height: int,
    width: int,
    priority: list[int],
    block_size: int,
    segments: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), _BG, dtype=int)
    legend_h = len(priority) * block_size
    legend_w = block_size
    legend_r0 = height - legend_h
    legend_c0 = width - legend_w

    for i, color in enumerate(priority):
        r0 = legend_r0 + i * block_size
        inp[r0 : r0 + block_size, legend_c0 : legend_c0 + block_size] = int(color)

    for seg in segments:
        (r0, c0), (r1, c1) = seg["endpoints"]
        inp[r0, c0] = int(seg["color"])
        inp[r1, c1] = int(seg["color"])

    out = inp.copy()
    legend_mask = np.zeros((height, width), dtype=bool)
    legend_mask[legend_r0:, legend_c0:] = True
    for color in reversed(priority):
        seg = next(item for item in segments if int(item["color"]) == int(color))
        for r, c in seg["cells"]:
            if not legend_mask[r, c]:
                out[r, c] = int(color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(_MIN_COLORS, _MAX_COLORS)
    block_size = random.choice((1, 2))
    legend_h = n_colors * block_size
    legend_w = block_size

    min_h = max(_GRID_MIN, legend_h + 4)
    min_w = max(_GRID_MIN, legend_w + 6)
    if min_h > _GRID_MAX or min_w > _GRID_MAX:
        return None

    height = random.randint(min_h, _GRID_MAX)
    width = random.randint(min_w, _GRID_MAX)

    legend_r0 = height - legend_h
    legend_c0 = width - legend_w
    valid_mask = np.ones((height, width), dtype=bool)
    valid_mask[max(legend_r0 - 1, 0) :, max(legend_c0 - 1, 0) :] = False
    if int(valid_mask.sum()) < n_colors * 10:
        return None

    colors = random.sample(_COLORS, n_colors)
    priority = colors.copy()
    random.shuffle(priority)

    need_cross = n_colors >= 3 and random.random() < 0.67
    segments: list[dict] = []
    used_crossings: set[tuple[int, int]] = set()
    used_orientations: list[str] = []

    if need_cross:
        pair = _make_crossing_pair(height, width, valid_mask)
        if pair is None:
            return None
        for seg, color in zip(pair, colors[:2]):
            seg["color"] = int(color)
            segments.append(seg)
            used_orientations.append(seg["orientation"])
        used_crossings.add(tuple(map(int, next(iter(pair[0]["cell_set"] & pair[1]["cell_set"])))))
        remaining_colors = colors[2:]
    else:
        remaining_colors = colors

    orientations = _sample_orientations(len(remaining_colors), used_orientations)
    for color, orientation in zip(remaining_colors, orientations):
        placed = False
        for _ in range(_SEGMENT_TRIES):
            cand = _random_segment(height, width, valid_mask, orientation)
            if cand is None:
                continue
            ok, hits = _can_add_segment(cand, segments, used_crossings, allow_crossings=need_cross)
            if not ok:
                continue
            cand["color"] = int(color)
            segments.append(cand)
            used_orientations.append(orientation)
            used_crossings.update(hits)
            placed = True
            break
        if not placed:
            return None

    if len(segments) != n_colors:
        return None

    crossing_count = _count_crossings(segments)
    if need_cross and crossing_count == 0:
        return None
    if not need_cross and crossing_count != 0:
        return None

    inp, expected = _render_from_spec(height, width, priority, block_size, segments)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo_new seed1852178230/5/2")
