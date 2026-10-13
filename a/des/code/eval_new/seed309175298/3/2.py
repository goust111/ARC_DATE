from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _rect_perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        coords.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _legend_candidate(grid: np.ndarray, top: bool, left: bool) -> tuple[int, int, int, list[int]] | None:
    h, w = grid.shape
    r0 = 0 if top else h - 2
    if r0 < 0 or r0 + 1 >= h:
        return None

    if left:
        start = 0
        n = 0
        while n < w and int(grid[r0, n]) not in (BG, GREY):
            n += 1
    else:
        n = 0
        while n < w and int(grid[r0, w - 1 - n]) not in (BG, GREY):
            n += 1
        start = w - n

    if n < 2:
        return None

    colors = [int(v) for v in grid[r0, start : start + n]]
    if len(set(colors)) != n:
        return None
    if not np.all(grid[r0 + 1, start : start + n] == GREY):
        return None

    if start > 0:
        if int(grid[r0, start - 1]) != BG or int(grid[r0 + 1, start - 1]) != BG:
            return None
    if start + n < w:
        if int(grid[r0, start + n]) != BG or int(grid[r0 + 1, start + n]) != BG:
            return None

    probe_row = r0 + 2 if top else r0 - 1
    if 0 <= probe_row < h and np.any(grid[probe_row, start : start + n] != BG):
        return None

    return r0, start, n, colors


def _detect_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates = []
    for top in (True, False):
        for left in (True, False):
            cand = _legend_candidate(grid, top, left)
            if cand is not None:
                candidates.append(cand)
    if not candidates:
        raise ValueError("legend not found")
    candidates.sort(key=lambda item: item[2], reverse=True)
    r0, c0, n, colors = candidates[0]
    return (r0, c0, r0 + 1, c0 + n - 1), colors


def _track_paths(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    mask = np.zeros_like(g, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True
    grey_pts = np.argwhere((g == GREY) & ~mask)
    if grey_pts.size == 0:
        return []

    r0, c0 = map(int, grey_pts.min(axis=0))
    r1, c1 = map(int, grey_pts.max(axis=0))

    grey_boxes: list[tuple[int, int, int, int]] = []
    offset = 0
    while True:
        rr0, cc0 = r0 + offset, c0 + offset
        rr1, cc1 = r1 - offset, c1 - offset
        if rr1 - rr0 < 2 or cc1 - cc0 < 2:
            break
        box = (rr0, cc0, rr1, cc1)
        if all(int(g[r, c]) == GREY for r, c in _rect_perimeter_coords(*box)):
            grey_boxes.append(box)
            offset += 2
        else:
            break

    paths: list[list[tuple[int, int]]] = []
    for rr0, cc0, rr1, cc1 in grey_boxes[:-1]:
        paths.append(_rect_perimeter_coords(rr0 + 1, cc0 + 1, rr1 - 1, cc1 - 1))
    return paths


def _extract_fragments(values: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(values)
    while i < n:
        color = values[i]
        j = i + 1
        while j < n and values[j] == color:
            j += 1
        if color not in (BG, GREY):
            runs.append((int(color), int(j - i)))
        i = j

    if (
        len(runs) >= 2
        and values
        and values[0] not in (BG, GREY)
        and values[-1] == values[0]
    ):
        first_color, first_len = runs[0]
        _, last_len = runs[-1]
        runs[0] = (first_color, first_len + last_len)
        runs.pop()
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each orbit's colored fragments to follow the corner legend's color priority.

    Input structure:
    - bg = black.
    - A 2xN legend sits in one corner; its first row lists the active non-bg, non-grey colors from left to right, and its second row is grey.
    - Away from the legend, several concentric grey rectangles act as orbit boundaries; the 1-pixel rectangular lanes between adjacent grey rings contain colored fragments arranged along clockwise perimeter paths.

    Rule:
    - Read the legend's first row to get the global color priority.
    - For each orbit lane, traverse the lane from its top-left cell clockwise, collect the colored contiguous fragments, and stably sort those fragments by the legend priority.
    - Clear that lane to bg, then repack the sorted fragments from the lane's top-left cell clockwise with no extra gaps between successive fragments.
    - Keep the legend and all grey orbit boundaries unchanged.

    Key insight: The corner legend defines an explicit visible sort order, while the grey concentric rectangles reveal that each set of fragments lives on a separate cyclic rectangular path that must be linearized from the top-left corner.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    legend_bbox, order = _detect_legend(g)
    priority = {color: idx for idx, color in enumerate(order)}

    for path in _track_paths(g, legend_bbox):
        values = [int(g[r, c]) for r, c in path]
        fragments = _extract_fragments(values)
        indexed = list(enumerate(fragments))
        indexed.sort(key=lambda item: (priority.get(item[1][0], len(priority)), item[0]))

        for r, c in path:
            out[r, c] = BG

        write_pos = 0
        for _, (color, length) in indexed:
            for _ in range(length):
                if write_pos >= len(path):
                    break
                r, c = path[write_pos]
                out[r, c] = int(color)
                write_pos += 1

    return out


def _random_odd(lo: int, hi: int) -> int:
    lo = lo if lo % 2 == 1 else lo + 1
    hi = hi if hi % 2 == 1 else hi - 1
    if lo > hi:
        raise ValueError("no odd number in range")
    return random.randrange(lo, hi + 1, 2)


def _compose(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _choose_colors(k: int, order: list[int]) -> list[int]:
    if len(order) == 2:
        a, b = random.sample(order, 2)
        if random.random() < 0.5:
            a, b = b, a
        return [a if idx % 2 == 0 else b for idx in range(k)]

    for _ in range(200):
        active = random.sample(order, random.randint(2, len(order)))
        if len(active) == 2 and k % 2 == 1:
            continue
        seq = [random.choice(active)]
        for idx in range(1, k):
            choices = [c for c in active if c != seq[-1]]
            if idx == k - 1:
                choices = [c for c in choices if c != seq[0]]
            if not choices:
                break
            seq.append(random.choice(choices))
        if len(seq) == k and len(set(seq)) >= 2 and seq[0] != seq[-1]:
            return seq
    raise RuntimeError("failed to choose segment colors")


def _place_legend(grid: np.ndarray, colors: list[int], corner: str) -> None:
    h, w = grid.shape
    n = len(colors)
    r0 = 0 if corner in ("tl", "tr") else h - 2
    c0 = 0 if corner in ("tl", "bl") else w - n
    grid[r0, c0 : c0 + n] = np.array(colors, dtype=int)
    grid[r0 + 1, c0 : c0 + n] = GREY


def _track_segment_spec(path_len: int, order: list[int]) -> list[tuple[int, int]]:
    max_segments = min(8, max(2, path_len // 2))
    min_segments = min(max_segments, 4)
    if len(order) == 2:
        candidates = [value for value in range(min_segments, max_segments + 1) if value % 2 == 0]
        if not candidates:
            candidates = [2]
        k = random.choice(candidates)
    else:
        k = random.randint(min_segments, max_segments)
    lengths = _compose(path_len, k)
    colors = _choose_colors(k, order)
    return list(zip(colors, lengths))


def _sorted_spec(spec: list[tuple[int, int]], order: list[int]) -> list[tuple[int, int]]:
    priority = {color: idx for idx, color in enumerate(order)}
    indexed = list(enumerate(spec))
    indexed.sort(key=lambda item: (priority[item[1][0]], item[0]))
    return [item[1] for item in indexed]


def _paint_spec(grid: np.ndarray, path: list[tuple[int, int]], spec: list[tuple[int, int]]) -> None:
    idx = 0
    for color, length in spec:
        for _ in range(length):
            r, c = path[idx]
            grid[r, c] = int(color)
            idx += 1


def _margin_bases(corner: str, legend_n: int) -> tuple[int, int, int, int]:
    top = 3 if corner in ("tl", "tr") else 1
    bottom = 3 if corner in ("bl", "br") else 1
    left = legend_n + 2 if corner in ("tl", "bl") else 1
    right = legend_n + 2 if corner in ("tr", "br") else 1
    return top, bottom, left, right


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.randint(2, 5)
    order = random.sample(PALETTE, num_colors)
    corner = random.choice(("tl", "tr", "bl", "br"))
    tracks = random.randint(2, 4)

    base_top, base_bottom, base_left, base_right = _margin_bases(corner, len(order))
    min_outer = 4 * tracks + 3
    max_outer_h = 30 - base_top - base_bottom
    max_outer_w = 30 - base_left - base_right
    if max_outer_h < min_outer or max_outer_w < min_outer:
        return None

    outer_h = _random_odd(min_outer, max_outer_h)
    outer_w = _random_odd(min_outer, max_outer_w)

    extra_top = random.randint(0, min(3, 30 - (outer_h + base_top + base_bottom)))
    extra_bottom = random.randint(0, min(3, 30 - (outer_h + base_top + base_bottom + extra_top)))
    extra_left = random.randint(0, min(3, 30 - (outer_w + base_left + base_right)))
    extra_right = random.randint(0, min(3, 30 - (outer_w + base_left + base_right + extra_left)))

    top_margin = base_top + extra_top
    bottom_margin = base_bottom + extra_bottom
    left_margin = base_left + extra_left
    right_margin = base_right + extra_right

    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _place_legend(grid, order, corner)
    _place_legend(out, order, corner)

    r0, c0 = top_margin, left_margin
    r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1

    for inset in range(0, 2 * (tracks + 1), 2):
        rr0, cc0 = r0 + inset, c0 + inset
        rr1, cc1 = r1 - inset, c1 - inset
        for rr, cc in _rect_perimeter_coords(rr0, cc0, rr1, cc1):
            grid[rr, cc] = GREY
            out[rr, cc] = GREY

    used_colors: Counter[int] = Counter()
    changed = False
    for ring_idx in range(tracks):
        path = _rect_perimeter_coords(r0 + 1 + 2 * ring_idx, c0 + 1 + 2 * ring_idx, r1 - 1 - 2 * ring_idx, c1 - 1 - 2 * ring_idx)
        spec = _track_segment_spec(len(path), order)
        sorted_spec = _sorted_spec(spec, order)
        _paint_spec(grid, path, spec)
        _paint_spec(out, path, sorted_spec)
        used_colors.update(color for color, _ in spec)
        if spec != sorted_spec:
            changed = True

    if set(used_colors) != set(order):
        return None
    if not changed or np.array_equal(grid, out):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
