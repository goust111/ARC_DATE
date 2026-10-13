from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PATH = 3
PALETTE_POOL = (1, 2, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
MIN_DIM = 10
MAX_DIM = 20


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


def _components_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in cells:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in _neighbors4(cur[0], cur[1], h, w):
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _path_turns_and_segments(path_cells: set[tuple[int, int]], h: int, w: int) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]]]:
    assert path_cells, "path must be non-empty"

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in path_cells:
        neighbors = [nxt for nxt in _neighbors4(cell[0], cell[1], h, w) if nxt in path_cells]
        neighbor_map[cell] = neighbors

    endpoints = [cell for cell, neighbors in neighbor_map.items() if len(neighbors) == 1]
    assert len(endpoints) == 2, f"path must have exactly two endpoints, got {endpoints}"
    assert all(r in (0, h - 1) or c in (0, w - 1) for r, c in endpoints), "endpoints must touch the border"

    for cell, neighbors in neighbor_map.items():
        deg = len(neighbors)
        assert deg in (1, 2), f"path must be a simple non-branching line, found degree {deg} at {cell}"

    seen: set[tuple[int, int]] = set()
    q = deque([endpoints[0]])
    seen.add(endpoints[0])
    while q:
        cur = q.popleft()
        for nxt in neighbor_map[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == path_cells, "path cells must form one connected component"

    turns: set[tuple[int, int]] = set()
    for cell, neighbors in neighbor_map.items():
        if len(neighbors) != 2:
            continue
        (r1, c1), (r2, c2) = neighbors
        if r1 != r2 and c1 != c2:
            turns.add(cell)

    segment_cells = path_cells - turns
    segments = _components_from_cells(segment_cells, h, w)
    assert segments, "removing turns must leave at least one straight segment"
    for segment in segments:
        rows = {r for r, _ in segment}
        cols = {c for _, c in segment}
        assert len(rows) == 1 or len(cols) == 1, "every segment must be horizontal or vertical"
    return turns, segments


def _extract_palette(grid: np.ndarray, bg: int, path_color: int) -> list[int]:
    h, w = grid.shape
    counts = Counter(int(v) for v in grid.ravel() if int(v) not in (bg, path_color))
    assert counts, "palette must exist"
    assert all(count == 1 for count in counts.values()), "every palette color must appear exactly once"

    cells: list[tuple[int, int, int]] = []
    for color in counts:
        pts = np.argwhere(grid == color)
        assert len(pts) == 1
        r, c = map(int, pts[0])
        cells.append((r, c, int(color)))

    rows = {r for r, _, _ in cells}
    assert len(rows) == 1, "palette must occupy exactly one row"
    row = rows.pop()
    assert row in (0, h - 1), "palette row must sit on the top or bottom edge"

    cols = sorted(c for _, c, _ in cells)
    assert cols == list(range(cols[0], cols[-1] + 1)), "palette colors must be tightly packed"
    assert cols[0] == 0 or cols[-1] == w - 1, "palette must touch a corner"

    by_col = {c: color for _, c, color in cells}
    palette = [by_col[c] for c in cols]
    assert len(set(palette)) == len(palette)

    inner_row = 1 if row == 0 else h - 2
    if 0 <= inner_row < h:
        assert all(int(grid[inner_row, c]) == bg for c in cols), "palette must have a clear inward buffer row"
    return palette


def _analyze_grid(grid: np.ndarray) -> tuple[int, int, list[int], set[tuple[int, int]], set[tuple[int, int]], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape

    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    path_candidates = [color for color, count in counts.items() if count > 1]
    assert len(path_candidates) == 1, f"expected exactly one multi-cell path color, got {path_candidates}"
    path_color = int(path_candidates[0])

    palette = _extract_palette(g, bg, path_color)
    path_cells = {(int(r), int(c)) for r, c in np.argwhere(g == path_color)}
    turns, segments = _path_turns_and_segments(path_cells, h, w)
    return bg, path_color, palette, path_cells, turns, segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the orthogonal path at its bends, map each straight-segment length through the corner palette,
    and recolor that segment. Bend pixels themselves stay green as the separators.
    """
    g = np.asarray(grid, dtype=int)
    _bg, path_color, palette, _path_cells, turns, segments = _analyze_grid(g)

    out = g.copy()
    for segment in segments:
        length = len(segment)
        new_color = int(palette[length - 1]) if length <= len(palette) else int(path_color)
        for r, c in segment:
            out[r, c] = new_color

    for r, c in turns:
        out[r, c] = int(path_color)
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _build_base_path(lengths: list[int], start_axis: int, orth_signs: list[int]) -> tuple[set[tuple[int, int]], int, int]:
    assert len(lengths) in (3, 5)
    assert len(lengths) % 2 == 1
    assert len(orth_signs) == len(lengths) // 2

    r = 0
    c = 0
    vertices = [(r, c)]
    sign_index = 0
    for i, length in enumerate(lengths):
        delta = int(length + (1 if 0 < i < len(lengths) - 1 else 0))
        if i % 2 == 0:
            if start_axis == 0:
                c += delta
            else:
                r += delta
        else:
            sign = orth_signs[sign_index]
            sign_index += 1
            if start_axis == 0:
                r += sign * delta
            else:
                c += sign * delta
        vertices.append((r, c))

    min_r = min(r0 for r0, _ in vertices)
    min_c = min(c0 for _, c0 in vertices)
    vertices = [(r0 - min_r, c0 - min_c) for r0, c0 in vertices]

    cells: set[tuple[int, int]] = set()
    for a, b in zip(vertices, vertices[1:]):
        cells.update(_line_cells(a, b))

    h = max(r0 for r0, _ in cells) + 1
    w = max(c0 for _, c0 in cells) + 1
    _path_turns_and_segments(cells, h, w)
    return cells, h, w


def _flip_cells(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    flip_rows: bool,
    flip_cols: bool,
) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        rr = h - 1 - r if flip_rows else r
        cc = w - 1 - c if flip_cols else c
        out.add((rr, cc))
    return out


def _palette_positions(h: int, w: int, corner: str, palette_len: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, c) for c in range(palette_len)]
    if corner == "tr":
        start = w - palette_len
        return [(0, c) for c in range(start, w)]
    if corner == "bl":
        return [(h - 1, c) for c in range(palette_len)]
    if corner == "br":
        start = w - palette_len
        return [(h - 1, c) for c in range(start, w)]
    raise ValueError(corner)


def _forbidden_cells_for_palette(h: int, w: int, corner: str, palette_len: int) -> set[tuple[int, int]]:
    palette_cells = set(_palette_positions(h, w, corner, palette_len))
    out = set(palette_cells)
    if corner.startswith("t"):
        inner_row = 1
    else:
        inner_row = h - 2
    if 0 <= inner_row < h:
        cols = [c for _, c in palette_cells]
        out.update((inner_row, c) for c in cols)
    return out


def _choose_lengths(seg_count: int) -> list[int]:
    lengths = [random.randint(1, 6) for _ in range(seg_count)]
    if seg_count == 5 and random.random() < 0.5:
        lengths[2] = random.randint(1, 7)
    return lengths


def _choose_palette_len(lengths: list[int], max_palette_len: int, need_overlong: bool) -> int | None:
    min_len = min(lengths)
    max_len = max(lengths)
    if max_palette_len < 3:
        return None
    if need_overlong:
        lo = max(3, min_len)
        hi = min(max_palette_len, max_len - 1)
        if lo > hi:
            return None
        return random.randint(lo, hi)
    lo = max(3, max_len)
    if lo > max_palette_len:
        return None
    return random.randint(lo, max_palette_len)


def _place_path(
    base_cells: set[tuple[int, int]],
    base_h: int,
    base_w: int,
    *,
    final_h: int,
    final_w: int,
    corner: str,
    palette_len: int,
    start_axis: int,
) -> set[tuple[int, int]] | None:
    forbidden = _forbidden_cells_for_palette(final_h, final_w, corner, palette_len)
    if start_axis == 0:
        slack = final_h - base_h
        offsets = list(range(slack + 1))
        random.shuffle(offsets)
        for off in offsets:
            placed = {(r + off, c) for r, c in base_cells}
            if placed.isdisjoint(forbidden):
                return placed
        return None

    slack = final_w - base_w
    offsets = list(range(slack + 1))
    random.shuffle(offsets)
    for off in offsets:
        placed = {(r, c + off) for r, c in base_cells}
        if placed.isdisjoint(forbidden):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    seg_count = random.choice((3, 5))
    lengths = _choose_lengths(seg_count)
    if len(set(lengths)) < 2:
        return None

    start_axis = random.choice((0, 1))
    orth_signs = [random.choice((-1, 1)) for _ in range(seg_count // 2)]
    try:
        base_cells, base_h, base_w = _build_base_path(lengths, start_axis, orth_signs)
    except AssertionError:
        return None

    if start_axis == 0:
        base_cells = _flip_cells(
            base_cells,
            base_h,
            base_w,
            flip_rows=bool(random.getrandbits(1)),
            flip_cols=bool(random.getrandbits(1)),
        )
        min_final_h = max(base_h + 2, MIN_DIM)
        if min_final_h > MAX_DIM or not (MIN_DIM <= base_w <= MAX_DIM):
            return None
        final_h = random.randint(min_final_h, MAX_DIM)
        final_w = base_w
    else:
        base_cells = _flip_cells(
            base_cells,
            base_h,
            base_w,
            flip_rows=bool(random.getrandbits(1)),
            flip_cols=bool(random.getrandbits(1)),
        )
        min_final_w = max(base_w + 2, MIN_DIM)
        if min_final_w > MAX_DIM or not (MIN_DIM <= base_h <= MAX_DIM):
            return None
        final_h = base_h
        final_w = random.randint(min_final_w, MAX_DIM)

    need_overlong = bool(random.getrandbits(1))
    palette_len = _choose_palette_len(lengths, min(len(PALETTE_POOL), final_w - 2), need_overlong)
    if palette_len is None:
        return None
    if not any(length <= palette_len for length in lengths):
        return None
    if need_overlong and not any(length > palette_len for length in lengths):
        return None
    if not need_overlong and not all(length <= palette_len for length in lengths):
        return None

    corner = random.choice(("tl", "tr", "bl", "br"))
    placed_cells = _place_path(
        base_cells,
        base_h,
        base_w,
        final_h=final_h,
        final_w=final_w,
        corner=corner,
        palette_len=palette_len,
        start_axis=start_axis,
    )
    if placed_cells is None:
        return None

    grid = np.full((final_h, final_w), BG, dtype=int)
    for r, c in placed_cells:
        grid[r, c] = PATH

    palette_colors = random.sample(PALETTE_POOL, palette_len)
    for (r, c), color in zip(_palette_positions(final_h, final_w, corner, palette_len), palette_colors):
        if grid[r, c] != BG:
            return None
        grid[r, c] = color

    try:
        _bg, _path_color, parsed_palette, _path_cells, turns, segments = _analyze_grid(grid)
    except AssertionError:
        return None

    segment_lengths = sorted(len(segment) for segment in segments)
    if segment_lengths != sorted(lengths):
        return None
    if len(turns) != seg_count - 1:
        return None
    if parsed_palette != palette_colors:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
