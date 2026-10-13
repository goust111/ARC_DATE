from __future__ import annotations

import random
from collections import deque

import numpy as np


Rect = tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive


def _flood_component(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    seen: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return coords


def _extract_regions(body: np.ndarray) -> list[tuple[Rect, int, list[tuple[int, int]]]]:
    h, w = body.shape
    seen = np.zeros((h, w), dtype=bool)
    covered = np.zeros((h, w), dtype=bool)
    regions: list[tuple[Rect, int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(body[r, c])
            if color == 0 or seen[r, c]:
                continue
            coords = _flood_component(body, (r, c), color, seen)
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = body[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all((patch == color) | (patch == 0)):
                raise AssertionError("expected solid rectangles possibly containing black markers")
            regions.append(((r0, c0, r1, c1), color, coords))
            covered[r0 : r1 + 1, c0 : c1 + 1] = True

    if not np.all(covered):
        raise AssertionError("expected the body to be tiled by rectangular regions")
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a full divider through each marked rectangle, using geometry by default but flipping direction when the aligned reference color matches.

    Input structure:
    - The top row is a reference strip made of per-column colored pixels.
    - All lower cells are tiled by non-square monochrome rectangles, some of
      which contain a single black marker pixel.
    - Every marker aligns vertically with exactly one reference-strip pixel in
      the same column, and the strip itself is unchanged by the transformation.

    Rule:
    - For each marked rectangle, find its bounding box and compare the
      rectangle's fill color with the top-row reference color in the marker's
      column.
    - If the colors differ, draw a black divider through the marker along the
      rectangle's short axis: vertical for wide rectangles, horizontal for tall
      rectangles.
    - If the colors match, flip that direction and instead draw along the
      rectangle's long axis, still spanning only the rectangle itself.

    Key insight:
    - Rectangle aspect ratio suggests the default divider direction, but the
      aligned reference strip encodes an exception that reverses the orientation
      exactly when the colors match.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 2:
        raise AssertionError("expected a 2D grid with a reference row")

    out = g.copy()
    body = g[1:, :]
    regions = _extract_regions(body)
    marker_mask = body == 0
    assigned_markers = np.zeros_like(marker_mask, dtype=bool)

    for (r0, c0, r1, c1), color, _coords in regions:
        patch = body[r0 : r1 + 1, c0 : c1 + 1]
        markers = np.argwhere(patch == 0)
        if len(markers) == 0:
            continue

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == width:
            raise AssertionError("expected all rectangles to be non-square")

        for mr, mc in markers:
            rr = int(r0 + mr)
            cc = int(c0 + mc)
            assigned_markers[rr, cc] = True
            matches = int(g[0, cc]) == color
            draw_vertical = (width > height) != matches
            if draw_vertical:
                out[1 + r0 : 1 + r1 + 1, cc] = 0
            else:
                out[1 + rr, c0 : c1 + 1] = 0

    if not np.array_equal(assigned_markers, marker_mask):
        raise AssertionError("expected every black body pixel to be a rectangle marker")
    return out


_GEN_COLORS = (1, 2, 3, 4)
_GEN_BODY_MIN = 12
_GEN_BODY_MAX = 18
_GEN_RECT_COUNT_RANGE = (6, 7)
_GEN_MIN_SIDE = 3
_GEN_MARKED_MIN = 4
_GEN_MARKED_MAX = 6
_GEN_RETRIES = 400


def _rect_size(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _can_split_child(rect: Rect) -> bool:
    h, w = _rect_size(rect)
    return h >= 2 * _GEN_MIN_SIDE or w >= 2 * _GEN_MIN_SIDE


def _split_options(rect: Rect) -> list[tuple[Rect, Rect]]:
    r0, c0, r1, c1 = rect
    h, w = _rect_size(rect)
    options: list[tuple[Rect, Rect]] = []

    if h >= 2 * _GEN_MIN_SIDE:
        for cut in range(r0 + _GEN_MIN_SIDE, r1 - _GEN_MIN_SIDE + 2):
            top = (r0, c0, cut - 1, c1)
            bottom = (cut, c0, r1, c1)
            options.append((top, bottom))
    if w >= 2 * _GEN_MIN_SIDE:
        for cut in range(c0 + _GEN_MIN_SIDE, c1 - _GEN_MIN_SIDE + 2):
            left = (r0, c0, r1, cut - 1)
            right = (r0, cut, r1, c1)
            options.append((left, right))
    return options


def _adjacent(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_overlap = max(ar0, br0) <= min(ar1, br1)
    col_overlap = max(ac0, bc0) <= min(ac1, bc1)
    if ar1 + 1 == br0 or br1 + 1 == ar0:
        return col_overlap
    if ac1 + 1 == bc0 or bc1 + 1 == ac0:
        return row_overlap
    return False


def _assign_colors(rects: list[Rect]) -> list[int] | None:
    order = sorted(range(len(rects)), key=lambda i: sum(_adjacent(rects[i], rects[j]) for j in range(len(rects))), reverse=True)
    neighbors = {
        i: [j for j in range(len(rects)) if i != j and _adjacent(rects[i], rects[j])]
        for i in range(len(rects))
    }
    assigned: list[int | None] = [None] * len(rects)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        forbidden = {assigned[j] for j in neighbors[idx] if assigned[j] is not None}
        palette = list(_GEN_COLORS)
        random.shuffle(palette)
        for color in palette:
            if color in forbidden:
                continue
            assigned[idx] = color
            if dfs(pos + 1):
                return True
        assigned[idx] = None
        return False

    if not dfs(0):
        return None
    return [int(color) for color in assigned]


def _build_partition(body_h: int, body_w: int, target_count: int) -> list[Rect] | None:
    rects = [(0, 0, body_h - 1, body_w - 1)]
    while len(rects) < target_count:
        candidates = [(idx, _split_options(rect)) for idx, rect in enumerate(rects)]
        candidates = [(idx, opts) for idx, opts in candidates if opts]
        if not candidates:
            return None
        idx, opts = random.choice(candidates)
        a, b = random.choice(opts)
        rects[idx : idx + 1] = [a, b]

    if any((_rect_size(rect)[0] == _rect_size(rect)[1]) for rect in rects):
        return None
    wide = sum(_rect_size(rect)[1] > _rect_size(rect)[0] for rect in rects)
    tall = sum(_rect_size(rect)[0] > _rect_size(rect)[1] for rect in rects)
    if wide < 2 or tall < 2:
        return None
    if any(not _can_split_child(rect) and min(_rect_size(rect)) < _GEN_MIN_SIDE for rect in rects):
        return None
    return rects


def _interior_columns(rect: Rect) -> list[int]:
    _r0, c0, _r1, c1 = rect
    return list(range(c0 + 1, c1))


def _pick_distinct_columns(rects: list[Rect]) -> list[int] | None:
    order = sorted(range(len(rects)), key=lambda i: len(_interior_columns(rects[i])))
    chosen: list[int | None] = [None] * len(rects)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        cols = _interior_columns(rects[idx])
        random.shuffle(cols)
        for col in cols:
            if col in used:
                continue
            used.add(col)
            chosen[idx] = col
            if dfs(pos + 1):
                return True
            used.remove(col)
            chosen[idx] = None
        return False

    if not dfs(0):
        return None
    return [int(col) for col in chosen]


def _choose_marked(rects: list[Rect]) -> tuple[list[int], dict[int, bool]] | None:
    wide = [i for i, rect in enumerate(rects) if _rect_size(rect)[1] > _rect_size(rect)[0]]
    tall = [i for i, rect in enumerate(rects) if _rect_size(rect)[0] > _rect_size(rect)[1]]
    if len(wide) < 2 or len(tall) < 2:
        return None

    random.shuffle(wide)
    random.shuffle(tall)
    statuses = {
        wide[0]: True,
        wide[1]: False,
        tall[0]: True,
        tall[1]: False,
    }
    marked = list(statuses)

    remaining = [i for i in range(len(rects)) if i not in statuses]
    random.shuffle(remaining)
    extra = random.randint(0, min(_GEN_MARKED_MAX - len(marked), len(remaining)))
    for idx in remaining[:extra]:
        marked.append(idx)
        statuses[idx] = bool(random.randint(0, 1))

    if len(marked) < _GEN_MARKED_MIN:
        return None
    return marked, statuses


def _render_input(
    body_h: int,
    body_w: int,
    rects: list[Rect],
    colors: list[int],
    marked: list[int],
    match_status: dict[int, bool],
) -> np.ndarray | None:
    grid = np.empty((body_h + 1, body_w), dtype=int)
    grid[0, :] = np.random.choice(_GEN_COLORS, size=body_w)

    for rect, color in zip(rects, colors):
        r0, c0, r1, c1 = rect
        grid[1 + r0 : 1 + r1 + 1, c0 : c1 + 1] = color

    marked_rects = [rects[idx] for idx in marked]
    marker_cols = _pick_distinct_columns(marked_rects)
    if marker_cols is None:
        return None

    for local_idx, rect_idx in enumerate(marked):
        r0, c0, r1, c1 = rects[rect_idx]
        color = colors[rect_idx]
        marker_col = marker_cols[local_idx]
        marker_row = random.randint(r0 + 1, r1 - 1)
        grid[1 + marker_row, marker_col] = 0
        if match_status[rect_idx]:
            grid[0, marker_col] = color
        else:
            grid[0, marker_col] = random.choice([candidate for candidate in _GEN_COLORS if candidate != color])

    return grid


def _gen_attempt() -> np.ndarray | None:
    body_h = random.randint(_GEN_BODY_MIN, _GEN_BODY_MAX)
    body_w = random.randint(_GEN_BODY_MIN, _GEN_BODY_MAX)
    target_count = random.randint(*_GEN_RECT_COUNT_RANGE)

    rects = _build_partition(body_h, body_w, target_count)
    if rects is None:
        return None
    colors = _assign_colors(rects)
    if colors is None:
        return None

    marked_plan = _choose_marked(rects)
    if marked_plan is None:
        return None
    marked, statuses = marked_plan

    grid = _render_input(body_h, body_w, rects, colors, marked, statuses)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid[1:, :] == 0)) != len(marked):
        return None
    if sum(statuses[idx] for idx in marked) < 2 or sum(not statuses[idx] for idx in marked) < 2:
        return None
    if np.unique(grid[0, :]).size < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
