from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]

BG = 8
BEND_COLOR = 6
PAIR_COLORS = (1, 2, 3, 4, 7, 9)

SHAPES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)

PAIR_COUNT_RANGE = (3, 5)
PAIR_LAYOUTS: dict[int, tuple[tuple[int, int], ...]] = {
    3: ((1, 3), (2, 2)),
    4: ((2, 2),),
    5: ((2, 3),),
}
SLOT_GAP = 1
SLOT_H_RANGE = (7, 10)
SLOT_W_RANGE = (7, 8)
PAIR_PLACE_RETRIES = 200
LAYOUT_RETRIES = 100
GENERATE_RETRIES = 500
MIN_MARGIN = 1
MAX_MARGIN = 3


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _anchor_of(comp: list[Coord]) -> Coord:
    return min(comp, key=lambda rc: (int(rc[0]), int(rc[1])))


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1) + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1) + step, step)]
    raise ValueError("segment must be horizontal or vertical")


def _path_cells(anchor_a: Coord, anchor_b: Coord) -> list[Coord]:
    a, b = sorted((anchor_a, anchor_b), key=lambda rc: (int(rc[0]), int(rc[1])))
    if a[0] == b[0] or a[1] == b[1]:
        return _segment_cells(a, b)
    corner = (int(a[0]), int(b[1]))
    return _segment_cells(a, corner) + _segment_cells(corner, b)[1:]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color object pair by a fixed horizontal-then-vertical path.

    Input structure:
    - The background is azure and the top-left corner contains one magenta marker pixel.
    - There are 3-5 colors chosen from red, blue, green, yellow, orange, and maroon.
    - Each chosen color appears as exactly two separated 4-connected monochrome objects of area 2-4.

    Rule:
    - For each object, take its top-leftmost pixel as the anchor.
    - Order the two anchors of a same-color pair in reading order, then connect them by
      moving first horizontally to the second anchor's column and then vertically to the
      second anchor's row.
    - If the two anchors already share a row or column, the connection is a straight line
      drawn in the pair's own color; otherwise the L-shaped connection is drawn in the
      corner marker's magenta color.
    - Only recolor azure background cells along the path; keep all object pixels and the
      corner marker unchanged.

    Key insight:
    The route is not an arbitrary shortest path: every pair uses the same ordered
    horizontal-then-vertical geometry, and the lone magenta corner marker supplies the
    color specifically for bent connections.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    bend_color = int(g[0, 0])

    components_by_color: dict[int, list[list[Coord]]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _components_of_color(g, color)
        kept: list[list[Coord]] = []
        for comp in comps:
            if color == bend_color and len(comp) == 1 and comp[0] == (0, 0):
                continue
            kept.append(comp)
        if kept:
            components_by_color[int(color)] = kept

    out = g.copy()
    for color, comps in components_by_color.items():
        assert len(comps) == 2, f"expected one pair for color {color}, got {len(comps)} components"
        anchors = [_anchor_of(comp) for comp in comps]
        a, b = sorted(anchors, key=lambda rc: (int(rc[0]), int(rc[1])))
        path = _path_cells(a, b)
        draw_color = int(color) if (a[0] == b[0] or a[1] == b[1]) else int(bend_color)
        for r, c in path:
            if int(out[r, c]) == int(bg):
                out[r, c] = draw_color

    out[0, 0] = bend_color
    return out


def _shape_dims(shape: tuple[Coord, ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _place_shape(anchor: Coord, shape: tuple[Coord, ...]) -> set[Coord]:
    ar, ac = anchor
    return {(int(ar + dr), int(ac + dc)) for dr, dc in shape}


def _cells_too_close(a: set[Coord], b: set[Coord]) -> bool:
    for r0, c0 in a:
        for r1, c1 in b:
            if max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) <= 1:
                return True
    return False


def _sample_pair_kinds(pair_count: int) -> list[str]:
    options = ("h", "v", "l")
    for _ in range(100):
        kinds = [random.choice(options) for _ in range(pair_count)]
        if "l" in kinds and any(kind != "l" for kind in kinds):
            return kinds
    raise RuntimeError("failed to sample branch mix")


def _sample_margins(total_inner: int, limit: int = 30) -> tuple[int, int] | None:
    slack = int(limit) - int(total_inner)
    if slack < 2 * MIN_MARGIN:
        return None
    lo = MIN_MARGIN
    hi = min(MAX_MARGIN, slack - MIN_MARGIN)
    if hi < lo:
        return None
    first = random.randint(lo, hi)
    second_hi = min(MAX_MARGIN, slack - first)
    second_lo = MIN_MARGIN
    if second_hi < second_lo:
        return None
    second = random.randint(second_lo, second_hi)
    return int(first), int(second)


def _sample_layout(pair_count: int) -> tuple[int, int, int, int, int, int, int, int] | None:
    for _ in range(LAYOUT_RETRIES):
        rows, cols = random.choice(PAIR_LAYOUTS[pair_count])
        max_slot_h = min(SLOT_H_RANGE[1], (30 - 2 * MIN_MARGIN - (rows - 1) * SLOT_GAP) // rows)
        max_slot_w = min(SLOT_W_RANGE[1], (30 - 2 * MIN_MARGIN - (cols - 1) * SLOT_GAP) // cols)
        if max_slot_h < SLOT_H_RANGE[0] or max_slot_w < SLOT_W_RANGE[0]:
            continue
        slot_h = random.randint(SLOT_H_RANGE[0], max_slot_h)
        slot_w = random.randint(SLOT_W_RANGE[0], max_slot_w)

        inner_h = rows * slot_h + (rows - 1) * SLOT_GAP
        inner_w = cols * slot_w + (cols - 1) * SLOT_GAP
        vertical = _sample_margins(inner_h)
        horizontal = _sample_margins(inner_w)
        if vertical is None or horizontal is None:
            continue
        top, bottom = vertical
        left, right = horizontal
        if top + bottom + inner_h > 30 or left + right + inner_w > 30:
            continue
        return rows, cols, slot_h, slot_w, top, bottom, left, right
    return None


def _sample_anchors(kind: str, shape_a: tuple[Coord, ...], shape_b: tuple[Coord, ...], slot_h: int, slot_w: int) -> tuple[Coord, Coord] | None:
    ha, wa = _shape_dims(shape_a)
    hb, wb = _shape_dims(shape_b)

    if kind == "h":
        row_max = slot_h - max(ha, hb)
        if row_max < 0:
            return None
        row = random.randint(0, row_max)
        c0 = random.randint(0, slot_w - wa)
        c1 = random.randint(0, slot_w - wb)
        if abs(c0 - c1) < 3:
            return None
        return (row, c0), (row, c1)

    if kind == "v":
        col_max = slot_w - max(wa, wb)
        if col_max < 0:
            return None
        col = random.randint(0, col_max)
        r0 = random.randint(0, slot_h - ha)
        r1 = random.randint(0, slot_h - hb)
        if abs(r0 - r1) < 3:
            return None
        return (r0, col), (r1, col)

    if kind == "l":
        r0 = random.randint(0, slot_h - ha)
        c0 = random.randint(0, slot_w - wa)
        r1 = random.randint(0, slot_h - hb)
        c1 = random.randint(0, slot_w - wb)
        if abs(r0 - r1) < 2 or abs(c0 - c1) < 2:
            return None
        if r0 == r1 or c0 == c1:
            return None
        return (r0, c0), (r1, c1)

    raise ValueError(f"unknown kind: {kind}")


def _place_pair(slot_h: int, slot_w: int, kind: str) -> tuple[set[Coord], set[Coord], Coord, Coord] | None:
    for _ in range(PAIR_PLACE_RETRIES):
        shape_a = random.choice(SHAPES)
        shape_b = random.choice(SHAPES)
        anchors = _sample_anchors(kind, shape_a, shape_b, slot_h, slot_w)
        if anchors is None:
            continue
        anchor_a, anchor_b = anchors
        cells_a = _place_shape(anchor_a, shape_a)
        cells_b = _place_shape(anchor_b, shape_b)
        if cells_a & cells_b:
            continue
        if _cells_too_close(cells_a, cells_b):
            continue

        ordered_a, ordered_b = sorted((anchor_a, anchor_b), key=lambda rc: (int(rc[0]), int(rc[1])))
        path = set(_path_cells(ordered_a, ordered_b))
        if path & (cells_a | cells_b) != {ordered_a, ordered_b}:
            continue
        return cells_a, cells_b, anchor_a, anchor_b
    return None


def _draw_pair(grid: np.ndarray, cells_a: set[Coord], cells_b: set[Coord], color: int, origin: Coord) -> None:
    orow, ocol = origin
    for r, c in cells_a | cells_b:
        grid[orow + r, ocol + c] = int(color)


def _build_candidate() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    layout = _sample_layout(pair_count)
    if layout is None:
        return None
    rows, cols, slot_h, slot_w, top, bottom, left, right = layout
    height = top + bottom + rows * slot_h + (rows - 1) * SLOT_GAP
    width = left + right + cols * slot_w + (cols - 1) * SLOT_GAP

    grid = np.full((height, width), BG, dtype=int)
    grid[0, 0] = BEND_COLOR

    colors = random.sample(list(PAIR_COLORS), pair_count)
    kinds = _sample_pair_kinds(pair_count)

    slot_indices = list(range(rows * cols))
    random.shuffle(slot_indices)
    used_slots = slot_indices[:pair_count]

    for color, kind, slot_index in zip(colors, kinds, used_slots):
        local = _place_pair(slot_h, slot_w, kind)
        if local is None:
            return None
        cells_a, cells_b, _anchor_a, _anchor_b = local
        slot_r = slot_index // cols
        slot_c = slot_index % cols
        origin = (top + slot_r * (slot_h + SLOT_GAP), left + slot_c * (slot_w + SLOT_GAP))
        _draw_pair(grid, cells_a, cells_b, color, origin)

    return grid


def _is_valid_generated(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(inp[0, 0]) != BEND_COLOR:
        return False
    if _bg_color(inp) != BG:
        return False

    object_colors = [int(c) for c in np.unique(inp) if int(c) not in (BG, BEND_COLOR)]
    if not (PAIR_COUNT_RANGE[0] <= len(object_colors) <= PAIR_COUNT_RANGE[1]):
        return False

    for color in object_colors:
        comps = _components_of_color(inp, color)
        if len(comps) != 2:
            return False
        if not all(2 <= len(comp) <= 4 for comp in comps):
            return False

    out = main(inp)
    changed = (out != inp)
    if int(changed.sum()) == 0:
        return False

    bend_pixels = np.argwhere(changed & (out == BEND_COLOR))
    straight_pixels = np.argwhere(changed & np.isin(out, object_colors))
    if bend_pixels.size == 0 or straight_pixels.size == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        candidate = _build_candidate()
        if candidate is not None and _is_valid_generated(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid ARC input")
