import random
from collections import Counter, deque

import numpy as np


BG = 8
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7]
MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    if BG in vals:
        bg_count = int(counts[np.where(vals == BG)[0][0]])
        if bg_count == int(counts.max()):
            return BG
    return int(vals[counts.argmax()])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _touch_edges(coords: list[tuple[int, int]], h: int, w: int) -> list[str]:
    edges = set()
    for r, c in coords:
        if r == 0:
            edges.add("top")
        if r == h - 1:
            edges.add("bottom")
        if c == 0:
            edges.add("left")
        if c == w - 1:
            edges.add("right")
    return sorted(edges)


def _scale2_about_center(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    cr = (min(rows) + max(rows)) / 2.0
    cc = (min(cols) + max(cols)) / 2.0
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        br = int(round(cr + 2 * (r - cr)))
        bc = int(round(cc + 2 * (c - cc)))
        out.add((br, bc))
        out.add((br + 1, bc))
        out.add((br, bc + 1))
        out.add((br + 1, bc + 1))
    return out


def _rotate180_about_touch_center(
    coords: list[tuple[int, int]], edge: str, h: int, w: int
) -> set[tuple[int, int]]:
    if edge == "top":
        cols = [c for r, c in coords if r == 0]
        pr, pc = 0.0, (min(cols) + max(cols)) / 2.0
    elif edge == "bottom":
        cols = [c for r, c in coords if r == h - 1]
        pr, pc = float(h - 1), (min(cols) + max(cols)) / 2.0
    elif edge == "left":
        rows = [r for r, c in coords if c == 0]
        pr, pc = (min(rows) + max(rows)) / 2.0, 0.0
    else:
        rows = [r for r, c in coords if c == w - 1]
        pr, pc = (min(rows) + max(rows)) / 2.0, float(w - 1)

    out: set[tuple[int, int]] = set()
    for r, c in coords:
        nr = int(round(2 * pr - r))
        nc = int(round(2 * pc - c))
        out.add((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        color = int(comp["color"])
        coords = comp["coords"]
        edges = _touch_edges(coords, h, w)
        touched = len(edges)

        if touched == 0:
            painted = _scale2_about_center(coords)
            paint_color = color
        elif touched == 1:
            painted = _rotate180_about_touch_center(coords, edges[0], h, w)
            paint_color = color
        elif touched == 2:
            painted = set(coords)
            paint_color = MAROON
        else:
            painted = set()
            paint_color = color

        for r, c in painted:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = paint_color

    return out


def _random_connected_shape(height: int, width: int, cells: int) -> set[tuple[int, int]]:
    cells = max(1, min(cells, height * width))
    start = (random.randrange(height), random.randrange(width))
    shape = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in shape:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(shape) < cells and frontier:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        shape.add(pick)
        add_frontier(*pick)
    return shape


def _trim_shape(shape: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return {(r - min_r, c - min_c) for r, c in shape}


def _make_corner_shape() -> set[tuple[int, int]]:
    height = random.randint(2, 4)
    width = random.randint(2, 4)
    cells = random.randint(max(3, (height + width) // 2), max(3, height * width - 1))
    shape = _random_connected_shape(height, width, cells)
    shape.add((0, 0))
    shape = _trim_shape(shape)
    if max(r for r, _ in shape) == 0:
        shape.add((1, 0))
    if max(c for _, c in shape) == 0:
        shape.add((0, 1))
    return _trim_shape(shape)


def _transform_shape(shape: set[tuple[int, int]], flip_r: bool, flip_c: bool) -> set[tuple[int, int]]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    out = set()
    for r, c in shape:
        nr = h - 1 - r if flip_r else r
        nc = w - 1 - c if flip_c else c
        out.add((nr, nc))
    return _trim_shape(out)


def _shape_to_cells(shape: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _reserve_ok(occupied: set[tuple[int, int]], cells: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _make_three_edge_shape(kind: str, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if kind == "top-left-right":
        for c in range(w):
            cells.add((0, c))
        stem_c = random.randint(max(1, w // 4), min(w - 2, (3 * w) // 4))
        stem_len = random.randint(1, max(1, min(4, h - 2)))
        for r in range(1, stem_len + 1):
            cells.add((r, stem_c))
    elif kind == "bottom-left-right":
        for c in range(w):
            cells.add((h - 1, c))
        stem_c = random.randint(max(1, w // 4), min(w - 2, (3 * w) // 4))
        stem_len = random.randint(1, max(1, min(4, h - 2)))
        for r in range(h - 2, h - 2 - stem_len, -1):
            cells.add((r, stem_c))
    elif kind == "left-top-bottom":
        for r in range(h):
            cells.add((r, 0))
        stem_r = random.randint(max(1, h // 4), min(h - 2, (3 * h) // 4))
        stem_len = random.randint(1, max(1, min(4, w - 2)))
        for c in range(1, stem_len + 1):
            cells.add((stem_r, c))
    else:
        for r in range(h):
            cells.add((r, w - 1))
        stem_r = random.randint(max(1, h // 4), min(h - 2, (3 * h) // 4))
        stem_len = random.randint(1, max(1, min(4, w - 2)))
        for c in range(w - 2, w - 2 - stem_len, -1):
            cells.add((stem_r, c))
    return cells


def _place_internal_shape(
    occupied: set[tuple[int, int]], h: int, w: int
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        sh = random.randint(2, 4)
        sw = random.randint(2, 4)
        cells = random.randint(max(3, (sh * sw) // 2), sh * sw)
        shape = _random_connected_shape(sh, sw, cells)
        top = random.randint(2, h - sh - 2)
        left = random.randint(2, w - sw - 2)
        placed = _shape_to_cells(shape, top, left)
        if _reserve_ok(occupied, placed, h, w):
            return placed
    return None


def _place_single_edge_shape(
    occupied: set[tuple[int, int]], h: int, w: int, side: str, blocked_corner: str
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        sh = random.randint(2, 4)
        sw = random.randint(2, 4)
        cells = random.randint(max(3, (sh * sw) // 2), sh * sw)
        shape = _random_connected_shape(sh, sw, cells)

        if side == "top":
            top = 0
            left_lo = 1 if blocked_corner == "right" else 2
            left_hi = w - sw - (2 if blocked_corner == "left" else 1)
            if left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)
        elif side == "bottom":
            top = h - sh
            left_lo = 1 if blocked_corner == "right" else 2
            left_hi = w - sw - (2 if blocked_corner == "left" else 1)
            if left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)
        elif side == "left":
            left = 0
            top_lo = 1 if blocked_corner == "bottom" else 2
            top_hi = h - sh - (2 if blocked_corner == "top" else 1)
            if top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)
        else:
            left = w - sw
            top_lo = 1 if blocked_corner == "bottom" else 2
            top_hi = h - sh - (2 if blocked_corner == "top" else 1)
            if top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)

        placed = _shape_to_cells(shape, top, left)
        touches = _touch_edges(list(placed), h, w)
        if len(touches) != 1 or touches[0] != side:
            continue
        if _reserve_ok(occupied, placed, h, w):
            return placed
    return None


def _place_corner_shape(
    occupied: set[tuple[int, int]], h: int, w: int, corner: str
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        shape = _make_corner_shape()
        if corner == "tl":
            placed = shape
        elif corner == "tr":
            placed = _shape_to_cells(_transform_shape(shape, False, True), 0, w - max(c for _, c in shape) - 1)
        elif corner == "bl":
            placed = _shape_to_cells(_transform_shape(shape, True, False), h - max(r for r, _ in shape) - 1, 0)
        else:
            placed = _shape_to_cells(
                _transform_shape(shape, True, True),
                h - max(r for r, _ in shape) - 1,
                w - max(c for _, c in shape) - 1,
            )
        touches = _touch_edges(list(placed), h, w)
        if len(touches) != 2:
            continue
        if _reserve_ok(occupied, placed, h, w):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    three_kind = random.choice(
        ["top-left-right", "bottom-left-right", "left-top-bottom", "right-top-bottom"]
    )
    branch_specs = {
        "top-left-right": ("bottom", ["bl", "br"]),
        "bottom-left-right": ("top", ["tl", "tr"]),
        "left-top-bottom": ("right", ["tr", "br"]),
        "right-top-bottom": ("left", ["tl", "bl"]),
    }
    free_side, corner_choices = branch_specs[three_kind]
    corner = random.choice(corner_choices)

    three_cells = _make_three_edge_shape(three_kind, h, w)
    if not _reserve_ok(occupied, three_cells, h, w):
        return None
    occupied |= three_cells

    corner_cells = _place_corner_shape(occupied, h, w, corner)
    if corner_cells is None:
        return None
    occupied |= corner_cells

    blocked_corner = {
        ("top", "tl"): "left",
        ("top", "tr"): "right",
        ("bottom", "bl"): "left",
        ("bottom", "br"): "right",
        ("left", "tl"): "top",
        ("left", "bl"): "bottom",
        ("right", "tr"): "top",
        ("right", "br"): "bottom",
    }[(free_side, corner)]
    one_edge_cells = _place_single_edge_shape(occupied, h, w, free_side, blocked_corner)
    if one_edge_cells is None:
        return None
    occupied |= one_edge_cells

    internal_cells = _place_internal_shape(occupied, h, w)
    if internal_cells is None:
        return None

    colors = random.sample(PALETTE, 4)
    _paint(grid, three_cells, colors[0])
    _paint(grid, corner_cells, colors[1])
    _paint(grid, one_edge_cells, colors[2])
    _paint(grid, internal_cells, colors[3])

    if random.random() < 0.35:
        extra = _place_internal_shape(occupied | internal_cells, h, w)
        if extra is not None:
            _paint(grid, extra, random.choice([c for c in PALETTE if c not in colors]))

    comps = _find_components(grid, BG)
    branch_counts = Counter(len(_touch_edges(comp["coords"], h, w)) for comp in comps)
    if not all(branch_counts.get(k, 0) >= 1 for k in [0, 1, 2, 3]):
        return None

    out = main(grid)
    fg = set(map(tuple, np.argwhere(out != BG)))
    if not fg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
