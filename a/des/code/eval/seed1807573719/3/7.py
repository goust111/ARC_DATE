from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _touches_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _layer_dist(region: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    if not region:
        return {}

    frontier = deque()
    dist: dict[tuple[int, int], int] = {}
    for cell in region:
        r, c = cell
        if any((nr, nc) not in region for nr, nc in _neighbors4(r, c)):
            dist[cell] = 1
            frontier.append(cell)

    while frontier:
        r, c = frontier.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in region and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                frontier.append(nxt)
    return dist


def _palette_and_seed_positions(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    h, w = grid.shape
    counts = Counter(map(int, grid.ravel()))

    palette_cells: dict[int, tuple[int, int]] = {}
    seed_cells: dict[int, tuple[int, int]] = {}
    for color, count in counts.items():
        if color == bg or count != 2:
            continue

        pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        border_pts = [pt for pt in pts if pt[0] in (0, h - 1) or pt[1] in (0, w - 1)]
        if len(border_pts) != 1:
            continue

        border_pt = border_pts[0]
        inner_pts = [pt for pt in pts if pt != border_pt]
        if len(inner_pts) != 1:
            continue

        palette_cells[color] = border_pt
        seed_cells[color] = inner_pts[0]

    assert palette_cells, "expected a palette bar with seed colors"

    rows = {r for r, _c in palette_cells.values()}
    cols = {c for _r, c in palette_cells.values()}
    if len(rows) == 1:
        ordered = sorted(palette_cells.items(), key=lambda item: (item[1][1], item[1][0]))
    elif len(cols) == 1:
        ordered = sorted(palette_cells.items(), key=lambda item: (item[1][0], item[1][1]))
    else:
        ordered = sorted(palette_cells.items(), key=lambda item: (item[1][0], item[1][1]))

    order = {color: idx + 1 for idx, (color, _pt) in enumerate(ordered)}
    return order, seed_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow concentric shrinking rings inside closed outlines, where ring depth is
    determined by the seed color's position in the edge palette bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    palette_order, seed_cells = _palette_and_seed_positions(g, bg)
    passable = g == bg
    for r, c in seed_cells.values():
        passable[r, c] = True

    out = g.copy()
    for comp in _components(passable):
        if _touches_border(comp, h, w):
            continue

        region = set(comp)
        comp_seed_colors = [color for color, pos in seed_cells.items() if pos in region]
        if len(comp_seed_colors) != 1:
            continue

        seed_color = int(comp_seed_colors[0])
        max_layers = int(palette_order[seed_color])
        dists = _layer_dist(region)
        for (r, c), dist in dists.items():
            if dist <= max_layers and g[r, c] == bg:
                out[r, c] = seed_color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 30)
SHAPE_COUNT_CHOICES = (2, 3, 3, 4, 4)
BAR_EDGES = ("top", "bottom")
SHAPE_MODES = ("rect", "corner_cut", "side_notch", "double_cut")
SHAPE_H_RANGE = (6, 10)
SHAPE_W_RANGE = (6, 10)
OUTER_RETRIES = 400
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120
PALETTE_SIZE = 10


def _filled_mask(mode: str, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)

    if mode == "rect":
        return mask

    if mode == "corner_cut":
        cut_h = random.randint(1, h - 3)
        cut_w = random.randint(1, w - 3)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
        return mask

    if mode == "side_notch":
        side = random.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            notch_w = random.randint(1, max(1, w - 4))
            start = random.randint(1, w - notch_w - 1)
            depth = random.randint(1, h - 3)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            mask[rows, start : start + notch_w] = False
        else:
            notch_h = random.randint(1, max(1, h - 4))
            start = random.randint(1, h - notch_h - 1)
            depth = random.randint(1, w - 3)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            mask[start : start + notch_h, cols] = False
        return mask

    first = _filled_mask("corner_cut", h, w)
    second = first.copy()
    for _ in range(2):
        cut_h = random.randint(1, h - 4)
        cut_w = random.randint(1, w - 4)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            second[:cut_h, :cut_w] = False
        elif corner == "tr":
            second[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            second[h - cut_h :, :cut_w] = False
        else:
            second[h - cut_h :, w - cut_w :] = False
    return second


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled)
    for r, c in np.argwhere(filled):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not filled[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _make_shape() -> dict[str, object] | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(*SHAPE_H_RANGE)
        w = random.randint(*SHAPE_W_RANGE)
        mode = random.choice(SHAPE_MODES)
        filled = _filled_mask(mode, h, w)

        if not np.any(filled):
            continue
        if len(_components(filled)) != 1:
            continue

        boundary = _boundary_mask(filled)
        interior = filled & (~boundary)
        comps = _components(interior)
        if len(comps) != 1:
            continue

        region = set(comps[0])
        dists = _layer_dist(region)
        if not dists:
            continue

        seed_choices = [pt for pt, dist in dists.items() if dist >= 2]
        if not seed_choices:
            continue

        return {
            "filled": filled,
            "boundary": boundary,
            "interior_dist": dists,
            "seed_local": random.choice(seed_choices),
            "shape": filled.shape,
            "mode": mode,
        }
    return None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_shapes(grid_h: int, grid_w: int, bar_edge: str, shapes: list[dict[str, object]]) -> bool:
    top_gap = 2 if bar_edge == "top" else 1
    bottom_gap = 2 if bar_edge == "bottom" else 1
    occupied: list[tuple[int, int, int, int]] = []

    random.shuffle(shapes)
    for spec in shapes:
        sh, sw = map(int, spec["shape"])
        min_r = top_gap
        max_r = grid_h - bottom_gap - sh
        min_c = 1
        max_c = grid_w - 1 - sw
        if min_r > max_r or min_c > max_c:
            return False

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(min_r, max_r)
            c0 = random.randint(min_c, max_c)
            bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            padded = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
            if any(_rects_overlap(padded, other) for other in occupied):
                continue

            spec["origin"] = (r0, c0)
            occupied.append(padded)
            placed = True
            break

        if not placed:
            return False
    return True


def _render_case(grid_h: int, grid_w: int, bg: int, border: int, bar_edge: str, shapes: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    seed_colors = random.sample([c for c in range(PALETTE_SIZE) if c not in (bg, border)], len(shapes))
    random.shuffle(seed_colors)
    for spec, color in zip(shapes, seed_colors, strict=False):
        spec["seed_color"] = int(color)

    palette_order_colors = seed_colors[:]
    random.shuffle(palette_order_colors)
    palette_order = {color: idx + 1 for idx, color in enumerate(palette_order_colors)}

    bar_row = 0 if bar_edge == "top" else grid_h - 1
    start_c = random.randint(1, grid_w - len(shapes) - 1)
    for offset, color in enumerate(palette_order_colors):
        grid[bar_row, start_c + offset] = int(color)

    for spec in shapes:
        r0, c0 = spec["origin"]
        boundary = spec["boundary"]
        for lr, lc in np.argwhere(boundary):
            grid[r0 + int(lr), c0 + int(lc)] = border
        sr, sc = spec["seed_local"]
        grid[r0 + sr, c0 + sc] = int(spec["seed_color"])

    out = grid.copy()
    for spec in shapes:
        r0, c0 = spec["origin"]
        seed_color = int(spec["seed_color"])
        max_layers = int(palette_order[seed_color])
        for (lr, lc), dist in spec["interior_dist"].items():
            if dist <= max_layers:
                gr = r0 + lr
                gc = c0 + lc
                if out[gr, gc] == bg:
                    out[gr, gc] = seed_color

    return grid, out


def _valid_input(grid: np.ndarray, expected: np.ndarray, bg: int, border: int, shape_count: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False
    if counts[border] <= shape_count:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    color_counts = Counter(map(int, grid.ravel()))
    seed_like = [color for color, count in color_counts.items() if color not in (bg, border) and count == 2]
    return len(seed_like) == shape_count


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    shape_count = random.choice(SHAPE_COUNT_CHOICES)
    bar_edge = random.choice(BAR_EDGES)

    colors = random.sample(range(PALETTE_SIZE), 2)
    bg, border = int(colors[0]), int(colors[1])

    shapes: list[dict[str, object]] = []
    for _ in range(shape_count):
        shape = _make_shape()
        if shape is None:
            return None
        shapes.append(shape)

    if not _place_shapes(grid_h, grid_w, bar_edge, shapes):
        return None

    grid, expected = _render_case(grid_h, grid_w, bg, border, bar_edge, shapes)
    if not _valid_input(grid, expected, bg, border, shape_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
