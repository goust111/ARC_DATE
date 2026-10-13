import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS_SHAPE = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
VBAR_SHAPE = frozenset({(0, 0), (1, 0), (2, 0)})

LEGEND_H = 6
LEGEND_W = 13
MAX_GEN_RETRIES = 300
MAX_PLACE_RETRIES = 250


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells):
    r0, c0, _, _ = _bbox(cells)
    return sorted((r - r0, c - c0) for r, c in cells)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _perimeter_cells(bbox):
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_aligned_singleton(singletons, col, min_row):
    candidates = [info for info in singletons if info["row"] > min_row and info["col"] == col]
    if not candidates:
        return None
    return min(candidates, key=lambda info: info["row"])


def _parse_legend(inner: np.ndarray, bg: int, border_color: int):
    infos = []
    for color in map(int, np.unique(inner)):
        if color in (bg, border_color):
            continue
        for cells in _components(inner == color):
            r0, c0, r1, c1 = _bbox(cells)
            infos.append(
                {
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                    "color": color,
                    "norm": frozenset(_normalize(cells)),
                    "row": r0,
                    "col": c0,
                    "center_col": (c0 + c1) // 2,
                }
            )

    pluses = [info for info in infos if info["area"] == 5 and info["norm"] == PLUS_SHAPE]
    bars = [info for info in infos if info["area"] == 3 and info["norm"] == VBAR_SHAPE]
    singletons = [info for info in infos if info["area"] == 1]
    if len(pluses) != 1 or len(bars) != 1 or len(singletons) < 4:
        return None

    node_shape = pluses[0]
    channel_shape = bars[0]
    node_color_info = _find_aligned_singleton(singletons, node_shape["center_col"], node_shape["bbox"][2])
    channel_color_info = _find_aligned_singleton(singletons, channel_shape["center_col"], channel_shape["bbox"][2])
    if node_color_info is None or channel_color_info is None:
        return None

    endpoint_pairs = []
    for info in singletons:
        below = _find_aligned_singleton(singletons, info["center_col"], info["row"])
        if below is not None:
            endpoint_pairs.append((info, below))
    if len(endpoint_pairs) != 1:
        return None
    endpoint_shape, endpoint_color_info = endpoint_pairs[0]

    colors = [endpoint_color_info["color"], channel_color_info["color"], node_color_info["color"]]
    if len(set(colors)) != 3:
        return None

    return {
        1: int(endpoint_color_info["color"]),
        2: int(channel_color_info["color"]),
        3: int(node_color_info["color"]),
    }


def _find_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for cells in _components(grid == color):
            bbox = _bbox(cells)
            r0, c0, r1, c1 = bbox
            if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 11:
                continue
            if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
                continue
            if set(cells) != _perimeter_cells(bbox):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            mapping = _parse_legend(inner, bg, color)
            if mapping is not None:
                candidates.append((bbox, color, mapping))
    assert len(candidates) == 1, "expected exactly one parseable legend frame"
    return candidates[0]


def _style_of_cell(cell, cell_set):
    r, c = cell
    neighbors = []
    for dr, dc in DIR4:
        if (r + dr, c + dc) in cell_set:
            neighbors.append((dr, dc))
    degree = len(neighbors)
    if degree >= 3:
        return 3, {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    if degree <= 1:
        return 1, {(0, 0)}

    has_h = any(dc != 0 for _, dc in neighbors)
    has_v = any(dr != 0 for dr, _ in neighbors)
    offsets = {(0, 0)}
    # A corner channel belongs to both a horizontal and a vertical segment,
    # so the local thickening is the union of the two perpendicular bars.
    if has_h:
        offsets.add((-1, 0))
        offsets.add((1, 0))
    if has_v:
        offsets.add((0, -1))
        offsets.add((0, 1))
    return 2, offsets


def _expanded_cells(cells):
    cell_set = set(cells)
    out = set()
    for cell in cells:
        _, offsets = _style_of_cell(cell, cell_set)
        r, c = cell
        for dr, dc in offsets:
            out.add((r + dr, c + dc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect colored one-pixel skeletons, classify each pixel by 4-neighbor topology,
    expand it with the legend-selected style, and repaint it with the legend colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    bg = _most_common_color(g)

    legend_bbox, legend_color, mapping = _find_legend(g, bg)
    r0, c0, r1, c1 = legend_bbox
    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = np.full_like(g, bg)
    out[legend_mask] = g[legend_mask]
    priority = np.zeros_like(g, dtype=int)
    paint = np.zeros_like(g, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, legend_color)]
    for color in colors:
        mask = (g == color) & ~legend_mask
        for cells in _components(mask):
            cell_set = set(cells)
            for cell in cells:
                kind, offsets = _style_of_cell(cell, cell_set)
                sr, sc = cell
                for dr, dc in offsets:
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and not legend_mask[rr, cc]:
                        if kind > priority[rr, cc]:
                            priority[rr, cc] = kind
                            paint[rr, cc] = mapping[kind]

    out[priority > 0] = paint[priority > 0]
    return out


def _random_transform(cells, rng: np.random.Generator):
    rot = int(rng.integers(0, 4))
    flip_r = bool(rng.integers(0, 2))
    flip_c = bool(rng.integers(0, 2))
    transformed = []
    for r, c in cells:
        rr, cc = r, c
        for _ in range(rot):
            rr, cc = cc, -rr
        if flip_r:
            rr = -rr
        if flip_c:
            cc = -cc
        transformed.append((rr, cc))
    return _normalize(transformed)


def _make_line(rng: np.random.Generator):
    length = int(rng.integers(4, 8))
    return [(0, c) for c in range(length)]


def _make_l(rng: np.random.Generator):
    a = int(rng.integers(3, 7))
    b = int(rng.integers(3, 7))
    cells = {(0, c) for c in range(a)}
    cells |= {(r, 0) for r in range(b)}
    return sorted(cells)


def _make_zigzag(rng: np.random.Generator):
    lengths = [int(rng.integers(2, 5)) for _ in range(3)]
    cells = {(0, 0)}
    r, c = 0, 0
    for dr, dc, steps in ((0, 1, lengths[0]), (1, 0, lengths[1]), (0, 1, lengths[2])):
        for _ in range(steps):
            r += dr
            c += dc
            cells.add((r, c))
    return sorted(cells)


def _make_t(rng: np.random.Generator):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    missing = int(rng.integers(0, 4))
    cells = {(0, 0)}
    for idx, (dr, dc) in enumerate(dirs):
        if idx == missing:
            continue
        length = int(rng.integers(1, 4))
        for step in range(1, length + 1):
            cells.add((dr * step, dc * step))
    return sorted(cells)


def _make_cross(rng: np.random.Generator):
    cells = {(0, 0)}
    for dr, dc in DIR4:
        length = int(rng.integers(1, 4))
        for step in range(1, length + 1):
            cells.add((dr * step, dc * step))
    return sorted(cells)


def _make_tree(rng: np.random.Generator):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    missing = int(rng.integers(0, 4))
    arm_lengths = {}
    cells = {(0, 0)}
    active_dirs = []
    for idx, (dr, dc) in enumerate(dirs):
        if idx == missing:
            continue
        length = int(rng.integers(1, 4))
        arm_lengths[(dr, dc)] = length
        active_dirs.append((dr, dc))
        for step in range(1, length + 1):
            cells.add((dr * step, dc * step))

    base_dir = active_dirs[int(rng.integers(0, len(active_dirs)))]
    end_r = base_dir[0] * arm_lengths[base_dir]
    end_c = base_dir[1] * arm_lengths[base_dir]
    if base_dir[0] == 0:
        bend_choices = [(-1, 0), (1, 0)]
    else:
        bend_choices = [(0, -1), (0, 1)]
    bend_dir = bend_choices[int(rng.integers(0, 2))]
    bend_len = int(rng.integers(2, 5))
    for step in range(1, bend_len + 1):
        cells.add((end_r + bend_dir[0] * step, end_c + bend_dir[1] * step))
    return sorted(cells)


def _buffer_cells(cells, shape, radius=1):
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _legend_bbox_for_edge(edge: str, h: int, w: int, rng: np.random.Generator):
    if edge == "top":
        return 0, int(rng.integers(0, w - LEGEND_W + 1)), LEGEND_H - 1, int(rng.integers(0, 1))
    if edge == "bottom":
        return h - LEGEND_H, int(rng.integers(0, w - LEGEND_W + 1)), LEGEND_H - 1, int(rng.integers(0, 1))
    if edge == "left":
        return int(rng.integers(0, h - LEGEND_H + 1)), 0, int(rng.integers(0, 1)), LEGEND_W - 1
    return int(rng.integers(0, h - LEGEND_H + 1)), w - LEGEND_W, int(rng.integers(0, 1)), LEGEND_W - 1


def _draw_legend(grid: np.ndarray, top: int, left: int, border_color: int, example_color: int, mapping):
    grid[top : top + LEGEND_H, left : left + LEGEND_W] = 0
    grid[top, left : left + LEGEND_W] = border_color
    grid[top + LEGEND_H - 1, left : left + LEGEND_W] = border_color
    grid[top : top + LEGEND_H, left] = border_color
    grid[top : top + LEGEND_H, left + LEGEND_W - 1] = border_color

    inner_r = top + 1
    inner_c = left + 1
    grid[inner_r + 1, inner_c + 1] = example_color
    grid[inner_r + 0 : inner_r + 3, inner_c + 5] = example_color
    grid[inner_r + 1, inner_c + 8 : inner_c + 11] = example_color
    grid[inner_r + 0, inner_c + 9] = example_color
    grid[inner_r + 2, inner_c + 9] = example_color
    grid[inner_r + 3, inner_c + 1] = mapping[1]
    grid[inner_r + 3, inner_c + 5] = mapping[2]
    grid[inner_r + 3, inner_c + 9] = mapping[3]


def _place_shape(local_cells, reserved_buffer, shape, rng: np.random.Generator):
    h, w = shape
    expanded = _expanded_cells(local_cells)
    er0, ec0, er1, ec1 = _bbox(expanded)
    min_r = 1 - er0
    max_r = (h - 2) - er1
    min_c = 1 - ec0
    max_c = (w - 2) - ec1
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(MAX_PLACE_RETRIES):
        off_r = int(rng.integers(min_r, max_r + 1))
        off_c = int(rng.integers(min_c, max_c + 1))
        placed_cells = {(r + off_r, c + off_c) for r, c in local_cells}
        placed_expanded = {(r + off_r, c + off_c) for r, c in expanded}
        placed_buffer = _buffer_cells(placed_expanded, shape, radius=1)
        if placed_buffer & reserved_buffer:
            continue
        return placed_cells, placed_buffer
    return None


def _gen_shape(kind: str, rng: np.random.Generator):
    makers = {
        "line": _make_line,
        "turn": lambda rr: _make_l(rr) if bool(rr.integers(0, 2)) else _make_zigzag(rr),
        "branch": lambda rr: [_make_t, _make_cross, _make_tree][int(rr.integers(0, 3))](rr),
        "any": lambda rr: [
            _make_line,
            _make_l,
            _make_zigzag,
            _make_t,
            _make_cross,
            _make_tree,
        ][int(rr.integers(0, 6))](rr),
    }
    return _random_transform(makers[kind](rng), rng)


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(22, 31))
    w = int(rng.integers(22, 31))
    n_objects = int(rng.integers(2, 5))

    if h < LEGEND_H + 4 or w < LEGEND_W + 4:
        return None

    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    border_color = palette[0]
    example_color = palette[1]
    mapping = {1: palette[2], 2: palette[3], 3: palette[4]}
    skeleton_colors = palette[5 : 5 + n_objects]
    if len(skeleton_colors) != n_objects:
        return None

    grid = np.zeros((h, w), dtype=int)
    edge = ["top", "bottom", "left", "right"][int(rng.integers(0, 4))]
    if edge == "top":
        legend_top = 0
        legend_left = int(rng.integers(0, w - LEGEND_W + 1))
    elif edge == "bottom":
        legend_top = h - LEGEND_H
        legend_left = int(rng.integers(0, w - LEGEND_W + 1))
    elif edge == "left":
        legend_top = int(rng.integers(0, h - LEGEND_H + 1))
        legend_left = 0
    else:
        legend_top = int(rng.integers(0, h - LEGEND_H + 1))
        legend_left = w - LEGEND_W

    _draw_legend(grid, legend_top, legend_left, border_color, example_color, mapping)

    legend_cells = {
        (r, c)
        for r in range(legend_top, legend_top + LEGEND_H)
        for c in range(legend_left, legend_left + LEGEND_W)
    }
    reserved_buffer = _buffer_cells(legend_cells, grid.shape, radius=1)

    shape_kinds = ["branch", "turn"] + ["any"] * (n_objects - 2)
    rng.shuffle(shape_kinds)

    for color, kind in zip(skeleton_colors, shape_kinds):
        placed = None
        for _ in range(MAX_PLACE_RETRIES):
            local_cells = _gen_shape(kind, rng)
            if len(local_cells) < 3:
                continue
            placed = _place_shape(local_cells, reserved_buffer, grid.shape, rng)
            if placed is not None:
                break
        if placed is None:
            return None
        placed_cells, placed_buffer = placed
        for r, c in placed_cells:
            grid[r, c] = color
        reserved_buffer |= placed_buffer

    try:
        out = main(grid)
        legend_bbox, _, parsed_mapping = _find_legend(grid, 0)
        r0, c0, r1, c1 = legend_bbox
        legend_mask = np.zeros(grid.shape, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert parsed_mapping == mapping
        assert np.array_equal(out[legend_mask], grid[legend_mask])
        painted = set(map(int, np.unique(out[~legend_mask]))) - {0}
        assert painted == {mapping[1], mapping[2], mapping[3]}
        for color in skeleton_colors:
            assert np.any(grid == color)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_RETRIES} attempts")
