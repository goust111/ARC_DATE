from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MARKER_POOL = (1, 2, 3, 4)
OBJECT_POOL = (6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_marker_positions(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    top: dict[int, int] = {}
    left: dict[int, int] = {}
    for c, value in enumerate(grid[0]):
        color = int(value)
        if color != bg:
            top[color] = int(c)
    for r, value in enumerate(grid[:, 0]):
        color = int(value)
        if color != bg:
            left[color] = int(r)

    shared = [color for color in top if color in left]
    assert shared, "expected paired top/left marker colors"
    return {int(color): (int(left[color]), int(top[color])) for color in shared}


def _find_scale_square(grid: np.ndarray, bg: int, marker_colors: set[int]) -> tuple[int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg or color in marker_colors:
            continue
        for comp in _color_components(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if r0 == 0 or c0 == 0:
                continue
            if height < 2 or height != width:
                continue
            if int(comp.shape[0]) != height * width:
                continue
            candidates.append((r1 + c1, r1, c1, color, height))

    assert candidates, "expected a filled square size marker"
    _score, _r1, _c1, color, size = max(candidates)
    return int(color), int(size)


def _topological_layers(crop: np.ndarray, colors: list[int]) -> list[int]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    areas: dict[int, int] = {}
    for color in colors:
        coords = np.argwhere(crop == color)
        boxes[color] = _bbox(coords)
        areas[color] = int(coords.shape[0])

    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for i, a in enumerate(colors):
        ar0, ac0, ar1, ac1 = boxes[a]
        for b in colors[i + 1 :]:
            br0, bc0, br1, bc1 = boxes[b]
            r0 = max(ar0, br0)
            c0 = max(ac0, bc0)
            r1 = min(ar1, br1)
            c1 = min(ac1, bc1)
            if r0 > r1 or c0 > c1:
                continue
            overlap = crop[r0 : r1 + 1, c0 : c1 + 1]
            sees_a = bool(np.any(overlap == a))
            sees_b = bool(np.any(overlap == b))
            if sees_a and not sees_b:
                edges[b].add(a)
            elif sees_b and not sees_a:
                edges[a].add(b)

    for lower in colors:
        for upper in edges[lower]:
            indeg[upper] += 1

    lower_first: list[int] = []
    queue = [color for color in colors if indeg[color] == 0]
    queue.sort(key=lambda color: (-areas[color], boxes[color]))
    while queue:
        color = queue.pop(0)
        lower_first.append(color)
        for nxt in sorted(edges[color], key=lambda other: (-areas[other], boxes[other])):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
        queue.sort(key=lambda color: (-areas[color], boxes[color]))

    if len(lower_first) != len(colors):
        lower_first = sorted(colors, key=lambda color: (_bbox(np.argwhere(crop == color)), -areas[color]))
    return list(reversed(lower_first))


def _decompose_crop(crop: np.ndarray, bg: int, grey: int) -> np.ndarray:
    size = int(crop.shape[0])
    colors = [int(color) for color in np.unique(crop) if int(color) not in (bg, grey)]
    if not colors:
        return np.full((size, size), bg, dtype=int)

    top_first = _topological_layers(crop, colors)
    panel_h = len(top_first) * size + (len(top_first) - 1)
    panel = np.full((panel_h, size), bg, dtype=int)

    for i, color in enumerate(top_first):
        coords = np.argwhere(crop == color)
        r0, c0, r1, c1 = _bbox(coords)
        start = i * (size + 1)
        panel[start + r0 : start + r1 + 1, c0 : c1 + 1] = color
        if i != len(top_first) - 1:
            panel[start + size, :] = grey
    return panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop marker-selected windows and split overlapping rectangles into ordered layer slices.

    Input structure:
    - The background is black, with single-pixel color markers on the top row and left column.
    - Each marker color appears exactly once on the top border and once on the left border, encoding one crop origin.
    - Interior cells contain several solid colored rectangles that may overlap by overpainting.
    - A small filled grey square near the lower-right interior corner gives the crop size S.

    Rule:
    - Pair each shared marker color to get a crop top-left coordinate (left-row, top-column), then extract its S x S window.
    - Inside each window, treat every visible color as one axis-aligned rectangle clipped by the window.
    - Infer the layer order from which rectangle color is visible inside pairwise bbox overlaps, then restore each rectangle to its full clipped bbox.
    - Output the windows from left to right by top-marker position, with each window rendered as vertically stacked layers separated by grey rows and different windows separated by grey columns.

    Key insight:
    - The border markers define where to crop, while the visible overlap pattern inside each crop reveals which rectangles are above or below and therefore how to reconstruct the hidden pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _most_common_color(g)

    markers = _find_marker_positions(g, bg)
    grey, size = _find_scale_square(g, bg, set(markers))

    ordered = sorted(markers.items(), key=lambda item: item[1][1])
    panels = []
    for _color, (row, col) in ordered:
        crop = g[row : row + size, col : col + size]
        assert crop.shape == (size, size), "marker crop must stay inside the grid"
        panels.append(_decompose_crop(crop, bg, grey))

    out_h = max(panel.shape[0] for panel in panels)
    out_w = len(panels) * size + (len(panels) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for i, panel in enumerate(panels):
        ph, pw = panel.shape
        out[:ph, c : c + pw] = panel
        c += pw
        if i != len(panels) - 1:
            out[:, c] = grey
            c += 1
    return out


def _layer_options(size: int) -> list[int]:
    options = [1, 2]
    if size >= 6:
        options.append(3)
    return options


def _shape_options(size: int, layers: int) -> list[tuple[int, int]]:
    if layers == 1:
        options = []
        for h in range(1, size + 1):
            for w in range(1, size + 1):
                if h != w and h * w >= 2:
                    options.append((h, w))
        return options

    step = 2 * (layers - 1)
    options = []
    for inner_h in range(1, size - step + 1):
        for inner_w in range(1, size - step + 1):
            if inner_h != inner_w and inner_h * inner_w >= 2:
                options.append((inner_h, inner_w))
    return options


def _window_scene(rng: np.random.Generator, size: int, layers: int) -> np.ndarray:
    scene = np.full((size, size), BG, dtype=int)
    colors = list(rng.choice(OBJECT_POOL, size=layers, replace=False))
    shape_choices = _shape_options(size, layers)
    inner_h, inner_w = shape_choices[int(rng.integers(0, len(shape_choices)))]

    if layers == 1:
        outer_h = inner_h
        outer_w = inner_w
    else:
        outer_h = inner_h + 2 * (layers - 1)
        outer_w = inner_w + 2 * (layers - 1)

    r0 = int(rng.integers(0, size - outer_h + 1))
    c0 = int(rng.integers(0, size - outer_w + 1))

    boxes = []
    for depth in range(layers):
        top = r0 + depth
        left = c0 + depth
        bottom = r0 + outer_h - depth - 1
        right = c0 + outer_w - depth - 1
        boxes.append((top, left, bottom, right))

    for color, (top, left, bottom, right) in zip(colors, boxes):
        scene[top : bottom + 1, left : right + 1] = int(color)
    return scene


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    size = int(rng.integers(4, 7))
    layers = int(rng.choice(_layer_options(size)))
    window_count = int(rng.integers(2, 5))

    width = 1 + window_count * size + (window_count - 1) + 1
    row_upper = 30 - size - 2
    if width > 30 or row_upper < window_count:
        return None

    cols = [1 + i * (size + 1) for i in range(window_count)]
    row_candidates = list(range(1, row_upper + 1))
    if len(row_candidates) < window_count:
        return None
    rows = sorted(int(v) for v in rng.choice(row_candidates, size=window_count, replace=False))

    grey_row = max(rows) + size + 1
    height = grey_row + size + 1
    if height > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    marker_colors = list(rng.choice(MARKER_POOL, size=window_count, replace=False))

    for color, row, col in zip(marker_colors, rows, cols):
        grid[0, col] = int(color)
        grid[row, 0] = int(color)
        grid[row : row + size, col : col + size] = _window_scene(rng, size, layers)

    grey_col = width - size
    grid[grey_row : grey_row + size, grey_col : grey_col + size] = GREY
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
