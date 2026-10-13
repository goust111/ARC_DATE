from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

# Normalized 2x2 L-triomino orientations.
L_PATTERNS: dict[int, frozenset[tuple[int, int]]] = {
    0: frozenset({(0, 0), (0, 1), (1, 0)}),  # missing bottom-right
    1: frozenset({(0, 0), (0, 1), (1, 1)}),  # missing bottom-left
    2: frozenset({(0, 0), (1, 0), (1, 1)}),  # missing top-right
    3: frozenset({(0, 1), (1, 0), (1, 1)}),  # missing top-left
}
PATTERN_TO_ID = {pattern: pid for pid, pattern in L_PATTERNS.items()}

# Under the anchored assembly convention in main(), these families yield a unique
# completed rectangle shape.
ORIENTATION_FAMILIES = {
    (2, 3): [(2, 2), (2, 3), (3, 2)],
    (3, 2): [(1, 1), (1, 3), (3, 1)],
}

GEN_ATTEMPTS = 400


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _normalize(coords: np.ndarray | set[tuple[int, int]] | frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        return frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _rotate90(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    return _normalize({(c, h - 1 - r) for r, c in shape})


def _rotations(shape: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    out: list[frozenset[tuple[int, int]]] = []
    cur = shape
    for _ in range(4):
        out.append(cur)
        cur = _rotate90(cur)
    return out


def _is_l_triomino(shape: frozenset[tuple[int, int]]) -> bool:
    if len(shape) != 3:
        return False
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) == 1 and max(cols) == 1 and shape in PATTERN_TO_ID


def _infer_rectangle_dims(shapes: list[frozenset[tuple[int, int]]]) -> tuple[int, int]:
    dims_found: set[tuple[int, int]] = set()
    assembly_options = [((2, 3), (0, 1)), ((3, 2), (1, 0))]

    for fixed_idx in (0, 1):
        fixed = shapes[fixed_idx]
        other = shapes[1 - fixed_idx]
        for dims, offset in assembly_options:
            rect = {(r, c) for r in range(dims[0]) for c in range(dims[1])}
            for rotated in _rotations(other):
                placed_other = {(r + offset[0], c + offset[1]) for r, c in rotated}
                if fixed & placed_other:
                    continue
                if fixed | placed_other == rect:
                    dims_found.add(dims)

    assert len(dims_found) == 1, "same-color L pair must imply a unique rectangle size"
    return next(iter(dims_found))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color L triominoes, infer the unique completed rectangle, and redraw that rectangle at the matching anchor.

    Input structure:
    - The background is black.
    - Every non-black color appears as exactly two 4-connected L-shaped triominoes, each occupying three cells of a 2x2 box.
    - The same color also appears once more as a singleton anchor pixel, separated from the L pieces.

    Rule:
    - Group components by color; for each color, identify its two L triominoes and its singleton anchor.
    - Treat the anchor as the top-left corner of the final assembled rectangle.
    - Test the two L shapes under the anchored assembly convention: one L occupies the output rectangle's upper-left 2x2 corner, while the other may rotate and fill the overlapping 2x2 block immediately to its right (forming a 2x3 rectangle) or immediately below it (forming a 3x2 rectangle).
    - If every valid complement agrees on one rectangle size, erase the original L pieces and fill that anchored rectangle with the same color.
    - The anchor pixel is covered by the rectangle.

    Key insight:
    - Color first pairs the fragments, then the fragment geometry determines whether the anchored completion must be wide (2x3) or tall (3x2).
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(h, w) <= 30

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    specs = []
    for color in colors:
        comps = _component_coords(g == color)
        anchors = [comp for comp in comps if len(comp) == 1]
        l_comps = [comp for comp in comps if _is_l_triomino(_normalize(comp))]
        assert len(comps) == 3, "each color must have exactly two L pieces and one anchor"
        assert len(anchors) == 1 and len(l_comps) == 2

        shapes = [_normalize(comp) for comp in l_comps]
        dims = _infer_rectangle_dims(shapes)
        anchor = tuple(map(int, anchors[0][0]))
        ar, ac = anchor
        assert ar + dims[0] <= h and ac + dims[1] <= w, "anchored rectangle must fit in the grid"

        erase_cells = [tuple(map(int, cell)) for comp in comps for cell in comp]
        specs.append({
            "color": color,
            "anchor": anchor,
            "dims": dims,
            "erase": erase_cells,
        })

    out = g.copy()
    for spec in specs:
        for r, c in spec["erase"]:
            out[r, c] = BG

    for spec in specs:
        ar, ac = spec["anchor"]
        rh, rw = spec["dims"]
        assert np.all(out[ar:ar + rh, ac:ac + rw] == BG), "completed rectangles must not overlap"
        out[ar:ar + rh, ac:ac + rw] = int(spec["color"])

    return out


def _buffer(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _translated(shape: frozenset[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _all_placements(
    h: int,
    w: int,
    shape: frozenset[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int, set[tuple[int, int]]]]:
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    placements = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            cells = _translated(shape, top, left)
            if cells & forbidden:
                continue
            placements.append((top, left, cells))
    return placements


def _sample_grid_shape() -> tuple[int, int]:
    return random.randint(12, 30), random.randint(12, 30)


def _sample_color_count(h: int, w: int) -> int:
    area = h * w
    if area < 240:
        cap = 2
    elif area < 420:
        cap = 3
    elif area < 650:
        cap = 4
    else:
        cap = 5
    return random.randint(1, cap)


def _render_expected(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        ar, ac = spec["anchor"]
        rh, rw = spec["dims"]
        out[ar:ar + rh, ac:ac + rw] = int(spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_grid_shape()
    n_colors = _sample_color_count(h, w)
    colors = list(range(1, 10))
    random.shuffle(colors)
    colors = colors[:n_colors]

    specs: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    branches = [(2, 3), (3, 2)]
    for color in colors:
        dims = random.choice(branches)
        orient_pair = random.choice(ORIENTATION_FAMILIES[dims])
        rect_shape = frozenset((r, c) for r in range(dims[0]) for c in range(dims[1]))
        placements = _all_placements(h, w, rect_shape, reserved)
        if not placements:
            return None
        top, left, rect_cells = random.choice(placements)
        reserved |= _buffer(rect_cells, h, w, pad=1)
        specs.append({
            "color": color,
            "dims": dims,
            "anchor": (top, left),
            "orient_pair": orient_pair,
            "pieces": [],
        })

    forbidden = set(reserved)
    place_order = list(range(len(specs)))
    random.shuffle(place_order)
    for idx in place_order:
        spec = specs[idx]
        orientations = list(spec["orient_pair"])
        random.shuffle(orientations)
        pieces = []
        for orient in orientations:
            shape = L_PATTERNS[orient]
            placements = _all_placements(h, w, shape, forbidden)
            if not placements:
                return None
            _top, _left, cells = random.choice(placements)
            forbidden |= _buffer(cells, h, w, pad=1)
            pieces.append(cells)
        spec["pieces"] = pieces

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        ar, ac = spec["anchor"]
        grid[ar, ac] = color
        for piece in spec["pieces"]:
            for r, c in piece:
                grid[r, c] = color

    expected = _render_expected(specs, (h, w))
    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-L puzzle")
