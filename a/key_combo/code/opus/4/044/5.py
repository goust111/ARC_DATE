from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
LEFT_COLORS = (1, 2, 3, 4)
GREY = 5
PANEL_SIZES = (4, 5)
GEN_ATTEMPTS = 400


RNG = random.Random()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(sorted(coords))
    return parts


def _majority_color(grid: np.ndarray, coords: list[tuple[int, int]]) -> int:
    counts = Counter(int(grid[r, c]) for r, c in coords)
    [(color, best), *rest] = counts.most_common()
    if rest:
        assert best > rest[0][1], "component majority color must be unique"
    return int(color)


def _recolored_stamp(template_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(template_mask.shape, dtype=int)
    patch[template_mask] = int(color)
    return patch


def _stamp_by_component(left: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    n = int(left.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for coords in _components(left != BLACK):
        rep = _majority_color(left, coords)
        patch = _recolored_stamp(template_mask, rep)
        for r, c in coords:
            r0 = r * n
            c0 = c * n
            out[r0 : r0 + n, c0 : c0 + n][template_mask] = patch[template_mask]
    return out


def _stamp_per_pixel(left: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    n = int(left.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for r, c in np.argwhere(left != BLACK):
        color = int(left[r, c])
        r0 = int(r) * n
        c0 = int(c) * n
        out[r0 : r0 + n, c0 : c0 + n][template_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the right-panel template into output blocks using each left-panel component's majority color.

    Input structure:
    - The grid has two equal square panels side by side, with the left panel as the source and the right panel as the template.
    - The left panel uses black as background and places a small set of blue/red/green/yellow pixels.
    - The right panel uses black as background and contains one 4-connected grey template shape.

    Rule:
    - Split the input into two N x N panels, where N is the grid height.
    - Find 4-connected components of the non-black pixels in the left panel while ignoring color differences.
    - For each component, take its unique most frequent color as that component's representative color.
    - Build an N^2 x N^2 black output made of N x N blocks, and for every non-black left-panel pixel at position (r, c), stamp the grey template into output block (r, c) recolored to that pixel's component representative color.
    - Leave blocks corresponding to black left-panel cells fully black.

    Key insight:
    - Color is chosen once per 4-connected source component, so minority colors inside a component are overwritten by the component majority when the template is stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, w = map(int, g.shape)
    assert w == 2 * n, "expected two equal square panels"

    left = g[:, :n]
    right = g[:, n:]
    assert set(np.unique(right)).issubset({BLACK, GREY}), "template panel must only use black and grey"

    template_mask = right == GREY
    template_components = _components(template_mask)
    assert len(template_components) == 1, "template must be one 4-connected grey shape"

    return _stamp_by_component(left, template_mask)


BASE_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 0),),),
    2: (((0, 0), (0, 1)),),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ),
    5: (
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ),
}


def _normalize(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in shape)
    c0 = min(c for _, c in shape)
    return tuple(sorted((r - r0, c - c0) for r, c in shape))


def _transform_shape(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = list(shape)
    for _ in range(rot % 4):
        pts = [(-c, r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize(tuple(pts))


def _shape_variants(size: int) -> list[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    for shape in BASE_SHAPES[size]:
        norm = _normalize(shape)
        for rot in range(4):
            for flip in (False, True):
                out.add(_transform_shape(norm, rot, flip))
    return sorted(out)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


LEFT_SHAPES = {size: _shape_variants(size) for size in (1, 2, 3, 4)}
TEMPLATE_SHAPES = {size: _shape_variants(size) for size in (3, 4, 5)}


def _component_size_specs(n: int) -> list[int]:
    options = [
        [3, 2, 1],
        [3, 2, 1, 1],
        [4, 2, 1],
        [3, 3, 1],
        [4, 2, 1, 1],
    ]
    if n == 5:
        options.extend([[4, 3, 1], [3, 2, 2, 1], [3, 3, 1, 1]])
    return list(RNG.choice(options))


def _can_place(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], n: int) -> bool:
    cell_set = set(cells)
    if (0, 0) in cell_set:
        return False
    for r, c in cell_set:
        if not (0 <= r < n and 0 <= c < n):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied and (nr, nc) not in cell_set:
                return False
    return True


def _place_components(n: int, sizes: list[int]) -> list[list[tuple[int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    placed: list[list[tuple[int, int]]] = []
    for size in sorted(sizes, reverse=True):
        variants = LEFT_SHAPES[size][:]
        RNG.shuffle(variants)
        candidates: list[list[tuple[int, int]]] = []
        for shape in variants:
            h, w = _shape_dims(shape)
            for r0 in range(n - h + 1):
                for c0 in range(n - w + 1):
                    cells = [(r0 + r, c0 + c) for r, c in shape]
                    if _can_place(cells, occupied, n):
                        candidates.append(sorted(cells))
        if not candidates:
            return None
        cells = RNG.choice(candidates)
        placed.append(cells)
        occupied.update(cells)
    RNG.shuffle(placed)
    return placed


def _assign_component_colors(cells: list[tuple[int, int]], force_mixed_top_left: bool) -> tuple[dict[tuple[int, int], int], int]:
    size = len(cells)
    cells = sorted(cells)
    if size == 1:
        color = RNG.choice(LEFT_COLORS)
        return {cells[0]: color}, color

    if size == 2:
        rep = RNG.choice(LEFT_COLORS)
        return {cell: rep for cell in cells}, rep

    rep = RNG.choice(LEFT_COLORS)
    other_colors = [c for c in LEFT_COLORS if c != rep]
    mixed = force_mixed_top_left or RNG.random() < 0.6

    if not mixed:
        return {cell: rep for cell in cells}, rep

    layout = {cell: rep for cell in cells}
    if size == 4 and RNG.random() < 0.35:
        minority_cells = [cells[0], cells[-1]] if force_mixed_top_left else RNG.sample(cells, 2)
        minority_main = RNG.choice(other_colors)
        tertiary = RNG.choice([c for c in other_colors if c != minority_main])
        layout[minority_cells[0]] = minority_main
        layout[minority_cells[1]] = tertiary
    else:
        minority_cell = cells[0] if force_mixed_top_left else RNG.choice(cells)
        layout[minority_cell] = RNG.choice(other_colors)
    return layout, rep


def _build_left_panel(n: int) -> np.ndarray | None:
    sizes = _component_size_specs(n)
    components = _place_components(n, sizes)
    if components is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    rep_colors: list[int] = []
    mixed_found = False
    top_left_mismatch = False

    multi_components = [coords for coords in components if len(coords) >= 3]
    if not multi_components:
        return None
    forced_mixed = min(multi_components, key=lambda coords: coords[0])

    for coords in components:
        force = coords == forced_mixed
        color_map, rep = _assign_component_colors(coords, force)
        rep_colors.append(rep)
        component_colors = {color_map[cell] for cell in coords}
        if len(component_colors) > 1:
            mixed_found = True
            if color_map[min(coords)] != rep:
                top_left_mismatch = True
        for cell, color in color_map.items():
            grid[cell] = color

    if len(set(rep_colors)) < 2:
        return None
    if not mixed_found or not top_left_mismatch:
        return None
    return grid


def _build_template_panel(n: int) -> np.ndarray:
    panel = np.zeros((n, n), dtype=int)
    size = RNG.choice((3, 4, 5))
    variants = TEMPLATE_SHAPES[size][:]
    RNG.shuffle(variants)
    for shape in variants:
        h, w = _shape_dims(shape)
        if h > n or w > n:
            continue
        r0 = RNG.randrange(n - h + 1)
        c0 = RNG.randrange(n - w + 1)
        for r, c in shape:
            panel[r0 + r, c0 + c] = GREY
        if len(np.unique(np.argwhere(panel == GREY)[:, 0])) >= 2 and len(np.unique(np.argwhere(panel == GREY)[:, 1])) >= 2:
            return panel
        panel.fill(BLACK)
    shape = RNG.choice(TEMPLATE_SHAPES[size])
    h, w = _shape_dims(shape)
    r0 = RNG.randrange(n - h + 1)
    c0 = RNG.randrange(n - w + 1)
    for r, c in shape:
        panel[r0 + r, c0 + c] = GREY
    return panel


def _validate_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    n, w = grid.shape
    if n not in PANEL_SIZES or w != 2 * n:
        return False

    left = grid[:, :n]
    right = grid[:, n:]

    if left[0, 0] != BLACK:
        return False
    if not set(np.unique(left)).issubset({BLACK, *LEFT_COLORS}):
        return False
    if not set(np.unique(right)).issubset({BLACK, GREY}):
        return False

    comps = _components(left != BLACK)
    if sum(len(comp) == 1 for comp in comps) < 1:
        return False
    if sum(len(comp) >= 2 for comp in comps) < 2:
        return False
    if not (5 <= int((left != BLACK).sum()) <= 8):
        return False

    reps: list[int] = []
    mixed_found = False
    top_left_mismatch = False
    for comp in comps:
        colors = [int(left[r, c]) for r, c in comp]
        counts = Counter(colors).most_common()
        if len(counts) > 1 and counts[0][1] <= counts[1][1]:
            return False
        rep = int(counts[0][0])
        reps.append(rep)
        if len(comp) > 1 and len(set(colors)) > 1:
            mixed_found = True
            if int(left[min(comp)]) != rep:
                top_left_mismatch = True

    if len(set(reps)) < 2 or not mixed_found or not top_left_mismatch:
        return False

    template_comps = _components(right == GREY)
    if len(template_comps) != 1:
        return False
    if len(template_comps[0]) not in (3, 4, 5):
        return False

    out = main(grid)
    if out.shape != (n * n, n * n) or max(out.shape) > 30:
        return False
    if np.array_equal(out, _stamp_per_pixel(left, right == GREY)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = RNG.choice(PANEL_SIZES)
        left = _build_left_panel(n)
        if left is None:
            continue
        right = _build_template_panel(n)
        grid = np.concatenate([left, right], axis=1)
        if _validate_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid connectivity-majority stamp input")
