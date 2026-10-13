from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
LEGEND_BG = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4)
TRANSFORMS = ("cw", "ccw", "flip", "rot180")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_TAILS = {
    "up": {(1, -1), (1, 1)},
    "down": {(-1, -1), (-1, 1)},
    "left": {(-1, 1), (1, 1)},
    "right": {(-1, -1), (1, -1)},
}

LEGEND_H = 12
LEGEND_W = 8
GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 34)
OBJECT_COUNT_RANGE = (4, 6)
SHAPE_SIZES = (3, 4, 5)
GEN_ATTEMPTS = 400

# Use a 3x3 asymmetric L so the four listed transforms stay visually distinct.
LEGEND_L = np.array(
    [
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 0],
    ],
    dtype=bool,
)


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int):
    if connectivity == 4:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    h, w = shape
    for dr, dc in offsets:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((int(rr), int(cc)))
            for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((int(nr), int(nc)))
        parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(int)
    return tuple(tuple(int(v) for v in row) for row in cropped)


def _transform_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "cw":
        return np.rot90(mask, -1)
    if kind == "ccw":
        return np.rot90(mask, 1)
    if kind == "flip":
        return np.fliplr(mask)
    if kind == "rot180":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown transform {kind}")


def _legend_transform_lookup() -> dict[tuple[tuple[int, ...], ...], str]:
    return {_mask_signature(_transform_mask(LEGEND_L, kind)): kind for kind in TRANSFORMS}


def _transform_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], kind: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        dr = r - cr
        dc = c - cc
        if kind == "cw":
            nr = cr + dc
            nc = cc - dr
        elif kind == "ccw":
            nr = cr - dc
            nc = cc + dr
        elif kind == "flip":
            nr = cr + dr
            nc = cc - dc
        elif kind == "rot180":
            nr = cr - dr
            nc = cc - dc
        else:
            raise AssertionError(f"unknown transform {kind}")
        if abs(nr - round(nr)) > 1e-9 or abs(nc - round(nc)) > 1e-9:
            raise AssertionError("transform must stay on integer lattice")
        out.add((int(round(nr)), int(round(nc))))
    return out


def _legend_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    assert int(grid[0, 0]) != bg
    mask = np.zeros(grid.shape, dtype=bool)
    queue = deque([(0, 0)])
    mask[0, 0] = True
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors(r, c, grid.shape, 4):
            if mask[nr, nc]:
                continue
            if int(grid[nr, nc]) == bg:
                continue
            mask[nr, nc] = True
            queue.append((int(nr), int(nc)))
    return mask


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, str], np.ndarray]:
    legend_mask = _legend_mask(grid, bg)
    r0, c0, r1, c1 = _bbox(list(zip(*np.where(legend_mask), strict=False)))
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    panel_bg = int(grid[0, 0])
    colors = {int(v) for v in np.unique(legend) if int(v) != panel_bg}
    assert len(colors) == 4

    lookup = _legend_transform_lookup()
    mapping: dict[int, str] = {}
    for color in colors:
        comps = _components(legend == color, connectivity=4)
        sizes = sorted(len(comp) for comp in comps)
        assert sizes == [1, 4]
        shape_comp = max(comps, key=len)
        rr0, cc0, rr1, cc1 = _bbox(shape_comp)
        patch = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        for r, c in shape_comp:
            patch[r - rr0, c - cc0] = True
        sig = _mask_signature(patch)
        assert sig in lookup
        mapping[int(color)] = lookup[sig]

    return mapping, legend_mask


def _parse_arrow(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    cell_set = set(cells)
    assert len(cell_set) == 3
    for tip in cell_set:
        rels = {(r - tip[0], c - tip[1]) for r, c in cell_set if (r, c) != tip}
        for name, tails in ARROW_TAILS.items():
            if rels == tails:
                return tip, DIRS[name]
    raise AssertionError("invalid arrow shape")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the color->transform legend, apply each arrow's transform to the grey
    shape it points to, erase the arrows, and keep the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(np.bincount(g.ravel()).argmax())

    transform_by_color, legend_mask = _parse_legend(g, bg)
    nonlegend = ~legend_mask
    outside_colors = {int(v) for v in np.unique(g[nonlegend]) if int(v) != bg}
    grey_candidates = outside_colors - set(transform_by_color)
    assert len(grey_candidates) == 1
    grey = grey_candidates.pop()

    grey_mask = (g == grey) & nonlegend
    grey_comps = _components(grey_mask, connectivity=4)
    assert grey_comps
    comp_id = np.full(g.shape, -1, dtype=int)
    for idx, comp in enumerate(grey_comps):
        for r, c in comp:
            comp_id[r, c] = idx

    targeted: dict[int, str] = {}
    arrow_cells: list[tuple[int, int]] = []
    for color, kind in transform_by_color.items():
        arrow_mask = (g == color) & nonlegend
        for comp in _components(arrow_mask, connectivity=8):
            assert len(comp) == 3
            tip, (dr, dc) = _parse_arrow(comp)
            step = 1
            target = None
            while True:
                rr = tip[0] + dr * step
                cc = tip[1] + dc * step
                assert 0 <= rr < h and 0 <= cc < w
                val = int(g[rr, cc])
                if val == grey:
                    target = int(comp_id[rr, cc])
                    assert target >= 0
                    break
                assert val == bg
                step += 1
            assert target not in targeted
            targeted[target] = kind
            arrow_cells.extend(comp)

    out = g.copy()
    for r, c in arrow_cells:
        out[r, c] = bg

    for cid, kind in targeted.items():
        for r, c in grey_comps[cid]:
            out[r, c] = bg
        moved = _transform_cells(grey_comps[cid], kind)
        for r, c in moved:
            assert 0 <= r < h and 0 <= c < w
            assert out[r, c] == bg
        for r, c in moved:
            out[r, c] = grey

    return out


def _random_shape(size: int) -> np.ndarray | None:
    for _ in range(200):
        target = random.randint(size + 1, size * size - 2)
        mask = np.zeros((size, size), dtype=bool)
        seed = (random.randrange(size), random.randrange(size))
        mask[seed] = True
        frontier = {cell for cell in _neighbors(seed[0], seed[1], mask.shape, 4) if not mask[cell]}
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nb in _neighbors(cell[0], cell[1], mask.shape, 4):
                if not mask[nb]:
                    frontier.add(nb)

        if int(mask.sum()) != target:
            continue
        rows, cols = np.where(mask)
        if int(rows.min()) != 0 or int(rows.max()) != size - 1:
            continue
        if int(cols.min()) != 0 or int(cols.max()) != size - 1:
            continue
        if any(np.array_equal(mask, _transform_mask(mask, kind)) for kind in TRANSFORMS):
            continue
        return mask
    return None


def _arrow_cells_for(mask: np.ndarray, r0: int, c0: int, direction: str) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    size = mask.shape[0]
    if direction == "right":
        options = [r for r in range(size) if mask[r, 0]]
        target_r = random.choice(options)
        tip = (r0 + target_r, c0 - 2)
        cells = {tip, (tip[0] - 1, tip[1] - 1), (tip[0] + 1, tip[1] - 1)}
    elif direction == "left":
        options = [r for r in range(size) if mask[r, size - 1]]
        target_r = random.choice(options)
        tip = (r0 + target_r, c0 + size + 1)
        cells = {tip, (tip[0] - 1, tip[1] + 1), (tip[0] + 1, tip[1] + 1)}
    elif direction == "down":
        options = [c for c in range(size) if mask[0, c]]
        target_c = random.choice(options)
        tip = (r0 - 2, c0 + target_c)
        cells = {tip, (tip[0] - 1, tip[1] - 1), (tip[0] - 1, tip[1] + 1)}
    elif direction == "up":
        options = [c for c in range(size) if mask[size - 1, c]]
        target_c = random.choice(options)
        tip = (r0 + size + 1, c0 + target_c)
        cells = {tip, (tip[0] + 1, tip[1] - 1), (tip[0] + 1, tip[1] + 1)}
    else:
        raise AssertionError(f"bad direction {direction}")
    gap = (tip[0] + DIRS[direction][0], tip[1] + DIRS[direction][1])
    return cells, gap


def _expand(cells: set[tuple[int, int]], h: int, w: int, pad: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_legend(grid: np.ndarray, transform_by_color: dict[int, str]) -> None:
    grid[:LEGEND_H, :LEGEND_W] = LEGEND_BG
    row_colors = list(ARROW_COLORS)
    random.shuffle(row_colors)
    for i, color in enumerate(row_colors):
        band_r = 3 * i
        grid[band_r + 1, 0] = color
        patch = _transform_mask(LEGEND_L, transform_by_color[color])
        patch_rows, patch_cols = np.where(patch)
        for r, c in zip(patch_rows.tolist(), patch_cols.tolist(), strict=False):
            grid[band_r + int(r), 2 + int(c)] = color


def _expected_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        for r, c in obj["arrow_cells"]:
            out[r, c] = BG
        for r, c in obj["shape_cells"]:
            out[r, c] = BG
    for obj in objects:
        moved = _transform_cells(obj["shape_cells"], obj["kind"])
        for r, c in moved:
            assert out[r, c] == BG
        for r, c in moved:
            out[r, c] = GREY
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    transform_list = list(TRANSFORMS)
    random.shuffle(transform_list)
    transform_by_color = {color: kind for color, kind in zip(ARROW_COLORS, transform_list, strict=False)}
    _place_legend(grid, transform_by_color)

    occupied = np.zeros((h, w), dtype=bool)
    for r, c in _expand({(rr, cc) for rr in range(LEGEND_H) for cc in range(LEGEND_W)}, h, w, pad=1):
        occupied[r, c] = True

    n_obj = random.randint(*OBJECT_COUNT_RANGE)
    colors = list(ARROW_COLORS)
    random.shuffle(colors)
    colors.extend(random.choices(ARROW_COLORS, k=n_obj - len(colors)))
    directions = list(DIRS)
    random.shuffle(directions)
    directions.extend(random.choices(tuple(DIRS), k=n_obj - len(directions)))
    specs = list(zip(colors[:n_obj], directions[:n_obj], strict=False))
    random.shuffle(specs)

    objects: list[dict] = []
    for color, direction in specs:
        placed = False
        for _ in range(500):
            size = random.choice(SHAPE_SIZES)
            mask = _random_shape(size)
            if mask is None:
                continue
            r0 = random.randint(1, h - size - 2)
            c0 = random.randint(1, w - size - 2)
            arrow_cells, gap = _arrow_cells_for(mask, r0, c0, direction)
            shape_cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
            bbox_cells = {(rr, cc) for rr in range(r0, r0 + size) for cc in range(c0, c0 + size)}
            footprint = bbox_cells | shape_cells | arrow_cells | {gap}
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in footprint):
                continue
            reserve = _expand(footprint, h, w, pad=1)
            if any(occupied[rr, cc] for rr, cc in reserve):
                continue

            moved = _transform_cells(shape_cells, transform_by_color[color])
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in moved):
                continue
            if any(cell not in bbox_cells for cell in moved):
                continue

            for r, c in shape_cells:
                grid[r, c] = GREY
            for r, c in arrow_cells:
                grid[r, c] = color
            for rr, cc in reserve:
                occupied[rr, cc] = True
            objects.append(
                {
                    "shape_cells": shape_cells,
                    "arrow_cells": arrow_cells,
                    "kind": transform_by_color[color],
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, objects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
