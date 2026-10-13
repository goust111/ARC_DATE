import random

import numpy as np


BG = 0
AZURE = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
QUADRANT_FROM_MISSING = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}


def _neighbors(r: int, c: int, shape: tuple[int, int], *, connectivity: int) -> list[tuple[int, int]]:
    dirs = DIRS4 if connectivity == 4 else DIRS8
    out = []
    h, w = shape
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _mask_components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, mask.shape, connectivity=connectivity):
                    if bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for cells in _mask_components(grid == AZURE, connectivity=4):
        cell_set = set(cells)
        assert len(cell_set) == 3, "each arrow must have exactly three azure cells"
        rows = [r for r, _c in cell_set]
        cols = [c for _r, c in cell_set]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must occupy a 2x2 box with one missing corner"
        full_box = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
        missing = list(full_box - cell_set)
        assert len(missing) == 1
        mr, mc = missing[0]
        direction = QUADRANT_FROM_MISSING[(mr - r0, mc - c0)]
        arrows.append(
            {
                "cells": cell_set,
                "direction": direction,
            }
        )
    assert arrows, "expected at least one arrow"
    return arrows


def _vertical_run(grid: np.ndarray, start_r: int, col: int, color: int) -> set[tuple[int, int]]:
    top = start_r
    while top - 1 >= 0 and int(grid[top - 1, col]) == color:
        top -= 1
    bottom = start_r
    while bottom + 1 < grid.shape[0] and int(grid[bottom + 1, col]) == color:
        bottom += 1
    return {(r, col) for r in range(top, bottom + 1)}


def _extract_portals(grid: np.ndarray, arrows: list[dict]) -> list[dict]:
    portals_by_cells: dict[frozenset[tuple[int, int]], dict] = {}
    for arrow in arrows:
        adjacent = set()
        for r, c in arrow["cells"]:
            for nr, nc in _neighbors(r, c, grid.shape, connectivity=4):
                color = int(grid[nr, nc])
                if color not in (BG, AZURE):
                    adjacent.add((nr, nc))
        assert adjacent, "each arrow must touch a portal"

        groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for cell in adjacent:
            key = (int(grid[cell]), cell[1])
            groups.setdefault(key, set()).add(cell)

        candidates = []
        for (color, col), group_cells in groups.items():
            seed_r = next(iter(group_cells))[0]
            cells = _vertical_run(grid, seed_r, col, color)
            if len(cells) < 2:
                continue
            if not group_cells <= cells:
                continue
            candidates.append((len(cells), cells, color, col))

        assert candidates, "arrow must identify a vertical portal segment"
        max_len = max(length for length, _cells, _color, _col in candidates)
        best = [(cells, color, col) for length, cells, color, col in candidates if length == max_len]
        assert len(best) == 1, "arrow must touch exactly one portal segment"
        cells, color, col = best[0]
        rows = sorted(r for r, _c in cells)
        assert rows == list(range(rows[0], rows[-1] + 1)), "portal must be contiguous"
        info = portals_by_cells.setdefault(
            frozenset(cells),
            {
                "cells": cells,
                "color": color,
                "col": col,
                "top": rows[0],
                "bottom": rows[-1],
                "direction": arrow["direction"],
            },
        )
        assert info["direction"] == arrow["direction"], "portal must have a unique arrow"
    portals = list(portals_by_cells.values())
    assert portals, "expected at least one portal"

    colors_to_portals: dict[int, list[dict]] = {}
    for portal in portals:
        colors_to_portals.setdefault(int(portal["color"]), []).append(portal)
    for same_color in colors_to_portals.values():
        assert len(same_color) == 2, "each portal color must appear exactly twice"
    return portals


def _crop_patch(grid: np.ndarray, cells: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _placement_origin(portal: dict, shape: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    dr, dc = direction
    row0 = int(portal["top"]) - h + 1 if dr < 0 else int(portal["bottom"])
    col0 = int(portal["col"]) - w if dc < 0 else int(portal["col"]) + 1
    return row0, col0


def _stamp_non_bg(grid: np.ndarray, patch: np.ndarray, row0: int, col0: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            color = int(patch[r, c])
            if color == BG:
                continue
            rr, cc = row0 + r, col0 + c
            assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1], "placement must stay in bounds"
            assert int(grid[rr, cc]) == BG, "teleported object must land on empty cells"
            grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each multi-color object through the same-color partner portal and
    re-emit it in the exit arrow's open-quadrant direction after a horizontal mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows = _extract_arrows(g)
    portals = _extract_portals(g, arrows)
    portals_by_color: dict[int, list[dict]] = {}
    portal_by_cell: dict[tuple[int, int], dict] = {}
    for portal in portals:
        portals_by_color.setdefault(int(portal["color"]), []).append(portal)
        for cell in portal["cells"]:
            portal_by_cell[cell] = portal

    portal_cells = set(portal_by_cell)
    object_mask = (g != BG) & (g != AZURE)
    for r, c in portal_cells:
        object_mask[r, c] = False

    objects = _mask_components(object_mask, connectivity=8)
    assert objects, "expected at least one object"

    out = g.copy()
    moves: list[tuple[np.ndarray, int, int]] = []
    for cells in objects:
        adjacent_portals = set()
        for r, c in cells:
            for nr, nc in _neighbors(r, c, g.shape, connectivity=4):
                portal = portal_by_cell.get((nr, nc))
                if portal is not None:
                    adjacent_portals.add(id(portal))
        assert len(adjacent_portals) == 1, "each object must touch exactly one entrance portal"

        entrance = next(portal for portal in portals if id(portal) in adjacent_portals)
        partner_options = [p for p in portals_by_color[int(entrance["color"])] if p is not entrance]
        assert len(partner_options) == 1
        exit_portal = partner_options[0]

        patch, _bbox = _crop_patch(g, cells)
        mirrored = np.fliplr(patch)
        row0, col0 = _placement_origin(exit_portal, mirrored.shape, exit_portal["direction"])
        moves.append((mirrored, row0, col0))

        for r, c in cells:
            out[r, c] = BG

    for patch, row0, col0 in moves:
        _stamp_non_bg(out, patch, row0, col0)
    return out


# --------------- generator ---------------

GEN_HEIGHT_MIN = 14
GEN_HEIGHT_MAX = 24
GEN_WIDTH_MIN = 16
GEN_WIDTH_MAX = 24
GEN_PAIR_BAND = 7
GEN_PORTAL_LENGTH = 3
GEN_PORTAL_ROW_OFFSET = 2
GEN_MAX_PAIRS = 3
GEN_PORTAL_GAP_MIN = 8
GEN_PORTAL_COL_MIN = 3
GEN_OBJECT_SIZES = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
GEN_RETRIES = 400
GEN_OBJECT_ATTEMPTS = 80


def _arrow_patch(direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    missing = {
        (-1, -1): (0, 0),
        (-1, 1): (0, 1),
        (1, -1): (1, 0),
        (1, 1): (1, 1),
    }[(dr, dc)]
    patch = np.full((2, 2), AZURE, dtype=int)
    patch[missing] = BG
    return patch


def _arrow_origin(portal: dict) -> tuple[int, int]:
    dr, dc = portal["direction"]
    row0 = int(portal["top"]) if dr < 0 else int(portal["bottom"]) - 1
    col0 = int(portal["col"]) - 2 if dc > 0 else int(portal["col"]) + 1
    return row0, col0


def _rect_cells(row0: int, col0: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    return {(row0 + r, col0 + c) for r in range(h) for c in range(w)}


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, row0: int, col0: int) -> None:
    h, w = patch.shape
    grid[row0 : row0 + h, col0 : col0 + w] = patch


def _random_pattern(shape: tuple[int, int], colors: list[int]) -> np.ndarray:
    h, w = shape
    max_colors = min(len(colors), h * w, 4)
    for _ in range(100):
        count = random.randint(2, max_colors)
        palette = random.sample(colors, count)
        patch = np.array(
            [[random.choice(palette) for _c in range(w)] for _r in range(h)],
            dtype=int,
        )
        if len(set(map(int, patch.ravel()))) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to build a non-symmetric object pattern")


def _sample_band_starts(height: int, count: int) -> list[int] | None:
    starts = list(range(height - GEN_PAIR_BAND + 1))
    for _ in range(100):
        choice = sorted(random.sample(starts, count))
        if all(choice[i + 1] - choice[i] >= GEN_PAIR_BAND for i in range(count - 1)):
            return choice
    return None


def _apply_random_flip(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    choice = random.randrange(4)
    if choice & 1:
        inp = np.flipud(inp).copy()
        out = np.flipud(out).copy()
    if choice & 2:
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()
    return inp, out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GEN_HEIGHT_MIN, GEN_HEIGHT_MAX)
    width = random.randint(GEN_WIDTH_MIN, GEN_WIDTH_MAX)
    max_pairs = min(GEN_MAX_PAIRS, height // GEN_PAIR_BAND)
    if max_pairs == 0:
        return None

    pair_count = random.randint(1, max_pairs)
    band_starts = _sample_band_starts(height, pair_count)
    if band_starts is None:
        return None

    if width < GEN_PORTAL_COL_MIN + GEN_PORTAL_GAP_MIN + 4:
        return None

    pair_colors = random.sample(GEN_PALETTE, pair_count)
    static = np.full((height, width), BG, dtype=int)
    pair_specs = []
    static_cells: set[tuple[int, int]] = set()

    for band_top, color in zip(band_starts, pair_colors):
        left_col_max = width - GEN_PORTAL_GAP_MIN - 5
        if left_col_max < GEN_PORTAL_COL_MIN:
            return None
        left_col = random.randint(GEN_PORTAL_COL_MIN, left_col_max)
        right_col = random.randint(left_col + GEN_PORTAL_GAP_MIN, width - 4)
        top = band_top + GEN_PORTAL_ROW_OFFSET
        bottom = top + GEN_PORTAL_LENGTH - 1

        portals = []
        for col in (left_col, right_col):
            direction = random.choice(GEN_DIRECTIONS)
            portal = {
                "color": color,
                "col": col,
                "top": top,
                "bottom": bottom,
                "direction": direction,
            }
            portals.append(portal)

            for r in range(top, bottom + 1):
                static[r, col] = color
                static_cells.add((r, col))

            arrow = _arrow_patch(direction)
            ar0, ac0 = _arrow_origin(portal)
            _stamp_patch(static, arrow, ar0, ac0)
            for r, c in _rect_cells(ar0, ac0, arrow.shape):
                if int(arrow[r - ar0, c - ac0]) != BG:
                    static_cells.add((r, c))

        pair_specs.append(portals)

    object_count = random.randint(1, pair_count)
    used_pairs = random.sample(range(pair_count), object_count)

    inp = static.copy()
    out = static.copy()
    occupied_in = set(static_cells)
    occupied_out = set(static_cells)

    object_color_pool = [c for c in GEN_PALETTE if c not in pair_colors]
    for pair_idx in used_pairs:
        portals = pair_specs[pair_idx]
        placed = False
        for _ in range(GEN_OBJECT_ATTEMPTS):
            entrance_idx = random.randrange(2)
            exit_idx = 1 - entrance_idx
            shape = random.choice(GEN_OBJECT_SIZES)
            patch = _random_pattern(shape, object_color_pool)
            src_dir = (-portals[entrance_idx]["direction"][0], -portals[entrance_idx]["direction"][1])
            dst_patch = np.fliplr(patch)
            src_r0, src_c0 = _placement_origin(portals[entrance_idx], patch.shape, src_dir)
            dst_r0, dst_c0 = _placement_origin(portals[exit_idx], dst_patch.shape, portals[exit_idx]["direction"])

            src_cells = _rect_cells(src_r0, src_c0, patch.shape)
            dst_cells = _rect_cells(dst_r0, dst_c0, dst_patch.shape)
            if not src_cells or not dst_cells:
                continue
            if min(r for r, _c in src_cells) < 0 or max(r for r, _c in src_cells) >= height:
                continue
            if min(c for _r, c in src_cells) < 0 or max(c for _r, c in src_cells) >= width:
                continue
            if min(r for r, _c in dst_cells) < 0 or max(r for r, _c in dst_cells) >= height:
                continue
            if min(c for _r, c in dst_cells) < 0 or max(c for _r, c in dst_cells) >= width:
                continue
            if src_cells & occupied_in:
                continue
            if dst_cells & occupied_out:
                continue
            if dst_cells & occupied_in:
                continue

            _stamp_patch(inp, patch, src_r0, src_c0)
            _stamp_patch(out, dst_patch, dst_r0, dst_c0)
            occupied_in |= src_cells
            occupied_out |= dst_cells
            placed = True
            break
        if not placed:
            return None

    inp, out = _apply_random_flip(inp, out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid portal sample")
