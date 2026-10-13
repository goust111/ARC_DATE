from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
AZURE = 8
MAROON = 9

ROLE_COLORS = {GREY, AZURE, MAROON}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DELTA = {
    0: (-1, 0),  # top
    1: (0, 1),   # right
    2: (1, 0),   # bottom
    3: (0, -1),  # left
}


def _most_common_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.reshape(-1)))
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        coords = []
        while queue:
            cr, cc = queue.popleft()
            coords.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _midpoint_offset(length: int) -> int:
    return int((length - 1) // 2)


def _inside(box: tuple[int, int, int, int], pt: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = pt
    return r0 <= r <= r1 and c0 <= c <= c1


def _find_divider(g: np.ndarray) -> tuple[str, int]:
    h, w = g.shape
    row_candidates = [
        r
        for r in range(h)
        if np.all(np.isin(g[r, :], [GREY, AZURE])) and int(np.sum(g[r, :] == GREY)) >= max(1, w // 2)
    ]
    col_candidates = [
        c
        for c in range(w)
        if np.all(np.isin(g[:, c], [GREY, AZURE])) and int(np.sum(g[:, c] == GREY)) >= max(1, h // 2)
    ]
    if row_candidates and not col_candidates:
        return ("h", int(row_candidates[0]))
    if col_candidates and not row_candidates:
        return ("v", int(col_candidates[0]))
    if len(row_candidates) == 1 and len(col_candidates) == 0:
        return ("h", int(row_candidates[0]))
    if len(col_candidates) == 1 and len(row_candidates) == 0:
        return ("v", int(col_candidates[0]))
    raise AssertionError("expected exactly one divider line")


def _tile_side(marker: tuple[int, int], box: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = box
    mr, mc = marker
    if mr == r0 and c0 <= mc <= c1:
        return 0
    if mc == c1 and r0 <= mr <= r1:
        return 1
    if mr == r1 and c0 <= mc <= c1:
        return 2
    if mc == c0 and r0 <= mr <= r1:
        return 3
    return None


def _cell_side(marker_rel: tuple[int, int], box_rel: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, c1 = box_rel
    mr, mc = marker_rel
    if mr == r0 - 1 and c0 <= mc <= c1:
        return 0
    if mc == c1 + 1 and r0 <= mr <= r1:
        return 1
    if mr == r1 + 1 and c0 <= mc <= c1:
        return 2
    if mc == c0 - 1 and r0 <= mr <= r1:
        return 3
    return None


def _rotate_dims(h: int, w: int, k: int) -> tuple[int, int]:
    return (h, w) if k % 2 == 0 else (w, h)


def _frame_data(g: np.ndarray, divider_axis: str, divider_idx: int) -> dict:
    grey_mask = g == GREY
    if divider_axis == "h":
        grey_mask[divider_idx, :] = False
    else:
        grey_mask[:, divider_idx] = False
    grey_comps = _components(grey_mask)
    assert grey_comps, "expected a grey frame component"
    frame_coords = max(grey_comps, key=lambda comp: int(comp.shape[0]))
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)
    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    wall_mask = (sub == GREY) | (sub == MAROON)
    cell_comps = _components(~wall_mask)
    frame_markers_rel = [tuple(map(int, pos)) for pos in np.argwhere(sub == MAROON)]
    cells = []
    marker_to_cell = {}
    for idx, comp in enumerate(cell_comps):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(comp.shape[0]) == h * w, "cells must be solid rectangles"
        matches = []
        for marker_rel in frame_markers_rel:
            side = _cell_side(marker_rel, (r0, c0, r1, c1))
            if side is not None:
                matches.append((marker_rel, side))
        assert len(matches) == 1, "each frame cell must have exactly one marker"
        marker_rel, side = matches[0]
        cell = {
            "id": idx,
            "bbox_rel": (r0, c0, r1, c1),
            "shape": (h, w),
            "marker_rel": marker_rel,
            "side": int(side),
        }
        cells.append(cell)
        marker_abs = (fr0 + marker_rel[0], fc0 + marker_rel[1])
        marker_to_cell[marker_abs] = cell
    return {
        "bbox": (fr0, fc0, fr1, fc1),
        "subgrid": sub,
        "cells": cells,
        "marker_to_cell": marker_to_cell,
    }


def _tile_data(g: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> dict:
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in ROLE_COLORS | {bg})
    tile_markers = [
        tuple(map(int, pos))
        for pos in np.argwhere(g == MAROON)
        if not _inside(frame_bbox, tuple(map(int, pos)))
    ]
    tiles = []
    marker_to_tile = {}
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 1, "each tile color must be unique"
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(comp.shape[0]) == h * w - 1, "tile body must be a rectangle with one marker cutout"
        matches = []
        for marker in tile_markers:
            side = _tile_side(marker, (r0, c0, r1, c1))
            if side is not None:
                matches.append((marker, side))
        assert len(matches) == 1, "each tile must have exactly one marker"
        marker, side = matches[0]
        tile = {
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "shape": (h, w),
            "marker": marker,
            "side": int(side),
        }
        tiles.append(tile)
        marker_to_tile[marker] = tile
    return {"tiles": tiles, "marker_to_tile": marker_to_tile}


def _line_pairs(g: np.ndarray, marker_to_tile: dict, marker_to_cell: dict) -> list[tuple[dict, dict]]:
    azure_comps = _components(g == AZURE)
    pairs = []
    used_tile_markers = set()
    used_cell_markers = set()
    for comp in azure_comps:
        adjacent = set()
        for r, c in comp:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == MAROON:
                    adjacent.add((nr, nc))
        assert len(adjacent) == 2, "each line must connect exactly two markers"
        tile_markers = [marker for marker in adjacent if marker in marker_to_tile]
        cell_markers = [marker for marker in adjacent if marker in marker_to_cell]
        assert len(tile_markers) == 1 and len(cell_markers) == 1
        tile_marker = tile_markers[0]
        cell_marker = cell_markers[0]
        used_tile_markers.add(tile_marker)
        used_cell_markers.add(cell_marker)
        pairs.append((marker_to_tile[tile_marker], marker_to_cell[cell_marker]))
    assert len(used_tile_markers) == len(marker_to_tile)
    assert len(used_cell_markers) == len(marker_to_cell)
    return pairs


def _rotation_needed(tile: dict, cell: dict) -> int:
    tile_h, tile_w = tile["shape"]
    cell_h, cell_w = cell["shape"]
    candidates = [
        k
        for k in range(4)
        if _rotate_dims(tile_h, tile_w, k) == (cell_h, cell_w)
        and (tile["side"] + k) % 4 == cell["side"]
    ]
    assert len(candidates) == 1, "tile orientation must be uniquely determined"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored tile to a frame cell by tracing the azure line, then rotate
    the tile so its maroon-marked edge faces the cell wall marked in maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    divider_axis, divider_idx = _find_divider(g)
    frame = _frame_data(g, divider_axis, divider_idx)
    tiles = _tile_data(g, bg, frame["bbox"])
    pairs = _line_pairs(g, tiles["marker_to_tile"], frame["marker_to_cell"])

    out = np.full(frame["subgrid"].shape, bg, dtype=int)
    out[frame["subgrid"] == GREY] = GREY
    for tile, cell in pairs:
        _rotation_needed(tile, cell)
        r0, c0, r1, c1 = cell["bbox_rel"]
        out[r0 : r1 + 1, c0 : c1 + 1] = tile["color"]
        mr, mc = cell["marker_rel"]
        out[mr, mc] = tile["color"]
    return out


# --------------- generator ---------------

PALETTE = [0, 1, 2, 3, 4, 6, 7]
GRID_MIN = 22
GRID_MAX = 30
CELL_COUNT_RANGE = (2, 5)
CELL_SPAN_RANGE = (2, 5)
MIN_FRAME_MARGIN = 2
MIN_TILE_BAND = 7
MAX_GENERATION_ATTEMPTS = 400


def _build_layout(n_cells: int) -> tuple[str, int, int, list[dict]]:
    strip = random.choice(["row", "col"])
    cells = []
    if strip == "row":
        cell_h = random.randint(CELL_SPAN_RANGE[0], CELL_SPAN_RANGE[1] - 1)
        width_choices = [d for d in range(*CELL_SPAN_RANGE) if d != cell_h]
        widths = [random.choice(width_choices) for _ in range(n_cells)]
        frame_h = cell_h + 2
        frame_w = sum(widths) + n_cells + 1
        cursor = 1
        for idx, cell_w in enumerate(widths):
            r0 = 1
            c0 = cursor
            r1 = r0 + cell_h - 1
            c1 = c0 + cell_w - 1
            sides = [0, 2]
            if idx == 0:
                sides.append(3)
            if idx == n_cells - 1:
                sides.append(1)
            cells.append({"bbox_rel": (r0, c0, r1, c1), "side": random.choice(sides)})
            cursor = c1 + 2
    else:
        cell_w = random.randint(CELL_SPAN_RANGE[0], CELL_SPAN_RANGE[1] - 1)
        height_choices = [d for d in range(*CELL_SPAN_RANGE) if d != cell_w]
        heights = [random.choice(height_choices) for _ in range(n_cells)]
        frame_h = sum(heights) + n_cells + 1
        frame_w = cell_w + 2
        cursor = 1
        for idx, cell_h in enumerate(heights):
            r0 = cursor
            c0 = 1
            r1 = r0 + cell_h - 1
            c1 = c0 + cell_w - 1
            sides = [1, 3]
            if idx == 0:
                sides.append(0)
            if idx == n_cells - 1:
                sides.append(2)
            cells.append({"bbox_rel": (r0, c0, r1, c1), "side": random.choice(sides)})
            cursor = r1 + 2
    return strip, frame_h, frame_w, cells


def _marker_on_cell(cell_box: tuple[int, int, int, int], side: int) -> tuple[int, int]:
    r0, c0, r1, c1 = cell_box
    if side == 0:
        return (r0 - 1, c0 + _midpoint_offset(c1 - c0 + 1))
    if side == 1:
        return (r0 + _midpoint_offset(r1 - r0 + 1), c1 + 1)
    if side == 2:
        return (r1 + 1, c0 + _midpoint_offset(c1 - c0 + 1))
    return (r0 + _midpoint_offset(r1 - r0 + 1), c0 - 1)


def _marker_on_tile(tile_box: tuple[int, int, int, int], side: int) -> tuple[int, int]:
    r0, c0, r1, c1 = tile_box
    if side == 0:
        return (r0, c0 + _midpoint_offset(c1 - c0 + 1))
    if side == 1:
        return (r0 + _midpoint_offset(r1 - r0 + 1), c1)
    if side == 2:
        return (r1, c0 + _midpoint_offset(c1 - c0 + 1))
    return (r0 + _midpoint_offset(r1 - r0 + 1), c0)


def _choose_tile_specs(cells: list[dict], colors: list[int]) -> list[dict]:
    specs = []
    for cell, color in zip(cells, colors):
        r0, c0, r1, c1 = cell["bbox_rel"]
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        if random.randrange(2):
            k = random.choice([1, 3])
            tile_h, tile_w = cell_w, cell_h
        else:
            k = random.choice([0, 2])
            tile_h, tile_w = cell_h, cell_w
        tile_side = (cell["side"] - k) % 4
        specs.append(
            {
                "color": int(color),
                "shape": (int(tile_h), int(tile_w)),
                "side": int(tile_side),
                "rotation": int(k),
            }
        )
    return specs


def _choose_canvas(frame_h: int, frame_w: int) -> tuple[int, int, str, bool, int]:
    for _ in range(200):
        divider_axis = random.choice(["h", "v"])
        frame_on_positive_side = bool(random.randrange(2))
        if divider_axis == "v":
            min_w = MIN_TILE_BAND + frame_w + 2 * MIN_FRAME_MARGIN + 1
            min_h = max(GRID_MIN, frame_h + 2 * MIN_FRAME_MARGIN)
            if min_w > GRID_MAX or min_h > GRID_MAX:
                continue
            h = random.randint(min_h, GRID_MAX)
            w = random.randint(min_w, GRID_MAX)
            if frame_on_positive_side:
                divider = random.randint(MIN_TILE_BAND, w - (frame_w + 2 * MIN_FRAME_MARGIN) - 1)
            else:
                divider = random.randint(frame_w + 2 * MIN_FRAME_MARGIN, w - MIN_TILE_BAND - 1)
            return h, w, divider_axis, frame_on_positive_side, divider
        min_h = MIN_TILE_BAND + frame_h + 2 * MIN_FRAME_MARGIN + 1
        min_w = max(GRID_MIN, frame_w + 2 * MIN_FRAME_MARGIN)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue
        h = random.randint(min_h, GRID_MAX)
        w = random.randint(min_w, GRID_MAX)
        if frame_on_positive_side:
            divider = random.randint(MIN_TILE_BAND, h - (frame_h + 2 * MIN_FRAME_MARGIN) - 1)
        else:
            divider = random.randint(frame_h + 2 * MIN_FRAME_MARGIN, h - MIN_TILE_BAND - 1)
        return h, w, divider_axis, frame_on_positive_side, divider
    raise RuntimeError("failed to size canvas")


def _frame_origin(
    h: int,
    w: int,
    frame_h: int,
    frame_w: int,
    divider_axis: str,
    divider: int,
    frame_on_positive_side: bool,
) -> tuple[int, int]:
    if divider_axis == "v":
        if frame_on_positive_side:
            col_lo = divider + 1 + MIN_FRAME_MARGIN
            col_hi = w - MIN_FRAME_MARGIN - frame_w
        else:
            col_lo = MIN_FRAME_MARGIN
            col_hi = divider - MIN_FRAME_MARGIN - frame_w
        row_lo = MIN_FRAME_MARGIN
        row_hi = h - MIN_FRAME_MARGIN - frame_h
    else:
        if frame_on_positive_side:
            row_lo = divider + 1 + MIN_FRAME_MARGIN
            row_hi = h - MIN_FRAME_MARGIN - frame_h
        else:
            row_lo = MIN_FRAME_MARGIN
            row_hi = divider - MIN_FRAME_MARGIN - frame_h
        col_lo = MIN_FRAME_MARGIN
        col_hi = w - MIN_FRAME_MARGIN - frame_w
    assert row_lo <= row_hi and col_lo <= col_hi
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _tile_region(
    h: int,
    w: int,
    divider_axis: str,
    divider: int,
    frame_on_positive_side: bool,
) -> tuple[int, int, int, int]:
    if divider_axis == "v":
        if frame_on_positive_side:
            return (0, 0, h - 1, divider - 1)
        return (0, divider + 1, h - 1, w - 1)
    if frame_on_positive_side:
        return (0, 0, divider - 1, w - 1)
    return (divider + 1, 0, h - 1, w - 1)


def _place_tiles(
    region: tuple[int, int, int, int],
    tile_specs: list[dict],
) -> list[dict] | None:
    rr0, rc0, rr1, rc1 = region
    occupied = np.zeros((rr1 - rr0 + 1, rc1 - rc0 + 1), dtype=bool)
    placed: list[dict] = []
    order = sorted(range(len(tile_specs)), key=lambda idx: tile_specs[idx]["shape"][0] * tile_specs[idx]["shape"][1], reverse=True)
    result: list[dict | None] = [None] * len(tile_specs)
    for idx in order:
        tile = tile_specs[idx]
        th, tw = tile["shape"]
        row_min = rr0 + 1
        row_max = rr1 - th
        col_min = rc0 + 1
        col_max = rc1 - tw
        if row_min > row_max or col_min > col_max:
            return None
        for _ in range(300):
            top = random.randint(row_min, row_max)
            left = random.randint(col_min, col_max)
            lr0 = top - rr0
            lc0 = left - rc0
            er0 = lr0 - 1
            ec0 = lc0 - 1
            er1 = lr0 + th
            ec1 = lc0 + tw
            if er0 < 0 or ec0 < 0 or er1 >= occupied.shape[0] or ec1 >= occupied.shape[1]:
                continue
            if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            box = (top, left, top + th - 1, left + tw - 1)
            marker = _marker_on_tile(box, tile["side"])
            start = (marker[0] + SIDE_TO_DELTA[tile["side"]][0], marker[1] + SIDE_TO_DELTA[tile["side"]][1])
            placed_tile = dict(tile)
            placed_tile["bbox"] = box
            placed_tile["marker"] = marker
            placed_tile["line_end"] = start
            result[idx] = placed_tile
            placed.append(placed_tile)
            break
        else:
            return None
    return [tile for tile in result if tile is not None]


def _bfs_path(free: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    if start == end:
        return [start]
    h, w = free.shape
    queue = deque([start])
    prev = {start: None}
    while queue:
        r, c = queue.popleft()
        neighbors = []
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and (nr, nc) not in prev:
                neighbors.append((nr, nc))
        neighbors.sort(key=lambda pt: abs(pt[0] - end[0]) + abs(pt[1] - end[1]))
        for nxt in neighbors:
            prev[nxt] = (r, c)
            if nxt == end:
                path = [end]
                cur = end
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            queue.append(nxt)
    return None


def _route_lines(
    grid: np.ndarray,
    bg: int,
    divider_axis: str,
    divider: int,
    pairs: list[tuple[dict, dict]],
) -> np.ndarray | None:
    h, w = grid.shape
    walkable = grid == bg
    if divider_axis == "h":
        walkable[divider, :] |= grid[divider, :] == GREY
    else:
        walkable[:, divider] |= grid[:, divider] == GREY

    marker_neighbors = np.zeros((h, w), dtype=bool)
    endpoint_set = set()
    for tile, cell in pairs:
        endpoint_set.add(tile["line_end"])
        endpoint_set.add(cell["line_end"])
        for marker in [tile["marker"], cell["marker"]]:
            mr, mc = marker
            for dr, dc in DIR4:
                nr = mr + dr
                nc = mc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    marker_neighbors[nr, nc] = True

    order = pairs[:]
    random.shuffle(order)
    used = np.zeros((h, w), dtype=bool)
    out = grid.copy()
    for tile, cell in sorted(order, key=lambda pair: abs(pair[0]["line_end"][0] - pair[1]["line_end"][0]) + abs(pair[0]["line_end"][1] - pair[1]["line_end"][1]), reverse=True):
        start = tile["line_end"]
        end = cell["line_end"]
        allow = np.zeros((h, w), dtype=bool)
        allow[start] = True
        allow[end] = True
        free = walkable & ~used & ~(marker_neighbors & ~allow)
        free[start] = True
        free[end] = True
        path = _bfs_path(free, start, end)
        if path is None:
            return None
        for r, c in path:
            used[r, c] = True
            out[r, c] = AZURE
    return out


def _expected_output(frame_h: int, frame_w: int, cells: list[dict], tile_specs: list[dict]) -> np.ndarray:
    out = np.full((frame_h, frame_w), GREY, dtype=int)
    for cell, tile in zip(cells, tile_specs):
        r0, c0, r1, c1 = cell["bbox_rel"]
        out[r0 : r1 + 1, c0 : c1 + 1] = tile["color"]
        mr, mc = cell["marker_rel"]
        out[mr, mc] = tile["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    n_cells = random.randint(CELL_COUNT_RANGE[0], CELL_COUNT_RANGE[1] - 1)
    color_pool = [color for color in PALETTE if color != bg]
    if n_cells > len(color_pool):
        return None
    _strip, frame_h, frame_w, cells = _build_layout(n_cells)
    if len(cells) != n_cells:
        return None
    chosen_colors = random.sample(color_pool, n_cells)
    tile_specs = _choose_tile_specs(cells, chosen_colors)
    h, w, divider_axis, frame_on_positive_side, divider = _choose_canvas(frame_h, frame_w)
    frame_top, frame_left = _frame_origin(h, w, frame_h, frame_w, divider_axis, divider, frame_on_positive_side)
    tile_region = _tile_region(h, w, divider_axis, divider, frame_on_positive_side)
    placed_tiles = _place_tiles(tile_region, tile_specs)
    if placed_tiles is None or len(placed_tiles) != n_cells:
        return None

    grid = np.full((h, w), bg, dtype=int)
    if divider_axis == "h":
        grid[divider, :] = GREY
    else:
        grid[:, divider] = GREY

    frame = np.full((frame_h, frame_w), GREY, dtype=int)
    placed_cells = []
    for cell in cells:
        r0, c0, r1, c1 = cell["bbox_rel"]
        frame[r0 : r1 + 1, c0 : c1 + 1] = bg
        abs_box = (frame_top + r0, frame_left + c0, frame_top + r1, frame_left + c1)
        marker_rel = _marker_on_cell((r0, c0, r1, c1), cell["side"])
        frame[marker_rel] = MAROON
        placed_cell = dict(cell)
        placed_cell["bbox"] = abs_box
        placed_cell["marker_rel"] = marker_rel
        placed_cell["marker"] = (frame_top + marker_rel[0], frame_left + marker_rel[1])
        delta = SIDE_TO_DELTA[cell["side"]]
        placed_cell["line_end"] = (placed_cell["marker"][0] + delta[0], placed_cell["marker"][1] + delta[1])
        placed_cells.append(placed_cell)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame

    for tile in placed_tiles:
        r0, c0, r1, c1 = tile["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = tile["color"]
        mr, mc = tile["marker"]
        grid[mr, mc] = MAROON

    pairs = list(zip(placed_tiles, placed_cells))
    routed = _route_lines(grid, bg, divider_axis, divider, pairs)
    if routed is None:
        return None
    if _most_common_color(routed) != bg:
        return None

    expected = _expected_output(frame_h, frame_w, placed_cells, placed_tiles)
    try:
        actual = main(routed)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return routed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle 5 input")
