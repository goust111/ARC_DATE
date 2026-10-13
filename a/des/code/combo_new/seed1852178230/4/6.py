from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


Coord = tuple[int, int]


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _neighbors4(mask: np.ndarray, cell: Coord) -> list[Coord]:
    r, c = cell
    h, w = mask.shape
    out: list[Coord] = []
    if r > 0 and mask[r - 1, c]:
        out.append((r - 1, c))
    if r + 1 < h and mask[r + 1, c]:
        out.append((r + 1, c))
    if c > 0 and mask[r, c - 1]:
        out.append((r, c - 1))
    if c + 1 < w and mask[r, c + 1]:
        out.append((r, c + 1))
    return out


def _trace_path(start: Coord, neighbor_map: dict[Coord, list[Coord]]) -> list[Coord]:
    path = [start]
    prev: Coord | None = None
    curr = start
    seen = {start}

    while True:
        nbrs = neighbor_map[curr]
        if prev is None:
            assert len(nbrs) == 1, "a path must start from a degree-1 endpoint"
            nxt = nbrs[0]
        else:
            if len(nbrs) == 1:
                break
            if len(nbrs) == 2:
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
            else:
                assert len(nbrs) == 4, "only 4-way crossings are allowed as junctions"
                dr = curr[0] - prev[0]
                dc = curr[1] - prev[1]
                nxt = (curr[0] + dr, curr[1] + dc)
                assert nxt in nbrs, "a crossing must continue straight for each original path"

        assert nxt not in seen, "paths must be simple chains"
        path.append(nxt)
        seen.add(nxt)
        prev, curr = curr, nxt

    return path


def _extract_paths(grid: np.ndarray) -> tuple[int, dict[Coord, list[Coord]], list[list[Coord]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    mask = g != bg
    if not np.any(mask):
        return bg, {}, [], []

    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    neighbor_map = {cell: _neighbors4(mask, cell) for cell in occupied}
    endpoints = sorted(cell for cell in occupied if len(neighbor_map[cell]) == 1)
    assert len(endpoints) % 2 == 0 and endpoints, "expected open paths with paired endpoints"

    endpoint_set = set(endpoints)
    used_endpoints: set[Coord] = set()
    paths: list[list[Coord]] = []
    colors: list[int] = []

    for start in endpoints:
        if start in used_endpoints:
            continue
        coords = _trace_path(start, neighbor_map)
        end = coords[-1]
        assert end in endpoint_set and end != start, "each traced path must end at a distinct endpoint"
        used_endpoints.add(start)
        used_endpoints.add(end)

        if end < start:
            coords.reverse()

        color = int(g[coords[0]])
        assert color != bg
        for cell in coords:
            if len(neighbor_map[cell]) != 4:
                assert int(g[cell]) == color, "non-crossing pixels must keep the path color"

        paths.append(coords)
        colors.append(color)

    coverage: dict[Coord, int] = defaultdict(int)
    for coords in paths:
        for cell in coords:
            coverage[cell] += 1

    for cell in occupied:
        deg = len(neighbor_map[cell])
        assert deg in (1, 2, 4), "only endpoints, simple path pixels, and 4-way crossings are valid"
        expected = 2 if deg == 4 else 1
        assert coverage[cell] == expected, "every occupied cell must belong to the expected number of paths"

    return bg, neighbor_map, paths, colors


def _recolor_paths(
    shape: tuple[int, int],
    bg: int,
    paths: list[list[Coord]],
    original_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    if not paths:
        return out

    members: dict[Coord, list[int]] = defaultdict(list)
    for path_id, coords in enumerate(paths):
        for cell in coords:
            members[cell].append(path_id)

    crossings = {cell: ids for cell, ids in members.items() if len(ids) == 2}
    assert all(len(ids) <= 2 for ids in members.values())

    crossing_order: list[list[Coord]] = []
    crossing_cuts: list[list[int]] = []
    coord_index: list[dict[Coord, int]] = []
    for coords in paths:
        index = {cell: i for i, cell in enumerate(coords)}
        ordered = [cell for cell in coords if cell in crossings]
        assert len(ordered) % 2 == 0, "every path must meet an even number of crossings"
        coord_index.append(index)
        crossing_order.append(ordered)
        crossing_cuts.append([index[cell] for cell in ordered])

    current_colors = [int(color) for color in original_colors]
    next_crossing = [0 for _ in paths]
    downstream_colors: list[list[int]] = [[] for _ in paths]
    crossing_colors: dict[Coord, int] = {}
    processed: set[Coord] = set()

    while len(processed) < len(crossings):
        progress = False
        for path_id, ordered in enumerate(crossing_order):
            idx = next_crossing[path_id]
            if idx >= len(ordered):
                continue

            cell = ordered[idx]
            if cell in processed:
                continue

            a, b = crossings[cell]
            other = b if a == path_id else a
            other_idx = next_crossing[other]
            if other_idx >= len(crossing_order[other]):
                continue
            if crossing_order[other][other_idx] != cell:
                continue

            cut_a = coord_index[path_id][cell]
            cut_b = coord_index[other][cell]
            assert cut_a > 0 and cut_b > 0
            prev_a = paths[path_id][cut_a - 1]
            prev_b = paths[other][cut_b - 1]

            incoming_a = current_colors[path_id]
            incoming_b = current_colors[other]
            crossing_colors[cell] = incoming_a if prev_a < prev_b else incoming_b

            current_colors[path_id], current_colors[other] = incoming_b, incoming_a
            downstream_colors[path_id].append(current_colors[path_id])
            downstream_colors[other].append(current_colors[other])
            next_crossing[path_id] += 1
            next_crossing[other] += 1
            processed.add(cell)
            progress = True

        assert progress, "crossing order is cyclic or ambiguous"

    for path_id, coords in enumerate(paths):
        cuts = crossing_cuts[path_id]
        if not cuts:
            for r, c in coords:
                out[r, c] = int(original_colors[path_id])
            continue

        for r, c in coords[: cuts[0]]:
            out[r, c] = int(original_colors[path_id])

        for idx, cut in enumerate(cuts):
            color = int(downstream_colors[path_id][idx])
            next_cut = cuts[idx + 1] if idx + 1 < len(cuts) else len(coords)
            for r, c in coords[cut + 1 : next_cut]:
                out[r, c] = color

    for (r, c), color in crossing_colors.items():
        out[r, c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint orthogonal paths by swapping the carried colors at grey crossing switches.

    Input structure:
    - The background is black.
    - Foreground objects are one-pixel-wide 4-connected orthogonal paths with no branching outside crossings.
    - A crossing is a shared pixel where exactly two paths overlap; that pixel is shown in grey and has four neighbors.
    - Each path has one visible color away from crossings and meets an even number of crossings.

    Rule:
    - Recover every original path by tracing between its two endpoints, continuing straight through each grey 4-way crossing.
    - Orient each path from its lexicographically earlier endpoint (topmost, then leftmost) and carry its own color forward from that end.
    - Whenever two active path traces reach the same crossing, they swap carried colors, so each downstream segment takes the other path's current color.
    - Repaint each crossing pixel with the current color of the incoming branch whose previous pixel is in the smaller row; break ties by the smaller column.

    Key insight:
    - The grey pixel is not a final color mark but a switch: first recover the hidden straight-through paths, then propagate colors along those paths while exchanging the active colors at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    bg, _neighbor_map, paths, colors = _extract_paths(g)
    return _recolor_paths(g.shape, bg, paths, colors)


def _polyline(points: list[Coord]) -> list[Coord]:
    coords = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        assert r0 == r1 or c0 == c1
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        r, c = r0, c0
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            coords.append((r, c))
    assert len(coords) == len(set(coords)), "generated paths must be simple"
    return coords


def _sample_spaced(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_row_intervals(h: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(300):
        intervals: list[tuple[int, int]] = []
        ok = True
        for _ in range(count):
            top = random.randint(1, h - 5)
            bottom = random.randint(top + 2, min(h - 2, top + 5))
            if any(not (bottom + 2 < a or top > b + 2) for a, b in intervals):
                ok = False
                break
            intervals.append((top, bottom))
        if ok:
            return sorted(intervals)
    return None


def _left_turn_col(trunks: list[int], w: int, cross_count: int) -> int | None:
    if cross_count == len(trunks):
        lo, hi = trunks[-1] + 1, w - 3
    else:
        lo, hi = trunks[cross_count - 1] + 1, trunks[cross_count] - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _right_turn_col(trunks: list[int], cross_count: int) -> int | None:
    first_idx = len(trunks) - cross_count
    if first_idx == 0:
        lo, hi = 2, trunks[0] - 1
    else:
        lo, hi = trunks[first_idx - 1] + 1, trunks[first_idx] - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _safe_cells(cells: list[Coord], occupied: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in occupied:
                return False
    return True


def _maybe_isolated_path(shape: tuple[int, int], occupied: set[Coord]) -> list[Coord] | None:
    h, w = shape
    variants: list[list[Coord]] = []
    for col in range(1, min(4, w - 2)):
        for dr in range(2, min(5, h - 2)):
            for dc in range(2, min(5, w - col)):
                variants.append(_polyline([(0, col), (dr, col), (dr, col + dc)]))
    for col in range(max(1, w - 4), w - 1):
        for dr in range(2, min(5, h - 2)):
            for dc in range(2, min(5, col + 1)):
                variants.append(_polyline([(0, col), (dr, col), (dr, col - dc)]))
    for col in range(1, min(4, w - 2)):
        for dr in range(2, min(5, h - 2)):
            for dc in range(2, min(5, w - col)):
                variants.append(_polyline([(h - 1, col), (h - 1 - dr, col), (h - 1 - dr, col + dc)]))
    for col in range(max(1, w - 4), w - 1):
        for dr in range(2, min(5, h - 2)):
            for dc in range(2, min(5, col + 1)):
                variants.append(_polyline([(h - 1, col), (h - 1 - dr, col), (h - 1 - dr, col - dc)]))

    random.shuffle(variants)
    for coords in variants:
        if _safe_cells(coords, occupied, shape):
            return coords
    return None


def _build_grid(shape: tuple[int, int], paths: list[list[Coord]], colors: list[int]) -> np.ndarray | None:
    h, w = shape
    cell_to_paths: dict[Coord, list[int]] = defaultdict(list)
    for path_id, coords in enumerate(paths):
        for cell in coords:
            cell_to_paths[cell].append(path_id)

    if any(len(ids) > 2 for ids in cell_to_paths.values()):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for cell, ids in cell_to_paths.items():
        if len(ids) == 1:
            grid[cell] = int(colors[ids[0]])
        else:
            grid[cell] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    n_trunks = random.choice([1, 2])
    n_u_paths = random.choice([1, 2])

    trunks = _sample_spaced(n_trunks, 3, w - 4, 4)
    intervals = _sample_row_intervals(h, n_u_paths)
    if trunks is None or intervals is None:
        return None

    paths: list[list[Coord]] = []
    for col in trunks:
        paths.append(_polyline([(0, col), (h - 1, col)]))

    for top, bottom in intervals:
        side = random.choice(["left", "right"])
        cross_count = random.randint(1, n_trunks)
        if side == "left":
            turn = _left_turn_col(trunks, w, cross_count)
            if turn is None:
                return None
            coords = _polyline([(top, 0), (top, turn), (bottom, turn), (bottom, 0)])
        else:
            turn = _right_turn_col(trunks, cross_count)
            if turn is None:
                return None
            coords = _polyline([(top, w - 1), (top, turn), (bottom, turn), (bottom, w - 1)])
        paths.append(coords)

    occupied = {cell for coords in paths for cell in coords}
    if random.random() < 0.5:
        isolated = _maybe_isolated_path((h, w), occupied)
        if isolated is not None:
            paths.append(isolated)

    colors = random.sample(COLOR_POOL, len(paths))
    grid = _build_grid((h, w), paths, colors)
    if grid is None:
        return None

    rotation = random.choice([0, 1, 2, 3])
    if rotation:
        grid = np.rot90(grid, rotation).copy()

    try:
        bg, neighbor_map, extracted_paths, extracted_colors = _extract_paths(grid)
        if len(extracted_paths) != len(paths):
            return None
        if len(set(extracted_colors)) != len(extracted_colors):
            return None
        crossing_counts = [sum(1 for cell in coords if len(neighbor_map[cell]) == 4) for coords in extracted_paths]
        if any(count % 2 != 0 for count in crossing_counts):
            return None
        if not any(count > 0 for count in crossing_counts):
            return None
        out = _recolor_paths(grid.shape, bg, extracted_paths, extracted_colors)
        if np.any(out == GREY):
            return None
        if not np.any(out != grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-switch input")
