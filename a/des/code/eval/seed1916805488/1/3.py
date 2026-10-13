from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _edge_key(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int, int]:
    r, c = cell
    h, w = shape
    return (min(r, c, h - 1 - r, w - 1 - c), r, c)


def _find_filter_blocks(grid: np.ndarray) -> list[dict]:
    filters: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        comps = _components(grid == color)
        for pts in comps:
            if len(pts) != 4:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if (r1 - r0, c1 - c0) != (1, 1):
                continue
            cells = {(int(r), int(c)) for r, c in pts}
            if cells != {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}:
                continue
            filters.append({"color": color, "cells": cells})
    return filters


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    filters = _find_filter_blocks(g)
    if not filters:
        return out

    component_info: dict[tuple[int, int], tuple[int, int]] = {}
    comps_by_color: dict[int, list[set[tuple[int, int]]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp_idx, pts in enumerate(_components(g == color)):
            cell_set = {(int(r), int(c)) for r, c in pts}
            comps_by_color[color].append(cell_set)
            for cell in cell_set:
                component_info[cell] = (color, comp_idx)

    for filt in filters:
        filt_cells = filt["cells"]
        touched_ids: set[tuple[int, int]] = set()
        touched_colors: set[int] = set()
        for r, c in filt_cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(g.shape, nr, nc) and (nr, nc) in component_info:
                    comp_id = component_info[(nr, nc)]
                    if comp_id[0] == int(filt["color"]):
                        continue
                    touched_ids.add(comp_id)
                    touched_colors.add(comp_id[0])

        assert len(touched_ids) == 2, "each filter must touch exactly two path segments"
        assert len(touched_colors) == 1, "the two path segments around one filter must share a color"

        color = next(iter(touched_colors))
        comp_ids = sorted(touched_ids)
        segments = [comps_by_color[color][comp_idx] for _comp_color, comp_idx in comp_ids]

        real_endpoints: list[tuple[int, int]] = []
        for seg in segments:
            endpoints: list[tuple[int, int]] = []
            real: list[tuple[int, int]] = []
            for r, c in seg:
                degree = sum((nr, nc) in seg for nr, nc in _neighbors4(r, c))
                if degree == 1:
                    endpoints.append((r, c))
                    if all((nr, nc) not in filt_cells for nr, nc in _neighbors4(r, c)):
                        real.append((r, c))
            assert len(endpoints) == 2, "each path segment must remain a simple open path"
            assert len(real) == 1, "each side of a filter must contribute one real endpoint"
            real_endpoints.append(real[0])

        start_seg_idx = 0 if _edge_key((h, w), real_endpoints[0]) < _edge_key((h, w), real_endpoints[1]) else 1
        recolor_seg = segments[1 - start_seg_idx]
        for r, c in recolor_seg:
            out[r, c] = int(filt["color"])

    return out


GRID_MIN = 14
GRID_MAX = 22
PATH_COUNT_RANGE = (1, 4)
SEGMENT_COUNT_RANGE = (3, 6)
SEGMENT_LEN_RANGE = (2, 5)
GEN_RETRIES = 400


def _dir_from_edge(edge: str) -> tuple[int, int]:
    return {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}[edge]


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(dc, dr), (-dc, -dr)]


def _edge_start(rng: random.Random, h: int, w: int, tie_mode: bool) -> tuple[tuple[int, int], str]:
    if tie_mode:
        edge = "top"
    else:
        edge = rng.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        return (0, rng.randint(2, w - 3)), edge
    if edge == "bottom":
        return (h - 1, rng.randint(2, w - 3)), edge
    if edge == "left":
        return (rng.randint(2, h - 3), 0), edge
    return (rng.randint(2, h - 3), w - 1), edge


def _endpoint_degree(path_cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((nr, nc) in path_cells for nr, nc in _neighbors4(r, c))


def _is_clear_for_path(
    cell: tuple[int, int],
    shape: tuple[int, int],
    current: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    prev: tuple[int, int] | None,
) -> bool:
    r, c = cell
    if not _in_bounds(shape, r, c) or cell in current or cell in blocked:
        return False
    for nr in range(r - 1, r + 2):
        for nc in range(c - 1, c + 2):
            if (nr, nc) in blocked:
                return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in current and (prev is None or (nr, nc) != prev):
            return False
    return True


def _build_path(
    rng: random.Random,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    tie_mode: bool,
) -> list[tuple[int, int]] | None:
    h, w = shape
    start, edge = _edge_start(rng, h, w, tie_mode)
    path = [start]
    current = {start}
    direction = _dir_from_edge(edge)
    seg_count = rng.randint(*SEGMENT_COUNT_RANGE)

    for seg_idx in range(seg_count):
        directions = [direction] if seg_idx == 0 else _perpendiculars(direction)
        rng.shuffle(directions)
        placed = False
        for cand_dir in directions:
            lengths = list(range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1))
            rng.shuffle(lengths)
            if tie_mode and seg_idx == seg_count - 1:
                last_r, last_c = path[-1]
                if cand_dir[0] == 1:
                    lengths = [h - 1 - last_r]
                elif cand_dir[0] == -1:
                    lengths = [last_r]
                elif cand_dir[1] == 1:
                    lengths = [w - 1 - last_c]
                else:
                    lengths = [last_c]
            for length in lengths:
                if length < 2:
                    continue
                new_cells: list[tuple[int, int]] = []
                rr, cc = path[-1]
                ok = True
                prev = path[-1]
                for _ in range(length):
                    rr += cand_dir[0]
                    cc += cand_dir[1]
                    cell = (rr, cc)
                    current_preview = current | set(new_cells)
                    if not _is_clear_for_path(cell, shape, current_preview, blocked, prev):
                        ok = False
                        break
                    new_cells.append(cell)
                    prev = cell
                if not ok:
                    continue
                path.extend(new_cells)
                current.update(new_cells)
                direction = cand_dir
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if len(path) < 10:
        return None

    start_key = _edge_key(shape, path[0])
    end_key = _edge_key(shape, path[-1])
    if tie_mode:
        if start_key[0] != end_key[0]:
            return None
        if start_key >= end_key:
            return None
    else:
        if not (start_key < end_key):
            return None
        if start_key[0] == end_key[0]:
            return None

    cells = set(path)
    if _endpoint_degree(cells, path[0]) != 1 or _endpoint_degree(cells, path[-1]) != 1:
        return None
    return path


def _find_filter_slot(
    path: list[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[int, set[tuple[int, int]]] | None:
    path_set = set(path)
    for i in range(2, len(path) - 3):
        a, b, c, d = path[i - 1], path[i], path[i + 1], path[i + 2]
        if not ((abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1) and (abs(b[0] - c[0]) + abs(b[1] - c[1]) == 1) and (abs(c[0] - d[0]) + abs(c[1] - d[1]) == 1)):
            continue
        if a[0] == b[0] == c[0] == d[0]:
            r = b[0]
            c0 = min(b[1], c[1])
            for side in (-1, 1):
                cells = {(r, c0), (r, c0 + 1), (r + side, c0), (r + side, c0 + 1)}
                if all(_in_bounds(shape, rr, cc) for rr, cc in cells) and _slot_is_clear(
                    cells, path_set, blocked, {a, d}
                ):
                    return i, cells
        if a[1] == b[1] == c[1] == d[1]:
            ccol = b[1]
            r0 = min(b[0], c[0])
            for side in (-1, 1):
                cells = {(r0, ccol), (r0 + 1, ccol), (r0, ccol + side), (r0 + 1, ccol + side)}
                if all(_in_bounds(shape, rr, cc) for rr, cc in cells) and _slot_is_clear(
                    cells, path_set, blocked, {a, d}
                ):
                    return i, cells
    return None


def _slot_is_clear(
    filter_cells: set[tuple[int, int]],
    path_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    allowed_contacts: set[tuple[int, int]],
) -> bool:
    extras = filter_cells - path_set
    if extras & blocked:
        return False
    for r, c in extras:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in blocked:
                    return False
    remaining = path_set - (filter_cells & path_set)
    contacts: set[tuple[int, int]] = set()
    for r, c in filter_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in remaining:
                contacts.add((nr, nc))
    if contacts != allowed_contacts:
        return False
    return True


def _reserve(cells: set[tuple[int, int]], blocked: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(shape, nr, nc):
                    blocked.add((nr, nc))


def _paint_scene(
    shape: tuple[int, int],
    paths: list[dict],
    recolored: bool,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for item in paths:
        path = item["path"]
        idx = item["filter_index"]
        filt = item["filter_cells"]
        path_color = item["path_color"]
        filter_color = item["filter_color"]
        for j, (r, c) in enumerate(path):
            if j in (idx, idx + 1):
                continue
            color = path_color
            if recolored and j >= idx + 2:
                color = filter_color
            grid[r, c] = color
        for r, c in filt:
            grid[r, c] = filter_color
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        max_paths = min(PATH_COUNT_RANGE[1], len(PALETTE) // 2)
        path_count = rng.randint(PATH_COUNT_RANGE[0], max_paths)
        colors = rng.sample(PALETTE, 2 * path_count)
        blocked: set[tuple[int, int]] = set()
        paths: list[dict] = []
        force_tie = rng.random() < 0.5
        ok = True

        for path_idx in range(path_count):
            tie_mode = force_tie if path_idx == 0 else rng.random() < 0.5
            built = None
            for _attempt in range(80):
                path = _build_path(rng, (h, w), blocked, tie_mode)
                if path is None:
                    continue
                slot = _find_filter_slot(path, (h, w), blocked)
                if slot is None:
                    continue
                filter_index, filter_cells = slot
                built = (path, filter_index, filter_cells)
                break
            if built is None:
                ok = False
                break

            path_color = colors[2 * path_idx]
            filter_color = colors[2 * path_idx + 1]
            path, filter_index, filter_cells = built
            occupied = (set(path) - {path[filter_index], path[filter_index + 1]}) | set(filter_cells)
            _reserve(occupied, blocked, (h, w))
            paths.append(
                {
                    "path": path,
                    "filter_index": filter_index,
                    "filter_cells": filter_cells,
                    "path_color": path_color,
                    "filter_color": filter_color,
                }
            )

        if not ok or not paths:
            continue

        inp = _paint_scene((h, w), paths, recolored=False)
        expected = _paint_scene((h, w), paths, recolored=True)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("generate_input failed")
