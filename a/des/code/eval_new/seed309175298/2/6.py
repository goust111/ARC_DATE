import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

STAMP_COLORS = (BLUE, RED, YELLOW, ORANGE)
NEXT_COLOR = {
    RED: ORANGE,
    ORANGE: YELLOW,
    YELLOW: BLUE,
    BLUE: RED,
}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
COMBO_COUNT_OPTIONS = (1, 2, 3)
SEGMENT_COUNT_OPTIONS = (1, 2, 3)
SEGMENT_LEN_RANGE = (2, 6)
GEN_RETRIES = 400
PLACE_RETRIES = 120
HALO = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _mono_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    allowed = set(colors)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in allowed or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, pts))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_stamps(grid: np.ndarray) -> list[dict[str, object]]:
    stamps: list[dict[str, object]] = []
    for color, pts in _mono_components(grid, STAMP_COLORS):
        if len(pts) != 4:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        cell_set = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if set(pts) != cell_set:
            continue
        stamps.append(
            {
                "color": int(color),
                "top_left": (int(r0), int(c0)),
                "cells": cell_set,
            }
        )
    return stamps


def _find_endpoint_mask(grid: np.ndarray, stamp_cells: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    endpoint_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in STAMP_COLORS or (r, c) in stamp_cells:
                continue
            if any(int(grid[nr, nc]) in (GREY, MAROON) for nr, nc in _neighbors4(r, c, h, w)):
                endpoint_mask[r, c] = True
    return endpoint_mask


def _adjacent_stamp(coord: tuple[int, int], stamps: list[dict[str, object]]) -> int | None:
    r, c = coord
    hits: list[int] = []
    for idx, stamp in enumerate(stamps):
        for sr, sc in stamp["cells"]:
            if abs(sr - r) + abs(sc - c) == 1:
                hits.append(idx)
                break
    if len(hits) == 1:
        return hits[0]
    return None


def _order_visible_path(
    comp: list[tuple[int, int]],
    endpoint_mask: np.ndarray,
    stamps: list[dict[str, object]],
    h: int,
    w: int,
) -> tuple[int, list[tuple[int, int]]]:
    comp_set = set(comp)
    neighbors = {
        p: [q for q in _neighbors4(p[0], p[1], h, w) if q in comp_set]
        for p in comp
    }
    ends = [p for p in comp if len(neighbors[p]) == 1]
    comp_endpoints = [p for p in comp if bool(endpoint_mask[p[0], p[1]])]
    assert len(comp_endpoints) == 1
    finish = comp_endpoints[0]

    start = None
    stamp_idx = None
    for p in ends:
        idx = _adjacent_stamp(p, stamps)
        if idx is not None:
            start = p
            stamp_idx = idx
            break
    if start is None or stamp_idx is None:
        raise AssertionError("could not identify the track start")

    ordered = [start]
    prev = None
    curr = start
    while curr != finish:
        nexts = [p for p in neighbors[curr] if p != prev]
        assert len(nexts) == 1
        prev, curr = curr, nexts[0]
        ordered.append(curr)
    return int(stamp_idx), ordered


def _reconstruct_full_path(
    stamp_top_left: tuple[int, int],
    visible_path: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0 = stamp_top_left
    full = [(r0, c0)]
    if not visible_path:
        return full

    first = visible_path[0]
    if first == (r0, c0 + 2):
        full.append((r0, c0 + 1))
    elif first == (r0 + 2, c0):
        full.append((r0 + 1, c0))
    full.extend(visible_path)
    return full


def _paint_trace(
    out: np.ndarray,
    positions: list[tuple[int, int]],
    marker_coords: set[tuple[int, int]],
    start_color: int,
) -> None:
    current = int(start_color)
    for i, (r, c) in enumerate(positions):
        if i > 0 and (r, c) in marker_coords:
            current = NEXT_COLOR[current]
        out[r : r + 2, c : c + 2] = current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 2x2 colored stamp along its grey track and paint the swept trail with color changes at maroon markers.

    Input structure:
    - Black background with several separate stamp-track systems.
    - Each stamp is a solid 2x2 block in red, orange, yellow, or blue.
    - A one-pixel-wide track of grey cells extends from each stamp; maroon track cells are color-change triggers.
    - The visible track ends at a single endpoint pixel that matches that stamp's initial color.

    Rule:
    - Treat the track as the path of the stamp's top-left corner from the start stamp to the matching endpoint.
    - The stamp keeps painting 2x2 squares along that path, so the output is the union of all swept stamp positions.
    - When the top-left lands on a maroon track pixel, advance the stamp color in the cycle red -> orange -> yellow -> blue -> red before painting that step.
    - Covered track cells and the endpoint disappear under the painted trail; untouched background and unrelated tracks remain unchanged.

    Key insight:
    - The grey pixels are not the trail itself but the movement path of a 2x2 stamp, and maroon pixels are triggers that recolor later path segments according to a fixed visible cycle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    stamps = _find_stamps(g)
    stamp_cells = set()
    for stamp in stamps:
        stamp_cells.update(stamp["cells"])

    endpoint_mask = _find_endpoint_mask(g, stamp_cells)
    path_mask = (g == GREY) | (g == MAROON) | endpoint_mask
    marker_coords = set(map(tuple, np.argwhere(g == MAROON)))

    out = g.copy()
    for comp in _mask_components(path_mask):
        stamp_idx, visible_path = _order_visible_path(comp, endpoint_mask, stamps, h, w)
        full_path = _reconstruct_full_path(stamps[stamp_idx]["top_left"], visible_path)
        _paint_trace(out, full_path, marker_coords, int(stamps[stamp_idx]["color"]))
    return out


def _visible_path_positions(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0 = path[0]
    hidden = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
    return [p for p in path[1:] if p not in hidden]


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    if len(set(path)) != len(path):
        return False
    for i, (r1, c1) in enumerate(path):
        for j in range(i + 1, len(path)):
            r2, c2 = path[j]
            if abs(r1 - r2) + abs(c1 - c2) == 1 and j != i + 1:
                return False
    return True


def _build_relative_path() -> list[tuple[int, int]] | None:
    segment_count = random.choice(SEGMENT_COUNT_OPTIONS)
    directions = [random.choice(DIR4)]
    for _ in range(1, segment_count):
        prev = directions[-1]
        if prev[0] == 0:
            directions.append(random.choice(((1, 0), (-1, 0))))
        else:
            directions.append(random.choice(((0, 1), (0, -1))))

    lengths = [random.randint(*SEGMENT_LEN_RANGE) for _ in range(segment_count)]
    path = [(0, 0)]
    r, c = 0, 0
    for (dr, dc), seg_len in zip(directions, lengths):
        for _ in range(seg_len):
            r += dr
            c += dc
            path.append((r, c))

    hidden = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for i, p in enumerate(path[1:], start=1):
        if p in hidden and not (i == 1 and p in {(0, 1), (1, 0)}):
            return None

    visible = _visible_path_positions(path)
    if len(visible) < 2:
        return None
    if not _is_simple_path(path):
        return None
    return path


def _choose_marker_indices(path: list[tuple[int, int]]) -> set[int] | None:
    visible = _visible_path_positions(path)
    visible_lookup = {p: i for i, p in enumerate(path)}
    candidates = [visible_lookup[p] for p in visible[:-1]]
    if not candidates:
        return None

    marker_count = random.randint(1, min(3, len(candidates)))
    random.shuffle(candidates)
    picked: list[int] = []
    for idx in candidates:
        if all(abs(idx - prev) >= 2 for prev in picked):
            picked.append(idx)
        if len(picked) == marker_count:
            break
    if len(picked) != marker_count:
        return None
    return set(picked)


def _footprint_cells(path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in path:
        for dr in (0, 1):
            for dc in (0, 1):
                cells.add((r + dr, c + dc))
    return cells


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - HALO, r + HALO + 1):
            for cc in range(c - HALO, c + HALO + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _shift_points(points: set[tuple[int, int]] | list[tuple[int, int]], dr: int, dc: int):
    return [(r + dr, c + dc) for r, c in points]


def _expected_output(grid: np.ndarray, combos: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for combo in combos:
        _paint_trace(
            out,
            combo["path"],
            combo["markers"],
            int(combo["color"]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    combo_count = random.choice(COMBO_COUNT_OPTIONS)
    occupied_halo: set[tuple[int, int]] = set()
    combos: list[dict[str, object]] = []

    for _ in range(combo_count):
        placed = False
        for _place_try in range(PLACE_RETRIES):
            path = _build_relative_path()
            if path is None:
                continue
            marker_indices = _choose_marker_indices(path)
            if marker_indices is None:
                continue

            footprint = _footprint_cells(path)
            min_r = min(r for r, _ in footprint)
            max_r = max(r for r, _ in footprint)
            min_c = min(c for _, c in footprint)
            max_c = max(c for _, c in footprint)
            lo_r, hi_r = -min_r, h - 1 - max_r
            lo_c, hi_c = -min_c, w - 1 - max_c
            if lo_r > hi_r or lo_c > hi_c:
                continue

            for _shift_try in range(PLACE_RETRIES):
                shift_r = random.randint(lo_r, hi_r)
                shift_c = random.randint(lo_c, hi_c)
                shifted_path = _shift_points(path, shift_r, shift_c)
                shifted_footprint = set(_shift_points(footprint, shift_r, shift_c))
                if shifted_footprint & occupied_halo:
                    continue

                start_r, start_c = shifted_path[0]
                visible_path = _visible_path_positions(shifted_path)
                endpoint = visible_path[-1]
                marker_coords = {shifted_path[i] for i in marker_indices}
                if endpoint in marker_coords:
                    continue

                color = random.choice(STAMP_COLORS)
                stamp_cells = {
                    (start_r, start_c),
                    (start_r, start_c + 1),
                    (start_r + 1, start_c),
                    (start_r + 1, start_c + 1),
                }
                input_cells = set(stamp_cells) | set(visible_path)
                if any(grid[r, c] != BG for r, c in input_cells):
                    continue

                for r, c in stamp_cells:
                    grid[r, c] = color
                for r, c in visible_path[:-1]:
                    grid[r, c] = MAROON if (r, c) in marker_coords else GREY
                grid[endpoint] = color

                combos.append(
                    {
                        "color": color,
                        "path": shifted_path,
                        "markers": marker_coords,
                    }
                )
                occupied_halo |= _halo(shifted_footprint, h, w)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _expected_output(grid, combos)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
