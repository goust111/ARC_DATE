from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
_GEN_CALL_INDEX = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _endpoint_open_dirs(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    mask = grid == GREY
    endpoints: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        neigh = [(nr, nc) for nr, nc in _neighbors4(rr, cc) if _in_bounds(grid.shape, nr, nc) and mask[nr, nc]]
        if len(neigh) != 1:
            continue
        nr, nc = neigh[0]
        endpoints[(rr, cc)] = (rr - nr, cc - nc)
    return endpoints


def _nearest_visible_endpoint(grid: np.ndarray, marker: tuple[int, int], endpoints: dict[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0 = marker
    candidates: list[tuple[int, int, int]] = []
    for dr, dc in DIRS4:
        r, c = r0 + dr, c0 + dc
        dist = 1
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value == BACKGROUND:
                r += dr
                c += dc
                dist += 1
                continue
            if value == GREY and (r, c) in endpoints:
                candidates.append((dist, r, c))
            break
    assert candidates, f"marker at {marker} must see a grey endpoint"
    candidates.sort()
    best_dist = candidates[0][0]
    best = [pt for dist, *pt in candidates if dist == best_dist]
    assert len(best) == 1, f"marker at {marker} must have a unique nearest endpoint"
    return int(best[0][0]), int(best[0][1])


def _facing_pair(p0: tuple[int, int], d0: tuple[int, int], p1: tuple[int, int], d1: tuple[int, int]) -> bool:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1 and c0 < c1:
        return d0 == (0, 1) and d1 == (0, -1)
    if r0 == r1 and c1 < c0:
        return d0 == (0, -1) and d1 == (0, 1)
    if c0 == c1 and r0 < r1:
        return d0 == (1, 0) and d1 == (-1, 0)
    if c0 == c1 and r1 < r0:
        return d0 == (-1, 0) and d1 == (1, 0)
    return False


def _repair_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    original_endpoints = _endpoint_open_dirs(g)
    for r, c in np.argwhere(g == RED):
        marker = (int(r), int(c))
        _nearest_visible_endpoint(g, marker, original_endpoints)
        g[marker] = GREY

    endpoints = _endpoint_open_dirs(g)
    fill_cells: set[tuple[int, int]] = set()
    items = sorted(endpoints.items())
    for i, (p0, d0) in enumerate(items):
        r0, c0 = p0
        for p1, d1 in items[i + 1 :]:
            r1, c1 = p1
            if r0 != r1 and c0 != c1:
                continue
            if not _facing_pair(p0, d0, p1, d1):
                continue
            if r0 == r1:
                left, right = sorted([c0, c1])
                gap = right - left - 1
                if not (1 <= gap <= 3):
                    continue
                if np.all(g[r0, left + 1 : right] == BACKGROUND):
                    for c in range(left + 1, right):
                        fill_cells.add((r0, c))
            else:
                top, bottom = sorted([r0, r1])
                gap = bottom - top - 1
                if not (1 <= gap <= 3):
                    continue
                if np.all(g[top + 1 : bottom, c0] == BACKGROUND):
                    for r in range(top + 1, bottom):
                        fill_cells.add((r, c0))
    for r, c in fill_cells:
        g[r, c] = GREY
    return g


def _ranked_components(grid: np.ndarray) -> list[np.ndarray]:
    comps = _components(grid == GREY)
    assert 3 <= len(comps) <= 6, "repaired grid must have 3 to 6 grey components"
    areas = [int(len(comp)) for comp in comps]
    assert len(set(areas)) == len(areas), "repaired component areas must be unique"
    return sorted(comps, key=lambda comp: (int(len(comp)), _bbox(comp)[0], _bbox(comp)[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the broken grey line network and recolor each repaired component by ascending area rank.

    Input structure:
    - The input is a black background grid containing one-pixel-wide grey horizontal/vertical line fragments.
    - Red singleton markers lie on the same row or column as a visible grey endpoint and indicate which fragment should be extended.
    - The grey fragments may already include turns, but every repair action happens along straight row or column gaps.

    Rule:
    - Recolor each red marker to grey after assigning it to its unique nearest visible grey endpoint.
    - On the repaired grey mask, find every pair of grey endpoints in one row or one column that face each other across an all-black gap of length 1 to 3, and fill that gap with grey.
    - Compute the 4-connected grey components after all repairs, sort them by area from small to large, and break ties by the top-left component bbox corner.
    - Recolor the ranked components with the repeating palette green, red, blue.

    Key insight:
    - The red markers do not just add pixels: they move endpoints outward, which can enable short facing-gap repairs that merge fragments and completely change the component-area ranking used for the final colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, BLUE, RED, GREEN, GREY}), "unexpected colors"

    repaired = _repair_grid(g)
    ranked = _ranked_components(repaired)
    palette = [GREEN, RED, BLUE]

    out = np.zeros_like(repaired)
    for rank, comp in enumerate(ranked):
        color = palette[rank % len(palette)]
        out[comp[:, 0], comp[:, 1]] = color
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _sample_component(
    area: int, rng: random.Random, force_turn: bool, forced_gap: int | None
) -> dict[str, object] | None:
    for _ in range(200):
        gap = forced_gap if forced_gap is not None else rng.randint(1, 3)
        tail_len_choices = [0, 2, 3, 4] if not force_turn else [2, 3, 4]
        rng.shuffle(tail_len_choices)
        for tail_len in tail_len_choices:
            main_len = area - tail_len
            if main_len < gap + 5:
                continue
            total_frag = main_len - gap - 1
            left_min = max(2, total_frag - 8)
            left_max = min(8, total_frag - 2)
            if left_min > left_max:
                continue
            left_len = rng.randint(left_min, left_max)
            right_len = total_frag - left_len
            if not (2 <= right_len <= 8):
                continue

            horizontal = bool(rng.randint(0, 1))
            marker_on_left = bool(rng.randint(0, 1))
            cells = set()
            grey = set()
            if horizontal:
                for c in range(main_len):
                    cells.add((0, c))
                if marker_on_left:
                    marker = (0, left_len)
                    grey |= {(0, c) for c in range(left_len)}
                    grey |= {(0, c) for c in range(left_len + gap + 1, main_len)}
                else:
                    marker = (0, left_len + gap)
                    grey |= {(0, c) for c in range(left_len)}
                    grey |= {(0, c) for c in range(left_len + gap + 1, main_len)}
                if tail_len:
                    anchor = (0, 0) if bool(rng.randint(0, 1)) else (0, main_len - 1)
                    dr = -1 if bool(rng.randint(0, 1)) else 1
                    for step in range(1, tail_len + 1):
                        cells.add((anchor[0] + dr * step, anchor[1]))
                        grey.add((anchor[0] + dr * step, anchor[1]))
            else:
                for r in range(main_len):
                    cells.add((r, 0))
                if marker_on_left:
                    marker = (left_len, 0)
                    grey |= {(r, 0) for r in range(left_len)}
                    grey |= {(r, 0) for r in range(left_len + gap + 1, main_len)}
                else:
                    marker = (left_len + gap, 0)
                    grey |= {(r, 0) for r in range(left_len)}
                    grey |= {(r, 0) for r in range(left_len + gap + 1, main_len)}
                if tail_len:
                    anchor = (0, 0) if bool(rng.randint(0, 1)) else (main_len - 1, 0)
                    dc = -1 if bool(rng.randint(0, 1)) else 1
                    for step in range(1, tail_len + 1):
                        cells.add((anchor[0], anchor[1] + dc * step))
                        grey.add((anchor[0], anchor[1] + dc * step))

            min_r = min(r for r, _ in cells)
            min_c = min(c for _, c in cells)
            cells = {(r - min_r, c - min_c) for r, c in cells}
            grey = {(r - min_r, c - min_c) for r, c in grey}
            marker = (marker[0] - min_r, marker[1] - min_c)
            comp_mask = np.zeros(
                (
                    max(r for r, _ in cells) + 1,
                    max(c for _, c in cells) + 1,
                ),
                dtype=bool,
            )
            for r, c in cells:
                comp_mask[r, c] = True
            comp = _components(comp_mask)[0]
            if len(comp) != area:
                continue
            if len(_endpoint_open_dirs(np.where(comp_mask, GREY, BACKGROUND))) != 2:
                continue
            return {
                "cells": cells,
                "grey": grey,
                "marker": marker,
                "shape": comp_mask.shape,
                "area": area,
            }
    return None


def _component_bbox(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _can_place_component(
    cells: set[tuple[int, int]], offset: tuple[int, int], blocked: np.ndarray
) -> bool:
    off_r, off_c = offset
    h, w = blocked.shape
    for r, c in cells:
        rr, cc = r + off_r, c + off_c
        if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
            return False
    return True


def _mark_component_blocked(
    blocked: np.ndarray, cells: set[tuple[int, int]], offset: tuple[int, int]
) -> None:
    off_r, off_c = offset
    h, w = blocked.shape
    for r, c in cells:
        rr, cc = r + off_r, c + off_c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_components(
    shape: tuple[int, int],
    areas: list[int],
    forced_gap: int,
    force_turn: bool,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []
    for i, area in enumerate(areas):
        comp = _sample_component(area, rng, force_turn and i == 0, forced_gap if i == 0 else None)
        if comp is None:
            return None
        ch, cw = _component_bbox(comp["cells"])
        if ch > h or cw > w:
            return None
        placed_one = False
        for _ in range(200):
            off_r = rng.randint(0, h - ch)
            off_c = rng.randint(0, w - cw)
            if not _can_place_component(comp["cells"], (off_r, off_c), blocked):
                continue
            placed_comp = {
                "cells": {(r + off_r, c + off_c) for r, c in comp["cells"]},
                "grey": {(r + off_r, c + off_c) for r, c in comp["grey"]},
                "marker": (comp["marker"][0] + off_r, comp["marker"][1] + off_c),
            }
            placed.append(placed_comp)
            _mark_component_blocked(blocked, comp["cells"], (off_r, off_c))
            placed_one = True
            break
        if not placed_one:
            return None
    return placed


def _build_input_grid(shape: tuple[int, int], components: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    for comp in components:
        for r, c in comp["grey"]:
            if grid[r, c] != BACKGROUND:
                return None
            grid[r, c] = GREY
    for comp in components:
        r, c = comp["marker"]
        if grid[r, c] != BACKGROUND:
            return None
        grid[r, c] = RED
    return grid


def _expected_repaired(shape: tuple[int, int], components: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for comp in components:
        for r, c in comp["cells"]:
            grid[r, c] = GREY
    return grid


def _valid_marker_assignments(grid: np.ndarray) -> bool:
    endpoints = _endpoint_open_dirs(np.where(grid == GREY, GREY, BACKGROUND))
    try:
        for r, c in np.argwhere(grid == RED):
            _nearest_visible_endpoint(grid, (int(r), int(c)), endpoints)
    except AssertionError:
        return False
    return True


def _generate_attempt(rng: random.Random, count: int, forced_gap: int, force_turn: bool) -> np.ndarray | None:
    h = rng.randint(15, 20)
    w = rng.randint(15, 20)
    area_pool = list(range(6, 17))
    areas = sorted(rng.sample(area_pool, count))
    components = _place_components((h, w), areas, forced_gap, force_turn, rng)
    if components is None:
        return None

    inp = _build_input_grid((h, w), components)
    if inp is None or not _valid_marker_assignments(inp):
        return None

    repaired = _repair_grid(inp)
    expected = _expected_repaired((h, w), components)
    if not np.array_equal(repaired, expected):
        return None
    ranked = _ranked_components(repaired)
    if len(ranked) != count:
        return None
    if np.array_equal(main(inp), inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX
    _GEN_CALL_INDEX += 1

    schedule = [
        (3, 1, False),
        (4, 2, True),
        (5, 3, False),
        (6, 1, False),
        (3, 2, True),
        (4, 3, False),
        (5, 1, True),
        (6, 2, False),
        (3, 3, False),
        (4, 1, True),
        (5, 2, False),
        (6, 1, False),
    ]
    count, forced_gap, force_turn = schedule[(_GEN_CALL_INDEX - 1) % len(schedule)]

    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng, count, forced_gap, force_turn)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
