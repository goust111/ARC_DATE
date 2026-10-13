import random
from functools import lru_cache

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

INPUT_COLORS = (1, 2, 3, 6, 7, 8, 9)
BODY_COLORS = (3, 6, 7, 8, 9)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
NUM_POLYGONS_OPTIONS = (1, 2)
NUM_DISTRACTORS_RANGE = (0, 3)
RECT_SIDE_RANGE = (5, 11)
PLACEMENT_PADDING = 2
DISTRACTOR_LENGTH_RANGE = (3, 7)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_segments(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != YELLOW)
    segments: list[dict] = []
    for comp in _components(mask):
        if len(comp) < 2:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if len(set(rows)) == 1:
            row = rows[0]
            ordered = sorted(comp, key=lambda p: p[1])
            expected = list(range(ordered[0][1], ordered[-1][1] + 1))
            if [c for _, c in ordered] != expected:
                continue
            axis = "h"
        elif len(set(cols)) == 1:
            col = cols[0]
            ordered = sorted(comp, key=lambda p: p[0])
            expected = list(range(ordered[0][0], ordered[-1][0] + 1))
            if [r for r, _ in ordered] != expected:
                continue
            axis = "v"
        else:
            continue

        endpoints = [ordered[0], ordered[-1]]
        endpoint_colors = [int(grid[p]) for p in endpoints]
        segments.append(
            {
                "coords": ordered,
                "axis": axis,
                "endpoints": endpoints,
                "endpoint_colors": endpoint_colors,
            }
        )
    return segments


def _path_is_clear(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> bool:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        if hi - lo < 1:
            return False
        if hi - lo == 1:
            return True
        return bool(np.all(grid[r0, lo + 1 : hi] == BG))
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        if hi - lo < 1:
            return False
        if hi - lo == 1:
            return True
        return bool(np.all(grid[lo + 1 : hi, c0] == BG))
    return False


def _maximum_matching(endpoints: list[dict], edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    n = len(endpoints)
    adjacency = [set() for _ in range(n)]
    lengths: dict[tuple[int, int], int] = {}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
        p0 = endpoints[i]["coord"]
        p1 = endpoints[j]["coord"]
        lengths[(min(i, j), max(i, j))] = abs(p0[0] - p1[0]) + abs(p0[1] - p1[1])

    @lru_cache(maxsize=None)
    def search(mask: int):
        if mask == 0:
            return 0, 0, ()

        bit = mask & -mask
        i = bit.bit_length() - 1
        best = search(mask ^ bit)
        best = (best[0], best[1], best[2])

        for j in adjacency[i]:
            if not (mask & (1 << j)):
                continue
            nxt = search(mask ^ bit ^ (1 << j))
            key = (min(i, j), max(i, j))
            candidate = (
                nxt[0] + 2,
                nxt[1] - lengths[key],
                ((i, j),) + nxt[2],
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    return list(search((1 << n) - 1)[2])


def _match_endpoints(grid: np.ndarray, segments: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    endpoints: list[dict] = []
    for seg_id, segment in enumerate(segments):
        for endpoint, color in zip(segment["endpoints"], segment["endpoint_colors"]):
            endpoints.append({"seg_id": seg_id, "coord": endpoint, "color": color})

    by_color: dict[int, list[int]] = {}
    for idx, endpoint in enumerate(endpoints):
        by_color.setdefault(endpoint["color"], []).append(idx)

    matched_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for indices in by_color.values():
        if len(indices) < 2:
            continue
        local_edges: list[tuple[int, int]] = []
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                i = indices[a_pos]
                j = indices[b_pos]
                if endpoints[i]["seg_id"] == endpoints[j]["seg_id"]:
                    continue
                if _path_is_clear(grid, endpoints[i]["coord"], endpoints[j]["coord"]):
                    local_edges.append((a_pos, b_pos))

        if not local_edges:
            continue
        local_endpoints = [endpoints[i] for i in indices]
        for ai, bi in _maximum_matching(local_endpoints, local_edges):
            matched_pairs.append((local_endpoints[ai]["coord"], local_endpoints[bi]["coord"]))

    return matched_pairs


def _draw_connector(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = GREY
    else:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = GREY


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = _find_segments(g)
    matched_pairs = _match_endpoints(g, segments)

    out = g.copy()
    for p0, p1 in matched_pairs:
        _draw_connector(out, p0, p1)

    boundary = (out != BG) & (out != YELLOW)
    free = ~boundary

    filled_regions: list[dict] = []
    for comp in _components(free):
        touches_border = any(
            r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in comp
        )
        if touches_border:
            continue

        touches_grey = False
        yellow_count = 0
        for r, c in comp:
            if out[r, c] == YELLOW:
                yellow_count += 1
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and boundary[nr, nc]:
                    if out[nr, nc] == GREY:
                        touches_grey = True
        if not touches_grey:
            continue

        fill_color = BLUE if yellow_count % 2 == 1 else RED
        for r, c in comp:
            out[r, c] = fill_color
        filled_regions.append({"cells": comp, "yellow_count": yellow_count, "fill_color": fill_color})

    meta = {"matched_pairs": matched_pairs, "filled_regions": filled_regions, "segments": segments}
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)[0]


def _expanded_box(r0: int, c0: int, r1: int, c1: int, pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    er0, ec0, er1, ec1 = _expanded_box(r0, c0, r1, c1, pad, occupied.shape[0], occupied.shape[1])
    return not bool(np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1]))


def _paint_rect_fragment_polygon(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], int] | None:
    h, w = grid.shape
    max_side_h = min(RECT_SIDE_RANGE[1], h - 2 * PLACEMENT_PADDING - 1)
    max_side_w = min(RECT_SIDE_RANGE[1], w - 2 * PLACEMENT_PADDING - 1)
    if max_side_h < RECT_SIDE_RANGE[0] or max_side_w < RECT_SIDE_RANGE[0]:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        side_h = rng.randint(RECT_SIDE_RANGE[0], max_side_h)
        side_w = rng.randint(RECT_SIDE_RANGE[0], max_side_w)
        r0 = rng.randint(PLACEMENT_PADDING, h - side_h - PLACEMENT_PADDING - 1)
        c0 = rng.randint(PLACEMENT_PADDING, w - side_w - PLACEMENT_PADDING - 1)
        r1 = r0 + side_h
        c1 = c0 + side_w
        if not _box_is_free(occupied, r0, c0, r1, c1, PLACEMENT_PADDING):
            continue

        gap_top = rng.randint(1, side_w - 2)
        gap_right = rng.randint(1, side_h - 2)
        gap_bottom = rng.randint(1, side_w - 2)
        gap_left = rng.randint(1, side_h - 2)

        corner_colors = rng.sample(INPUT_COLORS, 4)
        top_left_color, top_right_color, bottom_right_color, bottom_left_color = corner_colors

        body_choices = []
        endpoint_pairs = (
            (top_left_color, top_right_color),
            (top_right_color, bottom_right_color),
            (bottom_left_color, bottom_right_color),
            (top_left_color, bottom_left_color),
        )
        for color_a, color_b in endpoint_pairs:
            candidates = [c for c in BODY_COLORS if c not in (color_a, color_b)]
            body_choices.append(rng.choice(candidates))
        top_body, right_body, bottom_body, left_body = body_choices

        interior_cells = [
            (rr, cc)
            for rr in range(r0 + 1, r1)
            for cc in range(c0 + 1, c1)
        ]
        if len(interior_cells) < 4:
            continue

        choose_odd = rng.choice([False, True])
        yellow_options = [1, 3] if choose_odd else [0, 2, 4]
        yellow_options = [k for k in yellow_options if k <= len(interior_cells)]
        if not yellow_options:
            continue
        yellow_count = rng.choice(yellow_options)

        top_end = (r0, c1 - gap_top)
        right_end = (r1 - gap_right, c1)
        bottom_end = (r1, c0 + gap_bottom)
        left_end = (r0 + gap_left, c0)

        segment_cells = set()
        segment_cells.update((r0, c) for c in range(c0, c1 - gap_top + 1))
        segment_cells.update((r, c1) for r in range(r0, r1 - gap_right + 1))
        segment_cells.update((r1, c) for c in range(c0 + gap_bottom, c1 + 1))
        segment_cells.update((r, c0) for r in range(r0 + gap_left, r1 + 1))
        if any(grid[r, c] != BG for r, c in segment_cells):
            continue

        for c in range(c0, c1 - gap_top + 1):
            grid[r0, c] = top_body
        grid[r0, c0] = top_left_color
        grid[top_end] = top_right_color

        for r in range(r0, r1 - gap_right + 1):
            grid[r, c1] = right_body
        grid[r0, c1] = top_right_color
        grid[right_end] = bottom_right_color

        for c in range(c0 + gap_bottom, c1 + 1):
            grid[r1, c] = bottom_body
        grid[r1, c1] = bottom_right_color
        grid[bottom_end] = bottom_left_color

        for r in range(r0 + gap_left, r1 + 1):
            grid[r, c0] = left_body
        grid[left_end] = top_left_color
        grid[r1, c0] = bottom_left_color

        marker_cells = rng.sample(interior_cells, yellow_count)
        for r, c in marker_cells:
            grid[r, c] = YELLOW

        er0, ec0, er1, ec1 = _expanded_box(r0, c0, r1, c1, PLACEMENT_PADDING, h, w)
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True

        expected_pairs = {
            tuple(sorted((top_end, (r0, c1)))),
            tuple(sorted((right_end, (r1, c1)))),
            tuple(sorted((bottom_end, (r1, c0)))),
            tuple(sorted((left_end, (r0, c0)))),
        }
        return expected_pairs, yellow_count

    return None


def _paint_distractor(grid: np.ndarray, occupied: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        axis = rng.choice(("h", "v"))
        length = rng.randint(DISTRACTOR_LENGTH_RANGE[0], DISTRACTOR_LENGTH_RANGE[1])
        if axis == "h":
            if length >= w - 2:
                continue
            r = rng.randint(1, h - 2)
            c0 = rng.randint(1, w - length - 1)
            r0, c1 = r, c0 + length - 1
            if not _box_is_free(occupied, r, c0, r, c1, 1):
                continue
            left_color, right_color = rng.sample(INPUT_COLORS, 2)
            body_color = rng.choice([c for c in BODY_COLORS if c not in (left_color, right_color)])
            grid[r, c0 : c1 + 1] = body_color
            grid[r, c0] = left_color
            grid[r, c1] = right_color
            er0, ec0, er1, ec1 = _expanded_box(r, c0, r, c1, 1, h, w)
        else:
            if length >= h - 2:
                continue
            c = rng.randint(1, w - 2)
            r0 = rng.randint(1, h - length - 1)
            r1 = r0 + length - 1
            if not _box_is_free(occupied, r0, c, r1, c, 1):
                continue
            top_color, bottom_color = rng.sample(INPUT_COLORS, 2)
            body_color = rng.choice([c for c in BODY_COLORS if c not in (top_color, bottom_color)])
            grid[r0 : r1 + 1, c] = body_color
            grid[r0, c] = top_color
            grid[r1, c] = bottom_color
            er0, ec0, er1, ec1 = _expanded_box(r0, c, r1, c, 1, h, w)

        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return True
    return False


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    expected_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    expected_yellow_counts: list[int] = []
    num_polygons = rng.choice(NUM_POLYGONS_OPTIONS)
    for _ in range(num_polygons):
        result = _paint_rect_fragment_polygon(grid, occupied, rng)
        if result is None:
            return None
        pairs, yellow_count = result
        expected_pairs.update(pairs)
        expected_yellow_counts.append(yellow_count)

    for _ in range(rng.randint(*NUM_DISTRACTORS_RANGE)):
        _paint_distractor(grid, occupied, rng)

    out, meta = _solve(grid)
    actual_pairs = {tuple(sorted(pair)) for pair in meta["matched_pairs"]}
    actual_counts = sorted(region["yellow_count"] for region in meta["filled_regions"])

    if actual_pairs != expected_pairs:
        return None
    if len(meta["filled_regions"]) != num_polygons:
        return None
    if actual_counts != sorted(expected_yellow_counts):
        return None
    if not np.any(out == BLUE) and not np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
