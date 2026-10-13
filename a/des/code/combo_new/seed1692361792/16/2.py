from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 8
GREY = 5
RED = 2
GREEN = 3
SEGMENT_COLORS = [0, 1, 4, 6, 7, 9]

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
EVEN_TURNS = [0, 2]
ODD_TURNS = [1, 3]

GRID_MIN = 16
GRID_MAX = 28
PAIR_COUNT_OPTIONS = [2, 3]
SEGMENT_LEN_RANGE = (3, 5)
STEP_PAD = 6
INTERIOR_STEP_RANGE = (1, 6)
MIN_CORRIDOR_CELLS = 2
PATH_ATTEMPTS = 200
PLACE_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _segment_endpoints(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    endpoints = []
    for r, c in cells:
        deg = sum((r + dr, c + dc) in cell_set for dr, dc in DIRS)
        if deg == 1:
            endpoints.append((r, c))
    return endpoints


def _order_path(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) == 1:
        return list(cells)
    cell_set = set(cells)
    neigh = {}
    for r, c in cells:
        neigh[(r, c)] = [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in cell_set]
    endpoints = [cell for cell, ns in neigh.items() if len(ns) == 1]
    assert len(endpoints) == 2, "grey corridor must be a simple path"
    order = [min(endpoints)]
    prev = None
    cur = order[0]
    while len(order) < len(cells):
        nxts = [nxt for nxt in neigh[cur] if nxt != prev]
        assert len(nxts) == 1, "grey corridor must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def _count_turns(seq: list[tuple[int, int]]) -> int:
    if len(seq) < 3:
        return 0
    dirs = []
    for (r0, c0), (r1, c1) in zip(seq, seq[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each paired path-and-corridor system by the parity of its corridor turns.

    Input structure:
    - The background is a uniform azure field.
    - Every non-background, non-grey color appears as exactly two separate 4-connected straight path fragments.
    - Grey pixels form one-cell-wide 4-connected corridors, each touching the endpoint of exactly two same-colored fragments.

    Rule:
    - For each grey corridor, find the two same-colored fragments that it connects and treat their union as one full path.
    - Count the number of 90-degree direction changes along the connecting corridor.
    - If the corridor turn count is odd, recolor the entire full path red; if it is even, recolor the entire full path green.

    Key insight:
    - The original fragment color only identifies which two fragments belong together; the final color is determined solely by the corridor geometry, specifically the parity of its turns.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    colored = (g != bg) & (g != GREY)
    comp_id = -np.ones((h, w), dtype=int)
    components = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not colored[r, c] or seen[r, c]:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                comp_id[rr, cc] = len(components)
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if colored[nr, nc] and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append({"color": color, "cells": cells})

    by_color: dict[int, list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        by_color[int(comp["color"])].append(idx)
    for ids in by_color.values():
        assert len(ids) == 2, "each color must appear in exactly two fragments"

    grey_components = _components_of_mask(g == GREY)
    out = g.copy()
    seen_component_ids = set()
    seen_colors = set()

    for grey_cells in grey_components:
        grey_set = set(grey_cells)
        adjacent_ids = set()
        for r, c in grey_cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if comp_id[nr, nc] >= 0:
                    adjacent_ids.add(int(comp_id[nr, nc]))
        assert len(adjacent_ids) == 2, "each grey corridor must connect exactly two fragments"
        id1, id2 = sorted(adjacent_ids)
        color1 = int(components[id1]["color"])
        color2 = int(components[id2]["color"])
        assert color1 == color2, "corridor must connect same-colored fragments"
        assert color1 not in seen_colors, "each color pair must use a unique corridor"
        seen_colors.add(color1)

        grey_order = _order_path(grey_cells)

        touching_endpoints = {}
        for cid in (id1, id2):
            cells = components[cid]["cells"]
            endpoints = _segment_endpoints(cells)
            candidates = [
                cell
                for cell in endpoints
                if any((nr, nc) in grey_set for nr, nc in _neighbors4(cell[0], cell[1], h, w))
            ]
            if not candidates:
                candidates = [
                    cell
                    for cell in cells
                    if any((nr, nc) in grey_set for nr, nc in _neighbors4(cell[0], cell[1], h, w))
                ]
            assert len(candidates) == 1, "each fragment must touch its corridor at one endpoint"
            touching_endpoints[cid] = candidates[0]

        if len(grey_order) == 1:
            start_endpoint = touching_endpoints[id1]
            end_endpoint = touching_endpoints[id2]
        else:
            start_candidates = [
                touching_endpoints[cid]
                for cid in (id1, id2)
                if touching_endpoints[cid] in set(_neighbors4(grey_order[0][0], grey_order[0][1], h, w))
            ]
            end_candidates = [
                touching_endpoints[cid]
                for cid in (id1, id2)
                if touching_endpoints[cid] in set(_neighbors4(grey_order[-1][0], grey_order[-1][1], h, w))
            ]
            assert len(start_candidates) == 1 and len(end_candidates) == 1
            start_endpoint = start_candidates[0]
            end_endpoint = end_candidates[0]

        turns = _count_turns([start_endpoint] + grey_order + [end_endpoint])
        final_color = RED if turns % 2 == 1 else GREEN

        for cid in (id1, id2):
            for r, c in components[cid]["cells"]:
                out[r, c] = final_color
                seen_component_ids.add(cid)
        for r, c in grey_cells:
            out[r, c] = final_color

    assert seen_component_ids == set(range(len(components))), "every fragment must be connected by a grey corridor"
    return out


def _turn_choices(direction: tuple[int, int]) -> list[tuple[int, int]]:
    idx = DIRS.index(direction)
    return [DIRS[(idx - 1) % 4], DIRS[(idx + 1) % 4]]


def _is_simple_path(seq: list[tuple[int, int]]) -> bool:
    if len(set(seq)) != len(seq):
        return False
    cell_set = set(seq)
    for idx, (r, c) in enumerate(seq):
        expected = set()
        if idx > 0:
            expected.add(seq[idx - 1])
        if idx + 1 < len(seq):
            expected.add(seq[idx + 1])
        actual = {(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in cell_set}
        if actual != expected:
            return False
    return True


def _build_local_path(turns: int, len1: int, len2: int) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        directions = [random.choice(DIRS)]
        for _ in range(turns):
            directions.append(random.choice(_turn_choices(directions[-1])))

        if turns == 0:
            steps = [random.randint(len1 + len2 + MIN_CORRIDOR_CELLS - 1, len1 + len2 + STEP_PAD)]
        else:
            steps = []
            for i in range(len(directions)):
                if i == 0:
                    lo = len1
                    hi = len1 + STEP_PAD
                elif i == len(directions) - 1:
                    lo = len2
                    hi = len2 + STEP_PAD
                else:
                    lo, hi = INTERIOR_STEP_RANGE
                steps.append(random.randint(lo, hi))

        seq = [(0, 0)]
        r = 0
        c = 0
        for (dr, dc), step_count in zip(directions, steps):
            for _ in range(step_count):
                r += dr
                c += dc
                seq.append((r, c))

        corridor = seq[len1:-len2]
        if len(corridor) < MIN_CORRIDOR_CELLS:
            continue
        if _count_turns(seq) != turns:
            continue
        if _is_simple_path(seq):
            return seq
    return None


def _normalize_path(seq: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in seq)
    min_c = min(c for _, c in seq)
    shifted = [(r - min_r, c - min_c) for r, c in seq]
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1
    return shifted, height, width


def _can_place(seq: list[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    return all(0 <= r < h and 0 <= c < w and not blocked[r, c] for r, c in seq)


def _stamp_blocked(seq: list[tuple[int, int]], blocked: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in seq:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    parities = [0, 1]
    while len(parities) < pair_count:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)
    colors = random.sample(SEGMENT_COLORS, pair_count)

    for color, parity in zip(colors, parities):
        placed = False
        turn_count = random.choice(EVEN_TURNS if parity == 0 else ODD_TURNS)
        for _ in range(PLACE_ATTEMPTS):
            len1 = random.randint(*SEGMENT_LEN_RANGE)
            len2 = random.randint(*SEGMENT_LEN_RANGE)
            local = _build_local_path(turn_count, len1, len2)
            if local is None:
                continue
            if random.random() < 0.5:
                local = list(reversed(local))
            local, path_h, path_w = _normalize_path(local)
            max_r0 = h - path_h - 1
            max_c0 = w - path_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue
            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            seq = [(r + r0, c + c0) for r, c in local]
            if not _can_place(seq, blocked):
                continue

            segment1 = seq[:len1]
            segment2 = seq[-len2:]
            corridor = seq[len1:-len2]
            if len(corridor) < MIN_CORRIDOR_CELLS:
                continue

            for r, c in segment1 + segment2:
                grid[r, c] = color
            for r, c in corridor:
                grid[r, c] = GREY

            final_color = RED if parity == 1 else GREEN
            for r, c in seq:
                expected[r, c] = final_color
            _stamp_blocked(seq, blocked)
            placed = True
            break
        if not placed:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
