from __future__ import annotations

import random
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PORTAL_COLORS = [BLUE, RED, GREEN, MAGENTA, CYAN]
NON_PORTAL_COLORS = {BLACK, GREY, YELLOW, ORANGE, MAROON}
GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(sorted(coords))
    return components


def _portal_shape(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    assert len(coords) == 2, "portal component must have area 2"
    (r0, c0), (r1, c1) = coords
    if r0 == r1 and abs(c0 - c1) == 1:
        return sorted(coords, key=lambda rc: rc[1]), "horizontal"
    if c0 == c1 and abs(r0 - r1) == 1:
        return sorted(coords, key=lambda rc: rc[0]), "vertical"
    raise AssertionError("portal must be a 1x2 or 2x1 segment")


def _parse_portals(grid: np.ndarray) -> tuple[dict[int, dict], dict[tuple[int, int], tuple[int, tuple[int, int]]]]:
    portals: dict[int, dict] = {}
    lookup: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    portal_colors = [int(color) for color in np.unique(grid) if int(color) not in NON_PORTAL_COLORS]
    assert portal_colors, "expected at least one portal color"

    for color in portal_colors:
        comps = _color_components(grid, color)
        assert len(comps) == 2, "each portal color must appear as exactly two components"
        parsed = []
        for comp in comps:
            cells, orientation = _portal_shape(comp)
            parsed.append((cells, orientation))

        orientations = {orientation for _cells, orientation in parsed}
        assert len(orientations) == 1, "paired portals must have parallel orientation"
        orientation = parsed[0][1]
        segments = [cells for cells, _orientation in parsed]
        segments.sort(key=lambda segment: segment[0])

        exit_map: dict[tuple[int, int], tuple[int, int]] = {}
        for index in range(2):
            a = segments[0][index]
            b = segments[1][index]
            exit_map[a] = b
            exit_map[b] = a
            lookup[a] = (color, b)
            lookup[b] = (color, a)

        portals[color] = {
            "color": color,
            "orientation": orientation,
            "segments": segments,
            "cells": segments[0] + segments[1],
            "exit_map": exit_map,
            "used": False,
        }

    return portals, lookup


def _parse_movers(grid: np.ndarray) -> list[dict]:
    movers: list[dict] = []
    used_oranges: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid == YELLOW):
        rr = int(r)
        cc = int(c)
        directions = []
        orange_cells = []
        for dr, dc in CARDINALS:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == ORANGE:
                directions.append((dr, dc))
                orange_cells.append((nr, nc))
        assert len(directions) == 1, "each yellow object must have exactly one orange direction pixel"
        movers.append({"start": (rr, cc), "direction": directions[0]})
        used_oranges.add(orange_cells[0])

    orange_pixels = {tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)}
    assert orange_pixels == used_oranges, "every orange pixel must belong to exactly one mover"
    movers.sort(key=lambda item: item["start"])
    assert movers, "expected at least one moving object"
    return movers


def _simulate_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    portals, portal_lookup = _parse_portals(g)
    movers = _parse_movers(g)

    env = g.copy()
    env[(env == YELLOW) | (env == ORANGE)] = BLACK

    traces: list[dict] = []
    final_positions: list[tuple[int, int]] = []
    for mover in movers:
        r, c = mover["start"]
        dr, dc = mover["direction"]
        used_colors: list[int] = []
        blocked_by_used = False

        while True:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            cell = int(env[nr, nc])
            if cell == BLACK:
                r, c = nr, nc
                continue
            if cell == GREY or cell == MAROON:
                blocked_by_used = cell == MAROON
                break

            portal_info = portal_lookup.get((nr, nc))
            if portal_info is None:
                break

            portal_color, exit_cell = portal_info
            portal = portals[portal_color]
            if portal["used"]:
                blocked_by_used = True
                break

            portal["used"] = True
            used_colors.append(portal_color)
            for pr, pc in portal["cells"]:
                env[pr, pc] = MAROON
            r, c = exit_cell

        final_positions.append((r, c))
        traces.append(
            {
                "start": mover["start"],
                "direction": mover["direction"],
                "end": (r, c),
                "teleports": tuple(used_colors),
                "blocked_by_used": blocked_by_used,
            }
        )

    out = env.copy()
    for r, c in final_positions:
        out[r, c] = YELLOW
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each yellow object through black corridors, teleporting through same-color wall portals until it stops.

    Input structure:
    - The background is black, with grey walls that partition the grid into straight rooms or channels.
    - Each portal is a colored 2-cell gap embedded inside a grey wall, and every portal color appears exactly twice with the same orientation.
    - Each moving object is a single yellow pixel with one adjacent orange pixel indicating its cardinal motion direction.

    Rule:
    - Process movers in top-to-bottom, then left-to-right order, while ignoring the orange markers as terrain.
    - A mover travels straight through black cells; if its next cell is grey, or a previously used maroon portal, it stops on the previous cell.
    - If the mover reaches an unused portal cell, it teleports to the corresponding cell of the paired portal, keeps the same direction, and that whole portal pair turns maroon.

    Key insight:
    - The colored wall gaps are not decorations but paired spatial jumps, so solving the task requires matching portal pairs and simulating each mover step by step with persistent portal state.
    """
    out, _traces = _simulate_grid(np.asarray(grid, dtype=int))
    return out


def _place_mover(grid: np.ndarray, yellow: tuple[int, int], direction: tuple[int, int]) -> bool:
    r, c = yellow
    dr, dc = direction
    or_r, or_c = r + dr, c + dc
    if not _in_bounds(grid.shape, r, c) or not _in_bounds(grid.shape, or_r, or_c):
        return False
    if int(grid[r, c]) != BLACK or int(grid[or_r, or_c]) != BLACK:
        return False
    grid[r, c] = YELLOW
    grid[or_r, or_c] = ORANGE
    return True


def _pick_two_spans(length: int) -> list[int] | None:
    candidates = list(range(2, length - 3))
    random.shuffle(candidates)
    chosen: list[int] = []
    for start in candidates:
        if all(abs(start - other) >= 3 for other in chosen):
            chosen.append(start)
            if len(chosen) == 2:
                return sorted(chosen)
    return None


def _sample_horizontal_bands() -> tuple[int, int, int, int, int, int]:
    while True:
        top = random.randint(3, 6)
        band12 = random.randint(3, 6)
        band23 = random.randint(4, 7)
        band34 = random.randint(3, 6)
        bottom = random.randint(3, 6)
        height = top + band12 + band23 + band34 + bottom + 6
        if height <= 30:
            return top, band12, band23, band34, bottom, height


def _sample_vertical_bands() -> tuple[int, int, int, int, int, int]:
    while True:
        left = random.randint(3, 6)
        band12 = random.randint(3, 6)
        band23 = random.randint(4, 7)
        band34 = random.randint(3, 6)
        right = random.randint(3, 6)
        width = left + band12 + band23 + band34 + right + 6
        if width <= 30:
            return left, band12, band23, band34, right, width


def _validate_case(grid: np.ndarray, expected: dict[tuple[int, int], dict]) -> bool:
    try:
        out, traces = _simulate_grid(grid)
    except AssertionError:
        return False

    trace_map = {trace["start"]: trace for trace in traces}
    if set(trace_map) != set(expected):
        return False

    finals = [trace["end"] for trace in traces]
    if len(set(finals)) != len(finals):
        return False
    if max(out.shape) > 30:
        return False

    for start, need in expected.items():
        trace = trace_map[start]
        if len(trace["teleports"]) != need["teleports"]:
            return False
        if trace["blocked_by_used"] != need["blocked"]:
            return False
    return True


def _generate_horizontal_input() -> np.ndarray | None:
    top, band12, band23, band34, bottom, height = _sample_horizontal_bands()
    width = random.randint(17, 29)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    wall1 = 1 + top
    wall2 = wall1 + 1 + band12
    wall3 = wall2 + 1 + band23
    wall4 = wall3 + 1 + band34
    for row in (wall1, wall2, wall3, wall4):
        grid[row, 1:-1] = GREY

    starts_top = _pick_two_spans(width)
    starts_mid = _pick_two_spans(width)
    if starts_top is None or starts_mid is None:
        return None
    p1_top, p3_top = starts_top
    p1_mid, p3_mid = starts_mid

    extra_mid_choices = list(range(2, width - 3))
    p2_bottom = random.choice(extra_mid_choices)
    colors = random.sample(PORTAL_COLORS, 4 if random.random() < 0.5 else 3)

    grid[wall1, p1_top : p1_top + 2] = colors[0]
    grid[wall2, p1_mid : p1_mid + 2] = colors[0]
    grid[wall3, p1_mid : p1_mid + 2] = colors[1]
    grid[wall4, p2_bottom : p2_bottom + 2] = colors[1]
    grid[wall1, p3_top : p3_top + 2] = colors[2]
    grid[wall2, p3_mid : p3_mid + 2] = colors[2]

    all_portal_cols = {
        p1_top,
        p1_top + 1,
        p1_mid,
        p1_mid + 1,
        p2_bottom,
        p2_bottom + 1,
        p3_top,
        p3_top + 1,
        p3_mid,
        p3_mid + 1,
    }
    if len(colors) == 4:
        extra_top_choices = [c for c in range(2, width - 3) if all(abs(c - x) >= 3 for x in (p1_mid, p3_mid))]
        extra_bottom_choices = [c for c in range(2, width - 3) if all(abs(c - x) >= 3 for x in (p2_bottom,))]
        if not extra_top_choices or not extra_bottom_choices:
            return None
        p4_mid = random.choice(extra_top_choices)
        p4_bottom = random.choice(extra_bottom_choices)
        grid[wall3, p4_mid : p4_mid + 2] = colors[3]
        grid[wall4, p4_bottom : p4_bottom + 2] = colors[3]
        all_portal_cols.update({p4_mid, p4_mid + 1, p4_bottom, p4_bottom + 1})

    expected: dict[tuple[int, int], dict] = {}

    chain_offset = random.randint(0, 1)
    chain_row = random.randint(1, wall1 - 2)
    chain_start = (chain_row, p1_top + chain_offset)
    if not _place_mover(grid, chain_start, (1, 0)):
        return None
    expected[chain_start] = {"teleports": 2, "blocked": False}

    once_offset = random.randint(0, 1)
    once_row = random.randint(wall2 + 2, wall3 - 1)
    once_start = (once_row, p3_mid + once_offset)
    if not _place_mover(grid, once_start, (-1, 0)):
        return None
    expected[once_start] = {"teleports": 1, "blocked": False}

    safe_cols = [c for c in range(1, width - 1) if c not in all_portal_cols]
    if not safe_cols:
        return None
    stop_col = random.choice(safe_cols)
    stop_row = random.randint(wall3 + 1, wall4 - 2)
    stop_start = (stop_row, stop_col)
    stop_direction = random.choice([(1, 0), (-1, 0)])
    if stop_direction == (-1, 0) and stop_row == wall3 + 1:
        stop_row += 1
        stop_start = (stop_row, stop_col)
    if stop_direction == (1, 0) and stop_row == wall4 - 2:
        stop_row -= 1
        stop_start = (stop_row, stop_col)
    if not _place_mover(grid, stop_start, stop_direction):
        return None
    expected[stop_start] = {"teleports": 0, "blocked": False}

    if random.random() < 0.7:
        block_offset = random.randint(0, 1)
        block_row = random.randint(wall4 + 2, height - 2)
        block_start = (block_row, p2_bottom + block_offset)
        if not _place_mover(grid, block_start, (-1, 0)):
            return None
        expected[block_start] = {"teleports": 0, "blocked": True}

    if not _validate_case(grid, expected):
        return None
    return grid


def _generate_vertical_input() -> np.ndarray | None:
    left, band12, band23, band34, right, width = _sample_vertical_bands()
    height = random.randint(17, 29)

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    wall1 = 1 + left
    wall2 = wall1 + 1 + band12
    wall3 = wall2 + 1 + band23
    wall4 = wall3 + 1 + band34
    for col in (wall1, wall2, wall3, wall4):
        grid[1:-1, col] = GREY

    starts_left = _pick_two_spans(height)
    starts_mid = _pick_two_spans(height)
    if starts_left is None or starts_mid is None:
        return None
    p1_left, p3_left = starts_left
    p1_mid, p3_mid = starts_mid

    extra_mid_choices = list(range(2, height - 3))
    p2_right = random.choice(extra_mid_choices)
    colors = random.sample(PORTAL_COLORS, 4 if random.random() < 0.5 else 3)

    grid[p1_left : p1_left + 2, wall1] = colors[0]
    grid[p1_mid : p1_mid + 2, wall2] = colors[0]
    grid[p1_mid : p1_mid + 2, wall3] = colors[1]
    grid[p2_right : p2_right + 2, wall4] = colors[1]
    grid[p3_left : p3_left + 2, wall1] = colors[2]
    grid[p3_mid : p3_mid + 2, wall2] = colors[2]

    all_portal_rows = {
        p1_left,
        p1_left + 1,
        p1_mid,
        p1_mid + 1,
        p2_right,
        p2_right + 1,
        p3_left,
        p3_left + 1,
        p3_mid,
        p3_mid + 1,
    }
    if len(colors) == 4:
        extra_left_choices = [r for r in range(2, height - 3) if all(abs(r - x) >= 3 for x in (p1_mid, p3_mid))]
        extra_right_choices = [r for r in range(2, height - 3) if all(abs(r - x) >= 3 for x in (p2_right,))]
        if not extra_left_choices or not extra_right_choices:
            return None
        p4_mid = random.choice(extra_left_choices)
        p4_right = random.choice(extra_right_choices)
        grid[p4_mid : p4_mid + 2, wall3] = colors[3]
        grid[p4_right : p4_right + 2, wall4] = colors[3]
        all_portal_rows.update({p4_mid, p4_mid + 1, p4_right, p4_right + 1})

    expected: dict[tuple[int, int], dict] = {}

    chain_offset = random.randint(0, 1)
    chain_col = random.randint(1, wall1 - 2)
    chain_start = (p1_left + chain_offset, chain_col)
    if not _place_mover(grid, chain_start, (0, 1)):
        return None
    expected[chain_start] = {"teleports": 2, "blocked": False}

    once_offset = random.randint(0, 1)
    once_col = random.randint(wall2 + 2, wall3 - 1)
    once_start = (p3_mid + once_offset, once_col)
    if not _place_mover(grid, once_start, (0, -1)):
        return None
    expected[once_start] = {"teleports": 1, "blocked": False}

    safe_rows = [r for r in range(1, height - 1) if r not in all_portal_rows]
    if not safe_rows:
        return None
    stop_row = random.choice(safe_rows)
    stop_col = random.randint(wall3 + 1, wall4 - 2)
    stop_start = (stop_row, stop_col)
    stop_direction = random.choice([(0, 1), (0, -1)])
    if stop_direction == (0, -1) and stop_col == wall3 + 1:
        stop_col += 1
        stop_start = (stop_row, stop_col)
    if stop_direction == (0, 1) and stop_col == wall4 - 2:
        stop_col -= 1
        stop_start = (stop_row, stop_col)
    if not _place_mover(grid, stop_start, stop_direction):
        return None
    expected[stop_start] = {"teleports": 0, "blocked": False}

    if random.random() < 0.7:
        block_offset = random.randint(0, 1)
        block_col = random.randint(wall4 + 2, width - 2)
        block_start = (p2_right + block_offset, block_col)
        if not _place_mover(grid, block_start, (0, -1)):
            return None
        expected[block_start] = {"teleports": 0, "blocked": True}

    if not _validate_case(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        if random.random() < 0.5:
            grid = _generate_horizontal_input()
        else:
            grid = _generate_vertical_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal puzzle input")
