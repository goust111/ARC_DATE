from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

UP, RIGHT, DOWN, LEFT = range(4)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_CCW = "L"
TURN_CW = "R"

LEGEND_H = 2
LEGEND_SLOT_STARTS = (1, 4, 7)
OBSTACLE_COLORS = (RED, GREEN, YELLOW)
FILL_COLORS = (BLUE, RED, GREEN, YELLOW, PINK)
GEN_ATTEMPTS = 8000


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rot(direction: int, turn: str) -> int:
    return (direction - 1) % 4 if turn == TURN_CCW else (direction + 1) % 4


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_legend(legend: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for start in LEGEND_SLOT_STARTS:
        if start + 1 >= legend.shape[1]:
            continue
        slot = legend[:, start : start + 2]
        present = [int(v) for v in np.unique(slot) if int(v) != GREY]
        if not present:
            continue
        assert len(present) == 1, "each legend slot must encode exactly one color"
        color = present[0]
        coords = np.argwhere(slot == color)
        assert len(coords) == 2, "legend bars must be 2 pixels tall"
        assert {int(r) for r, _c in coords} == {0, 1}, "legend bar must span both legend rows"
        side = int(coords[0, 1])
        assert all(int(c) == side for _r, c in coords), "legend bar must be vertical"
        mapping[color] = TURN_CCW if side == 0 else TURN_CW
    assert 2 <= len(mapping) <= 3, "legend must encode 2-3 obstacle colors"
    return mapping


def _find_emitter(base: np.ndarray) -> tuple[int, int]:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(base == TEAL)]
    assert len(pts) == 9, "input must contain only the 3x3 teal emitter"
    r0, c0, r1, c1 = _bbox(pts)
    assert (r1 - r0, c1 - c0) == (2, 2), "teal pixels must form a 3x3 square"
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            assert int(base[r, c]) == TEAL
    return r0, c0


def _obstacle_blocks(base: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    blocks: list[dict] = []
    cell_to_block: dict[tuple[int, int], int] = {}
    for color in OBSTACLE_COLORS:
        for comp in _components(base == color):
            r0, c0, r1, c1 = _bbox(comp)
            assert len(comp) == 2, "each obstacle block must be a domino"
            assert (r1 - r0, c1 - c0) in {(0, 1), (1, 0)}, "obstacles must be 1x2 or 2x1"
            block_id = len(blocks)
            blocks.append({"color": color, "cells": tuple(sorted(comp))})
            for cell in comp:
                cell_to_block[cell] = block_id
    assert 3 <= len(blocks) <= 5, "main grid must contain 3-5 obstacle blocks"
    return blocks, cell_to_block


def _trace_rays(base: np.ndarray, turn_map: dict[int, str]) -> tuple[np.ndarray, dict]:
    out = base.copy()
    emitter_r0, emitter_c0 = _find_emitter(base)
    emitter_r1, emitter_c1 = emitter_r0 + 2, emitter_c0 + 2
    blocks, cell_to_block = _obstacle_blocks(base)

    ray_specs = [
        (emitter_r0, emitter_c0 + 1, UP),
        (emitter_r1, emitter_c0 + 1, DOWN),
        (emitter_r0 + 1, emitter_c0, LEFT),
        (emitter_r0 + 1, emitter_c1, RIGHT),
    ]

    trace = {
        "rays": [],
        "used_block_ids": set(),
        "used_colors": set(),
        "total_hits": 0,
        "blocks": blocks,
    }

    for start_r, start_c, start_dir in ray_specs:
        r, c, direction = start_r, start_c, start_dir
        seen_states: set[tuple[int, int, int]] = set()
        painted = 0
        hits: list[int] = []
        termination = "edge"
        while True:
            state = (r, c, direction)
            assert state not in seen_states, "ray path must not cycle"
            seen_states.add(state)

            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not _in_bounds(out.shape, nr, nc):
                termination = "edge"
                break

            value = int(out[nr, nc])
            if value == BLACK:
                out[nr, nc] = TEAL
                r, c = nr, nc
                painted += 1
                continue

            if value == TEAL:
                termination = "path"
                break

            assert value in turn_map, "every obstacle color must be present in the legend"
            block_id = cell_to_block[(nr, nc)]
            trace["used_block_ids"].add(block_id)
            trace["used_colors"].add(value)
            trace["total_hits"] += 1
            hits.append(block_id)
            direction = _rot(direction, turn_map[value])

        trace["rays"].append(
            {
                "start": (start_r, start_c, start_dir),
                "painted": painted,
                "hits": tuple(hits),
                "termination": termination,
            }
        )

    return out, trace


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    regions: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == grid.shape[0] - 1 or cc == 0 or cc == grid.shape[1] - 1:
                    touches_edge = True
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                r0, c0, r1, c1 = _bbox(coords)
                regions.append(
                    {
                        "coords": coords,
                        "area": len(coords),
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    return regions


def _fill_regions(traced: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = traced.copy()
    ordered = sorted(regions, key=lambda region: region["area"])
    assert len(ordered) <= len(FILL_COLORS), "too many enclosed regions for the fixed fill palette"
    for fill_color, region in zip(FILL_COLORS, ordered):
        for r, c in region["coords"]:
            out[r, c] = fill_color
    return out


def _solve_base(base: np.ndarray, turn_map: dict[int, str]) -> tuple[np.ndarray, dict, list[dict]]:
    traced, trace = _trace_rays(base, turn_map)
    regions = _enclosed_regions(traced)
    areas = [region["area"] for region in regions]
    assert len(areas) == len(set(areas)), "enclosed-region areas must be distinct"
    return _fill_regions(traced, regions), trace, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace teal rays that turn at legend-coded obstacle colors, then fill the newly enclosed black regions by increasing area.

    Input structure:
    - The top 2 rows are a grey legend band containing 2-3 colored vertical bars; each bar color matches one obstacle color and its left/right placement within a 2-column slot encodes the turn direction.
    - Below the legend is a black main grid with one centered 3x3 teal emitter square and 3-5 colored obstacle dominoes in red, green, and/or yellow.
    - No teal ray path is present in the input; only the emitter and the obstacle blocks are pre-drawn.

    Rule:
    - Remove the top legend band in the output.
    - Decode each legend bar: left position means a 90-degree counterclockwise turn, right position means a 90-degree clockwise turn.
    - Emit one 1-pixel-wide teal ray from the middle cell of each side of the 3x3 teal square; rays advance over black cells, stop at edges or existing teal path, and turn in place when the next cell is an obstacle color.
    - After all rays terminate, find every 4-connected black region that cannot reach the outer border without crossing a non-black cell, sort those enclosed regions by area, and fill them with colors [blue, red, green, yellow, pink] from smallest to largest area.

    Key insight:
    - The legend does not label the target regions directly; it controls the turning behavior of the rays, and those ray-built walls determine which enclosed regions exist and what their relative areas are.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > LEGEND_H
    legend = g[:LEGEND_H, :]
    base = g[LEGEND_H:, :].copy()
    turn_map = _parse_legend(legend)
    out, _trace, _regions = _solve_base(base, turn_map)
    return out


def _legend_rows(width: int, turn_map: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, width), GREY, dtype=int)
    colors = list(turn_map)
    random.shuffle(colors)
    assert len(colors) <= len(LEGEND_SLOT_STARTS)
    for color, start in zip(colors, LEGEND_SLOT_STARTS):
        col = start if turn_map[color] == TURN_CCW else start + 1
        legend[:, col] = color
    return legend


def _expand_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _random_turn_map(colors: tuple[int, ...]) -> dict[int, str]:
    if len(colors) == 2:
        signs = [TURN_CCW, TURN_CW]
        random.shuffle(signs)
        return {color: sign for color, sign in zip(colors, signs)}
    signs = [TURN_CCW, TURN_CW, random.choice([TURN_CCW, TURN_CW])]
    random.shuffle(signs)
    return {color: sign for color, sign in zip(colors, signs)}


def _candidate_turn_maps(colors: tuple[int, ...]) -> list[dict[int, str]]:
    maps: list[dict[int, str]] = []
    if len(colors) == 2:
        maps.append({colors[0]: TURN_CCW, colors[1]: TURN_CW})
        maps.append({colors[0]: TURN_CW, colors[1]: TURN_CCW})
    else:
        for signs in (
            (TURN_CCW, TURN_CCW, TURN_CW),
            (TURN_CCW, TURN_CW, TURN_CCW),
            (TURN_CW, TURN_CCW, TURN_CCW),
            (TURN_CW, TURN_CW, TURN_CCW),
            (TURN_CW, TURN_CCW, TURN_CW),
            (TURN_CCW, TURN_CW, TURN_CW),
        ):
            maps.append({color: sign for color, sign in zip(colors, signs)})
    random.shuffle(maps)
    return maps


def _place_random_blocks(height: int, width: int, emitter_r0: int, emitter_c0: int) -> list[dict] | None:
    block_count = random.choices([4, 5], weights=[1, 3])[0]
    color_count = random.choice([2, 3])
    palette = random.sample(OBSTACLE_COLORS, color_count)
    block_colors = list(palette)
    while len(block_colors) < block_count:
        block_colors.append(random.choice(palette))
    random.shuffle(block_colors)

    forbidden = {(r, c) for r in range(emitter_r0, emitter_r0 + 3) for c in range(emitter_c0, emitter_c0 + 3)}
    blocks: list[dict] = []

    for color in block_colors:
        placed = False
        for _ in range(300):
            band = random.choice(["top", "bottom", "left", "right", "free"])
            if band == "top":
                orientation = "h" if random.random() < 0.7 else "v"
                r = random.randint(1, max(1, emitter_r0 + 1))
                c = random.randint(max(1, emitter_c0 - 7), min(width - 3, emitter_c0 + 5))
            elif band == "bottom":
                orientation = "h" if random.random() < 0.7 else "v"
                r = random.randint(min(height - 3, emitter_r0 + 1), height - 3)
                c = random.randint(max(1, emitter_c0 - 7), min(width - 3, emitter_c0 + 5))
            elif band == "left":
                orientation = "v" if random.random() < 0.7 else "h"
                r = random.randint(max(1, emitter_r0 - 7), min(height - 3, emitter_r0 + 5))
                c = random.randint(1, max(1, emitter_c0))
            elif band == "right":
                orientation = "v" if random.random() < 0.7 else "h"
                r = random.randint(max(1, emitter_r0 - 7), min(height - 3, emitter_r0 + 5))
                c = random.randint(min(width - 3, emitter_c0 + 2), width - 3)
            else:
                orientation = random.choice(["h", "v"])
                r = random.randint(1, height - 3)
                c = random.randint(1, width - 3)
            cells = [(r, c), (r, c + 1)] if orientation == "h" else [(r, c), (r + 1, c)]
            if any(cell in forbidden for cell in cells):
                continue
            blocks.append({"color": color, "cells": tuple(cells)})
            forbidden.update(cells)
            placed = True
            break
        if not placed:
            return None
    return blocks


def _base_grid(height: int, width: int, emitter_r0: int, emitter_c0: int, blocks: list[dict]) -> np.ndarray:
    base = np.zeros((height, width), dtype=int)
    base[emitter_r0 : emitter_r0 + 3, emitter_c0 : emitter_c0 + 3] = TEAL
    for block in blocks:
        color = block["color"]
        for r, c in block["cells"]:
            base[r, c] = color
    return base


def _spatial_order(regions: list[dict]) -> list[tuple[int, int, int]]:
    ordered = []
    for idx, region in enumerate(regions):
        r0, c0, _r1, _c1 = region["bbox"]
        ordered.append((r0, c0, idx))
    return sorted(ordered)


def _alternate_output(base: np.ndarray, turn_map: dict[int, str]) -> np.ndarray | None:
    try:
        out, _trace, _regions = _solve_base(base, turn_map)
        return out
    except AssertionError:
        return None


def _validate_candidate(base: np.ndarray, turn_map: dict[int, str]) -> np.ndarray | None:
    try:
        expected, trace, regions = _solve_base(base, turn_map)
    except AssertionError:
        return None

    if not (2 <= len(regions) <= 5):
        return None

    areas = [region["area"] for region in regions]
    if len(areas) != len(set(areas)):
        return None

    used_block_ids = sorted(trace["used_block_ids"])
    if len(used_block_ids) < 2:
        return None

    used_turns = {turn_map[int(trace["blocks"][block_id]["color"])] for block_id in used_block_ids}
    if used_turns != {TURN_CCW, TURN_CW}:
        return None

    if sum(bool(ray["hits"]) for ray in trace["rays"]) < 2:
        return None

    terminations = {ray["termination"] for ray in trace["rays"]}
    if "path" not in terminations or "edge" not in terminations:
        return None

    traced, _trace_again = _trace_rays(base, turn_map)
    descending = traced.copy()
    for fill_color, region in zip(FILL_COLORS, sorted(regions, key=lambda region: region["area"], reverse=True)):
        for r, c in region["coords"]:
            descending[r, c] = fill_color
    if np.array_equal(descending, expected):
        return None

    for color in sorted(trace["used_colors"]):
        flipped = dict(turn_map)
        flipped[color] = TURN_CW if turn_map[color] == TURN_CCW else TURN_CCW
        altered = _alternate_output(base, flipped)
        if altered is None or np.array_equal(altered, expected):
            return None

    for block_id in used_block_ids:
        variant = base.copy()
        for r, c in trace["blocks"][block_id]["cells"]:
            variant[r, c] = BLACK
        altered = _alternate_output(variant, turn_map)
        if altered is None or np.array_equal(altered, expected):
            return None

    full_input = np.vstack([_legend_rows(base.shape[1], turn_map), base])
    try:
        actual = main(full_input)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return full_input


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 22)
    width = random.randint(18, 24)
    emitter_r0 = random.randint(height // 2 - 2, height // 2 + 1)
    emitter_c0 = random.randint(width // 2 - 2, width // 2 + 1)

    blocks = _place_random_blocks(height, width, emitter_r0, emitter_c0)
    if blocks is None:
        return None

    colors = tuple(sorted({block["color"] for block in blocks}))
    if len(colors) < 2:
        return None
    base = _base_grid(height, width, emitter_r0, emitter_c0, blocks)
    for turn_map in _candidate_turn_maps(colors):
        grid = _validate_candidate(base, turn_map)
        if grid is not None:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
