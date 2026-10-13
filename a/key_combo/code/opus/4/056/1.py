from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MARKER_COLORS = (BLUE, GREEN, YELLOW)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 600
GRID_SIDE_RANGE = (15, 18)
BLOCK_COUNT_RANGE = (4, 6)
MARKER_DISTANCE_MIN = 2
MARKER_DISTANCE_MAX = 6
CROSSING_PROBABILITY = 0.65


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _find_red_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _components_of_color(grid, RED)
    blocks: list[tuple[int, int]] = []
    for coords in comps:
        assert len(coords) == 4, "each red object must be a 2x2 block"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each red object must have 2x2 bbox"
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                assert int(grid[rr, cc]) == RED, "red bbox must be solid"
        blocks.append((r0, c0))
    assert 4 <= len(blocks) <= 6, "expected 4 to 6 red 2x2 blocks"
    return sorted(blocks)


def _block_pixel_lookup(blocks: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, (r0, c0) in enumerate(blocks):
        for rr in range(r0, r0 + 2):
            for cc in range(c0, c0 + 2):
                lookup[(rr, cc)] = idx
    return lookup


def _visible_landing_for_marker(
    grid: np.ndarray,
    marker: tuple[int, int],
    block_lookup: dict[tuple[int, int], int],
) -> tuple[int, tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    r0, c0 = marker
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIRS4:
        rr, cc = r0 + dr, c0 + dc
        while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
            rr += dr
            cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if int(grid[rr, cc]) != RED:
            continue
        bid = block_lookup.get((rr, cc))
        if bid is None:
            continue
        landing = (rr - dr, cc - dc)
        candidates.append((bid, (dr, dc), landing))
    assert len(candidates) == 1, "each marker must see exactly one red block along one axis"
    return candidates[0]


def _slide_markers(
    grid: np.ndarray,
    blocks: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    block_lookup = _block_pixel_lookup(blocks)
    out = g.copy()
    moved: dict[int, list[tuple[int, int]]] = {color: [] for color in MARKER_COLORS}

    markers = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(np.isin(g, MARKER_COLORS))]
    assert len(markers) == 6, "expected six colored markers"
    for color in MARKER_COLORS:
        assert sum(1 for _r, _c, v in markers if v == color) == 2, "expected two markers of each non-red color"

    landings: list[tuple[int, int, int]] = []
    for r, c, color in markers:
        _bid, _approach, landing = _visible_landing_for_marker(g, (r, c), block_lookup)
        assert landing != (r, c), "marker must actually slide"
        landings.append((landing[0], landing[1], color))
    assert len({(r, c) for r, c, _color in landings}) == 6, "slid marker positions must be distinct"

    out[np.isin(out, MARKER_COLORS)] = BG
    for r, c, color in landings:
        assert int(out[r, c]) == BG, "slid marker must land on empty background"
        out[r, c] = color
        moved[color].append((r, c))
    return out, moved


def _draw_segment(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        for cc in range(min(c0, c1), max(c0, c1) + 1):
            if int(out[r0, cc]) != RED:
                out[r0, cc] = color
        return
    assert c0 == c1, "same-color pair must share a row or a column"
    for rr in range(min(r0, r1), max(r0, r1) + 1):
        if int(out[rr, c0]) != RED:
            out[rr, c0] = color


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        return [(r0, cc) for cc in range(min(c0, c1), max(c0, c1) + 1)]
    assert c0 == c1
    return [(rr, c0) for rr in range(min(r0, r1), max(r0, r1) + 1)]


def _line_output_from_markers(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    markers: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0 in blocks:
        out[r0 : r0 + 2, c0 : c0 + 2] = RED
    for color, coords in markers.items():
        for r, c in coords:
            out[r, c] = color
    horizontal: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    vertical: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for color in MARKER_COLORS:
        pts = sorted(markers[color])
        assert len(pts) == 2, "each color must have exactly two markers"
        a, b = pts
        assert (a[0] == b[0]) ^ (a[1] == b[1]), "same-color pair must align along exactly one axis"
        if a[0] == b[0]:
            horizontal.append((a, b, color))
        else:
            vertical.append((a, b, color))
    for a, b, color in horizontal:
        _draw_segment(out, a, b, color)
    for a, b, color in vertical:
        _draw_segment(out, a, b, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each colored marker to its unique visible red 2x2 block, then connect same-color landing pairs with axis-aligned lines.

    Input structure:
    - The background is black.
    - There are 4 to 6 separated red 2x2 solid blocks.
    - There are exactly six single-pixel markers: two blue, two green, and two yellow.
    - Each marker has exactly one unobstructed horizontal or vertical sightline to a red 2x2 block.

    Rule:
    - For each marker, follow its unique visible axis toward the nearest red block and slide it to the background cell directly adjacent to that block.
    - Clear the original marker positions, then pair the two landing markers of each color.
    - Draw every horizontal same-color segment first and every vertical same-color segment second, keeping red block cells unchanged.

    Key insight:
    - The red 2x2 blocks are magnetic targets: the markers must move first, and only the moved positions determine which same-color line segments exist and where they cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert set(map(int, np.unique(g))) <= {BG, BLUE, RED, GREEN, YELLOW}, "unexpected colors present"
    assert _get_bg(g) == BG, "background must be black"

    blocks = _find_red_blocks(g)
    slid, markers = _slide_markers(g, blocks)
    out = _line_output_from_markers(slid.shape, blocks, markers)
    assert np.array_equal(out[slid == RED], slid[slid == RED]), "red blocks must be preserved"
    return out


def _block_cells(block: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = block
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _block_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    cells_a = _block_cells(a)
    cells_b = _block_cells(b)
    return min(abs(ra - rb) + abs(ca - cb) for ra, ca in cells_a for rb, cb in cells_b)


def _landing_candidates(shape: tuple[int, int], blocks: list[tuple[int, int]]) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    occupied = set().union(*(_block_cells(block) for block in blocks))
    for bid, (r0, c0) in enumerate(blocks):
        specs = (
            (r0 - 1, c0, (1, 0), (-1, 0)),
            (r0 - 1, c0 + 1, (1, 0), (-1, 0)),
            (r0 + 2, c0, (-1, 0), (1, 0)),
            (r0 + 2, c0 + 1, (-1, 0), (1, 0)),
            (r0, c0 - 1, (0, 1), (0, -1)),
            (r0 + 1, c0 - 1, (0, 1), (0, -1)),
            (r0, c0 + 2, (0, -1), (0, 1)),
            (r0 + 1, c0 + 2, (0, -1), (0, 1)),
        )
        for r, c, approach, away in specs:
            if not (0 <= r < h and 0 <= c < w):
                continue
            if (r, c) in occupied:
                continue
            candidates.append(
                {
                    "block": bid,
                    "landing": (r, c),
                    "approach": approach,
                    "away": away,
                }
            )
    return candidates


def _pair_records(shape: tuple[int, int], blocks: list[tuple[int, int]], candidates: list[dict]) -> list[dict]:
    block_cells = set().union(*(_block_cells(block) for block in blocks))
    pairs: list[dict] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            p0 = a["landing"]
            p1 = b["landing"]
            same_row = p0[0] == p1[0]
            same_col = p0[1] == p1[1]
            if same_row == same_col:
                continue
            cells = _segment_cells(p0, p1)
            if len(cells) < 5:
                continue
            interior = cells[1:-1]
            if any(cell in block_cells for cell in interior):
                continue
            orientation = "h" if same_row else "v"
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "orientation": orientation,
                    "cells": cells,
                    "interior": set(interior),
                }
            )
    return pairs


def _pair_selection_ok(selected: list[dict], need_cross: bool, *, final: bool) -> bool:
    used_landings = set()
    endpoints = []
    for pair in selected:
        for endpoint in (pair["a"]["landing"], pair["b"]["landing"]):
            if endpoint in used_landings:
                return False
            used_landings.add(endpoint)
            endpoints.append(endpoint)

    for idx, pair in enumerate(selected):
        other_endpoints = set(endpoints) - {pair["a"]["landing"], pair["b"]["landing"]}
        if pair["interior"] & other_endpoints:
            return False
        for jdx in range(idx + 1, len(selected)):
            other = selected[jdx]
            if pair["orientation"] == other["orientation"]:
                if pair["orientation"] == "h" and pair["a"]["landing"][0] == other["a"]["landing"][0]:
                    return False
                if pair["orientation"] == "v" and pair["a"]["landing"][1] == other["a"]["landing"][1]:
                    return False

    crossings = 0
    for idx, pair in enumerate(selected):
        for jdx in range(idx + 1, len(selected)):
            other = selected[jdx]
            if pair["orientation"] == other["orientation"]:
                continue
            if pair["orientation"] == "h":
                cross = (pair["a"]["landing"][0], other["a"]["landing"][1])
            else:
                cross = (other["a"]["landing"][0], pair["a"]["landing"][1])
            if cross in pair["interior"] and cross in other["interior"]:
                crossings += 1
    if final and need_cross and crossings == 0:
        return False
    if final and not any(pair["orientation"] == "h" for pair in selected):
        return False
    if final and not any(pair["orientation"] == "v" for pair in selected):
        return False
    return True


def _choose_pairs(shape: tuple[int, int], blocks: list[tuple[int, int]], rng: random.Random) -> list[dict] | None:
    candidates = _landing_candidates(shape, blocks)
    pairs = _pair_records(shape, blocks, candidates)
    if not pairs:
        return None

    need_cross = rng.random() < CROSSING_PROBABILITY
    h_target = 2 if rng.random() < 0.5 else 1
    v_target = 3 - h_target
    rng.shuffle(pairs)

    def rec(start: int, selected: list[dict], h_count: int, v_count: int) -> list[dict] | None:
        if len(selected) == 3:
            if h_count == h_target and v_count == v_target and _pair_selection_ok(selected, need_cross, final=True):
                return [dict(pair) for pair in selected]
            return None
        for idx in range(start, len(pairs)):
            pair = pairs[idx]
            next_h = h_count + (pair["orientation"] == "h")
            next_v = v_count + (pair["orientation"] == "v")
            if next_h > h_target or next_v > v_target:
                continue
            selected.append(pair)
            if _pair_selection_ok(selected, False, final=False):
                found = rec(idx + 1, selected, next_h, next_v)
                if found is not None:
                    return found
            selected.pop()
        return None

    return rec(0, [], 0, 0)


def _candidate_marker_positions(shape: tuple[int, int], landing: tuple[int, int], away: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    positions: list[tuple[int, int]] = []
    for dist in range(MARKER_DISTANCE_MIN, MARKER_DISTANCE_MAX + 1):
        rr = landing[0] + away[0] * dist
        cc = landing[1] + away[1] * dist
        if 0 <= rr < h and 0 <= cc < w:
            positions.append((rr, cc))
    return positions


def _build_input_grid(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    assignment: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0 in blocks:
        grid[r0 : r0 + 2, c0 : c0 + 2] = RED
    for item in assignment:
        r, c = item["source"]
        grid[r, c] = item["color"]
    return grid


def _path_clear_of_other_sources(a: dict, b: dict) -> bool:
    sr, sc = a["source"]
    lr, lc = a["landing"]
    dr = 0 if lr == sr else (1 if lr > sr else -1)
    dc = 0 if lc == sc else (1 if lc > sc else -1)
    rr, cc = sr + dr, sc + dc
    while (rr, cc) != (lr, lc):
        if (rr, cc) == b["source"]:
            return False
        rr += dr
        cc += dc
    return True


def _naive_unslid_output(inp: np.ndarray) -> np.ndarray:
    grid = np.asarray(inp, dtype=int)
    blocks = _find_red_blocks(grid)
    markers = {color: [] for color in MARKER_COLORS}
    for r, c in np.argwhere(np.isin(grid, MARKER_COLORS)):
        markers[int(grid[r, c])].append((int(r), int(c)))
    out = np.zeros_like(grid)
    for r0, c0 in blocks:
        out[r0 : r0 + 2, c0 : c0 + 2] = RED
    for color in MARKER_COLORS:
        pts = markers[color]
        for r, c in pts:
            out[r, c] = color
        if len(pts) != 2:
            continue
        a, b = pts
        if a[0] == b[0] or a[1] == b[1]:
            _draw_segment(out, a, b, color)
    return out


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
        blocks = _find_red_blocks(inp)
        slid, landed = _slide_markers(inp, blocks)
    except AssertionError:
        return False

    if not np.array_equal(out, _line_output_from_markers(inp.shape, blocks, landed)):
        return False
    if np.array_equal(inp, out):
        return False
    if np.array_equal(out, _naive_unslid_output(inp)):
        return False

    original = {color: [] for color in MARKER_COLORS}
    for r, c in np.argwhere(np.isin(inp, MARKER_COLORS)):
        original[int(inp[r, c])].append((int(r), int(c)))
    for color in MARKER_COLORS:
        if len(original[color]) != 2 or len(landed[color]) != 2:
            return False
        a0, b0 = original[color]
        if a0[0] == b0[0] or a0[1] == b0[1]:
            return False
        a1, b1 = landed[color]
        if not ((a1[0] == b1[0]) ^ (a1[1] == b1[1])):
            return False
    return True


def _assign_sources(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    pairs: list[dict],
    colors: list[int],
    rng: random.Random,
) -> list[dict] | None:
    entries = []
    for color, pair in zip(colors, pairs):
        for endpoint in (pair["a"], pair["b"]):
            positions = _candidate_marker_positions(shape, endpoint["landing"], endpoint["away"])
            rng.shuffle(positions)
            if not positions:
                return None
            entries.append(
                {
                    "color": color,
                    "landing": endpoint["landing"],
                    "approach": endpoint["approach"],
                    "positions": positions,
                }
            )
    order = sorted(range(len(entries)), key=lambda idx: len(entries[idx]["positions"]))
    assigned: list[dict] = []
    used: set[tuple[int, int]] = set()

    def rec(pos: int) -> list[dict] | None:
        if pos == len(order):
            color_to_sources = {color: [] for color in MARKER_COLORS}
            for item in assigned:
                color_to_sources[item["color"]].append(item["source"])
            for color in MARKER_COLORS:
                if len(color_to_sources[color]) != 2:
                    return None
                a, b = color_to_sources[color]
                if a[0] == b[0] or a[1] == b[1]:
                    return None
            grid = _build_input_grid(shape, blocks, assigned)
            try:
                block_lookup = _block_pixel_lookup(blocks)
                for item in assigned:
                    _bid, _approach, landing = _visible_landing_for_marker(grid, item["source"], block_lookup)
                    if landing != item["landing"]:
                        return None
            except AssertionError:
                return None
            return [dict(item) for item in assigned]

        entry = entries[order[pos]]
        for source in entry["positions"]:
            if source in used:
                continue
            candidate = {
                "color": entry["color"],
                "landing": entry["landing"],
                "source": source,
                "approach": entry["approach"],
            }
            if not all(_path_clear_of_other_sources(candidate, other) and _path_clear_of_other_sources(other, candidate) for other in assigned):
                continue
            assigned.append(
                {
                    "color": candidate["color"],
                    "landing": candidate["landing"],
                    "source": candidate["source"],
                    "approach": candidate["approach"],
                }
            )
            used.add(source)
            result = rec(pos + 1)
            if result is not None:
                return result
            used.remove(source)
            assigned.pop()
        return None

    return rec(0)


def _place_blocks(shape: tuple[int, int], n_blocks: int, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(2, h - 4) for c in range(2, w - 4)]
    rng.shuffle(candidates)
    blocks: list[tuple[int, int]] = []
    for r, c in candidates:
        block = (r, c)
        if all(_block_distance(block, other) >= 4 for other in blocks):
            blocks.append(block)
            if len(blocks) == n_blocks:
                return blocks
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)
    shape = (h, w)
    n_blocks = rng.randint(*BLOCK_COUNT_RANGE)
    blocks = _place_blocks(shape, n_blocks, rng)
    if blocks is None:
        return None

    pairs = _choose_pairs(shape, blocks, rng)
    if pairs is None:
        return None

    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    assignment = _assign_sources(shape, blocks, pairs, colors, rng)
    if assignment is None:
        return None

    inp = _build_input_grid(shape, blocks, assignment)
    if not _semantic_check(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for magnetic sliding wiring task")
