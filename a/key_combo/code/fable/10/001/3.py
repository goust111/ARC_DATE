from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

ARROW_PATTERNS = {
    "up": np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    "down": np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    "left": np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    "right": np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
}

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

TIP_OFFSETS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}

NON_ARROW_COLORS = {BLACK, GREY}
GEN_ARROW_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GEN_RETRIES = 400

ARROW_PARSE_DATA = {}
for _name, _pattern in ARROW_PATTERNS.items():
    _coords = np.argwhere(_pattern)
    _r0, _c0 = _coords.min(axis=0)
    _r1, _c1 = _coords.max(axis=0)
    ARROW_PARSE_DATA[_name] = {
        "pattern": _pattern[_r0 : _r1 + 1, _c0 : _c1 + 1],
        "tip": (TIP_OFFSETS[_name][0] - _r0, TIP_OFFSETS[_name][1] - _c0),
    }


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    accounted = np.zeros_like(grid, dtype=bool)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in NON_ARROW_COLORS):
        for comp in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(comp)
            block = (grid[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)
            matches = [name for name, data in ARROW_PARSE_DATA.items() if np.array_equal(block, data["pattern"])]
            assert len(matches) == 1, "failed to decode a colored 3x3 arrow glyph"
            direction = matches[0]
            tip_r = r0 + ARROW_PARSE_DATA[direction]["tip"][0]
            tip_c = c0 + ARROW_PARSE_DATA[direction]["tip"][1]
            arrow_cells = set(comp)
            for cell_r, cell_c in comp:
                accounted[cell_r, cell_c] = True
            arrows.append(
                {
                    "color": color,
                    "direction": direction,
                    "delta": DIRS[direction],
                    "tip": (tip_r, tip_c),
                    "bbox": (r0, c0, r1, c1),
                    "cells": arrow_cells,
                }
            )

    non_special = (grid != BLACK) & (grid != GREY)
    assert np.array_equal(accounted, non_special), "all non-black non-grey cells must belong to arrows"
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1], item["color"]))
    assert arrows, "expected at least one arrow"
    return arrows


def _extract_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirrors: dict[tuple[int, int], str] = {}
    for comp in _components(grid == GREY, connectivity=8):
        assert len(comp) >= 2, "mirror segments must have length at least 2"
        by_row = sorted(comp)
        diffs = {(r - c) for r, c in comp}
        sums = {(r + c) for r, c in comp}
        if len(diffs) == 1:
            orient = "\\"
            expected = [(by_row[0][0] + i, by_row[0][1] + i) for i in range(len(by_row))]
            assert by_row == expected, "backslash mirror must be a contiguous main-diagonal segment"
        elif len(sums) == 1:
            orient = "/"
            expected = [(by_row[0][0] + i, by_row[0][1] - i) for i in range(len(by_row))]
            assert by_row == expected, "slash mirror must be a contiguous anti-diagonal segment"
        else:
            raise AssertionError("grey components must each be one diagonal segment")
        for cell in comp:
            assert cell not in mirrors
            mirrors[cell] = orient
    return mirrors


def _reflect(delta: tuple[int, int], orient: str) -> tuple[int, int]:
    if orient == "\\":
        return delta[1], delta[0]
    if orient == "/":
        return -delta[1], -delta[0]
    raise AssertionError(f"unknown mirror orientation: {orient}")


def _paint_beam_cell(out: np.ndarray, r: int, c: int, color: int) -> None:
    current = int(out[r, c])
    if current == BLACK:
        out[r, c] = color
    elif current == color or current == RED:
        return
    else:
        out[r, c] = RED


def _simulate(
    grid: np.ndarray,
    arrows: list[dict],
    mirrors: dict[tuple[int, int], str],
    *,
    stop_on_arrows: bool = True,
    use_mirrors: bool = True,
    use_red_crossing: bool = True,
    return_info: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    arrow_cells = {cell for arrow in arrows for cell in arrow["cells"]}
    info_list: list[dict] = []

    for arrow in arrows:
        color = int(arrow["color"])
        dr, dc = arrow["delta"]
        current = arrow["tip"]
        seen_states: set[tuple[int, int, int, int]] = set()
        painted: list[tuple[int, int]] = []
        mirror_orients: list[str] = []
        termination = "edge"

        while True:
            state = (current[0], current[1], dr, dc)
            assert state not in seen_states, "beam loop detected"
            seen_states.add(state)

            nr = current[0] + dr
            nc = current[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                termination = "edge"
                break
            if stop_on_arrows and (nr, nc) in arrow_cells:
                termination = "arrow"
                break
            if use_mirrors and (nr, nc) in mirrors:
                current = (nr, nc)
                mirror_orients.append(mirrors[(nr, nc)])
                dr, dc = _reflect((dr, dc), mirrors[(nr, nc)])
                continue
            if int(grid[nr, nc]) != BLACK and (nr, nc) not in arrow_cells:
                raise AssertionError("beams may only traverse black cells or mirrors")
            if use_red_crossing:
                _paint_beam_cell(out, nr, nc, color)
            else:
                current_color = int(out[nr, nc])
                if current_color == BLACK:
                    out[nr, nc] = color
            current = (nr, nc)
            painted.append(current)

        info_list.append(
            {
                "color": color,
                "painted": painted,
                "mirrors": mirror_orients,
                "termination": termination,
            }
        )

    if return_info:
        return out, info_list
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored arrow beam through grey diagonal mirrors and mark cross-color intersections red.

    Input structure:
    - The background is black.
    - The non-grey foreground objects are 3x3 monochrome arrow glyphs pointing up, down, left, or right.
    - The grey foreground objects are disjoint diagonal line segments, each following either "\" or "/".

    Rule:
    - Each arrow emits a beam of its own color, starting one cell in front of the arrow tip and moving cell by cell in the arrow's pointing direction.
    - When a beam enters a grey mirror cell, that mirror cell stays grey and the beam turns 90 degrees according to the mirror orientation before continuing.
    - A beam stops when its next step would leave the grid or enter any arrow glyph cell.
    - Every traversed black cell is painted with the beam color, and any cell visited by beams of different colors becomes red while mirrors and arrows remain unchanged.

    Key insight:
    - The grey diagonals are not drawable targets; they are fixed reflectors, so the output is the superposition of all reflected beam paths plus the red cells where differently colored paths meet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    arrows = _extract_arrows(g)
    mirrors = _extract_mirrors(g)
    return _simulate(g, arrows, mirrors)


def _place_arrow(grid: np.ndarray, top: int, left: int, direction: str, color: int) -> None:
    pattern = ARROW_PATTERNS[direction]
    block = grid[top : top + 3, left : left + 3]
    assert block.shape == (3, 3)
    assert np.all(block == BLACK)
    block[pattern.astype(bool)] = color


def _diag_segment_cells(
    hit: tuple[int, int],
    orient: str,
    length: int,
    hit_index: int,
) -> list[tuple[int, int]]:
    hr, hc = hit
    if orient == "\\":
        start_r = hr - hit_index
        start_c = hc - hit_index
        return [(start_r + i, start_c + i) for i in range(length)]
    start_r = hr - hit_index
    start_c = hc + hit_index
    return [(start_r + i, start_c - i) for i in range(length)]


def _touching(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            if max(abs(ra - rb), abs(ca - cb)) <= 1:
                return True
    return False


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.choice([False, True]):
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _build_template() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(GEN_ARROW_COLORS, 3)

    len1 = random.randint(2, 4)
    len2 = random.randint(2, 4)
    hit_idx1 = random.randrange(len1)
    hit_idx2 = random.randrange(len2)

    r1_low = max(3, hit_idx1)
    r1_high = min(6, h - (len1 - hit_idx1))
    if r1_low > r1_high:
        return None
    r1 = random.randint(r1_low, r1_high)

    c1_low = 10
    c1_high = w - max(6, hit_idx2 + 2)
    if c1_low > c1_high:
        return None
    c1 = random.randint(c1_low, c1_high)

    r2_low = r1 + 6
    r2_high = min(h - 6, h - (len2 - hit_idx2))
    if r2_low > r2_high:
        return None
    r2 = random.randint(r2_low, r2_high)

    c_a = random.randint(1, 3)
    c_c_high = c1 - 7
    if c_c_high < 2:
        return None
    c_c = random.randint(2, c_c_high)

    r_cross_low = r1 + 2
    r_cross_high = r2 - 2
    if r_cross_low > r_cross_high:
        return None
    r_cross = random.randint(r_cross_low, r_cross_high)

    c_b_high = c1 - 8
    if c_b_high < 1:
        return None
    c_b = random.randint(1, c_b_high)

    arrow_specs = [
        (r1 - 1, c_a, "right", colors[0]),
        (r_cross - 1, c_b, "right", colors[1]),
        (r2 - 1, c_c, "down", colors[2]),
    ]
    arrow_sets: list[set[tuple[int, int]]] = []
    for top, left, direction, _color in arrow_specs:
        if not (0 <= top <= h - 3 and 0 <= left <= w - 3):
            return None
        cells = {
            (top + r, left + c)
            for r in range(3)
            for c in range(3)
            if ARROW_PATTERNS[direction][r, c]
        }
        if any(max(abs(r - top), abs(c - left)) < 0 for r, c in cells):
            return None
        arrow_sets.append(cells)

    for idx, cells_a in enumerate(arrow_sets):
        for cells_b in arrow_sets[idx + 1 :]:
            if _touching(cells_a, cells_b):
                return None

    mirror1 = set(_diag_segment_cells((r1, c1), "\\", len1, hit_idx1))
    mirror2 = set(_diag_segment_cells((r2, c1), "/", len2, hit_idx2))
    if _touching(mirror1, mirror2):
        return None
    for mirror_cells in (mirror1, mirror2):
        for cells in arrow_sets:
            if _touching(mirror_cells, cells):
                return None

    for top, left, direction, color in arrow_specs:
        _place_arrow(grid, top, left, direction, color)
    for mr, mc in mirror1 | mirror2:
        if not (0 <= mr < h and 0 <= mc < w):
            return None
        if grid[mr, mc] != BLACK:
            return None
        grid[mr, mc] = GREY

    return _random_symmetry(grid)


def _validate_generation(grid: np.ndarray) -> bool:
    try:
        arrows = _extract_arrows(grid)
        mirrors = _extract_mirrors(grid)
        out, info = _simulate(grid, arrows, mirrors, return_info=True)
        no_red = _simulate(grid, arrows, mirrors, use_red_crossing=False)
    except AssertionError:
        return False

    arrow_mask = np.zeros_like(grid, dtype=bool)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            arrow_mask[r, c] = True

    mirrorless = grid.copy()
    for r, c in mirrors:
        mirrorless[r, c] = BLACK
    try:
        no_mirror = _simulate(mirrorless, arrows, {})
    except AssertionError:
        return False
    for r, c in mirrors:
        no_mirror[r, c] = GREY

    arrowless = grid.copy()
    arrowless[arrow_mask] = BLACK
    try:
        no_arrow_stop = _simulate(arrowless, arrows, mirrors, stop_on_arrows=False)
    except AssertionError:
        return False
    no_arrow_stop[arrow_mask] = grid[arrow_mask]

    if len(arrows) != 3 or len(mirrors) < 4:
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, no_mirror):
        return False
    if np.array_equal(out, no_arrow_stop):
        return False
    if np.array_equal(out, no_red):
        return False
    if int(np.count_nonzero(out == RED)) < 1:
        return False

    used_orients = {orient for item in info for orient in item["mirrors"]}
    if used_orients != {"\\", "/"}:
        return False
    if not any(item["termination"] == "arrow" for item in info):
        return False
    if not any(item["termination"] == "edge" for item in info):
        return False
    if any(len(item["painted"]) < 3 for item in info):
        return False

    total_painted = sum(len(item["painted"]) for item in info)
    if total_painted < 18 or total_painted > 80:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_template()
        if grid is None:
            continue
        if _validate_generation(grid):
            return grid
    raise RuntimeError("failed to generate a valid arrow-mirror ARC input")
