from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GRID_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (14, 18)
SIZE_PAIRS = ((2, 3), (2, 5), (3, 4), (3, 5), (4, 5))
MARKER_COUNT_OPTIONS = (2, 3)
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_filled_square(grid: np.ndarray, color: int) -> tuple[int, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0, f"expected a {color}-colored square"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w, "expected a square parameter object"
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == color), "parameter square must be filled"
    assert len(coords) == side_h * side_w, "square must be the only component of its color"
    return int(side_h), (r0, c0, r1, c1)


def _separator_positions(side: int, limit: int) -> tuple[int, ...]:
    return tuple(range(int(side), int(limit), int(side) + 1))


def _grid_mask(shape: tuple[int, int], side: int) -> np.ndarray:
    h, w = shape
    rows = _separator_positions(side, h)
    cols = _separator_positions(side, w)
    mask = np.zeros((h, w), dtype=bool)
    if rows:
        mask[list(rows), :] = True
    if cols:
        mask[:, list(cols)] = True
    return mask


def _overlay_grid(shape: tuple[int, int], small_side: int, large_side: int) -> np.ndarray:
    red_mask = _grid_mask(shape, large_side)
    blue_mask = _grid_mask(shape, small_side)
    out = np.zeros(shape, dtype=int)
    out[red_mask] = RED
    out[blue_mask] = BLUE
    return out


def _region_from_seed(
    line_grid: np.ndarray, start: tuple[int, int]
) -> tuple[frozenset[tuple[int, int]], bool]:
    sr, sc = start
    assert line_grid[sr, sc] not in (BLUE, RED), "seed must be inside a chamber"

    h, w = line_grid.shape
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    touches_border = sr in (0, h - 1) or sc in (0, w - 1)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if (nr, nc) in seen or line_grid[nr, nc] in (BLUE, RED):
                continue
            seen.add((nr, nc))
            if nr in (0, h - 1) or nc in (0, w - 1):
                touches_border = True
            q.append((nr, nc))
    return frozenset(seen), bool(touches_border)


def _all_regions(line_grid: np.ndarray) -> list[dict]:
    h, w = line_grid.shape
    visited = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or line_grid[r, c] in (BLUE, RED):
                continue
            cells, touches_border = _region_from_seed(line_grid, (r, c))
            for rr, cc in cells:
                visited[rr, cc] = True
            coords = np.array(sorted(cells), dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            regions.append(
                {
                    "cells": cells,
                    "touches_border": bool(touches_border),
                    "area": len(cells),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return regions


def _pick_visual_cell(cells: frozenset[tuple[int, int]]) -> tuple[int, int]:
    coords = sorted(cells)
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    center_r = (min(rs) + max(rs)) / 2.0
    center_c = (min(cs) + max(cs)) / 2.0
    return min(coords, key=lambda rc: (abs(rc[0] - center_r) + abs(rc[1] - center_c), rc))


def _fill_marked_chambers(
    shape: tuple[int, int], small_side: int, large_side: int, markers: list[tuple[int, int]]
) -> np.ndarray:
    out = _overlay_grid(shape, small_side, large_side)
    filled: set[tuple[int, int]] = set()
    for marker in markers:
        region, touches_border = _region_from_seed(out, marker)
        assert not touches_border, "markers must sit in enclosed chambers"
        filled.update(region)
    for r, c in filled:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two square-driven lattices and fill the marker-selected enclosed chambers.

    Input structure:
    - The background is black.
    - A blue filled square and a larger red filled square sit near the top-left and encode two lattice spacings.
    - Two or three green single-pixel markers lie strictly inside future black chambers, away from every future lattice line.

    Rule:
    - Read the blue square side length `s1` and the red square side length `s2`, with `s1 < s2`.
    - On a same-sized black canvas, draw blue full rows/columns at indices `s1, s1+(s1+1), ...` and red full rows/columns at
      `s2, s2+(s2+1), ...`; if both lattices cover a cell, keep blue.
    - Erase the original filled squares, then for each green marker flood-fill its entire enclosed 4-connected chamber with green.

    Key insight:
    - The two top-left squares are only spacing controls; the target chamber boundaries come from the union of both lattices, so
      at least one marked chamber must be interpreted with both scales together rather than with either lattice alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    small_side, blue_box = _find_filled_square(g, BLUE)
    large_side, red_box = _find_filled_square(g, RED)
    assert small_side < large_side, "blue square must be the finer lattice control"
    assert abs(blue_box[0] - red_box[0]) <= 1, "squares should share a row band"
    assert blue_box[1] < red_box[1], "blue square should be left of red square"
    assert blue_box[3] < red_box[1], "parameter squares must not overlap"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert 2 <= len(markers) <= 3, "expected two or three markers"

    return _fill_marked_chambers(g.shape, small_side, large_side, markers)


def _squares_mask(
    shape: tuple[int, int],
    blue_box: tuple[int, int, int, int],
    red_box: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in (blue_box, red_box):
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _placement_options(
    h: int, w: int, small_side: int, large_side: int
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    placements = []
    top_limit_small = min(3, h - small_side)
    top_limit_large = min(4, h - large_side)
    right_limit = max(w // 2 + 2, large_side + 7)

    for blue_r0 in range(0, top_limit_small + 1):
        for delta in (-1, 0, 1):
            red_r0 = blue_r0 + delta
            if red_r0 < 0 or red_r0 > top_limit_large:
                continue
            for blue_c0 in range(0, min(3, w - small_side) + 1):
                for gap in (1, 2, 3):
                    red_c0 = blue_c0 + small_side + gap
                    if red_c0 + large_side > w - 1:
                        continue
                    if red_c0 + large_side > right_limit:
                        continue
                    blue_box = (blue_r0, blue_c0, blue_r0 + small_side - 1, blue_c0 + small_side - 1)
                    red_box = (red_r0, red_c0, red_r0 + large_side - 1, red_c0 + large_side - 1)
                    placements.append((blue_box, red_box))
    return placements


def _candidate_regions(
    shape: tuple[int, int],
    small_side: int,
    large_side: int,
    squares_mask: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    combined = _overlay_grid(shape, small_side, large_side)
    blue_only = _overlay_grid(shape, small_side, small_side + 100)
    red_only = _overlay_grid(shape, large_side, large_side + 100)

    split_candidates: list[dict] = []
    stable_candidates: list[dict] = []
    for region in _all_regions(combined):
        if region["touches_border"] or region["area"] < 2:
            continue
        if any(squares_mask[r, c] for r, c in region["cells"]):
            continue

        cell = _pick_visual_cell(region["cells"])
        blue_region, _blue_touches = _region_from_seed(blue_only, cell)
        red_region, _red_touches = _region_from_seed(red_only, cell)

        needs_red = region["cells"] != blue_region
        needs_blue = region["cells"] != red_region
        if not needs_blue:
            continue

        info = {
            "cells": region["cells"],
            "area": region["area"],
            "cell": cell,
            "needs_red": bool(needs_red),
            "needs_blue": bool(needs_blue),
        }
        if needs_red:
            split_candidates.append(info)
        else:
            stable_candidates.append(info)
    return split_candidates, stable_candidates


def _choose_markers(
    split_candidates: list[dict], stable_candidates: list[dict]
) -> list[tuple[int, int]] | None:
    if not split_candidates:
        return None
    target_count = random.choice(MARKER_COUNT_OPTIONS)

    random.shuffle(split_candidates)
    random.shuffle(stable_candidates)
    split_candidates.sort(key=lambda item: (item["area"], item["cell"]))
    stable_candidates.sort(key=lambda item: (-item["area"], item["cell"]))

    chosen: list[dict] = [split_candidates[0]]
    used_cells = {split_candidates[0]["cell"]}

    if stable_candidates and target_count >= 2:
        for cand in stable_candidates:
            if cand["cell"] not in used_cells:
                chosen.append(cand)
                used_cells.add(cand["cell"])
                break

    pool = split_candidates[1:] + stable_candidates[1:]
    random.shuffle(pool)
    pool.sort(key=lambda item: (-item["area"], item["cell"]))
    for cand in pool:
        if len(chosen) >= target_count:
            break
        if cand["cell"] in used_cells:
            continue
        chosen.append(cand)
        used_cells.add(cand["cell"])

    if len(chosen) < 2:
        return None
    if all(item["needs_red"] for item in chosen) and stable_candidates:
        # Keep both "split by the second lattice" and "already fine-grid sized" examples in circulation.
        return None
    if max(item["area"] for item in chosen) < 3:
        return None
    return [item["cell"] for item in chosen]


def _build_input(
    shape: tuple[int, int],
    blue_box: tuple[int, int, int, int],
    red_box: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    br0, bc0, br1, bc1 = blue_box
    rr0, rc0, rr1, rc1 = red_box
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = BLUE
    grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = RED
    for r, c in markers:
        grid[r, c] = GREEN
    return grid


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    small_side: int,
    large_side: int,
    blue_box: tuple[int, int, int, int],
    red_box: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False

    overlay = _overlay_grid(inp.shape, small_side, large_side)
    blue_mask = _grid_mask(inp.shape, small_side)
    red_mask = _grid_mask(inp.shape, large_side)
    overlap_mask = blue_mask & red_mask
    if not np.any(overlap_mask):
        return False
    if not np.all(overlay[overlap_mask] == BLUE):
        return False

    if not np.array_equal(main(inp), out):
        return False

    enclosed = [r for r in _all_regions(overlay) if not r["touches_border"]]
    if len(enclosed) <= len(markers):
        return False

    seen_regions: set[frozenset[tuple[int, int]]] = set()
    saw_needs_red = False
    saw_area_gain = False
    saw_stable = False

    blue_only = _overlay_grid(inp.shape, small_side, small_side + 100)
    red_only = _overlay_grid(inp.shape, large_side, large_side + 100)

    for marker in markers:
        if overlay[marker] != BLACK:
            return False
        region, touches_border = _region_from_seed(overlay, marker)
        if touches_border or region in seen_regions:
            return False
        seen_regions.add(region)

        blue_region, _ = _region_from_seed(blue_only, marker)
        red_region, _ = _region_from_seed(red_only, marker)
        needs_red = region != blue_region
        needs_blue = region != red_region
        if not needs_blue:
            return False
        saw_needs_red = saw_needs_red or needs_red
        saw_stable = saw_stable or (not needs_red)
        saw_area_gain = saw_area_gain or (len(region) > 1)

        for r, c in region:
            if out[r, c] != GREEN:
                return False

    if not saw_needs_red or not saw_stable or not saw_area_gain:
        return False

    unfilled_enclosed = 0
    for region in enclosed:
        if any(out[r, c] == BLACK for r, c in region["cells"]):
            unfilled_enclosed += 1
    if unfilled_enclosed == 0:
        return False

    square_mask = _squares_mask(inp.shape, blue_box, red_box)
    if np.any(square_mask & (inp == GREEN)):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)

    feasible_pairs = [
        (s1, s2)
        for s1, s2 in SIZE_PAIRS
        if len(_separator_positions(s1, h)) >= 2
        and len(_separator_positions(s1, w)) >= 2
        and len(_separator_positions(s2, h)) >= 2
        and len(_separator_positions(s2, w)) >= 2
    ]
    if not feasible_pairs:
        return None
    small_side, large_side = random.choice(feasible_pairs)

    placements = _placement_options(h, w, small_side, large_side)
    if not placements:
        return None
    blue_box, red_box = random.choice(placements)
    square_mask = _squares_mask(shape, blue_box, red_box)

    split_candidates, stable_candidates = _candidate_regions(shape, small_side, large_side, square_mask)
    markers = _choose_markers(split_candidates, stable_candidates)
    if markers is None:
        return None

    inp = _build_input(shape, blue_box, red_box, markers)
    out = _fill_marked_chambers(shape, small_side, large_side, markers)

    if not _validate_instance(inp, out, small_side, large_side, blue_box, red_box, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-grid chamber input")
