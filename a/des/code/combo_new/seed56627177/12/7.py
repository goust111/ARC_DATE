from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_TO_DIR = {
    7: (0, 1),   # orange -> right
    6: (1, 0),   # magenta -> down
    8: (0, -1),  # azure -> left
    9: (-1, 0),  # maroon -> up
}
DIR_TO_SEED = {direction: color for color, direction in SEED_TO_DIR.items()}
BODY_COLORS = [1, 2, 3, 4, 5]
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 120


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[set[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)

    return components


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _interior_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    interior: list[tuple[int, int]] = []
    for r, c in cells:
        if all((r + dr, c + dc) in cells for dr, dc in DIR4):
            interior.append((r, c))
    return interior


def _directional_frontier(cells: set[tuple[int, int]], direction: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = direction
    return {
        (r + dr, c + dc)
        for r, c in cells
        if (r + dr, c + dc) not in cells
    }


def _touches_edge(new_cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1 for r, c in new_cells)


def _touches_other_region(new_cells: set[tuple[int, int]], other_cells: set[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        if (r, c) in other_cells:
            return True
        for dr, dc in DIR4:
            if (r + dr, c + dc) in other_cells:
                return True
    return False


def _sets_conflict(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    return _touches_other_region(a, b)


def _parse_regions(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    regions: list[dict] = []

    for comp in _component_sets(g != bg):
        colors = Counter(int(g[r, c]) for r, c in comp)
        assert len(colors) == 2, "each region must contain a body color and one seed color"
        main_color, main_count = max(colors.items(), key=lambda kv: kv[1])
        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) != int(main_color)]
        assert len(seed_cells) == 1, "each region must contain exactly one seed pixel"
        assert main_count == len(comp) - 1, "the seed must be the unique minority pixel"
        seed = seed_cells[0]
        seed_color = int(g[seed])
        assert seed_color in SEED_TO_DIR, "seed color must encode one of four directions"
        regions.append(
            {
                "cells": set(comp),
                "main_color": int(main_color),
                "seed": (int(seed[0]), int(seed[1])),
                "seed_color": seed_color,
                "direction": SEED_TO_DIR[seed_color],
            }
        )

    regions.sort(key=lambda region: (min(r for r, _ in region["cells"]), min(c for _, c in region["cells"])))
    return bg, regions


def _grow_region_independent(
    cells: set[tuple[int, int]],
    direction: tuple[int, int],
    other_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], int, str]:
    current = set(cells)
    steps = 0
    stop_reason = "none"

    while True:
        new_cells = _directional_frontier(current, direction)
        if not new_cells:
            return current, steps, stop_reason
        if _touches_edge(new_cells, h, w):
            return current, steps, "edge"
        if _touches_other_region(new_cells, other_cells):
            return current, steps, "other"
        current |= new_cells
        steps += 1


def _grown_regions_independent(regions: list[dict], h: int, w: int) -> list[dict]:
    all_cells = set().union(*(region["cells"] for region in regions))
    grown: list[dict] = []
    for region in regions:
        other_cells = all_cells - region["cells"]
        final_cells, steps, stop_reason = _grow_region_independent(
            region["cells"], region["direction"], other_cells, h, w
        )
        grown.append({**region, "final_cells": final_cells, "steps": steps, "stop_reason": stop_reason})
    return grown


def _simulate_lockstep(regions: list[dict], h: int, w: int) -> list[set[tuple[int, int]]]:
    states = [set(region["cells"]) for region in regions]
    blocked = [False] * len(regions)

    while True:
        any_growth = False
        candidates = [set() for _ in regions]

        for i, region in enumerate(regions):
            if blocked[i]:
                continue
            new_cells = _directional_frontier(states[i], region["direction"])
            if not new_cells:
                blocked[i] = True
                continue
            other_current = set().union(*(states[j] for j in range(len(states)) if j != i))
            if _touches_edge(new_cells, h, w) or _touches_other_region(new_cells, other_current):
                blocked[i] = True
                continue
            candidates[i] = new_cells

        conflicted = [False] * len(regions)
        for i in range(len(regions)):
            if not candidates[i]:
                continue
            for j in range(i + 1, len(regions)):
                if candidates[j] and _sets_conflict(candidates[i], candidates[j]):
                    conflicted[i] = True
                    conflicted[j] = True

        for i in range(len(regions)):
            if candidates[i] and not conflicted[i]:
                states[i] |= candidates[i]
                any_growth = True
            elif candidates[i] and conflicted[i]:
                blocked[i] = True

        if not any_growth:
            break

    return states


def _paint_output(shape: tuple[int, int], grown_regions: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for region in grown_regions:
        for r, c in region["final_cells"]:
            out[r, c] = int(region["main_color"])
    return out


def _paint_states(shape: tuple[int, int], states: list[set[tuple[int, int]]], regions: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for state, region in zip(states, regions):
        for r, c in state:
            out[r, c] = int(region["main_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seeded color region in the direction encoded by its internal marker until the next full layer would hit another region or the grid edge.

    Input structure:
    - The background is black.
    - Each foreground object is a 4-connected region whose body is almost monochrome.
    - Exactly one pixel inside each region has a different color; that seed color is one of orange, magenta, azure, or maroon.
    - Different regions are separated by background and do not initially touch.

    Rule:
    - Treat each non-background connected component as one region; its majority color is the body color and its unique minority pixel is the seed.
    - Decode the seed color as a growth direction: orange -> right, magenta -> down, azure -> left, maroon -> up.
    - For one region at a time, repeatedly add the whole directional frontier one column or one row farther in that direction.
    - Stop before a new layer would touch the grid edge or become 4-neighbor adjacent to any pixel of another original region.
    - Replace the seed with the body color in the output.

    Key insight:
    - The odd-colored interior pixel is not decorative: it is a direction code, and the region grows by repeatedly extruding its entire exposed side rather than by moving the seed itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, regions = _parse_regions(g)
    grown_regions = _grown_regions_independent(regions, *g.shape)
    return _paint_output(g.shape, grown_regions, bg)


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _random_shape() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        cells = {(r, c) for r in range(h) for c in range(w)}
        mode = random.choice(["rect", "cut1", "cut2", "tab", "tab_cut", "double_tab"])

        if mode in {"cut1", "cut2", "tab_cut"}:
            corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            cut_count = 1 if mode == "cut1" else 2
            if mode == "tab_cut":
                cut_count = 1
            for cell in random.sample(corners, cut_count):
                cells.discard(cell)

        if mode in {"tab", "tab_cut", "double_tab"}:
            tab_count = 1 if mode != "double_tab" else 2
            used_sides: set[str] = set()
            for _tab in range(tab_count):
                side = random.choice([s for s in ("top", "bottom", "left", "right") if s not in used_sides])
                used_sides.add(side)
                if side in {"top", "bottom"}:
                    span = random.randint(1, max(1, w - 1))
                    start = random.randint(0, w - span)
                    row = -1 if side == "top" else h
                    for c in range(start, start + span):
                        cells.add((row, c))
                else:
                    span = random.randint(1, max(1, h - 1))
                    start = random.randint(0, h - span)
                    col = -1 if side == "left" else w
                    for r in range(start, start + span):
                        cells.add((r, col))

        cells = _normalize(cells)
        if not _is_connected(cells):
            continue
        if len(cells) < 6:
            continue
        interior = _interior_cells(cells)
        if not interior:
            continue
        seed = random.choice(interior)
        return cells, seed

    raise RuntimeError("failed to generate a shape with an interior seed")


def _dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _sample_overlapping_start(base_start: int, base_len: int, band_start: int, band_len: int, other_len: int) -> int | None:
    low = band_start
    high = band_start + band_len - other_len
    overlap_low = max(low, base_start - other_len + 1)
    overlap_high = min(high, base_start + base_len - 1)
    if overlap_low > overlap_high:
        return None
    return random.randint(overlap_low, overlap_high)


def _sample_nonoverlap_start(base_start: int, base_len: int, band_start: int, band_len: int, other_len: int) -> int | None:
    low = band_start
    high = band_start + band_len - other_len
    candidates = []
    for start in range(low, high + 1):
        end = start + other_len - 1
        if end < base_start or start > base_start + base_len - 1:
            candidates.append(start)
    if not candidates:
        return None
    return random.choice(candidates)


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r0, c0 in a:
        for r1, c1 in b:
            best = min(best, abs(r0 - r1) + abs(c0 - c1))
            if best <= 2:
                return best
    return best


def _build_ring_layout(include_center: bool) -> tuple[tuple[int, int], list[dict]] | None:
    block_b, block_d = random.choice(
        [(False, False), (True, False), (False, True), (True, True)] if include_center
        else [(False, False), (True, False), (False, True)]
    )

    shapes = {}
    for key in ("A", "B", "C", "D"):
        cells, seed = _random_shape()
        shapes[key] = {"cells": cells, "seed": seed}
    if include_center:
        cells, seed = _random_shape()
        shapes["E"] = {"cells": cells, "seed": seed}

    hA, wA = _dims(shapes["A"]["cells"])
    hB, wB = _dims(shapes["B"]["cells"])
    hC, wC = _dims(shapes["C"]["cells"])
    hD, wD = _dims(shapes["D"]["cells"])

    left_band = max(wA, wD) + random.randint(0, 2)
    right_band = max(wB, wC) + random.randint(0, 2)
    if not block_d:
        left_band = max(left_band, wA + wD + 1 + random.randint(0, 2))
    if not block_b:
        right_band = max(right_band, wB + wC + 1 + random.randint(0, 2))

    top_band = max(hA, hB) + random.randint(0, 2)
    bottom_band = max(hC, hD) + random.randint(0, 2)

    hole_h = random.randint(6, 11 if include_center else 9)
    hole_w = random.randint(6, 11 if include_center else 9)
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)

    H = margin_top + top_band + hole_h + bottom_band + margin_bottom
    W = margin_left + left_band + hole_w + right_band + margin_right
    if H > 30 or W > 30:
        return None

    top_start = margin_top
    bottom_start = H - margin_bottom - bottom_band
    left_start = margin_left
    right_start = W - margin_right - right_band

    rA = top_start + random.randint(0, top_band - hA)
    cA = left_start + random.randint(0, left_band - wA)

    rB = _sample_overlapping_start(rA, hA, top_start, top_band, hB)
    if rB is None:
        return None
    cB = right_start + random.randint(0, right_band - wB)

    rC = bottom_start + random.randint(0, bottom_band - hC)
    if block_b:
        cC = _sample_overlapping_start(cB, wB, right_start, right_band, wC)
    else:
        cC = _sample_nonoverlap_start(cB, wB, right_start, right_band, wC)
    if cC is None:
        return None

    rD = _sample_overlapping_start(rC, hC, bottom_start, bottom_band, hD)
    if rD is None:
        return None
    if block_d:
        cD = _sample_overlapping_start(cA, wA, left_start, left_band, wD)
    else:
        cD = _sample_nonoverlap_start(cA, wA, left_start, left_band, wD)
    if cD is None:
        return None

    placements = {
        "A": (rA, cA, (0, 1)),
        "B": (rB, cB, (1, 0)),
        "C": (rC, cC, (0, -1)),
        "D": (rD, cD, (-1, 0)),
    }

    if include_center:
        hE, wE = _dims(shapes["E"]["cells"])
        hole_top = top_start + top_band
        hole_left = left_start + left_band
        if hE > hole_h or wE > hole_w:
            return None
        rE = hole_top + random.randint(0, hole_h - hE)
        cE = hole_left + random.randint(0, hole_w - wE)
        placements["E"] = (rE, cE, random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)]))

    objects: list[dict] = []
    for body_color, key in zip(random.sample(BODY_COLORS, len(placements)), placements):
        top, left, direction = placements[key]
        abs_cells = _translate(shapes[key]["cells"], top, left)
        seed_r, seed_c = shapes[key]["seed"]
        seed = (top + seed_r, left + seed_c)
        objects.append(
            {
                "cells": abs_cells,
                "seed": seed,
                "main_color": body_color,
                "seed_color": DIR_TO_SEED[direction],
                "direction": direction,
            }
        )

    return (H, W), objects


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["main_color"])
        sr, sc = obj["seed"]
        grid[sr, sc] = int(obj["seed_color"])
    return grid


def _valid_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        bg, regions = _parse_regions(grid)
    except AssertionError:
        return False

    if len(regions) not in (4, 5):
        return False

    for i, region in enumerate(regions):
        if region["seed"] not in _interior_cells(region["cells"]):
            return False
        for other in regions[i + 1 :]:
            if _min_manhattan(region["cells"], other["cells"]) <= 2:
                return False

    grown_regions = _grown_regions_independent(regions, *grid.shape)
    if any(region["steps"] < 1 for region in grown_regions):
        return False

    reasons = {region["stop_reason"] for region in grown_regions}
    if not {"edge", "other"} <= reasons:
        return False

    final_sets = [region["final_cells"] for region in grown_regions]
    for i, cells in enumerate(final_sets):
        for other in final_sets[i + 1 :]:
            if _sets_conflict(cells, other):
                return False

    strict_states = _simulate_lockstep(regions, *grid.shape)
    independent_out = _paint_output(grid.shape, grown_regions, bg)
    strict_out = _paint_states(grid.shape, strict_states, regions, bg)
    if not np.array_equal(independent_out, strict_out):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(out, independent_out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        include_center = random.random() < 0.5
        built = _build_ring_layout(include_center)
        if built is None:
            continue
        shape, objects = built
        grid = _render_input(shape, objects)
        if _valid_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for task 7")
