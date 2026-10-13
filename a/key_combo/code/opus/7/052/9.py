from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

FILL_BY_SIDE = {5: BLUE, 7: YELLOW, 9: GREEN}
OPPOSITE_FILL = {BLUE: GREEN, GREEN: BLUE, YELLOW: ORANGE}

RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300
FRAME_COUNT_CHOICES = (3, 4, 5)
SIDE_CHOICES = (5, 7, 9)
MIN_GRID_SIZE = 18
MAX_GRID_SIZE = 30
MIN_GAP = 1
MAX_GRID_SLACK = 4


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_square_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side not in FILL_BY_SIDE:
        return False
    if len(comp) != 4 * side - 4:
        return False
    comp_set = set(comp)
    for c in range(c0, c1 + 1):
        if (r0, c) not in comp_set or (r1, c) not in comp_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in comp_set or (r, c1) not in comp_set:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each red hollow square by side length, then mark exposed interior defects with an opposite-color halo.

    Input structure:
    - The grid has a black background with 3 to 5 separated red hollow square frames of side 5, 7, or 9.
    - Each frame contains a single centered red marker, and some frames also contain 0 to 2 gray defect pixels inside.
    - Defects are isolated from the border and appear subtle in the black interior before filling.

    Rule:
    - Detect every red hollow square frame and keep its red border and centered red marker unchanged.
    - Fill each black interior by frame side: 5 -> blue, 7 -> yellow, 9 -> green.
    - Any interior pixel that is neither the fill color nor red after filling is a defect.
    - For each defect, recolor fill-colored cells in its clipped 3x3 neighborhood to the corresponding opposite halo color, then recolor the defect center itself teal.

    Key insight:
    - The defects are only identifiable after the size-based fill exposes them against the interior background, and the fill color also determines the halo color used to mark each defect.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    red_components = _components(g == RED)
    markers = {comp[0] for comp in red_components if len(comp) == 1}
    frames = []
    for comp in red_components:
        if len(comp) > 1 and _is_square_frame(g, comp):
            frames.append(_bbox(comp))

    frames.sort()
    for r0, c0, r1, c1 in frames:
        side = r1 - r0 + 1
        fill_color = FILL_BY_SIDE[side]
        center = (r0 + side // 2, c0 + side // 2)
        if center not in markers:
            raise AssertionError("missing centered red marker")

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color

        defects = np.argwhere((interior != fill_color) & (interior != RED))
        halo_color = OPPOSITE_FILL[fill_color]
        for dr, dc in defects:
            rr = r0 + 1 + int(dr)
            cc = c0 + 1 + int(dc)
            rs = slice(max(0, rr - 1), min(out.shape[0], rr + 2))
            cs = slice(max(0, cc - 1), min(out.shape[1], cc + 2))
            patch = out[rs, cs]
            patch[patch == fill_color] = halo_color
            out[rr, cc] = TEAL

    return out


def _candidate_defect_offsets(side: int) -> list[tuple[int, int]]:
    center = side // 2
    offsets = []
    for dr in range(2, side - 2):
        for dc in range(2, side - 2):
            if (dr, dc) != (center, center):
                offsets.append((dr, dc))
    return offsets


def _choose_defect_offsets(side: int, count: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_defect_offsets(side)
    RNG.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for dr, dc in candidates:
        if all(max(abs(dr - rr), abs(dc - cc)) >= 3 for rr, cc in chosen):
            chosen.append((dr, dc))
            if len(chosen) == count:
                return chosen
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, side: int) -> None:
    grid[top, left : left + side] = RED
    grid[top + side - 1, left : left + side] = RED
    grid[top : top + side, left] = RED
    grid[top : top + side, left + side - 1] = RED
    grid[top + side // 2, left + side // 2] = RED


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _feasible_row_assignments(specs: list[dict], row_count: int) -> list[list[list[int]]]:
    assignments: list[list[list[int]]] = []

    def backtrack(index: int, rows: list[list[int]]) -> None:
        if index == len(specs):
            if all(rows):
                assignments.append([row[:] for row in rows])
            return

        order = list(range(row_count))
        RNG.shuffle(order)
        for row_idx in order:
            rows[row_idx].append(index)
            backtrack(index + 1, rows)
            rows[row_idx].pop()

    backtrack(0, [[] for _ in range(row_count)])
    return assignments


def _layout_rows(specs: list[dict]) -> tuple[list[list[int]], int, int] | None:
    row_counts = [n for n in (2, 3) if n <= len(specs)]
    RNG.shuffle(row_counts)
    for row_count in row_counts:
        assignments = _feasible_row_assignments(specs, row_count)
        RNG.shuffle(assignments)
        for rows in assignments:
            rows = [row[:] for row in rows]
            for row in rows:
                RNG.shuffle(row)
            row_heights = [max(int(specs[idx]["side"]) for idx in row) for row in rows]
            row_widths = [
                sum(int(specs[idx]["side"]) for idx in row) + MIN_GAP * (len(row) - 1)
                for row in rows
            ]
            min_h = 2 + sum(row_heights) + MIN_GAP * (row_count - 1)
            min_w = 2 + max(row_widths)
            if min_h <= MAX_GRID_SIZE and min_w <= MAX_GRID_SIZE:
                return rows, min_h, min_w
    return None


def _distribute_extra(extra: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(extra):
        parts[int(RNG.integers(slots))] += 1
    return parts


def _place_rows(specs: list[dict], rows: list[list[int]], height: int, width: int) -> list[tuple[int, int, int, int]]:
    row_heights = [max(int(specs[idx]["side"]) for idx in row) for row in rows]
    min_vertical = [1] * (len(rows) + 1)
    used_h = sum(row_heights) + sum(min_vertical)
    vertical_extra = _distribute_extra(height - used_h, len(min_vertical))
    vertical_gaps = [a + b for a, b in zip(min_vertical, vertical_extra)]

    placements: list[tuple[int, int, int, int]] = []
    top = vertical_gaps[0]
    for row_index, row in enumerate(rows):
        row_height = row_heights[row_index]
        min_horizontal = [1] * (len(row) + 1)
        used_w = sum(int(specs[idx]["side"]) for idx in row) + sum(min_horizontal)
        horizontal_extra = _distribute_extra(width - used_w, len(min_horizontal))
        horizontal_gaps = [a + b for a, b in zip(min_horizontal, horizontal_extra)]

        left = horizontal_gaps[0]
        for side_index, frame_idx in enumerate(row):
            side = int(specs[frame_idx]["side"])
            row_slack = row_height - side
            offset = int(RNG.integers(row_slack + 1)) if row_slack > 0 else 0
            placements.append((frame_idx, top + offset, left, side))
            left += side + horizontal_gaps[side_index + 1]

        top += row_height + vertical_gaps[row_index + 1]

    return placements


def _choose_frame_specs(num_frames: int) -> list[dict]:
    sides = [int(RNG.choice(SIDE_CHOICES)) for _ in range(num_frames)]
    if not any(side in (7, 9) for side in sides):
        sides[int(RNG.integers(num_frames))] = int(RNG.choice((7, 9)))

    eligible = [idx for idx, side in enumerate(sides) if side in (7, 9)]
    max_defect_frames = min(len(eligible), num_frames - 1)
    defect_frame_count = int(RNG.integers(1, max_defect_frames + 1))
    defect_frames = set(RNG.choice(eligible, size=defect_frame_count, replace=False).tolist())

    specs: list[dict] = []
    for idx, side in enumerate(sides):
        if idx not in defect_frames:
            defect_count = 0
        elif side == 7:
            defect_count = 1
        else:
            defect_count = 2 if RNG.random() < 0.45 else 1
        specs.append({"side": side, "defect_count": defect_count})
    return specs


def _render_expected(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in sorted(frames, key=lambda item: (item["top"], item["left"])):
        top = int(frame["top"])
        left = int(frame["left"])
        side = int(frame["side"])
        fill_color = FILL_BY_SIDE[side]
        interior = out[top + 1 : top + side - 1, left + 1 : left + side - 1]
        interior[interior == BLACK] = fill_color
        halo_color = OPPOSITE_FILL[fill_color]
        for rr, cc in sorted(frame["defects"]):
            patch = out[rr - 1 : rr + 2, cc - 1 : cc + 2]
            patch[patch == fill_color] = halo_color
            out[rr, cc] = TEAL
    return out


def _build_input_from_specs(specs: list[dict]) -> np.ndarray | None:
    layout = _layout_rows(specs)
    if layout is None:
        return None
    rows, min_h, min_w = layout

    height_hi = min(MAX_GRID_SIZE, max(MIN_GRID_SIZE, min_h) + MAX_GRID_SLACK)
    width_hi = min(MAX_GRID_SIZE, max(MIN_GRID_SIZE, min_w) + MAX_GRID_SLACK)
    height_lo = max(MIN_GRID_SIZE, min_h)
    width_lo = max(MIN_GRID_SIZE, min_w)
    if height_lo > height_hi or width_lo > width_hi:
        return None

    height = int(RNG.integers(height_lo, height_hi + 1))
    width = int(RNG.integers(width_lo, width_hi + 1))
    placements = _place_rows(specs, rows, height, width)
    if len(placements) != len(specs):
        return None

    boxes = []
    for _frame_idx, top, left, side in placements:
        box = (top, left, top + side - 1, left + side - 1)
        boxes.append(box)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if not _boxes_have_gap(boxes[i], boxes[j]):
                return None

    grid = np.full((height, width), BLACK, dtype=int)
    frames: list[dict] = []
    for frame_idx, top, left, side in placements:
        spec = specs[frame_idx]
        _draw_frame(grid, top, left, side)
        defects_needed = int(spec["defect_count"])
        offsets = _choose_defect_offsets(side, defects_needed)
        if defects_needed and offsets is None:
            return None
        defects = []
        for dr, dc in offsets or []:
            rr = top + dr
            cc = left + dc
            grid[rr, cc] = GREY
            defects.append((rr, cc))
        frames.append({"top": top, "left": left, "side": side, "defects": defects})

    has_defect = any(frame["defects"] for frame in frames)
    has_clean = any(not frame["defects"] for frame in frames)
    if not (has_defect and has_clean):
        return None

    expected = _render_expected(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        num_frames = int(RNG.choice(FRAME_COUNT_CHOICES))
        specs = _choose_frame_specs(num_frames)
        grid = _build_input_from_specs(specs)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-9 input")
