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

FILL_BY_RANGE = {
    range(1, 4): BLUE,
    range(4, 7): YELLOW,
    range(7, 11): GREEN,
}

AREA_OPTIONS = {
    "small": [(1, 1), (2, 1), (1, 3)],
    "medium": [(1, 5), (2, 3)],
    "large": [(1, 7), (3, 3), (2, 5)],
}

GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            parts.append(comp)
    return parts


def _mirror_repair(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != GREY:
                continue
            mc = w - 1 - c
            mirror_value = int(out[r, mc])
            assert mirror_value != GREY, "grey mask must be asymmetric across the vertical axis"
            out[r, c] = mirror_value
    return out


def _fill_color_for_area(area: int) -> int:
    for area_range, color in FILL_BY_RANGE.items():
        if area in area_range:
            return color
    raise AssertionError(f"unexpected enclosed area {area}")


def _enclosed_void_regions(grid: np.ndarray, wall_color: int = RED) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = g != wall_color
    regions: list[list[tuple[int, int]]] = []
    seen = np.zeros((h, w), dtype=bool)
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_edge = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            if not touches_edge:
                regions.append(comp)
    return regions


def _red_is_connected(grid: np.ndarray) -> bool:
    comps = _components(np.asarray(grid, dtype=int) == RED)
    return len(comps) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a left-right symmetric red pattern and color each enclosed black hole by its area bucket.

    Input structure:
    - The grid is square with a black background.
    - Red pixels form a vertically mirror-symmetric connected pattern, but some red boundary pixels are replaced by grey mask cells.
    - The repaired pattern contains 3-4 enclosed voids whose black areas fall in the ranges 1-3, 4-6, and 7-10.

    Rule:
    - Replace every grey pixel with the color at its vertical mirror position to restore the symmetric red pattern.
    - In the repaired grid, find every 4-connected enclosed black region fully surrounded by red.
    - Fill each enclosed black region by area: 1-3 -> blue, 4-6 -> yellow, 7-10 -> green. Keep red boundary pixels and edge-connected background black.

    Key insight:
    - Grey cells are asymmetric damage on a symmetric wall pattern, so the wall must be repaired first; only then do the enclosed regions and their areas become correct.
    """
    repaired = _mirror_repair(np.asarray(grid, dtype=int))
    out = repaired.copy()
    for region in _enclosed_void_regions(repaired, wall_color=RED):
        colors = {int(repaired[r, c]) for r, c in region}
        assert colors == {BLACK}, "repaired enclosed regions must be purely black"
        fill = _fill_color_for_area(len(region))
        for r, c in region:
            out[r, c] = fill
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, hole_h: int, hole_w: int) -> dict:
    outer_h = hole_h + 2
    outer_w = hole_w + 2
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    return {
        "frame_bbox": (r0, c0, r1, c1),
        "hole_bbox": (r0 + 1, c0 + 1, r0 + hole_h, c0 + hole_w),
        "area": hole_h * hole_w,
        "hole_h": hole_h,
        "hole_w": hole_w,
    }


def _naive_without_repair(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for region in _enclosed_void_regions(g, wall_color=RED):
        black_cells = [(r, c) for r, c in region if int(g[r, c]) == BLACK]
        if not black_cells:
            continue
        fill = _fill_color_for_area(len(black_cells))
        for r, c in black_cells:
            out[r, c] = fill
    return out


def _choose_specs() -> list[tuple[str, int, int]]:
    count = random.choice([3, 4])
    bins = ["small", "medium", "large"]
    if count == 4:
        bins.append(random.choice(bins))
    random.shuffle(bins)

    used_areas: dict[str, set[int]] = {name: set() for name in AREA_OPTIONS}
    specs: list[tuple[str, int, int]] = []
    for area_bin in bins:
        options = AREA_OPTIONS[area_bin][:]
        random.shuffle(options)
        chosen = None
        for hole_h, hole_w in options:
            area = hole_h * hole_w
            if area not in used_areas[area_bin]:
                chosen = (area_bin, hole_h, hole_w)
                used_areas[area_bin].add(area)
                break
        assert chosen is not None
        specs.append(chosen)
    return specs


def _build_candidate() -> tuple[np.ndarray, list[dict]] | None:
    specs = _choose_specs()
    outer_heights = [hole_h + 2 for _, hole_h, _hole_w in specs]
    outer_widths = [hole_w + 2 for _, _hole_h, hole_w in specs]
    gaps = [random.randint(1, 2) for _ in range(len(specs) - 1)]
    stack_h = sum(outer_heights) + sum(gaps)

    min_side = max(13, max(outer_widths) + random.randint(4, 5), stack_h + random.randint(2, 4))
    side = min_side if min_side % 2 == 1 else min_side + 1
    side += random.choice([0, 2])
    if side > 25:
        return None

    center_c = side // 2
    free_rows = side - stack_h
    if free_rows < 2:
        return None
    top_pad = random.randint(1, free_rows - 1)

    grid = np.zeros((side, side), dtype=int)
    frames: list[dict] = []
    cur_r = top_pad
    for idx, (_area_bin, hole_h, hole_w) in enumerate(specs):
        outer_w = hole_w + 2
        c0 = center_c - outer_w // 2
        frame = _draw_frame(grid, cur_r, c0, hole_h, hole_w)
        frames.append(frame)
        cur_r += hole_h + 2
        if idx < len(gaps):
            cur_r += gaps[idx]

    for i in range(len(frames) - 1):
        prev_r1 = frames[i]["frame_bbox"][2]
        next_r0 = frames[i + 1]["frame_bbox"][0]
        grid[prev_r1 : next_r0 + 1, center_c] = RED

    if random.random() < 0.6:
        top_stub = random.randint(0, 2)
        if top_stub:
            top_r0 = max(0, frames[0]["frame_bbox"][0] - top_stub)
            top_r1 = frames[0]["frame_bbox"][0]
            grid[top_r0 : top_r1 + 1, center_c] = RED
    if random.random() < 0.6:
        bottom_stub = random.randint(0, 2)
        if bottom_stub:
            bot_r0 = frames[-1]["frame_bbox"][2]
            bot_r1 = min(side - 1, frames[-1]["frame_bbox"][2] + bottom_stub)
            grid[bot_r0 : bot_r1 + 1, center_c] = RED

    min_damaged = 2 if len(frames) >= 3 else 1
    damage_count = random.randint(min_damaged, len(frames))
    for idx in random.sample(range(len(frames)), damage_count):
        r0, c0, r1, c1 = frames[idx]["frame_bbox"]
        side_name = random.choice(["left", "right"])
        col = c0 if side_name == "left" else c1
        rows = list(range(r0 + 1, r1))
        gap_len = 1 if len(rows) == 1 else random.randint(1, min(2, len(rows)))
        start = random.randint(0, len(rows) - gap_len)
        for r in rows[start : start + gap_len]:
            mirror_col = side - 1 - col
            if mirror_col == col or grid[r, mirror_col] != RED:
                return None
            grid[r, col] = GREY

    return grid, frames


def _validate_candidate(grid: np.ndarray, frames: list[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h != w or h > 30:
        return False
    if not np.any(grid == GREY):
        return False

    repaired = _mirror_repair(grid)
    if not np.array_equal(repaired, np.fliplr(repaired)):
        return False
    if not _red_is_connected(repaired):
        return False

    for r, c in np.argwhere(grid == GREY):
        mc = w - 1 - int(c)
        if int(grid[int(r), mc]) != RED or int(repaired[int(r), int(c)]) != RED:
            return False

    repaired_regions = _enclosed_void_regions(repaired, wall_color=RED)
    raw_regions = _enclosed_void_regions(grid, wall_color=RED)
    if len(repaired_regions) != len(frames):
        return False
    if len(raw_regions) >= len(repaired_regions):
        return False

    planned_areas = sorted(int(frame["area"]) for frame in frames)
    actual_areas = sorted(len(region) for region in repaired_regions)
    if actual_areas != planned_areas:
        return False
    if {_fill_color_for_area(area) for area in actual_areas} != {BLUE, YELLOW, GREEN}:
        return False

    if any({int(repaired[r, c]) for r, c in region} != {BLACK} for region in repaired_regions):
        return False

    correct = main(grid)
    if np.array_equal(correct, repaired):
        return False
    if np.array_equal(correct, _naive_without_repair(grid)):
        return False

    present = set(map(int, np.unique(correct)))
    if not {BLUE, YELLOW, GREEN}.issubset(present):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        grid, frames = built
        if _validate_candidate(grid, frames):
            return grid
    raise AssertionError("failed to generate a valid symmetry-repair enclosed-area task")
