from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

STEP_LIMIT = 200
INTERIOR_SIZES = (3, 5, 7)
GEN_ATTEMPTS = 400


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _on_ring(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1 and (r in (r0, r1) or c in (c0, c1))


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _trace_ray(shape: tuple[int, int], start: tuple[int, int], step_limit: int = STEP_LIMIT) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, shape)
    r, c = map(int, start)
    dr, dc = -1, 1
    positions = [(r, c)]
    move_dirs: list[tuple[int, int]] = []
    for _ in range(step_limit):
        nr = r + dr
        nc = c + dc
        if nr < 0 or nr >= height:
            dr = -dr
            nr = r + dr
        if nc < 0 or nc >= width:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
        positions.append((r, c))
        move_dirs.append((dr, dc))
        if (r, c) == start:
            break
    return positions, move_dirs


def _gray_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rectangles: list[dict[str, object]] = []
    for comp in _gray_components(grid):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        if outer_h < 5 or outer_w < 5:
            raise AssertionError("grey component is too small to be a valid hollow rectangle")
        ring = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        cells = set(comp)
        if cells != ring:
            raise AssertionError("grey component must be a 1-cell-thick hollow rectangle")
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        if inner_h not in INTERIOR_SIZES or inner_w not in INTERIOR_SIZES:
            raise AssertionError("rectangle interior must be odd-sized in {3,5,7}")
        rectangles.append(
            {
                "outer": (r0, c0, r1, c1),
                "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "inner_shape": (inner_h, inner_w),
            }
        )
    rectangles.sort(key=lambda rect: tuple(rect["outer"]))
    return rectangles


def _checker_color(inner_bbox: tuple[int, int, int, int], r: int, c: int, entry_dc: int) -> int:
    ir0, ic0, _ir1, _ic1 = inner_bbox
    same_parity = ((r - ir0) + (c - ic0)) % 2 == 0
    if entry_dc > 0:
        return RED if same_parity else GREEN
    return GREEN if same_parity else RED


def _render_output(grid: np.ndarray, rectangles: list[dict[str, object]], positions: Iterable[tuple[int, int]], entries: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    path_cells = set((int(r), int(c)) for r, c in positions)
    for r, c in path_cells:
        if int(out[r, c]) != GREY:
            out[r, c] = BLUE
    for idx, rect in enumerate(rectangles):
        if idx not in entries:
            continue
        inner = rect["inner"]
        ir0, ic0, ir1, ic1 = inner
        for r in range(ir0, ir1 + 1):
            for c in range(ic0, ic1 + 1):
                if (r, c) in path_cells:
                    continue
                out[r, c] = _checker_color(inner, r, c, entries[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a bouncing diagonal ray, activate the hollow rectangles whose interiors it traverses, and fill each activated interior with a direction-sensitive checkerboard.

    Input structure:
    - The grid is a black background with one blue start pixel and several non-overlapping grey hollow rectangles.
    - Each rectangle has a 1-cell-thick border and an odd-sized black interior.
    - The start pixel lies outside every rectangle.

    Rule:
    - Launch a blue ray from the start in the up-right diagonal direction and reflect it only on the four grid boundaries.
    - Paint every visited non-grey cell blue, stop when the path returns to the start cell or exceeds 200 steps, and let the ray pass through grey borders without reflecting.
    - A rectangle is activated exactly when at least one visited cell lies in its interior.
    - For each activated rectangle, look at the first step that enters its interior: if that step moves right, fill the remaining interior black cells with a red-starting checkerboard; if it moves left, use the green-starting checkerboard. Interior cells already visited by the ray stay blue.

    Key insight:
    - The same bouncing path simultaneously determines which rectangles matter and which checkerboard phase each one receives, because activation and color order both come from the ray's first interior entry segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "ARC grids must stay within 30x30"
    assert height >= 7 and width >= 7, "grid is too small for this puzzle"

    starts = np.argwhere(g == BLUE)
    assert starts.shape[0] == 1, "expected exactly one blue start pixel"
    start = tuple(map(int, starts[0]))

    rectangles = _find_rectangles(g)
    assert len(rectangles) >= 1, "expected at least one hollow rectangle"
    for rect in rectangles:
        assert not _point_in_bbox(start, rect["outer"]), "start pixel must lie outside every rectangle"

    positions, move_dirs = _trace_ray(g.shape, start)
    entries: dict[int, int] = {}
    for step_index, (dr, dc) in enumerate(move_dirs, start=1):
        prev = positions[step_index - 1]
        cur = positions[step_index]
        for rect_index, rect in enumerate(rectangles):
            inner = rect["inner"]
            if rect_index in entries:
                continue
            if _point_in_bbox(cur, inner) and not _point_in_bbox(prev, inner):
                entries[rect_index] = int(dc)

    return _render_output(g, rectangles, positions, entries)


def _active_rectangles_from_metadata(rectangles: list[dict[str, object]]) -> list[dict[str, object]]:
    return [rect for rect in rectangles if rect.get("active", False)]


def _build_input(shape: tuple[int, int], start: tuple[int, int], rectangles: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[start] = BLUE
    for rect in rectangles:
        r0, c0, r1, c1 = rect["outer"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
    return grid


def _make_active_rect_candidate(
    shape: tuple[int, int],
    positions: list[tuple[int, int]],
    move_dirs: list[tuple[int, int]],
    step_index: int,
    inner_h: int,
    inner_w: int,
    offset_r: int,
    offset_c: int,
    start: tuple[int, int],
) -> dict[str, object] | None:
    height, width = shape
    cur_r, cur_c = positions[step_index]
    prev = positions[step_index - 1]
    dc = int(move_dirs[step_index - 1][1])
    r0 = cur_r - offset_r - 1
    c0 = cur_c - offset_c - 1
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    outer = (r0, c0, r1, c1)
    inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    if r0 < 1 or c0 < 1 or r1 >= height - 1 or c1 >= width - 1:
        return None
    if _point_in_bbox(start, outer) or _point_in_bbox(prev, inner):
        return None

    first_inside = None
    inside_cells: set[tuple[int, int]] = set()
    ring_touch = False
    for idx, point in enumerate(positions):
        if _point_in_bbox(point, inner):
            inside_cells.add(point)
            if first_inside is None:
                first_inside = idx
        elif _on_ring(point, outer):
            ring_touch = True
    if first_inside != step_index:
        return None
    inner_area = inner_h * inner_w
    if len(inside_cells) == 0 or len(inside_cells) > inner_area - 2:
        return None
    return {
        "outer": outer,
        "inner": inner,
        "active": True,
        "entry_dc": dc,
        "entry_index": step_index,
        "inside_cells": inside_cells,
        "ring_touch": ring_touch,
    }


def _enumerate_active_candidates(shape: tuple[int, int], start: tuple[int, int], positions: list[tuple[int, int]], move_dirs: list[tuple[int, int]]) -> list[dict[str, object]]:
    by_bbox: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for step_index in range(1, len(positions)):
        for inner_h in INTERIOR_SIZES:
            for inner_w in INTERIOR_SIZES:
                for offset_r in range(inner_h):
                    for offset_c in range(inner_w):
                        cand = _make_active_rect_candidate(
                            shape,
                            positions,
                            move_dirs,
                            step_index,
                            inner_h,
                            inner_w,
                            offset_r,
                            offset_c,
                            start,
                        )
                        if cand is None:
                            continue
                        bbox = cand["outer"]
                        prev = by_bbox.get(bbox)
                        if prev is None or (
                            len(cand["inside_cells"]) < len(prev["inside_cells"])
                            or (
                                len(cand["inside_cells"]) == len(prev["inside_cells"])
                                and cand["entry_index"] < prev["entry_index"]
                            )
                        ):
                            by_bbox[bbox] = cand
    candidates = list(by_bbox.values())
    random.shuffle(candidates)
    return candidates


def _enumerate_inactive_candidates(
    shape: tuple[int, int],
    start: tuple[int, int],
    positions: list[tuple[int, int]],
    blocked: list[tuple[int, int, int, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    height, width = shape
    touch: list[dict[str, object]] = []
    clear: list[dict[str, object]] = []
    for inner_h in INTERIOR_SIZES:
        for inner_w in INTERIOR_SIZES:
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            for r0 in range(1, height - outer_h):
                for c0 in range(1, width - outer_w):
                    outer = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                    inner = (r0 + 1, c0 + 1, r0 + outer_h - 2, c0 + outer_w - 2)
                    if _point_in_bbox(start, outer):
                        continue
                    if any(not _bboxes_separated(outer, bbox, gap=2) for bbox in blocked):
                        continue
                    inside = False
                    ring_touch = False
                    for point in positions:
                        if _point_in_bbox(point, inner):
                            inside = True
                            break
                        if _on_ring(point, outer):
                            ring_touch = True
                    if inside:
                        continue
                    cand = {
                        "outer": outer,
                        "inner": inner,
                        "active": False,
                        "ring_touch": ring_touch,
                    }
                    if ring_touch:
                        touch.append(cand)
                    else:
                        clear.append(cand)
    random.shuffle(touch)
    random.shuffle(clear)
    return touch, clear


def _pick_active_rectangles(candidates: list[dict[str, object]], target_active_count: int) -> list[dict[str, object]] | None:
    right = [cand for cand in candidates if int(cand["entry_dc"]) > 0]
    left = [cand for cand in candidates if int(cand["entry_dc"]) < 0]
    random.shuffle(right)
    random.shuffle(left)
    for rcand in right:
        for lcand in left:
            if not _bboxes_separated(rcand["outer"], lcand["outer"], gap=2):
                continue
            chosen = [rcand, lcand]
            if target_active_count == 2:
                return chosen
            pool = [cand for cand in candidates if cand not in chosen]
            random.shuffle(pool)
            for extra in pool:
                if all(_bboxes_separated(extra["outer"], rect["outer"], gap=2) for rect in chosen):
                    return chosen + [extra]
    return None


def _pick_inactive_rectangles(
    touch_candidates: list[dict[str, object]],
    clear_candidates: list[dict[str, object]],
    chosen: list[dict[str, object]],
    need: int,
) -> list[dict[str, object]] | None:
    picked: list[dict[str, object]] = []
    blocked = [rect["outer"] for rect in chosen]

    def try_add(cands: list[dict[str, object]]) -> bool:
        for cand in cands:
            if all(_bboxes_separated(cand["outer"], bbox, gap=2) for bbox in blocked):
                picked.append(cand)
                blocked.append(cand["outer"])
                return True
        return False

    if need <= 0:
        return []
    if touch_candidates:
        try_add(touch_candidates)
    while len(picked) < need:
        if not try_add(clear_candidates) and not try_add(touch_candidates):
            return None
    return picked


def _render_expected_from_metadata(shape: tuple[int, int], start: tuple[int, int], rectangles: list[dict[str, object]]) -> np.ndarray:
    grid = _build_input(shape, start, rectangles)
    positions, _move_dirs = _trace_ray(shape, start)
    entries = {
        idx: int(rect["entry_dc"])
        for idx, rect in enumerate(rectangles)
        if rect.get("active", False)
    }
    return _render_output(grid, rectangles, positions, entries)


def _validate_generation(shape: tuple[int, int], start: tuple[int, int], rectangles: list[dict[str, object]]) -> np.ndarray | None:
    grid = _build_input(shape, start, rectangles)
    positions, move_dirs = _trace_ray(shape, start)

    if positions[-1] != start and len(move_dirs) < STEP_LIMIT:
        return None

    active_count = 0
    entry_dirs = set()
    for rect in rectangles:
        inner = rect["inner"]
        inside_steps = [idx for idx, point in enumerate(positions) if _point_in_bbox(point, inner)]
        if rect["active"]:
            if not inside_steps:
                return None
            first = inside_steps[0]
            if first == 0:
                return None
            actual_dc = int(move_dirs[first - 1][1])
            if actual_dc != int(rect["entry_dc"]):
                return None
            visible_blue = {positions[idx] for idx in inside_steps}
            inner_area = (inner[2] - inner[0] + 1) * (inner[3] - inner[1] + 1)
            if len(visible_blue) > inner_area - 2:
                return None
            active_count += 1
            entry_dirs.add(actual_dc)
        elif inside_steps:
            return None

    if active_count < 2 or active_count >= len(rectangles):
        return None
    if entry_dirs != {-1, 1}:
        return None

    inactive = [rect for rect in rectangles if not rect["active"]]
    if not inactive:
        return None
    if not any(bool(rect.get("ring_touch", False)) for rect in inactive):
        return None

    detected = _find_rectangles(grid)
    if len(detected) != len(rectangles):
        return None
    if any(tuple(detected[idx]["outer"]) != tuple(rectangles[idx]["outer"]) for idx in range(len(rectangles))):
        return None

    expected = _render_expected_from_metadata(shape, start, rectangles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    blue_count = int(np.sum(actual == BLUE))
    if blue_count <= 1:
        return None
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    start = (random.randint(4, height - 5), random.randint(2, width - 4))
    positions, move_dirs = _trace_ray((height, width), start)

    candidates = _enumerate_active_candidates((height, width), start, positions, move_dirs)
    if not candidates:
        return None

    total_rectangles = random.choice((3, 4))
    target_active_count = 2 if total_rectangles == 3 else random.choice((2, 3))
    active_rects = _pick_active_rectangles(candidates, target_active_count)
    if active_rects is None:
        return None

    blocked = [rect["outer"] for rect in active_rects]
    touch_candidates, clear_candidates = _enumerate_inactive_candidates((height, width), start, positions, blocked)
    inactive_rects = _pick_inactive_rectangles(touch_candidates, clear_candidates, active_rects, total_rectangles - len(active_rects))
    if inactive_rects is None:
        return None

    rectangles = [dict(rect) for rect in active_rects + inactive_rects]
    rectangles.sort(key=lambda rect: tuple(rect["outer"]))
    return _validate_generation((height, width), start, rectangles)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-rectangle ARC input")
