from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SOURCE_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE]
RANK_COLORS = [BLUE, RED, GREEN, YELLOW]

GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_RANGE = (13, 18)
OBJECT_AREA_RANGE = range(3, 11)
MAX_OBJECT_HEIGHT = 4
MAX_OBJECT_WIDTH = 4
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _continuous_runs(cols: list[int]) -> list[tuple[int, int]]:
    if not cols:
        return []
    runs = []
    start = prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _guide_info(grid: np.ndarray) -> dict:
    guide_pixels = np.argwhere(grid == GREY)
    assert guide_pixels.size > 0, "expected a grey guide bar"
    guide_rows = sorted(set(int(r) for r in guide_pixels[:, 0]))
    assert len(guide_rows) == 1, "guide bar must occupy exactly one row"
    guide_row = guide_rows[0]
    h, w = grid.shape
    assert 0 < guide_row < h - 1, "guide row needs space on both sides"

    guide_cols = sorted(int(c) for c in np.where(grid[guide_row] == GREY)[0])
    assert guide_cols, "guide row must contain grey cells"
    span_start, span_end = guide_cols[0], guide_cols[-1]
    segments = _continuous_runs(guide_cols)
    gap_cols = [c for c in range(span_start, span_end + 1) if grid[guide_row, c] != GREY]
    assert 1 <= len(_continuous_runs(gap_cols)) <= 2, "guide must contain one or two internal gaps"
    return {
        "row": guide_row,
        "cols": set(guide_cols),
        "segments": segments,
        "gap_cols": set(gap_cols),
        "span": (span_start, span_end),
        "width": w,
    }


def _extract_objects(grid: np.ndarray, guide: dict) -> list[dict]:
    objects = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BLACK, GREY)):
        comps = _components_mask(grid == color)
        assert len(comps) == 1, "each source color should form exactly one object"
        coords = comps[0]
        rows = coords[:, 0]
        cols = coords[:, 1]
        if int(rows.max()) < guide["row"]:
            side = "top"
            distance = guide["row"] - int(rows.max()) - 1
            shift = distance
        else:
            assert int(rows.min()) > guide["row"], "objects must sit wholly above or below the guide"
            side = "bottom"
            distance = int(rows.min()) - guide["row"] - 1
            shift = -distance
        assert distance >= 1, "objects must be separated from the guide by a black row"
        ideal = coords.copy()
        ideal[:, 0] += shift
        hit_gap = any(grid[guide["row"], int(c)] != GREY for c in set(map(int, cols)))
        objects.append(
            {
                "color": color,
                "coords": coords,
                "ideal_coords": ideal,
                "side": side,
                "distance": distance,
                "area": int(coords.shape[0]),
                "cols": sorted(set(map(int, cols))),
                "hit_gap": hit_gap,
            }
        )
    return objects


def _stack_side(objects: list[dict], side: str, shape: tuple[int, int]) -> list[dict]:
    h, _ = shape
    occupied = np.zeros(shape, dtype=bool)
    placed = []
    for obj in sorted(objects, key=lambda item: (item["distance"], item["area"], item["color"])):
        base = obj["ideal_coords"]
        offset = 0
        while True:
            candidate = base.copy()
            candidate[:, 0] += -offset if side == "top" else offset
            if candidate[:, 0].min() < 0 or candidate[:, 0].max() >= h:
                raise AssertionError("stacking pushed an object out of bounds")
            if not occupied[candidate[:, 0], candidate[:, 1]].any():
                break
            offset += 1
        occupied[candidate[:, 0], candidate[:, 1]] = True
        placed.append({**obj, "out_coords": candidate, "stacked": offset > 0})
    return placed


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    guide = _guide_info(g)
    objects = _extract_objects(g, guide)
    assert len(objects) >= 4, "expected multiple source objects"
    areas = [obj["area"] for obj in objects]
    assert len(areas) == len(set(areas)), "object areas must be distinct"

    survivors = [obj for obj in objects if not obj["hit_gap"]]
    erased = [obj for obj in objects if obj["hit_gap"]]
    assert len(survivors) >= 1 and len(erased) >= 1

    placed = _stack_side([obj for obj in survivors if obj["side"] == "top"], "top", g.shape)
    placed += _stack_side([obj for obj in survivors if obj["side"] == "bottom"], "bottom", g.shape)
    assert 1 <= len(placed) <= 4, "ranking palette only supports up to four surviving objects"

    out = np.zeros_like(g)
    out[guide["row"], list(sorted(guide["cols"]))] = GREY
    ranked = sorted(placed, key=lambda item: item["area"], reverse=True)
    for rank, obj in enumerate(ranked):
        color = RANK_COLORS[rank]
        out[obj["out_coords"][:, 0], obj["out_coords"][:, 1]] = color

    details = {
        "guide": guide,
        "objects": objects,
        "survivors": placed,
        "erased": erased,
        "ranked": ranked,
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter objects by guide-gap projection, compress the survivors against the guide, then recolor them by area rank.

    Input structure:
    - A black background contains a single horizontal one-cell-thick grey guide bar with one or two internal gaps.
    - Several single-color 4-connected objects of distinct colors sit wholly above or wholly below the guide, with at least one blank row of separation.
    - Object colors are non-black and non-grey, and object areas are all distinct.

    Rule:
    - For each object, inspect the set of columns occupied by its pixels and project those columns onto the guide row.
    - If any projected column lands on a black cell of the guide row, erase that entire object.
    - Move each surviving object vertically toward the guide until its nearest row is adjacent to the guide; if survivors on the same side would overlap after compression, keep the nearer one fixed and push the farther one outward just enough to avoid overlap.
    - Sort the surviving objects by pixel count in descending order and recolor them blue, red, green, then yellow by rank; keep the guide grey and all remaining cells black.

    Key insight:
    - The guide row serves both as the projection filter and as the compression anchor, so gap positions decide survival before the surviving objects can be packed and ranked.
    """
    return _solve(grid)[0]


def _random_composition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    values = []
    remaining = total
    for i in range(parts):
        rest = parts - i - 1
        lo = max(low, remaining - high * rest)
        hi = min(high, remaining - low * rest)
        if lo > hi:
            return None
        values.append(random.randint(lo, hi))
        remaining -= values[-1]
    random.shuffle(values)
    return values if sum(values) == total else None


def _make_guide(width: int) -> dict | None:
    min_span = max(8, math.ceil(width * 0.72))
    for _ in range(200):
        n_gaps = random.choice([1, 2])
        gap_widths = [random.randint(1, 2) for _ in range(n_gaps)]
        n_segments = n_gaps + 1
        min_segment_total = 2 * n_segments
        max_segment_total = width - 2 - sum(gap_widths)
        if max_segment_total < min_segment_total:
            continue
        segment_total = random.randint(max(min_span - sum(gap_widths), min_segment_total), max_segment_total)
        seg_lengths = _random_composition(segment_total, n_segments, 2, max_segment_total)
        if seg_lengths is None or max(seg_lengths) < 4:
            continue
        span_len = segment_total + sum(gap_widths)
        if span_len > width - 2:
            continue
        start = random.randint(1, width - span_len - 1)
        segments = []
        cursor = start
        gap_cols = []
        for i, seg_len in enumerate(seg_lengths):
            segments.append((cursor, cursor + seg_len - 1))
            cursor += seg_len
            if i < n_gaps:
                gap_start = cursor
                gap_end = cursor + gap_widths[i] - 1
                gap_cols.extend(range(gap_start, gap_end + 1))
                cursor = gap_end + 1
        return {"segments": segments, "gap_cols": gap_cols}
    return None


def _feasible_widths(area: int, max_len: int) -> list[int]:
    lo = max(1, math.ceil(area / MAX_OBJECT_HEIGHT))
    hi = min(MAX_OBJECT_WIDTH, max_len, area)
    return list(range(lo, hi + 1))


def _interval_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _interval_candidates_for_gap(area: int, segments: list[tuple[int, int]], gap_cols: list[int]) -> list[tuple[int, int]]:
    if not gap_cols:
        return []
    span_start = segments[0][0]
    span_end = segments[-1][1]
    candidates = []
    guide_cols = {c for s, e in segments for c in range(s, e + 1)}
    for width in _feasible_widths(area, span_end - span_start + 1):
        for start in range(span_start, span_end - width + 2):
            end = start + width - 1
            cols = set(range(start, end + 1))
            if not (cols & set(gap_cols)):
                continue
            if not (cols & guide_cols):
                continue
            candidates.append((start, end))
    random.shuffle(candidates)
    return candidates


def _profile_for_area(area: int, width: int) -> list[int] | None:
    profiles = []
    for _ in range(50):
        profile = _random_composition(area, width, 1, MAX_OBJECT_HEIGHT)
        if profile is None:
            continue
        if width == 1 or len(set(profile)) > 1 or max(profile) > 1:
            profiles.append(profile)
    return random.choice(profiles) if profiles else None


def _make_ideal_coords(side: str, guide_row: int, start_col: int, heights: list[int]) -> np.ndarray:
    coords: list[tuple[int, int]] = []
    for dc, height in enumerate(heights):
        col = start_col + dc
        if side == "top":
            for dr in range(height):
                coords.append((guide_row - 1 - dr, col))
        else:
            for dr in range(height):
                coords.append((guide_row + 1 + dr, col))
    return np.asarray(coords, dtype=int)


def _assign_input_positions(plans: list[dict], height: int, guide_row: int) -> bool:
    by_side = {"top": [], "bottom": []}
    for plan in plans:
        by_side[plan["side"]].append(plan)

    for side, side_plans in by_side.items():
        random.shuffle(side_plans)
        if side == "top":
            cursor = guide_row - 2
            for plan in side_plans:
                obj_height = int(plan["ideal_coords"][:, 0].max() - plan["ideal_coords"][:, 0].min() + 1)
                gap = random.randint(0, 1)
                bottom = cursor - gap
                top = bottom - obj_height + 1
                if top < 0:
                    return False
                distance = (guide_row - 1) - bottom
                shifted = plan["ideal_coords"].copy()
                shifted[:, 0] -= distance
                plan["distance"] = distance
                plan["input_coords"] = shifted
                cursor = top - 2
        else:
            cursor = guide_row + 2
            for plan in side_plans:
                obj_height = int(plan["ideal_coords"][:, 0].max() - plan["ideal_coords"][:, 0].min() + 1)
                gap = random.randint(0, 1)
                top = cursor + gap
                bottom = top + obj_height - 1
                if bottom >= height:
                    return False
                distance = top - (guide_row + 1)
                shifted = plan["ideal_coords"].copy()
                shifted[:, 0] += distance
                plan["distance"] = distance
                plan["input_coords"] = shifted
                cursor = bottom + 2
    return True


def _build_grid(height: int, width: int, guide_row: int, guide: dict, plans: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for start, end in guide["segments"]:
        grid[guide_row, start : end + 1] = GREY
    for plan in plans:
        grid[plan["input_coords"][:, 0], plan["input_coords"][:, 1]] = plan["color"]
    return grid


def _has_clearance(grid: np.ndarray, coords: np.ndarray) -> bool:
    h, w = grid.shape
    for r0, c0 in coords:
        r = int(r0)
        c = int(c0)
        r1 = max(0, r - 1)
        r2 = min(h, r + 2)
        c1 = max(0, c - 1)
        c2 = min(w, c + 2)
        block = grid[r1:r2, c1:c2]
        if np.any(block != BLACK):
            return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out, details = _solve(grid)
    except Exception:
        return False

    objects = details["objects"]
    survivors = details["survivors"]
    erased = details["erased"]
    if len(survivors) not in (3, 4) or len(erased) != 1:
        return False
    if not any(obj["stacked"] for obj in survivors):
        return False
    if not any((set(obj["cols"]) & details["guide"]["gap_cols"]) and (set(obj["cols"]) - details["guide"]["gap_cols"]) for obj in erased):
        return False
    if {obj["side"] for obj in survivors} != {"top", "bottom"}:
        return False

    ranked = details["ranked"]
    output_colors = [int(out[obj["out_coords"][0, 0], obj["out_coords"][0, 1]]) for obj in ranked]
    if output_colors != RANK_COLORS[: len(ranked)]:
        return False
    if sum(obj["color"] != RANK_COLORS[i] for i, obj in enumerate(ranked)) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    guide_row = random.randint(max(5, height // 2 - 1), min(height - 6, height // 2 + 1))
    guide = _make_guide(width)
    if guide is None:
        return None

    survivors_count = random.choice([3, 4])
    stack_side = random.choice(["top", "bottom"])
    other_side = "bottom" if stack_side == "top" else "top"
    if survivors_count == 3:
        survivor_sides = [stack_side, stack_side, other_side]
    else:
        survivor_sides = [stack_side, stack_side, other_side, other_side]
        if random.random() < 0.5:
            survivor_sides[-1] = stack_side
    erased_side = random.choice(["top", "bottom"])

    object_count = survivors_count + 1
    areas = random.sample(list(OBJECT_AREA_RANGE), object_count)
    colors = random.sample(SOURCE_COLORS, object_count)

    plans = []
    for i, side in enumerate(survivor_sides):
        plans.append({"kind": "survivor", "side": side, "area": areas[i], "color": colors[i]})
    plans.append({"kind": "erased", "side": erased_side, "area": areas[-1], "color": colors[-1]})

    long_segments = [seg for seg in guide["segments"] if seg[1] - seg[0] + 1 >= 4]
    if not long_segments:
        return None
    stack_segment = random.choice(long_segments)

    stackers = [plan for plan in plans if plan["kind"] == "survivor" and plan["side"] == stack_side][:2]
    if len(stackers) < 2:
        return None

    pair_candidates = []
    for width_a in _feasible_widths(stackers[0]["area"], stack_segment[1] - stack_segment[0] + 1):
        for start_a in range(stack_segment[0], stack_segment[1] - width_a + 2):
            interval_a = (start_a, start_a + width_a - 1)
            for width_b in _feasible_widths(stackers[1]["area"], stack_segment[1] - stack_segment[0] + 1):
                for start_b in range(stack_segment[0], stack_segment[1] - width_b + 2):
                    interval_b = (start_b, start_b + width_b - 1)
                    if _interval_overlaps(interval_a, interval_b):
                        pair_candidates.append((interval_a, interval_b))
    if not pair_candidates:
        return None
    interval_a, interval_b = random.choice(pair_candidates)
    stackers[0]["interval"] = interval_a
    stackers[1]["interval"] = interval_b

    used_by_side = {
        "top": [plan["interval"] for plan in plans if plan.get("interval") and plan["side"] == "top"],
        "bottom": [plan["interval"] for plan in plans if plan.get("interval") and plan["side"] == "bottom"],
    }

    for plan in plans:
        if "interval" in plan:
            continue
        if plan["kind"] == "survivor":
            candidates = []
            for segment in guide["segments"]:
                seg_len = segment[1] - segment[0] + 1
                for width_obj in _feasible_widths(plan["area"], seg_len):
                    for start in range(segment[0], segment[1] - width_obj + 2):
                        interval = (start, start + width_obj - 1)
                        if any(_interval_overlaps(interval, taken) for taken in used_by_side[plan["side"]]):
                            continue
                        candidates.append(interval)
            if not candidates:
                return None
            plan["interval"] = random.choice(candidates)
        else:
            candidates = [
                interval
                for interval in _interval_candidates_for_gap(plan["area"], guide["segments"], guide["gap_cols"])
                if not any(_interval_overlaps(interval, taken) for taken in used_by_side[plan["side"]])
            ]
            if not candidates:
                return None
            plan["interval"] = random.choice(candidates)
        used_by_side[plan["side"]].append(plan["interval"])

    for plan in plans:
        width_obj = plan["interval"][1] - plan["interval"][0] + 1
        heights = _profile_for_area(plan["area"], width_obj)
        if heights is None:
            return None
        plan["heights"] = heights
        plan["ideal_coords"] = _make_ideal_coords(plan["side"], guide_row, plan["interval"][0], heights)

    if not _assign_input_positions(plans, height, guide_row):
        return None

    grid = np.zeros((height, width), dtype=int)
    for start, end in guide["segments"]:
        grid[guide_row, start : end + 1] = GREY
    for plan in plans:
        if not _has_clearance(grid, plan["input_coords"]):
            return None
        grid[plan["input_coords"][:, 0], plan["input_coords"][:, 1]] = plan["color"]

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
