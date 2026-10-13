from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paste


BG = 0
FENCE = 5
WALL_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
HEIGHT_RANGE = (12, 18)
WIDTH_RANGE = (20, 27)
GENERATION_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _row_profile(coords: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(r) for r in coords[:, 0]}))


def _normalize(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_objects(grid: np.ndarray, color: int, bg: int) -> list[dict]:
    inner = np.asarray(grid, dtype=int).copy()
    inner[:, 0] = bg
    inner[:, -1] = bg
    return find_components(
        inner,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )


def _fence_columns(grid: np.ndarray, bg: int) -> list[int]:
    cols = []
    for c in range(1, grid.shape[1] - 1):
        col_vals = set(map(int, np.unique(grid[:, c])))
        if FENCE in col_vals:
            assert col_vals <= {bg, FENCE}, "fence columns must contain only background or grey"
            cols.append(c)
    assert cols, "expected at least one fence column"
    return cols


def _gate_intervals(grid: np.ndarray, col: int) -> list[tuple[int, int]]:
    vals = np.asarray(grid[:, col], dtype=int)
    gates = []
    r = 0
    while r < len(vals):
        while r < len(vals) and int(vals[r]) == FENCE:
            r += 1
        if r >= len(vals):
            break
        start = r
        while r < len(vals) and int(vals[r]) != FENCE:
            r += 1
        gates.append((int(start), int(r - 1)))
    return gates


def _profile_fits_gate(rows: tuple[int, ...], gates: list[tuple[int, int]]) -> bool:
    for r0, r1 in gates:
        if all(r0 <= row <= r1 for row in rows):
            return True
    return False


def _sort_key(comp: dict, wall_col: int, width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if wall_col == 0:
        return int(c0), int(r0), int(c1), int(r1)
    return int(width - 1 - c1), int(r0), int(c0), int(r1)


def _simulate_one(
    coords: np.ndarray,
    *,
    direction: int,
    occupied_kind: np.ndarray,
    fence_cols: list[int],
    gates_by_col: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, dict]:
    cur = np.asarray(coords, dtype=int).copy()
    rows = _row_profile(cur)
    passed: list[int] = []

    while True:
        proposed = cur + np.array([0, direction], dtype=int)
        if (
            proposed[:, 0].min() < 0
            or proposed[:, 0].max() >= occupied_kind.shape[0]
            or proposed[:, 1].min() < 0
            or proposed[:, 1].max() >= occupied_kind.shape[1]
        ):
            return cur, {"reason": "wall", "passed_fences": tuple(passed), "blocked_fence": None}

        min_c = int(cur[:, 1].min())
        max_c = int(cur[:, 1].max())
        prop_min = int(proposed[:, 1].min())
        prop_max = int(proposed[:, 1].max())

        entering = []
        if direction == -1:
            for fence_col in fence_cols:
                if prop_min <= fence_col < min_c:
                    entering.append(int(fence_col))
        else:
            for fence_col in fence_cols:
                if max_c < fence_col <= prop_max:
                    entering.append(int(fence_col))
        assert len(entering) <= 1, "object should encounter at most one fence per step"
        if entering:
            fence_col = entering[0]
            if not _profile_fits_gate(rows, gates_by_col[fence_col]):
                return cur, {"reason": "fence", "passed_fences": tuple(passed), "blocked_fence": int(fence_col)}
            passed.append(int(fence_col))

        hit = occupied_kind[proposed[:, 0], proposed[:, 1]]
        if np.any(hit == 2):
            return cur, {"reason": "object", "passed_fences": tuple(passed), "blocked_fence": None}
        if np.any(hit == 1):
            return cur, {"reason": "wall", "passed_fences": tuple(passed), "blocked_fence": None}
        cur = proposed


def _solve(grid: np.ndarray, *, traces: bool = False) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])
    assert left_color != right_color
    assert left_color not in {bg, FENCE} and right_color not in {bg, FENCE}
    assert np.all(g[:, 0] == left_color)
    assert np.all(g[:, w - 1] == right_color)

    fence_cols = _fence_columns(g, bg)
    gates_by_col = {col: _gate_intervals(g, col) for col in fence_cols}

    out = np.full_like(g, bg)
    out[:, 0] = left_color
    out[:, -1] = right_color
    out[g == FENCE] = FENCE

    all_traces: list[dict] = []
    for color, wall_col, direction in ((left_color, 0, -1), (right_color, w - 1, 1)):
        fixed = np.zeros_like(g, dtype=np.uint8)
        fixed[:, 0] = 1
        fixed[:, -1] = 1
        fixed[g == FENCE] = 1
        occupied_kind = fixed.copy()

        comps = _extract_objects(g, color, bg)
        comps.sort(key=lambda comp: _sort_key(comp, wall_col, w))

        for comp in comps:
            final_coords, trace = _simulate_one(
                comp["coords"],
                direction=direction,
                occupied_kind=occupied_kind,
                fence_cols=fence_cols,
                gates_by_col=gates_by_col,
            )
            assert not np.any(out[final_coords[:, 0], final_coords[:, 1]] != bg), "cross-color overlap"
            out[final_coords[:, 0], final_coords[:, 1]] = color
            occupied_kind[final_coords[:, 0], final_coords[:, 1]] = 2
            if traces:
                all_traces.append(
                    {
                        "color": int(color),
                        "start_bbox": _bbox(comp["coords"]),
                        "final_bbox": _bbox(final_coords),
                        "stop_reason": trace["reason"],
                        "blocked_fence": trace["blocked_fence"],
                        "passed_fences": trace["passed_fences"],
                    }
                )

    assert int(np.sum(out == left_color)) == int(np.sum(g == left_color))
    assert int(np.sum(out == right_color)) == int(np.sum(g == right_color))
    return out, all_traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each object sideways toward its same-colored wall, letting only row profiles that fit a single gate pass each grey fence.

    Input structure:
    - The leftmost and rightmost columns are full-height solid walls of two different colors.
    - Interior 8-connected objects use exactly those wall colors and sit away from their matching wall.
    - One or more grey fence columns stand between the walls, each with one or more vertical gate openings.

    Rule:
    - For each wall color separately, process that color's objects from nearest to the matching wall to farthest.
    - Each object keeps its rows and rigid shape, sliding only horizontally toward its same-colored wall.
    - When an object first reaches a fence, it may pass only if all rows occupied by the object lie inside one continuous gate of that fence; otherwise it stops flush against the fence.
    - A passing object keeps sliding until it hits the wall or a previously settled object of the same color. Walls and fences stay fixed.

    Key insight:
    - The decisive feature is the object's row profile across the whole component: distance alone does not determine the result, because different fences accept or reject different row bands.
    """
    return _solve(np.asarray(grid, dtype=int), traces=False)[0]


def _random_connected_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    start = rng.randint(0, max(0, width - 1))
    length = rng.randint(1, width)
    end = min(width - 1, start + length - 1)
    rows.append((start, end))
    for _ in range(1, height):
        prev_start, prev_end = rows[-1]
        lo = max(0, prev_start - 1)
        hi = min(width - 1, prev_end + 1)
        start = rng.randint(lo, hi)
        max_len = width - start
        min_len = 1
        length = rng.randint(min_len, max_len)
        end = start + length - 1
        if end < prev_start - 1 or start > prev_end + 1:
            start = max(0, min(width - 1, prev_start))
            end = min(width - 1, max(start, prev_end))
        rows.append((start, end))
    mask = np.zeros((height, width), dtype=bool)
    for r, (c0, c1) in enumerate(rows):
        mask[r, c0 : c1 + 1] = True
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return _normalize(mask)


def _sample_bands(height: int, rng: random.Random) -> dict[str, tuple[int, int]]:
    for _ in range(200):
        top_h = rng.randint(2, 4)
        mid_h = rng.randint(2, 4)
        bot_h = rng.randint(2, 4)
        gap1 = rng.randint(1, 2)
        gap2 = rng.randint(1, 2)
        used = top_h + mid_h + bot_h + gap1 + gap2
        if used > height - 2:
            continue
        offset = rng.randint(0, height - 2 - used)
        top0 = 1 + offset
        mid0 = top0 + top_h + gap1
        bot0 = mid0 + mid_h + gap2
        return {
            "top": (top0, top0 + top_h - 1),
            "mid": (mid0, mid0 + mid_h - 1),
            "bottom": (bot0, bot0 + bot_h - 1),
        }
    raise RuntimeError("failed to sample row bands")


def _interval_inside(band: tuple[int, int], rng: random.Random, *, force_height: int | None = None) -> tuple[int, int]:
    r0, r1 = band
    band_h = r1 - r0 + 1
    height = force_height if force_height is not None else rng.randint(2, band_h)
    start = rng.randint(r0, r1 - height + 1)
    return int(start), int(start + height - 1)


def _fence_plan(fence_cols: list[int], bands: dict[str, tuple[int, int]], rng: random.Random) -> dict[int, list[tuple[int, int]]]:
    plan: dict[int, list[tuple[int, int]]] = {}
    if len(fence_cols) == 2:
        base = [["top", "mid"], ["top", "bottom"]]
    else:
        base = [["top", "mid"], ["top", "mid", "bottom"], ["top", "bottom"]]
    used_rows = {row for interval in bands.values() for row in range(interval[0], interval[1] + 1)}
    spare_rows = [r for r in range(1, max(used_rows) + 2) if r not in used_rows]
    for col, names in zip(fence_cols, base):
        gates = [bands[name] for name in names]
        if spare_rows and rng.random() < 0.5:
            row = rng.choice(spare_rows)
            gates.append((row, row))
        plan[int(col)] = sorted(gates)
    return plan


def _draw_fences(grid: np.ndarray, gates_by_col: dict[int, list[tuple[int, int]]]) -> None:
    for col, gates in gates_by_col.items():
        grid[:, col] = FENCE
        for r0, r1 in gates:
            grid[r0 : r1 + 1, col] = BG


def _compose_widths(width: int, n_fences: int, rng: random.Random) -> tuple[list[int], list[int]]:
    free = width - 2 - n_fences
    if n_fences == 2:
        left_outer = rng.randint(5, 7)
        right_outer = rng.randint(6, 8)
        middle = free - left_outer - right_outer
        if middle < 3:
            raise ValueError("width too small for two fences")
        spans = [left_outer, middle, right_outer]
    else:
        left_outer = rng.randint(5, 7)
        right_outer = rng.randint(6, 8)
        gap1 = rng.randint(2, 4)
        gap2 = free - left_outer - right_outer - gap1
        if gap2 < 2:
            raise ValueError("width too small for three fences")
        spans = [left_outer, gap1, gap2, right_outer]

    fence_cols = []
    cursor = 0
    for span in spans[:-1]:
        cursor += span
        fence_cols.append(cursor + len(fence_cols) + 1)
    return spans, fence_cols


def _build_object_patch(color: int, row_interval: tuple[int, int], width: int, rng: random.Random) -> np.ndarray:
    height = row_interval[1] - row_interval[0] + 1
    mask = _random_connected_mask(height, width, rng)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _candidate_left_cols(start: int, end: int, widths: list[int], rng: random.Random) -> list[int]:
    if len(widths) < 2:
        raise ValueError("need at least two left objects")
    first = int(start)
    second = int(first + widths[0] + 1)
    if second > end:
        raise ValueError("left top pair cannot fit")
    choices = list(range(second + 1, end + 1))
    if len(choices) < len(widths) - 2:
        raise ValueError("not enough left columns")
    others = sorted(rng.sample(choices, len(widths) - 2))
    return [first, second, *others]


def _candidate_right_maxes(start: int, end: int, widths: list[int], rng: random.Random) -> list[int]:
    if len(widths) < 2:
        raise ValueError("need at least two right objects")
    first = int(end)
    second = int(first - widths[0] - 1)
    if second < start:
        raise ValueError("right top pair cannot fit")
    choices = list(range(start, second))
    if len(choices) < len(widths) - 2:
        raise ValueError("not enough right columns")
    others = sorted(rng.sample(choices, len(widths) - 2), reverse=True)
    return [first, second, *others]


def _make_trace_key(color: int, bbox: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int, int]]:
    return int(color), tuple(map(int, bbox))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    n_fences = rng.choice([2, 3])
    try:
        spans, fence_cols = _compose_widths(width, n_fences, rng)
    except ValueError:
        return None

    bands = _sample_bands(height, rng)
    gates_by_col = _fence_plan(fence_cols, bands, rng)
    left_color, right_color = rng.sample(WALL_PALETTE, 2)

    grid = np.full((height, width), BG, dtype=int)
    grid[:, 0] = left_color
    grid[:, -1] = right_color
    _draw_fences(grid, gates_by_col)

    left_top_interval = _interval_inside(bands["top"], rng)
    right_top_interval = _interval_inside(bands["top"], rng)
    left_mid_interval = _interval_inside(bands["mid"], rng)
    left_bottom_interval = _interval_inside(bands["bottom"], rng)
    right_mid_interval = _interval_inside(bands["mid"], rng)
    right_bottom_interval = _interval_inside(bands["bottom"], rng)

    left_specs = [
        ("left_wall", left_top_interval, rng.randint(1, 2)),
        ("left_stack", left_top_interval, rng.randint(1, 2)),
        ("left_first_fence", left_mid_interval, rng.randint(1, 2)),
        ("left_late_fence", left_bottom_interval, rng.randint(1, 2)),
    ]
    right_specs = [
        ("right_wall", right_top_interval, rng.randint(1, 2)),
        ("right_stack", right_top_interval, rng.randint(1, 2)),
        ("right_late_fence", right_mid_interval, rng.randint(1, 2)),
        ("right_first_fence", right_bottom_interval, rng.randint(1, 2)),
    ]

    right_outer_start = fence_cols[-1] + 1
    right_outer_end = width - 2
    left_outer_start = 1
    left_outer_end = fence_cols[0] - 1
    if right_outer_end - right_outer_start + 1 < 6 or left_outer_end - left_outer_start + 1 < 6:
        return None

    try:
        left_cols = _candidate_left_cols(
            right_outer_start,
            right_outer_end,
            [spec[2] for spec in left_specs],
            rng,
        )
        right_maxes = _candidate_right_maxes(
            left_outer_start,
            left_outer_end,
            [spec[2] for spec in right_specs],
            rng,
        )
    except ValueError:
        return None

    plan_expectations: dict[tuple[int, tuple[int, int, int, int]], tuple[str, int | None, int]] = {}

    for (label, interval, width_cells), c0 in zip(left_specs, left_cols):
        patch = _build_object_patch(left_color, interval, width_cells, rng)
        r0 = interval[0]
        if c0 + patch.shape[1] - 1 > right_outer_end:
            return None
        if np.any(grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] != BG):
            return None
        paste(grid, patch, r0, c0, transparent=BG)
        bbox = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
        if label == "left_wall":
            plan_expectations[_make_trace_key(left_color, bbox)] = ("wall", None, 0)
        elif label == "left_stack":
            plan_expectations[_make_trace_key(left_color, bbox)] = ("object", None, 0)
        elif label == "left_first_fence":
            plan_expectations[_make_trace_key(left_color, bbox)] = ("fence", fence_cols[-1], 0)
        else:
            plan_expectations[_make_trace_key(left_color, bbox)] = ("fence", fence_cols[0], 1)

    for (label, interval, width_cells), max_c in zip(right_specs, right_maxes):
        patch = _build_object_patch(right_color, interval, width_cells, rng)
        c0 = max_c - patch.shape[1] + 1
        r0 = interval[0]
        if c0 < left_outer_start:
            return None
        if np.any(grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] != BG):
            return None
        paste(grid, patch, r0, c0, transparent=BG)
        bbox = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
        if label == "right_wall":
            plan_expectations[_make_trace_key(right_color, bbox)] = ("wall", None, 0)
        elif label == "right_stack":
            plan_expectations[_make_trace_key(right_color, bbox)] = ("object", None, 0)
        elif label == "right_late_fence":
            plan_expectations[_make_trace_key(right_color, bbox)] = ("fence", fence_cols[-1], 1)
        else:
            plan_expectations[_make_trace_key(right_color, bbox)] = ("fence", fence_cols[0], 0)

    try:
        out, traces = _solve(grid, traces=True)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if len(traces) != len(plan_expectations):
        return None

    seen = set()
    first_block = False
    late_block = False
    hit_object = False
    hit_wall = False
    for trace in traces:
        key = _make_trace_key(trace["color"], trace["start_bbox"])
        if key not in plan_expectations or key in seen:
            return None
        seen.add(key)
        expect_reason, expect_fence, min_passed = plan_expectations[key]
        if trace["stop_reason"] != expect_reason:
            return None
        if expect_fence is not None and trace["blocked_fence"] != expect_fence:
            return None
        if len(trace["passed_fences"]) < min_passed:
            return None
        if trace["stop_reason"] == "object":
            hit_object = True
        if trace["stop_reason"] == "wall":
            hit_wall = True
        if trace["stop_reason"] == "fence" and len(trace["passed_fences"]) == 0:
            first_block = True
        if trace["stop_reason"] == "fence" and len(trace["passed_fences"]) >= 1:
            late_block = True

    if not (first_block and late_block and hit_object and hit_wall):
        return None
    if len(set(map(int, np.unique(grid)))) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-profile fence puzzle")
