from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
RED = 2
GREEN = 3

GRID_H_RANGE = (17, 25)
GRID_W_RANGE = (17, 25)
BAR_COUNT_RANGE = (2, 4)
BAR_LENGTH_RANGE = (3, 7)
ORIENTATIONS = ("H", "V")
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bar_info(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(coords.shape[0])
    assert h == 1 or w == 1, "every red component must be a straight bar"
    assert area == h * w, "bar must be solid"
    length = max(h, w)
    assert length >= 3, "bar must be long enough to define a diamond"
    orientation = "H" if h == 1 else "V"
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "orientation": orientation,
        "length": int(length),
        "radius": int((length - 1) // 2),
    }


def _paint_diamond(mask: np.ndarray, bar: dict, value: int) -> None:
    r0, c0, r1, c1 = bar["bbox"]
    radius = bar["radius"]
    if bar["orientation"] == "H":
        row = r0
        mask[row, c0 : c1 + 1] = value
        for d in range(1, radius + 1):
            left = c0 + d
            right = c1 - d
            mask[row - d, left : right + 1] = value
            mask[row + d, left : right + 1] = value
    else:
        col = c0
        mask[r0 : r1 + 1, col] = value
        for d in range(1, radius + 1):
            top = r0 + d
            bottom = r1 - d
            mask[top : bottom + 1, col - d] = value
            mask[top : bottom + 1, col + d] = value


def _start_setup(grid: np.ndarray) -> tuple[tuple[int, int], int, int, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "expected a single green start pixel"
    r, c = map(int, pts[0])
    h, w = grid.shape
    if r == 0:
        return (r, c), 0, 1, h - 1
    if r == h - 1:
        return (r, c), 0, -1, 0
    if c == 0:
        return (r, c), 1, 1, w - 1
    if c == w - 1:
        return (r, c), 1, -1, 0
    raise AssertionError("green start must lie on the border")


def _set_axes(main_axis: int, main_value: int, cross_value: int) -> tuple[int, int]:
    return (main_value, cross_value) if main_axis == 0 else (cross_value, main_value)


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert int(np.count_nonzero(g == GREEN)) == 1, "expected one start pixel"

    start, main_axis, forward_sign, goal_value = _start_setup(g)
    cross_size = w if main_axis == 0 else h

    red_components = _components(g == RED)
    assert 2 <= len(red_components) <= 4, "expected 2-4 red bars"
    bars = [_bar_info(comp) for comp in red_components]

    forbidden = np.zeros((h, w), dtype=bool)
    comp_id = np.full((h, w), -1, dtype=int)
    projections: list[dict] = []
    for idx, bar in enumerate(bars):
        temp = np.zeros((h, w), dtype=int)
        _paint_diamond(temp, bar, 1)
        diamond = temp.astype(bool)
        assert not (forbidden & diamond).any(), "diamond keep-out zones must not overlap"
        forbidden |= diamond
        comp_id[diamond] = idx
        coords = np.argwhere(diamond)
        r0, c0, r1, c1 = _bbox(coords)
        if main_axis == 0:
            projections.append(
                {
                    "main_lo": r0,
                    "main_hi": r1,
                    "cross_lo": c0,
                    "cross_hi": c1,
                }
            )
        else:
            projections.append(
                {
                    "main_lo": c0,
                    "main_hi": c1,
                    "cross_lo": r0,
                    "cross_hi": r1,
                }
            )

    sr, sc = start
    forbidden[sr, sc] = False
    comp_id[sr, sc] = -1

    out = g.copy()
    pos_main = sr if main_axis == 0 else sc
    pos_cross = sc if main_axis == 0 else sr
    detours: list[dict] = []

    def paint(main_value: int, cross_value: int) -> None:
        rr, cc = _set_axes(main_axis, main_value, cross_value)
        if out[rr, cc] != RED:
            out[rr, cc] = GREEN

    paint(pos_main, pos_cross)
    step_guard = h * w * 4

    while pos_main != goal_value:
        assert step_guard > 0, "path tracing did not terminate"
        step_guard -= 1

        next_main = pos_main + forward_sign
        rr, cc = _set_axes(main_axis, next_main, pos_cross)
        assert 0 <= rr < h and 0 <= cc < w, "path left the grid before reaching the opposite edge"

        if not forbidden[rr, cc]:
            pos_main = next_main
            paint(pos_main, pos_cross)
            continue

        blocked_id = int(comp_id[rr, cc])
        assert blocked_id >= 0, "blocked cell must belong to a unique diamond"
        proj = projections[blocked_id]

        options = []
        for side_sign in (-1, 1):
            cur_cross = pos_cross
            steps = 0
            while True:
                next_cross = cur_cross + side_sign
                if not (0 <= next_cross < cross_size):
                    break
                sr2, sc2 = _set_axes(main_axis, pos_main, next_cross)
                if forbidden[sr2, sc2]:
                    break
                cur_cross = next_cross
                steps += 1
                if cur_cross < proj["cross_lo"] or cur_cross > proj["cross_hi"]:
                    options.append(
                        {
                            "side_sign": side_sign,
                            "steps": steps,
                            "target_cross": cur_cross,
                        }
                    )
                    break

        assert options, "at least one lateral bypass must exist"
        min_steps = min(opt["steps"] for opt in options)
        best = [opt for opt in options if opt["steps"] == min_steps]
        if len(best) > 1:
            center = (cross_size - 1) / 2.0
            best.sort(
                key=lambda opt: (
                    abs(opt["target_cross"] - center),
                    abs((pos_cross + opt["side_sign"]) - center),
                    opt["target_cross"],
                )
            )
        choice = best[0]
        old_cross = pos_cross
        for _ in range(choice["steps"]):
            pos_cross += choice["side_sign"]
            paint(pos_main, pos_cross)
        detours.append(
            {
                "comp_id": blocked_id,
                "at_main": pos_main,
                "from_cross": old_cross,
                "to_cross": pos_cross,
                "steps": choice["steps"],
            }
        )

    return out, {
        "forbidden": forbidden,
        "bars": bars,
        "detours": detours,
        "projections": projections,
        "start": start,
        "main_axis": main_axis,
        "forward_sign": forward_sign,
        "goal_value": goal_value,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the green path from the edge start to the opposite edge while treating each red bar's hidden diamond envelope as a forbidden zone.

    Input structure:
    - The background is azure (8).
    - There are 2-4 red (2) solid one-cell-thick horizontal or vertical bars of length 3-7.
    - A single green (3) start pixel lies on one border edge.

    Rule:
    - For each red bar, compute its diamond envelope with radius floor((length - 1) / 2): the bar is the middle line and each step away shortens the span by one cell at both ends.
    - The entire diamond envelope, including the bar itself, is a forbidden zone that the path may not enter.
    - Starting from the green pixel, extend the green path toward the opposite parallel edge; when the next forward cell is forbidden, move sideways along the current row/column, choosing the shorter bypass, and if both sides are equally short choose the side closer to the grid center; once outside that diamond's span, resume forward motion.

    Key insight:
    - The red bar does not just block its own pixels: its geometric length implicitly determines a larger hidden diamond keep-out region, and the visible path is the result of routing around those invisible envelopes.
    """
    out, _trace = _solve_with_trace(grid)
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _projected_width(orientation: str, length: int) -> int:
    radius = (length - 1) // 2
    return length if orientation == "H" else 2 * radius + 1


def _diamond_height(orientation: str, length: int) -> int:
    radius = (length - 1) // 2
    return 2 * radius + 1 if orientation == "H" else length


def _transform_from_top(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid.T).copy()
    raise ValueError(edge)


def _choose_bar_specs(count: int) -> list[dict]:
    specs = []
    for _ in range(count):
        specs.append(
            {
                "orientation": random.choice(ORIENTATIONS),
                "length": random.randint(*BAR_LENGTH_RANGE),
            }
        )
    return specs


def _interval_candidates(width: int, cur_cross: int, spec: dict) -> list[dict]:
    proj_width = _projected_width(spec["orientation"], spec["length"])
    candidates = []
    lo_min = max(0, cur_cross - proj_width + 1)
    lo_max = min(cur_cross, width - proj_width)
    center = (width - 1) / 2.0

    for lo in range(lo_min, lo_max + 1):
        hi = lo + proj_width - 1
        left_steps = cur_cross - (lo - 1)
        right_steps = (hi + 1) - cur_cross
        left_target = lo - 1
        right_target = hi + 1
        if left_target < 0 or right_target >= width:
            continue

        if left_steps < right_steps:
            chosen = "left"
            target = left_target
            steps = left_steps
        elif right_steps < left_steps:
            chosen = "right"
            target = right_target
            steps = right_steps
        else:
            if abs(cur_cross - center) < 0.5:
                continue
            chosen = "right" if cur_cross < center else "left"
            target = right_target if chosen == "right" else left_target
            steps = left_steps

        mode = "tie" if left_steps == right_steps else chosen
        candidates.append(
            {
                "projection_lo": lo,
                "projection_hi": hi,
                "target_cross": target,
                "steps": steps,
                "mode": mode,
                "chosen": chosen,
                "proj_width": proj_width,
            }
        )
    return candidates


def _choose_interval(width: int, cur_cross: int, spec: dict, center_bias: bool) -> dict | None:
    candidates = _interval_candidates(width, cur_cross, spec)
    if not candidates:
        return None

    if center_bias:
        weighted = []
        for cand in candidates:
            weight = 1
            if cand["mode"] == "tie":
                weight = 5
            weighted.extend([cand] * weight)
        return random.choice(weighted)
    return random.choice(candidates)


def _build_top_instance() -> tuple[np.ndarray, dict] | None:
    count = random.randint(*BAR_COUNT_RANGE)
    specs = _choose_bar_specs(count)
    heights = [_diamond_height(spec["orientation"], spec["length"]) for spec in specs]
    min_h = sum(heights) + (count + 1)
    max_h = GRID_H_RANGE[1] - 1
    if min_h > max_h:
        return None

    h = random.randint(max(GRID_H_RANGE[0], min_h), max_h)
    w = random.randint(*GRID_W_RANGE)
    start_c = random.randint(1, w - 2)
    force_tie = random.random() < 0.45
    tie_index = random.randrange(count) if force_tie else -1

    placements = []
    cur_cross = start_c
    tie_used = False
    for idx, spec in enumerate(specs):
        need_tie = force_tie and not tie_used and idx == tie_index
        if need_tie:
            tie_candidates = [cand for cand in _interval_candidates(w, cur_cross, spec) if cand["mode"] == "tie"]
            interval = random.choice(tie_candidates) if tie_candidates else _choose_interval(w, cur_cross, spec, center_bias=True)
        else:
            interval = _choose_interval(w, cur_cross, spec, center_bias=False)
        if interval is None:
            return None
        tie_used = tie_used or interval["mode"] == "tie"
        placements.append({**spec, **interval})
        cur_cross = interval["target_cross"]

    slack = h - min_h
    gaps = [g + 1 for g in _random_composition(slack, count + 1)]
    top = gaps[0]
    for i, placement in enumerate(placements):
        placement["top"] = top
        top += heights[i] + gaps[i + 1]

    grid = np.full((h, w), BG, dtype=int)
    grid[0, start_c] = GREEN
    used = np.zeros((h, w), dtype=bool)
    for placement in placements:
        length = placement["length"]
        radius = (length - 1) // 2
        lo = placement["projection_lo"]
        if placement["orientation"] == "H":
            bar_row = placement["top"] + radius
            grid[bar_row, lo : lo + length] = RED
            temp = np.zeros((h, w), dtype=int)
            _paint_diamond(
                temp,
                {
                    "bbox": (bar_row, lo, bar_row, lo + length - 1),
                    "orientation": "H",
                    "length": length,
                    "radius": radius,
                },
                1,
            )
        else:
            col = lo + radius
            top_row = placement["top"]
            grid[top_row : top_row + length, col] = RED
            temp = np.zeros((h, w), dtype=int)
            _paint_diamond(
                temp,
                {
                    "bbox": (top_row, col, top_row + length - 1, col),
                    "orientation": "V",
                    "length": length,
                    "radius": radius,
                },
                1,
            )
        diamond = temp.astype(bool)
        if (used & diamond).any():
            return None
        used |= diamond

    return grid, {"placements": placements, "start_c": start_c}


def _green_path_cells(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    return (out == GREEN) & ((inp == BG) | (inp == GREEN))


def _validate_generated(inp: np.ndarray, meta: dict) -> bool:
    try:
        out, trace = _solve_with_trace(inp)
    except Exception:
        return False

    if not np.any(out[-1, :] == GREEN):
        return False
    if len(trace["detours"]) != len(trace["bars"]):
        return False
    if {d["comp_id"] for d in trace["detours"]} != set(range(len(trace["bars"]))):
        return False

    green_mask = _green_path_cells(inp, out)
    if np.any(trace["forbidden"] & green_mask):
        return False

    start = trace["start"]
    if out[start] != GREEN:
        return False
    if int(np.count_nonzero(inp == GREEN)) != 1:
        return False

    for detour in trace["detours"]:
        if detour["steps"] <= 0 or detour["from_cross"] == detour["to_cross"]:
            return False

    for bar in trace["bars"]:
        if not (3 <= bar["length"] <= 7):
            return False

    # Semantic check: the straight-ahead cell immediately before each obstacle must be blocked.
    for detour in trace["detours"]:
        forward_main = detour["at_main"] + trace["forward_sign"]
        rr, cc = _set_axes(trace["main_axis"], forward_main, detour["from_cross"])
        if not trace["forbidden"][rr, cc]:
            return False

    _ = meta
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_top_instance()
        if built is None:
            continue
        canonical, meta = built
        if not _validate_generated(canonical, meta):
            continue
        edge = random.choice(("top", "bottom", "left", "right"))
        candidate = _transform_from_top(canonical, edge)
        try:
            out = main(candidate)
        except Exception:
            continue
        if candidate.shape[0] > 30 or candidate.shape[1] > 30:
            continue
        if out.shape[0] > 30 or out.shape[1] > 30:
            continue
        return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
