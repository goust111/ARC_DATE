# Combo variant of db695cfb (index 20)
# combo_spec: D3/1+D10/1+D17/1
# Name: 网格扩展+固定长度+距离编码

"""
Design analysis for the combo variant.

a) Concrete behavior and trace
- The visible ingredients stay close to the source task: blue endpoint pairs define 45-degree
  segments, and magenta trigger pixels can lie on those segments.
- Original task: fill the blue diagonal, then let every triggered magenta ray continue along the
  perpendicular diagonal until it hits the current border.
- Combo task: still fill the same blue diagonal, but every triggered magenta ray now has a fixed
  length equal to floor(blue endpoint distance / 2). After drawing those finite rays in absolute
  coordinates, enlarge the output canvas if needed so the whole drawing fits.
- Example: blue endpoints at (2,2) and (10,10) give segment length 8, so every trigger on that
  segment emits 4 magenta cells in each perpendicular direction. A trigger at (3,3) and another
  at (7,7) both emit length 4 even though their distances to the nearest endpoint differ; this is
  the key difference from the pure D3 variant. If one ray reaches row -1 or col 13, the output is
  expanded with background rows or columns so those coordinates become visible.

b) Complexity calibration
- Only one numeric code is introduced: ray_len = floor(segment_len / 2). That keeps the rule
  inferable from a few examples.
- To make the distance code observable, the generator always uses two intended segments with
  different segment lengths, so at least two ray-length tiers appear in one sample.
- To separate this combo from the trigger-local D3 rule, one segment always carries two triggers
  at different positions but with the same ray length.

c) Implicit-assumption review of the original main()
- `out = grid.copy()` is no longer sufficient because legal output cells may lie outside the input
  rectangle. The combo solver paints in absolute coordinates and materializes the expanded canvas
  afterward.
- The original trigger collector stored only directions because the border decided the stop point.
  The combo solver must store a per-direction finite reach.
- The original while-loop stop condition was the current border. That assumption is invalid once
  the true stop condition is the encoded reach and the border is derived afterward.
- The original longest same-color diagonal-pair heuristic for identifying blue is preserved, but
  the generator must prevent magenta distractors from creating a longer same-color diagonal pair.
- Grouping blue points by diagonal key and connecting each group's extremes still assumes at most
  one intended segment per diagonal family key; the generator enforces unique diagonal ids.

d) Placement legality
- Generated input pixels never overlap one another.
- Output-only blue fills and full finite magenta rays are reserved in one absolute coordinate
  system, so later segments cannot collide with already planned geometry even outside the input.
- Distractors are placed only on cells absent from both the input occupancy set and the reserved
  output set.

e) Visual observability
- Every generated sample has at least one true expansion event, so the larger output shape is not
  optional.
- Every generated sample has at least two distinct encoded ray lengths, so longer blue gaps visibly
  lead to longer magenta extensions.
- Every generated sample differs from the nearest-endpoint D3 baseline because one segment has two
  triggers at different endpoint distances but the same emitted length.

f) Visual cleanliness
- The scene is capped at two intended segments with modest distractor counts.
- Rays and segment interiors are kept non-overlapping, including outside the input frame, which
  avoids dense diagonal tangles in the expanded output.

g) Color inference
- No new color remapping is introduced. Blue remains the segment color and magenta remains the
  trigger/ray color, so the only extra reasoning burden is the length code.

i) New solver assumptions guaranteed by generation
- At least one diagonal-aligned same-color blue pair exists, and no other color forms a longer
  same-color diagonal pair.
- Intended blue endpoints do not form accidental extra diagonal pairs across segments, so the
  solver's "connect extremes on each key" logic recovers exactly the generated segments.
- Cross-segment magenta triggers are not diagonally aligned, so the blue-vs-magenta color choice
  remains unambiguous.
- Full encoded rays may leave the input, but they never overlap reserved geometry, so painting
  background-only cells is well-defined on the expanded canvas.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from arc_utils import get_bg


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _choose_line_color(by_color: dict[int, list[tuple[int, int]]]) -> int:
    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return int(best[1])


def _detect_segments(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], int], list[dict[str, object]]]:
    h, w = grid.shape
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    seed_cells: dict[tuple[int, int], int] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))
        seed_cells[(r, c)] = int(col)

    line_color = _choose_line_color(by_color)
    pts = by_color[line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[dict[str, object]] = []

    def add_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        segments.append(
            {
                "path": path,
                "perp_dirs": [(dr, -dc), (-dr, dc)],
                "ray_limit": abs(q[0] - p[0]) // 2,
            }
        )

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            add_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            add_segment(group[0], group[-1])

    return int(bg), int(line_color), seed_cells, segments


def _paint_expanded_canvas(
    base_shape: tuple[int, int],
    bg: int,
    line_color: int,
    seed_cells: dict[tuple[int, int], int],
    segment_paths: list[list[tuple[int, int]]],
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]],
) -> np.ndarray:
    h, w = base_shape
    painted = {(int(r), int(c)): int(v) for (r, c), v in seed_cells.items()}

    for path in segment_paths:
        for r, c in path:
            if painted.get((r, c), bg) == bg:
                painted[(r, c)] = int(line_color)

    for (r0, c0, tcol), dir_to_reach in trigger_reaches.items():
        for (ddr, ddc), reach in dir_to_reach.items():
            for step in range(1, int(reach) + 1):
                cell = (int(r0 + step * ddr), int(c0 + step * ddc))
                if painted.get(cell, bg) == bg:
                    painted[cell] = int(tcol)

    rows = [0, h - 1]
    cols = [0, w - 1]
    rows.extend(r for r, _ in painted)
    cols.extend(c for _, c in painted)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for (r, c), v in painted.items():
        out[r - min_r, c - min_c] = int(v)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs, emit fixed-length perpendicular rays, and expand the canvas to fit them.

    Input structure:
    - The background is the most common color. One non-background color provides diagonal endpoint pairs, and differently-colored trigger pixels may sit on those same 45-degree diagonals.

    Rule:
    - Fill each detected blue diagonal segment from one extreme endpoint to the other.
    - Whenever that segment passes through a trigger pixel, emit the two perpendicular diagonal rays in the trigger color.
    - Each emitted ray has a fixed length equal to floor(the blue endpoint distance / 2), and the output grid is expanded with background padding whenever those full rays leave the input rectangle.

    Key insight:
    - The blue spacing carries a numeric instruction: every trigger on the same blue segment inherits the same finite ray length, and the final canvas size must be inferred from that encoded reach rather than from the original border.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg, line_color, seed_cells, segments = _detect_segments(grid)

    segment_paths: list[list[tuple[int, int]]] = []
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}
    for seg in segments:
        path = seg["path"]
        perp_dirs = seg["perp_dirs"]
        ray_limit = int(seg["ray_limit"])
        segment_paths.append(path)
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                key = (int(r), int(c), v)
                caps = trigger_reaches.setdefault(key, {})
                for direction in perp_dirs:
                    caps[direction] = max(caps.get(direction, 0), ray_limit)

    return _paint_expanded_canvas((h, w), bg, line_color, seed_cells, segment_paths, trigger_reaches)


# --------------- generator (combo) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (10, 20)
_W_RANGE = (10, 20)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = 2
_SEG_LEN_RANGE = (6, 14)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 1)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 140
_DISTRACTOR_RETRIES = 100
_GEN_RETRIES = 600


def _full_ray_cells(
    trigger_pos: tuple[int, int],
    perp_dirs: list[tuple[int, int]],
    ray_limit: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    tr, tc = trigger_pos
    for ddr, ddc in perp_dirs:
        for step in range(1, ray_limit + 1):
            cells.add((tr + step * ddr, tc + step * ddc))
    return cells


def _ray_exits_input(
    trigger_pos: tuple[int, int],
    perp_dirs: list[tuple[int, int]],
    ray_limit: int,
    h: int,
    w: int,
) -> bool:
    tr, tc = trigger_pos
    for ddr, ddc in perp_dirs:
        for step in range(1, ray_limit + 1):
            r = tr + step * ddr
            c = tc + step * ddc
            if not (0 <= r < h and 0 <= c < w):
                return True
    return False


def _nearest_endpoint_distance(idx: int, seg_len: int) -> int:
    return min(idx, seg_len - idx)


def _aligned_with_any(point: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = point
    return any(abs(r - rr) == abs(c - cc) and r != rr for rr, cc in others)


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    line_points: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            path = [(r0 + k, c0 + k) for k in range(seg_len + 1)]
            diag_key = r0 - c0
            if diag_key in used_backslash:
                continue
            dc = 1
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            path = [(r0 + k, c0 - k) for k in range(seg_len + 1)]
            diag_key = r0 + c0
            if diag_key in used_slash:
                continue
            dc = -1

        ep0, ep1 = path[0], path[-1]
        if any(cell in occupied or cell in blocked for cell in path):
            continue
        if _aligned_with_any(ep0, line_points) or _aligned_with_any(ep1, line_points):
            continue
        return path, dc
    return None


def _candidate_triggers(
    path: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    ray_limit: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[dict[str, object]]:
    seg_len = len(path) - 1
    candidates: list[dict[str, object]] = []
    for idx in range(1, seg_len):
        pos = path[idx]
        ray_cells = _full_ray_cells(pos, perp_dirs, ray_limit)
        if ray_cells & (occupied | blocked):
            continue
        candidates.append(
            {
                "pos": pos,
                "idx": idx,
                "nearest": _nearest_endpoint_distance(idx, seg_len),
                "ray_cells": ray_cells,
                "expands": _ray_exits_input(pos, perp_dirs, ray_limit, h, w),
            }
        )
    return candidates


def _choose_primary_pair(
    candidates: list[dict[str, object]],
    prior_triggers: set[tuple[int, int]],
    require_expand: bool,
) -> list[dict[str, object]] | None:
    pairs = []
    for a, b in combinations(candidates, 2):
        if int(a["nearest"]) == int(b["nearest"]):
            continue
        pos_a = a["pos"]
        pos_b = b["pos"]
        if _aligned_with_any(pos_a, prior_triggers) or _aligned_with_any(pos_b, prior_triggers):
            continue
        expands = bool(a["expands"]) or bool(b["expands"])
        if require_expand and not expands:
            continue
        diff = abs(int(a["nearest"]) - int(b["nearest"]))
        pairs.append((expands, diff, [a, b]))
    if not pairs:
        return None
    pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_expand = pairs[0][0]
    best_diff = pairs[0][1]
    shortlist = [pair for expands, diff, pair in pairs if expands == best_expand and diff == best_diff]
    return random.choice(shortlist)


def _choose_single_trigger(
    candidates: list[dict[str, object]],
    prior_triggers: set[tuple[int, int]],
    require_expand: bool,
) -> dict[str, object] | None:
    usable = []
    for cand in candidates:
        pos = cand["pos"]
        if _aligned_with_any(pos, prior_triggers):
            continue
        if require_expand and not bool(cand["expands"]):
            continue
        usable.append(cand)
    if not usable:
        return None
    usable.sort(key=lambda cand: (bool(cand["expands"]), -int(cand["nearest"])), reverse=True)
    best = bool(usable[0]["expands"])
    shortlist = [cand for cand in usable if bool(cand["expands"]) == best]
    return random.choice(shortlist)


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    h, w = inp.shape
    seed_cells = {
        (int(r), int(c)): int(inp[r, c])
        for r in range(h)
        for c in range(w)
        if int(inp[r, c]) != int(bg)
    }
    segment_paths = [seg["path"] for seg in segments]
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}
    for seg in segments:
        for trigger_pos in seg["trigger_positions"]:
            key = (int(trigger_pos[0]), int(trigger_pos[1]), int(_TRIGGER_COLOR))
            caps = trigger_reaches.setdefault(key, {})
            for direction in seg["perp_dirs"]:
                caps[direction] = max(caps.get(direction, 0), int(seg["ray_limit"]))
    return _paint_expanded_canvas((h, w), bg, _LINE_COLOR, seed_cells, segment_paths, trigger_reaches)


def _build_nearest_distance_baseline(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    h, w = inp.shape
    seed_cells = {
        (int(r), int(c)): int(inp[r, c])
        for r in range(h)
        for c in range(w)
        if int(inp[r, c]) != int(bg)
    }
    segment_paths = [seg["path"] for seg in segments]
    trigger_reaches: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}
    for seg in segments:
        for trigger_pos, nearest in seg["trigger_nearest"].items():
            key = (int(trigger_pos[0]), int(trigger_pos[1]), int(_TRIGGER_COLOR))
            caps = trigger_reaches.setdefault(key, {})
            for direction in seg["perp_dirs"]:
                caps[direction] = max(caps.get(direction, 0), int(nearest))
    return _paint_expanded_canvas((h, w), bg, _LINE_COLOR, seed_cells, segment_paths, trigger_reaches)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    line_points: set[tuple[int, int]] = set()
    trigger_points: set[tuple[int, int]] = set()
    used_backslash: set[int] = set()
    used_slash: set[int] = set()
    segments: list[dict[str, object]] = []
    expanding_seen = False

    # Segment 1: two triggers at different endpoint distances, same encoded ray length.
    first_segment = None
    for _ in range(_SEGMENT_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        result = _try_place_segment(
            slope_type,
            seg_len,
            h,
            w,
            occupied,
            blocked,
            line_points,
            used_backslash,
            used_slash,
        )
        if result is None:
            continue
        path, dc = result
        ray_limit = seg_len // 2
        perp_dirs = [(1, -dc), (-1, dc)]
        candidates = _candidate_triggers(path, perp_dirs, ray_limit, h, w, occupied, blocked)
        chosen = _choose_primary_pair(candidates, trigger_points, require_expand=False)
        if chosen is None:
            continue
        trigger_positions = [cand["pos"] for cand in chosen]
        trigger_nearest = {cand["pos"]: int(cand["nearest"]) for cand in chosen}
        ray_cells = set().union(*(cand["ray_cells"] for cand in chosen))
        first_segment = {
            "path": path,
            "perp_dirs": perp_dirs,
            "ray_limit": ray_limit,
            "trigger_positions": trigger_positions,
            "trigger_nearest": trigger_nearest,
            "seg_len": seg_len,
            "slope_type": slope_type,
            "diag_key": path[0][0] - path[0][1] if slope_type == "backslash" else path[0][0] + path[0][1],
            "ray_cells": ray_cells,
            "expands": any(bool(cand["expands"]) for cand in chosen),
        }
        break
    if first_segment is None:
        return None

    segments.append(first_segment)
    occupied.update([first_segment["path"][0], first_segment["path"][-1]])
    occupied.update(first_segment["trigger_positions"])
    blocked.update(set(first_segment["path"][1:-1]) - set(first_segment["trigger_positions"]))
    blocked.update(first_segment["ray_cells"])
    line_points.update([first_segment["path"][0], first_segment["path"][-1]])
    trigger_points.update(first_segment["trigger_positions"])
    if first_segment["slope_type"] == "backslash":
        used_backslash.add(int(first_segment["diag_key"]))
    else:
        used_slash.add(int(first_segment["diag_key"]))
    expanding_seen = expanding_seen or bool(first_segment["expands"])

    # Segment 2: different blue spacing, hence different encoded ray length.
    second_segment = None
    for _ in range(_SEGMENT_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        ray_limit = seg_len // 2
        if ray_limit == int(first_segment["ray_limit"]):
            continue
        result = _try_place_segment(
            slope_type,
            seg_len,
            h,
            w,
            occupied,
            blocked,
            line_points,
            used_backslash,
            used_slash,
        )
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]
        candidates = _candidate_triggers(path, perp_dirs, ray_limit, h, w, occupied, blocked)
        chosen = _choose_single_trigger(candidates, trigger_points, require_expand=not expanding_seen)
        if chosen is None:
            continue
        trigger_positions = [chosen["pos"]]
        trigger_nearest = {chosen["pos"]: int(chosen["nearest"])}
        second_segment = {
            "path": path,
            "perp_dirs": perp_dirs,
            "ray_limit": ray_limit,
            "trigger_positions": trigger_positions,
            "trigger_nearest": trigger_nearest,
            "seg_len": seg_len,
            "slope_type": slope_type,
            "diag_key": path[0][0] - path[0][1] if slope_type == "backslash" else path[0][0] + path[0][1],
            "ray_cells": set(chosen["ray_cells"]),
            "expands": bool(chosen["expands"]),
        }
        break
    if second_segment is None:
        return None

    segments.append(second_segment)
    occupied.update([second_segment["path"][0], second_segment["path"][-1]])
    occupied.update(second_segment["trigger_positions"])
    blocked.update(set(second_segment["path"][1:-1]) - set(second_segment["trigger_positions"]))
    blocked.update(second_segment["ray_cells"])
    line_points.update([second_segment["path"][0], second_segment["path"][-1]])
    trigger_points.update(second_segment["trigger_positions"])
    if second_segment["slope_type"] == "backslash":
        used_backslash.add(int(second_segment["diag_key"]))
    else:
        used_slash.add(int(second_segment["diag_key"]))
    expanding_seen = expanding_seen or bool(second_segment["expands"])

    if not expanding_seen:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for seg in segments:
        inp[seg["path"][0]] = _LINE_COLOR
        inp[seg["path"][-1]] = _LINE_COLOR
        for pos in seg["trigger_positions"]:
            inp[pos] = _TRIGGER_COLOR

    all_line_positions = set(line_points)
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            pos = (random.randint(0, h - 1), random.randint(0, w - 1))
            if pos in occupied or pos in blocked:
                continue
            if _aligned_with_any(pos, all_line_positions):
                continue
            inp[pos] = _LINE_COLOR
            occupied.add(pos)
            all_line_positions.add(pos)
            break

    all_trigger_positions = set(trigger_points)
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            pos = (random.randint(0, h - 1), random.randint(0, w - 1))
            if pos in occupied or pos in blocked:
                continue
            if _aligned_with_any(pos, all_trigger_positions):
                continue
            inp[pos] = _TRIGGER_COLOR
            occupied.add(pos)
            all_trigger_positions.add(pos)
            break

    expected = _build_expected(inp, bg, segments)
    nearest_baseline = _build_nearest_distance_baseline(inp, bg, segments)
    if expected.shape == inp.shape:
        return None
    if np.array_equal(expected, nearest_baseline):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
