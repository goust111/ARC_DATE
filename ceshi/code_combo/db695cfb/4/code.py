# Combo variant of db695cfb (index 4)
# combo_spec: D2/3+D5/0
# Name: 单侧延伸+颜色循环

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "solvers"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The stable scaffold from db695cfb stays intact: blue endpoints define 45-degree
#      diagonals, the output completes each such diagonal, and magenta pixels on the
#      completed diagonal act as triggers.
#    - Two source-stage behaviors are changed together.
#      * D2/3: a trigger no longer emits both perpendicular rays; it emits only the branch
#        selected by its position relative to the visible global diagonal r=c.
#      * D5/0: the emitted ray no longer keeps the trigger's magenta color; the ray color is
#        chosen by trigger order along that completed blue segment.
#    - Small fused trace on a slash segment:
#      * Input endpoints at (1, 6) and (6, 1) imply the completed "/" path
#        (1,6),(2,5),(3,4),(4,3),(5,2),(6,1).
#      * Put triggers at (2,5) and (4,3). In the source task, both would emit two magenta
#        "\" rays.
#      * In the combo task, path order is top-to-bottom. Trigger (2,5) is first and satisfies
#        c>r, so it emits only the upward perpendicular branch and that branch is red.
#      * Trigger (4,3) is second and satisfies c<r, so it emits only the downward
#        perpendicular branch and that branch is green.
#      * The visible differences from the source are therefore simultaneous: half the rays
#        disappear, and the surviving rays are ordered red then green instead of all magenta.
#
# b) Complexity calibration
#    - The side rule has only two categories: above r=c versus below r=c.
#    - The color rule uses a short fixed cycle [red, green, yellow], which is enough to show
#      ordering without flooding the grid with colors.
#    - The generator forces one showcase slash segment whose active triggers appear on both
#      sides of r=c and whose count is at least two. That single object demonstrates the
#      one-sided geometry and the color cycle at once.
#    - The cycle resets per segment. A global counter across unrelated diagonals would be much
#      harder to infer from 2-4 ARC examples.
#
# c) Audit of source-code assumptions broken by the combo
#    - Original `draw_segment()` assumed every trigger contributes both perpendicular
#      directions. That is false now; each active trigger contributes exactly one direction.
#    - Original ray painting assumed the trigger color itself should be extended. That is also
#      false now; ray color depends on local trigger index along the segment.
#    - Combining the two changes introduces a new segment-local assumption: trigger order must
#      be recovered along each completed diagonal before rays are drawn.
#    - Another new assumption is that active triggers on r=c would be ambiguous for D2/3.
#      The solver ignores such cases, and the generator forbids them for active triggers.
#    - Segment overlap would make segment-local ordering and direction assignment ambiguous, so
#      the generator reserves all path and ray cells and rejects colliding placements.
#
# d) Placement legality
#    - Every endpoint, active trigger, distractor, completed path cell, and emitted ray cell
#      must stay in bounds and pairwise disjoint except that a trigger lies on its own segment.
#    - The generator constructs output-first, reserves diagonal fills and one-sided rays, then
#      adds distractors only on untouched background cells.
#    - Reusing the same solved diagonal is disallowed because the solver connects the extreme
#      endpoints on each diagonal group.
#
# e) Visual observability
#    - The side cue uses a human-visible spatial landmark, the main diagonal r=c; no hidden
#      ordering by color id or object creation time is used.
#    - Each generated puzzle contains at least one segment with two active triggers ordered
#      along the same blue path, so the red->green progression is directly observable.
#    - The same showcase segment also contains triggers on both sides of r=c, so the opposite
#      one-sided branches are visible in one training example instead of being scattered across
#      unrelated objects.
#
# f) Visual cleanliness
#    - Segment count is capped below the original maximum, and any path/ray collision is
#      rejected. This avoids layered multi-color intersections that would hide the fused rule.
#    - Off-diagonal magenta distractors are kept away from completed paths and off r=c so they
#      do not suggest a nonexistent "main-diagonal trigger" case.
#
# g) Change inferability
#    - Only three ray colors are introduced, and they are tied to a structural cue: first,
#      second, third active trigger along a completed segment.
#    - Because the showcase segment simultaneously exposes order and side, a human can explain
#      the output using two independent visible rules instead of memorizing arbitrary colors.


_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_RAY_COLOR_CYCLE = (2, 3, 4)  # red, green, yellow

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: combo visibility guards ===
_SHOWCASE_TRIGGER_RANGE = (2, 3)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _ray_color_for_trigger(index: int) -> int:
    return int(_RAY_COLOR_CYCLE[index % len(_RAY_COLOR_CYCLE)])


def _side_of_main_diagonal(r: int, c: int) -> int:
    if c > r:
        return 1
    if c < r:
        return -1
    return 0


def _single_perp_dir_from_segment(dr: int, dc: int, r: int, c: int) -> tuple[int, int] | None:
    """
    Return the only surviving perpendicular direction for an active trigger.

    Above r=c means "upward" along the perpendicular diagonal; below r=c means "downward".
    """
    side = _side_of_main_diagonal(r, c)
    if side == 0:
        return None
    slope_sign = dr * dc  # +1 for "\", -1 for "/"
    if side > 0:
        return -1, slope_sign
    return 1, -slope_sign


def _single_perp_dir_from_dc(dc: int, r: int, c: int) -> tuple[int, int] | None:
    """Generator helper: segments are stored in top-to-bottom order, so dr=+1."""
    side = _side_of_main_diagonal(r, c)
    if side == 0:
        return None
    if side > 0:
        return -1, dc
    return 1, -dc


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _find_line_color(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (distance, color)
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
    _, line_color = best
    return line_color, by_color[line_color]


def _collect_segments(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    pts: list[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], int, list[tuple[int, int, tuple[int, int]]]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[tuple[list[tuple[int, int]], int, list[tuple[int, int, tuple[int, int]]]]] = []

    def add_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        ordered_triggers: list[tuple[int, int, tuple[int, int]]] = []
        for r, c in path:
            v = int(grid[r, c])
            if v == bg or v == line_color:
                continue
            single_dir = _single_perp_dir_from_segment(dr, dc, r, c)
            if single_dir is None:
                continue
            ordered_triggers.append((r, c, single_dir))
        segments.append((path, dc, ordered_triggers))

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            add_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            add_segment(g[0], g[-1])

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal endpoint pairs, then let each trigger emit one ordered colored perpendicular ray.

    Input structure:
    - The background is the dominant color. One non-background color marks endpoint pairs on 45-degree diagonals, and magenta pixels may lie on the completed diagonals as candidate triggers. Valid active triggers are off the global main diagonal r=c.

    Rule:
    - Fill each diagonal segment between the extreme same-color endpoints on that diagonal.
    - Walk each completed segment from its upper endpoint to its lower endpoint. The 1st active trigger on that segment emits a red perpendicular ray, the 2nd emits green, the 3rd emits yellow, then the cycle repeats.
    - Each active trigger emits only one of the two perpendicular branches: cells with c>r use the upward branch, while cells with c<r use the downward branch.

    Key insight:
    - Two independent visible cues cooperate: trigger order along a finished blue diagonal determines the ray color, while trigger position relative to r=c determines which half of the rotated beam survives.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    h, w = grid.shape

    bg = get_bg(grid)
    line_color, pts = _find_line_color(grid, bg)
    segments = _collect_segments(grid, bg, line_color, pts)

    seen_triggers: set[tuple[int, int]] = set()
    trigger_specs: list[tuple[int, int, int, tuple[int, int]]] = []

    for path, _, ordered_triggers in segments:
        for r, c in path:
            if int(out[r, c]) == bg:
                out[r, c] = line_color

        for idx, (r, c, ray_dir) in enumerate(ordered_triggers):
            if (r, c) in seen_triggers:
                continue
            seen_triggers.add((r, c))
            trigger_specs.append((r, c, _ray_color_for_trigger(idx), ray_dir))

    for r0, c0, ray_color, (ddr, ddc) in trigger_specs:
        r, c = r0 + ddr, c0 + ddc
        while 0 <= r < h and 0 <= c < w:
            if int(out[r, c]) == bg:
                out[r, c] = ray_color
            r += ddr
            c += ddc

    return out


def _compute_single_ray_cells(
    trigger_positions: list[tuple[int, int]],
    dc: int,
    height: int,
    width: int,
) -> tuple[set[tuple[int, int]] | None, dict[tuple[int, int], tuple[int, int]] | None]:
    cells: set[tuple[int, int]] = set()
    trigger_dirs: dict[tuple[int, int], tuple[int, int]] = {}
    for tr, tc in trigger_positions:
        single_dir = _single_perp_dir_from_dc(dc, tr, tc)
        if single_dir is None:
            return None, None
        trigger_dirs[(tr, tc)] = single_dir
        ddr, ddc = single_dir
        r, c = tr + ddr, tc + ddc
        while 0 <= r < height and 0 <= c < width:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells, trigger_dirs


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]],
) -> np.ndarray:
    height, width = inp.shape
    expected = inp.copy()
    for path, trigger_positions, trigger_dirs in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for idx, (tr, tc) in enumerate(trigger_positions):
            ray_color = _ray_color_for_trigger(idx)
            ddr, ddc = trigger_dirs[(tr, tc)]
            r, c = tr + ddr, tc + ddc
            while 0 <= r < height and 0 <= c < width:
                if int(expected[r, c]) == bg:
                    expected[r, c] = ray_color
                r += ddr
                c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_diag_keys: set[tuple[str, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if height - 1 - seg_len < 0 or width - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, height - 1 - seg_len)
            c0 = random.randint(0, width - 1 - seg_len)
            dc = 1
        else:
            if height - 1 - seg_len < 0 or seg_len > width - 1:
                return None
            r0 = random.randint(0, height - 1 - seg_len)
            c0 = random.randint(seg_len, width - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        diag_key = ("a", path[0][0] - path[0][1]) if dc == 1 else ("b", path[0][0] + path[0][1])
        if diag_key in used_diag_keys:
            continue
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _ordered_showcase_triggers(path: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    interior = path[1:-1]
    pos_cells = [cell for cell in interior if _side_of_main_diagonal(*cell) > 0]
    neg_cells = [cell for cell in interior if _side_of_main_diagonal(*cell) < 0]
    if not pos_cells or not neg_cells:
        return None

    valid_cells = [cell for cell in interior if _side_of_main_diagonal(*cell) != 0]
    max_triggers = min(_SHOWCASE_TRIGGER_RANGE[1], len(valid_cells))
    if max_triggers < _SHOWCASE_TRIGGER_RANGE[0]:
        return None

    n_triggers = random.randint(_SHOWCASE_TRIGGER_RANGE[0], max_triggers)
    chosen = [random.choice(pos_cells), random.choice(neg_cells)]
    remaining = [cell for cell in valid_cells if cell not in set(chosen)]
    if n_triggers > 2:
        chosen.extend(random.sample(remaining, n_triggers - 2))
    chosen = sorted(chosen, key=path.index)
    return chosen


def _ordered_regular_triggers(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    valid_cells = [cell for cell in path[1:-1] if _side_of_main_diagonal(*cell) != 0]
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(valid_cells))
    if max_triggers <= 0:
        return []
    n_triggers = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
    if n_triggers == 0:
        return []
    chosen = random.sample(valid_cells, n_triggers)
    return sorted(chosen, key=path.index)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_H_RANGE)
    width = random.randint(*_W_RANGE)
    forbidden_bg = {_LINE_COLOR, _TRIGGER_COLOR, *_RAY_COLOR_CYCLE}
    bg = random.choice([c for c in range(10) if c not in forbidden_bg])

    n_segments = random.randint(*_SEGMENT_COUNT)
    showcase_idx = random.randrange(n_segments)
    max_seg_len = min(height - 1, width - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    used_diag_keys: set[tuple[str, int]] = set()
    segments: list[tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]] = []

    for seg_idx in range(n_segments):
        slope_choices = ["slash"] if seg_idx == showcase_idx else ["backslash", "slash"]
        placed = False

        for _ in range(_SEGMENT_PLACE_RETRIES):
            slope_type = random.choice(slope_choices)
            seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
            result = _try_place_segment(slope_type, seg_len, height, width, occupied, drawn, used_diag_keys)
            if result is None:
                continue
            path, dc = result

            if seg_idx == showcase_idx:
                trigger_positions = _ordered_showcase_triggers(path)
            else:
                trigger_positions = _ordered_regular_triggers(path)
            if trigger_positions is None:
                continue

            ray_cells, trigger_dirs = _compute_single_ray_cells(trigger_positions, dc, height, width)
            if ray_cells is None or trigger_dirs is None:
                continue
            if any(rc in occupied or rc in drawn for rc in ray_cells):
                continue

            segments.append((path, trigger_positions, trigger_dirs))
            occupied.update([path[0], path[-1]])
            occupied.update(trigger_positions)
            drawn.update(set(path[1:-1]) - set(trigger_positions))
            drawn.update(ray_cells)
            diag_key = ("a", path[0][0] - path[0][1]) if dc == 1 else ("b", path[0][0] + path[0][1])
            used_diag_keys.add(diag_key)
            placed = True
            break

        if not placed:
            return None

    showcase_trigger_count = len(segments[showcase_idx][1])
    showcase_sides = {_side_of_main_diagonal(r, c) for r, c in segments[showcase_idx][1]}
    if showcase_trigger_count < 2 or showcase_sides != {-1, 1}:
        return None

    inp = np.full((height, width), bg, dtype=int)
    for path, trigger_positions, _ in segments:
        inp[path[0]] = _LINE_COLOR
        inp[path[-1]] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, height - 1), random.randint(0, width - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, height - 1), random.randint(0, width - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if _side_of_main_diagonal(r, c) == 0:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
