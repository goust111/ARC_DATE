# Combo variant of db695cfb (index 33)
# combo_spec: D8/0+D16/0
# Name: 选择性触发+四方向延伸

"""
Design analysis for the D8/0 + D16/0 combo variant
==================================================

source_task:
  reference_task: db695cfb baseline solver/spec plus the D8 single-variant reference
  preserved_core:
    - Blue endpoint pairs still define completed 45-degree diagonals.
    - Magenta pixels still sit on those diagonals and act as the only ray sources.
    - Rays still travel diagonally and never overwrite existing non-background cells.

fused_rule:
  selective_trigger:
    - Each real segment contains multiple on-segment magenta pixels.
    - Only the first magenta encountered from the upper endpoint toward the lower
      endpoint is active.
    - The lower magenta stays visible but inert.
  four_way_extension:
    - The chosen active magenta emits in all four diagonal directions.
    - This includes the two directions parallel to the completed blue segment and
      the two directions perpendicular to it, creating a full X-shaped spread.
  fixed_choice_for_inference:
    - The combo description says "first or last"; this implementation fixes
      "first from the upper endpoint" so the rule is stable enough to infer from
      a few ARC examples.

concretization_and_trace:
  example_segment:
    endpoints: [(2, 2), (7, 7)]
    on_segment_magenta: [(3, 3), (6, 6)]
  original_behavior:
    - Fill (2,2)..(7,7) in blue.
    - Both magenta pixels emit only along the perpendicular diagonal.
  D8_only_behavior:
    - Fill the same blue diagonal.
    - Only (3,3) emits, still only on the perpendicular diagonal.
  fused_behavior:
    - Fill the same blue diagonal.
    - Only (3,3) emits.
    - From (3,3), rays run to the border in all four diagonal directions.
    - Along the segment's own diagonal, the ray passes through the lower magenta
      and blue endpoints without overwriting them, then continues beyond them on
      background cells.
  visible_difference:
    - Compared with the baseline, one magenta no longer branches at all.
    - Compared with D8-only, the active magenta no longer makes a single slash;
      it produces a full X with additional same-diagonal tails beyond the blue
      endpoints.

complexity_calibration:
  kept_from_original:
    - Same color roles, same background rule, same endpoint-pair discovery rule,
      same distractor categories, same output-first generation style.
  simplified_for_observability:
    - Each real segment is generated with exactly two on-segment magenta pixels.
    - Segment count is capped at 3 and distractors are reduced because four-way
      rays occupy more space and clutter rises quickly.
    - Endpoints are kept one cell away from the border so the new same-diagonal
      ray branches are always visually present.
  why_this_is_not_too_simple:
    - Multiple on-segment magenta pixels still force the solver to infer the
      selection rule.
    - Four-way emission creates a strong geometric contrast from the original
      task instead of a barely noticeable tweak.

implicit_assumption_audit:
  baseline_main_dependencies_broken_by_combo:
    - The original solver assumed every trigger on a completed path is active.
      That is invalid now; path scanning must collect candidates and keep only
      the first.
    - The original solver stored only perpendicular directions per trigger.
      That is invalid now; the active trigger must always receive the same four
      diagonal directions regardless of segment slope.
    - The original generator reserved ray cells for every trigger. That would
      incorrectly forbid cells that should remain background once only one
      trigger is active.
  assumptions_kept_valid:
    - The true line color is still the color with the longest same-color
      diagonal pair, so generation must still guard against magenta becoming the
      selected color.
    - Segment endpoints are still the extreme blue points on one diagonal.
    - Sorting a segment's path by row is sufficient to define "first", because
      both allowed diagonal orientations move monotonically downward.
  new_main_assumptions_and_generator_guarantees:
    - Off-diagonal magenta distractors never lie on a real completed segment.
    - Real segments use unique diagonal keys within each orientation, so one
      group of blue points corresponds to one intended segment.
    - Every real segment has two on-segment magenta pixels, and the combo output
      differs from both the baseline and the D8-only behavior.

placement_legality:
  segment_placement:
    - Paths must stay in-bounds, avoid existing occupied cells, and avoid all
      previously reserved blue-fill or ray cells.
    - Endpoints are placed away from the outer border so same-diagonal ray
      branches can extend beyond them.
  ray_legality:
    - Only the active trigger's four-way rays are reserved.
    - Rays may pass through the current segment's own endpoints or inactive
      trigger, because the solver skips non-background cells and continues.
    - Rays may not collide with previously placed objects or reserved output
      cells from other segments.
  distractor_legality:
    - Unpaired blue distractors must not create another diagonal blue pair.
    - Off-diagonal magenta distractors must avoid all reserved path/ray cells.

visual_observability:
  human_visible_ordering:
    - "First trigger" is defined by visible top-to-bottom position, not color
      ids, scan order, or hidden metadata.
  support_in_examples:
    - Every generated real segment contains two magenta pixels on the same blue
      diagonal, so the inactive-vs-active distinction is repeatedly visible.
    - Four-way emission is supported by a one-cell border margin, guaranteeing
      non-empty same-diagonal branches beyond the endpoints.
  anti-degeneracy:
    - Generation rejects cases where the combo output matches D8-only, D16-only,
      or the original baseline behavior on the same input.

cleanliness_and_inference:
  clutter_control:
    - Segment count and distractor counts are reduced relative to the baseline.
    - Diagonal keys are unique, preventing stacked blue structures that would
      make it ambiguous which magenta belongs to which segment.
  inference_story:
    - Across examples, solvers can see that multiple magenta pixels appear on a
      completed blue diagonal, yet only the upper one produces branches.
    - They can also see that the chosen magenta branches both perpendicular to
      the blue segment and along the blue segment's continuation, which is the
      cue for the "all four diagonal directions" part of the combo.
"""

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


_ALL_DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _best_diagonal_pair(by_color: dict[int, list[tuple[int, int]]]) -> tuple[int, int] | None:
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
    return best


def _collect_line_groups(
    points: list[tuple[int, int]],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in points:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))
    return groups_a, groups_b


def _compute_ray_cells(
    trigger: tuple[int, int],
    dirs: list[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells = set()
    tr, tc = trigger
    for ddr, ddc in dirs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _render_output(
    inp: np.ndarray,
    bg: int,
    segments: list[dict],
    *,
    activate_all: bool,
    four_way: bool,
) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for segment in segments:
        for r, c in segment["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        trigger_positions = (
            segment["trigger_positions"] if activate_all else [segment["active_trigger"]]
        )
        dirs = _ALL_DIAG_DIRS if four_way else segment["perp_dirs"]
        for tr, tc in trigger_positions:
            for ddr, ddc in dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _line_color_is_selected(inp: np.ndarray, bg: int) -> bool:
    by_color: dict[int, list[tuple[int, int]]] = {}
    h, w = inp.shape
    for r in range(h):
        for c in range(w):
            v = int(inp[r, c])
            if v == bg:
                continue
            by_color.setdefault(v, []).append((r, c))
    best = _best_diagonal_pair(by_color)
    return best is not None and best[1] == _LINE_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each blue endpoint diagonal, then let only one on-line trigger fire in four diagonal directions.

    Input structure:
    - The background is the most common color. One non-background color forms endpoint pairs on shared 45-degree diagonals, and a different color may appear multiple times between those endpoints on the same diagonal.

    Rule:
    - Connect the extreme same-color endpoints on every occupied diagonal.
    - If completed diagonals contain trigger pixels, keep only the first trigger encountered from the upper endpoint toward the lower endpoint as active, and extend that trigger's color to the border in all four diagonal directions.

    Key insight:
    - The puzzle combines two filters at once: multiple magenta pixels can lie on one blue diagonal, but only the topmost one matters, and its effect is a full X-shaped diagonal burst rather than the original perpendicular-only pair of rays.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = _best_diagonal_pair(by_color)
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]
    groups_a, groups_b = _collect_line_groups(pts)
    active_triggers: dict[tuple[int, int, int], None] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        candidates: list[tuple[int, int, int]] = []
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                candidates.append((r, c, v))

        if candidates:
            active_triggers[candidates[0]] = None

        for r, c in path:
            if int(out[r, c]) == bg:
                out[r, c] = line_color

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for r0, c0, tcol in active_triggers:
        for ddr, ddc in _ALL_DIAG_DIRS:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (4, 14)
_TRIGGERS_PER_SEG = 2

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 500


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_diag_keys: set[tuple[str, int]],
) -> tuple[list[tuple[int, int]], int, tuple[str, int]] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - seg_len - 3 < 0 or w - seg_len - 3 < 0:
                return None
            r0 = random.randint(1, h - seg_len - 2)
            c0 = random.randint(1, w - seg_len - 2)
            dc = 1
            diag_key = ("backslash", r0 - c0)
        else:
            if h - seg_len - 3 < 0 or (seg_len + 1) > (w - 2):
                return None
            r0 = random.randint(1, h - seg_len - 2)
            c0 = random.randint(seg_len + 1, w - 2)
            dc = -1
            diag_key = ("slash", r0 + c0)

        if diag_key in used_diag_keys:
            continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(cell in occupied or cell in drawn for cell in path):
            continue
        return path, dc, diag_key
    return None


def _choose_trigger_positions(path_interior: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    n = len(path_interior)
    if n < _TRIGGERS_PER_SEG:
        return None

    upper_limit = max(1, n // 2)
    first_choices = list(range(upper_limit))
    random.shuffle(first_choices)
    for first_idx in first_choices:
        min_second_idx = max(first_idx + 2, n // 2)
        if min_second_idx >= n:
            min_second_idx = first_idx + 1
        if min_second_idx >= n:
            continue
        second_choices = list(range(min_second_idx, n))
        random.shuffle(second_choices)
        second_idx = second_choices[0]
        return [path_interior[first_idx], path_interior[second_idx]]
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(h - 3, w - 3, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    used_diag_keys: set[tuple[str, int]] = set()
    segments: list[dict] = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)

        placed = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn, used_diag_keys)
        if placed is None:
            continue
        path, dc, diag_key = placed
        path_interior = path[1:-1]
        trigger_positions = _choose_trigger_positions(path_interior)
        if trigger_positions is None:
            continue

        active_trigger = trigger_positions[0]
        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(active_trigger, _ALL_DIAG_DIRS, h, w)

        if any(cell in occupied or cell in drawn for cell in ray_cells):
            continue

        used_diag_keys.add(diag_key)
        segments.append(
            {
                "endpoints": (path[0], path[-1]),
                "trigger_positions": trigger_positions,
                "active_trigger": active_trigger,
                "path": path,
                "perp_dirs": perp_dirs,
            }
        )
        occupied.update([path[0], path[-1]])
        occupied.update(trigger_positions)
        drawn.update(path)
        drawn.update(ray_cells)

    if not segments:
        return None

    inp = np.full((h, w), bg, dtype=int)
    for segment in segments:
        ep0, ep1 = segment["endpoints"]
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in segment["trigger_positions"]:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
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
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    if not _line_color_is_selected(inp, bg):
        return None

    combo_expected = _render_output(inp, bg, segments, activate_all=False, four_way=True)
    d8_only_expected = _render_output(inp, bg, segments, activate_all=False, four_way=False)
    d16_only_expected = _render_output(inp, bg, segments, activate_all=True, four_way=True)
    baseline_expected = _render_output(inp, bg, segments, activate_all=True, four_way=False)

    if np.array_equal(combo_expected, d8_only_expected):
        return None
    if np.array_equal(combo_expected, d16_only_expected):
        return None
    if np.array_equal(combo_expected, baseline_expected):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, combo_expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
