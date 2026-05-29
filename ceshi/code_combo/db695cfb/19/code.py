# Combo variant of db695cfb (index 19)
# combo_spec: D8/0+D13/1
# Name: 选择性触发+品红变蓝

"""
Design analysis for the combo variant
=====================================

source_task:
  reference_task: db695cfb
  combo_spec: D8/0 + D13/1
  preserved_core:
    - Blue endpoint pairs still define completed 45-degree diagonals.
    - Perpendicular diagonal rays still come from magenta pixels that lie on
      those completed diagonals.

fused_rule:
  selective_trigger:
    - Real segments deliberately contain two magenta pixels on their interior.
    - Only the first magenta encountered when walking from the upper endpoint
      to the lower endpoint is active.
    - The later magenta remains visible but inert.
  recolor_effect:
    - The active magenta is absorbed by the blue diagonal in the output, so its
      own cell becomes blue.
    - Its emitted perpendicular rays are still magenta.

concretization_and_trace:
  example_segment:
    endpoints: [(1, 1), (6, 6)]
    interior_magenta: [(2, 2), (5, 5)]
  original_behavior:
    - The full diagonal from (1,1) to (6,6) is filled blue.
    - Both magenta cells emit magenta rays.
    - Both magenta cells remain magenta on the diagonal.
  fused_behavior:
    - The same diagonal is filled blue.
    - Only (2,2), the magenta nearest the upper endpoint, emits rays.
    - Cell (2,2) itself turns blue, so it visually merges into the main line.
    - Cell (5,5) stays magenta and emits nothing.
  observable_difference:
    - Compared with the original, one ray family disappears.
    - Compared with D8-only selective triggering, the active trigger no longer
      remains magenta; it is recolored blue.

complexity_calibration:
  kept:
    - Same canvas scale, same colors, same diagonal geometry, same output-first
      generation strategy, same distractor categories.
  adjusted_for_visibility:
    - Every real segment has exactly two on-segment magenta pixels so each
      training sample exposes both "which trigger is chosen" and "chosen one
      turns blue".
    - Segment count is capped at 3 and distractors are light, which keeps the
      absorbed trigger and the inert trigger easy to compare by eye.
  avoided:
    - The rule is not allowed to alternate between "first" and "last" trigger,
      because that would be too hard to infer from 2-4 ARC examples.

implicit_assumption_audit:
  original_main_dependencies_broken_by_combo:
    - The original path scan activated every non-blue cell on the segment.
      That is no longer valid; the scan must gather all candidate triggers,
      choose the first one only, and remember the rest as inert.
    - The original fill step only painted background cells blue.
      That is no longer sufficient; the selected trigger cell must also be
      overwritten to blue, while later triggers must not be overwritten.
    - The original generator reserved ray cells for every trigger on the
      segment. In the combo, only the selected trigger's rays should be
      reserved; otherwise the generator would prohibit valid background cells.
  new_main_assumptions_and_generator_guarantees:
    - Sorting segment endpoints by row defines a visible top-to-bottom order.
      The generator keeps each real segment on its own diagonal key so this
      ordering is unambiguous.
    - Off-diagonal magenta distractors never lie on any real completed segment,
      so the solver's on-path trigger scan only sees intended candidates.
    - Blue must remain the color chosen by the longest diagonal-aligned pair.
      The generator checks this explicitly to avoid magenta accidental wins.
    - Rays from the active trigger must not overwrite any input pixel; segment
      placement rejects such cases.

placement_legality_and_cleanliness:
  legality:
    - Segment paths cannot overlap prior input pixels or reserved output cells.
    - Only the active trigger's ray cells are reserved.
    - Distractors avoid every occupied or reserved cell.
  cleanliness:
    - The active trigger is chosen from the earlier half of the segment and the
      inert trigger from the later half, making the ordering cue visually clear.
    - Ray overlaps are rejected to prevent muddled outputs.

post_change_audit_of_main:
  grouping_loops:
    - Each diagonal group is assumed to correspond to one true segment defined
      by extreme blue endpoints. The generator enforces unique diagonal keys.
  trigger_selection_loop:
    - Candidate triggers are read from the input grid, not the partially drawn
      output, so recoloring one segment cannot affect later selection logic.
  ray_emission_loop:
    - Rays only paint background cells, which matches the generated promise
      that no emitted ray needs to overwrite an existing pixel.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _best_diagonal_pair(by_color: dict[int, list[tuple[int, int]]]) -> tuple[int, int] | None:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each blue diagonal, but only absorb and activate the earliest on-line magenta.

    Input structure:
    - The dominant color is background. Blue pixels appear as endpoint pairs on 45-degree diagonals, and magenta pixels may sit between those endpoints on the same diagonal as candidate triggers.

    Rule:
    - Connect the extreme blue endpoints on each occupied diagonal.
    - If a completed diagonal contains magenta candidates, only the first one encountered from the upper endpoint toward the lower endpoint is active: it turns blue and emits magenta rays along the perpendicular diagonal to the border. Later magenta pixels stay magenta and emit nothing.

    Key insight:
    - Position along the diagonal decides both behaviors at once: the uppermost on-line magenta is the only trigger, and that same chosen pixel is absorbed into the blue line while the later magenta remains an inert marker.
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

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]

        candidates: list[tuple[int, int, int]] = []
        for r, c in path:
            v = int(grid[r, c])
            if v != bg and v != line_color:
                candidates.append((r, c, v))

        active = candidates[0] if candidates else None
        active_pos = None if active is None else (active[0], active[1])

        for r, c in path:
            if int(out[r, c]) == bg or (active_pos is not None and (r, c) == active_pos):
                out[r, c] = line_color

        if active is not None:
            triggers.setdefault(active, set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
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
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = 2

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 500


def _compute_ray_cells(
    trigger: tuple[int, int],
    perp_dirs: list[tuple[int, int]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    cells = set()
    tr, tc = trigger
    for ddr, ddc in perp_dirs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[dict],
    *,
    activate_all: bool,
    recolor_active: bool,
) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for segment in segments:
        active_trigger = segment["active_trigger"]
        active_pos = (active_trigger[0], active_trigger[1])
        for r, c in segment["path"]:
            if expected[r, c] == bg or (recolor_active and (r, c) == active_pos):
                expected[r, c] = _LINE_COLOR

        triggers = segment["trigger_positions"] if activate_all else [active_trigger]
        for tr, tc in triggers:
            for ddr, ddc in segment["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < h and 0 <= c < w:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    used_diag_keys: set[tuple[str, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if h - 1 - seg_len < 0 or w - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(0, w - 1 - seg_len)
            dc = 1
            diag_key = ("backslash", r0 - c0)
        else:
            if h - 1 - seg_len < 0 or seg_len > w - 1:
                return None
            r0 = random.randint(0, h - 1 - seg_len)
            c0 = random.randint(seg_len, w - 1)
            dc = -1
            diag_key = ("slash", r0 + c0)

        if diag_key in used_diag_keys:
            continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(cell in occupied or cell in drawn for cell in path):
            continue
        return path, dc
    return None


def _choose_trigger_positions(path_interior: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(path_interior) < _TRIGGERS_PER_SEG:
        return None

    split = max(1, len(path_interior) // 2)
    early_pool = path_interior[:split]
    late_pool = path_interior[split:]
    if not early_pool or not late_pool:
        return None

    first = random.choice(early_pool)
    first_idx = path_interior.index(first)
    later_candidates = [pt for pt in late_pool if path_interior.index(pt) > first_idx]
    if not later_candidates:
        return None
    second = random.choice(later_candidates)
    return [first, second]


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


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_H_RANGE)
    w = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(h - 1, w - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    used_diag_keys: set[tuple[str, int]] = set()
    segments: list[dict] = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, h, w, occupied, drawn, used_diag_keys)
        if result is None:
            continue
        path, dc = result
        path_interior = path[1:-1]
        trigger_positions = _choose_trigger_positions(path_interior)
        if trigger_positions is None:
            continue

        perp_dirs = [(1, -dc), (-1, dc)]
        active_trigger = trigger_positions[0]
        ray_cells = _compute_ray_cells(active_trigger, perp_dirs, h, w)
        if any(cell in occupied or cell in drawn for cell in ray_cells):
            continue

        diag_key = ("backslash", path[0][0] - path[0][1]) if slope_type == "backslash" else ("slash", path[0][0] + path[0][1])
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
        drawn.update(set(path_interior) - set(trigger_positions))
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

    fused_expected = _build_expected(inp, bg, segments, activate_all=False, recolor_active=True)
    selective_no_recolor = _build_expected(inp, bg, segments, activate_all=False, recolor_active=False)
    original_like = _build_expected(inp, bg, segments, activate_all=True, recolor_active=False)
    if np.array_equal(fused_expected, selective_no_recolor):
        return None
    if np.array_equal(fused_expected, original_like):
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    if np.array_equal(result, fused_expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
