# Variant of db695cfb
# Dimension: D8 — 触发像素数量与排列 (obstacle)
# Change: 对角线上可能有多个品红像素，每个独立触发延伸... → 仅第一个/最后一个品红像素触发

"""
Design analysis for the D8 variant
==================================

source_task:
  reference_task: none
  baseline_reading: original solver/spec only

variant_rule:
  core_change:
    - Each blue endpoint pair still defines one completed 45-degree diagonal.
    - There may be multiple magenta pixels on that diagonal.
    - Only the first magenta encountered when traversing from the upper endpoint
      to the lower endpoint is active; lower magenta pixels on the same diagonal
      remain visible but do not emit rays.
  visible cue:
    - The ordering rule is geometric and human-visible: "nearest to the top
      endpoint", not based on color id, scan order outside the segment, or any
      hidden metadata.

concretization_and_trace:
  example_segment:
    endpoints: [(1, 1), (6, 6)]
    on_segment_magenta: [(2, 2), (5, 5)]
  original_behavior:
    - Fill the whole main diagonal from (1,1) to (6,6).
    - Both (2,2) and (5,5) emit rays on the perpendicular diagonal.
  variant_behavior:
    - Fill the same main diagonal.
    - Only (2,2), the first trigger from the upper endpoint, emits rays.
    - (5,5) stays magenta in place but produces no extension.
  difference:
    - Input construction must include multiple on-segment magenta pixels.
    - Solver logic must filter them before ray emission.
    - Output generation must use exactly one active trigger per segment.

complexity_calibration:
  kept_from_original:
    - Same canvas scale, same endpoint/ray geometry, same colors, same
      background logic, same distractor categories.
  adjusted_for observability:
    - Real segments are generated with exactly two on-segment magenta pixels so
      every training example exposes the new selection rule.
    - Segment count is capped at 3 and off-diagonal distractors are slightly
      reduced to keep the output readable once inactive magenta pixels are left
      behind on completed diagonals.
  why not more branches:
    - Mixing "sometimes first, sometimes last" would be hard to infer from a few
      ARC examples. The variant therefore fixes a single positional rule.

implicit_assumption_audit:
  original_main_dependencies:
    - Choosing the line color by the longest same-color diagonal pair remains
      valid, but generation must still avoid cases where magenta accidentally
      becomes the best pair.
    - The original draw_segment() activated every non-line pixel found on the
      path. That assumption is invalid now; path scanning must separate
      candidate triggers from the final active trigger.
    - The original generator computed ray occupancy from all on-segment
      triggers. That must change to active-trigger-only occupancy, otherwise the
      generator would reserve cells that should remain background in the variant.
  new_main_assumptions:
    - Endpoints of a true segment are the extreme blue points on one diagonal.
    - Ordering along a segment is top-to-bottom, so sorting endpoints by row is
      enough to define "first trigger".
    - Off-diagonal magenta distractors never sit on a completed blue segment.
  generator_guarantees:
    - Each real segment uses a unique diagonal key within its orientation, so
      solver grouping by extreme endpoints remains correct.
    - Every real segment has two on-segment magenta pixels, and expected output
      differs from the original "all triggers fire" rule.
    - Rays from the chosen trigger never overwrite any input pixel.

placement_legality_and_cleanliness:
  legality:
    - No segment path overlaps previous input pixels or previously drawn output
      cells.
    - Only the active trigger's rays are reserved as output-only cells.
    - Distractors avoid all reserved cells.
  cleanliness:
    - Two on-segment magenta pixels are separated by position along the segment,
      making it easy to compare the active upper trigger with the inactive lower
      one.
    - Low distractor counts keep the completed diagonal and single emitted ray
      family legible.
"""

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


def _collect_line_groups(points: list[tuple[int, int]]) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in points:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))
    return groups_a, groups_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal endpoint pairs, but let only one on-line trigger emit rays.

    Input structure:
    - The dominant color is background. One non-background color forms endpoint pairs on 45-degree diagonals, and one or more differently-colored pixels may lie between those endpoints on the same diagonal.

    Rule:
    - Connect the extreme same-color endpoints on each occupied diagonal.
    - If the finished diagonal contains trigger pixels, only the first trigger encountered from the upper endpoint toward the lower endpoint sends perpendicular diagonal rays to the border; later triggers on that same diagonal stay inert.

    Key insight:
    - Multiple magenta pixels can sit on one completed diagonal, but position filters them: the active splitter is the one nearest the top endpoint, not every trigger on the line.
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
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a, groups_b = _collect_line_groups(pts)

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

        if candidates:
            key = candidates[0]
            triggers.setdefault(key, set()).update(perp_dirs)

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
_GEN_RETRIES = 400


def _compute_ray_cells(trigger: tuple[int, int], perp_dirs: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    tr, tc = trigger
    for ddr, ddc in perp_dirs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict], *, activate_all: bool) -> np.ndarray:
    h, w = inp.shape
    expected = inp.copy()
    for segment in segments:
        for r, c in segment["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        triggers = segment["trigger_positions"] if activate_all else [segment["active_trigger"]]
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
    if not late_pool:
        late_pool = path_interior[1:]
    if not early_pool or not late_pool:
        return None

    for _ in range(_DISTRACTOR_RETRIES):
        first = random.choice(early_pool)
        later_candidates = [pt for pt in late_pool if path_interior.index(pt) > path_interior.index(first)]
        if not later_candidates:
            continue
        second = random.choice(later_candidates)
        return [first, second]
    return None


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
        if any(cell in occupied for cell in ray_cells):
            continue

        if slope_type == "backslash":
            used_diag_keys.add(("backslash", path[0][0] - path[0][1]))
        else:
            used_diag_keys.add(("slash", path[0][0] + path[0][1]))

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

    expected = _build_expected(inp, bg, segments, activate_all=False)
    original_like = _build_expected(inp, bg, segments, activate_all=True)
    if np.array_equal(expected, original_like):
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
