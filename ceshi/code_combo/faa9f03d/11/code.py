# Combo variant of faa9f03d (index 11)
# combo_spec: D6/1+D9/0
# Name: 同色同行/列全部填充 + 纯直线段无转角

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import components_mask, get_bg


H = W = 12
BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


# Design analysis for D6/1 + D9/0
# - Concrete behavior: if a blue horizontal line is broken into
#   `..11.2..2.11..`, the original task could refuse the marker-marker gap
#   because both bridge endpoints are markers and the gap is longer than 1.
#   This combo removes that restriction, so the whole interval between the
#   leftmost and rightmost same-color sightings is filled. D9/0 also removes
#   turns, so each color behaves as a 1-D row segment or column segment.
# - Complexity calibration: keeping 4 colors as 2 full/partial rows and
#   2 full/partial columns preserves crossings and repeated evidence while
#   avoiding the visual clutter of multi-turn polylines. At least one segment
#   always carries two separated red markers so the aggressive D6 fill is
#   visible rather than latent.
# - Implicit assumptions removed from the original solver:
#   1. No axis/perpendicular support checks are needed before filling.
#   2. Pure-filled crossings no longer need marker-bracketing heuristics.
#   3. Output components are asserted to stay straight, not rectilinear.
#   Yellow controls still flip one arm to the opposite side, but because all
#   components are straight, the erased/painted parts are single rays.
# - Placement legality: generator keeps rows/cols spaced apart, never puts
#   red markers on crossings, and chooses yellow pivots far enough from the
#   opposite crossings so the marker sees only its intended color.
# - Observability / cleanliness: every sample has repeated row/column lines,
#   visible crossings, and at least one long marker-marker gap that becomes a
#   continuous straight segment in the output. That makes the combo legible
#   from a few examples without relying on hidden color-id conventions.


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _segment_cells(orient: str, fixed: int, lo: int, hi: int) -> set[tuple[int, int]]:
    if orient == "h":
        return {(fixed, c) for c in range(lo, hi + 1)}
    return {(r, fixed) for r in range(lo, hi + 1)}


def _is_straight_component(comp: np.ndarray) -> bool:
    rows = {int(r) for r, _ in comp}
    cols = {int(c) for _, c in comp}
    return len(rows) == 1 or len(cols) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair singleton-marked straight line fragments by recoloring the markers,
    filling every same-color row/column gap, and flipping yellow-controlled
    arms into edge-facing rays.

    Input structure:
    - A 12x12 grid with a dominant background.
    - Non-marker colors appear only as one-cell-thick horizontal or vertical
      straight segments; there are no turns.
    - Marker colors are singleton components: red marks missing cells on a
      straight segment, and yellow sits next to the endpoint that should be
      extended toward the marker side.

    Rule:
    - Assign each marker to the nearest visible line color seen in the same
      row or column, preferring an endpoint already supported along that axis.
    - After recoloring the markers, aggressively fill every gap between
      consecutive same-color sightings in each row and each column.
    - For a yellow marker, find its adjacent pivot, erase the old arm on the
      far side of that pivot, then paint the opposite ray from the pivot
      through the marker direction until the grid edge.
    - At overlaps, yellow-control rays override everything; otherwise a cell
      that already showed one candidate color uses the more fragmented
      pre-fill color, and a purely filled overlap uses the lower color id.

    Key insight:
    - Once turns are forbidden, each color is reconstructed independently as
      1-D row/column continuity. The essential signal is simply which same-
      color observations belong to the same straight segment, with yellow
      markers only deciding which side of a pivot survives.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, "foreground color must have at least one component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < h and 0 <= c < w:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break
        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        candidates.sort()
        _, _, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(h):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                additions.update((r, c) for c in range(c1 + 1, c2))
        for c in range(w):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                additions.update((r, c) for r in range(r1 + 1, r2))
        masks[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in _pts(g == 4):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned pivot"

            old_start = (pivot[0] + dr, pivot[1] + dc)
            if old_start in masks[color]:
                stack = [old_start]
                old_arm: set[tuple[int, int]] = set()
                while stack:
                    q = stack.pop()
                    if q == pivot or q in old_arm or q not in masks[color]:
                        continue
                    old_arm.add(q)
                    qr, qc = q
                    for ndr, ndc in DIRS4:
                        stack.append((qr + ndr, qc + ndc))
                for q in old_arm:
                    masks[color].discard(q)
                    marker_pts[color].discard(q)
                    control_ray[color].discard(q)

            r, c = pivot
            pr, pc = -dr, -dc
            while 0 <= r < h and 0 <= c < w:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = [color for color in colors_here if pt in control_ray[color]]
        if forced:
            out[pt] = forced[0]
        elif len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            original_crossing = [color for color in colors_here if int(g[pt]) == color]
            if original_crossing:
                out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            assert _is_straight_component(comp), f"color {color} formed a turned component"

    return out


def _spaced_pair() -> tuple[int, int]:
    vals = list(range(2, 10))
    for _ in range(100):
        a, b = sorted(random.sample(vals, 2))
        if b - a >= 4:
            return a, b
    raise RuntimeError("failed to choose spaced coordinates")


def _choose_crossing_display(
    visible_claims: dict[tuple[int, int], list[int]],
    preferred_loser: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    display: dict[tuple[int, int], int] = {}
    changed = False
    for pt, colors_here in visible_claims.items():
        if len(colors_here) == 1:
            display[pt] = colors_here[0]
            continue
        if pt in preferred_loser and preferred_loser[pt] in colors_here:
            display[pt] = preferred_loser[pt]
            changed = True
        else:
            display[pt] = random.choice(colors_here)
    if not changed:
        multi = [pt for pt, colors_here in visible_claims.items() if len(colors_here) > 1]
        if multi:
            pt = random.choice(multi)
            colors_here = visible_claims[pt]
            display[pt] = max(colors_here)
    return display


def _pick_gap_markers(
    segment: dict,
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    orient = segment["orient"]
    fixed = segment["fixed"]
    crosses = segment["cross_scalars"]
    scalar_values = sorted(segment["visible_scalars"])
    if not scalar_values:
        return None

    blocked_scalars = set(crosses)
    runs: list[list[int]] = []
    run: list[int] = []
    for v in scalar_values:
        if v in blocked_scalars:
            if run:
                runs.append(run)
            run = []
            continue
        if not run or v == run[-1] + 1:
            run.append(v)
        else:
            runs.append(run)
            run = [v]
    if run:
        runs.append(run)

    random.shuffle(runs)
    for run in runs:
        if len(run) < 5:
            continue
        for i in range(len(run)):
            for j in range(i + 3, len(run)):
                a, b = run[i], run[j]
                if any(abs(a - cross) < 2 or abs(b - cross) < 2 for cross in crosses):
                    continue
                if orient == "h":
                    pts = [(fixed, a), (fixed, b)]
                    gap = {(fixed, c) for c in range(a + 1, b)}
                else:
                    pts = [(a, fixed), (b, fixed)]
                    gap = {(r, fixed) for r in range(a + 1, b)}
                if any(pt in forbidden for pt in pts):
                    continue
                if any(nb in forbidden for pt in pts for nb in [pt] + _neighbors(pt)):
                    continue
                return pts, gap
    return None


def _extra_red_candidates(
    segments: list[dict],
    used_markers: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for segment in segments:
        orient = segment["orient"]
        fixed = segment["fixed"]
        for scalar in sorted(segment["visible_scalars"]):
            if scalar in segment["cross_scalars"]:
                continue
            pt = (fixed, scalar) if orient == "h" else (scalar, fixed)
            if pt in used_markers:
                continue
            if any(nb in used_markers for nb in [pt] + _neighbors(pt)):
                continue
            out.append(pt)
    random.shuffle(out)
    return out


def _render_input(
    segments: list[dict],
    red_markers: dict[tuple[int, int], int],
    yellow_marker: tuple[int, int] | None,
    yellow_color: int | None,
    crossing_display: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    claims: dict[tuple[int, int], list[int]] = {}
    for segment in segments:
        for pt in segment["input_cells"]:
            claims.setdefault(pt, []).append(segment["color"])

    for pt, colors_here in claims.items():
        if pt in red_markers or pt == yellow_marker:
            continue
        if len(colors_here) == 1:
            grid[pt] = colors_here[0]
        else:
            grid[pt] = crossing_display[pt]

    for pt in red_markers:
        grid[pt] = 2
    if yellow_marker is not None and yellow_color is not None:
        grid[yellow_marker] = 4
    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    rows = _spaced_pair()
    cols = _spaced_pair()
    colors = random.sample(LINE_PALETTE, 4)

    segments = [
        {"color": colors[0], "orient": "h", "fixed": rows[0]},
        {"color": colors[1], "orient": "h", "fixed": rows[1]},
        {"color": colors[2], "orient": "v", "fixed": cols[0]},
        {"color": colors[3], "orient": "v", "fixed": cols[1]},
    ]

    for segment in segments:
        if segment["orient"] == "h":
            segment["target_lo"], segment["target_hi"] = 0, W - 1
            segment["target_cells"] = _segment_cells("h", segment["fixed"], 0, W - 1)
            segment["input_cells"] = set(segment["target_cells"])
            segment["visible_scalars"] = set(range(W))
            segment["cross_scalars"] = set(cols)
        else:
            segment["target_lo"], segment["target_hi"] = 0, H - 1
            segment["target_cells"] = _segment_cells("v", segment["fixed"], 0, H - 1)
            segment["input_cells"] = set(segment["target_cells"])
            segment["visible_scalars"] = set(range(H))
            segment["cross_scalars"] = set(rows)
        segment["yellow_target_cells"] = set()
        segment["yellow_old_arm_cells"] = set()

    yellow_segment: dict | None = None
    yellow_marker: tuple[int, int] | None = None
    yellow_side: str | None = None
    if random.random() < 0.6:
        yellow_segment = random.choice(segments)
        if yellow_segment["orient"] == "h":
            left_ok = max(cols) + 1 <= W - 3
            right_ok = 2 <= min(cols) - 1
            sides = []
            if left_ok:
                sides.append("left")
            if right_ok:
                sides.append("right")
            if not sides:
                return None
            yellow_side = random.choice(sides)
            r = yellow_segment["fixed"]
            if yellow_side == "left":
                pivot = random.randint(max(cols) + 1, W - 3)
                yellow_marker = (r, pivot - 1)
                yellow_segment["target_cells"] = _segment_cells("h", r, 0, pivot)
                yellow_segment["input_cells"] = _segment_cells("h", r, pivot, W - 1)
                yellow_segment["visible_scalars"] = set(range(pivot, W))
                yellow_segment["yellow_target_cells"] = _segment_cells("h", r, 0, pivot)
                yellow_segment["yellow_old_arm_cells"] = _segment_cells("h", r, pivot + 1, W - 1)
            else:
                pivot = random.randint(2, min(cols) - 1)
                yellow_marker = (r, pivot + 1)
                yellow_segment["target_cells"] = _segment_cells("h", r, pivot, W - 1)
                yellow_segment["input_cells"] = _segment_cells("h", r, 0, pivot)
                yellow_segment["visible_scalars"] = set(range(0, pivot + 1))
                yellow_segment["yellow_target_cells"] = _segment_cells("h", r, pivot, W - 1)
                yellow_segment["yellow_old_arm_cells"] = _segment_cells("h", r, 0, pivot - 1)
        else:
            top_ok = max(rows) + 1 <= H - 3
            bottom_ok = 2 <= min(rows) - 1
            sides = []
            if top_ok:
                sides.append("top")
            if bottom_ok:
                sides.append("bottom")
            if not sides:
                return None
            yellow_side = random.choice(sides)
            c = yellow_segment["fixed"]
            if yellow_side == "top":
                pivot = random.randint(max(rows) + 1, H - 3)
                yellow_marker = (pivot - 1, c)
                yellow_segment["target_cells"] = _segment_cells("v", c, 0, pivot)
                yellow_segment["input_cells"] = _segment_cells("v", c, pivot, H - 1)
                yellow_segment["visible_scalars"] = set(range(pivot, H))
                yellow_segment["yellow_target_cells"] = _segment_cells("v", c, 0, pivot)
                yellow_segment["yellow_old_arm_cells"] = _segment_cells("v", c, pivot + 1, H - 1)
            else:
                pivot = random.randint(2, min(rows) - 1)
                yellow_marker = (pivot + 1, c)
                yellow_segment["target_cells"] = _segment_cells("v", c, pivot, H - 1)
                yellow_segment["input_cells"] = _segment_cells("v", c, 0, pivot)
                yellow_segment["visible_scalars"] = set(range(0, pivot + 1))
                yellow_segment["yellow_target_cells"] = _segment_cells("v", c, pivot, H - 1)
                yellow_segment["yellow_old_arm_cells"] = _segment_cells("v", c, 0, pivot - 1)

    forbidden_markers: set[tuple[int, int]] = set()
    if yellow_marker is not None:
        forbidden_markers.add(yellow_marker)
        forbidden_markers.update(_neighbors(yellow_marker))

    gap_host_candidates = [
        segment for segment in segments
        if segment["orient"] == "h"
        and segment is not yellow_segment
        and len(segment["visible_scalars"]) >= 5
    ]
    random.shuffle(gap_host_candidates)
    forced_gap_color = None
    forced_gap_cells: set[tuple[int, int]] = set()
    red_markers: dict[tuple[int, int], int] = {}
    for segment in gap_host_candidates:
        picked = _pick_gap_markers(segment, forbidden_markers)
        if picked is None:
            continue
        pts, gap_cells = picked
        for pt in pts:
            red_markers[pt] = segment["color"]
            forbidden_markers.add(pt)
            forbidden_markers.update(_neighbors(pt))
        segment["input_cells"] -= gap_cells
        if segment["orient"] == "h":
            segment["visible_scalars"] -= {c for _, c in gap_cells}
        else:
            segment["visible_scalars"] -= {r for r, _ in gap_cells}
        forced_gap_color = segment["color"]
        forced_gap_cells = gap_cells
        break
    if forced_gap_color is None or not forced_gap_cells:
        return None

    extra_budget = random.randint(0, 2)
    for pt in _extra_red_candidates(segments, set(red_markers) | forbidden_markers):
        if extra_budget == 0:
            break
        owner = next(segment["color"] for segment in segments if pt in segment["input_cells"])
        red_markers[pt] = owner
        forbidden_markers.add(pt)
        forbidden_markers.update(_neighbors(pt))
        extra_budget -= 1

    visible_claims: dict[tuple[int, int], list[int]] = {}
    for segment in segments:
        for pt in segment["input_cells"]:
            if pt in red_markers:
                continue
            visible_claims.setdefault(pt, []).append(segment["color"])

    preferred_loser = {pt: max(colors_here) for pt, colors_here in visible_claims.items() if len(colors_here) > 1}
    crossing_display = _choose_crossing_display(visible_claims, preferred_loser)
    inp = _render_input(
        segments,
        red_markers,
        yellow_marker,
        None if yellow_segment is None else yellow_segment["color"],
        crossing_display,
    )

    for color in (2, 4):
        for pt in _pts(inp == color):
            if any(int(inp[nb]) == color for nb in _neighbors(pt)):
                return None

    try:
        out = main(inp)
    except Exception:
        return None

    changed = int(np.count_nonzero(out != inp))
    if changed < 6:
        return None

    line_colors = sorted(set(map(int, out.ravel())) - {BG})
    if 2 in line_colors or 4 in line_colors:
        return None
    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            if not _is_straight_component(comp):
                return None

    if any(int(out[pt]) != forced_gap_color for pt in forced_gap_cells):
        return None

    yellow_info = None
    if yellow_segment is not None:
        target_check = yellow_segment["yellow_target_cells"] - {
            (rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])
        }
        old_check = yellow_segment["yellow_old_arm_cells"] - {
            (rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])
        }
        if not target_check:
            return None
        if any(int(out[pt]) != yellow_segment["color"] for pt in target_check):
            return None
        if any(int(out[pt]) == yellow_segment["color"] for pt in old_check):
            return None
        yellow_info = {
            "color": yellow_segment["color"],
            "side": yellow_side,
            "target_cells": sorted(target_check),
            "old_cells": sorted(old_check),
        }

    meta = {
        "forced_gap_color": forced_gap_color,
        "forced_gap_cells": sorted(forced_gap_cells),
        "yellow": yellow_info,
        "changed": changed,
    }
    return inp, out, meta


def _gen_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(GEN_RETRIES):
        try:
            built = _build_case()
        except Exception:
            built = None
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid D6/1+D9/0 faa9f03d input")


def generate_input() -> np.ndarray:
    return _gen_case()[0]
