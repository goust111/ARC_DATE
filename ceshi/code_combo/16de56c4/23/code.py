# Combo variant of 16de56c4 (index 23)
# combo_spec: D5/3+D7/0+D17/1
# Name: 对称填充+标记擦除+旁观者擦除

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# 1) Concrete behavior / trace:
#    - The original task depended on a unique shooting edge and kept both marker dots
#      as part of the painted ray. This combo removes that edge notion entirely.
#    - A fused active line now looks like: two same-color interior markers, plus
#      optionally one extra dot. The repeated pair defines the step. Expansion is
#      bilateral: left from the first marker and right from the second marker.
#    - Example row of length 11:
#        input  : [0,0,0,4,0,0,4,0,2,0,0]
#        markers: positions 3 and 6, so step = 3
#        extra  : position 8 is off-beat, therefore a bystander
#        D5-only output would paint beat cells 0/3/6/9.
#        This combo then applies D7 and D17: erase marker positions 3 and 6, erase
#        bystander position 8, keep only the symmetric trajectory cells 0 and 9.
#        final  : [4,0,0,0,0,0,0,0,0,4,0]
#    - If the extra dot is aligned instead, it is a companion obstacle on one side:
#      it caps that side's walk, supplies the paint color, and survives only because
#      it lies on the final trajectory itself.
#
# 2) Complexity calibration:
#    - Keep exactly one repeated color per active line and at most one extra dot.
#      That preserves a stable "decode line by line" grammar.
#    - Keep 4-9 active lines with empty gaps between them, so humans can compare
#      several repeated instances without overlap clutter.
#    - Force the marker pair into the interior with at least one outward beat on both
#      sides. Otherwise the D5 change would visually collapse back into one-sided fill
#      or yield too little output after marker erasure.
#    - Require both companion lines and bystander lines across generated samples.
#      This makes all three fused ingredients visible: bilateral growth, marker erase,
#      and bystander erase.
#
# 3) Hidden assumptions broken by the fusion:
#    - Original edge counting is invalid because no marker is anchored to a border.
#      The solver must infer the active axis structurally instead.
#    - Original `out = g.copy()` is now wrong in spirit: it would preserve markers and
#      bystanders unless each one were explicitly removed. Starting from pure bg is
#      cleaner and directly matches the "only trajectories survive" spec.
#    - Original fill loops were allowed to repaint marker cells. That assumption fails
#      because markers are instructions only; they must never remain in the output.
#    - Bystander handling can no longer be an afterthought. Once the output is rebuilt
#      from scratch, every surviving non-bg cell must be justified as part of a beat
#      trajectory; off-beat extras must disappear.
#    - Companion detection still assumes at most one aligned foreign dot and that it
#      sits outside the marker pair. The generator enforces that explicitly.
#
# 4) Placement legality:
#    - Marker positions are chosen as [a, a + step] with a >= step and a + 2*step < L,
#      guaranteeing at least one legal outward beat on both sides.
#    - A companion, if present, is placed on the same beat strictly left of a or
#      strictly right of b, never between them.
#    - A bystander, if present, is chosen from off-beat cells distinct from both
#      markers, so it cannot be mistaken for a companion.
#    - Active lines are spaced apart and each line contributes at most three dots,
#      so no placements overlap and row/column parses stay visually clean.
#
# 5) Visual observability and cleanliness:
#    - The output is intentionally sparse and minimal: the colored instruction dots
#      from the input disappear, leaving only periodic residues on both sides.
#    - Companion lines demonstrate color takeover on the whole trajectory.
#    - Bystander lines demonstrate that some extra dots vanish entirely rather than
#      acting as blockers.
#    - Because every line is isolated and follows the same local grammar, a human can
#      infer the rule from a few examples without relying on color ids or ordering.
#
# 6) New solver assumptions that the generator must satisfy:
#    - Exactly one of row-wise or column-wise parsing is valid.
#    - Every non-empty active line has 2 or 3 non-bg cells, with exactly one repeated
#      color appearing twice.
#    - Any aligned extra dot is unique and outside the marker pair.
#    - Every generated line produces non-empty output on both sides after erasing the
#      two marker cells, so the bilateral rule remains observable.


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))

_LINE_WEIGHTS = {
    "marker_only": 0.25,
    "companion": 0.35,
    "bystander": 0.40,
}

_MAX_RETRIES = 500


def _cell(by_row: bool, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if by_row else (pos, line)


def _parse_axis(
    g: np.ndarray, bg: int, by_row: bool
) -> list[dict[str, int | None | str]] | None:
    num_lines = g.shape[0] if by_row else g.shape[1]
    line_len = g.shape[1] if by_row else g.shape[0]

    parsed: list[dict[str, int | None | str]] = []
    saw_nonempty = False

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        total = 0
        for p in range(line_len):
            value = int(g[_cell(by_row, li, p)])
            if value != bg:
                colors[value].append(p)
                total += 1

        if total == 0:
            continue

        saw_nonempty = True
        if total not in (2, 3):
            return None

        repeating = [(color, positions) for color, positions in colors.items() if len(positions) == 2]
        if len(repeating) != 1:
            return None

        marker, marker_pos = repeating[0]
        a, b = sorted(marker_pos)
        step = b - a
        if step <= 0:
            return None

        companion_color = None
        companion_pos = None
        bystander_pos = None
        line_type = "marker_only"

        for color, positions in colors.items():
            if color == marker:
                continue
            if len(positions) != 1:
                return None
            pos = positions[0]
            if (pos - a) % step == 0:
                if companion_pos is not None:
                    return None
                if a < pos < b:
                    return None
                companion_color = color
                companion_pos = pos
                line_type = "companion"
            else:
                if bystander_pos is not None:
                    return None
                bystander_pos = pos
                line_type = "bystander"

        parsed.append(
            {
                "line": li,
                "marker": marker,
                "a": a,
                "b": b,
                "step": step,
                "companion_color": companion_color,
                "companion_pos": companion_pos,
                "bystander_pos": bystander_pos,
                "line_type": line_type,
            }
        )

    if not saw_nonempty:
        return None
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow periodic rays symmetrically from interior marker pairs and keep only the final trajectories.

    Input structure:
    - bg is the most common color. Sparse non-bg dots lie on either several rows or several
      columns, and exactly one of those two axis interpretations is structurally valid.
    - Every active line contains exactly one repeated color with exactly 2 interior dots; their
      spacing is the step of a periodic beat.
    - A line may also contain one extra single dot: either an aligned companion on that same
      beat class, or an off-beat bystander.

    Rule:
    - Infer whether rows or columns are the active lines by finding the unique axis whose
      non-empty lines all match the "marker pair + optional one extra dot" grammar.
    - For each active line, extend the beat symmetrically: walk left from the first marker and
      right from the second marker, stopping a side at the first border or aligned companion.
    - Without a companion, paint the trajectory in the marker color. With a companion, paint the
      whole trajectory in the companion color.
    - Markers and bystanders are erased to background. The output keeps only non-marker beat cells,
      including a companion cell if it lies on the trajectory.

    Key insight: the repeated pair is no longer part of the answer and no boundary launches the ray.
    The pair is just a local instruction seed for a bilateral arithmetic progression, while any
    off-beat extra dot is noise that disappears.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    candidates: list[tuple[bool, list[dict[str, int | None | str]]]] = []
    for by_row in (True, False):
        parsed = _parse_axis(g, bg, by_row)
        if parsed is not None:
            candidates.append((by_row, parsed))

    assert len(candidates) == 1, f"Ambiguous structured axis: {[c[0] for c in candidates]}"
    by_row, parsed_lines = candidates[0]
    line_len = g.shape[1] if by_row else g.shape[0]
    out = np.full_like(g, bg)

    for info in parsed_lines:
        li = int(info["line"])
        marker = int(info["marker"])
        a = int(info["a"])
        b = int(info["b"])
        step = int(info["step"])
        companion_color = info["companion_color"]
        companion_pos = info["companion_pos"]

        paint = marker if companion_color is None else int(companion_color)
        fill_positions: set[int] = set()

        pos = a - step
        while pos >= 0:
            fill_positions.add(pos)
            if companion_pos is not None and pos == int(companion_pos):
                break
            pos -= step

        pos = b + step
        while pos < line_len:
            fill_positions.add(pos)
            if companion_pos is not None and pos == int(companion_pos):
                break
            pos += step

        for p in fill_positions:
            out[_cell(by_row, li, p)] = paint

    return out


def _choose_dims(by_row: bool, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        h = random.randint(lo, _ROWS_RANGE[1])
        w = random.randint(*_COLS_RANGE)
        return h, w, h, w

    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    w = random.randint(lo, _COLS_RANGE[1])
    h = random.randint(*_ROWS_RANGE)
    return h, w, w, h


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _weighted_choice(options: list[str]) -> str:
    weights = [_LINE_WEIGHTS[option] for option in options]
    return random.choices(options, weights=weights, k=1)[0]


def _choose_marker_pair(line_len: int) -> tuple[int, int] | None:
    feasible_steps = []
    for step in range(_STEP_RANGE[0], min(_STEP_RANGE[1], (line_len - 1) // 3) + 1):
        starts = list(range(step, line_len - 2 * step))
        if starts:
            feasible_steps.append(step)
    if not feasible_steps:
        return None
    step = random.choice(feasible_steps)
    a = random.choice(list(range(step, line_len - 2 * step)))
    return a, step


def _build_line_dots(
    line_len: int, marker_color: int
) -> tuple[list[tuple[int, int]], dict[str, int | str | None]]:
    marker_pair = _choose_marker_pair(line_len)
    if marker_pair is None:
        raise ValueError("line too short for interior bilateral markers")

    a, step = marker_pair
    b = a + step
    dots = [(a, marker_color), (b, marker_color)]

    left_beats = list(range(a - step, -1, -step))
    right_beats = list(range(b + step, line_len, step))
    other_colors = [color for color in _COLORS if color != marker_color]
    bystander_positions = [p for p in range(line_len) if (p - a) % step != 0 and p not in (a, b)]

    feasible = ["marker_only"]
    if other_colors and (left_beats or right_beats):
        feasible.append("companion")
    if other_colors and bystander_positions:
        feasible.append("bystander")

    kind = _weighted_choice(feasible)
    meta: dict[str, int | str | None] = {
        "a": a,
        "b": b,
        "step": step,
        "type": kind,
        "companion_pos": None,
        "companion_color": None,
        "bystander_pos": None,
    }

    if kind == "companion":
        sides = []
        if left_beats:
            sides.append(("left", left_beats))
        if right_beats:
            sides.append(("right", right_beats))
        side, beats = random.choice(sides)
        companion_pos = random.choice(beats)
        companion_color = random.choice(other_colors)
        dots.append((companion_pos, companion_color))
        meta["companion_pos"] = companion_pos
        meta["companion_color"] = companion_color
        meta["companion_side"] = side
    elif kind == "bystander":
        bystander_pos = random.choice(bystander_positions)
        dots.append((bystander_pos, random.choice(other_colors)))
        meta["bystander_pos"] = bystander_pos

    return dots, meta


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    parsed_rows = _parse_axis(inp, bg, True)
    parsed_cols = _parse_axis(inp, bg, False)
    assert (parsed_rows is None) != (parsed_cols is None)

    by_row = parsed_rows is not None
    parsed = parsed_rows if parsed_rows is not None else parsed_cols
    assert parsed is not None

    erased_markers = 0
    erased_bystanders = 0
    companion_lines = 0
    bystander_lines = 0
    bilateral_lines = 0

    for info in parsed:
        li = int(info["line"])
        a = int(info["a"])
        b = int(info["b"])

        for pos in (a, b):
            if int(out[_cell(by_row, li, pos)]) == bg:
                erased_markers += 1

        companion_pos = info["companion_pos"]
        if companion_pos is not None:
            companion_lines += 1

        bystander_pos = info["bystander_pos"]
        if bystander_pos is not None:
            bystander_lines += 1
            if int(out[_cell(by_row, li, int(bystander_pos))]) == bg:
                erased_bystanders += 1

        has_left = False
        has_right = False
        line_len = inp.shape[1] if by_row else inp.shape[0]
        for p in range(line_len):
            if int(out[_cell(by_row, li, p)]) == bg:
                continue
            if p < a:
                has_left = True
            if p > b:
                has_right = True
        if has_left and has_right:
            bilateral_lines += 1

    new_trail_cells = int(np.count_nonzero((inp == bg) & (out != bg)))
    kept_input_nonbg = int(np.count_nonzero((inp != bg) & (out != bg)))

    return {
        "active_lines": len(parsed),
        "erased_markers": erased_markers,
        "erased_bystanders": erased_bystanders,
        "companion_lines": companion_lines,
        "bystander_lines": bystander_lines,
        "bilateral_lines": bilateral_lines,
        "new_trail_cells": new_trail_cells,
        "kept_input_nonbg": kept_input_nonbg,
    }


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    by_row = random.choice((True, False))

    dims = _choose_dims(by_row, n_active)
    if dims is None:
        return None
    h, w, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    inp = np.zeros((h, w), dtype=int)
    kinds: list[str] = []

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        dots, meta = _build_line_dots(line_len, marker_color)

        for pos, color in dots:
            rc = _cell(by_row, li, pos)
            if inp[rc] != 0:
                return None
            inp[rc] = color

        kinds.append(str(meta["type"]))

    bg = int(get_bg(inp))
    parsed_rows = _parse_axis(inp, bg, True)
    parsed_cols = _parse_axis(inp, bg, False)
    if (parsed_rows is None) == (parsed_cols is None):
        return None
    if (parsed_rows is not None) != by_row:
        return None

    if "companion" not in kinds or "bystander" not in kinds:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["active_lines"] != n_active:
        return None
    if metrics["erased_markers"] != 2 * n_active:
        return None
    if metrics["erased_bystanders"] < 1:
        return None
    if metrics["companion_lines"] < 1 or metrics["bystander_lines"] < 1:
        return None
    if metrics["bilateral_lines"] != n_active:
        return None
    if metrics["new_trail_cells"] < n_active:
        return None
    if metrics["kept_input_nonbg"] != metrics["companion_lines"]:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
