# Variant of 16de56c4
# Dimension: D5 — 终止条件 (dst)
# Change: 无伴随点时填满整行；有伴随点时填充标记点与伴随点的span范围内... → 对称填充

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


"""
Design analysis for the D5 variant
==================================

spec:
  preserved_structure:
    grid: 7-12 rows x 9-21 cols
    background: 0 / most common color
    global_axis: exactly one of row-wise or col-wise is structurally valid
    active_lines: 4-9 non-empty lines on that axis, with gaps >= 1
    per_active_line:
      marker: exactly one color appears exactly twice; the gap is the step
      extra_dot: optional single companion or optional single bystander
      total_dots_per_line: 2 or 3
  D5_change:
    old_behavior:
      - marker pair was tied to a unique shooting edge
      - no companion -> fill the whole line on the marker beat
      - companion -> fill only the span between marker and companion
    new_behavior:
      - remove the shooting-edge concept entirely
      - marker pair is an interior seed that defines a periodic beat
      - extend the same beat class in both directions from that seed
      - each side stops independently at the first edge or aligned obstacle
      - if a companion exists, it still sets the paint color for the whole fill
      - unaligned bystanders stay unchanged

behavior_trace:
  original_line_example:
    line_len: 11
    markers: [0, 3]
    companion: [9]
    output: fill 0,3,6,9 only (one-sided from the edge)
  variant_line_example:
    line_len: 11
    markers: [3, 6]
    companion: [0]
    bystander: none
    output: fill 0,3,6,9 with companion color
    difference:
      - the same arithmetic progression is used
      - but the known pair now sits in the interior
      - growth is bilateral: left side reaches the obstacle at 0,
        right side independently continues to the far edge-side beat at 9

complexity_calibration:
  - Keep exactly one repeated color per active line, so the step is still easy to infer.
  - Keep at most one aligned companion and at most one unaligned bystander,
    so the rule stays learnable from 2-4 examples.
  - Place the marker pair away from the border, so bilateral growth is visible
    and the task does not visually collapse back to the original edge-shooting regime.

implicit_assumption_audit:
  - Original edge counting is invalid now because no marker is forced onto any border.
    Replace it with axis parsing: the intended axis is the only one whose non-empty
    lines all have the expected "two same-color markers + optional one extra dot" form.
  - Original lo/hi logic assumed one-sided travel from a boundary. That is the exact
    D5 dependency, so it is replaced by two independent walks:
      left from the first marker and right from the second marker.
  - Original "companion color overrides paint color" remains valid and is preserved.
  - Original safe overwrite condition (only bg/marker cells get repainted) remains valid
    because the generator guarantees that bystanders are off-beat and companions are unique.

placement_legality:
  - Marker positions are chosen as [a, a+step] with a-step >= 0 and a+2*step < line_len.
    This guarantees at least one outward beat on both sides.
  - Companion, if present, is placed on the same beat class strictly outside the marker pair.
  - Bystander, if present, is placed off the beat class and away from the marker cells.
  - Active lines are distinct, so dots never overlap.

visual_observability:
  - Every generated active line has visible fill added on both sides of the marker pair.
  - Samples mix open lines and companion-limited lines often enough that humans can see:
    "same periodic rhythm, but now expansion starts from the middle rather than from an edge."

cleanliness_and_color_reasoning:
  - No color-id ordering is used.
  - Companion colors are visible as direct replacements of the marker pair and all newly
    filled beats on that line, which gives a clear structural cue for the color change.
"""


def _cell(by_row: bool, line: int, pos: int) -> tuple[int, int]:
    return (line, pos) if by_row else (pos, line)


def _parse_axis(
    g: np.ndarray, bg: int, by_row: bool
) -> list[dict[str, int | None]] | None:
    H, W = g.shape
    num_lines = H if by_row else W
    line_len = W if by_row else H

    parsed: list[dict[str, int | None]] = []
    saw_nonempty = False

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        total = 0
        for p in range(line_len):
            v = int(g[_cell(by_row, li, p)])
            if v != bg:
                colors[v].append(p)
                total += 1

        if total == 0:
            continue

        saw_nonempty = True
        if total not in (2, 3):
            return None

        repeating = [(color, ps) for color, ps in colors.items() if len(ps) == 2]
        if len(repeating) != 1:
            return None

        marker, marker_pos = repeating[0]
        a, b = sorted(marker_pos)
        step = b - a
        if step <= 0:
            return None

        aligned: list[tuple[int, int]] = []
        bystanders = 0
        for color, ps in colors.items():
            if color == marker:
                continue
            if len(ps) != 1:
                return None
            pos = ps[0]
            if (pos - a) % step == 0:
                aligned.append((color, pos))
            else:
                bystanders += 1

        if len(aligned) > 1 or bystanders > 1:
            return None

        companion_color = None
        companion_pos = None
        if aligned:
            companion_color, companion_pos = aligned[0]
            if a < companion_pos < b:
                return None

        parsed.append(
            {
                "line": li,
                "marker": marker,
                "a": a,
                "b": b,
                "step": step,
                "companion_color": companion_color,
                "companion_pos": companion_pos,
            }
        )

    if not saw_nonempty:
        return None
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic progressions by growing away from an interior marker pair.

    Input structure:
    - bg is the most common color. Sparse non-bg dots lie on either several rows
      or several columns, and only one of those two axis interpretations is valid.
    - Each active line contains exactly one repeated color with exactly 2 dots;
      those two dots are consecutive beats of a periodic progression, so their gap
      defines the step.
    - A line may also contain one extra single dot: either an aligned companion on
      the same beat class, or an unaligned bystander that should remain untouched.

    Rule:
    - Infer whether rows or columns are the active lines by finding the unique axis
      whose non-empty lines all match the "marker pair + optional one extra dot" pattern.
    - For each active line, use the two marker dots to recover the step.
    - Extend that beat class in both directions: left from the first marker and right
      from the second marker, stopping each side independently at the first border or
      aligned companion.
    - Without a companion, paint with the marker color. With a companion, paint every
      filled beat with the companion color. Unaligned bystanders are unchanged.

    Key insight: the repeated color no longer points back to a shooting edge; instead
    it acts as a local seed for an arithmetic progression, and termination is determined
    by bilateral growth from the seed rather than by a one-sided span.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    candidates: list[tuple[bool, list[dict[str, int | None]]]] = []
    for by_row in (True, False):
        parsed = _parse_axis(g, bg, by_row)
        if parsed is not None:
            candidates.append((by_row, parsed))

    assert len(candidates) == 1, f"Ambiguous structured axis: {[c[0] for c in candidates]}"
    by_row, parsed_lines = candidates[0]
    line_len = g.shape[1] if by_row else g.shape[0]

    for info in parsed_lines:
        li = int(info["line"])
        marker = int(info["marker"])
        a = int(info["a"])
        b = int(info["b"])
        step = int(info["step"])
        companion_color = info["companion_color"]
        companion_pos = info["companion_pos"]

        paint = marker if companion_color is None else int(companion_color)
        fill_positions = {a, b}

        pos = a
        while True:
            nxt = pos - step
            if nxt < 0:
                break
            fill_positions.add(nxt)
            if companion_pos is not None and nxt == int(companion_pos):
                break
            pos = nxt

        pos = b
        while True:
            nxt = pos + step
            if nxt >= line_len:
                break
            fill_positions.add(nxt)
            if companion_pos is not None and nxt == int(companion_pos):
                break
            pos = nxt

        for p in fill_positions:
            rc = _cell(by_row, li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.45

_MAX_RETRIES = 400


def _choose_dims(by_row: bool, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W

    lo = max(_COLS_RANGE[0], min_num_lines)
    if lo > _COLS_RANGE[1]:
        return None
    W = random.randint(lo, _COLS_RANGE[1])
    H = random.randint(*_ROWS_RANGE)
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _choose_marker_pair(line_len: int) -> tuple[int, int] | None:
    max_step = min(_STEP_RANGE[1], (line_len - 1) // 3)
    if max_step < _STEP_RANGE[0]:
        return None
    step = random.randint(_STEP_RANGE[0], max_step)
    starts = list(range(step, line_len - 2 * step))
    if not starts:
        return None
    a = random.choice(starts)
    return a, step


def _build_line_dots(line_len: int, marker_color: int) -> tuple[list[tuple[int, int]], dict[str, int | str]]:
    marker_pair = _choose_marker_pair(line_len)
    if marker_pair is None:
        raise ValueError("line too short for bilateral marker placement")

    a, step = marker_pair
    b = a + step
    dots = [(a, marker_color), (b, marker_color)]
    meta: dict[str, int | str] = {"a": a, "b": b, "step": step, "type": "marker_only"}

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        return dots, meta

    if roll < _P_MARKER_ONLY + _P_COMPANION:
        companion_side = random.choice(("left", "right"))
        if companion_side == "left":
            beats = list(range(a - step, -1, -step))
        else:
            beats = list(range(b + step, line_len, step))
        other_colors = [c for c in _COLORS if c != marker_color]
        if other_colors and beats:
            companion_pos = random.choice(beats)
            companion_color = random.choice(other_colors)
            dots.append((companion_pos, companion_color))
            meta.update(
                {
                    "type": "companion",
                    "companion_side": companion_side,
                    "companion_pos": companion_pos,
                    "companion_color": companion_color,
                }
            )
            return dots, meta

    other_colors = [c for c in _COLORS if c != marker_color]
    non_beat = [
        p
        for p in range(line_len)
        if (p - a) % step != 0 and p not in (a, b)
    ]
    if other_colors and non_beat:
        bystander_pos = random.choice(non_beat)
        dots.append((bystander_pos, random.choice(other_colors)))
        meta.update({"type": "bystander", "bystander_pos": bystander_pos})
    return dots, meta


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    parsed_rows = _parse_axis(inp, bg, True)
    parsed_cols = _parse_axis(inp, bg, False)
    assert (parsed_rows is None) != (parsed_cols is None)
    by_row = parsed_rows is not None
    parsed = parsed_rows if parsed_rows is not None else parsed_cols
    assert parsed is not None

    bilateral_lines = 0
    companion_lines = 0
    left_companions = 0
    right_companions = 0

    for info in parsed:
        li = int(info["line"])
        marker = int(info["marker"])
        a = int(info["a"])
        b = int(info["b"])
        row_has_left = False
        row_has_right = False

        line_len = inp.shape[1] if by_row else inp.shape[0]
        for p in range(line_len):
            rc = _cell(by_row, li, p)
            if int(out[rc]) != bg and int(inp[rc]) != int(out[rc]):
                if p < a:
                    row_has_left = True
                if p > b:
                    row_has_right = True

        if row_has_left and row_has_right:
            bilateral_lines += 1

        companion_pos = info["companion_pos"]
        if companion_pos is not None:
            companion_lines += 1
            if int(companion_pos) < a:
                left_companions += 1
            else:
                right_companions += 1

    return {
        "bilateral_lines": bilateral_lines,
        "companion_lines": companion_lines,
        "left_companions": left_companions,
        "right_companions": right_companions,
        "active_lines": len(parsed),
    }


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    by_row = random.choice((True, False))

    dims = _choose_dims(by_row, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    inp = np.zeros((H, W), dtype=int)
    line_meta: list[dict[str, int | str]] = []

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        dots, meta = _build_line_dots(line_len, marker_color)
        for pos, color in dots:
            r, c = _cell(by_row, li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
        line_meta.append(meta)

    has_companion = any(meta["type"] == "companion" for meta in line_meta)
    has_open_line = any(meta["type"] != "companion" for meta in line_meta)
    if not (has_companion and has_open_line):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["bilateral_lines"] < max(2, metrics["active_lines"] // 2):
        return None
    if metrics["companion_lines"] == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
