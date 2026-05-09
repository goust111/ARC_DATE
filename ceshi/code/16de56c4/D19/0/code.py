# Variant of 16de56c4
# Dimension: D19 — 符号编码 (symbol_instructions)
# Change: 无符号编码——所有规则参数直接从点的空间位置推导... → 图例区域

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D19 variant
# -----------------------------------
# a) Concrete layout and behavior
# - The original task infers the paint color directly from the line itself:
#   marker-only lines paint with the marker color, and companion lines paint
#   with the companion color.
# - The variant moves only that color-choice rule into a corner legend.  The
#   main geometry stays the same: a unique shooting edge, exactly two marker
#   dots on each active line, modular beat filling, optional aligned companion
#   as a stopper, optional unaligned bystander left untouched.
# - The legend is a tiny corner strip:
#   - row-based shooting (left/right) -> a 2-row horizontal legend
#   - column-based shooting (top/bottom) -> a 2-column vertical legend
#   Each legend pair is "marker color -> fill color", with a one-cell gap
#   between pairs so humans can read it as a lookup table.
# - Trace on one line:
#   old logic: marker 2 at positions 0 and 3, aligned companion 8 at 6
#              -> fill beat cells 0,3,6 with color 8
#   new logic: the same spatial pattern, but legend says 2 -> 5
#              -> companion still limits the span to 0..6, yet filled cells
#                 become color 5 instead of 8
#   Difference: the companion still controls geometry, but the legend now
#   controls the resulting color.
#
# b) Complexity calibration
# - The legend encodes 2-4 mappings per grid.  Every key color is reused on at
#   least one active line, so the mapping is not a one-off curiosity.
# - Fill colors are sampled distinct from marker colors to make the D19 change
#   visually obvious in a few examples.
# - Companion/bystander colors come from a neutral leftover pool when possible,
#   which avoids accidental "companion color = output color" cues.
#
# c) Implicit assumptions audit
# - Original code assumes any aligned non-marker color is also the paint color.
#   That is no longer true; main() must separate "which dot stops the ray" from
#   "which color paints the ray".
# - Original edge counting blindly uses all boundary dots.  The variant keeps
#   that viable by constructing the legend on non-shooting corners and forcing
#   the shooting edge to still have a strict boundary-dot majority.
# - Original per-line scan treats every non-bg cell on an active line as part of
#   the line logic.  The variant therefore reserves the legend's two rows or two
#   columns as inactive main-grid lines and main() skips them.
#
# d) Placement legality
# - Legend rows/cols are reserved, so legend pixels never overlap line markers.
# - Active lines are sampled with a gap >= 1 exactly as before, but only from
#   the non-legend rows/cols.
# - Companion dots remain beyond the interior marker along the same beat.
# - Bystanders are explicitly off-beat.
#
# e/f/g) Observability and cleanliness
# - The legend is always in a corner, compact, and isolated from the line field.
# - The mapping is readable from local adjacency alone; no color-id ordering or
#   hidden numeric convention is used.
# - Because fill colors differ from marker colors, and because companion colors
#   are not reused as paint colors, the legend effect is visible in the output.


def _edge_counts(g: np.ndarray, bg: int) -> dict[str, int]:
    H, W = g.shape
    return {
        "top": sum(1 for c in range(W) if int(g[0, c]) != bg),
        "bottom": sum(1 for c in range(W) if int(g[H - 1, c]) != bg),
        "left": sum(1 for r in range(H) if int(g[r, 0]) != bg),
        "right": sum(1 for r in range(H) if int(g[r, W - 1]) != bg),
    }


def _scan_horizontal_legend(
    g: np.ndarray, bg: int, *, top: bool, right: bool
) -> dict[str, object] | None:
    H, W = g.shape
    if H < 2:
        return None
    r0 = 0 if top else H - 2
    rows = slice(r0, r0 + 2)

    cols: list[int] = []
    idx = 0
    while True:
        c = W - 1 - 2 * idx if right else 2 * idx
        if not (0 <= c < W):
            break
        marker = int(g[r0, c])
        fill = int(g[r0 + 1, c])
        if marker == bg or fill == bg:
            break
        if idx > 0:
            sep = c + 1 if right else c - 1
            if int(g[r0, sep]) != bg or int(g[r0 + 1, sep]) != bg:
                return None
        cols.append(c)
        idx += 1

    if len(cols) < 2:
        return None

    allowed = np.zeros((2, W), dtype=bool)
    for c in cols:
        allowed[:, c] = True
    if ((g[rows, :] != bg) & (~allowed)).any():
        return None

    mapping: dict[int, int] = {}
    for c in cols:
        marker = int(g[r0, c])
        fill = int(g[r0 + 1, c])
        if marker in mapping:
            return None
        mapping[marker] = fill

    return {
        "mapping": mapping,
        "blocked_lines": {r0, r0 + 1},
        "cells": {(r0, c) for c in cols} | {(r0 + 1, c) for c in cols},
    }


def _scan_vertical_legend(
    g: np.ndarray, bg: int, *, bottom: bool, right: bool
) -> dict[str, object] | None:
    H, W = g.shape
    if W < 2:
        return None
    c0 = W - 2 if right else 0
    cols = slice(c0, c0 + 2)

    rows: list[int] = []
    idx = 0
    while True:
        r = H - 1 - 2 * idx if bottom else 2 * idx
        if not (0 <= r < H):
            break
        marker = int(g[r, c0])
        fill = int(g[r, c0 + 1])
        if marker == bg or fill == bg:
            break
        if idx > 0:
            sep = r + 1 if bottom else r - 1
            if int(g[sep, c0]) != bg or int(g[sep, c0 + 1]) != bg:
                return None
        rows.append(r)
        idx += 1

    if len(rows) < 2:
        return None

    allowed = np.zeros((H, 2), dtype=bool)
    for r in rows:
        allowed[r, :] = True
    if ((g[:, cols] != bg) & (~allowed)).any():
        return None

    mapping: dict[int, int] = {}
    for r in rows:
        marker = int(g[r, c0])
        fill = int(g[r, c0 + 1])
        if marker in mapping:
            return None
        mapping[marker] = fill

    return {
        "mapping": mapping,
        "blocked_lines": {c0, c0 + 1},
        "cells": {(r, c0) for r in rows} | {(r, c0 + 1) for r in rows},
    }


def _find_legend(g: np.ndarray, bg: int, edge: str) -> dict[str, object]:
    by_row = edge in ("left", "right")
    candidates: list[dict[str, object]] = []

    if by_row:
        if edge == "left":
            scans = [
                _scan_horizontal_legend(g, bg, top=True, right=True),
                _scan_horizontal_legend(g, bg, top=False, right=True),
            ]
        else:
            scans = [
                _scan_horizontal_legend(g, bg, top=True, right=False),
                _scan_horizontal_legend(g, bg, top=False, right=False),
            ]
    else:
        if edge == "top":
            scans = [
                _scan_vertical_legend(g, bg, bottom=True, right=False),
                _scan_vertical_legend(g, bg, bottom=True, right=True),
            ]
        else:
            scans = [
                _scan_vertical_legend(g, bg, bottom=False, right=False),
                _scan_vertical_legend(g, bg, bottom=False, right=True),
            ]

    for cand in scans:
        if cand is not None:
            candidates.append(cand)

    assert len(candidates) == 1, f"Expected exactly one legend, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a corner legend, then fill periodic rays from a unique shooting edge.

    Input structure:
    - The main field keeps the original sparse-dot geometry: one unique shooting
      edge, active parallel lines, and exactly two repeated marker dots per line.
    - A small corner legend lists marker-color to fill-color mappings in paired
      cells; its rows or columns are reserved and contain no main-grid markers.
    - A different-colored aligned companion still marks a stopping span, while an
      unaligned bystander remains an untouched spectator.

    Rule:
    - Identify the shooting edge from the boundary with the strict non-bg maximum.
    - Find the corner legend on the opposite side and decode marker -> fill color.
    - On each non-legend shooting-direction line, the marker pair still defines
      the period and phase, and an aligned companion still limits the fill span.
    - The paint color now comes from the legend entry of the repeating marker
      color, not from the companion dot itself.

    Key insight: the spatial part of the task is unchanged; only the color choice
    has been externalized into a compact legend, so solving requires combining a
    local lookup table with the original modular ray logic.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge_counts = _edge_counts(g, bg)
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    legend = _find_legend(g, bg, edge)
    mapping = legend["mapping"]
    blocked_lines = legend["blocked_lines"]

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        if li in blocked_lines:
            continue

        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) should have exactly one marker "
            f"color, got {dict(colors)}"
        )

        marker, mpos = next(iter(repeating.items()))
        assert marker in mapping, f"Marker color {marker} missing from legend"
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2): {mpos}"
        )

        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        companion = None
        for other_color, other_positions in colors.items():
            if other_color == marker:
                continue
            if any(pos % step == offset for pos in other_positions):
                companion = other_color
                break

        paint = int(mapping[marker])
        if companion is not None:
            all_positions = mpos + colors[companion]
            lo, hi = min(all_positions), max(all_positions)
        else:
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LEGEND_KEY_RANGE = (2, 4)
_MIN_ACTIVE_LINES = 4
_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.40
_MAX_RETRIES = 400


def _spaced_indices(n: int, start: int, length: int) -> list[int]:
    available = length - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [start + picks[i] + i for i in range(n)]


def _choose_geometry() -> tuple[int, int, str, int] | None:
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")
    H = random.randint(*_ROWS_RANGE)
    W = random.randint(*_COLS_RANGE)
    usable_lines = H - 2 if by_row else W - 2
    max_active = (usable_lines + 1) // 2
    if max_active < _MIN_ACTIVE_LINES:
        return None
    n_active = random.randint(_MIN_ACTIVE_LINES, max_active)
    return H, W, edge, n_active


def _sample_palette(k: int) -> tuple[list[int], dict[int, int], list[int]]:
    marker_colors = random.sample(_COLORS, k)
    remaining = [c for c in _COLORS if c not in marker_colors]
    fill_colors = random.sample(remaining, k)
    neutral = [c for c in remaining if c not in fill_colors]
    return marker_colors, dict(zip(marker_colors, fill_colors)), neutral


def _legend_layout(
    H: int, W: int, edge: str, k: int
) -> tuple[set[int], list[tuple[int, int, int, int]]]:
    by_row = edge in ("left", "right")
    if by_row:
        top = bool(random.getrandbits(1))
        right = edge == "left"
        r0 = 0 if top else H - 2
        blocked = {r0, r0 + 1}
        pairs = []
        for i in range(k):
            c = W - 1 - 2 * i if right else 2 * i
            pairs.append((r0, c, r0 + 1, c))
        return blocked, pairs

    bottom = edge == "top"
    right = bool(random.getrandbits(1))
    c0 = W - 2 if right else 0
    blocked = {c0, c0 + 1}
    pairs = []
    for i in range(k):
        r = H - 1 - 2 * i if bottom else 2 * i
        pairs.append((r, c0, r, c0 + 1))
    return blocked, pairs


def _active_lines(H: int, W: int, edge: str, blocked: set[int], n_active: int) -> list[int]:
    by_row = edge in ("left", "right")
    if by_row:
        if 0 in blocked:
            return _spaced_indices(n_active, 2, H - 2)
        return _spaced_indices(n_active, 0, H - 2)
    if 0 in blocked:
        return _spaced_indices(n_active, 2, W - 2)
    return _spaced_indices(n_active, 0, W - 2)


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    neutral_colors: list[int],
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    step = min(random.randint(*_STEP_RANGE), line_len - 1)
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
        forward_beats = list(range(inner_pos + step, line_len, step))
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
        forward_beats = list(range(inner_pos - step, -1, -step))
    offset = edge_pos % step

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    line_type = "marker_only"

    other_colors = neutral_colors or [c for c in _COLORS if c != marker_color]
    roll = random.random()
    if roll < _P_MARKER_ONLY:
        pass
    elif roll < _P_MARKER_ONLY + _P_COMPANION and other_colors and forward_beats:
        dots.append((random.choice(forward_beats), random.choice(other_colors)))
        line_type = "companion"
    else:
        non_beat = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in (edge_pos, inner_pos)
        ]
        if other_colors and non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))
            line_type = "bystander"

    spec = {
        "marker_color": marker_color,
        "step": step,
        "offset": offset,
        "line_type": line_type,
    }
    return dots, spec


def _render_expected(
    inp: np.ndarray,
    edge: str,
    line_specs: list[dict[str, object]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    H, W = inp.shape
    by_row = edge in ("left", "right")
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for spec in line_specs:
        li = int(spec["line"])
        marker_color = int(spec["marker_color"])
        marker_positions = list(spec["marker_positions"])
        companion_positions = list(spec["companion_positions"])
        step = int(spec["step"])
        offset = int(spec["offset"])
        paint = int(mapping[marker_color])

        if companion_positions:
            all_positions = marker_positions + companion_positions
            lo, hi = min(all_positions), max(all_positions)
        else:
            lo, hi = 0, line_len - 1

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            cur = int(out[rc])
            if cur == 0 or cur == marker_color:
                out[rc] = paint

    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    geom = _choose_geometry()
    if geom is None:
        return None
    H, W, edge, n_active = geom

    max_keys = min(_LEGEND_KEY_RANGE[1], n_active - 1, 4)
    if max_keys < _LEGEND_KEY_RANGE[0]:
        return None
    k = random.randint(_LEGEND_KEY_RANGE[0], max_keys)
    marker_colors, mapping, neutral_colors = _sample_palette(k)

    blocked, legend_pairs = _legend_layout(H, W, edge, k)
    active_lines = _active_lines(H, W, edge, blocked, n_active)
    if len(active_lines) != n_active:
        return None

    by_row = edge in ("left", "right")
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)

    for (r1, c1, r2, c2), marker_color in zip(legend_pairs, marker_colors):
        inp[r1, c1] = int(marker_color)
        inp[r2, c2] = int(mapping[marker_color])

    assigned_markers = marker_colors.copy()
    assigned_markers.extend(random.choice(marker_colors) for _ in range(n_active - k))
    random.shuffle(assigned_markers)

    line_specs: list[dict[str, object]] = []
    for li, marker_color in zip(active_lines, assigned_markers):
        dots, spec = _build_line_dots(edge, line_len, marker_color, neutral_colors)
        marker_positions: list[int] = []
        companion_positions: list[int] = []

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = int(color)
            if color == marker_color:
                marker_positions.append(pos)
            elif pos % int(spec["step"]) == int(spec["offset"]):
                companion_positions.append(pos)

        spec["line"] = li
        spec["marker_positions"] = tuple(sorted(marker_positions))
        spec["companion_positions"] = tuple(sorted(companion_positions))
        line_specs.append(spec)

    counts = _edge_counts(inp, 0)
    winners = [name for name, value in counts.items() if value == max(counts.values())]
    if winners != [edge]:
        return None

    expected = _render_expected(inp, edge, line_specs, mapping)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
