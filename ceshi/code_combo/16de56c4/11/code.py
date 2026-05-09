# Combo variant of 16de56c4 (index 11)
# combo_spec: D4/3+D19/0
# Name: 色块分组+图例区域

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior trace
# - The original task used two same-color pixels on a line; their gap was the
#   step directly. Here those two pixels become two equal 1D bars, but the bar
#   length is no longer the step itself.
# - Each active line now carries a grouped marker made of two equal bars:
#   one touches the shooting edge, the second sits further inside, and the empty
#   gap between the bars equals the bar length. That grouped shape tells you only
#   which size-class the line belongs to.
# - A corner legend supplies the missing step information. For row-based shooting
#   it is a 2-row strip; for column-based shooting it is a 2-column strip. Each
#   legend entry is a same-color pair of bars: the boundary-nearer bar shows a
#   marker size class, and the adjacent inner bar shows the step paired with that
#   size. Main-grid marker colors do not participate in this lookup.
# - Example on a left-edge row:
#     legend says size 2 -> step 3 and size 3 -> step 2
#     row marker bars are [0..1] and [4..5], so the line uses size 2
#     decoded step is 3, anchor is at position 0, so beats are 0,3,6,9,...
#     if an aligned extra pixel sits at 9, the fill is truncated to 0..9 and
#     the whole marker glyph is recolored to that companion color
#   The difference from the D4-only version is that size no longer numerically
#   equals the beat period; the legend must be consulted first.
#
# b) Complexity calibration
# - The generator uses exactly two size classes per grid and repeats both across
#   several active lines. That keeps the legend small and readable while still
#   making the lookup necessary.
# - The size->step mapping is always non-identity, so the fused task does not
#   collapse back to "bar width = step".
# - Every generated grid includes marker-only, companion, and bystander lines,
#   with counts kept roughly balanced, so solvers must confront every behavioral
#   branch of the original ray rule instead of learning from a one-branch subset.
# - At least one marker color is reused across both size classes, which prevents
#   a misleading "color determines step" shortcut and makes the size-based lookup
#   visually observable.
#
# c) Hidden-assumption audit against the original solver
# - Counting repeated pixel positions is invalid because a marker now contributes
#   contiguous runs, not isolated dots.
# - Treating the gap between the two same-color marks as the step is also invalid:
#   in this combo the two marker bars are spaced by their own size, while the
#   actual step comes from the legend.
# - Inferring paint color from the legend would be wrong here; the legend encodes
#   step only. Paint color still follows the old rule: marker color when there is
#   no aligned companion, companion color when there is one.
# - Raw boundary counts are polluted by the legend because the legend touches a
#   corner. The solver therefore detects the legend first and removes its cells
#   before deciding which edge is the shooter.
#
# d) Placement legality
# - Marker bars lie entirely on one row/column, never overlap, and require
#   `line_len >= 3 * size` so both bars plus their size-equal gap fit.
# - Active lines never use the legend's reserved two rows or two columns.
# - Companions are placed strictly beyond the interior marker along the shooting
#   direction; bystanders are explicitly off-beat.
#
# e/f/g) Observability, cleanliness, and inferability
# - Legend colors are unique to the legend and never reused in the main field, so
#   the corner strip reads as an instruction panel rather than another object.
# - The same marker color can appear with different bar sizes on different lines,
#   which makes it clear that size-class, not color identity, controls the step.
# - Because the size->step lookup is non-identity and repeated, several generated
#   samples visibly differ from any direct-width interpretation.
#
# New implicit assumptions introduced by main(), and how generation satisfies them
# - Exactly one valid legend exists among the eight corner/orientation scans:
#   generator reserves one 2-row or 2-column corner strip and keeps legend colors
#   absent elsewhere.
# - Every non-empty non-legend line has exactly one valid grouped marker:
#   generator places exactly two equal runs of one color, one touching the edge,
#   with gap equal to their size and with that size present in the legend.
# - There is at most one aligned extra color per line:
#   generator places at most one extra pixel on each active line.


def _scan_runs(
    g: np.ndarray, bg: int, line_len: int, cell
) -> dict[int, list[tuple[int, int]]]:
    """Return contiguous non-bg runs on one shooting-direction line."""
    runs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    p = 0
    while p < line_len:
        value = int(g[cell(p)])
        if value == bg:
            p += 1
            continue
        start = p
        while p + 1 < line_len and int(g[cell(p + 1)]) == value:
            p += 1
        runs[value].append((start, p))
        p += 1
    return runs


def _edge_counts(
    g: np.ndarray, bg: int, excluded: set[tuple[int, int]] | None = None
) -> dict[str, int]:
    H, W = g.shape
    blocked = excluded or set()
    return {
        "top": sum(
            1 for c in range(W) if int(g[0, c]) != bg and (0, c) not in blocked
        ),
        "bottom": sum(
            1
            for c in range(W)
            if int(g[H - 1, c]) != bg and (H - 1, c) not in blocked
        ),
        "left": sum(
            1 for r in range(H) if int(g[r, 0]) != bg and (r, 0) not in blocked
        ),
        "right": sum(
            1
            for r in range(H)
            if int(g[r, W - 1]) != bg and (r, W - 1) not in blocked
        ),
    }


def _scan_horizontal_legend(
    g: np.ndarray, bg: int, *, top: bool, right: bool
) -> dict[str, object] | None:
    H, W = g.shape
    if H < 2:
        return None

    size_row = 0 if top else H - 1
    step_row = 1 if top else H - 2
    rows = {size_row, step_row}
    cursor = W - 1 if right else 0

    mapping: dict[int, int] = {}
    colors: set[int] = set()
    cells: set[tuple[int, int]] = set()

    while 0 <= cursor < W:
        size_val = int(g[size_row, cursor])
        step_val = int(g[step_row, cursor])
        if size_val == bg and step_val == bg:
            break
        if size_val == bg or step_val == bg or size_val != step_val:
            return None
        color = size_val
        if color in colors:
            return None
        colors.add(color)

        if right:
            size_start = cursor
            while size_start - 1 >= 0 and int(g[size_row, size_start - 1]) == color:
                size_start -= 1
            step_start = cursor
            while step_start - 1 >= 0 and int(g[step_row, step_start - 1]) == color:
                step_start -= 1

            size_len = cursor - size_start + 1
            step_len = cursor - step_start + 1
            window_start = cursor - max(size_len, step_len) + 1
            if window_start < 0:
                return None

            for c in range(window_start, cursor + 1):
                want_size = size_start <= c <= cursor
                want_step = step_start <= c <= cursor
                top_v = int(g[size_row, c])
                bot_v = int(g[step_row, c])
                if want_size:
                    if top_v != color:
                        return None
                    cells.add((size_row, c))
                elif top_v != bg:
                    return None
                if want_step:
                    if bot_v != color:
                        return None
                    cells.add((step_row, c))
                elif bot_v != bg:
                    return None

            cursor = window_start - 1
            if cursor >= 0:
                if int(g[size_row, cursor]) != bg or int(g[step_row, cursor]) != bg:
                    return None
                cursor -= 1
        else:
            size_end = cursor
            while size_end + 1 < W and int(g[size_row, size_end + 1]) == color:
                size_end += 1
            step_end = cursor
            while step_end + 1 < W and int(g[step_row, step_end + 1]) == color:
                step_end += 1

            size_len = size_end - cursor + 1
            step_len = step_end - cursor + 1
            window_end = cursor + max(size_len, step_len) - 1
            if window_end >= W:
                return None

            for c in range(cursor, window_end + 1):
                want_size = cursor <= c <= size_end
                want_step = cursor <= c <= step_end
                top_v = int(g[size_row, c])
                bot_v = int(g[step_row, c])
                if want_size:
                    if top_v != color:
                        return None
                    cells.add((size_row, c))
                elif top_v != bg:
                    return None
                if want_step:
                    if bot_v != color:
                        return None
                    cells.add((step_row, c))
                elif bot_v != bg:
                    return None

            cursor = window_end + 1
            if cursor < W:
                if int(g[size_row, cursor]) != bg or int(g[step_row, cursor]) != bg:
                    return None
                cursor += 1

        if size_len in mapping or step_len in mapping.values():
            return None
        mapping[size_len] = step_len

    if len(mapping) < 2:
        return None

    for r in rows:
        for c in range(W):
            if int(g[r, c]) != bg and (r, c) not in cells:
                return None

    for r in range(H):
        for c in range(W):
            value = int(g[r, c])
            if value in colors and (r, c) not in cells:
                return None

    return {
        "orientation": "horizontal",
        "mapping": mapping,
        "blocked_lines": rows,
        "cells": cells,
        "colors": colors,
    }


def _scan_vertical_legend(
    g: np.ndarray, bg: int, *, bottom: bool, right: bool
) -> dict[str, object] | None:
    H, W = g.shape
    if W < 2:
        return None

    size_col = W - 1 if right else 0
    step_col = W - 2 if right else 1
    cols = {size_col, step_col}
    cursor = H - 1 if bottom else 0

    mapping: dict[int, int] = {}
    colors: set[int] = set()
    cells: set[tuple[int, int]] = set()

    while 0 <= cursor < H:
        size_val = int(g[cursor, size_col])
        step_val = int(g[cursor, step_col])
        if size_val == bg and step_val == bg:
            break
        if size_val == bg or step_val == bg or size_val != step_val:
            return None
        color = size_val
        if color in colors:
            return None
        colors.add(color)

        if bottom:
            size_start = cursor
            while size_start - 1 >= 0 and int(g[size_start - 1, size_col]) == color:
                size_start -= 1
            step_start = cursor
            while step_start - 1 >= 0 and int(g[step_start - 1, step_col]) == color:
                step_start -= 1

            size_len = cursor - size_start + 1
            step_len = cursor - step_start + 1
            window_start = cursor - max(size_len, step_len) + 1
            if window_start < 0:
                return None

            for r in range(window_start, cursor + 1):
                want_size = size_start <= r <= cursor
                want_step = step_start <= r <= cursor
                left_v = int(g[r, size_col])
                right_v = int(g[r, step_col])
                if want_size:
                    if left_v != color:
                        return None
                    cells.add((r, size_col))
                elif left_v != bg:
                    return None
                if want_step:
                    if right_v != color:
                        return None
                    cells.add((r, step_col))
                elif right_v != bg:
                    return None

            cursor = window_start - 1
            if cursor >= 0:
                if int(g[cursor, size_col]) != bg or int(g[cursor, step_col]) != bg:
                    return None
                cursor -= 1
        else:
            size_end = cursor
            while size_end + 1 < H and int(g[size_end + 1, size_col]) == color:
                size_end += 1
            step_end = cursor
            while step_end + 1 < H and int(g[step_end + 1, step_col]) == color:
                step_end += 1

            size_len = size_end - cursor + 1
            step_len = step_end - cursor + 1
            window_end = cursor + max(size_len, step_len) - 1
            if window_end >= H:
                return None

            for r in range(cursor, window_end + 1):
                want_size = cursor <= r <= size_end
                want_step = cursor <= r <= step_end
                left_v = int(g[r, size_col])
                right_v = int(g[r, step_col])
                if want_size:
                    if left_v != color:
                        return None
                    cells.add((r, size_col))
                elif left_v != bg:
                    return None
                if want_step:
                    if right_v != color:
                        return None
                    cells.add((r, step_col))
                elif right_v != bg:
                    return None

            cursor = window_end + 1
            if cursor < H:
                if int(g[cursor, size_col]) != bg or int(g[cursor, step_col]) != bg:
                    return None
                cursor += 1

        if size_len in mapping or step_len in mapping.values():
            return None
        mapping[size_len] = step_len

    if len(mapping) < 2:
        return None

    for c in cols:
        for r in range(H):
            if int(g[r, c]) != bg and (r, c) not in cells:
                return None

    for r in range(H):
        for c in range(W):
            value = int(g[r, c])
            if value in colors and (r, c) not in cells:
                return None

    return {
        "orientation": "vertical",
        "mapping": mapping,
        "blocked_lines": cols,
        "cells": cells,
        "colors": colors,
    }


def _find_legend(g: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []

    for top in (True, False):
        for right in (True, False):
            cand = _scan_horizontal_legend(g, bg, top=top, right=right)
            if cand is not None:
                candidates.append(cand)

    for bottom in (True, False):
        for right in (True, False):
            cand = _scan_vertical_legend(g, bg, bottom=bottom, right=right)
            if cand is not None:
                candidates.append(cand)

    unique: dict[frozenset[tuple[int, int]], dict[str, object]] = {}
    for cand in candidates:
        unique[frozenset(cand["cells"])] = cand
    candidates = list(unique.values())

    assert len(candidates) == 1, f"Expected exactly one legend, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays from one edge, but decode their step from a corner legend.

    Input structure:
    - Background is the most common color. The main field contains sparse active
      rows or columns, plus one compact legend tucked into a corner strip.
    - Each active line has exactly one grouped marker color: two equal 1D bars of
      that color on the line, one touching the shooting edge and one deeper in the
      grid. Their bar length and their separating gap are equal, so the glyph only
      reveals a size class, not the beat period itself.
    - The legend reserves exactly two rows or two columns. In each legend entry,
      the boundary-nearer bar gives a size class and the adjacent inner bar gives
      the step paired with that size.
    - A single extra pixel on an active line is either an aligned companion that
      stops the ray or an off-beat bystander that remains unchanged.

    Rule:
    - First identify the corner legend and read a size -> step mapping from it.
    - Ignore legend cells when counting boundary cells, then find the unique
      shooting edge; left/right imply row-wise shooting, top/bottom imply columns.
    - On each non-legend line, find the unique marker color whose two equal runs
      have one edge-touching run and a between-run gap equal to the run length.
      Use that run length as the legend key to recover the real step.
    - The edge-nearest end of the edge-touching bar defines the beat phase. If an
      extra pixel shares that modular class, it is a companion: it truncates the
      fill span and its color becomes the paint. Otherwise the marker color paints
      the whole line's beat positions. Companion lines also recolor both marker
      bars completely, mirroring the original absorption behavior.

    Key insight:
    - The grouped marker glyph no longer exposes the period directly. Its size
      must be matched against the corner legend to obtain the true step, and only
      after that lookup does the familiar modular ray logic become solvable.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    legend = _find_legend(g, bg)
    edge_counts = _edge_counts(g, bg, excluded=set(legend["cells"]))
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-legend non-bg cells on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    expected_orientation = "horizontal" if by_row else "vertical"
    assert legend["orientation"] == expected_orientation, (
        f"Legend orientation {legend['orientation']} incompatible with edge {edge}"
    )

    size_to_step = dict(legend["mapping"])
    blocked_lines = set(legend["blocked_lines"])
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def touches_edge(run: tuple[int, int]) -> bool:
        start, end = run
        return start == 0 if edge in ("left", "top") else end == line_len - 1

    for li in range(num_lines):
        if li in blocked_lines:
            continue

        runs = _scan_runs(g, bg, line_len, lambda p, li=li: cell(li, p))
        if not runs:
            continue

        marker_candidates: dict[int, tuple[list[tuple[int, int]], int]] = {}
        for color, color_runs in runs.items():
            if len(color_runs) != 2:
                continue
            lengths = [end - start + 1 for start, end in color_runs]
            if lengths[0] != lengths[1]:
                continue
            size = lengths[0]
            if size not in size_to_step:
                continue
            edge_runs = [run for run in color_runs if touches_edge(run)]
            if len(edge_runs) != 1:
                continue
            edge_run = edge_runs[0]
            inner_run = color_runs[0] if color_runs[1] == edge_run else color_runs[1]
            gap = (
                inner_run[0] - edge_run[1] - 1
                if edge in ("left", "top")
                else edge_run[0] - inner_run[1] - 1
            )
            if gap != size:
                continue
            marker_candidates[color] = (color_runs, size)

        assert len(marker_candidates) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has malformed grouped markers: {runs}"
        )

        marker, (marker_runs, size) = next(iter(marker_candidates.items()))
        step = int(size_to_step[size])
        edge_run = next(run for run in marker_runs if touches_edge(run))
        anchor = edge_run[0] if edge in ("left", "top") else edge_run[1]
        offset = anchor % step

        companion = None
        companion_positions: list[int] = []
        for color, color_runs in runs.items():
            if color == marker:
                continue
            positions = [
                pos for start, end in color_runs for pos in range(start, end + 1)
            ]
            if any(pos % step == offset for pos in positions):
                assert companion is None, (
                    f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned extras: {runs}"
                )
                companion = color
                companion_positions = positions

        if companion is not None:
            paint = companion
            marker_positions = [
                pos for start, end in marker_runs for pos in range(start, end + 1)
            ]
            all_positions = marker_positions + companion_positions
            lo, hi = min(all_positions), max(all_positions)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

        if companion is not None:
            for start, end in marker_runs:
                for pos in range(start, end + 1):
                    out[cell(li, pos)] = paint

    return out


_ROWS_RANGE = (9, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_SIZE_RANGE = (2, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.30
_P_COMPANION = 0.45

_MAX_RETRIES = 600


def _spaced_indices(n: int, start: int, length: int) -> list[int]:
    available = length - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [start + picks[i] + i for i in range(n)]


def _choose_geometry() -> tuple[int, int, str, int] | None:
    edge = random.choice(_EDGES)
    H = random.randint(*_ROWS_RANGE)
    W = random.randint(*_COLS_RANGE)
    by_row = edge in ("left", "right")
    line_len = W if by_row else H
    if line_len < 9:
        return None

    usable_lines = H - 2 if by_row else W - 2
    max_active = min(_ACTIVE_LINES_RANGE[1], (usable_lines + 1) // 2)
    if max_active < _ACTIVE_LINES_RANGE[0]:
        return None
    n_active = random.randint(_ACTIVE_LINES_RANGE[0], max_active)
    return H, W, edge, n_active


def _choose_size_mapping(edge: str, H: int, W: int) -> dict[int, int] | None:
    by_row = edge in ("left", "right")
    line_len = W if by_row else H
    legend_limit = W if by_row else H
    max_size = min(_SIZE_RANGE[1], line_len // 3)
    available_sizes = list(range(_SIZE_RANGE[0], max_size + 1))
    pairs = [
        (a, b)
        for a, b in combinations(available_sizes, 2)
        if 2 * max(a, b) + 1 <= legend_limit
    ]
    if not pairs:
        return None
    a, b = random.choice(pairs)
    return {a: b, b: a}


def _legend_layout(
    H: int, W: int, edge: str, mapping: dict[int, int], legend_colors: list[int]
) -> dict[str, object]:
    sizes = list(mapping.keys())
    by_row = edge in ("left", "right")

    cells: list[tuple[int, int, int]] = []
    legend_cells: set[tuple[int, int]] = set()

    if by_row:
        top = bool(random.getrandbits(1))
        right = edge == "left"
        size_row = 0 if top else H - 1
        step_row = 1 if top else H - 2
        blocked = {size_row, step_row}
        cursor = W - 1 if right else 0

        for size, color in zip(sizes, legend_colors):
            step = mapping[size]
            width = max(size, step)
            if right:
                start = cursor - width + 1
                for c in range(cursor - size + 1, cursor + 1):
                    cells.append((size_row, c, color))
                    legend_cells.add((size_row, c))
                for c in range(cursor - step + 1, cursor + 1):
                    cells.append((step_row, c, color))
                    legend_cells.add((step_row, c))
                cursor = start - 2
            else:
                start = cursor
                for c in range(start, start + size):
                    cells.append((size_row, c, color))
                    legend_cells.add((size_row, c))
                for c in range(start, start + step):
                    cells.append((step_row, c, color))
                    legend_cells.add((step_row, c))
                cursor = start + width + 1

        return {
            "orientation": "horizontal",
            "blocked_lines": blocked,
            "cells": cells,
            "legend_cells": legend_cells,
        }

    bottom = edge == "top"
    right = bool(random.getrandbits(1))
    size_col = W - 1 if right else 0
    step_col = W - 2 if right else 1
    blocked = {size_col, step_col}
    cursor = H - 1 if bottom else 0

    for size, color in zip(sizes, legend_colors):
        step = mapping[size]
        height = max(size, step)
        if bottom:
            start = cursor - height + 1
            for r in range(cursor - size + 1, cursor + 1):
                cells.append((r, size_col, color))
                legend_cells.add((r, size_col))
            for r in range(cursor - step + 1, cursor + 1):
                cells.append((r, step_col, color))
                legend_cells.add((r, step_col))
            cursor = start - 2
        else:
            start = cursor
            for r in range(start, start + size):
                cells.append((r, size_col, color))
                legend_cells.add((r, size_col))
            for r in range(start, start + step):
                cells.append((r, step_col, color))
                legend_cells.add((r, step_col))
            cursor = start + height + 1

    return {
        "orientation": "vertical",
        "blocked_lines": blocked,
        "cells": cells,
        "legend_cells": legend_cells,
    }


def _active_lines(
    H: int, W: int, edge: str, blocked_lines: set[int], n_active: int
) -> list[int]:
    by_row = edge in ("left", "right")
    if by_row:
        if 0 in blocked_lines:
            return _spaced_indices(n_active, 2, H - 2)
        return _spaced_indices(n_active, 0, H - 2)
    if 0 in blocked_lines:
        return _spaced_indices(n_active, 2, W - 2)
    return _spaced_indices(n_active, 0, W - 2)


def _marker_runs(
    edge: str, size: int, line_len: int
) -> tuple[tuple[int, int], tuple[int, int], int]:
    if edge in ("left", "top"):
        edge_run = (0, size - 1)
        inner_run = (2 * size, 3 * size - 1)
        anchor = edge_run[0]
    else:
        edge_run = (line_len - size, line_len - 1)
        inner_run = (line_len - 3 * size, line_len - 2 * size - 1)
        anchor = edge_run[1]
    return edge_run, inner_run, anchor


def _line_position_options(
    edge: str, size: int, line_len: int, step: int
) -> dict[str, object]:
    edge_run, inner_run, anchor = _marker_runs(edge, size, line_len)
    offset = anchor % step

    marker_positions = [
        pos
        for start, end in (edge_run, inner_run)
        for pos in range(start, end + 1)
    ]
    if edge in ("left", "top"):
        aligned = [p for p in range(inner_run[1] + 1, line_len) if p % step == offset]
    else:
        aligned = [p for p in range(0, inner_run[0]) if p % step == offset]

    blocked = set(marker_positions)
    non_beat = [p for p in range(line_len) if p not in blocked and p % step != offset]
    return {
        "edge_run": edge_run,
        "inner_run": inner_run,
        "anchor": anchor,
        "marker_positions": marker_positions,
        "aligned": aligned,
        "non_beat": non_beat,
    }


def _assign_line_kinds(
    n_active: int, companion_feasible: set[int]
) -> list[str] | None:
    if n_active < 3 or not companion_feasible:
        return None

    kinds: list[str | None] = [None] * n_active
    companion_idx = random.choice(sorted(companion_feasible))
    remaining = [i for i in range(n_active) if i != companion_idx]
    bystander_idx = random.choice(remaining)
    remaining = [i for i in remaining if i != bystander_idx]
    marker_only_idx = random.choice(remaining)
    remaining = [i for i in remaining if i != marker_only_idx]

    kinds[companion_idx] = "companion"
    kinds[bystander_idx] = "bystander"
    kinds[marker_only_idx] = "marker_only"

    counts = {"companion": 1, "bystander": 1, "marker_only": 1}
    for idx in random.sample(remaining, len(remaining)):
        options = ["bystander", "marker_only"]
        if idx in companion_feasible:
            options.append("companion")
        min_count = min(counts[kind] for kind in options)
        candidates = [kind for kind in options if counts[kind] == min_count]
        choice = random.choice(candidates)
        kinds[idx] = choice
        counts[choice] += 1

    return [kind for kind in kinds if kind is not None]


def _build_line_specs(
    edge: str,
    line_len: int,
    marker_color: int,
    size: int,
    step: int,
    extra_palette: list[int],
    forced_kind: str | None = None,
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    options = _line_position_options(edge, size, line_len, step)
    edge_run = options["edge_run"]
    inner_run = options["inner_run"]
    anchor = int(options["anchor"])
    aligned = list(options["aligned"])
    non_beat = list(options["non_beat"])

    cells: list[tuple[int, int]] = []
    marker_positions: list[int] = []
    for start, end in (edge_run, inner_run):
        for pos in range(start, end + 1):
            cells.append((pos, marker_color))
            marker_positions.append(pos)

    other_colors = [c for c in extra_palette if c != marker_color]

    line_kind = forced_kind or "marker_only"
    extra_pos = None
    extra_color = None
    if forced_kind is None:
        roll = random.random()
        if roll < _P_MARKER_ONLY:
            line_kind = "marker_only"
        elif roll < _P_MARKER_ONLY + _P_COMPANION:
            line_kind = "companion"
        else:
            line_kind = "bystander"

    if line_kind == "companion":
        if not aligned or not other_colors:
            return [], {}
        extra_pos = random.choice(aligned)
        extra_color = random.choice(other_colors)
        cells.append((extra_pos, extra_color))
    elif line_kind == "bystander":
        if not non_beat or not other_colors:
            return [], {}
        extra_pos = random.choice(non_beat)
        extra_color = random.choice(other_colors)
        cells.append((extra_pos, extra_color))
    elif line_kind != "marker_only":
        return [], {}

    spec = {
        "marker_color": marker_color,
        "size": size,
        "step": step,
        "anchor": anchor,
        "marker_runs": (edge_run, inner_run),
        "marker_positions": tuple(sorted(marker_positions)),
        "extra_pos": extra_pos,
        "extra_color": extra_color,
        "line_kind": line_kind,
    }
    return cells, spec


def _render_expected(
    inp: np.ndarray,
    edge: str,
    line_specs: list[dict[str, object]],
    *,
    use_lookup: bool,
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
        size = int(spec["size"])
        step = int(spec["step"]) if use_lookup else size
        anchor = int(spec["anchor"])
        offset = anchor % step
        marker_positions = list(spec["marker_positions"])
        extra_pos = spec["extra_pos"]
        extra_color = spec["extra_color"]

        companion = extra_pos is not None and int(extra_pos) % step == offset
        if companion:
            paint = int(extra_color)
            lo = min(marker_positions + [int(extra_pos)])
            hi = max(marker_positions + [int(extra_pos)])
        else:
            paint = marker_color
            lo, hi = 0, line_len - 1

        for pos in range(lo, hi + 1):
            if pos % step != offset:
                continue
            rc = cell(li, pos)
            cur = int(out[rc])
            if cur == 0 or cur == marker_color:
                out[rc] = paint

        if companion:
            for start, end in spec["marker_runs"]:
                for pos in range(start, end + 1):
                    out[cell(li, pos)] = paint

    return out


def _line_diff_count(
    expected: np.ndarray, baseline: np.ndarray, edge: str, active_lines: list[int]
) -> int:
    by_row = edge in ("left", "right")
    changed = 0
    for li in active_lines:
        if by_row:
            if not np.array_equal(expected[li, :], baseline[li, :]):
                changed += 1
        else:
            if not np.array_equal(expected[:, li], baseline[:, li]):
                changed += 1
    return changed


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    geom = _choose_geometry()
    if geom is None:
        return None
    H, W, edge, n_active = geom

    mapping = _choose_size_mapping(edge, H, W)
    if mapping is None:
        return None
    sizes = list(mapping.keys())

    legend_colors = random.sample(_COLORS, len(sizes))
    remaining = [c for c in _COLORS if c not in legend_colors]
    marker_palette_size = random.randint(2, min(3, len(remaining) - 1))
    marker_palette = random.sample(remaining, marker_palette_size)
    extra_palette = [c for c in remaining if c not in marker_palette]
    if not extra_palette:
        return None

    legend = _legend_layout(H, W, edge, mapping, legend_colors)
    active_lines = _active_lines(H, W, edge, set(legend["blocked_lines"]), n_active)
    if len(active_lines) != n_active:
        return None

    by_row = edge in ("left", "right")
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    for r, c, color in legend["cells"]:
        if inp[r, c] != 0:
            return None
        inp[r, c] = int(color)

    line_sizes = [sizes[i % len(sizes)] for i in range(n_active)]
    random.shuffle(line_sizes)
    if len(set(line_sizes)) != len(sizes):
        return None

    companion_feasible = {
        idx
        for idx, size in enumerate(line_sizes)
        if _line_position_options(edge, size, line_len, mapping[size])["aligned"]
    }
    line_kinds = _assign_line_kinds(n_active, companion_feasible)
    if line_kinds is None:
        return None

    shared_color = random.choice(marker_palette)
    line_markers = [random.choice(marker_palette) for _ in range(n_active)]
    idx_a = next(i for i, size in enumerate(line_sizes) if size == sizes[0])
    idx_b = next(i for i, size in enumerate(line_sizes) if size == sizes[1])
    line_markers[idx_a] = shared_color
    line_markers[idx_b] = shared_color

    line_specs: list[dict[str, object]] = []
    line_kind_counts: defaultdict[str, int] = defaultdict(int)
    for li, size, marker_color, forced_kind in zip(
        active_lines, line_sizes, line_markers, line_kinds
    ):
        dots, spec = _build_line_specs(
            edge=edge,
            line_len=line_len,
            marker_color=marker_color,
            size=size,
            step=mapping[size],
            extra_palette=extra_palette,
            forced_kind=forced_kind,
        )
        if not spec:
            return None

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = int(color)

        spec["line"] = li
        line_specs.append(spec)
        line_kind_counts[str(spec["line_kind"])] += 1

    edge_counts = _edge_counts(inp, 0, excluded=set(legend["legend_cells"]))
    winners = [name for name, value in edge_counts.items() if value == max(edge_counts.values())]
    if winners != [edge]:
        return None

    expected = _render_expected(inp, edge, line_specs, use_lookup=True)
    baseline = _render_expected(inp, edge, line_specs, use_lookup=False)
    try:
        actual = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    size_color_sets: defaultdict[int, set[int]] = defaultdict(set)
    for spec in line_specs:
        size_color_sets[int(spec["size"])].add(int(spec["marker_color"]))
    shared_cross_size = int(
        any(
            color in size_color_sets[sizes[0]] and color in size_color_sets[sizes[1]]
            for color in marker_palette
        )
    )

    metrics = {
        "active_lines": n_active,
        "legend_dependent_lines": sum(
            1 for spec in line_specs if int(spec["size"]) != int(spec["step"])
        ),
        "changed_lines_vs_direct_size": _line_diff_count(
            expected, baseline, edge, active_lines
        ),
        "companion_lines": line_kind_counts["companion"],
        "bystander_lines": line_kind_counts["bystander"],
        "marker_only_lines": line_kind_counts["marker_only"],
        "shared_color_cross_size": shared_cross_size,
    }

    if metrics["legend_dependent_lines"] != n_active:
        return None
    if metrics["changed_lines_vs_direct_size"] == 0:
        return None
    if metrics["shared_color_cross_size"] == 0:
        return None

    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")


def variant_metrics(samples: int = 32) -> dict[str, float]:
    totals = defaultdict(int)
    for _ in range(samples):
        result = None
        while result is None:
            result = _gen_attempt()
        _, metrics = result
        for key, value in metrics.items():
            totals[key] += value

    return {
        "avg_active_lines": totals["active_lines"] / samples,
        "avg_legend_dependent_lines": totals["legend_dependent_lines"] / samples,
        "avg_changed_lines_vs_direct_size": totals["changed_lines_vs_direct_size"] / samples,
        "avg_companion_lines": totals["companion_lines"] / samples,
        "avg_bystander_lines": totals["bystander_lines"] / samples,
        "avg_marker_only_lines": totals["marker_only_lines"] / samples,
        "shared_color_cross_size_rate": totals["shared_color_cross_size"] / samples,
    }
