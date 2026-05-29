# Variant of 16de56c4
# Dimension: D4 — 源对象结构 (src)
# Change: 每行恰好2个同色标记点（一个在射击边，一个在内部），定义步长和偏移... → 色块分组

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D4 variant
# - Reference check: the workspace does not contain a solver file for source task
#   `b5ca7ac4`; only its ARC json is present, so the implementation uses only the
#   stated "color_grouping" idea: replace informative single pixels with small
#   connected same-color groups that must be decoded as one object.
# - Concrete object change: on each active shooting line, the repeating marker is
#   now two equal-length 1D bars of one color, not two isolated pixels. The bar
#   length itself is the step. One bar touches the shooting edge; the second bar
#   lies further inside the grid. The gap between the two bars is also one step,
#   so humans can read "bar width = empty gap = period cue" directly.
# - Behaviour trace on a left-edge row with step 3:
#     original input marker: positions {0, 3}
#     new input marker bars: [0..2] and [6..8]
#     inferred anchor/beat: position 0 modulo 3
#     output fill: beat cells 0,3,6,9,...; the extra pixels inside the bars are
#     just thickened tags. If a companion sits on the same beat, those marker bars
#     are recolored to the companion color, mirroring the old "marker dots get
#     absorbed" behaviour.
# - Complexity calibration: bars stay 1-cell thick and only extend along the
#   shooting direction, so linewise reasoning still works. Steps are limited to
#   2-5 when possible to keep the variant visually distinct from the original
#   dot-based task; step 1 is avoided in generation.
# - Hidden-assumption audit against the original solver:
#   1. Counting raw positions per color no longer works because a bar contributes
#      several pixels. The solver must read contiguous runs, not individual cells.
#   2. The old `len(mpos) == 2` assumption is replaced by "exactly two runs of the
#      same length, exactly one touching the shooting edge, and the inter-run gap
#      equals the run length".
#   3. Recoloring only beat cells is no longer enough when a companion absorbs the
#      marker: non-anchor pixels inside the marker bars must be recolored too.
#   4. Companion/bystander detection can still operate on positions modulo step,
#      but the generator must ensure at most one aligned non-marker color per line.
# - Placement legality: bars are fully contained in one row/column, never overlap,
#   and use `line_len >= 3 * step` so the second bar fits with a one-step gap.
#   Companion pixels are placed strictly beyond the inner bar along the shooting
#   direction; bystanders are placed off-beat and outside both bars.
# - Visual observability / cleanliness: each active line contains one dominant
#   grouped marker color and at most one extra single pixel. The bars are isolated
#   from neighbouring active lines by keeping active line indices separated. The
#   shooting edge remains unique because only the edge-touching bar contributes one
#   boundary pixel per active line, and generation rejects ambiguous boundary
#   counts.


def _scan_runs(g: np.ndarray, bg: int, line_len: int, cell) -> dict[int, list[tuple[int, int]]]:
    """Return contiguous non-bg runs on one shooting-direction line."""
    runs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    p = 0
    while p < line_len:
        v = int(g[cell(p)])
        if v == bg:
            p += 1
            continue
        start = p
        while p + 1 < line_len and int(g[cell(p + 1)]) == v:
            p += 1
        runs[v].append((start, p))
        p += 1
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend periodic beats launched from one dominant edge, with grouped markers.

    Input structure:
    - bg is the most common color. Non-bg cells appear only on a sparse set of
      shooting-direction lines.
    - One grid edge has strictly more non-bg boundary cells than the others; that
      edge defines the shooting direction (left/right -> rows, top/bottom -> cols).
    - Each active line contains exactly one repeating marker color, represented by
      two equal connected bars on that line. One bar touches the shooting edge,
      the other sits inside the grid, and the bar length equals the step.
    - A different-colored extra pixel on the same line is either an aligned
      companion (same beat class) or an off-beat bystander.

    Rule:
    - Find the unique shooting edge by boundary counts.
    - On each active line, detect the unique marker color by its two equal runs:
      one touches the shooting edge, and the gap between the runs equals their
      length. That run length is the step.
    - Use the edge-nearest end of the edge-touching run as the beat anchor. A
      different-colored aligned pixel absorbs the ray, becomes the paint color,
      and limits filling to the span between marker and companion. An unaligned
      pixel is a bystander and stays unchanged.
    - Without a companion, paint all beat cells across the whole line with the
      marker color. With a companion, also recolor both marker bars to the
      companion color.

    Key insight:
    - The grouped marker bars are not themselves the periodic fill; they encode
      the period via their size. Reading contiguous same-color runs reveals the
      step, while the edge-touching run supplies the phase, and any aligned extra
      pixel still acts as the wall that changes color and truncates the ray.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg cells on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def touches_edge(run: tuple[int, int]) -> bool:
        start, end = run
        return start == 0 if edge in ("left", "top") else end == line_len - 1

    for li in range(num_lines):
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
            step = lengths[0]
            edge_runs = [run for run in color_runs if touches_edge(run)]
            if len(edge_runs) != 1:
                continue
            edge_run = edge_runs[0]
            inner_run = color_runs[0] if color_runs[1] == edge_run else color_runs[1]
            gap = inner_run[0] - edge_run[1] - 1 if edge in ("left", "top") else edge_run[0] - inner_run[1] - 1
            if gap != step:
                continue
            marker_candidates[color] = (color_runs, step)

        assert len(marker_candidates) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has malformed grouped markers: {runs}"
        )
        marker, (marker_runs, step) = next(iter(marker_candidates.items()))
        edge_run = next(run for run in marker_runs if touches_edge(run))
        anchor = edge_run[0] if edge in ("left", "top") else edge_run[1]
        offset = anchor % step

        companion = None
        companion_positions: list[int] = []
        for color, color_runs in runs.items():
            if color == marker:
                continue
            positions = [p for start, end in color_runs for p in range(start, end + 1)]
            if any(p % step == offset for p in positions):
                assert companion is None, (
                    f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned companions: {runs}"
                )
                companion = color
                companion_positions = positions

        if companion is not None:
            paint = companion
            marker_positions = [p for start, end in marker_runs for p in range(start, end + 1)]
            all_ps = marker_positions + companion_positions
            lo, hi = min(all_ps), max(all_ps)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step != offset:
                continue
            rc = cell(li, p)
            cur = int(out[rc])
            if cur == bg or cur == marker:
                out[rc] = paint

        if companion is not None:
            for start, end in marker_runs:
                for p in range(start, end + 1):
                    out[cell(li, p)] = paint

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (2, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_P_MARKER_ONLY = 0.35
_P_COMPANION = 0.35

_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int):
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
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


def _marker_runs(edge: str, step: int, line_len: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    if edge in ("left", "top"):
        edge_run = (0, step - 1)
        inner_run = (2 * step, 3 * step - 1)
        anchor = edge_run[0]
    else:
        edge_run = (line_len - step, line_len - 1)
        inner_run = (line_len - 3 * step, line_len - 2 * step - 1)
        anchor = edge_run[1]
    return edge_run, inner_run, anchor


def _build_line_specs(edge: str, line_len: int, marker_color: int, step: int):
    edge_run, inner_run, anchor = _marker_runs(edge, step, line_len)
    offset = anchor % step

    marker_cells = []
    for start, end in (edge_run, inner_run):
        for pos in range(start, end + 1):
            marker_cells.append((pos, marker_color))

    if edge in ("left", "top"):
        aligned_companions = [p for p in range(inner_run[1] + 1, line_len) if p % step == offset]
    else:
        aligned_companions = [p for p in range(0, inner_run[0]) if p % step == offset]

    blocked = {pos for pos, _ in marker_cells}
    non_beat = [p for p in range(line_len) if p not in blocked and p % step != offset]
    other_colors = [c for c in _COLORS if c != marker_color]

    line_kind = "marker"
    extra = None
    roll = random.random()
    if roll < _P_MARKER_ONLY:
        line_kind = "marker"
    elif roll < _P_MARKER_ONLY + _P_COMPANION and aligned_companions and other_colors:
        line_kind = "companion"
        extra = (random.choice(aligned_companions), random.choice(other_colors))
    elif non_beat and other_colors:
        line_kind = "bystander"
        extra = (random.choice(non_beat), random.choice(other_colors))

    if extra is not None:
        marker_cells.append(extra)
    return marker_cells, line_kind, step


def _count_other_edge_dots(dots_rc: list[tuple[int, int]], edge: str, H: int, W: int) -> dict[str, int]:
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts = {}
    for other_edge in _EDGES:
        if other_edge == edge:
            continue
        check = boundary[other_edge]
        counts[other_edge] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    max_step = min(_STEP_RANGE[1], line_len // 3)
    if max_step < _STEP_RANGE[0]:
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    line_kinds: defaultdict[str, int] = defaultdict(int)
    steps_used: list[int] = []

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        step = random.randint(_STEP_RANGE[0], max_step)
        dots, line_kind, _ = _build_line_specs(edge, line_len, marker_color, step)
        line_kinds[line_kind] += 1
        steps_used.append(step)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    metrics = {
        "step_gt1_lines": sum(1 for step in steps_used if step > 1),
        "companion_lines": line_kinds["companion"],
        "bystander_lines": line_kinds["bystander"],
        "marker_only_lines": line_kinds["marker"],
        "max_step": max(steps_used),
    }
    if metrics["step_gt1_lines"] < max(2, n_active // 2):
        return None

    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")


def variant_metrics(samples: int = 32) -> dict[str, float]:
    """Quantify whether generation is exercising the grouped-marker variant."""
    totals = defaultdict(int)
    for _ in range(samples):
        result = None
        while result is None:
            result = _gen_attempt()
        _, metrics = result
        for key, value in metrics.items():
            totals[key] += value

    return {
        "avg_step_gt1_lines": totals["step_gt1_lines"] / samples,
        "avg_companion_lines": totals["companion_lines"] / samples,
        "avg_bystander_lines": totals["bystander_lines"] / samples,
        "avg_marker_only_lines": totals["marker_only_lines"] / samples,
        "avg_max_step": totals["max_step"] / samples,
    }
