# Variant of 16de56c4
# Dimension: D3 — 轨迹填充模式 (trajectory)
# Change: 等间距连续填充：在beat位置(pos % step == offset)全部填充相同颜色... → 交替颜色填充

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D3 variant
# ----------------------------------
# spec:
#   preserved_structure:
#     - One shooting edge is still identified by having the strict boundary-dot maximum.
#     - Left/right edges still fire along rows; top/bottom edges still fire along columns.
#     - Each active line still has exactly one repeated marker color with exactly two dots.
#     - A same-beat companion still limits the fill span; an off-beat bystander still stays unchanged.
#   variant_core:
#     - The fill on a companion line is no longer monochrome.
#     - Beat positions inside the companion-bounded span alternate marker/companion colors.
#     - Phase is anchored at the shooting-edge marker: beat index 0 uses the marker color,
#       beat index 1 uses the companion color, then ABAB onward.
#   concrete_trace:
#     - Example line (left edge): marker A at positions 0 and 2, companion B at position 6.
#     - Old rule: fill positions 0,2,4,6 all as B.
#     - New rule: fill positions 0,2,4,6 as A,B,A,B.
#     - The changed behavior is local to the trajectory painter; edge detection, step inference,
#       companion detection, and bystander preservation stay the same.
#   complexity_guardrails:
#     - Marker-only lines are still allowed so the task does not become a different puzzle,
#       but the generator forces at least two companion lines so the new pattern is visible.
#     - Companion positions are generated on odd beat indices (3,5,7,...) from the shooting edge.
#       This keeps the companion dot's input color stable in the output and makes the ABAB rule
#       legible from a few examples.
#     - A small per-grid palette is used to avoid noisy color clutter while keeping repeated evidence.
#   implicit_assumptions_checked:
#     - The original write condition only overwrote bg/marker cells. That was safe for monochrome
#       companion fills, but it would fail here because some companion cells may need to stay or be
#       repainted according to parity. The new code overwrites companion cells too, while still
#       protecting unaligned bystanders.
#     - Alternation needs an oriented origin. Sorting the marker positions is not enough because
#       right/bottom-fired lines run in the reverse direction. The solver now anchors parity at the
#       marker that lies on the shooting edge.
#   generator_legality:
#     - Dots are placed on distinct rows/columns, so active lines do not overlap.
#     - Companion lines only use steps that admit an odd beat index >= 3 inside the line bounds.
#     - Bystanders are sampled only from off-beat cells, so they cannot be mistaken for companions.
#     - Every candidate is validated by the solver before being returned.


def _shooting_edge(g: np.ndarray, bg: int) -> str:
    H, W = g.shape
    edge_counts = {
        'top': sum(1 for c in range(W) if g[0, c] != bg),
        'bottom': sum(1 for c in range(W) if g[H - 1, c] != bg),
        'left': sum(1 for r in range(H) if g[r, 0] != bg),
        'right': sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [e for e, c in edge_counts.items() if c == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    return winners[0]


def _line_axes(edge: str, shape: tuple[int, int]):
    H, W = shape
    by_row = edge in ('left', 'right')
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    return by_row, num_lines, line_len, cell


def _iter_line_colors(g: np.ndarray, bg: int, line: int, line_len: int, cell):
    colors: dict[int, list[int]] = defaultdict(list)
    for p in range(line_len):
        v = int(g[cell(line, p)])
        if v != bg:
            colors[v].append(p)
    return colors


def _paint_for_beat(marker: int, companion: int, edge_anchor: int, step: int, pos: int) -> int:
    beat_idx = abs(pos - edge_anchor) // step
    return marker if beat_idx % 2 == 0 else companion


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire periodic rays from a unique grid edge and stripe companion-bounded beats.

    Input structure:
    - The background is the most common color and all informative pixels are sparse dots.
    - One boundary (top, bottom, left, or right) uniquely has the most non-background dots;
      that boundary is the shooting edge and defines whether lines are rows or columns.
    - On each active shooting-direction line, exactly one color repeats twice; those two dots
      fix the beat spacing. A different-colored aligned dot on the same beat progression is a
      companion, while an unaligned extra dot is only a bystander.

    Rule:
    - Find the shooting edge, then inspect each row/column in that firing direction separately.
    - The repeated color's two positions define `step`; the boundary-side marker also fixes the
      alternation origin.
    - Without a companion, fill every beat position on the whole line with the marker color.
    - With a companion, fill only the span covered by the marker pair and the companion, but
      alternate colors on successive beat positions: marker color, companion color, marker color,
      companion color, and so on from the shooting edge inward.
    - Off-beat bystanders are preserved.

    Key insight: the geometry is still decoded exactly as in the base task, but a companion no
    longer turns the whole ray into one solid color; instead it supplies the second stripe color,
    and the firing edge determines the ABAB phase.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row, num_lines, line_len, cell = _line_axes(edge, g.shape)

    for li in range(num_lines):
        colors = _iter_line_colors(g, bg, li, line_len, cell)
        if not colors:
            continue

        repeating = {c: ps for c, ps in colors.items() if len(ps) >= 2}
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        for marker, mpos in repeating.items():
            assert len(mpos) == 2, (
                f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
                f"has {len(mpos)} dots (expected 2): {mpos}"
            )
            edge_anchor = min(mpos) if edge in ('left', 'top') else max(mpos)
            inner_marker = max(mpos) if edge in ('left', 'top') else min(mpos)
            step = abs(inner_marker - edge_anchor)
            assert step > 0, f"Line {li} ({'row' if by_row else 'col'}) has zero step"
            offset = edge_anchor % step

            companion = None
            for oc, ops in colors.items():
                if oc == marker:
                    continue
                if any(p % step == offset for p in ops):
                    companion = oc
                    break

            if companion is None:
                lo, hi = 0, line_len - 1
                overwrite = {bg, marker}
                for p in range(lo, hi + 1):
                    if p % step != offset:
                        continue
                    rc = cell(li, p)
                    if int(out[rc]) in overwrite:
                        out[rc] = marker
                continue

            all_ps = mpos + colors[companion]
            lo, hi = min(all_ps), max(all_ps)
            overwrite = {bg, marker, companion}
            for p in range(lo, hi + 1):
                if p % step != offset:
                    continue
                rc = cell(li, p)
                if int(out[rc]) not in overwrite:
                    continue
                out[rc] = _paint_for_beat(marker, companion, edge_anchor, step, p)

    return out


# --------------- generator (D3 alternating-trajectory variant) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ('top', 'bottom', 'left', 'right')

_P_MARKER_ONLY = 0.20
_P_COMPANION = 0.60

_MIN_COMPANION_LINES = 2
_PALETTE_SIZE_RANGE = (3, 6)
_MAX_RETRIES = 400


def _choose_dims(edge: str, n_active: int):
    min_num_lines = 2 * n_active - 1
    by_row = edge in ('left', 'right')
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


def _choose_line_types(n_active: int) -> list[str]:
    while True:
        kinds = []
        for _ in range(n_active):
            roll = random.random()
            if roll < _P_MARKER_ONLY:
                kinds.append('marker_only')
            elif roll < _P_MARKER_ONLY + _P_COMPANION:
                kinds.append('companion')
            else:
                kinds.append('bystander')
        if kinds.count('companion') >= _MIN_COMPANION_LINES:
            return kinds


def _valid_steps(edge: str, line_len: int, kind: str) -> list[int]:
    del edge
    upper = min(_STEP_RANGE[1], line_len - 1)
    steps = list(range(_STEP_RANGE[0], upper + 1))
    if kind == 'companion':
        return [s for s in steps if (line_len - 1) // s >= 3]
    if kind == 'bystander':
        valid = []
        edge_pos = 0
        for s in steps:
            offset = edge_pos % s
            if any(p % s != offset for p in range(line_len) if p not in (edge_pos, edge_pos + s)):
                valid.append(s)
        return valid
    return steps


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    step: int,
    kind: str,
):
    if edge in ('left', 'top'):
        edge_pos = 0
        direction = 1
        inner_pos = step
    else:
        edge_pos = line_len - 1
        direction = -1
        inner_pos = line_len - 1 - step
    offset = edge_pos % step

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    meta = {'kind': kind, 'companion_index': None}

    other_colors = [c for c in palette if c != marker_color]
    if not other_colors:
        other_colors = [c for c in _ALL_COLORS if c != marker_color]

    if kind == 'companion':
        max_beats = (line_len - 1) // step
        odd_indices = list(range(3, max_beats + 1, 2))
        if not odd_indices or not other_colors:
            return None
        beat_idx = random.choice(odd_indices)
        pos = edge_pos + direction * beat_idx * step
        dots.append((pos, random.choice(other_colors)))
        meta['companion_index'] = beat_idx
        return dots, meta

    if kind == 'bystander':
        non_beat = [
            p
            for p in range(line_len)
            if p % step != offset and p not in (edge_pos, inner_pos)
        ]
        if not non_beat or not other_colors:
            return None
        dots.append((random.choice(non_beat), random.choice(other_colors)))
        return dots, meta

    return dots, meta


def _count_other_edge_dots(dots_rc: list[tuple[int, int]], edge: str, H: int, W: int) -> dict[str, int]:
    boundary = {
        'top': lambda r, c: r == 0,
        'bottom': lambda r, c: r == H - 1,
        'left': lambda r, c: c == 0,
        'right': lambda r, c: c == W - 1,
    }
    counts = {}
    for oe in _EDGES:
        if oe == edge:
            continue
        check = boundary[oe]
        counts[oe] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _alternating_visibility_score(meta: list[dict]) -> int:
    score = 0
    for item in meta:
        if item['kind'] != 'companion':
            continue
        beat_idx = item['companion_index']
        score += 0 if beat_idx is None else beat_idx + 1
    return score


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_ALL_COLORS, palette_size)
    line_types = _choose_line_types(n_active)

    by_row = edge in ('left', 'right')

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    meta: list[dict] = []

    for li, kind in zip(active_lines, line_types):
        valid_steps = _valid_steps(edge, line_len, kind)
        if not valid_steps:
            return None

        marker_color = random.choice(palette)
        step = random.choice(valid_steps)
        built = _build_line_dots(edge, line_len, marker_color, palette, step, kind)
        if built is None:
            return None
        dots, line_meta = built
        meta.append(line_meta)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(cnt >= shooting_edge_dots for cnt in other_counts.values()):
        return None

    if _alternating_visibility_score(meta) < 8:
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
