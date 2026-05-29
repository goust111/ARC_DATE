# Variant of 16de56c4
# Dimension: D15 — 伴随点数量 (meta)
# Change: 每行0或1个伴随点... → 多伴随点

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(_ROOT / "solvers"))
from arc_utils import get_bg


# Design analysis for the D15 variant
# -----------------------------------
# a) Concrete behavior:
# - The original line logic had one repeating marker color (exactly 2 dots) and
#   at most one aligned companion dot, so a line either filled fully with the
#   marker color or formed one bounded segment in a companion color.
# - This variant keeps the same shooting-edge detection and the same marker
#   semantics, but changes the aligned-dot case: a line may have 2 or 3 aligned
#   companion dots of different colors, all beyond the inner marker.  Ordered
#   along the ray, they split the beat positions into consecutive segments:
#   first segment = from the shooting-edge marker to the first companion,
#   later segments = from just after the previous companion to the current one.
# - Example on a left-shooting row with step 2:
#   input dots at positions 0/2 (marker color 4), companions at 6/10/14 with
#   colors 7/3/8.  Old logic would stop at the first aligned companion and
#   paint beats 0,2,4,6 with color 7.  New logic paints:
#   0,2,4,6 -> 7; 8,10 -> 3; 12,14 -> 8.
#   The difference is visible as multiple color bands on the same dotted ray.
# b) Complexity calibration:
# - Companion count is restricted to 2 or 3 when companions exist, not an
#   arbitrary number, so the pattern is richer than the original but still
#   learnable from a few examples.
# - Marker-only lines are still allowed to preserve the original contrast
#   between unbounded and bounded rays, but each generated grid is forced to
#   contain multiple multi-companion lines so the variant is actually observed.
# c) Hidden assumptions reviewed against the old solver:
# - The old `for oc, ops ... break` relied on "at most one aligned companion".
#   That would silently discard later companions in this variant, so it must be
#   replaced with ordered collection of all aligned companions.
# - The old bounded fill used `lo, hi = min/max(marker+companion)` which was
#   harmless for a single segment.  With multiple companions that collapses all
#   intervals into one span and destroys the segment colors, so segment-wise
#   filling is required.
# - The old write guard `cur == bg or cur == marker` remains valid because each
#   segment starts after the previous companion, so companions never need to be
#   repainted by later segments.
# d) Placement legality:
# - Every active line still has exactly one repeating color with exactly two
#   dots, one on the shooting edge and one interior.
# - Every companion is on the same beat class as the marker and lies strictly
#   beyond the inner marker in the shooting direction.
# - Companion colors are distinct within a line and all non-marker colors are
#   singletons, preventing accidental second repeating colors.
# - Optional bystanders are placed off-beat and off occupied cells, so they stay
#   untouched and do not interfere with segment detection.
# e) Visual observability:
# - The new rule is visible because one dotted ray now changes color multiple
#   times along the same periodic scaffold.  This can be inferred from 2-4
#   examples without relying on color ids or hidden ordering rules.
# f) Visual cleanliness:
# - Only one sparse beat scaffold exists per active line; segments lie on the
#   same line rather than stacking multiple structures.  Active lines remain
#   spaced apart and bystanders are sparse, which keeps the grids legible.
# g) Color-change inferability:
# - Companion colors are read directly from the input dots that terminate each
#   segment.  The color mapping is local and structural: "segment ending at this
#   aligned dot uses that dot's color".
# h) Generator anti-degeneration policy:
# - Generated puzzles require at least two multi-companion lines, and companion
#   counts are only 2 or 3.  That prevents the dataset from collapsing back to
#   the original single-companion regime.
# i) New solver assumptions audited:
# - `main()` assumes each non-empty line has exactly one repeating marker color.
#   The generator enforces this by giving every non-marker color only one dot.
# - `main()` assumes aligned companions, if present, are beyond the inner marker
#   and already ordered by physical position after sorting by shooting direction.
#   The generator constructs them directly from beat indices >= 2.


def _edge_info(edge: str, line_len: int) -> tuple[bool, int]:
    forward = edge in ("left", "top")
    edge_pos = 0 if forward else line_len - 1
    return forward, edge_pos


def _segment_positions(prev_anchor: int, target: int, step: int, forward: bool, first: bool) -> range:
    if forward:
        start = prev_anchor if first else prev_anchor + step
        return range(start, target + 1, step)
    start = prev_anchor if first else prev_anchor - step
    return range(start, target - 1, -step)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic rays emitted from a unique grid edge, now with segmented companion colors.

    Input structure:
    - Background is the most common color; non-background cells are sparse dots.
    - One boundary has strictly more non-background dots than the other three;
      that boundary is the shooting edge, so left/right imply row-wise rays and
      top/bottom imply column-wise rays.
    - Every active shooting line contains exactly one marker color with exactly
      two dots: one on the shooting edge and one interior.  Their gap is the
      step of the periodic beat.
    - Additional single dots on that line may be aligned companions (same beat
      class as the marker) or unaligned bystanders.  In this variant there can
      be 2 or 3 aligned companions of different colors on one line.

    Rule:
    - Detect the unique shooting edge, then process each shooting-direction line
      independently.
    - The marker pair defines the beat positions.  If a line has no aligned
      companions, fill the entire beat sequence with the marker color.
    - If aligned companions exist, sort them from the shooting edge outward.
      Each companion ends one segment and donates its own color to that segment:
      from the edge marker to the first companion, then from just beyond each
      previous companion to the next one.
    - Any unaligned dot is a bystander and remains unchanged.

    Key insight: the old task had one absorbing companion, but here every
    aligned companion acts as a successive boundary.  The marker pair still
    reveals the period; the ordered aligned dots reveal how the ray is chopped
    into multiple color-coded intervals.
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
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    forward, edge_pos = _edge_info(edge, line_len)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

        if not colors:
            continue

        repeating = {color: ps for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one "
            f"repeating marker color: {dict(colors)}"
        )

        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} has "
            f"{len(mpos)} dots (expected 2): {mpos}"
        )
        assert edge_pos in mpos, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} must "
            f"touch the shooting edge: {mpos}"
        )
        a, b = sorted(mpos)
        step = b - a
        assert step > 0, f"Line {li} has coincident marker dots: {mpos}"
        offset = edge_pos % step

        aligned_companions: list[tuple[int, int]] = []
        for color, positions in colors.items():
            if color == marker:
                continue
            assert len(positions) == 1, (
                f"Line {li} ({'row' if by_row else 'col'}) non-marker color "
                f"{color} appears multiple times: {positions}"
            )
            pos = positions[0]
            if pos % step == offset:
                aligned_companions.append((pos, color))

        aligned_companions.sort(key=lambda item: item[0], reverse=not forward)
        if aligned_companions:
            assert 2 <= len(aligned_companions) <= 3, (
                f"Line {li} has {len(aligned_companions)} aligned companions "
                f"(expected 2 or 3): {aligned_companions}"
            )
            prev_anchor = edge_pos
            for idx, (target_pos, paint) in enumerate(aligned_companions):
                for p in _segment_positions(prev_anchor, target_pos, step, forward, idx == 0):
                    rc = cell(li, p)
                    cur = int(out[rc])
                    if cur == bg or cur == marker:
                        out[rc] = paint
                prev_anchor = target_pos
        else:
            full_range = range(edge_pos, line_len, step) if forward else range(edge_pos, -1, -step)
            for p in full_range:
                rc = cell(li, p)
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = marker

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 4)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_MULTI_COMPANION_COUNTS = (2, 3)
_PALETTE_SIZE_RANGE = (5, 7)
_P_MARKER_ONLY = 0.20
_P_BYSTANDER = 0.25

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


def _pick_step(line_len: int, n_companions: int) -> int | None:
    min_required = max(1, n_companions + 1)
    candidates = [
        step
        for step in range(_STEP_RANGE[0], _STEP_RANGE[1] + 1)
        if step <= line_len - 1 and (line_len - 1) >= min_required * step
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _beat_positions(edge: str, line_len: int, step: int) -> list[int]:
    forward, edge_pos = _edge_info(edge, line_len)
    if forward:
        return list(range(edge_pos, line_len, step))
    return list(range(edge_pos, -1, -step))


def _available_bystander_positions(edge: str, line_len: int, step: int, occupied: set[int]) -> list[int]:
    forward, edge_pos = _edge_info(edge, line_len)
    beat_set = set(_beat_positions(edge, line_len, step))
    candidates = []
    for p in range(line_len):
        if p in occupied or p in beat_set:
            continue
        if p in (0, line_len - 1) and p != edge_pos:
            continue
        candidates.append(p)
    return candidates


def _choose_companion_positions(
    companion_pool: list[int],
    n_companions: int,
    step: int,
    forward: bool,
) -> list[int] | None:
    if len(companion_pool) < n_companions:
        return None
    valid = [
        combo
        for combo in combinations(companion_pool, n_companions)
        if any(abs(b - a) > step for a, b in zip(combo, combo[1:]))
    ]
    if valid:
        ordered = list(random.choice(valid))
        if not forward:
            ordered.reverse()
        return ordered
    return None


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    force_multi: bool,
) -> list[tuple[int, int]] | None:
    if force_multi:
        n_companions = random.choice(_MULTI_COMPANION_COUNTS)
    elif random.random() < _P_MARKER_ONLY:
        n_companions = 0
    else:
        n_companions = random.choice(_MULTI_COMPANION_COUNTS)

    step = _pick_step(line_len, n_companions)
    if step is None:
        return None

    beats = _beat_positions(edge, line_len, step)
    forward, edge_pos = _edge_info(edge, line_len)
    inner_pos = beats[1]
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    occupied = {edge_pos, inner_pos}
    if n_companions:
        companion_pool = beats[2:]
        assert len(companion_pool) >= n_companions
        far_edge = line_len - 1 if forward else 0
        if far_edge in companion_pool and len(companion_pool) > n_companions:
            trimmed = [p for p in companion_pool if p != far_edge]
            if len(trimmed) >= n_companions:
                companion_pool = trimmed
        chosen_positions = _choose_companion_positions(companion_pool, n_companions, step, forward)
        if chosen_positions is None:
            return None
        companion_colors = random.sample([c for c in palette if c != marker_color], n_companions)
        for pos, color in zip(chosen_positions, companion_colors):
            dots.append((pos, color))
            occupied.add(pos)

    if random.random() < _P_BYSTANDER:
        bystander_candidates = [
            c for c in palette if c != marker_color and all(color != c for _, color in dots)
        ]
        free_positions = _available_bystander_positions(edge, line_len, step, occupied)
        if bystander_candidates and free_positions:
            dots.append((random.choice(free_positions), random.choice(bystander_candidates)))

    return dots


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    min_multi = max(2, n_active // 2)
    multi_count = random.randint(min_multi, n_active)
    multi_lines = set(random.sample(active_lines, multi_count))

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_COLORS, palette_size)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    actual_multi_lines = 0

    for li in active_lines:
        marker_color = random.choice(palette)
        dots = _build_line_dots(edge, line_len, marker_color, palette, li in multi_lines)
        if dots is None:
            return None

        aligned_count = 0
        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        # Count aligned companions to verify the anti-degeneration constraint.
        marker_positions = sorted(pos for pos, color in dots if color == marker_color)
        step = marker_positions[1] - marker_positions[0]
        offset = marker_positions[0] % step
        aligned_count = sum(
            1
            for pos, color in dots
            if color != marker_color and pos % step == offset
        )
        if aligned_count >= 2:
            actual_multi_lines += 1

        shooting_edge_dots += 1

    if actual_multi_lines < 2:
        return None

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
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
