# Variant of 16de56c4
# Dimension: D10 — 伴随点与旁观者的判定 (meta)
# Change: 同一行上与标记点同模(pos % step == offset)的异色点为伴随点；不同模的异色点为旁观者... → 全部为伴随

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D10 variant
# -----------------------------------
# spec:
#   preserved_core:
#     grid: 7-12 rows x 9-21 cols
#     background: 0 is the dominant color
#     shooting_edge: exactly one boundary has the largest non-bg count
#     direction: left/right -> rows, top/bottom -> columns
#     active_lines: 4-9 non-empty shooting lines, spaced by at least one blank line
#     marker: each active line still has exactly one repeating color with exactly 2 dots;
#       one sits on the shooting edge and the gap to the inner copy defines the step 1-5
#   d10_change:
#     old_behavior: only same-beat foreign dots were companions; off-beat foreign dots were ignored
#     new_behavior: every foreign dot on the line is a companion
#     segmentation_rule:
#       - normalize positions by distance from the shooting edge
#       - sort all companion dots by this distance
#       - segment 1 spans from the edge marker to companion 1
#       - segment k spans from companion k-1 to companion k
#       - each segment paints only beat positions of the marker rhythm
#       - segment k uses the color of companion k
#       - after the last companion, the ray stops; with no companions, the marker fills the whole line
#   concrete_trace:
#     example_input_line: edge marker M at t=0, inner marker M at t=3, companions A at t=5 and B at t=8
#     original: A and B are both off-beat, so they are both bystanders; the whole line fills with M
#     variant: beats t=0,3 are painted A because they lie in (start,A]; beat t=6 is painted B because it lies in (A,B]
#     visible_difference: a single line now contains multiple colored beat segments instead of one global fill
#   complexity_calibration:
#     - multiple companions are allowed but capped at 3 so the rule stays visible in 2-4 examples
#     - companions keep distinct colors within a line to avoid accidental second repeating colors
#     - the palette per grid is capped at 4-8 colors so color changes remain learnable
#   implicit_assumption_audit:
#     - original code assumed at most one relevant foreign color and could stop at the first aligned match
#     - original fill range used one [min,max] span; this breaks once different companions should own different subranges
#     - original "paint if bg or marker" remains safe: segment colors should not overwrite previously placed companions
#     - solver now assumes the only repeated color on a line is the marker; generator enforces that
#   placement_legality:
#     - companions are placed strictly beyond the inner marker and never on the opposite edge
#     - each companion uses a unique position and unique non-marker color on its line
#     - active lines are separated and boundary counts are checked so the shooting edge stays unambiguous
#   visual_observability:
#     - generator rejects samples that never exercise the variant (no multi-companion line and no off-beat companion)
#     - generator also rejects samples whose output would match the legacy single-companion rule on the same input
#     - many lines intentionally contain 2-3 companions, producing obvious multi-color segmented rays
#   neatness:
#     - all interaction stays on one shooting line, so there is no cross-line overlap
#     - opposite-edge placements are avoided to reduce boundary clutter


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint periodic rays whose foreign dots all become segment endpoints.

    Input structure:
    - The background is the most common color, with sparse non-background dots.
    - One boundary has strictly more non-background dots than the other three,
      which identifies the shooting edge and therefore whether rows or columns
      are the active shooting lines.
    - Every active line contains exactly one repeated marker color with two dots:
      one on the shooting edge and one deeper inside the line, and their gap is
      the period of the ray.
    - Any other dot on that same line is a companion, even if it is off the
      marker beat.

    Rule:
    - Detect the unique shooting edge; left/right edges mean row-wise rays,
      top/bottom edges mean column-wise rays.
    - On each active line, read the marker step from the two repeated dots and
      convert positions into distance from the shooting edge.
    - Sort every different-colored dot on that line by its distance from the
      edge.  If there are no companions, the marker color fills every beat
      position across the full line.
    - Otherwise, the ray is cut into consecutive segments: from the edge marker
      to the first companion, then from each companion to the next one.  Beat
      positions inside segment k take the color of companion k, and nothing is
      painted beyond the last companion.

    Key insight:
    The repeated color still only defines the rhythm; the stopping logic now
    comes from the ordered list of all foreign dots, so the output is recovered
    by walking outward from the shooting edge and recoloring the beat positions
    segment by segment.
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
    forward = edge in ("left", "top")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def to_t(pos: int) -> int:
        return pos if forward else (line_len - 1 - pos)

    def from_t(t: int) -> int:
        return t if forward else (line_len - 1 - t)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = {color: sorted(ps) for color, ps in colors.items() if len(ps) >= 2}
        assert repeating, (
            f"Line {li} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple repeating colors: "
            f"{dict(repeating)}"
        )

        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2): {mpos}"
        )

        marker_ts = sorted(to_t(p) for p in mpos)
        assert marker_ts[0] == 0, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"must touch the shooting edge: {mpos}"
        )
        step = marker_ts[1] - marker_ts[0]
        assert step > 0, f"Line {li} ({'row' if by_row else 'col'}) has zero step"

        companions: list[tuple[int, int]] = []
        for color, ps in colors.items():
            if color == marker:
                continue
            assert len(ps) == 1, (
                f"Line {li} ({'row' if by_row else 'col'}) non-marker color {color} "
                f"appears {len(ps)} times: {ps}"
            )
            companions.append((to_t(ps[0]), color))

        beat_ts = list(range(0, line_len, step))
        if not companions:
            for t in beat_ts:
                rc = cell(li, from_t(t))
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = marker
            continue

        companions.sort(key=lambda item: item[0])
        prev_t = -1
        for idx, (end_t, paint) in enumerate(companions):
            assert end_t > prev_t, (
                f"Line {li} ({'row' if by_row else 'col'}) companions are not strictly ordered"
            )
            for t in beat_ts:
                in_segment = (t <= end_t) if idx == 0 else (prev_t < t <= end_t)
                if not in_segment:
                    continue
                rc = cell(li, from_t(t))
                cur = int(out[rc])
                if cur == bg or cur == marker:
                    out[rc] = paint
            prev_t = end_t

    return out


def _legacy_main_like(grid: np.ndarray) -> np.ndarray:
    """Reference the original one-companion semantics to avoid generator degeneration."""
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
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1
    edge = winners[0]

    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = {color: sorted(ps) for color, ps in colors.items() if len(ps) >= 2}
        assert len(repeating) == 1
        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2
        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        companion = None
        for color, ps in colors.items():
            if color == marker:
                continue
            if any(p % step == offset for p in ps):
                companion = color
                break

        if companion is not None:
            paint = companion
            all_ps = mpos + colors[companion]
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

    return out


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_COMPANION_COUNT_WEIGHTS = ((0, 0.18), (1, 0.27), (2, 0.33), (3, 0.22))
_MAX_RETRIES = 600


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
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


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
) -> dict[str, int]:
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


def _sample_companion_count(line_len: int, palette_size: int) -> int:
    counts, weights = zip(*_COMPANION_COUNT_WEIGHTS)
    count = random.choices(counts, weights=weights, k=1)[0]
    return min(count, palette_size - 1, max(0, line_len - 3))


def _build_line_dots(
    edge: str,
    line_len: int,
    palette: list[int],
    marker_color: int,
) -> tuple[list[tuple[int, int]], dict[str, object]] | None:
    forward = edge in ("left", "top")

    def from_t(t: int) -> int:
        return t if forward else (line_len - 1 - t)

    companion_count = _sample_companion_count(line_len, len(palette))

    while companion_count > 0:
        max_step = min(_STEP_RANGE[1], line_len - companion_count - 2)
        if max_step >= 1:
            break
        companion_count -= 1
    if companion_count == 0:
        max_step = min(_STEP_RANGE[1], line_len - 2)
    if max_step < 1:
        return None

    step = random.randint(1, max_step)
    marker_ts = [0, step]
    dots = [(from_t(t), marker_color) for t in marker_ts]

    candidate_ts = list(range(step + 1, line_len - 1))
    if companion_count > len(candidate_ts):
        return None

    chosen_ts = sorted(random.sample(candidate_ts, companion_count))
    companion_colors = random.sample([c for c in palette if c != marker_color], companion_count)
    for t, color in zip(chosen_ts, companion_colors):
        dots.append((from_t(t), color))

    meta = {
        "step": step,
        "companion_ts": chosen_ts,
        "has_offbeat": any(t % step != 0 for t in chosen_ts),
        "has_multi": companion_count >= 2,
    }
    return dots, meta


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

    palette_size = random.randint(4, min(8, len(_ALL_COLORS)))
    palette = random.sample(_ALL_COLORS, palette_size)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    has_variant_line = False

    for li in active_lines:
        marker_color = random.choice(palette)
        built = _build_line_dots(edge, line_len, palette, marker_color)
        if built is None:
            return None
        dots, meta = built

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1
        has_variant_line = has_variant_line or bool(
            meta["has_multi"] or meta["has_offbeat"]
        )

    if not has_variant_line:
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
    if np.array_equal(out, _legacy_main_like(inp)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
