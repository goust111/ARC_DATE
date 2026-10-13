from __future__ import annotations
from collections import defaultdict
import numpy as np
from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill periodic progressions that shoot from a unique grid edge.

    Input structure:
    - bg = most common color.  Sparse non-bg dots of several colors.
    - All repeating colors originate from the same grid edge (top, bottom,
      left, or right — the one with the most non-bg dots).
    - Each repeating color has exactly 2 dots per shooting-direction line;
      their gap defines the step.
    - Every non-bg dot lies on a shooting-direction line that contains a
      repeating color.

    Rule:
    - Determine the unique shooting edge; left/right → rows, top/bottom → columns.
    - For each shooting-direction line, the repeating color has exactly 2 dots;
      step = gap between them, offset = first position mod step.
    - A different-colored dot aligned to the same beat (pos % step == offset) is
      a companion: it absorbs the ray and becomes the fill color, limiting the
      fill range to the span between repeating and companion dots.
    - An unaligned dot is a bystander and left untouched.
    - Without a companion the repeating color fills the entire line.

    Key insight: the shooting edge is identified by counting boundary dots;
    the gap between the two marker dots is the period; a companion sharing the
    modular class acts as a wall that stops the ray and dictates the paint color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    # --- Step 1: determine the unique shooting edge ---
    edge_counts = {
        'top':    sum(1 for c in range(W) if g[0, c] != bg),
        'bottom': sum(1 for c in range(W) if g[H - 1, c] != bg),
        'left':   sum(1 for r in range(H) if g[r, 0] != bg),
        'right':  sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [e for e, c in edge_counts.items() if c == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]

    by_row = edge in ('left', 'right')
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    # --- Step 2: process each shooting-direction line ---
    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            v = int(g[cell(li, p)])
            if v != bg:
                colors[v].append(p)

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
            a, b = sorted(mpos)
            step = b - a
            offset = a % step

            companion = None
            for oc, ops in colors.items():
                if oc == marker:
                    continue
                if any(p % step == offset for p in ops):
                    companion = oc
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

# --------------- generator (claude-opus-attempt1) ---------------

# --- Configuration ---
_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)       # number of non-empty shooting lines
_STEP_RANGE = (1, 5)               # marker dot spacing
_COLORS = list(range(1, 10))       # non-bg palette
_EDGES = ('top', 'bottom', 'left', 'right')

# Line type probabilities: marker-only / companion / bystander
_P_MARKER_ONLY = 0.40
_P_COMPANION = 0.35                # cumulative threshold = 0.75

_MAX_RETRIES = 300


def _choose_dims(edge: str, n_active: int):
    """Pick (H, W, num_lines, line_len) compatible with edge and active count.

    Returns None if the active count cannot fit the dimension range.
    """
    min_num_lines = 2 * n_active - 1  # minimum to fit n_active lines with gap >= 1
    by_row = edge in ('left', 'right')
    if by_row:
        lo = max(_ROWS_RANGE[0], min_num_lines)
        if lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(lo, _ROWS_RANGE[1])
        W = random.randint(*_COLS_RANGE)
        return H, W, H, W
    else:
        lo = max(_COLS_RANGE[0], min_num_lines)
        if lo > _COLS_RANGE[1]:
            return None
        W = random.randint(lo, _COLS_RANGE[1])
        H = random.randint(*_ROWS_RANGE)
        return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    """Choose *n* indices from [0, total) with gaps >= 1 between consecutive ones."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _build_line_dots(edge: str, line_len: int, marker_color: int, step: int):
    """Build the dot positions for one active line.

    Returns (dot_list, edge_pos) where each dot is (pos, color).
    """
    if edge in ('left', 'top'):
        edge_pos, inner_pos = 0, step
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
    offset = edge_pos % step

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]

    roll = random.random()
    if roll < _P_MARKER_ONLY:
        pass  # marker only
    elif roll < _P_MARKER_ONLY + _P_COMPANION:
        # Companion: aligned to beat, beyond inner marker
        other_colors = [c for c in _COLORS if c != marker_color]
        if edge in ('left', 'top'):
            beats = list(range(inner_pos + step, line_len, step))
        else:
            beats = list(range(inner_pos - step, -1, -step))
        if other_colors and beats:
            dots.append((random.choice(beats), random.choice(other_colors)))
    else:
        # Bystander: NOT aligned to beat
        other_colors = [c for c in _COLORS if c != marker_color]
        non_beat = [p for p in range(line_len)
                    if p % step != offset and p not in (edge_pos, inner_pos)]
        if other_colors and non_beat:
            dots.append((random.choice(non_beat), random.choice(other_colors)))

    return dots, edge_pos


def _count_other_edge_dots(dots_rc: list[tuple[int, int]], edge: str,
                           H: int, W: int) -> dict[str, int]:
    """Count how many placed dots land on each non-shooting edge."""
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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input."""
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ('left', 'right')

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc = []  # (r, c) of every placed dot

    for li in active_lines:
        marker_color = random.choice(_COLORS)
        step = random.randint(*_STEP_RANGE)
        step = min(step, line_len - 1)

        dots, edge_pos = _build_line_dots(edge, line_len, marker_color, step)

        for pos, color in dots:
            r, c = cell(li, pos)
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1  # marker always has one dot on the shooting edge

    # Verify shooting edge is unambiguous
    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(cnt >= shooting_edge_dots for cnt in other_counts.values()):
        return None

    # Validate with solver
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
