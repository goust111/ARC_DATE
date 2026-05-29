# Variant of 16de56c4
# Dimension: D7 — 变换规则 (transform)
# Change: 背景和标记色单元格被填充为轨迹颜色，旁观者点保持不变... → 旁观者条件变换

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the D7 variant
# a) Concrete behavior and trace:
#    The original puzzle already has periodic rays: the two same-color marker
#    dots define a beat, and an aligned farther dot can stop the ray and donate
#    the fill color.  The new behavior keeps that geometry but changes paint
#    priority.  A non-marker dot can now sit on the same beat before the true
#    absorber.  Example on a left-shooting row with step 3:
#    input positions [0:M, 3:M, 6:B, 9:C] produce output beat cells
#    [0,3,6,9] all colored C.  Under the old solver, position 6 stayed color B
#    because only background/marker cells were overwritten; under the variant,
#    the ray visibly passes through 6, so B is repainted to C.
# b) Complexity calibration:
#    Only one new branch is introduced: an aligned bystander before the farthest
#    aligned dot.  We keep at most one such covered bystander per line and still
#    allow marker-only, companion-only, and off-beat bystander lines, so the
#    task stays readable and does not explode into many special cases.
# c) Implicit-assumption audit:
#    The old assignment guard `if cur == bg or cur == marker` relied on the old
#    invariant that non-marker dots never sat on the filled trajectory unless
#    they were the final companion.  That invariant is no longer true, so the
#    write step must repaint every visited beat cell, not only bg/marker cells.
#    Companion detection also cannot stop at the first aligned non-marker dot
#    anymore; it must pick the farthest aligned dot away from the shooting edge.
# d) Placement legality:
#    Generator lines still contain exactly one repeated marker color with two
#    dots, one on the shooting edge and one one-step inward.  Covered-bystander
#    lines require two more aligned beat positions deeper inside the grid, so
#    their step is capped by floor((line_len - 1) / 3).  Distinct active lines
#    avoid overlap; per-line positions are unique and in bounds by construction.
# e) Visual observability:
#    Every generated puzzle includes at least one covered-bystander line and at
#    least one off-beat bystander line.  This creates a direct contrast: some
#    extra dots get overwritten exactly when they lie on the periodic path,
#    while others remain because they miss the beat.
# f) Visual cleanliness:
#    Interaction stays one-dimensional per active row/column.  There is no
#    stacking between different rays, and each line has at most four dots, so
#    the overwritten cell is easy to spot in the input/output pair.
# g) Color inference:
#    The overwritten bystander is always recolored to the same trajectory color
#    already evidenced by markers/background cells on that line; the puzzle does
#    not ask the solver to infer arbitrary color mappings.
# h) New solver assumptions and generator guarantees:
#    Solver assumes any aligned non-marker candidates that matter lie farther
#    from the edge than the inner marker, and that the farthest such candidate
#    is unique.  The generator guarantees this by placing at most one covered
#    bystander and one final companion on a line, with strictly increasing depth.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint periodic rays launched from one dominant grid edge.

    Input structure:
    - Background is the most frequent color; non-background cells are sparse.
    - One boundary (top, bottom, left, or right) has strictly the most dots and
      determines the shooting direction for all active lines.
    - On each active shooting line, exactly one color repeats twice: one marker
      sits on the shooting edge and the other sits one step inward, so their
      gap fixes the beat.
    - Extra single dots may appear on the same line.  Some miss the beat,
      while some lie on the beat farther from the edge than the inner marker.

    Rule:
    - Detect the unique shooting edge; left/right shoot across rows, top/bottom
      shoot across columns.
    - For each active line, use the two repeated marker dots to compute the
      periodic beat class.
    - Among non-marker dots that share that beat and lie deeper inside the ray,
      the farthest one is the absorber: its color becomes the trajectory color
      and it truncates the fill span to the marker/absorber extent.
    - Any nearer on-beat bystander is now covered by that same trajectory color;
      off-beat bystanders are untouched.
    - If there is no aligned absorber, the marker color fills the entire line
      on its beat.

    Key insight:
    - The puzzle is still solved by reading a single edge, a single period, and
      a single controlling aligned endpoint per line; the D7 twist is that the
      ray now repaints every visited beat cell, including non-marker bystanders
      that used to survive unchanged.
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
    edge_pos = 0 if edge in ("left", "top") else line_len - 1
    forward = 1 if edge in ("left", "top") else -1

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def distance_from_edge(pos: int) -> int:
        return forward * (pos - edge_pos)

    for li in range(num_lines):
        colors: dict[int, list[int]] = defaultdict(list)
        for p in range(line_len):
            value = int(g[cell(li, p)])
            if value != bg:
                colors[value].append(p)

        if not colors:
            continue

        repeating = [(color, positions) for color, positions in colors.items() if len(positions) == 2]
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeated marker color: "
            f"{dict(colors)}"
        )
        marker, mpos = repeating[0]
        a, b = sorted(mpos)
        step = b - a
        offset = a % step
        inner_pos = b if edge in ("left", "top") else a

        aligned_candidates: list[tuple[int, int]] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for p in positions:
                if p % step != offset:
                    continue
                if distance_from_edge(p) > distance_from_edge(inner_pos):
                    aligned_candidates.append((p, other_color))

        if aligned_candidates:
            companion_pos, paint = max(aligned_candidates, key=lambda item: distance_from_edge(item[0]))
            lo = min(a, b, companion_pos)
            hi = max(a, b, companion_pos)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        for p in range(lo, hi + 1):
            if p % step == offset:
                out[cell(li, p)] = paint

    return out


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_LINE_TYPES = (
    "marker_only",
    "companion_only",
    "offbeat_bystander",
    "covered_bystander",
)
_LINE_TYPE_WEIGHTS = (0.20, 0.25, 0.25, 0.30)
_MAX_RETRIES = 400


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


def _edge_params(edge: str, line_len: int) -> tuple[int, int]:
    if edge in ("left", "top"):
        return 0, 1
    return line_len - 1, -1


def _pick_distinct_colors(k: int, banned: set[int]) -> list[int] | None:
    choices = [color for color in _COLORS if color not in banned]
    if len(choices) < k:
        return None
    return random.sample(choices, k)


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    line_type: str,
) -> dict[str, object] | None:
    edge_pos, forward = _edge_params(edge, line_len)

    if line_type == "covered_bystander":
        max_step = min(_STEP_RANGE[1], (line_len - 1) // 3)
    elif line_type == "companion_only":
        max_step = min(_STEP_RANGE[1], (line_len - 1) // 2)
    else:
        max_step = min(_STEP_RANGE[1], line_len - 1)
    if max_step < _STEP_RANGE[0]:
        return None

    step = random.randint(_STEP_RANGE[0], max_step)
    inner_pos = edge_pos + forward * step
    dots: list[tuple[int, int]] = [(edge_pos, marker_color), (inner_pos, marker_color)]

    beats = []
    depth = 0
    while True:
        pos = edge_pos + forward * depth * step
        if not (0 <= pos < line_len):
            break
        beats.append(pos)
        depth += 1
    deeper_beats = beats[2:]

    if line_type == "marker_only":
        return {"dots": dots, "step": step, "line_type": line_type}

    if line_type == "companion_only":
        if not deeper_beats:
            return None
        colors = _pick_distinct_colors(1, {marker_color})
        if colors is None:
            return None
        dots.append((random.choice(deeper_beats), colors[0]))
        return {"dots": dots, "step": step, "line_type": line_type}

    if line_type == "offbeat_bystander":
        offset = edge_pos % step
        non_beat = [
            pos
            for pos in range(line_len)
            if pos not in (edge_pos, inner_pos) and pos % step != offset
        ]
        if not non_beat:
            return None
        colors = _pick_distinct_colors(1, {marker_color})
        if colors is None:
            return None
        dots.append((random.choice(non_beat), colors[0]))
        return {"dots": dots, "step": step, "line_type": line_type}

    if len(deeper_beats) < 2:
        return None
    cover_index = random.randint(0, len(deeper_beats) - 2)
    bystander_pos = deeper_beats[cover_index]
    companion_choices = deeper_beats[cover_index + 1 :]
    colors = _pick_distinct_colors(2, {marker_color})
    if colors is None:
        return None
    bystander_color, companion_color = colors
    dots.append((bystander_pos, bystander_color))
    dots.append((random.choice(companion_choices), companion_color))
    return {"dots": dots, "step": step, "line_type": line_type}


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]],
    edge: str,
    H: int,
    W: int,
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


def _choose_line_types(n_active: int) -> list[str]:
    line_types = ["covered_bystander", "offbeat_bystander"]
    line_types.append(random.choice(["marker_only", "companion_only"]))
    while len(line_types) < n_active:
        line_types.append(random.choices(_LINE_TYPES, weights=_LINE_TYPE_WEIGHTS, k=1)[0])
    random.shuffle(line_types)
    return line_types


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

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    line_types = _choose_line_types(n_active)
    covered_count = 0

    for li, line_type in zip(active_lines, line_types):
        built = None
        for _ in range(25):
            marker_color = random.choice(_COLORS)
            built = _build_line_dots(edge, line_len, marker_color, line_type)
            if built is not None:
                break
        if built is None:
            return None

        dots = built["dots"]
        if built["line_type"] == "covered_bystander":
            covered_count += 1

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    if covered_count == 0:
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
