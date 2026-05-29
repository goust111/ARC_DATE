# Combo variant of 16de56c4 (index 5)
# combo_spec: D7/0+D17/1
# Name: 标记点擦除+旁观者擦除

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for combo D7/0 + D17/1
# --------------------------------------
# a) Concrete behavior / trace:
#    - Keep the original geometry: one dominant boundary picks the shooting direction, each
#      active row/column has one repeated marker color, and an optional third dot may be aligned
#      (companion) or off-beat (bystander).
#    - Example 1, companion line shooting from the left:
#      input row has marker 4 at positions 0 and 3, and companion 7 at position 9.
#      Original task filled 0/3/6/9 with 7, so markers stayed visible as recolored cells.
#      Combo output instead keeps only the produced trail: position 6 becomes 7, companion 9
#      remains 7, while marker cells 0 and 3 are erased to bg.
#    - Example 2, bystander line shooting from the left:
#      input row has marker 2 at positions 0 and 3, and bystander 8 at position 5.
#      Original task filled 0/3/6/9/... with 2 and kept the 8.
#      Combo output fills only non-marker beat cells such as 6/9/... with 2, erases 0 and 3
#      because they are markers, and also erases 5 because it is an off-beat bystander.
#    - Net effect: every instructional dot disappears unless it is an aligned companion, so the
#      visible output consists only of ray cells plus any companion that absorbed a ray.
# b) Complexity calibration:
#    - The object vocabulary stays small: exactly one repeated marker color per active line and
#      at most one extra dot. This keeps the rule inferable instead of introducing new objects.
#    - To stop the combo from collapsing into D7-only or D17-only behavior, the generator forces
#      both a true companion line and a true bystander line in every sample.
#    - Steps are restricted so marker erasure still leaves at least one newly painted non-marker
#      beat cell; otherwise a line could degenerate into pure deletion with no ray evidence.
# c) Hidden assumptions checked against the original solver:
#    - `out = g.copy()` is no longer sufficient because copied marker cells and copied bystanders
#      must both be cleared explicitly.
#    - Companion detection cannot be color-only; the extra dot must be classified by alignment to
#      the marker beat so companions survive while bystanders vanish.
#    - The original fill loop allowed repainting marker cells. That assumption is invalid now, so
#      fill skips marker positions and erasure happens afterward.
#    - The solver assumes one repeated color and at most one aligned extra per non-empty line; the
#      generator preserves exactly that layout.
# d) Placement legality:
#    - Active lines are spaced with at least one empty line between them, so row-wise or column-
#      wise rays never overlap each other.
#    - Marker positions are fixed to the shooting edge plus one interior beat; optional extras are
#      chosen from disjoint legal sets, so no overlap or off-grid placement occurs.
#    - Companion candidates are placed at least two beats beyond the inner marker, guaranteeing a
#      visible retained trail cell after both marker dots are erased.
# e) Visual observability:
#    - Every generated grid contains all three visual roles: disappearing repeated markers, at
#      least one surviving companion, and at least one disappearing off-beat bystander.
#    - This gives the human solver contrast: some third dots survive and cap the ray, while other
#      third dots vanish, and the repeated pair always vanishes.
# f) Visual cleanliness:
#    - One extra dot per line avoids stacked classifications on the same row/column.
#    - The bounded step range and spaced active lines keep rays legible rather than saturating the
#      entire board with dense parallel fills.
# g) Color inference:
#    - No rule depends on color ordering or numeric ids. Companion color propagation is signaled by
#      beat alignment and span truncation, not by a special palette convention.
# h) Fusion metric used in validation:
#    - For a grid with `n_active` active lines and `b` bystander lines, successful combo samples
#      must satisfy `erased_input_cells == 2 * n_active + b`, `gained_trail_cells > 0`,
#      `b >= 1`, and `companion_lines >= 1`.
#    - This metric proves both erasure dimensions are active: markers always disappear, and at
#      least one bystander also disappears beyond the marker loss.
# i) New solver assumptions and generator guarantees:
#    - Solver assumption: every non-empty line has exactly one marker pair whose spacing defines
#      a usable step. Generator guarantee: each active line is built from exactly one pair.
#    - Solver assumption: any surviving foreign dot on the beat is the single companion.
#      Generator guarantee: each line receives at most one extra dot, so aligned extras are unique.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow periodic rays from a unique shooting edge, but erase every non-companion instruction dot.

    Input structure:
    - The background is the most common color, and only a sparse set of rows or columns contains
      non-background dots.
    - Exactly one boundary has more non-background dots than the other three; that boundary is the
      shooting side, so left/right imply row-wise processing and top/bottom imply column-wise.
    - Each active shooting line contains exactly one repeated marker color with two dots whose gap
      defines the period, plus optionally one extra dot of another color.
    - An extra dot on the same beat is a companion; an extra dot off that beat is a bystander.

    Rule:
    - Detect the unique shooting edge, then process each row or column parallel to that direction.
    - For the repeated marker pair on a line, compute the step from the marker gap and compute the
      beat class from the first marker modulo that step.
    - If the extra dot is aligned, keep it as the companion, use its color for the ray, and limit
      filling to the span covered by the markers and that companion.
    - If there is no aligned extra, fill the full line on that beat using the marker color.
    - In all cases, erase the two marker cells and erase every off-beat bystander cell to the
      background, leaving only produced ray cells and any companion.

    Key insight:
    - The puzzle now separates instruction from product. The repeated pair and any off-beat extra
      are input-only clues that disappear, while aligned companions and the periodic trail remain.
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

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert len(repeating) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) must have exactly one repeated color: "
            f"{dict(colors)}"
        )

        marker, mpos = next(iter(repeating.items()))
        assert len(mpos) == 2, (
            f"Line {li} ({'row' if by_row else 'col'}) marker {marker} "
            f"has {len(mpos)} dots (expected 2): {mpos}"
        )
        a, b = sorted(mpos)
        step = b - a
        offset = a % step

        aligned_extras: list[tuple[int, int]] = []
        bystander_positions: list[int] = []
        for other_color, positions in colors.items():
            if other_color == marker:
                continue
            for pos in positions:
                if pos % step == offset:
                    aligned_extras.append((other_color, pos))
                else:
                    bystander_positions.append(pos)

        assert len(aligned_extras) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple companions: {aligned_extras}"
        )

        for pos in bystander_positions:
            out[cell(li, pos)] = bg

        if aligned_extras:
            paint, companion_pos = aligned_extras[0]
            lo = min(a, b, companion_pos)
            hi = max(a, b, companion_pos)
        else:
            paint = marker
            lo, hi = 0, line_len - 1

        marker_positions = set(mpos)
        for p in range(lo, hi + 1):
            if p % step != offset or p in marker_positions:
                continue
            rc = cell(li, p)
            if int(out[rc]) == bg:
                out[rc] = paint

        for pos in marker_positions:
            out[cell(li, pos)] = bg

    return out


# --------------- generator (combo D7/0 + D17/1) ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_LINE_WEIGHTS = {
    "marker_only": 0.35,
    "companion": 0.30,
    "bystander": 0.35,
}

_MAX_RETRIES = 300


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    """Pick (H, W, num_lines, line_len) compatible with edge and active count."""
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
    """Choose n indices from [0, total) with at least one empty index between neighbors."""
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _valid_steps(line_len: int) -> list[int]:
    hi = min(_STEP_RANGE[1], line_len - 1)
    return [step for step in range(_STEP_RANGE[0], hi + 1) if 2 * step <= line_len - 1]


def _companion_steps(line_len: int) -> list[int]:
    return [step for step in _valid_steps(line_len) if 3 * step <= line_len - 1]


def _bystander_steps(line_len: int) -> list[int]:
    return [step for step in _valid_steps(line_len) if step > 1]


def _weighted_choice(options: list[str]) -> str:
    weights = [_LINE_WEIGHTS[option] for option in options]
    return random.choices(options, weights=weights, k=1)[0]


def _choose_line_kinds(n_active: int, line_len: int) -> list[str] | None:
    kinds = ["marker_only"]

    companion_steps = _companion_steps(line_len)
    bystander_steps = _bystander_steps(line_len)
    if not companion_steps or not bystander_steps:
        return None

    kinds.append("companion")
    kinds.append("bystander")

    while len(kinds) < n_active:
        feasible = ["marker_only", "companion", "bystander"]
        kinds.append(_weighted_choice(feasible))

    random.shuffle(kinds)
    return kinds


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    step: int,
    kind: str,
    palette: list[int],
) -> list[tuple[int, int]]:
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
        companion_positions = list(range(inner_pos + 2 * step, line_len, step))
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
        companion_positions = list(range(inner_pos - 2 * step, -1, -step))

    offset = edge_pos % step
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    other_colors = [color for color in palette if color != marker_color]
    assert other_colors, "Palette must contain at least one non-marker color"

    if kind == "companion":
        assert companion_positions, f"No legal companion position for step={step}, len={line_len}"
        dots.append((random.choice(companion_positions), random.choice(other_colors)))
    elif kind == "bystander":
        bystander_positions = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in (edge_pos, inner_pos)
        ]
        assert bystander_positions, f"No legal bystander position for step={step}, len={line_len}"
        dots.append((random.choice(bystander_positions), random.choice(other_colors)))

    return dots


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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid combo input."""
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    by_row = edge in ("left", "right")

    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims

    line_kinds = _choose_line_kinds(n_active, line_len)
    if line_kinds is None:
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(2, 8)
    palette = random.sample(_ALL_COLORS, palette_size)

    valid_steps = _valid_steps(line_len)
    companion_steps = _companion_steps(line_len)
    bystander_steps = _bystander_steps(line_len)
    if not valid_steps or not companion_steps or not bystander_steps:
        return None

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    shooting_edge_dots = 0
    all_rc: list[tuple[int, int]] = []
    bystander_lines = 0
    companion_lines = 0

    for li, kind in zip(active_lines, line_kinds):
        marker_color = random.choice(palette)
        if kind == "companion":
            step = random.choice(companion_steps)
            companion_lines += 1
        elif kind == "bystander":
            step = random.choice(bystander_steps)
            bystander_lines += 1
        else:
            step = random.choice(valid_steps)

        dots = _build_line_dots(edge, line_len, marker_color, step, kind, palette)

        for pos, color in dots:
            r, c = cell(li, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    if bystander_lines < 1 or companion_lines < 1:
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

    erased = int(np.count_nonzero((inp != 0) & (out == 0)))
    gained_trail = int(np.count_nonzero((inp == 0) & (out != 0)))
    expected_erased = 2 * n_active + bystander_lines
    if erased != expected_erased or gained_trail <= 0:
        return None

    surviving_input_cells = int(np.count_nonzero((inp != 0) & (out != 0)))
    if surviving_input_cells < companion_lines:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
