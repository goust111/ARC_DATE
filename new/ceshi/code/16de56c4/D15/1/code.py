from __future__ import annotations

# Variant of 16de56c4
# Dimension: D15 — 伴随点数量 (meta)
# Change: 每行0或1个伴随点... → 伴随点对
#
# Design analysis for the D15 variant
# ===================================
# spec_variant:
#   preserved_backbone:
#     grid: 7-12 rows x 9-21 cols
#     background: 0 is still the dominant color
#     shooting_edge: still the unique boundary with the most non-bg dots
#     direction: left/right -> rows, top/bottom -> columns
#     active_lines: 4-9 spaced lines remain active
#     marker: still exactly one repeated color per active line that touches the
#       shooting edge and appears exactly twice; its gap is the step
#   changed_dimension:
#     old_companion: 0 or 1 aligned singleton
#     new_companion: 0 or 1 aligned pair, i.e. exactly two dots of the same
#       color on the marker beat, both beyond the inner marker and away from the
#       shooting edge
#     fill_interval: if a companion pair exists, only beat cells inside the
#       closed interval between those two companion dots are painted
#   retained_side_case:
#     bystander: 0 or 1 singleton dot not on the marker beat, unchanged in the
#       output
#
# a) Concrete behavior trace
#   Example line with left shooting edge, marker color 2, step 3:
#   input row: 2 . . 2 . . . . 5 . . 5 . . .
#   beats are positions 0,3,6,9,12,...
#   Original task with a single companion at 9 would paint 0..9 on the beat:
#   5 . . 5 . . 5 . . 5 ...
#   New variant with companion pair at 8? no, that would be off-beat, so the
#   pair must sit on beat cells such as 9 and 15. The output becomes:
#   2 . . 2 . . . . 5 . . 5 . . .
#   plus any missing beat cells only inside [9,15], i.e. position 12 becomes 5.
#   The important visible difference is that the early beat cells near the edge
#   remain unfilled and the marker pair keeps its own color.
#
# b) Complexity calibration
#   Two repeated colors can now coexist on one line, so the solver must not
#   iterate over every repeated color as if each were a marker. To keep the task
#   inferable, the marker is the only repeated color that touches the shooting
#   edge, while the companion pair never touches that edge.
#
# c) Hidden assumptions from the original solver that break here
#   - `repeating = {c: ps for ... if len(ps) >= 2}` used to identify exactly one
#     repeated color per line. This is no longer true.
#   - looping over every repeated color would incorrectly treat the companion
#     pair as another marker and invent a second period.
#   - `companion = oc` assumed one aligned singleton was sufficient. The new
#     rule requires two aligned dots of the same non-marker color.
#   - `all_ps = mpos + colors[companion]` used the marker plus a singleton to
#     define the fill span. The new span must come only from the two companion
#     dots.
#
# d) Placement legality enforced by the generator
#   - marker pair: positions {edge, edge +/- step}
#   - companion pair: same color, both on the same modular beat, both strictly
#     beyond the inner marker, and separated by at least one intermediate beat
#   - bystander: a singleton on a non-beat interior cell
#   - no overlaps and no out-of-bounds placements
#
# e) Visual observability
#   Each generated puzzle contains multiple companion-pair lines and at least
#   one no-pair line. That makes the new bounded-interval behavior visible in a
#   small set of examples instead of occasionally collapsing back to the old
#   full-line propagation.
#
# f) Visual cleanliness
#   Companion pairs and bystanders are mutually exclusive on a line, so there is
#   no stack of unrelated clues in the same row/column.
#
# g) Color inference
#   The fill color is still directly shown by the triggering dots themselves:
#   the companion pair color becomes the paint color, while marker-only lines
#   paint with the marker color. No invisible color ordering is used.
#
# h) New solver assumptions and generator guarantees
#   - assumption: exactly one repeated color on a non-empty line touches the
#     shooting edge -> generator always makes the marker the only such color
#   - assumption: any repeated non-marker color that matters is an aligned pair
#     -> generator never creates unrelated repeated colors
#   - assumption: pair lines should actually add cells -> generator separates the
#     companion endpoints by at least two step units

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint periodic beats fired from one unique grid edge, optionally clipped by
    paired companion dots.

    Input structure:
    - The background is the most frequent color and all other cells are sparse.
    - One boundary (top, bottom, left, or right) has strictly more non-bg dots
      than the others; that boundary is the shooting edge.
    - Every active shooting-direction line contains one marker color appearing
      exactly twice, with one marker on the shooting edge and one interior
      marker; their gap defines the period.
    - A line may also contain either one off-beat bystander dot or one aligned
      companion pair: two dots of the same non-marker color on that marker beat.

    Rule:
    - Find the unique shooting edge; left/right imply row-wise lines, top/bottom
      imply column-wise lines.
    - On each non-empty line, identify the marker as the only repeated color
      that touches the shooting edge. The marker pair determines the step and
      beat class.
    - If there is a second repeated color whose two dots lie on the same beat,
      those two dots form a closed interval and their color becomes the fill
      color, but only beat cells inside that interval are painted.
    - If no such companion pair exists, the marker color fills the full line on
      its beat. Off-beat bystanders are preserved unchanged.

    Key insight:
    - The shooting edge still reveals the propagation direction, but after the
      D15 change a line can have two repeated colors. The marker is the repeated
      pair anchored to the shooting edge; a second aligned same-color pair, when
      present, must be matched as a bounded gate that limits where the periodic
      fill appears.
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

        repeated = {color: sorted(ps) for color, ps in colors.items() if len(ps) == 2}
        marker_candidates = {
            color: ps for color, ps in repeated.items() if edge_pos in ps
        }
        assert len(marker_candidates) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) needs exactly one "
            f"edge-anchored repeated color: {dict(colors)}"
        )

        marker, mpos = next(iter(marker_candidates.items()))
        inner_pos = next(p for p in mpos if p != edge_pos)
        step = abs(inner_pos - edge_pos)
        assert step > 0, f"Line {li} has zero marker spacing"
        offset = edge_pos % step

        pair_candidates = []
        for color, ps in repeated.items():
            if color == marker:
                continue
            if edge_pos in ps:
                continue
            if all(p % step == offset for p in ps):
                pair_candidates.append((color, ps))

        assert len(pair_candidates) <= 1, (
            f"Line {li} ({'row' if by_row else 'col'}) has multiple aligned "
            f"companion pairs: {pair_candidates}"
        )
        assert len(repeated) == 1 + len(pair_candidates), (
            f"Line {li} ({'row' if by_row else 'col'}) has unsupported repeated "
            f"colors: {dict(colors)}"
        )

        if pair_candidates:
            paint, pair_pos = pair_candidates[0]
            lo, hi = min(pair_pos), max(pair_pos)
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
_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")
_PALETTE_SIZE_RANGE = (3, 6)

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


def _beats_beyond_inner(edge: str, line_len: int, step: int) -> list[int]:
    if edge in ("left", "top"):
        inner_pos = step
        return list(range(inner_pos + step, line_len, step))
    inner_pos = line_len - 1 - step
    return list(range(inner_pos - step, -1, -step))


def _valid_steps(edge: str, line_len: int, role: str) -> list[int]:
    max_step = min(_STEP_RANGE[1], line_len - 1)
    steps = []
    opposite_edge = line_len - 1 if edge in ("left", "top") else 0
    for step in range(_STEP_RANGE[0], max_step + 1):
        if role == "pair":
            beats = [p for p in _beats_beyond_inner(edge, line_len, step) if p != opposite_edge]
            if len(beats) >= 3:
                steps.append(step)
        elif role == "marker":
            if (line_len - 1) // step + 1 >= 3:
                steps.append(step)
        else:
            steps.append(step)
    return steps


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    role: str,
    palette: list[int],
):
    steps = _valid_steps(edge, line_len, role)
    if not steps:
        return None

    step = random.choice(steps)
    if edge in ("left", "top"):
        edge_pos, inner_pos = 0, step
    else:
        edge_pos, inner_pos = line_len - 1, line_len - 1 - step
    offset = edge_pos % step

    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    other_colors = [c for c in palette if c != marker_color]
    if not other_colors:
        return None

    if role == "pair":
        opposite_edge = line_len - 1 if edge in ("left", "top") else 0
        beats = [p for p in _beats_beyond_inner(edge, line_len, step) if p != opposite_edge]
        pairs = [
            (beats[i], beats[j])
            for i in range(len(beats))
            for j in range(i + 2, len(beats))
        ]
        if not pairs:
            return None
        a, b = random.choice(pairs)
        companion_color = random.choice(other_colors)
        dots.extend([(a, companion_color), (b, companion_color)])
    elif role == "bystander":
        non_beat = [
            p
            for p in range(1, line_len - 1)
            if p not in (edge_pos, inner_pos) and p % step != offset
        ]
        if not non_beat:
            return None
        dots.append((random.choice(non_beat), random.choice(other_colors)))

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


def _choose_roles(n_active: int) -> list[str]:
    n_pair = random.randint(max(2, n_active // 2), n_active - 1)
    roles = ["pair"] * n_pair
    roles.append("marker")
    while len(roles) < n_active:
        roles.append(random.choice(["marker", "bystander"]))
    random.shuffle(roles)
    return roles


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

    palette_size = random.randint(*_PALETTE_SIZE_RANGE)
    palette = random.sample(_COLORS, palette_size)
    roles = _choose_roles(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0

    for li, role in zip(active_lines, roles):
        marker_color = random.choice(palette)
        dots = _build_line_dots(edge, line_len, marker_color, role, palette)
        if dots is None:
            return None

        seen_pos = set()
        for pos, color in dots:
            if pos in seen_pos:
                return None
            seen_pos.add(pos)
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

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
