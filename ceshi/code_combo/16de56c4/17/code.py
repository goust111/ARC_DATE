# Combo variant of 16de56c4 (index 17)
# combo_spec: D3/0+D7/1
# Name: 交替颜色填充+保留源

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    Take a left-shooting row with marker A at positions 0 and 2, so step = 2.
#    If there is a same-beat companion B at position 10, the base task would fill
#    0,2,4,6,8,10 all as B. D3/0 alone would alternate them as A,B,A,B,A,B.
#    The fused rule keeps the two marker sources intact, so the actual combo row is
#    A,0,A,0,A,0,B,0,A,0,B. The beat phase is still anchored at the shooting-edge
#    marker, but source cells are exempt from repainting. This creates the intended
#    contrast: a visible ABAB trail with the original marker pair still standing out.
# b) Complexity calibration:
#    The puzzle should still be solved via edge detection + period inference, not by
#    memorizing a special case. We therefore keep marker-only, companion, and bystander
#    lines from the original task, but force multiple companion lines so both new effects
#    are observable across 2-4 examples. Companion lines prefer longer odd beat indices
#    so the alternating tail contains several blank products beyond the preserved sources.
# c) Broken implicit assumptions from the base solver:
#    - `a,b = sorted(mpos)` is no longer enough to define alternation phase, because
#      right/bottom firing runs in the opposite direction. The phase origin must be the
#      marker that lies on the shooting edge.
#    - `if cur == bg or cur == marker` is wrong under keep-source; marker cells must not
#      be overwritten even when the alternating color at that beat would be the companion.
#    - The D3-only solver allowed repainting companion cells to satisfy parity. That still
#      remains valid here; only marker sources are protected.
# d) Placement legality:
#    Active lines are spaced apart, so rows/columns do not overlap. Each line still has
#    exactly one repeated marker color and at most one extra dot. Companion dots are placed
#    on odd beat indices beyond the inner marker; bystanders are forced off-beat, so they
#    cannot be mistaken for companions.
# e) Visual observability:
#    Samples with only marker-only lines or only very short companion spans would hide the
#    combo. The generator therefore requires at least two companion lines, rejects cases
#    where the combo output equals either single-variant output, and scores visibility by
#    how many post-marker alternating beats are available.
# f) Visual cleanliness:
#    A small per-grid palette is used, and each active line contains at most one extra dot.
#    This keeps the striped rays readable and avoids clutter from many unrelated colors in
#    the same region.
# g) Inferability of color changes:
#    No color-id ordering is used. The two stripe colors are read directly from the repeated
#    marker color and the aligned companion dot. The preserved-source exception is also
#    spatially grounded: only the two repeated source dots resist repainting.
# h) New solver assumptions and generator guarantees:
#    - Every active line has exactly one repeated color with exactly two dots.
#    - Any aligned non-marker dot is the unique companion on that line.
#    - Companion lines use odd companion beat indices, so the endpoint itself supports the
#      alternating phase instead of contradicting it.
#    - The generator validates every candidate with the combo solver and also checks that
#      the combo output differs from base, D3-only, and D7-only behaviors on that same input.


def _shooting_edge(grid: np.ndarray, bg: int) -> str:
    H, W = grid.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if grid[0, c] != bg),
        "bottom": sum(1 for c in range(W) if grid[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if grid[r, 0] != bg),
        "right": sum(1 for r in range(H) if grid[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg dots on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    return winners[0]


def _line_axes(edge: str, shape: tuple[int, int]):
    H, W = shape
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    return by_row, num_lines, line_len, cell


def _line_colors(grid: np.ndarray, bg: int, line: int, line_len: int, cell):
    colors: dict[int, list[int]] = defaultdict(list)
    for pos in range(line_len):
        value = int(grid[cell(line, pos)])
        if value != bg:
            colors[value].append(pos)
    return colors


def _edge_anchor(edge: str, marker_positions: list[int]) -> int:
    return min(marker_positions) if edge in ("left", "top") else max(marker_positions)


def _paint_for_beat(marker: int, companion: int, edge_anchor: int, step: int, pos: int) -> int:
    beat_idx = abs(pos - edge_anchor) // step
    return marker if beat_idx % 2 == 0 else companion


def _solve_variant(
    grid: np.ndarray,
    *,
    alternating_companion: bool,
    keep_source: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge = _shooting_edge(g, bg)
    by_row, num_lines, line_len, cell = _line_axes(edge, g.shape)

    for line in range(num_lines):
        colors = _line_colors(g, bg, line, line_len, cell)
        if not colors:
            continue

        repeating = {color: positions for color, positions in colors.items() if len(positions) >= 2}
        assert repeating, (
            f"Line {line} ({'row' if by_row else 'col'}) has non-bg dots "
            f"{dict(colors)} but no repeating color"
        )

        for marker, marker_positions in repeating.items():
            assert len(marker_positions) == 2, (
                f"Line {line} ({'row' if by_row else 'col'}) marker {marker} "
                f"has {len(marker_positions)} dots (expected 2): {marker_positions}"
            )
            anchor = _edge_anchor(edge, marker_positions)
            inner = max(marker_positions) if edge in ("left", "top") else min(marker_positions)
            step = abs(inner - anchor)
            assert step > 0, f"Line {line} ({'row' if by_row else 'col'}) has zero step"
            offset = anchor % step

            companion = None
            companion_positions: list[int] = []
            for other_color, other_positions in colors.items():
                if other_color == marker:
                    continue
                if any(pos % step == offset for pos in other_positions):
                    companion = other_color
                    companion_positions = other_positions
                    break

            if companion is None:
                lo, hi = 0, line_len - 1
            else:
                all_positions = marker_positions + companion_positions
                lo, hi = min(all_positions), max(all_positions)

            for pos in range(lo, hi + 1):
                if pos % step != offset:
                    continue
                rc = cell(line, pos)
                if keep_source and int(g[rc]) == marker:
                    continue
                if companion is None or not alternating_companion:
                    paint = marker if companion is None else companion
                else:
                    paint = _paint_for_beat(marker, companion, anchor, step, pos)
                out[rc] = paint

    return out


def _base_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, alternating_companion=False, keep_source=False)


def _d3_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, alternating_companion=True, keep_source=False)


def _d7_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, alternating_companion=False, keep_source=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire periodic rays from a unique boundary and stripe them while preserving the source dots.

    Input structure:
    - The background is the most common color and the informative cells are sparse dots.
    - Exactly one boundary has strictly more non-background dots than the other three; that
      boundary is the shooting edge and determines whether rays run along rows or columns.
    - Each active firing line contains one repeated marker color appearing exactly twice; the
      boundary-side marker and the interior marker define the beat spacing. A third dot may be
      present: aligned to that beat progression as a companion, or off-beat as a bystander.

    Rule:
    - Detect the unique shooting edge, then process each shooting-direction row/column separately.
    - The gap between the two marker dots is the step, and the marker on the shooting edge fixes
      the alternation phase.
    - Without a companion, fill every beat position on the whole line with the marker color.
    - With a companion, fill only the beat positions inside the span covered by the marker pair
      and that companion. Those beat positions alternate marker color, companion color, marker
      color, companion color, moving inward from the shooting edge.
    - The two repeated marker dots are sources and stay unchanged; only non-source beat cells are
      painted. Off-beat bystanders also stay unchanged.

    Key insight:
    The structure is still decoded from boundary density and repeated-dot spacing, but the final
    painter now combines two effects: a companion supplies the second stripe color, and the marker
    pair remains visible as preserved sources embedded inside the alternating ray.
    """
    return _solve_variant(grid, alternating_companion=True, keep_source=True)


# --------------- generator ---------------

_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (1, 5)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("top", "bottom", "left", "right")

_MIN_COMPANION_LINES = 2
_PALETTE_SIZE_RANGE = (3, 6)
_MAX_RETRIES = 500


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


def _choose_line_types(n_active: int) -> list[str]:
    # Keep the fusion-visible companion lines dominant enough to expose the combo,
    # but guarantee all three branches appear in a typical puzzle.
    assert n_active >= _MIN_COMPANION_LINES + 2
    kinds = ["companion"] * _MIN_COMPANION_LINES + ["bystander", "marker_only"]
    extras = n_active - len(kinds)
    for _ in range(extras):
        kinds.append(random.choices(
            ["companion", "bystander", "marker_only"],
            weights=(2, 1, 1),
            k=1,
        )[0])
    random.shuffle(kinds)
    return kinds


def _edge_positions(edge: str, line_len: int, step: int) -> tuple[int, int, int]:
    if edge in ("left", "top"):
        edge_pos = 0
        inner_pos = step
    else:
        edge_pos = line_len - 1
        inner_pos = line_len - 1 - step
    return edge_pos, inner_pos, edge_pos % step


def _has_non_beat(edge: str, line_len: int, step: int) -> bool:
    edge_pos, inner_pos, offset = _edge_positions(edge, line_len, step)
    return any(
        pos % step != offset and pos not in (edge_pos, inner_pos)
        for pos in range(line_len)
    )


def _companion_beat_choices(line_len: int, step: int) -> list[int]:
    max_beats = (line_len - 1) // step
    odd = list(range(3, max_beats + 1, 2))
    preferred = [beat_idx for beat_idx in odd if beat_idx >= 5]
    return preferred or odd


def _allowed_steps(edge: str, line_len: int, kind: str) -> list[int]:
    steps = list(range(_STEP_RANGE[0], min(_STEP_RANGE[1], line_len - 1) + 1))
    if kind == "companion":
        return [step for step in steps if _companion_beat_choices(line_len, step)]
    if kind == "bystander":
        return [step for step in steps if _has_non_beat(edge, line_len, step)]
    return steps


def _build_line_dots(
    edge: str,
    line_len: int,
    marker_color: int,
    palette: list[int],
    step: int,
    kind: str,
):
    edge_pos, inner_pos, offset = _edge_positions(edge, line_len, step)
    direction = 1 if edge in ("left", "top") else -1
    dots = [(edge_pos, marker_color), (inner_pos, marker_color)]
    meta = {"kind": kind, "companion_index": None}

    other_colors = [color for color in palette if color != marker_color]
    if not other_colors:
        other_colors = [color for color in _ALL_COLORS if color != marker_color]

    if kind == "marker_only":
        return dots, meta

    if kind == "companion":
        beat_choices = _companion_beat_choices(line_len, step)
        if not beat_choices or not other_colors:
            return None
        beat_idx = random.choice(beat_choices)
        pos = edge_pos + direction * beat_idx * step
        dots.append((pos, random.choice(other_colors)))
        meta["companion_index"] = beat_idx
        return dots, meta

    if kind == "bystander":
        if not other_colors:
            return None
        non_beat = [
            pos
            for pos in range(line_len)
            if pos % step != offset and pos not in (edge_pos, inner_pos)
        ]
        if not non_beat:
            return None
        dots.append((random.choice(non_beat), random.choice(other_colors)))
        return dots, meta

    raise ValueError(f"Unknown line type: {kind}")


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


def _visibility_score(meta: list[dict]) -> int:
    score = 0
    for item in meta:
        if item["kind"] != "companion":
            continue
        beat_idx = item["companion_index"]
        if beat_idx is None:
            continue
        score += max(0, beat_idx - 2)
    return score


def _variant_deltas(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    return {
        "vs_base": int(np.count_nonzero(combo != _base_main(inp))),
        "vs_d3_only": int(np.count_nonzero(combo != _d3_only_main(inp))),
        "vs_d7_only": int(np.count_nonzero(combo != _d7_only_main(inp))),
    }


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
    palette = random.sample(_ALL_COLORS, palette_size)
    line_types = _choose_line_types(n_active)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    shooting_edge_dots = 0
    meta: list[dict] = []

    for line, kind in zip(active_lines, line_types):
        steps = _allowed_steps(edge, line_len, kind)
        if not steps:
            return None
        step = random.choice(steps)
        marker_color = random.choice(palette)

        built = _build_line_dots(edge, line_len, marker_color, palette, step, kind)
        if built is None:
            return None
        dots, line_meta = built
        meta.append(line_meta)

        for pos, color in dots:
            r, c = cell(line, pos)
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        shooting_edge_dots += 1

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    if _visibility_score(meta) < 7:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None

    deltas = _variant_deltas(inp)
    if min(deltas.values()) <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")
