# Combo variant of e87109e9 (index 18)
# combo_spec: D5/2+D14/0
# Name: 多源顺序发射

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(_ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask

# Design analysis
# ----------------
# spec_variant:
#   preserved_visual_language:
#     legend:
#       - keep the original 6x24 grey legend strip with four 4-column cells
#       - bar side still encodes the turn direction for that color
#     obstacles:
#       - keep edge-touching solid rectangles as in the base task
#       - allow one obstacle color to be reused on multiple edges so the legend can still fit 2-3 source colors
#     sources:
#       - replace the single teal square with 2-3 isolated 2x2 source squares of distinct colors
#       - every source color also appears in the legend, so later trail collisions can still be resolved from the same cue
#   core_change:
#     original_behavior:
#       - one teal source emitted four independent beams
#       - teal paint was never treated as a future obstacle, so beam order did not matter
#     fused_behavior:
#       - sources are processed in visible order: top-to-bottom, then left-to-right
#       - within each source, beams fire in the fixed order up, right, down, left
#       - once a beam finishes, its newly painted cells keep the source color and become real obstacles for all later beams
#       - therefore a later beam can turn on:
#           * a border obstacle color from the input
#           * an earlier beam of the same source
#           * an earlier beam of another source
#       - every turn, including trail-induced ones, is decoded from the legend by the color that was hit
#   concrete_small_example:
#       - suppose the earliest source is red and its up beam hits a top obstacle whose legend bar means clockwise
#       - that beam bends right and leaves a red horizontal trail near the top
#       - a lower blue source later fires its own up beam under that red segment
#       - under the old independent logic the blue beam would continue straight upward to the border
#       - under the fused logic it first meets the red trail, consults the red legend cell, turns, and paints a bent path
#       - the output difference comes from both dimensions at once: there are multiple sources, and their emission order matters
#   complexity_calibration:
#       - a fully random generator would rarely make different sources interact, so many samples would collapse to the old task
#       - to keep the new behavior visible, generation uses two mirrored cascade templates:
#           * early source hooks upward then toward the right wall
#           * early source hooks upward then toward the left wall
#       - later sources are placed under that early horizontal trail, so their up beams visibly turn on another source's path
#       - 2-source boards use an extra side obstacle so the earliest beam can form a richer hook
#       - some 3-source boards deliberately drop that side obstacle: the A->B interaction stays visible, while the third
#         source remains readable and the search space avoids unstable cyclic layouts
#       - all sources use 2x2 squares to keep multi-source detection unambiguous and the trail widths visually consistent
#   main_line_assumption_audit:
#       - the old solver assumed exactly one solid teal square; that is no longer valid, so source detection now scans for 2-3
#         isolated 2x2 square components whose colors are legend colors and whose components do not touch the border
#       - the old obstacle test discarded teal globally; that would destroy sequential behavior, so the new simulator instead
#         keeps a frozen obstacle grid and ignores only the current source square while checking the next footprint
#       - the current beam must not react to its own in-progress paint, only to already-frozen earlier beams; this is enforced
#         by updating the obstacle grid only after a beam completes
#       - source order must be human-visible rather than color-id based; sorting by source position preserves that
#       - mixed-color hits inside one footprint would be ambiguous, so the generator is built to keep each reachable collision
#         footprint monochrome; the solver asserts this assumption
#   placement_legality:
#       - sources are interior 2x2 squares with row gaps and column gaps large enough to avoid overlap
#       - top obstacles are chosen to cover only the first source's column, not the later source columns, so later up beams
#         need the earlier trail in order to bend
#       - side obstacles are shallow vertical rectangles near the turning row; they help the hooked beam bend again without
#         blocking the lower source squares directly; they are used only in the 2-source template
#       - all placements are validated by simulation, and invalid or degenerate attempts are rejected
#   observability_and_tidiness:
#       - source order depends only on geometric position, which a human can see
#       - trail color equals source color, so when a later beam turns on an earlier trail the responsible source remains legible
#       - border rectangles stay on the edges while the multi-source trails occupy the interior, keeping the output readable
#       - the generator rejects any sample whose sequential output matches the trail-ignoring baseline, preventing silent collapse

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

SOURCE_SIZE = 2
SOURCE_COUNT_3_PROB = 0.35
SOURCE_ROW_GAP = 4
SOURCE_EDGE_GAP = 4

OBS_THICKNESS_RANGE = (2, 4)
TOP_SPAN_MIN = 6
SIDE_SPAN_MIN = 5
SIDE_SPAN_MAX = 8

MAX_RETRIES = 400

_TEMPLATES = [
    {"name": "cascade_right", "hook_turn": "R", "side": _E},
    {"name": "cascade_left", "hook_turn": "L", "side": _W},
]


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode turn direction from the top legend strip."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turns: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        turns[color] = "L" if int(xs[0]) <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turns) == 4, "legend must encode four turn colors"
    return turns


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _is_full_square(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> bool:
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    return hh == ww and int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == color)) == hh * ww


def _find_sources(base: np.ndarray, legend_colors: set[int], bg: int) -> list[tuple[int, int, int, int]]:
    """Return sorted (r, c, size, color) source squares."""
    sources: list[tuple[int, int, int, int]] = []
    for color in sorted(legend_colors):
        for cells in components_mask(base == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(cells)
            if not _is_full_square(base, int(r0), int(c0), int(r1), int(c1), color):
                continue
            size = int(r1 - r0 + 1)
            if size != SOURCE_SIZE:
                continue
            if r0 == 0 or c0 == 0 or r1 == base.shape[0] - 1 or c1 == base.shape[1] - 1:
                continue
            sources.append((int(r0), int(c0), size, int(color)))
    sources.sort(key=lambda item: (item[0], item[1], item[3]))
    assert 2 <= len(sources) <= 3, "main grid must contain 2-3 interior source squares"
    assert len({color for _, _, _, color in sources}) == len(sources), "source colors must be distinct"
    return sources


def _simulate_sequential(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Process sources in position order; frozen earlier trails become later obstacles."""
    out = base.copy()
    static = base.copy()
    sources = _find_sources(base, set(turns), bg)
    frozen_owner = np.full(base.shape, -1, dtype=int)
    stats: list[dict[str, int]] = []

    for src_idx, (sr, sc, sz, color) in enumerate(sources):
        source_mask = np.zeros_like(base, dtype=bool)
        source_mask[sr : sr + sz, sc : sc + sz] = True
        source_stats = {
            "source_index": src_idx,
            "color": color,
            "self_trail_hits": 0,
            "other_trail_hits": 0,
            "static_hits": 0,
            "painted_cells": 0,
        }

        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

        def obstacle_at_next(nr: int, nc: int) -> tuple[int, str] | None:
            blk = static[nr : nr + sz, nc : nc + sz]
            mask = ~source_mask[nr : nr + sz, nc : nc + sz]
            vals = blk[mask]
            colors = set(map(int, np.unique(vals)))
            colors.discard(bg)
            if not colors:
                return None
            assert len(colors) == 1, "beam footprint must not hit mixed colors"
            hit_color = int(next(iter(colors)))
            owner_blk = frozen_owner[nr : nr + sz, nc : nc + sz][mask]
            owners = set(map(int, np.unique(owner_blk)))
            owners.discard(-1)
            if not owners:
                return hit_color, "static"
            assert len(owners) == 1, "trail collision footprint must belong to one earlier beam"
            owner = int(next(iter(owners)))
            return hit_color, "self_trail" if owner == src_idx else "other_trail"

        for d0 in range(4):
            r, c, d = sr, sc, d0
            seen: set[tuple[int, int, int]] = set()
            beam_mask = np.zeros_like(base, dtype=bool)
            while True:
                state = (r, c, d)
                assert state not in seen, "beam path must not cycle"
                seen.add(state)

                dr, dc = _DIRS[d]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    break

                hit = obstacle_at_next(nr, nc)
                if hit is None:
                    r, c = nr, nc
                    blk = out[r : r + sz, c : c + sz]
                    blk[blk == bg] = color
                    out[r : r + sz, c : c + sz] = blk
                    beam_mask[r : r + sz, c : c + sz] = True
                    beam_mask[source_mask] = False
                    continue

                hit_color, hit_kind = hit
                assert hit_color in turns, "every hit color must be encoded by the legend"
                source_stats[hit_kind + "_hits"] += 1
                d = _rot(d, turns[hit_color])

            static[beam_mask] = color
            frozen_owner[beam_mask] = src_idx
            source_stats["painted_cells"] += int(beam_mask.sum())

        stats.append(source_stats)

    return out, stats


def _simulate_independent(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    """Baseline behavior: all beams ignore previously painted trails."""
    out = base.copy()
    sources = _find_sources(base, set(turns), bg)

    for _src_idx, (sr, sc, sz, color) in enumerate(sources):
        source_mask = np.zeros_like(base, dtype=bool)
        source_mask[sr : sr + sz, sc : sc + sz] = True

        def in_bounds(r: int, c: int) -> bool:
            return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

        def obstacle_at_next(nr: int, nc: int) -> int | None:
            blk = base[nr : nr + sz, nc : nc + sz]
            mask = ~source_mask[nr : nr + sz, nc : nc + sz]
            vals = blk[mask]
            colors = set(map(int, np.unique(vals)))
            colors.discard(bg)
            if not colors:
                return None
            assert len(colors) == 1, "beam footprint must not hit mixed colors"
            return int(next(iter(colors)))

        for d0 in range(4):
            r, c, d = sr, sc, d0
            seen: set[tuple[int, int, int]] = set()
            while True:
                state = (r, c, d)
                assert state not in seen, "beam path must not cycle"
                seen.add(state)

                dr, dc = _DIRS[d]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    break

                oc = obstacle_at_next(nr, nc)
                if oc is None:
                    r, c = nr, nc
                    blk = out[r : r + sz, c : c + sz]
                    blk[blk == bg] = color
                    out[r : r + sz, c : c + sz] = blk
                    continue

                assert oc in turns, "every hit color must be encoded by the legend"
                d = _rot(d, turns[oc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint several colored beam systems whose earlier trails steer later ones.

    Input structure:
    - Top strip: the original grey 6x24 legend with four colored bars, where each bar sits on the left or right side of its cell.
    - Below it: one uniform-background playfield containing 2-3 isolated 2x2 source squares of different colors plus a few edge-touching obstacle rectangles.

    Rule:
    - Remove the legend strip in the output.
    - Sort the source squares by position (top-to-bottom, then left-to-right). For each source in that order, emit four sz x sz beams in the sequence up, then right, then down, then left.
    - A beam moves one cell at a time, painting only background cells with the source color. If its next footprint meets any previously existing colored block, look up that hit color in the legend and turn 90 degrees accordingly (left bar = counterclockwise, right bar = clockwise).
    - After a beam finishes, its newly painted trail stays on the board and becomes a real obstacle for all later beams from the same source and from later sources.
    - A beam stops only when its next step would leave the grid.

    Key insight: The legend now governs both border obstacles and old beam trails, because every source trail keeps the source color. Solving the task therefore requires tracking visible source order as well as the per-color turn rule.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    out, _stats = _simulate_sequential(base, turns, bg)
    return out


def _edge_rect(side: int, thickness: int, span: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _pick_rows(h: int, n_sources: int) -> list[int] | None:
    usable_lo = SOURCE_EDGE_GAP
    usable_hi = h - SOURCE_SIZE - SOURCE_EDGE_GAP
    if usable_hi - usable_lo + 1 < 1:
        return None
    candidates = list(range(usable_lo, usable_hi + 1))
    random.shuffle(candidates)
    rows: list[int] = []
    for row in sorted(candidates):
        if not rows or row - rows[-1] >= SOURCE_ROW_GAP:
            rows.append(row)
            if len(rows) == n_sources:
                return rows
    return None


def _choose_columns(
    template: dict[str, object],
    lane_side: int,
    n_sources: int,
    w: int,
) -> list[int] | None:
    margin = SOURCE_EDGE_GAP
    if template["side"] == _E:
        anchor_lo = 8 if n_sources == 3 else margin
        anchor_hi = lane_side - 10
        if anchor_lo > anchor_hi:
            return None
        sc0 = random.randint(anchor_lo, anchor_hi)
        lo = sc0 + 4
        hi = lane_side - 4
        if lo > hi:
            return None
        sc1 = random.randint(lo, hi)
        if n_sources == 2:
            return [sc0, sc1]
        extra_hi = sc0 - 4
        extra_lo = margin
        if extra_lo > extra_hi:
            return None
        sc2 = random.randint(extra_lo, extra_hi)
        return [sc0, sc1, sc2]

    anchor_lo = lane_side + 8
    anchor_hi = w - SOURCE_SIZE - (4 if n_sources == 3 else margin)
    if anchor_lo > anchor_hi:
        return None
    sc0 = random.randint(anchor_lo, anchor_hi)
    lo = lane_side + 2
    hi = sc0 - 4
    if lo > hi:
        return None
    sc1 = random.randint(lo, hi)
    if n_sources == 2:
        return [sc0, sc1]
    extra_lo = sc0 + 4
    extra_hi = w - SOURCE_SIZE - margin
    if extra_lo > extra_hi:
        return None
    sc2 = random.randint(extra_lo, extra_hi)
    return [sc0, sc1, sc2]


def _top_span_for_sources(
    template: dict[str, object],
    source_cols: list[int],
    w: int,
    side_thickness: int,
) -> tuple[int, int] | None:
    sc0 = source_cols[0]
    left_neighbors = [c for c in source_cols[1:] if c < sc0]
    right_neighbors = [c for c in source_cols[1:] if c > sc0]
    left_guard = (max(left_neighbors) + 2) if left_neighbors else 0
    right_guard = (min(right_neighbors) - 2) if right_neighbors else (w - side_thickness - 2)
    if template["side"] == _E:
        start_lo = max(0, sc0 - 3, left_guard)
        start_hi = sc0
        if start_lo > start_hi:
            return None
        start = random.randint(start_lo, start_hi)
        end_lo = max(sc0 + SOURCE_SIZE - 1, start + TOP_SPAN_MIN - 1)
        end_hi = min(right_guard, w - side_thickness - 2)
        if end_lo > end_hi:
            return None
        end = random.randint(end_lo, end_hi)
        return start, end

    end_lo = sc0 + SOURCE_SIZE - 1
    end_hi = min(w - side_thickness - 2, sc0 + 3, right_guard)
    if end_lo > end_hi:
        return None
    end = random.randint(end_lo, end_hi)
    start_lo = max(left_guard, end - 10)
    start_hi = min(sc0, end - TOP_SPAN_MIN + 1)
    if start_lo > start_hi:
        return None
    start = random.randint(start_lo, start_hi)
    return start, end


def _side_span(top_thickness: int, h: int) -> tuple[int, int] | None:
    start = max(1, top_thickness - 1)
    max_len = min(SIDE_SPAN_MAX, h - start - 2)
    if max_len < SIDE_SPAN_MIN:
        return None
    length = random.randint(SIDE_SPAN_MIN, max_len)
    return start, start + length - 1


def _turn_map(
    legend_colors: list[int],
    source_colors: list[int],
    obstacle_color: int,
    hook_turn: str,
) -> dict[int, str]:
    turns = {obstacle_color: hook_turn}
    for color in source_colors:
        turns[color] = hook_turn
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))
    return turns


def _render_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    n_sources = 3 if random.random() < SOURCE_COUNT_3_PROB else 2
    template = random.choice(_TEMPLATES)

    top_thickness = random.randint(*OBS_THICKNESS_RANGE)
    use_side_obstacle = n_sources == 2
    side_thickness = random.randint(*OBS_THICKNESS_RANGE) if use_side_obstacle else 0
    if use_side_obstacle:
        lane_side = w - side_thickness - SOURCE_SIZE if template["side"] == _E else side_thickness + 1
    else:
        lane_side = w - SOURCE_SIZE if template["side"] == _E else 0

    rows = _pick_rows(h, n_sources)
    cols = _choose_columns(template, lane_side, n_sources, w)
    if rows is None or cols is None:
        return None

    source_positions = list(zip(rows, cols))
    if any(abs(c - lane_side) < 3 for _, c in source_positions):
        return None

    top_span = _top_span_for_sources(template, cols, w, side_thickness)
    side_span = _side_span(top_thickness, h) if use_side_obstacle else None
    if top_span is None or (use_side_obstacle and side_span is None):
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    source_colors = legend_colors[:n_sources]
    obstacle_color = legend_colors[n_sources]
    turns = _turn_map(legend_colors, source_colors, obstacle_color, str(template["hook_turn"]))
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for (sr, sc), color in zip(source_positions, source_colors):
        if sr + SOURCE_SIZE > h or sc + SOURCE_SIZE > w:
            return None
        if occupied[sr : sr + SOURCE_SIZE, sc : sc + SOURCE_SIZE].any():
            return None
        occupied[sr : sr + SOURCE_SIZE, sc : sc + SOURCE_SIZE] = True
        base[sr : sr + SOURCE_SIZE, sc : sc + SOURCE_SIZE] = color

    edge_specs = [(_N, top_thickness, top_span)]
    if use_side_obstacle:
        edge_specs.append((int(template["side"]), side_thickness, side_span))
    for side, thickness, span in edge_specs:
        r0, r1, c0, c1 = _edge_rect(side, thickness, span, h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = obstacle_color

    try:
        expected, stats = _simulate_sequential(base, turns, bg)
        baseline = _simulate_independent(base, turns, bg)
    except AssertionError:
        return None

    inter_hits = sum(info["other_trail_hits"] for info in stats)
    total_paint = sum(info["painted_cells"] for info in stats)
    if inter_hits < 1:
        return None
    if total_paint < 28:
        return None
    if np.array_equal(expected, baseline):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _render_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
