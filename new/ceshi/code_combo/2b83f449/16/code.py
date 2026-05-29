# Combo variant of 2b83f449 (index 16)
# combo_spec: D3/1+D5/3
# Name: 颜色令牌+水平留痕
#
# Design analysis for the fusion:
# - Concrete behavior in pixels:
#   even rows are pipe rows, odd rows are wall rows, and each 1x3 gate bar still
#   opens into a 3-cell vertical post. As in D3/1, a token first slides
#   horizontally inside its current pipe segment until it is aligned with the only
#   available downward opening, then it drops by two rows. Every horizontal slide
#   paints the traversed cells with the token's own color, so the output contains
#   red and blue trace bands that show the horizontal motion explicitly.
# - Small example:
#   if a blue token starts at the left edge of a segment whose only exit is at
#   column 6, it paints the cells from the edge to column 6 in blue and then drops.
#   A red token on the right edge of a segment with only exit at column 9 paints its
#   route in red and then drops. When no exit exists, red tokens pack to the far end
#   of the final segment relative to their entry column, while blue tokens pack to
#   the near end. This keeps the D5/3 "color changes settling rule" idea active
#   instead of making color matter only for traces.
# - Complexity calibration:
#   the generator keeps the original single-exit pipe topology because the rotated
#   gate geometry naturally induces one downward opening per reachable segment. The
#   combo signal therefore comes from two visible effects at once: colored horizontal
#   traces and color-dependent near/far settling. Samples are rejected unless both
#   colors leave traces and the blue-near packing changes the result versus an
#   all-far baseline.
# - Implicit assumptions broken from the base solver:
#   the original code assumed one token color and one universal settling rule, then
#   recovered packing direction by rescanning the final segment for a unique
#   entry-from-above. The combo keeps the single-exit geometry but breaks the
#   one-color / one-rule assumption, so the new solver parses two token colors,
#   records the exact column used to enter the final segment, and computes packing
#   per token color.
# - Placement legality:
#   gates stay at least 5 columns apart within a wall row and at least 2 columns
#   away from gates in adjacent wall rows. All gates remain interior, so both the
#   closed 1x3 bars and opened posts fit in-bounds. Tokens stay on pipe-row edges in
#   the input and traces are only painted on bg cells inside the active segment, so
#   nothing overlaps illegally.
# - Visual observability:
#   the output shows long same-color horizontal bands plus same-color final token
#   piles. Blue traces come from left-edge sources and red traces from right-edge
#   sources, so a human can see that the token color identifies the moving token and
#   that the color also predicts how the final pile is arranged.
# - New solver assumptions and generator guarantees:
#   1. Token colors are exactly blue=1 and red=2, distinct from wall/gate/bg/post.
#      The generator always uses that palette.
#   2. Horizontal traces must actually matter. The generator requires non-zero trace
#      length for both colors and at least one token to descend 3+ levels.
#   3. Blue near-packing must matter. The generator compares against a "far for all"
#      baseline and rejects degenerate samples where the outputs would match.

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import paste

WALL = 0
BLUE = 1
RED = 2
OPEN = 6
GATE = 7
BG = 8

TOKEN_COLORS = (BLUE, RED)


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != BG:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == BG:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [
        c
        for c in range(a, b + 1)
        if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG
    ]


def _token_side(c: int, width: int) -> str:
    if c <= 1:
        return "left"
    if c >= width - 2:
        return "right"
    return "internal"


def _choose_exit(exits: list[int], color: int) -> int | None:
    if not exits:
        return None
    if len(exits) == 1:
        return int(exits[0])
    return int(max(exits) if color == BLUE else min(exits))


def _pack_end(seg: tuple[int, int], anchor: int, color: int) -> int:
    a, b = seg
    near_end = a if abs(anchor - a) <= abs(anchor - b) else b
    far_end = b if near_end == a else a
    return near_end if color == BLUE else far_end


def _build_opened_grid(gates: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in centers:
            grid[wr - 1, center] = OPEN
            grid[wr, center - 1] = BG
            grid[wr, center] = OPEN
            grid[wr, center + 1] = BG
            grid[wr + 1, center] = OPEN
    return grid


def _build_input(gates: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG

    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for center in centers:
            paste(grid, gate_patch, wr, center - 1)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = BLUE
        grid[r, width - 1] = RED

    return grid


def _parse_opened_from_input(grid: np.ndarray) -> tuple[np.ndarray, list[list[int]], list[dict[str, int | str]]]:
    g = np.array(grid, dtype=int)
    height, width = g.shape
    n_walls = (height - 1) // 2
    opened = g.copy()
    gates: list[list[int]] = [[] for _ in range(n_walls)]
    sources: list[dict[str, int | str]] = []

    for wall_idx in range(n_walls):
        wr = 2 * wall_idx + 1
        for c in range(1, width - 1):
            if int(g[wr, c - 1]) == GATE and int(g[wr, c]) == GATE and int(g[wr, c + 1]) == GATE:
                if not gates[wall_idx] or gates[wall_idx][-1] != c:
                    gates[wall_idx].append(c)
        for center in gates[wall_idx]:
            opened[wr, center - 1 : center + 2] = BG
            opened[wr - 1, center] = OPEN
            opened[wr, center] = OPEN
            opened[wr + 1, center] = OPEN

    order = 0
    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        for c in range(width):
            color = int(g[r, c])
            if color not in TOKEN_COLORS:
                continue
            sources.append(
                {
                    "r": r,
                    "c": c,
                    "color": color,
                    "side": _token_side(c, width),
                    "order": order,
                }
            )
            opened[r, c] = BG
            order += 1

    return opened, gates, sources


def _simulate(
    opened: np.ndarray,
    sources: list[dict[str, int | str]],
    *,
    color_sensitive_exit: bool,
    color_sensitive_pack: bool,
    draw_traces: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(opened, dtype=int, copy=True)
    records: list[dict[str, object]] = []
    metrics = {
        "blue_trace": 0,
        "red_trace": 0,
        "max_depth": 0,
    }

    for src in sorted(sources, key=lambda item: int(item["order"])):
        r = int(src["r"])
        c = int(src["c"])
        depth = 0
        entry_c: int | None = None
        color = int(src["color"])
        start_c = c

        while True:
            seg = _find_segment(opened, r, c)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg)
            if color_sensitive_exit:
                exit_c = _choose_exit(exits, color)
            else:
                exit_c = _choose_exit(exits, RED)
            if exit_c is None:
                break
            if draw_traces and exit_c != c:
                lo, hi = sorted((c, exit_c))
                out[r, lo : hi + 1] = color
                key = "blue_trace" if color == BLUE else "red_trace"
                metrics[key] += hi - lo + 1
            entry_c = exit_c
            r += 2
            c = exit_c
            depth += 1
            if r >= opened.shape[0]:
                break

        seg = _find_segment(opened, r, c)
        records.append(
            {
                "r": r,
                "c": c,
                "seg": seg if seg is not None else (c, c),
                "entry_c": entry_c,
                "start_c": start_c,
                "source": src,
                "depth": depth,
            }
        )
        metrics["max_depth"] = max(metrics["max_depth"], depth)

    buckets: dict[tuple[int, tuple[int, int], int], list[dict[str, object]]] = defaultdict(list)
    for rec in records:
        seg = rec["seg"]
        assert isinstance(seg, tuple)
        anchor = int(rec["entry_c"]) if rec["entry_c"] is not None else int(rec["start_c"])
        color = int(rec["source"]["color"])
        if color_sensitive_pack:
            pack_end = _pack_end(seg, anchor, color)
        else:
            pack_end = _pack_end(seg, anchor, RED)
        buckets[(int(rec["r"]), seg, pack_end)].append(rec)

    for (r, (a, b), pack_end), recs in buckets.items():
        ordered = sorted(recs, key=lambda item: int(item["source"]["order"]))
        if pack_end == a:
            for idx, rec in enumerate(ordered):
                cc = a + idx
                if cc <= b:
                    out[r, cc] = int(rec["source"]["color"])
        else:
            for idx, rec in enumerate(ordered):
                cc = b - idx
                if cc >= a:
                    out[r, cc] = int(rec["source"]["color"])

    return out, metrics


def main(grid: np.ndarray) -> np.ndarray:
    opened, _gates, sources = _parse_opened_from_input(np.array(grid, dtype=int))
    out, _metrics = _simulate(
        opened,
        sources,
        color_sensitive_exit=True,
        color_sensitive_pack=True,
        draw_traces=True,
    )
    return out


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    out: list[int] = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue
        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            out.append(c)
    return out


def _random_gates(rng: np.random.Generator) -> tuple[list[list[int]], int]:
    n_walls = int(rng.integers(5, 9))
    width = int(rng.integers(16, 20))
    gates: list[list[int]] = [[] for _ in range(n_walls)]

    center = int(rng.integers(3, width - 3))
    for row in range(n_walls):
        if row > 0:
            shifts = [d for d in (-4, -3, -2, 2, 3, 4) if 2 <= center + d <= width - 3]
            if shifts:
                center = int(center + rng.choice(shifts))
        gates[row].append(center)

    target_total = int(rng.integers(max(8, n_walls + 2), min(13, 3 * n_walls + 1)))
    while sum(len(row) for row in gates) < target_total:
        row = int(rng.integers(n_walls))
        candidates = _candidate_centers(gates, row, width)
        if not candidates:
            break
        gates[row].append(int(rng.choice(candidates)))
        gates[row].sort()

    return gates, width


def _core_behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    opened, _gates, sources = _parse_opened_from_input(inp)
    combo_out, combo_metrics = _simulate(
        opened,
        sources,
        color_sensitive_exit=True,
        color_sensitive_pack=True,
        draw_traces=True,
    )
    far_out, _ = _simulate(
        opened,
        sources,
        color_sensitive_exit=True,
        color_sensitive_pack=False,
        draw_traces=True,
    )
    no_trace_out, _ = _simulate(
        opened,
        sources,
        color_sensitive_exit=True,
        color_sensitive_pack=True,
        draw_traces=False,
    )
    metrics = dict(combo_metrics)
    metrics["pack_diff"] = int(not np.array_equal(combo_out, far_out))
    metrics["trace_diff"] = int(not np.array_equal(combo_out, no_trace_out))
    return metrics


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    gates, width = _random_gates(rng)
    inp = _build_input(gates, width)
    out = main(inp)
    metrics = _core_behavior_metrics(inp)
    if metrics["blue_trace"] <= 0 or metrics["red_trace"] <= 0:
        return None
    if metrics["pack_diff"] == 0 or metrics["trace_diff"] == 0:
        return None
    if metrics["max_depth"] < 3:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for task 2b83f449 index 16")
