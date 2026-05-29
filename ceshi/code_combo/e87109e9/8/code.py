# Combo variant of e87109e9 (index 8)
# combo_spec: D12/0+D14/0
# Name: 顺序发射加障碍物消失

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis
# ----------------
# spec_variant:
#   preserved_visual_language:
#     - keep the original 6x24 legend strip with 4 colored bars on a grey background
#     - keep one solid teal source square in the main grid
#     - keep 4 edge-touching solid obstacle rectangles whose colors come from the legend
#     - keep the output as the legend-cropped main grid with teal beam trails
#   fused_rule:
#     original_behavior:
#       - all 4 beams are traced independently, so their order is irrelevant
#       - colored obstacle rectangles never change after a hit
#     combo_behavior:
#       - fire beams in the fixed order up -> right -> down -> left
#       - once a beam finishes, every newly painted teal trail cell becomes a blocking obstacle for later beams
#       - when a beam hits a colored obstacle, it turns by the legend and that whole obstacle rectangle disappears
#       - later beams therefore depend on both earlier trail placement and earlier obstacle consumption
#   concrete_small_example:
#     - the up beam first hits the north block and turns left, then hits the west block and heads downward
#     - this consumes the west block and leaves a vertical teal trail near the left side
#     - the right beam later hits the east block, turns downward, then consumes the south block and exits right
#     - under the old task the down beam would still hit the south block and bend, but here that block is gone
#     - under the old task the left beam would travel farther left, but here it stops when it reaches the up beam's trail
#   complexity_calibration:
#     - a fully random generator would often hide one of the two dimensions, so the generator uses a single
#       deliberate 4-obstacle template:
#         * up beam uses north + west
#         * right beam uses east + south
#         * down beam demonstrates south-block erasure
#         * left beam demonstrates blocking by an earlier trail
#     - this keeps the fusion visible in every accepted sample and avoids ambiguous "maybe sequential, maybe not" grids
#     - trail interaction is implemented as blocking instead of inventing a new trail-turn legend, because the
#       combo description only says earlier trails become obstacles. Blocking is the most literal and the cleanest.
#   main_line_assumption_audit:
#     - legend parsing still assumes color 5 is confined to the top band; the generator preserves that
#     - the old solver treated color 8 as non-obstructing paint; that is no longer safe because earlier teal trails
#       now block later beams, so the solver uses a separate earlier_trail_mask
#     - obstacle erasure breaks the old assumption that the main grid is static during tracing, so the solver keeps
#       a mutable obstacle layer distinct from the painted output
#     - erasing by color assumes obstacle colors are unique on the board; the generator guarantees 4 distinct colors
#     - the source square must not count as an earlier trail obstacle; the generator still creates exactly one solid
#       teal square and the simulator removes that footprint from every trail mask
#     - because trail collisions now stop a beam, cycle-prevention asserts remain valid even when earlier trails exist
#   placement_legality:
#     - all 4 obstacle rectangles are edge-touching, pairwise non-overlapping, and separated from the source square
#     - top/left/right/bottom spans are clamped away from the occupied corners so the rectangles never overlap
#     - the south span is forced to cover both the right beam's lane and the source column, but to exclude the up
#       beam's left-side lane, ensuring only the right beam can consume it
#     - the source square is placed so each planned obstacle is reachable with visible run-up room
#   observability_and_tidiness:
#     - the legend still explains only colored-obstacle turns; order is inferred from repeated geometric evidence
#     - every accepted sample differs from both ablations:
#         * sequential-without-erasure
#         * erasure-without-earlier-trail blocking
#     - the blocking trail stays in the interior while the colored rectangles stay on borders, so the output remains
#       readable rather than collapsing into a dense teal knot

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_THICKNESS_RANGE = (2, 4)
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45

MAX_RETRIES = 400


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode left/right bar placement into L/R turns."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
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
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _obstacle_color_at(
    obstacles: np.ndarray, bg: int, nr: int, nc: int, sz: int
) -> int | None:
    blk = obstacles[nr : nr + sz, nc : nc + sz]
    colors = set(map(int, np.unique(blk)))
    colors.discard(bg)
    colors.discard(TEAL)
    if not colors:
        return None
    assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
    return int(next(iter(colors)))


def _simulate_variant(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    use_trails: bool,
    erase_obstacles: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Simulate the combo or an ablation of it."""
    out = base.copy()
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    h, w = out.shape

    source_mask = np.zeros_like(base, dtype=bool)
    source_mask[sr : sr + sz, sc : sc + sz] = True
    earlier_trail_mask = np.zeros_like(base, dtype=bool)
    obstacles = base.copy()
    obstacles[source_mask] = bg

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    beam_stats: list[dict[str, object]] = []
    for d0 in (_N, _E, _S, _W):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_mask = np.zeros_like(base, dtype=bool)
        trail_block = False
        hits: list[tuple[int, int, int, int]] = []
        erased: list[int] = []

        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            if use_trails and earlier_trail_mask[nr : nr + sz, nc : nc + sz].any():
                trail_block = True
                break

            oc = _obstacle_color_at(obstacles, bg, nr, nc, sz)
            if oc is not None:
                assert oc in turns, "obstacle color must be encoded by legend"
                hits.append((d, oc, nr, nc))
                if erase_obstacles:
                    obstacles[obstacles == oc] = bg
                    erased.append(oc)
                d = _rot(d, turns[oc])
                continue

            r, c = nr, nc
            blk = out[r : r + sz, c : c + sz]
            bg_mask = blk == bg
            if np.any(bg_mask):
                blk[bg_mask] = TEAL
                out[r : r + sz, c : c + sz] = blk
            beam_mask[r : r + sz, c : c + sz] = True
            beam_mask[source_mask] = False

        earlier_trail_mask |= beam_mask
        beam_stats.append(
            {
                "dir": d0,
                "trail_block": trail_block,
                "trail_cells": int(beam_mask.sum()),
                "hits": hits,
                "erased": erased,
            }
        )

    return out, beam_stats


def _simulate_combo(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    return _simulate_variant(
        base, turns, bg, use_trails=True, erase_obstacles=True
    )


def _simulate_no_erase(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    return _simulate_variant(
        base, turns, bg, use_trails=True, erase_obstacles=False
    )


def _simulate_no_trails(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    return _simulate_variant(
        base, turns, bg, use_trails=False, erase_obstacles=True
    )


def _simulate_independent(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    out, _stats = _simulate_variant(
        base, turns, bg, use_trails=False, erase_obstacles=False
    )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the teal beams one after another so earlier trails block later beams and hit obstacles vanish.

    Input structure:
    - The top strip is a grey legend band with four colored vertical bars, each bar placed on the left or right edge of its cell.
    - Below the legend is one uniform-background playfield containing a solid teal source square and several solid colored obstacle rectangles.

    Rule:
    - Remove the legend strip in the output.
    - Fire the four sz x sz beams in the fixed order up, then right, then down, then left.
    - A beam advances one cell at a time, painting teal only onto background cells.
    - If its next footprint meets a colored obstacle, read that obstacle's turn from the legend, rotate 90 degrees, and erase that whole obstacle from the grid before continuing.
    - If its next footprint meets an earlier beam's teal trail, the beam stops there; it does not pass through or turn.
    - A beam also stops when its next step would leave the grid.

    Key insight: The legend still controls only the colored obstacles, but the decisive new information is temporal: earlier beams both consume obstacles and leave behind blocking teal trails, so later beams must be simulated on the updated grid in the visible direction order.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(
        rows_with_legend, np.arange(legend_h)
    ), "legend color must be confined to the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])

    out, _stats = _simulate_combo(base, turns, bg)
    return out


def _pick_span(
    axis_len: int,
    required: list[tuple[int, int]],
    min_len: int,
    max_len: int,
    *,
    low: int = 0,
    high: int | None = None,
) -> tuple[int, int] | None:
    """Pick a span covering every required interval inside [low, high]."""
    if high is None:
        high = axis_len - 1
    if low > high:
        return None

    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    if req0 < low or req1 > high:
        return None

    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, high - low + 1)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(low, req1 - length + 1)
        start1 = min(req0, high - length + 1)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _obstacle_rect(
    side: int, thickness: int, span: tuple[int, int], h: int, w: int
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _template_turns(side_colors: dict[int, int], legend_colors: list[int]) -> dict[int, str]:
    turns = {
        side_colors[_N]: "L",
        side_colors[_W]: "L",
        side_colors[_E]: "R",
        side_colors[_S]: "L",
    }
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))
    return turns


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    min_len = SPAN_MIN_MULT * sz
    max_len = min(SPAN_MAX_MULT * sz, max(h, w))

    thickness = {
        _N: random.randint(*OBS_THICKNESS_RANGE),
        _E: random.randint(*OBS_THICKNESS_RANGE),
        _S: random.randint(*OBS_THICKNESS_RANGE),
        _W: random.randint(*OBS_THICKNESS_RANGE),
    }
    t_n, t_e, t_s, t_w = (
        thickness[_N],
        thickness[_E],
        thickness[_S],
        thickness[_W],
    )

    # Turn coordinates after a collision on each edge.
    r_n = t_n
    c_w = t_w
    c_e = w - t_e - sz
    r_s = h - t_s - sz

    min_r = max(SQUARE_EDGE_GAP, r_n + 3)
    max_r = min(h - sz - SQUARE_EDGE_GAP, r_s - 3)
    min_c = max(SQUARE_EDGE_GAP, c_w + sz + 2)
    max_c = min(w - sz - SQUARE_EDGE_GAP, c_e - 3)
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    n_span = _pick_span(
        w,
        [(sc, sc + sz - 1)],
        min_len,
        min(max_len, w),
        low=t_w,
        high=w - t_e - 1,
    )
    w_span = _pick_span(
        h,
        [(r_n, r_n + sz - 1)],
        min_len,
        min(max_len, h),
        low=t_n,
        high=h - t_s - 1,
    )
    e_span = _pick_span(
        h,
        [(sr, sr + sz - 1)],
        min_len,
        min(max_len, h),
        low=t_n,
        high=h - t_s - 1,
    )
    s_low = max(t_w + sz, sc)
    s_span = _pick_span(
        w,
        [(sc, sc + sz - 1), (c_e, c_e + sz - 1)],
        min_len,
        min(max_len, w),
        low=s_low,
        high=w - t_e - 1,
    )
    if None in (n_span, w_span, e_span, s_span):
        return None

    spans = {
        _N: n_span,
        _W: w_span,
        _E: e_span,
        _S: s_span,
    }

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_palette = random.sample(legend_colors, 4)
    side_colors = {
        _N: side_palette[0],
        _E: side_palette[1],
        _S: side_palette[2],
        _W: side_palette[3],
    }
    turns = _template_turns(side_colors, legend_colors)

    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    occupied[sr : sr + sz, sc : sc + sz] = True
    for side in (_N, _E, _S, _W):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_colors[side]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    try:
        combo, combo_stats = _simulate_combo(base, turns, bg)
        no_erase, no_erase_stats = _simulate_no_erase(base, turns, bg)
        no_trails, no_trail_stats = _simulate_no_trails(base, turns, bg)
        independent = _simulate_independent(base, turns, bg)
    except AssertionError:
        return None

    up_hits = [hit[1] for hit in combo_stats[_N]["hits"]]
    right_hits = [hit[1] for hit in combo_stats[_E]["hits"]]
    down_hits = [hit[1] for hit in combo_stats[_S]["hits"]]
    left_hits = [hit[1] for hit in combo_stats[_W]["hits"]]

    if up_hits != [side_colors[_N], side_colors[_W]]:
        return None
    if right_hits != [side_colors[_E], side_colors[_S]]:
        return None
    if down_hits:
        return None
    if left_hits:
        return None
    if not bool(combo_stats[_W]["trail_block"]):
        return None
    if bool(combo_stats[_E]["trail_block"]) or bool(combo_stats[_S]["trail_block"]):
        return None
    if combo_stats[_S]["trail_cells"] <= combo_stats[_W]["trail_cells"]:
        # Down should extend farther than the blocked left beam.
        return None

    if bool(no_trail_stats[_W]["trail_block"]):
        return None
    if [hit[1] for hit in no_erase_stats[_S]["hits"]] != [side_colors[_S]]:
        return None

    if np.array_equal(combo, no_erase):
        return None
    if np.array_equal(combo, no_trails):
        return None
    if np.array_equal(combo, independent):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
