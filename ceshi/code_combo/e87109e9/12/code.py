# Combo variant of e87109e9 (index 12)
# combo_spec: D4/1+D9/1
# Name: 虚线轨迹加可变宽度

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior
#    - The original task emits four teal beams that keep a constant source-sized thickness and
#      paint solid trails. Here we fuse two changes at once:
#         1. Every segment uses the same visible dash rhythm: paint, paint, gap, repeat.
#         2. Beam width starts at 2 and increases by exactly 1 after every obstacle turn.
#    - A concrete trace:
#         source 2x2 -> north segment moves upward with width 2 and paints cross-sections
#         step 1 on, step 2 on, step 3 off, ...
#         after hitting the top obstacle it turns, widens to 3, paints a 3x3 corner anchor,
#         then continues with the same 2-on-1-off rhythm.
#         If that beam hits a second obstacle, the third segment widens again to 4.
#    - The difference from the original is therefore visible in two independent but interacting
#      ways: gaps appear along the strip, and later segments occupy thicker lanes than earlier ones.
#
# b) Complexity calibration
#    - The dash period is fixed globally to a single easy-to-learn pattern: 2 painted slices, then
#      1 blank slice. There is no per-color or per-beam variation.
#    - Width growth is also single-rule: 2 -> 3 -> 4. We avoid shrink/grow branching so the solver
#      only has to count turns, not classify segment lengths.
#    - The generator forces exactly one two-turn beam in every sample, so width 4 is always present
#      and the fusion does not collapse back to "just dashed 2-wide beams".
#
# c) Hidden-assumption audit against the original main()
#    - Original bounds checks assumed width == source size forever; now every check must use the
#      current width.
#    - Original painting used sliding sz x sz blocks, which would hide a 2-on-1-off pattern because
#      adjacent width-2 blocks overlap. The combo instead paints one newly entered row/column slice
#      per step, while still using square footprints for movement and obstacle detection.
#    - Original cycle detection keyed only on (r,c,d). Width and dash phase now matter, so state
#      tracking must include the current width and the step count within the active segment.
#    - Original turn handling only changed direction. Here a turn also increments width, shifts the
#      anchor for south/east hits so the widened square stays flush to the obstacle, and paints a
#      new corner anchor before the next dashed segment begins.
#
# d) Placement legality
#    - Obstacles remain solid edge-touching rectangles, one on each side, with distinct colors that
#      all appear in the legend.
#    - Span selection is constraint-based: every obstacle must cover its own incoming 2-wide launch
#      lane; the single chain target must also cover one extra 3-wide lane from a turned beam; all
#      other potential second/third-hit lanes are explicitly excluded.
#    - The source is restricted so every direct segment and the forced chained segment have enough
#      free length to show at least one blank dash cell.
#
# e) Visual observability
#    - Both new control signals are geometric and directly visible. Humans can see the repeated
#      paint-paint-gap rhythm and can also count that the path gets wider after turns.
#    - No hidden use of color ids, ordering, randomness in the solver, or invisible metadata is
#      involved. Colors still only encode left-vs-right turning in the legend.
#
# f) Visual cleanliness
#    - We keep exactly one two-turn chain and make all other beams stop after one turn. That keeps
#      the board readable while still guaranteeing the 2 -> 3 -> 4 widening sequence exists.
#    - Obstacles stay on the borders, so the thicker later segments mostly expand into open interior
#      space instead of tangling around the source.
#
# g) Inferability of the fused change
#    - The dash rule is constant across all segments and all samples.
#    - The width rule depends only on turn count, not on obstacle color or segment length.
#    - Because every sample contains both width 3 and width 4 segments plus visible dash gaps,
#      the output cannot be explained by the original solid constant-width rule.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (21, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

SOURCE_SIZE = 2
INITIAL_WIDTH = 2
DASH_PERIOD = 3
DASH_ON_STEPS = 2

OBS_THICKNESS_RANGE = (2, 4)
SPAN_MIN = 7
SPAN_MAX = 16

FIRST_SEG_MIN = 4
CHAIN_SEG_MIN = 4
FINAL_SEG_MIN = 4
MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

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
        x = int(np.where(interior == color)[1][0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _turn_to_target(side: int, target: int) -> str:
    if target == (side - 1) % 4:
        return "L"
    if target == (side + 1) % 4:
        return "R"
    raise AssertionError("turn target must be adjacent")


def _turn_anchor(r: int, c: int, old_w: int, new_w: int, hit_side: int) -> tuple[int, int]:
    if hit_side == _N:
        return r, c
    if hit_side == _S:
        return r + old_w - new_w, c
    if hit_side == _W:
        return r, c
    return r, c + old_w - new_w


def _post_turn_interval(
    side: int, width: int, thickness: dict[int, int], h: int, w: int
) -> tuple[int, int]:
    if side == _N:
        return thickness[_N], thickness[_N] + width - 1
    if side == _S:
        return h - thickness[_S] - width, h - thickness[_S] - 1
    if side == _W:
        return thickness[_W], thickness[_W] + width - 1
    return w - thickness[_E] - width, w - thickness[_E] - 1


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=False):
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


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(0, req1 - length + 1)
        start1 = min(req0, axis_len - length)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
    compact = [(start, end) for length, start, end in choices if length <= cutoff]
    pool = compact if compact and random.random() < 0.65 else [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _paint_anchor(out: np.ndarray, bg: int, r: int, c: int, width: int) -> None:
    blk = out[r : r + width, c : c + width]
    blk[blk == bg] = TEAL


def _paint_leading_slice(out: np.ndarray, bg: int, r: int, c: int, width: int, d: int) -> None:
    if d == _N:
        sl = out[r, c : c + width]
        sl[sl == bg] = TEAL
        return
    if d == _S:
        sl = out[r + width - 1, c : c + width]
        sl[sl == bg] = TEAL
        return
    if d == _W:
        sl = out[r : r + width, c]
        sl[sl == bg] = TEAL
        return
    sl = out[r : r + width, c + width - 1]
    sl[sl == bg] = TEAL


def _trace_beams(
    base: np.ndarray,
    legend_turn: dict[int, str],
    *,
    dashed: bool,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    assert sz == SOURCE_SIZE, "this combo uses a fixed 2x2 teal source"

    out = base.copy()
    h, w = out.shape
    beam_meta: list[dict[str, object]] = []

    def in_bounds(r: int, c: int, width: int) -> bool:
        return 0 <= r and 0 <= c and r + width - 1 < h and c + width - 1 < w

    def obstacle_color_at_next(nr: int, nc: int, width: int) -> int | None:
        blk = out[nr : nr + width, nc : nc + width]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(TEAL)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d, width = sr, sc, d0, INITIAL_WIDTH
        seg_len = 0
        segment_lengths: list[int] = []
        widths_after_hits: list[int] = []
        blank_steps = 0
        seen: set[tuple[int, int, int, int, int]] = set()
        while True:
            state = (r, c, d, width, seg_len)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc, width):
                break

            oc = obstacle_color_at_next(nr, nc, width)
            if oc is None:
                r, c = nr, nc
                seg_len += 1
                on_step = (seg_len - 1) % DASH_PERIOD < DASH_ON_STEPS
                if (not dashed) or on_step:
                    _paint_leading_slice(out, bg, r, c, width, d)
                else:
                    blank_steps += 1
                continue

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            assert seg_len > 0, "each segment must travel before turning"
            segment_lengths.append(seg_len)
            new_width = width + 1
            widths_after_hits.append(new_width)
            incoming_side = d
            d = _rot(d, legend_turn[oc])
            r, c = _turn_anchor(r, c, width, new_width, hit_side=incoming_side)
            assert in_bounds(r, c, new_width), "beam must remain in bounds after widening"
            width = new_width
            seg_len = 0
            _paint_anchor(out, bg, r, c, width)

        beam_meta.append(
            {
                "segment_lengths": segment_lengths,
                "terminal_length": seg_len,
                "widths_after_hits": widths_after_hits,
                "blank_steps": blank_steps,
            }
        )

    return (out, beam_meta) if return_meta else out


def _simulate_combo(
    base: np.ndarray, legend_turn: dict[int, str], *, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    return _trace_beams(base, legend_turn, dashed=True, return_meta=return_meta)


def _simulate_solid_variable_width(base: np.ndarray, legend_turn: dict[int, str]) -> np.ndarray:
    return _trace_beams(base, legend_turn, dashed=False, return_meta=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot four teal rays whose later segments become thicker and dashed after turns.

    Input structure:
    - The top band is a grey legend split into four cells; each cell contains one colored vertical
      bar, and the bar's left/right placement still means counterclockwise/clockwise turning.
    - Under the legend sits the main board: a uniform background, one solid 2x2 teal source square,
      and four colored border obstacles whose colors are listed in the legend.

    Rule:
    - Remove the legend in the output.
    - Launch one ray from the teal square in each cardinal direction. The first segment is width 2.
    - Rays advance using square footprints for collision checks. While moving through empty space,
      they paint a dashed trail with a fixed 2-on-1-off rhythm. When a ray meets an obstacle color,
      it turns according to the legend, its next segment width increases by 1, and the widened
      corner anchor is painted before that next dashed segment continues. Rays stop at the grid edge.

    Key insight:
    - The legend still explains only the turning direction. The new work is purely geometric:
      follow the repeated dash cadence along each segment and count turns to know that widths progress
      from 2 to 3 to 4 as the beam keeps bouncing.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must occupy the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    return _simulate_combo(base, legend_turn, return_meta=False)


def _second_segment_length(chain_src: int, chain_tgt: int, sr: int, sc: int, thickness: dict[int, int], h: int, w: int) -> int:
    if chain_src == _N:
        return (w - thickness[_E] - 3 - sc) if chain_tgt == _E else (sc - thickness[_W])
    if chain_src == _S:
        return (w - thickness[_E] - 3 - sc) if chain_tgt == _E else (sc - thickness[_W])
    if chain_src == _W:
        return (sr - thickness[_N]) if chain_tgt == _N else (h - thickness[_S] - 3 - sr)
    return (sr - thickness[_N]) if chain_tgt == _N else (h - thickness[_S] - 3 - sr)


def _pick_source(
    h: int, w: int, thickness: dict[int, int], chain_src: int, chain_tgt: int
) -> tuple[int, int] | None:
    r_lo = thickness[_N] + FIRST_SEG_MIN
    r_hi = h - thickness[_S] - SOURCE_SIZE - FIRST_SEG_MIN
    c_lo = thickness[_W] + FIRST_SEG_MIN
    c_hi = w - thickness[_E] - SOURCE_SIZE - FIRST_SEG_MIN
    if r_lo > r_hi or c_lo > c_hi:
        return None

    candidates: list[tuple[int, int]] = []
    for sr in range(r_lo, r_hi + 1):
        for sc in range(c_lo, c_hi + 1):
            chain_len = _second_segment_length(chain_src, chain_tgt, sr, sc, thickness, h, w)
            if chain_len < CHAIN_SEG_MIN:
                continue
            candidates.append((sr, sc))
    if not candidates:
        return None

    # Bias away from perfect centering so turned segments vary visibly in length.
    def score(pos: tuple[int, int]) -> int:
        sr, sc = pos
        return abs(sr - (h // 2)) + abs(sc - (w // 2))

    candidates.sort(key=score, reverse=True)
    pool = candidates[: max(1, len(candidates) // 2)]
    return random.choice(pool)


def _preferred_target(side: int, sr: int, sc: int, h: int, w: int) -> int:
    if side in (_N, _S):
        return _W if sc >= w - 3 - sc else _E
    return _N if sr >= h - 3 - sr else _S


def _other_adjacent(side: int, blocked_adjacent: int) -> int:
    for target in ((side - 1) % 4, (side + 1) % 4):
        if target != blocked_adjacent:
            return target
    raise AssertionError("adjacent target must exist")


def _build_side_turns(
    chain_src: int, chain_tgt: int, sr: int, sc: int, h: int, w: int
) -> dict[int, str]:
    side_turn: dict[int, str] = {}
    side_turn[chain_src] = _turn_to_target(chain_src, chain_tgt)
    side_turn[chain_tgt] = _turn_to_target(chain_tgt, _other_adjacent(chain_tgt, chain_src))

    for side in (_N, _E, _S, _W):
        if side in side_turn:
            continue
        target = _preferred_target(side, sr, sc, h, w)
        side_turn[side] = _turn_to_target(side, target)
    return side_turn


def _compute_span_constraints(
    side_turn: dict[int, str],
    chain_src: int,
    chain_tgt: int,
    sr: int,
    sc: int,
    thickness: dict[int, int],
    h: int,
    w: int,
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    required = {side: [] for side in (_N, _E, _S, _W)}
    excluded = {side: [] for side in (_N, _E, _S, _W)}

    for side in (_N, _S):
        required[side].append((sc, sc + SOURCE_SIZE - 1))
    for side in (_W, _E):
        required[side].append((sr, sr + SOURCE_SIZE - 1))

    for side in (_N, _E, _S, _W):
        target = _rot(side, side_turn[side])
        interval = _post_turn_interval(side, 3, thickness, h, w)
        if side == chain_src and target == chain_tgt:
            required[target].append(interval)
        else:
            excluded[target].append(interval)

    third_target = _rot(chain_tgt, side_turn[chain_tgt])
    excluded[third_target].append(_post_turn_interval(chain_tgt, 4, thickness, h, w))
    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    thickness = {side: random.randint(*OBS_THICKNESS_RANGE) for side in (_N, _E, _S, _W)}

    chain_src = random.choice((_N, _E, _S, _W))
    chain_tgt = random.choice(((chain_src - 1) % 4, (chain_src + 1) % 4))

    src = _pick_source(h, w, thickness, chain_src, chain_tgt)
    if src is None:
        return None
    sr, sc = src

    side_turn = _build_side_turns(chain_src, chain_tgt, sr, sc, h, w)
    required, excluded = _compute_span_constraints(
        side_turn, chain_src, chain_tgt, sr, sc, thickness, h, w
    )

    spans: dict[int, tuple[int, int]] = {}
    for side in (_N, _E, _S, _W):
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(axis_len, required[side], excluded[side], SPAN_MIN, min(SPAN_MAX, axis_len))
        if span is None:
            return None
        spans[side] = span

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_colors = dict(zip((_N, _E, _S, _W), random.sample(legend_colors, 4), strict=False))
    turns = {side_colors[side]: side_turn[side] for side in (_N, _E, _S, _W)}
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + SOURCE_SIZE, sc : sc + SOURCE_SIZE] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    for side in (_N, _E, _S, _W):
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + SOURCE_SIZE or c1 < sc - 1 or c0 > sc + SOURCE_SIZE):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_colors[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])

    try:
        expected, meta = _simulate_combo(base, turns, return_meta=True)
        solid = _simulate_solid_variable_width(base, turns)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, solid):
        return None

    hit_counts = [len(m["segment_lengths"]) for m in meta]
    if sorted(hit_counts) != [1, 1, 1, 2]:
        return None

    first_lengths = [int(m["segment_lengths"][0]) for m in meta]
    if min(first_lengths) < FIRST_SEG_MIN:
        return None

    chained = [m for m in meta if len(m["segment_lengths"]) == 2]
    if len(chained) != 1:
        return None
    chain_meta = chained[0]
    if int(chain_meta["segment_lengths"][1]) < CHAIN_SEG_MIN:
        return None
    if int(chain_meta["terminal_length"]) < FINAL_SEG_MIN:
        return None

    widths_seen = {int(w1) for m in meta for w1 in m["widths_after_hits"]}
    if widths_seen != {3, 4}:
        return None

    if any(int(m["blank_steps"]) == 0 for m in meta):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
