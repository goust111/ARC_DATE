# Variant of e87109e9
# Dimension: D9 — 射线宽度 (trajectory)
# Change: 射线宽度为2像素（与源方块2x2的边长一致），射线占据2行或2列... → 射线宽度可变（由段长决定）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior
#    - The legend keeps its original role: each obstacle color still means turn-left or turn-right.
#    - The teal source is fixed to a 2x2 square so the initial beam width is always visibly 2.
#    - Each beam segment is painted as the union of sliding w x w teal blocks, where w is the
#      current beam width. Before the first hit, w=2. After every obstacle hit, the width changes
#      according to the length of the segment that just ended:
#         short segment (< 6 steps)  -> next width = max(1, w-1)
#         long  segment (>= 6 steps) -> next width = w+1
#    - Turn example:
#         original: a northward 2-wide beam hits the top obstacle and continues sideways as 2-wide.
#         variant:  if that northward segment had length 4, the sideways continuation is 1-wide;
#                   if it had length 8, the sideways continuation is 3-wide.
#      The difference is therefore visible only after the turn, which makes "segment length"
#      the control signal for the next segment's thickness.
#
# b) Complexity calibration
#    - The width rule uses only two visible categories, short vs long, with a fixed threshold 6.
#      This keeps the rule learnable from a few examples instead of creating many hard-to-infer bins.
#    - The generator guarantees both outcomes in a sample set: some beams shrink to width 1 and
#      others grow to width 3. That prevents degeneration back to the original constant-width task.
#
# c) Hidden-assumption audit against the original main()
#    - Original in_bounds() assumed beam width == source size forever. That is no longer valid:
#      every bounds check must use the current width.
#    - Original paint_block() always painted sz x sz. Here painting must use the current width,
#      otherwise widened or narrowed segments would be lost.
#    - Original obstacle_color_at_next() inspected a fixed-size next footprint. Here the next
#      footprint size depends on the current width, so obstacle detection must track w as well.
#    - Original cycle detection used only (r,c,d). With variable width, the same anchor and
#      direction at different widths are different states, so width must be part of the state.
#    - Original turns changed only direction. Here turns also reset the segment-length counter,
#      update width, and possibly shift the anchor for south/east bounces so the beam remains
#      flush against the wall it just bounced from.
#
# d) Placement legality
#    - Obstacles remain solid edge-touching rectangles. This keeps obstacle detection unambiguous.
#    - The generator always places four obstacles, one on each side, and fixes the source far enough
#      from every obstacle that the first segment length is at least 3.
#    - Each obstacle span must cover its own incoming 2-cell launch lane, but avoid the lane of the
#      beam that turns toward it. That guarantees exactly one hit per beam and a clear post-turn strip.
#
# e) Visual observability
#    - The only new control signal is visible geometry: distance from the source to the obstacle.
#      No hidden ordering by color id or any other invisible attribute is used.
#    - Because the generator places the source off-center, two beams are short and two are long,
#      so the output visibly mixes 1-wide and 3-wide continuations.
#
# f) Visual cleanliness
#    - We intentionally avoid multi-bounce chains here. The variant is about width change, and
#      one-turn beams keep the board readable while still forcing the solver to track width.
#    - Obstacles still sit on the borders, so widened strips mostly expand inward instead of
#      tangling through the center.
#
# g) Color inference
#    - Colors still only encode turn direction through the legend, exactly as in the original task.
#      Width changes do not depend on color at all, only on segment length.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (20, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

SOURCE_SIZE = 2
INITIAL_WIDTH = 2
LEN_THRESHOLD = 6
SHORT_LEN_RANGE = (3, 5)
LONG_LEN_MIN = 7

OBS_THICKNESS_RANGE = (2, 4)
SPAN_MIN = 7
SPAN_MAX = 15
MAX_RETRIES = 400

TURN_PATTERNS = [
    {_N: "L", _E: "L", _S: "L", _W: "L"},
    {_N: "R", _E: "R", _S: "R", _W: "R"},
    {_N: "R", _E: "L", _S: "R", _W: "L"},
    {_N: "L", _E: "R", _S: "L", _W: "R"},
]


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
        col = int(next(iter(colors)))
        x = int(np.where(interior == col)[1][0])
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
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


def _updated_width(width: int, seg_len: int) -> int:
    return width + 1 if seg_len >= LEN_THRESHOLD else max(1, width - 1)


def _turn_anchor(r: int, c: int, old_w: int, new_w: int, hit_side: int) -> tuple[int, int]:
    if hit_side == _N:
        return r, c
    if hit_side == _S:
        return r + old_w - new_w, c
    if hit_side == _W:
        return r, c
    return r, c + old_w - new_w


def _trace_beams(
    base: np.ndarray, legend_turn: dict[int, str], *, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, list[dict[str, list[int]]]]:
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    assert sz == SOURCE_SIZE, "this variant uses a 2x2 teal source"

    out = base.copy()
    h, w = out.shape
    beam_meta: list[dict[str, list[int]]] = []

    def in_bounds(r: int, c: int, width: int) -> bool:
        return 0 <= r and 0 <= c and r + width - 1 < h and c + width - 1 < w

    def paint_block(r: int, c: int, width: int) -> None:
        blk = out[r : r + width, c : c + width]
        blk[blk == bg] = TEAL
        out[r : r + width, c : c + width] = blk

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
        seg_lengths: list[int] = []
        widths_after_hits: list[int] = []
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
                paint_block(r, c, width)
                continue

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            assert seg_len > 0, "each beam must travel at least one step before turning"
            new_width = _updated_width(width, seg_len)
            seg_lengths.append(seg_len)
            widths_after_hits.append(new_width)
            hit_side = d
            d = _rot(d, legend_turn[oc])
            r, c = _turn_anchor(r, c, width, new_width, hit_side=hit_side)
            assert in_bounds(r, c, new_width), "beam must remain in bounds after width update"
            width = new_width
            seg_len = 0
            paint_block(r, c, width)

        beam_meta.append(
            {
                "segment_lengths": seg_lengths,
                "widths_after_hits": widths_after_hits,
            }
        )

    return (out, beam_meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four teal beams from a 2x2 source; each turn may thicken or thin the next segment.

    Input structure:
    - The top band is a 6x24 grey legend whose four colored bars still encode turn direction by side:
      a bar on the left half means counterclockwise, and a bar on the right half means clockwise.
    - Below the legend is the main board: one solid 2x2 teal square on a uniform background, plus
      colored border obstacles whose colors appear in the legend.

    Rule:
    - Remove the legend from the output.
    - Emit one beam from the teal square in each cardinal direction. The launch width is always 2.
    - As a beam moves, it paints teal over background cells only. When it hits an obstacle color C,
      it turns according to the legend, and the width for the following segment changes using the
      length of the segment that just ended: short (<6) segments shrink by 1, long (>=6) segments
      grow by 1. Beams stop when the next footprint would leave the board.

    Key insight:
    - The legend still tells you only how to turn; the new information comes from geometry. You must
      read how far a beam traveled before each turn, then use that visible segment length to decide
      whether the next segment becomes narrower or wider.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must occupy the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    return _trace_beams(base, legend_turn, return_meta=False)


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


def _choose_span(
    axis_len: int,
    required: tuple[int, int],
    excluded: tuple[int, int],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    req0, req1 = required
    ex0, ex1 = excluded
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(0, req1 - length + 1)
        start1 = min(req0, axis_len - length)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if end < ex0 or start > ex1:
                choices.append((start, end))
    return random.choice(choices) if choices else None


def _axis_lengths(available: int) -> tuple[int, int] | None:
    short_candidates = list(range(SHORT_LEN_RANGE[0], SHORT_LEN_RANGE[1] + 1))
    random.shuffle(short_candidates)
    for short_len in short_candidates:
        long_len = available - short_len
        if long_len >= LONG_LEN_MIN:
            if random.random() < 0.5:
                return short_len, long_len
            return long_len, short_len
    return None


def _post_turn_interval(side: int, width: int, thickness: dict[int, int], h: int, w: int) -> tuple[int, int]:
    if side == _N:
        return thickness[_N], thickness[_N] + width - 1
    if side == _S:
        return h - thickness[_S] - width, h - thickness[_S] - 1
    if side == _W:
        return thickness[_W], thickness[_W] + width - 1
    return w - thickness[_E] - width, w - thickness[_E] - 1


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    thickness = {side: random.randint(*OBS_THICKNESS_RANGE) for side in (_N, _E, _S, _W)}

    row_available = h - thickness[_N] - thickness[_S] - SOURCE_SIZE
    col_available = w - thickness[_W] - thickness[_E] - SOURCE_SIZE
    row_lengths = _axis_lengths(row_available)
    col_lengths = _axis_lengths(col_available)
    if row_lengths is None or col_lengths is None:
        return None

    north_len, south_len = row_lengths
    west_len, east_len = col_lengths
    sr = thickness[_N] + north_len
    sc = thickness[_W] + west_len

    side_turn = random.choice(TURN_PATTERNS)
    incoming: dict[int, int] = {}
    for side in (_N, _E, _S, _W):
        incoming[_rot(side, side_turn[side])] = side
    assert len(incoming) == 4, "turn pattern must be a permutation"

    first_len = {
        _N: north_len,
        _S: south_len,
        _W: west_len,
        _E: east_len,
    }
    next_width = {side: _updated_width(INITIAL_WIDTH, first_len[side]) for side in first_len}

    required = {
        _N: (sc, sc + SOURCE_SIZE - 1),
        _S: (sc, sc + SOURCE_SIZE - 1),
        _W: (sr, sr + SOURCE_SIZE - 1),
        _E: (sr, sr + SOURCE_SIZE - 1),
    }
    excluded = {
        side: _post_turn_interval(incoming[side], next_width[incoming[side]], thickness, h, w)
        for side in (_N, _E, _S, _W)
    }

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
        expected, meta = _trace_beams(base, turns, return_meta=True)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    hit_counts = [len(m["segment_lengths"]) for m in meta]
    hit_widths = [w1 for m in meta for w1 in m["widths_after_hits"]]
    if hit_counts != [1, 1, 1, 1]:
        return None
    if set(hit_widths) != {1, 3}:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 D9 variant input")
