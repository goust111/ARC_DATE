# Combo variant of e12f9a14 (index 12)
# combo_spec: D11/2+D14/0
# Name: 链式传递+按距离排序

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_behavior_trace:
#    - Keep the same visible objects as the base task: square grids, one shared frame color, 2x2 monochrome
#      inner blocks, and bg-colored structural gaps on 4x4 frame borders.
#    - The fused change removes the original "all gaps fire immediately" model. Instead, the frames form one
#      visible relay chain whose activation order is determined by center distance, not by colors or ids.
#    - Every generated frame has exactly 2 non-adjacent gaps, so there is no visible 1-gap source marker.
#      The closest frame to the grid center auto-starts. It emits from both of its gaps in its own color:
#      one ray is a side tail to the boundary, and the other ray reaches the entry gap of the second frame.
#    - Every later frame uses one gap as an entry and the other as an exit. When the predecessor ray lands on
#      the entry gap, that frame becomes active and emits only from its other gap in its own color.
#    - Small example: suppose three frames A, B, C are ordered by distance-to-center. In the base task,
#      all gaps of A/B/C would become seeds at once. Here, A fires first; one A-colored segment stops on B's
#      entry gap; only then does B emit its B-colored segment to C; only then does C emit its tail.
#      The observable difference is that the entry gap of each relay is painted by the predecessor color, and
#      there are no out-of-order beams from farther frames before the chain reaches them.
#
# b) complexity_calibration:
#    - If the source had a unique gap count, the distance-order rule would become unnecessary. To avoid that
#      degeneration, all generated frames use the same 2-gap vocabulary.
#    - A free interaction graph would be too hard to infer from 2-4 ARC examples, so the generator restricts
#      the layout to one directed chain. That preserves the multi-step reasoning while keeping the structure
#      human-recoverable.
#    - Block count is biased toward 3-4 so the chain and the center-order cue are usually visible, but the
#      search space stays small enough for reliable generation.
#
# c) implicit_assumptions_review:
#    - Original assumption: every structural gap is an immediate beam seed. Broken by the combo; now only the
#      nearest frame auto-starts, and later frames wait for activation.
#    - Original assumption: beam interactions are one global synchronous process. Broken again; this variant
#      uses one active relay step at a time, so older trails are fixed history rather than simultaneous peers.
#    - New solver assumption: each frame has exactly 2 structural gaps, the distance order is unique, and
#      consecutive ordered frames are connected by exactly one unobstructed ray-to-gap link. The generator
#      enforces all three properties and rejects ambiguous cases.
#
# d) placement_legality:
#    - Frames still occupy 4x4 bounding boxes around 2x2 blocks and remain separated by at least 2 cells.
#    - Consecutive chain links are placed on one of the 8 legal ray directions with a minimum span so that
#      the next frame's entry gap is reached exactly, without crossing any frame cell or previously used trail.
#    - The source tail and last tail are also checked against all frames, all other gaps, and all chain cells.
#
# e) visual_observability:
#    - Sorting depends only on visible geometry: squared distance from frame center to grid center.
#      No color id, palette order, or hidden numbering participates.
#    - Because every frame has 2 gaps, the auto-start frame is not explicitly marked. The intended source can
#      only be recovered by noticing which frame is closest to the center.
#    - The generator rejects samples where a simple row-major ordering would behave the same, so the distance
#      cue materially changes the execution.
#
# f) visual_tidiness:
#    - The chain is kept sparse: one incoming link and one outgoing/tail gap per frame, with no segment overlap.
#    - Rays stop on frame gaps instead of passing through, so color changes happen cleanly at frame boundaries.
#    - Source and terminal tails are short, readable side effects rather than a dense web of extra rays.
#
# g) change_inferability:
#    - Colors retain the original meaning: every emitted segment uses the inner block color of the frame that
#      emitted it. The only new inference burden is the activation order, and that order is paired with stable
#      geometric evidence: center distance plus the chain of aligned entry/exit gaps.
#
# solver_generator_contract:
#    - main() assumes at least 2 frames, exactly 2 gaps per frame, a unique center-distance order, and one
#      unique consecutive link between every adjacent pair in that order.
#    - generate_input() constructs only such layouts, validates them with main(), and also checks that using a
#      row-major fallback order would either fail or produce a different output.


_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_DIR_TO_LOCAL_GAPS = {
    (-1, -1): [(0, 0)],
    (-1, 0): [(0, 1), (0, 2)],
    (-1, 1): [(0, 3)],
    (0, 1): [(1, 3), (2, 3)],
    (1, 1): [(3, 3)],
    (1, 0): [(3, 2), (3, 1)],
    (1, -1): [(3, 0)],
    (0, -1): [(2, 0), (1, 0)],
}


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, bc + 1) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _gap_dir(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _scale(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _frame_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(4) for dc in range(4)}


def _frames_separated(a: tuple[int, int], b: tuple[int, int], gap: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _non_adjacent_local(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ia = _BORDER_INDEX[a]
    ib = _BORDER_INDEX[b]
    d = (ib - ia) % len(_BORDER_ORDER)
    return d not in (1, len(_BORDER_ORDER) - 1)


def _frame_center_dist2(r0: int, c0: int, h: int, w: int) -> int:
    dr2 = 2 * (r0 + 1) + 1 - (h - 1)
    dc2 = 2 * (c0 + 1) + 1 - (w - 1)
    return int(dr2 * dr2 + dc2 * dc2)


_PAIR_OPTIONS: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
for din, in_gaps in _DIR_TO_LOCAL_GAPS.items():
    for dout, out_gaps in _DIR_TO_LOCAL_GAPS.items():
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for gin in in_gaps:
            for gout in out_gaps:
                if gin != gout and _non_adjacent_local(gin, gout):
                    pairs.append((gin, gout))
        _PAIR_OPTIONS[(din, dout)] = pairs

_SECOND_GAPS: dict[tuple[int, int], list[tuple[int, int]]] = {
    gap: [other for other in _BORDER_ORDER if other != gap and _non_adjacent_local(gap, other)]
    for gap in _BORDER_ORDER
}


def _build_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    frames: list[dict[str, object]] = []
    blocks = _find_blocks_2x2(g, bg)
    assert len(blocks) >= 2, "expected at least two framed blocks"
    for color, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, color)
        abs_gaps = sorted(_frame_gaps_for_block(g, bg, frame_col, br, bc))
        assert len(abs_gaps) == 2, "combo variant expects exactly two structural gaps per frame"
        r0, c0 = br - 1, bc - 1
        local_gaps = [(gr - r0, gc - c0) for gr, gc in abs_gaps]
        frames.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "gaps": list(abs_gaps),
                "dirs": [_gap_dir(gap) for gap in local_gaps],
                "dist2": _frame_center_dist2(r0, c0, h, w),
            }
        )
    return frames


def _ray_hits_target_gap(
    g: np.ndarray,
    bg: int,
    gap_lookup: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    target: tuple[int, int],
) -> bool:
    h, w = g.shape
    cur = start
    while True:
        nxt = (cur[0] + direction[0], cur[1] + direction[1])
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            return False
        owner = gap_lookup.get(nxt)
        if owner is not None:
            return nxt == target
        if int(g[nxt[0], nxt[1]]) != bg:
            return False
        cur = nxt


def _infer_plan(
    g: np.ndarray,
    bg: int,
    frames: list[dict[str, object]],
    order: list[int],
) -> dict[str, object]:
    gap_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for fi, frame in enumerate(frames):
        for gi, gap in enumerate(frame["gaps"]):
            assert gap not in gap_lookup, "gap positions must be unique"
            gap_lookup[gap] = (fi, gi)

    dists = [int(frames[i]["dist2"]) for i in order]
    assert len(set(dists)) == len(dists), "expected unique center-distance order"

    links: list[tuple[int, int]] = []
    for idx in range(len(order) - 1):
        a = order[idx]
        b = order[idx + 1]
        candidates: list[tuple[int, int]] = []
        for ga, start in enumerate(frames[a]["gaps"]):
            direction = tuple(frames[a]["dirs"][ga])
            for gb, target in enumerate(frames[b]["gaps"]):
                if _ray_hits_target_gap(g, bg, gap_lookup, tuple(start), direction, tuple(target)):
                    candidates.append((ga, gb))
        assert len(candidates) == 1, "expected exactly one link between consecutive ordered frames"
        links.append(candidates[0])

    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    for idx, (ga, gb) in enumerate(links):
        a = order[idx]
        b = order[idx + 1]
        outgoing[a] = ga
        incoming[b] = gb

    source = order[0]
    tail_source = 1 - outgoing[source]
    assert tail_source in (0, 1), "expected a second source gap"

    for mid in order[1:-1]:
        assert mid in incoming and mid in outgoing, "middle relay must have both entry and exit"
        assert incoming[mid] != outgoing[mid], "entry and exit gaps must differ"

    last = order[-1]
    assert last in incoming, "last relay must have an incoming gap"
    tail_last = 1 - incoming[last]
    assert tail_last in (0, 1), "expected a second terminal gap"

    return {
        "order": list(order),
        "links": list(links),
        "incoming": incoming,
        "outgoing": outgoing,
        "source_tail": tail_source,
        "last_tail": tail_last,
    }


def _execute_plan(g: np.ndarray, bg: int, frames: list[dict[str, object]], plan: dict[str, object]) -> np.ndarray:
    out = g.copy()
    h, w = g.shape
    gap_cells = {tuple(gap) for frame in frames for gap in frame["gaps"]}
    order = list(plan["order"])
    links = list(plan["links"])
    incoming = dict(plan["incoming"])
    outgoing = dict(plan["outgoing"])

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    def paint(cell: tuple[int, int], color: int) -> None:
        r, c = cell
        assert int(g[r, c]) == bg, "beams may only overwrite background cells"
        assert int(out[r, c]) == bg, "beam segments must stay disjoint"
        out[r, c] = int(color)

    def trace_to_gap(frame_id: int, gap_idx: int, target: tuple[int, int]) -> None:
        frame = frames[frame_id]
        color = int(frame["color"])
        start = tuple(frame["gaps"][gap_idx])
        direction = tuple(frame["dirs"][gap_idx])
        if int(out[start[0], start[1]]) == bg:
            out[start[0], start[1]] = color
        cur = start
        while True:
            nxt = (cur[0] + direction[0], cur[1] + direction[1])
            assert in_bounds(nxt), "link ray must reach the next frame before the boundary"
            if nxt == target:
                assert int(out[nxt[0], nxt[1]]) == bg, "entry gap should be untouched before activation"
                out[nxt[0], nxt[1]] = color
                return
            assert nxt not in gap_cells, "link ray must not pass through another frame gap"
            paint(nxt, color)
            cur = nxt

    def trace_to_edge(frame_id: int, gap_idx: int) -> None:
        frame = frames[frame_id]
        color = int(frame["color"])
        start = tuple(frame["gaps"][gap_idx])
        direction = tuple(frame["dirs"][gap_idx])
        if int(out[start[0], start[1]]) == bg:
            out[start[0], start[1]] = color
        cur = start
        while True:
            nxt = (cur[0] + direction[0], cur[1] + direction[1])
            if not in_bounds(nxt):
                return
            assert nxt not in gap_cells, "tail ray must not hit another frame gap"
            paint(nxt, color)
            cur = nxt

    source = order[0]
    first_link_gap, first_entry_gap = links[0]
    trace_to_gap(source, first_link_gap, tuple(frames[order[1]]["gaps"][first_entry_gap]))
    trace_to_edge(source, int(plan["source_tail"]))

    for step in range(1, len(order) - 1):
        cur = order[step]
        out_gap = outgoing[cur]
        target_frame = order[step + 1]
        target_gap = incoming[target_frame]
        trace_to_gap(cur, out_gap, tuple(frames[target_frame]["gaps"][target_gap]))

    last = order[-1]
    trace_to_edge(last, int(plan["last_tail"]))
    return out


def _solve_with_order(grid: np.ndarray, order: list[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    plan = _infer_plan(g, bg, frames, order)
    return _execute_plan(g, bg, frames, plan)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay rays between framed blocks in center-distance order, with each reached frame handing off the chain.

    Input structure:
    - The grid is square with one dominant background color. Each object is a monochrome 2x2 block wrapped by a
      shared-color 4x4 frame.
    - Every frame exposes exactly two non-adjacent structural gaps on its border, and each gap defines one of the
      8 outward ray directions.

    Rule:
    - Rank the frames by distance from their centers to the grid center, nearest first; ties are not allowed in
      valid inputs.
    - The nearest frame starts automatically and emits from both of its gaps. One ray is a side tail to the edge;
      the other lands on the next frame's entry gap and stops there.
    - Whenever a frame is reached, it becomes active and emits only from its other gap in its own block color.
      This repeats down the distance-sorted chain until the farthest frame sends its final tail to the boundary.

    Key insight:
    - Nothing in the input marks the source by gap count. All frames look alike, so the hidden control variable is
      geometric centrality: first recover the center-near-to-center-far order, then follow the chained handoffs.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["dist2"]), int(frames[i]["r0"]), int(frames[i]["c0"])))
    plan = _infer_plan(g, bg, frames, order)
    return _execute_plan(g, bg, frames, plan)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    dist_order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["dist2"]), int(frames[i]["r0"]), int(frames[i]["c0"])))
    row_order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["r0"]), int(frames[i]["c0"])))
    out = _solve_with_order(g, dist_order)
    row_valid = 1
    try:
        alt = _solve_with_order(g, row_order)
        row_diff = int(np.count_nonzero(out != alt))
    except Exception:
        row_valid = 0
        row_diff = -1
    return {
        "frame_count": len(frames),
        "source_reading_rank": row_order.index(dist_order[0]),
        "row_order_valid": row_valid,
        "row_order_diff": row_diff,
        "painted_pixels": int(np.count_nonzero(out != g)),
    }


# --------------- generator (combo chain + center order) ---------------

BLOCK_COUNT_CHOICES = [2, 3, 4]
BLOCK_COUNT_WEIGHTS = [1, 4, 3]
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 16, 3: 20, 4: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
PLACEMENT_MARGIN = 2
FRAME_BBOX_GAP = 2
MIN_LINK_SPAN = 4
MAX_LINK_SPAN = 8
MIN_TAIL_CELLS = 3
MAX_SHIFT_TRIES = 80
MAX_TAIL_TRIES = 16
GEN_RETRIES = 700


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=block_count, replace=False)))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 16
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    cells = [start]
    cur = start
    for _ in range(steps):
        cur = _add(cur, direction)
        cells.append(cur)
    return cells


def _trace_to_edge(start: tuple[int, int], direction: tuple[int, int], n: int) -> list[tuple[int, int]]:
    cells = [start]
    cur = start
    while True:
        nxt = _add(cur, direction)
        if not (0 <= nxt[0] < n and 0 <= nxt[1] < n):
            return cells
        cells.append(nxt)
        cur = nxt


def _choose_link_dirs(count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    dirs = list(_DIRS8)
    out_dirs = [dirs[int(rng.integers(len(dirs)))]]
    for _ in range(1, count):
        entry_dir = _opp(out_dirs[-1])
        choices = [d for d in dirs if _PAIR_OPTIONS[(entry_dir, d)]]
        if not choices:
            return None
        out_dirs.append(choices[int(rng.integers(len(choices)))])
    return out_dirs


def _build_relative_chain(rng: np.random.Generator, count: int) -> list[dict[str, object]] | None:
    link_dirs = _choose_link_dirs(count - 1, rng)
    if link_dirs is None:
        return None

    frames: list[dict[str, object]] = []
    first_exit = tuple(_DIR_TO_LOCAL_GAPS[link_dirs[0]][int(rng.integers(len(_DIR_TO_LOCAL_GAPS[link_dirs[0]])))])
    frames.append({"pos": (0, 0), "entry": None, "exit": first_exit})

    for i in range(1, count - 1):
        entry_dir = _opp(link_dirs[i - 1])
        exit_dir = link_dirs[i]
        pairs = _PAIR_OPTIONS[(entry_dir, exit_dir)]
        if not pairs:
            return None
        entry_gap, exit_gap = pairs[int(rng.integers(len(pairs)))]
        prev = frames[-1]
        span = int(rng.integers(MIN_LINK_SPAN, MAX_LINK_SPAN + 1))
        prev_exit = _add(tuple(prev["pos"]), tuple(prev["exit"]))
        next_pos = _sub(_add(prev_exit, _scale(link_dirs[i - 1], span)), entry_gap)
        frames.append({"pos": next_pos, "entry": entry_gap, "exit": exit_gap})

    last_entry_dir = _opp(link_dirs[-1])
    last_entry = tuple(_DIR_TO_LOCAL_GAPS[last_entry_dir][int(rng.integers(len(_DIR_TO_LOCAL_GAPS[last_entry_dir])))] )
    prev = frames[-1]
    span = int(rng.integers(MIN_LINK_SPAN, MAX_LINK_SPAN + 1))
    prev_exit = _add(tuple(prev["pos"]), tuple(prev["exit"]))
    last_pos = _sub(_add(prev_exit, _scale(link_dirs[-1], span)), last_entry)
    frames.append({"pos": last_pos, "entry": last_entry, "exit": None})

    positions = [tuple(frame["pos"]) for frame in frames]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if not _frames_separated(positions[i], positions[j], FRAME_BBOX_GAP):
                return None
    return frames


def _frame_fit_ranges(positions: list[tuple[int, int]], n: int) -> tuple[range, range] | None:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    sr_lo = PLACEMENT_MARGIN - min(rows)
    sr_hi = n - 4 - PLACEMENT_MARGIN - max(rows)
    sc_lo = PLACEMENT_MARGIN - min(cols)
    sc_hi = n - 4 - PLACEMENT_MARGIN - max(cols)
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None
    return range(sr_lo, sr_hi + 1), range(sc_lo, sc_hi + 1)


def _build_rendered_grid(
    placed: list[dict[str, object]],
    block_colors: list[int],
    bg: int,
    frame_color: int,
    source_tail: tuple[int, int],
    last_tail: tuple[int, int],
    n: int,
) -> np.ndarray | None:
    grid = np.full((n, n), bg, dtype=int)
    frame_cells: set[tuple[int, int]] = set()
    gap_cells: set[tuple[int, int]] = set()
    gap_multiplicity: dict[tuple[int, int], int] = {}

    for idx, (frame, color) in enumerate(zip(placed, block_colors)):
        gaps = []
        if idx == 0:
            gaps = [tuple(frame["exit"]), source_tail]
        elif idx == len(placed) - 1:
            gaps = [tuple(frame["entry"]), last_tail]
        else:
            gaps = [tuple(frame["entry"]), tuple(frame["exit"])]
        _draw_framed_block(grid, int(frame["pos"][0]), int(frame["pos"][1]), bg, frame_color, int(color), (gaps[0], gaps[1]))
        local_gap_cells = {_add(tuple(frame["pos"]), gap) for gap in gaps}
        frame_cells |= _frame_cells(int(frame["pos"][0]), int(frame["pos"][1]))
        gap_cells |= local_gap_cells
        for cell in local_gap_cells:
            gap_multiplicity[cell] = gap_multiplicity.get(cell, 0) + 1

    if any(v != 1 for v in gap_multiplicity.values()):
        return None

    used_cells: set[tuple[int, int]] = set()

    for i in range(len(placed) - 1):
        start = _add(tuple(placed[i]["pos"]), tuple(placed[i]["exit"]))
        target = _add(tuple(placed[i + 1]["pos"]), tuple(placed[i + 1]["entry"]))
        diff = _sub(target, start)
        direction = _gap_dir(tuple(placed[i]["exit"]))
        if direction[0] == 0:
            if diff[0] != 0 or diff[1] == 0 or diff[1] % direction[1] != 0:
                return None
            steps = diff[1] // direction[1]
        elif direction[1] == 0:
            if diff[1] != 0 or diff[0] == 0 or diff[0] % direction[0] != 0:
                return None
            steps = diff[0] // direction[0]
        else:
            if diff[0] == 0 or diff[1] == 0:
                return None
            if diff[0] % direction[0] != 0 or diff[1] % direction[1] != 0:
                return None
            steps = diff[0] // direction[0]
            if steps != diff[1] // direction[1]:
                return None
        if steps < MIN_LINK_SPAN:
            return None
        cells = _segment_cells(start, direction, steps)
        if cells[-1] != target:
            return None
        for j, cell in enumerate(cells):
            if not (0 <= cell[0] < n and 0 <= cell[1] < n):
                return None
            if j == 0:
                if cell in used_cells:
                    return None
                continue
            if j == len(cells) - 1:
                if cell in used_cells or cell not in gap_cells:
                    return None
                continue
            if cell in used_cells or cell in frame_cells or cell in gap_cells:
                return None
        used_cells.update(cells)

    source_start = _add(tuple(placed[0]["pos"]), source_tail)
    source_tail_cells = _trace_to_edge(source_start, _gap_dir(source_tail), n)
    if len(source_tail_cells) - 1 < MIN_TAIL_CELLS:
        return None
    for j, cell in enumerate(source_tail_cells):
        if j == 0:
            if cell in used_cells:
                return None
            continue
        if cell in used_cells or cell in frame_cells or cell in gap_cells:
            return None
    used_cells.update(source_tail_cells)

    last_start = _add(tuple(placed[-1]["pos"]), last_tail)
    last_tail_cells = _trace_to_edge(last_start, _gap_dir(last_tail), n)
    if len(last_tail_cells) - 1 < MIN_TAIL_CELLS:
        return None
    for j, cell in enumerate(last_tail_cells):
        if j == 0:
            if cell in used_cells:
                return None
            continue
        if cell in used_cells or cell in frame_cells or cell in gap_cells:
            return None
    used_cells.update(last_tail_cells)

    return grid


def _realize_chain(
    relative: list[dict[str, object]],
    block_colors: list[int],
    bg: int,
    frame_color: int,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    positions = [tuple(frame["pos"]) for frame in relative]
    shift_ranges = _frame_fit_ranges(positions, n)
    if shift_ranges is None:
        return None
    row_range, col_range = shift_ranges
    shifts = [(sr, sc) for sr in row_range for sc in col_range]
    if not shifts:
        return None
    rng.shuffle(shifts)

    for shift in shifts[:MAX_SHIFT_TRIES]:
        placed = [{**frame, "pos": _add(tuple(frame["pos"]), shift)} for frame in relative]
        dist_order = list(range(len(placed)))
        dists = [_frame_center_dist2(int(frame["pos"][0]), int(frame["pos"][1]), n, n) for frame in placed]
        if any(dists[i] >= dists[i + 1] for i in range(len(dists) - 1)):
            continue

        row_order = sorted(range(len(placed)), key=lambda i: (int(placed[i]["pos"][0]), int(placed[i]["pos"][1])))
        if row_order[0] == 0:
            continue

        source_exit = tuple(placed[0]["exit"])
        last_entry = tuple(placed[-1]["entry"])
        source_tail_choices = list(_SECOND_GAPS[source_exit])
        last_tail_choices = list(_SECOND_GAPS[last_entry])
        rng.shuffle(source_tail_choices)
        rng.shuffle(last_tail_choices)

        for source_tail in source_tail_choices[:MAX_TAIL_TRIES]:
            for last_tail in last_tail_choices[:MAX_TAIL_TRIES]:
                grid = _build_rendered_grid(placed, block_colors, bg, frame_color, source_tail, last_tail, n)
                if grid is None:
                    continue
                return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    block_count = int(rng.choice(BLOCK_COUNT_CHOICES, p=np.array(BLOCK_COUNT_WEIGHTS) / sum(BLOCK_COUNT_WEIGHTS)))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    relative = _build_relative_chain(rng, block_count)
    if relative is None:
        return None

    grid = _realize_chain(relative, block_colors, bg, frame_color, n, rng)
    if grid is None:
        return None

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
        metrics = variant_metrics(grid)
    except Exception:
        return None

    if metrics["source_reading_rank"] <= 0:
        return None
    if metrics["row_order_valid"] == 1 and metrics["row_order_diff"] <= 0:
        return None
    if metrics["painted_pixels"] < block_count * 5:
        return None
    if np.count_nonzero(out != grid) < block_count * 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e12f9a14 input")
