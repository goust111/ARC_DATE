# Combo variant of e12f9a14 (index 26)
# combo_spec: D3/1+D6/0
# Name: 间歇跳跃轨迹+固定步数终止

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   objects:
#     - Keep the original 2x2 monochrome blocks inside 4x4 frames.
#     - Keep visible frame gaps as the beam emitters, but generator restricts them
#       to frame corners so the output emphasizes outward diagonal dashes.
#   fused_rule:
#     - Every emitted beam still moves one grid step at a time and still uses the
#       original collision mechanics if needed.
#     - Visibility and occupancy are no longer the same thing: the beam occupies
#       every traversed cell for collision purposes, but only paints every other
#       logical step, starting with the gap cell itself.
#     - Each frame also encodes a fixed total beam length from its visible gap
#       count. To keep the dashed outputs distinguishable, the encoded total cells
#       are 3/5/7/9 for 1/2/3/4 gaps. That yields visible dash counts 2/3/4/5.
#   behavior_trace:
#     - Original: a corner gap at NW paints a solid diagonal until the boundary or
#       a collision changes direction.
#     - D3-only idea: the same beam would still move through every diagonal cell,
#       but only cells 0,2,4,... along that path would be visible.
#     - D6-only idea: the beam would stop after a frame-coded number of cells even
#       if more background remains.
#     - Combo: the beam from a 3-gap frame takes exactly 6 moves after the gap
#       (7 cells total), but only cells with logical indices 0,2,4,6 are painted.
#       The blank cells between them are true traveled cells, so they can still
#       host later collisions even though they stay background-colored.
# complexity_calibration:
#   - The length code uses only four levels, matching the original 1-4 gap-count
#     categories, so examples can repeat the same relation within one grid.
#   - Using odd total lengths avoids the D3+D6 degeneracy where consecutive fixed
#     lengths would collapse to the same visible dash count.
#   - Generator prefers corner gaps and rejects beam intersections, so the fused
#     behavior is visually obvious instead of being drowned out by collision cases.
# hidden_assumptions_checked_in_main:
#   - Original code assumed "occupied trajectory cell" always meant "painted
#     output cell". That breaks under intermittent skips, so the combo solver keeps
#     separate occupancy metadata (`cell_owner`) and visible paint state.
#   - Original truncation logic only rewound stored path cells. Here the beam also
#     tracks a logical step index for the rewritten trajectory; truncation must
#     rewind that index to the kept head while not refunding remaining distance.
#   - Original fixed-length variant only needed remaining budget. The combo also
#     needs visibility parity to advance on collision steps that consume distance
#     but do not leave a painted cell.
# placement_legality:
#   - Generator still places 4x4 frames with clearance and validates in-bounds
#     travel.
#   - It now checks only the encoded number of steps plus a post-stop buffer, and
#     it rejects any chosen set of beams whose planned straight trajectories would
#     share cells or geometrically cross.
# observability_and_cleanliness:
#   - The visible effect is a diagonal dashed ray that stops in open space instead
#     of at the border; both parts are readable from raw pixels.
#   - No invisible property depends on raw color ids. The only code is the visible
#     number of frame openings, and that same count repeats across all beams from a
#     frame.
# new_main_assumptions_and_generator_guarantees:
#   - `_frame_gaps_for_block()` remains the solver's source of truth for counting
#     structural gaps, and the generator uses the same 4x4 border model.
#   - `generate_input()` guarantees all emitted beams have their full encoded
#     runway plus one extra empty cell after the stop point, and it rejects
#     straight-path collisions so the intended fused rule is observable.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}
_BEAM_LEN_BY_GAP_COUNT = {1: 3, 2: 5, 3: 7, 4: 9}


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
    """Find the unique non-bg color on the 4x4 border around a 2x2 block."""
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
        ok = True
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


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _beam_length_from_gap_count(gap_count: int) -> int:
    assert gap_count in _BEAM_LEN_BY_GAP_COUNT, "expected 1-4 structural frame gaps"
    return _BEAM_LEN_BY_GAP_COUNT[gap_count]


def _is_visible_step(step_idx: int) -> bool:
    return step_idx % 2 == 0


class _Beam:
    __slots__ = (
        "bid",
        "color",
        "pos",
        "pattern",
        "phase",
        "path",
        "path_steps",
        "enter_dirs",
        "active",
        "remaining",
        "trace_step",
    )

    def __init__(
        self,
        bid: int,
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        remaining: int,
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.path_steps = [0]
        self.enter_dirs = [pattern[0]]
        self.active = remaining > 0
        self.remaining = int(remaining)
        self.trace_step = 0

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit frame-launched beams whose visible trails are dashed and whose total travel is frame-encoded.

    Input structure:
    - The dominant color is the background. Each object is a 2x2 monochrome block enclosed by a shared-color 4x4 frame.
    - Frames contain 1-4 structural gaps. Each gap is a launch point, and the number of gaps on that frame encodes the beam's total allowed length.

    Rule:
    - Every gap emits a beam in the outward direction implied by its border position, using the inner block color.
    - A beam still moves cell by cell and still participates in the original collision rules, but it paints only every other logical step; skipped cells stay background-colored while remaining part of the occupied trajectory.
    - Beam length is fixed by frame gap count: 1/2/3/4 gaps map to total traversed cell counts 3/5/7/9 including the gap cell.
    - When a collision occurs before the budget is exhausted, preserve the original redirection mechanics while continuing to respect both the fixed remaining distance and the dashed visibility parity.

    Key insight: Count the frame openings first, then follow an invisible-visible-invisible-visible cadence along the emitted path; the dash spacing shows that beams pass through blank cells, and the mid-air stop shows that total travel is capped by the frame code rather than by the border.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    cell_visible: dict[tuple[int, int], bool] = {}

    def refresh_active(bid: int) -> None:
        beams[bid].active = beams[bid].remaining > 0

    def advance_trace(bid: int, spend_budget: bool) -> None:
        b = beams[bid]
        if spend_budget:
            assert b.remaining > 0, "beam step budget underflow"
            b.remaining -= 1
        b.trace_step += 1
        refresh_active(bid)

    def set_pattern(bid: int, pattern: tuple[tuple[int, int], ...]) -> None:
        beams[bid].pattern = pattern
        beams[bid].phase = 0
        refresh_active(bid)

    def occupy_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int], visible: bool) -> None:
        r, c = cell
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1
        cell_visible[cell] = visible
        out[r, c] = beams[bid].color if visible else bg

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        out[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)
        cell_visible.pop(cell, None)

    def append_head_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        b = beams[bid]
        b.path.append(cell)
        b.path_steps.append(b.trace_step)
        b.enter_dirs.append(enter_dir)
        b.pos = cell
        occupy_cell(bid, cell, enter_dir, _is_visible_step(b.trace_step))

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.path_steps = b.path_steps[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.trace_step = b.path_steps[-1]
        refresh_active(bid)

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        total_cells: int,
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, remaining=total_cells - 1)
        occupy_cell(bid, pos, pattern[0], True)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        total_cells = _beam_length_from_gap_count(len(gaps))
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), total_cells)

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        b = beams.get(bid)
        if b is None or not b.active:
            return None
        d = b.next_dir()
        partner: int | None = None
        for oid, ob in beams.items():
            if oid == bid or not ob.active:
                continue
            if ob.pattern != b.pattern or ob.phase != b.phase:
                continue
            if ob.next_dir() != d:
                continue
            if abs(ob.pos[0] - b.pos[0]) + abs(ob.pos[1] - b.pos[1]) != 1:
                continue
            if (b.pos[0] + d[0], b.pos[1] + d[1]) == ob.pos:
                continue
            if (ob.pos[0] + d[0], ob.pos[1] + d[1]) == b.pos:
                continue
            assert partner is None, "expected at most one adjacent lockstep partner"
            partner = oid
        return partner

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        a_consumes = {a}
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
            a_consumes.add(b)
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide on one cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            for gid in sorted(a_consumes):
                advance_trace(gid, spend_budget=True)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = {a, b}
            pa = find_partner(a)
            pb = find_partner(b)
            if pa is not None:
                group.add(pa)
            if pb is not None:
                group.add(pb)
            for gid in group:
                set_pattern(gid, (nd,))
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must involve exactly one cardinal and one diagonal beam"

        if is_card_a:
            u, v = da, db
            bid_u, bid_v = a, b
        else:
            u, v = db, da
            bid_u, bid_v = b, a

        for gid in sorted(a_consumes):
            advance_trace(gid, spend_budget=True)

        if cell_owner.get(cell) != bid_u:
            if cell_owner.get(cell) == bid_v:
                idx = cell_index[cell]
                truncate_beam(bid_v, idx)
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                idx = cell_index[cell]
                truncate_beam(bid_u, idx + 1)
            else:
                append_head_cell(bid_u, cell, u)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)

        if bid_v not in a_consumes:
            advance_trace(bid_v, spend_budget=False)
        bv = beams[bid_v]
        adj = (bv.pos[0] + u[0], bv.pos[1] + u[1])
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
        append_head_cell(bid_v, adj, u)

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell after 45-degree resolution"

        zig = (v, u)
        set_pattern(bid_u, zig)
        set_pattern(bid_v, zig)

    def diag_segment_owner(p: tuple[int, int], q: tuple[int, int]) -> int | None:
        dr, dc = q[0] - p[0], q[1] - p[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        oq = cell_owner.get(q)
        if oq is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return oq
        op = cell_owner.get(p)
        if op is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return op
        return None

    def resolve_cross_collision(a: int, b: int) -> None:
        da = beams[a].next_dir()
        db = beams[b].next_dir()
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        advance_trace(a, spend_budget=True)
        advance_trace(b, spend_budget=True)
        set_pattern(a, (u,))
        set_pattern(b, (u,))

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        seg_dir: tuple[int, int] | None = None
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert u != (0, 0), "expected segment crossing to resolve to a forward direction"

        advance_trace(a, spend_budget=True)
        group = {a, b}
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if pb is not None:
            group.add(pb)
        for gid in group:
            set_pattern(gid, (u,))

    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            if b.remaining <= 0:
                refresh_active(bid)
                continue
            d = b.next_dir()
            nr, nc = b.pos[0] + d[0], b.pos[1] + d[1]
            nxt = (nr, nc)
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit block/frame"
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            move_map[(beams[bid].pos, nxt)] = bid

        for (p, n), bid in list(move_map.items()):
            pr, pc = p
            nr, nc = n
            dr = nr - pr
            dc = nc - pc
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            obid = move_map.get((p2, n2))
            if obid is None or obid == bid:
                continue
            a, b = sorted((bid, obid))
            cross_pairs.add((a, b))

        seg_cross: set[tuple[int, int]] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            dr, dc = d
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            obid = diag_segment_owner(p2, n2)
            if obid is None or obid == bid:
                continue
            seg_cross.add((bid, obid))

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int, int, str]] = []
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            assert bidders, "expected each target cell to have at least one bidder"
            bidders_sorted = sorted(bidders)
            a = bidders_sorted[0]
            cand_bs: list[int] = []
            if owner is not None and owner != a:
                cand_bs.append(owner)
            for bb in bidders_sorted[1:]:
                if bb != a:
                    cand_bs.append(bb)
                    break
            if not cand_bs:
                continue
            b = cand_bs[0]
            events.append((0, cell[0], cell[1], a, b, "cell"))
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            keyr = min(ca[0], cb[0])
            keyc = min(ca[1], cb[1])
            events.append((1, keyr, keyc, a, b, "cross"))
        for a, b in seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            keyr = min(pr, pr + dr)
            keyc = min(pc, pc + dc)
            events.append((1, keyr, keyc, a, b, "segcross"))

        if events:
            events.sort()
            _, r, c, a, b, kind = events[0]
            if kind == "cell":
                resolve_cell_collision((r, c), a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision(a, b, prop)
            else:
                resolve_cross_collision(a, b)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            advance_trace(bid, spend_budget=True)
            append_head_cell(bid, nxt, d)
            if beams[bid].remaining > 0:
                beams[bid].advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    return out


# --------------- generator (combo) ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 14, 2: 18, 5: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_POST_STOP_BUFFER = 1
MIN_BEAM_PIXELS_PER_GAP = 2
GEN_RETRIES = 300

_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))
_GAP_COMBOS = {n: list(itertools.combinations(_CORNER_GAPS, n)) for n in range(1, 5)}


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _ray_cells(
    seed: tuple[int, int],
    direction: tuple[int, int],
    total_cells: int,
) -> list[tuple[int, int]]:
    r, c = seed
    dr, dc = direction
    cells = [seed]
    for _ in range(total_cells - 1):
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _rays_clear(
    grid: np.ndarray,
    bg: int,
    seeds: list[tuple[tuple[int, int], tuple[int, int], int]],
) -> bool:
    h, w = grid.shape
    for (r, c), (dr, dc), steps in seeds:
        nr, nc = r, c
        for _ in range(steps):
            nr += dr
            nc += dc
            if not (0 <= nr < h and 0 <= nc < w):
                return False
            if int(grid[nr, nc]) != bg:
                return False
        for _ in range(MIN_POST_STOP_BUFFER):
            nr += dr
            nc += dc
            if not (0 <= nr < h and 0 <= nc < w):
                return False
            if int(grid[nr, nc]) != bg:
                return False
    return True


def _segments_cross(seg_a: tuple[tuple[int, int], tuple[int, int]], seg_b: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    pa, na = seg_a
    pb, nb = seg_b
    if abs(na[0] - pa[0]) != 1 or abs(na[1] - pa[1]) != 1:
        return False
    if abs(nb[0] - pb[0]) != 1 or abs(nb[1] - pb[1]) != 1:
        return False
    r0a, r1a = sorted((pa[0], na[0]))
    c0a, c1a = sorted((pa[1], na[1]))
    r0b, r1b = sorted((pb[0], nb[0]))
    c0b, c1b = sorted((pb[1], nb[1]))
    if (r0a, r1a, c0a, c1a) != (r0b, r1b, c0b, c1b):
        return False
    return {pa, na} != {pb, nb} and not ({pa, na} & {pb, nb})


def _trajectories_compatible(seeds: list[tuple[tuple[int, int], tuple[int, int], int]]) -> bool:
    occupied_cells: set[tuple[int, int]] = set()
    buffer_cells: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for seed, direction, total_cells in seeds:
        cells = _ray_cells(seed, direction, total_cells)
        if any(cell in occupied_cells or cell in buffer_cells for cell in cells):
            return False
        dr, dc = direction
        stop_buffer = (cells[-1][0] + dr, cells[-1][1] + dc)
        if stop_buffer in occupied_cells or stop_buffer in buffer_cells:
            return False
        occupied_cells.update(cells)
        buffer_cells.add(stop_buffer)
        beam_segments = list(zip(cells[:-1], cells[1:]))
        for seg in beam_segments:
            for other in segments:
                if _segments_cross(seg, other):
                    return False
        segments.extend(beam_segments)
    return True


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        beam_len = _beam_length_from_gap_count(gap_count)
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc)), beam_len - 1) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _choose_gap_count(available: dict[int, list[tuple[tuple[int, int], ...]]], rng: np.random.Generator) -> int:
    counts = np.array(sorted(available), dtype=int)
    weights = counts.astype(float)
    if counts[-1] == 4:
        weights[-1] *= 2.5
    weights /= weights.sum()
    return int(rng.choice(counts, p=weights))


def _grid_size_min(block_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    available_by_block: list[dict[int, list[tuple[tuple[int, int], ...]]]] = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        available_by_block.append(available)

    chosen_counts = [_choose_gap_count(available, rng) for available in available_by_block]
    if block_count > 1 and len(set(chosen_counts)) == 1:
        base = chosen_counts[0]
        diversify = [i for i, available in enumerate(available_by_block) if any(count != base for count in available)]
        if diversify:
            idx = diversify[int(rng.integers(len(diversify)))]
            alts = [count for count in sorted(available_by_block[idx]) if count != base]
            chosen_counts[idx] = int(rng.choice(alts))

    chosen_gaps = []
    planned_seeds: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for (r0, c0), available, gap_count in zip(positions, available_by_block, chosen_counts):
        combos = available[gap_count]
        combo = combos[int(rng.integers(len(combos)))]
        chosen_gaps.append(combo)
        beam_len = _beam_length_from_gap_count(gap_count)
        for gap in combo:
            planned_seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap), beam_len))

    if not _trajectories_compatible(planned_seeds):
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        beam_len = _beam_length_from_gap_count(len(gaps))
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap), beam_len - 1))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e12f9a14 input")
