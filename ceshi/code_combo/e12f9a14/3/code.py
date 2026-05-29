# Combo variant of e12f9a14 (index 3)
# combo_spec: D1/1+D6/0
# Name: 反射弹跳+固定步数终止

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - Visible objects stay close to the base task: each colored 2x2 block sits inside a shared-color 4x4 frame.
#   - Generation uses corner gaps only, so every emitted beam starts diagonally as required by the combo spec.
#   - The beam no longer dies at the first boundary hit. Instead it mirrors on that edge and keeps going.
#   - The beam also no longer has unlimited lifetime. The gap cell is painted immediately, then the beam may
#     take exactly 4 further moves, because the surrounding frame side length is visibly 4.
#   - Small trace:
#       frame NE gap at seed (2, n-2) launches NE.
#       base task: paint until top/right boundary or collision, so the path keeps extending after the first edge.
#       combo task: step to (1, n-1), reflect on the right edge to NW and step to (0, n-2), reflect on the top
#       edge to SW and step to (1, n-3), then step to (2, n-4) and stop purely because the 4-step budget is spent.
#       The crucial difference is that both reflections happen before termination, yet the beam still stops with
#       empty runway remaining.
# complexity_calibration:
#   - The distance code has one visible level only: frame side length 4 -> four post-gap moves.
#   - To keep that simple code from becoming visually weak, generation forces at least one beam to bounce twice
#     before exhausting its budget. This makes the fusion visibly different from both "straight to edge" and
#     "bounce forever until blocked".
#   - Frames use one corner gap each in generation. Repetition across several blocks supports the constant-length
#     rule while keeping the output legible.
# implicit_assumption_audit:
#   - The base solver assumed "proposal out of bounds => beam stops". That is false now; proposal building must
#     reflect the direction before checking whether the next cell is legal.
#   - The D1 reflection variant allowed beams to revisit old territory, so self-reentry and self-segment crossing
#     must stop a beam defensively.
#   - The base and D1 solvers also assumed a beam can be reactivated after a collision redirect. With a finite
#     budget that is only valid while remaining steps stay positive, so all reactivation now flows through
#     refresh_active()/set_pattern().
#   - Collision handlers consume exactly one move for each moving participant even when the visible cell outcome is
#     a gap or a redirected zigzag step. Otherwise the fixed-step law would drift from the simulated path.
# placement_legality:
#   - Frames still obey square-grid fit and a minimum bbox clearance.
#   - A candidate gap is accepted only if a single-beam dry run on the obstacle grid can complete all 4 moves,
#     reflects as needed, and still has one more legal continuation cell after the budget is exhausted.
#   - The generator also rejects any pair of chosen beams whose traced paths overlap or whose diagonal segments
#     cross geometrically, so the final samples remain collision-free and clean.
# visual_observability:
#   - At least one generated beam reflects twice, so the bounce rule is visible in the output itself.
#   - Every generated beam stops after the same visible total length of 5 cells (gap cell + 4 moves), including
#     cases where further reflected travel would still be possible. That makes the fixed-budget rule inferable.
# visual_cleanliness:
#   - Collision-free generation avoids tangled outputs and keeps each beam attributable to its source frame.
#   - Larger frame spacing and corner-only gaps prevent ambiguous thick junctions.
# color_inference:
#   - No color remapping is introduced. The beam color is still inherited from the inner 2x2 block.
# solver_assumptions_enforced_by_generator:
#   - All generated gaps are structural frame gaps detected by _frame_gaps_for_block().
#   - Every generated beam is diagonal, has a full 4-step reflected runway, and has open runway beyond the budget.
#   - At least one beam has >= 2 reflections before stopping, and at least one beam stops by budget rather than
#     by obstacle or self-contact.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}
_FRAME_SIDE = 4
_BEAM_STEPS = _FRAME_SIDE
_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))
_SPECIAL_POS = (2, 3)
_SPECIAL_GAP = (0, 0)  # top-left frame corner -> NW launch


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


def _reflect_dir(d: tuple[int, int], flip_r: bool, flip_c: bool) -> tuple[int, int]:
    dr, dc = d
    if flip_r:
        dr = -dr
    if flip_c:
        dc = -dc
    return int(dr), int(dc)


def _reflect_pattern(
    pattern: tuple[tuple[int, int], ...], flip_r: bool, flip_c: bool
) -> tuple[tuple[int, int], ...]:
    return tuple(_reflect_dir(d, flip_r, flip_c) for d in pattern)


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _beam_steps_from_frame() -> int:
    return _BEAM_STEPS


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "remaining")

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
        self.enter_dirs = [pattern[0]]
        self.remaining = int(remaining)
        self.active = self.remaining > 0

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _solve(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0
    reflection_counts: dict[int, int] = {}

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    def refresh_active(bid: int) -> None:
        beams[bid].active = beams[bid].remaining > 0

    def consume_step(bid: int, steps: int = 1) -> None:
        b = beams[bid]
        assert b.remaining >= steps, "beam step budget underflow"
        b.remaining -= steps
        refresh_active(bid)

    def set_pattern(bid: int, pattern: tuple[tuple[int, int], ...]) -> None:
        beams[bid].pattern = pattern
        beams[bid].phase = 0
        refresh_active(bid)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        out[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        out[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        refresh_active(bid)

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        steps: int,
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, remaining=steps)
        reflection_counts[bid] = 0
        paint_cell(bid, pos, pattern[0])

    step_budget = _beam_steps_from_frame()
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), step_budget)

    assert beams, "expected at least one beam spawned from frame gaps"

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
        movers = {a}
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
            movers.add(b)
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide at one cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                truncate_beam(b, cell_index[cell])
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            for gid in sorted(movers):
                consume_step(gid)
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

        if cell_owner.get(cell) != bid_u:
            if cell_owner.get(cell) == bid_v:
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            if cell_owner.get(cell) == bid_u:
                truncate_beam(bid_u, cell_index[cell] + 1)
            else:
                bu = beams[bid_u]
                bu.path.append(cell)
                bu.enter_dirs.append(u)
                bu.pos = cell
                paint_cell(bid_u, cell, u)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)

        bv = beams[bid_v]
        adj = (bv.pos[0] + u[0], bv.pos[1] + u[1])
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
        bv.path.append(adj)
        bv.enter_dirs.append(u)
        bv.pos = adj
        paint_cell(bid_v, adj, u)

        for gid in sorted(movers):
            consume_step(gid)

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

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        db = prop[b][1]
        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected crossing diagonals to resolve forward"
        consume_step(a)
        consume_step(b)
        set_pattern(a, (nd,))
        set_pattern(b, (nd,))

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        seg_dir: tuple[int, int] | None = None
        if (
            cell_owner.get(n2) == b
            and cell_index.get(n2, 0) > 0
            and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1])
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif (
            cell_owner.get(p2) == b
            and cell_index.get(p2, 0) > 0
            and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1])
        ):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        nd = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert nd != (0, 0), "expected segment crossing to resolve forward"

        consume_step(a)
        group = {a, b}
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if pb is not None:
            group.add(pb)
        for gid in group:
            set_pattern(gid, (nd,))

    max_iters = h * w * 400
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
            reflected = False
            if not (0 <= nr < h and 0 <= nc < w):
                flip_r = not (0 <= nr < h)
                flip_c = not (0 <= nc < w)
                b.pattern = _reflect_pattern(b.pattern, flip_r, flip_c)
                d = b.next_dir()
                nr, nc = b.pos[0] + d[0], b.pos[1] + d[1]
                reflected = True
            nxt = (nr, nc)
            if not in_bounds(nxt) or not is_free(nxt):
                b.active = False
                continue
            if cell_owner.get(nxt) == bid:
                b.active = False
                continue
            if abs(d[0]) + abs(d[1]) == 2:
                pr, pc = b.pos
                p2 = (pr, pc + d[1])
                n2 = (pr + d[0], pc)
                if diag_segment_owner(p2, n2) == bid:
                    b.active = False
                    continue
            if reflected:
                reflection_counts[bid] += 1
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
            cross_pairs.add(tuple(sorted((bid, obid))))

        seg_cross: set[tuple[int, int]] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            p2 = (pr, pc + d[1])
            n2 = (pr + d[0], pc)
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
            events.append((0, cell[0], cell[1], a, cand_bs[0], "cell"))
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            events.append((1, min(ca[0], cb[0]), min(ca[1], cb[1]), a, b, "cross"))
        for a, b in seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            events.append((1, min(pr, pr + dr), min(pc, pc + dc), a, b, "segcross"))

        if events:
            events.sort()
            _, r, c, a, b, kind = events[0]
            if kind == "cell":
                resolve_cell_collision((r, c), a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision(a, b, prop)
            else:
                resolve_cross_collision(a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            consume_step(bid)
            if b.remaining > 0:
                b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    if not collect_stats:
        return out

    stats = {
        "beam_count": len(beams),
        "budget_exhausted_beams": sum(b.remaining == 0 for b in beams.values()),
        "total_reflections": sum(reflection_counts.values()),
        "reflected_beams": sum(v > 0 for v in reflection_counts.values()),
        "max_reflections_on_beam": max(reflection_counts.values(), default=0),
        "reflection_counts": dict(reflection_counts),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks, letting them bounce off the boundary but stopping them after a frame-sized number of moves.

    Input structure:
    - The background is the dominant color, and each colored 2x2 block is enclosed by a shared-color 4x4 frame.
    - Border gaps are the beam exits; this combo variant uses diagonal corner exits as the meaningful launch points.

    Rule:
    - Each gap paints a beam in the inner block's color, starting diagonally outward from the gap.
    - If the next step would leave the grid, the beam reflects on the hit boundary and continues with the mirrored diagonal direction.
    - The beam does not run indefinitely: after the gap cell is painted, it may take exactly 4 more moves, because that count is encoded by the visible side length of the surrounding 4x4 frame.
    - If beams meet before their budget is exhausted, keep the original collision laws: 90-degree meetings leave a blank collision cell and redirect to the bisector, while 45-degree meetings merge into the alternating zigzag path.

    Key insight:
    - The solver must combine two cues at once: follow the mirrored diagonal trajectory through one or more boundary bounces, but stop tracing after the frame-sized move budget is spent even if more reflected runway remains.
    """
    return _solve(grid, collect_stats=False)


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 5)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 12, 2: 15, 3: 18, 4: 22}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 3
PLACEMENT_MARGIN = 2
MIN_BEAM_PIXELS_PER_GAP = 5
GEN_RETRIES = 500


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
    special = _SPECIAL_POS
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected = [special]
    for _ in range(count - 1):
        viable = [p for p in all_positions if p not in selected and all(_frames_separated(p, q) for q in selected)]
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


def _segment_crosses(seg1: tuple[tuple[int, int], tuple[int, int]], seg2: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    a0, a1 = seg1
    b0, b1 = seg2
    if len({a0, a1, b0, b1}) < 4:
        return False
    if abs(a1[0] - a0[0]) != 1 or abs(a1[1] - a0[1]) != 1:
        return False
    if abs(b1[0] - b0[0]) != 1 or abs(b1[1] - b0[1]) != 1:
        return False
    rows = sorted({a0[0], a1[0], b0[0], b1[0]})
    cols = sorted({a0[1], a1[1], b0[1], b1[1]})
    if len(rows) != 2 or len(cols) != 2 or rows[1] - rows[0] != 1 or cols[1] - cols[0] != 1:
        return False
    slope1 = (a1[0] - a0[0]) * (a1[1] - a0[1])
    slope2 = (b1[0] - b0[0]) * (b1[1] - b0[1])
    return slope1 != slope2


def _trace_profile(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    d: tuple[int, int],
    steps: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    pos = seed
    cur_d = d
    path = [seed]
    visited = {seed}
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    reflections = 0

    for _ in range(steps):
        nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
        if not (0 <= nr < h and 0 <= nc < w):
            flip_r = not (0 <= nr < h)
            flip_c = not (0 <= nc < w)
            cur_d = _reflect_dir(cur_d, flip_r, flip_c)
            nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
            reflections += 1
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        nxt = (nr, nc)
        if int(grid[nr, nc]) != bg or nxt in visited:
            return None
        seg = (pos, nxt)
        for old_seg in segments:
            if _segment_crosses(seg, old_seg):
                return None
        path.append(nxt)
        visited.add(nxt)
        segments.append(seg)
        pos = nxt

    nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
    next_d = cur_d
    if not (0 <= nr < h and 0 <= nc < w):
        flip_r = not (0 <= nr < h)
        flip_c = not (0 <= nc < w)
        next_d = _reflect_dir(cur_d, flip_r, flip_c)
        nr, nc = pos[0] + next_d[0], pos[1] + next_d[1]
    open_after_budget = (
        0 <= nr < h
        and 0 <= nc < w
        and int(grid[nr, nc]) == bg
        and (nr, nc) not in visited
        and not any(_segment_crosses((pos, (nr, nc)), old_seg) for old_seg in segments)
    )
    return {
        "path": tuple(path),
        "segments": tuple(segments),
        "reflections": reflections,
        "open_after_budget": open_after_budget,
    }


def _available_traces(
    obstacles: np.ndarray, bg: int, r0: int, c0: int
) -> dict[tuple[int, int], dict[str, object]]:
    out: dict[tuple[int, int], dict[str, object]] = {}
    for gap in _CORNER_GAPS:
        profile = _trace_profile(obstacles, bg, (r0 + gap[0], c0 + gap[1]), _gap_dir(gap), _BEAM_STEPS)
        if profile is None or not profile["open_after_budget"]:
            continue
        out[gap] = profile
    return out


def _traces_compatible(a: dict[str, object], b: dict[str, object]) -> bool:
    if set(a["path"]) & set(b["path"]):
        return False
    for seg1 in a["segments"]:
        for seg2 in b["segments"]:
            if _segment_crosses(seg1, seg2):
                return False
    return True


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 12
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

    trace_by_pos: dict[tuple[int, int], dict[tuple[int, int], dict[str, object]]] = {}
    for r0, c0 in positions:
        available = _available_traces(obstacles, bg, r0, c0)
        if not available:
            return None
        trace_by_pos[(r0, c0)] = available

    special = _SPECIAL_POS
    special_traces = trace_by_pos.get(special, {})
    if _SPECIAL_GAP not in special_traces:
        return None
    if int(special_traces[_SPECIAL_GAP]["reflections"]) < 2:
        return None

    chosen_gaps: dict[tuple[int, int], tuple[int, int]] = {special: _SPECIAL_GAP}
    chosen_profiles = [special_traces[_SPECIAL_GAP]]

    for pos in positions:
        if pos == special:
            continue
        options = list(trace_by_pos[pos].items())
        rng.shuffle(options)
        choice: tuple[int, int] | None = None
        for gap, profile in options:
            if all(_traces_compatible(profile, prev) for prev in chosen_profiles):
                choice = gap
                chosen_profiles.append(profile)
                break
        if choice is None:
            return None
        chosen_gaps[pos] = choice

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        gap = chosen_gaps[(r0, c0)]
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), (gap,))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, collect_stats=True)
    except Exception:
        return None

    gap_total = len(positions)
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["reflected_beams"] < 1:
        return None
    if stats["max_reflections_on_beam"] < 2:
        return None
    if stats["budget_exhausted_beams"] != gap_total:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
