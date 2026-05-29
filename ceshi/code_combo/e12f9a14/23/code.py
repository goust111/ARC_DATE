# Combo variant of e12f9a14 (index 23)
# combo_spec: D6/0+D7/2
# Name: 可变框架大小+固定步数终止

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT / "arc2_solver_select") not in sys.path:
    sys.path.append(str(_ROOT / "arc2_solver_select"))

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - The original object "2x2 block inside a 4x4 frame" is generalized into a
#     family of hollow square frames with outer side lengths 3/4/5/6.
#   - Each frame still has a one-cell-thick border in one shared frame color and
#     encloses a solid monochrome square of side `frame_side - 2`.
#   - Border gaps still emit one beam each; corner gaps launch diagonal beams and
#     side gaps launch cardinal beams, exactly as in the source task logic.
#   - The D6 distance change is fused with D7 size variation by encoding travel
#     budget directly in the visible frame side length:
#       frame side 3/4/5/6 -> beam moves 3/4/5/6 steps after the gap cell.
# behavior_trace:
#   - Original: a 4x4 framed 2x2 block with one east gap would paint from the
#     gap to the grid edge unless a collision redirects it.
#   - Combo: a 5x5 frame with the same east gap paints the gap cell, then moves
#     exactly 5 more cells and stops in open background even if space remains.
#   - Another 3x3 frame with the same gap count in the same sample stops after 3
#     moves, so the solver must measure size rather than count openings.
# complexity_calibration:
#   - Only four size levels exist, which keeps the mapping learnable while making
#     the visual difference between short and long rays obvious.
#   - Gap count stays independent of beam length. The generator forces repeated
#     evidence where different frame sizes share the same gap count, preventing a
#     simpler "count gaps -> length" shortcut from fitting all examples.
#   - Multiple gaps on one frame provide repeated support for the same size-based
#     budget without introducing extra symbol systems or color remappings.
# hidden_assumptions_checked_in_main:
#   - `_find_blocks_2x2()` is no longer valid because the enclosed square is not
#     always 2x2. Detection must scan candidate framed squares of side 3..6.
#   - Original reactivation after collisions assumed unlimited lifetime. Here any
#     direction change must preserve the remaining move budget and stop at zero.
#   - Gap detection can no longer use hard-coded 4x4 coordinates. Structural gap
#     tests must work on an arbitrary square perimeter.
#   - The beam simulator still assumes beams never hit intact frames directly.
#     The generator therefore validates runways and rejects any sample for which
#     `main()` raises due to a blocked-by-frame trajectory.
# placement_legality:
#   - The generator places square frames with side-dependent bounding boxes and
#     keeps at least a two-cell clearance between those boxes.
#   - Every chosen gap must have at least `frame_side` clear cells in its initial
#     direction plus one extra background buffer cell, so fixed-stop behavior is
#     visible instead of coinciding with an edge stop.
#   - The final sample is accepted only if the solver runs successfully and adds
#     enough beam pixels beyond the input.
# observability_and_cleanliness:
#   - Size is directly visible from the square outline, and length is directly
#     visible from where the ray stops, so the key relation is human-observable.
#   - No invisible ordering by color ids is used. Colors only distinguish blocks.
#   - The output stays readable because size range is small, frame spacing is
#     enforced, and samples are rejected if the result is too trivial.
# new_main_assumptions_and_generator_guarantees:
#   - Solver assumes each valid frame has 1-4 non-adjacent structural gaps and a
#     solid interior square. Generator constructs exactly that object family.
#   - Solver assumes at least one pair of frames can demonstrate that gap count is
#     not the code. Generator preferentially emits multi-frame scenes with a same-
#     gap-count / different-size pair; otherwise the attempt is retried.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}
_FRAME_SIDES = (3, 4, 5, 6)
_FRAME_SIDES_DESC = tuple(sorted(_FRAME_SIDES, reverse=True))


def _border_order(side: int) -> tuple[tuple[int, int], ...]:
    pts: list[tuple[int, int]] = []
    for c in range(side):
        pts.append((0, c))
    for r in range(1, side - 1):
        pts.append((r, side - 1))
    for c in range(side - 1, -1, -1):
        pts.append((side - 1, c))
    for r in range(side - 2, 0, -1):
        pts.append((r, 0))
    return tuple(pts)


_BORDER_ORDER = {side: _border_order(side) for side in _FRAME_SIDES}
_BORDER_INDEX = {side: {p: i for i, p in enumerate(_BORDER_ORDER[side])} for side in _FRAME_SIDES}


def _gap_dir(side: int, gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == side - 1 else 0)
    dc = -1 if gc == 0 else (1 if gc == side - 1 else 0)
    return dr, dc


def _frame_steps_from_side(side: int) -> int:
    assert side in _FRAME_SIDES, "expected frame side in 3..6"
    return side


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


def _frame_gaps_for_square(
    g: np.ndarray,
    bg: int,
    wall: int,
    r0: int,
    c0: int,
    side: int,
) -> list[tuple[int, int]]:
    order = _BORDER_ORDER[side]
    vals = [int(g[r0 + dr, c0 + dc]) for dr, dc in order]
    gaps: list[tuple[int, int]] = []
    n = len(order)
    for i, ((dr, dc), v) in enumerate(zip(order, vals)):
        if v != bg:
            continue
        if vals[(i - 1) % n] == wall and vals[(i + 1) % n] == wall:
            gaps.append((r0 + dr, c0 + dc))
    return gaps


def _find_framed_squares(
    g: np.ndarray,
    bg: int,
) -> list[tuple[int, int, int, int, int, list[tuple[int, int]]]]:
    h, w = g.shape
    used = np.zeros((h, w), dtype=bool)
    found: list[tuple[int, int, int, int, int, list[tuple[int, int]]]] = []
    for side in _FRAME_SIDES_DESC:
        inner = side - 2
        order = _BORDER_ORDER[side]
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                if used[r0 : r0 + side, c0 : c0 + side].any():
                    continue
                interior = g[r0 + 1 : r0 + 1 + inner, c0 + 1 : c0 + 1 + inner]
                inner_colors = {int(v) for v in interior.ravel()}
                if len(inner_colors) != 1:
                    continue
                block_col = next(iter(inner_colors))
                if block_col == bg:
                    continue
                border_vals = [int(g[r0 + dr, c0 + dc]) for dr, dc in order]
                wall_colors = {v for v in border_vals if v != bg}
                if len(wall_colors) != 1:
                    continue
                wall = next(iter(wall_colors))
                if wall == block_col:
                    continue
                if any(v not in (bg, wall) for v in border_vals):
                    continue
                gaps = _frame_gaps_for_square(g, bg, wall, r0, c0, side)
                if not (1 <= len(gaps) <= 4):
                    continue
                if sum(v == bg for v in border_vals) != len(gaps):
                    continue
                found.append((block_col, wall, r0, c0, side, gaps))
                used[r0 : r0 + side, c0 : c0 + side] = True
    assert found, "expected at least one framed square in input"
    return found


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored beams from variable-size framed squares, with travel distance encoded by the frame side length.

    Input structure:
    - The background is the dominant color. Each object is a hollow square frame of side 3..6 in one shared frame color.
    - Inside each frame sits a solid monochrome square of a different color, and 1-4 border cells are missing as visible gaps.

    Rule:
    - Every gap emits one beam in the outward direction implied by that border location, using the inner square color.
    - The gap cell is painted immediately, then the beam moves a fixed number of additional steps equal to the frame side length.
    - Beams do not default to the grid edge anymore. They stop as soon as their size-encoded budget reaches zero.
    - If two beams meet before that, preserve the source task's collision behavior: 90-degree meetings create a blank intersection and redirect to the bisector, while 45-degree meetings become a zigzag. Any redirected beam still obeys its remaining budget.

    Key insight: Length is no longer inferred from where a ray can physically reach. Measure the frame first: bigger square frames authorize longer rays, while equal gap counts can still produce different stopping distances.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _find_framed_squares(g, bg)

    beams: dict[int, _Beam] = {}
    next_bid = 0

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
        paint_cell(bid, pos, pattern[0])

    for col, _wall, r0, c0, side, gaps in frames:
        steps = _frame_steps_from_side(side)
        for gr, gc in gaps:
            d = _normalize_dir(*_gap_dir(side, (gr - r0, gc - c0)))
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), steps)

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide at same cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            for gid in sorted(a_consumes):
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
                idx = cell_index[cell]
                truncate_beam(bid_v, idx)
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                idx = cell_index[cell]
                truncate_beam(bid_u, idx + 1)
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

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell after 45-degree resolution"

        for gid in sorted(a_consumes):
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

    def resolve_cross_collision(a: int, b: int) -> None:
        da = beams[a].next_dir()
        db = beams[b].next_dir()
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        consume_step(a)
        consume_step(b)
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

        consume_step(a)
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
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
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
            cross_pairs.add(tuple(sorted((bid, obid))))

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
                resolve_cross_collision(a, b)
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

    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 3: 18, 4: 22, 5: 26}
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
GEN_RETRIES = 400


def _gap_combos(side: int, size: int) -> list[tuple[tuple[int, int], ...]]:
    order = _BORDER_ORDER[side]
    n = len(order)
    idx_map = _BORDER_INDEX[side]
    combos: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(order, size):
        idxs = sorted(idx_map[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % n == 1 for i in range(size)):
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {
    side: {gap_count: _gap_combos(side, gap_count) for gap_count in range(1, 5)}
    for side in _FRAME_SIDES
}


def _frames_separated(
    a: tuple[int, int],
    side_a: int,
    b: tuple[int, int],
    side_b: int,
) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + side_a - 1, ac0 + side_a - 1
    br1, bc1 = br0 + side_b - 1, bc0 + side_b - 1
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(
    n: int,
    sides: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    count = len(sides)
    for _ in range(80):
        order = sorted(range(count), key=lambda i: (-sides[i], float(rng.random())))
        chosen: list[tuple[int, int] | None] = [None] * count
        ok = True
        for idx in order:
            side = sides[idx]
            margin = PLACEMENT_MARGIN
            positions = [(r, c) for r in range(margin, n - side - margin + 1) for c in range(margin, n - side - margin + 1)]
            viable = [
                p for p in positions
                if all(
                    chosen[j] is None or _frames_separated(p, side, chosen[j], sides[j])
                    for j in range(count)
                )
            ]
            if not viable:
                ok = False
                break
            chosen[idx] = viable[int(rng.integers(len(viable)))]
        if ok:
            return [p for p in chosen if p is not None]
    return None


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    side: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((side, side), frame_color, dtype=int)
    patch[1 : side - 1, 1 : side - 1] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


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


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    side: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    steps = _frame_steps_from_side(side)
    for gap_count, combos in _GAP_COMBOS[side].items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir(side, (gr, gc)), steps) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


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


def _pick_sides(block_count: int, rng: np.random.Generator) -> list[int]:
    for _ in range(50):
        sides = [int(rng.choice(_FRAME_SIDES)) for _ in range(block_count)]
        if len(set(sides)) >= 2:
            return sides
    return sides


def _pick_gap_counts(
    available_by_block: list[dict[int, list[tuple[tuple[int, int], ...]]]],
    sides: list[int],
    rng: np.random.Generator,
) -> list[int] | None:
    counts = [int(rng.choice(sorted(available))) for available in available_by_block]
    block_count = len(sides)
    if block_count < 2:
        return counts

    if any(sides[i] != sides[j] and counts[i] == counts[j] for i in range(block_count) for j in range(i + 1, block_count)):
        return counts

    pair_options: list[tuple[int, int, int]] = []
    for i in range(block_count):
        for j in range(i + 1, block_count):
            if sides[i] == sides[j]:
                continue
            common = sorted(set(available_by_block[i]).intersection(available_by_block[j]))
            for gap_count in common:
                pair_options.append((i, j, int(gap_count)))
    if not pair_options:
        return None

    i, j, gap_count = pair_options[int(rng.integers(len(pair_options)))]
    counts[i] = gap_count
    counts[j] = gap_count
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    sides = _pick_sides(block_count, rng)
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, sides, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), side, color in zip(positions, sides, block_colors):
        _draw_framed_block(obstacles, r0, c0, side, bg, frame_color, int(color), ())

    available_by_block: list[dict[int, list[tuple[tuple[int, int], ...]]]] = []
    for (r0, c0), side in zip(positions, sides):
        available = _clear_gap_combos(obstacles, bg, r0, c0, side)
        if not available:
            return None
        available_by_block.append(available)

    chosen_counts = _pick_gap_counts(available_by_block, sides, rng)
    if chosen_counts is None:
        return None

    chosen_gaps = []
    for available, side, gap_count in zip(available_by_block, sides, chosen_counts):
        combos = available.get(gap_count)
        if not combos:
            return None
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for (r0, c0), side, color, gaps in zip(positions, sides, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, side, bg, frame_color, int(color), gaps)
        steps = _frame_steps_from_side(side)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(side, gap), steps))

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
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
