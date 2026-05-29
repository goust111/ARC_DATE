# Variant of e12f9a14
# Dimension: D13 — 框架对射线绘制的影响 (transform)
# Change: 射线经过框架自身的缺口位置时会在缺口处绘制源颜色像素（替换原来的背景色），框架的绿色边框像素保持不变... → 射线发射后源框架消失

import numpy as np

from arc_utils import get_bg, paste
import itertools

# Design analysis
# ---------------
# design_analysis:
#   variant_goal:
#     - Keep the original frame/gap detection and beam-collision physics.
#     - Change only the transform stage: once a frame has emitted its beams, the 4x4 source object
#       is consumed and disappears from the output, matching the erase flavor of 3dc255db.
#   concretization:
#     - Input stays the same as the base task: 2x2 monochrome blocks inside a shared-color 4x4 frame,
#       with 1-4 structural border gaps.
#     - Output no longer preserves the green frame or the inner 2x2 color patch.
#     - The gap cell itself still becomes a beam-colored pixel, because it is the first visible ray cell
#       and is already background in the input.
#   behavior_trace:
#     - Base task: a red 2x2 block inside a green frame with a NE gap keeps the green border and red core,
#       and adds a red diagonal starting from the gap.
#     - Variant: the same ray is emitted with the same path, collisions, and stopping rules, but after
#       simulation every original non-background source cell in that 4x4 frame is erased to background.
#     - Difference: the visible evidence of the source object is removed; only the spent ray remains.
#   complexity_calibration:
#     - Do not reinterpret erased frames as empty space during propagation. Letting beams tunnel through
#       vanished sources would create a second rule change and alter collision layout.
#     - The variant is still visually obvious because each source contributes 12-15 erased pixels while
#       the beam seed at the gap remains colored.
#   implicit_assumptions_checked:
#     - _find_blocks_2x2 still works because detection reads the input, not the output.
#     - _infer_frame_color and _frame_gaps_for_block still require an intact input frame; generator keeps that.
#     - is_free() still consults the input grid, so frames remain blockers during simulation by design.
#     - Final erasure must clear only cells that were non-background in the input; clearing the full 4x4 box
#       would incorrectly delete the painted gap seed.
#   placement_legality:
#     - Reuse the original placement margins and frame separation so every 4x4 frame is in-bounds and
#       uniquely recoverable.
#     - Generator still rejects layouts where initial rays would immediately hit another frame.
#   observability_and_neatness:
#     - The rule is human-visible in 2-4 examples because every example shows intact framed sources in
#       the input and missing sources in the output, while the ray geometry stays consistent.
#     - No color-id-dependent ordering or hidden mapping is introduced; colors only label source/ray identity.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


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
                # It's a missing wall pixel on the would-be 4x4 frame border.
                gaps.append((r, c))
    # Keep only "structural" gaps: those that have wall pixels on the two inward-adjacent border cells.
    # This filters out arbitrary background on the frame perimeter when the frame touches the grid edge.
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        ok = True
        if dr == 0:  # side gap: require walls above/below on the frame border
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:  # top/bottom gap: require walls left/right on the border
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:  # corner gap: require the two inward orthogonal border cells to be walls
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
    # Some frames can include an extra decorative corner hole in addition to multiple true exits.
    # We only suppress the SE corner when there are at least 3 structural gaps (so 2-gap frames still emit both).
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


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        # enter_dirs[i] is the move used to enter path[i]; for start, store the first move dir.
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from gapped 4x4 frames, then erase the launching frames once their rays have been spent.

    Input structure:
    - bg is the dominant background. A shared frame color surrounds each monochrome 2x2 source block.
    - Every 4x4 frame has one or more structural border gaps that define outward beam launch sites.

    Rule:
    - Each structural gap emits a beam in the same outward direction as in the base task, using the inner block color.
    - Beam travel and all collision rules are unchanged: rays advance until the edge or a collision, 90-degree meetings leave a gap and turn, and 45-degree meetings merge into a zigzag.
    - The frame still behaves as a solid obstacle during simulation; only the rendered output changes.
    - After the beam system settles, erase every original non-background source pixel from each 4x4 frame, leaving only the colored beam trajectory (including the painted gap seed).

    Key insight: D13 changes the post-emission rendering rather than the beam physics: solve the same ray interactions first, then consume each source frame so the output records only where the emitted energy traveled.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    # Build initial beams from frame gaps.
    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

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
            # Only remove if still owned by this beam.
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = (dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            d = _normalize_dir(*d)
            spawn_beam(col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        """Return an active beam adjacent to bid that is moving in lockstep (same next_dir/pattern/phase)."""
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
            # Avoid \"chasing\" pairs where one would step into the other's current cell.
            if (b.pos[0] + d[0], b.pos[1] + d[1]) == ob.pos:
                continue
            if (ob.pos[0] + d[0], ob.pos[1] + d[1]) == b.pos:
                continue
            assert partner is None, "expected at most one adjacent lockstep partner"
            partner = oid
        return partner

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        # prop: dict bid -> (next_cell, move_dir)
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            # 90°: remove intersection cell and turn both to the bisector direction.
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically (odd parity), not collide at same cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)  # keep up to pre-cell (idx is collision cell index)
            # Ensure the intersection cell is permanently blank (gap).
            blocked.add(cell)
            unpaint_cell(cell)
            # Turning direction: normalized sum.
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = {a, b}
            pa = find_partner(a)
            pb = find_partner(b)
            if pa is not None:
                group.add(pa)
            if pb is not None:
                group.add(pb)
            for gid in group:
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                beams[gid].active = True
            return

        # 45°: merge and turn to a 22.5° zigzag (alternate diag+card), no gap.
        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45° collision must involve exactly one cardinal and one diagonal beam"

        if is_card_a:
            u, v = da, db
            bid_u, bid_v = a, b
        else:
            u, v = db, da
            bid_u, bid_v = b, a

        # Put the cardinal beam head at the collision cell (it "wins" the cell).
        if cell_owner.get(cell) != bid_u:
            # If the diagonal beam owned it, truncate it away.
            if cell_owner.get(cell) == bid_v:
                idx = cell_index[cell]
                truncate_beam(bid_v, idx)  # keep up to pre-cell (idx is collision cell index)
                unpaint_cell(cell)
            # Truncate cardinal beam to its current head, then append collision cell.
            # (If it already had the cell earlier, truncate to it.)
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
            # Cardinal beam owns the cell and may have raced past it — truncate back.
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)

        # Diagonal beam moves one step in the cardinal direction instead of into the collision cell.
        bv = beams[bid_v]
        adj = (bv.pos[0] + u[0], bv.pos[1] + u[1])
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
        bv.path.append(adj)
        bv.enter_dirs.append(u)
        bv.pos = adj
        paint_cell(bid_v, adj, u)

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell after 45-deg resolution"

        zig = (v, u)  # after the resolving cardinal step, next step is the diagonal component.
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True

    def diag_segment_owner(p: tuple[int, int], q: tuple[int, int]) -> int | None:
        """Return the beam id that currently owns the diagonal segment connecting p<->q, if any."""
        dr, dc = q[0] - p[0], q[1] - p[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        # Segment p->q exists if q was entered from p.
        oq = cell_owner.get(q)
        if oq is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return oq
        # Or segment q->p exists if p was entered from q.
        op = cell_owner.get(p)
        if op is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return op
        return None

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        db = prop[b][1]
        # Expect both diagonal.
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        beams[a].pattern = (u,)
        beams[a].phase = 0
        beams[b].pattern = (u,)
        beams[b].phase = 0
        beams[a].active = True
        beams[b].active = True
        # Do not spawn opposite-direction beams: the band continues only forward (matches train0 mid-gap).

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        """Handle a diagonal move that would cross an existing diagonal segment (geometric crossing)."""
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        # Determine the existing segment direction on b.
        seg_dir: tuple[int, int] | None = None
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            # Truncate b back to p2 (remove n2 and everything after it).
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            # Truncate b back to n2 (remove p2 and everything after it).
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert u != (0, 0), "expected segment crossing to resolve to a forward direction"

        group = {a, b}
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if pb is not None:
            group.add(pb)
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            beams[gid].active = True

    # Main loop: iterate until no motion or collision.
    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        # Proposals for active beams.
        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
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

        # Detect crossing collisions among diagonal moves.
        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            move_map[(beams[bid].pos, nxt)] = bid

        # Check for opposite-diagonal crossing in a 2x2 square:
        # (r,c)->(r+sgn, c+sgn) crosses with (r, c+sgn)->(r+sgn, c)
        for (p, n), bid in list(move_map.items()):
            pr, pc = p
            nr, nc = n
            dr = nr - pr
            dc = nc - pc
            # other diagonal endpoints
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            obid = move_map.get((p2, n2))
            if obid is None or obid == bid:
                continue
            a, b = sorted((bid, obid))
            cross_pairs.add((a, b))

        # Detect geometric crossings: a diagonal move crossing an already-drawn diagonal segment.
        seg_cross: set[tuple[int, int]] = set()  # (moving_bid, other_bid)
        for bid, (nxt, d) in prop.items():
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

        # Detect cell collisions (including moving into an occupied cell).
        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int, int, str]] = []
        # cell events
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            # Prefer a moving beam as `a` (must exist in `prop`).
            assert bidders, "expected each target cell to have at least one bidder"
            bidders_sorted = sorted(bidders)
            a = bidders_sorted[0]
            # Pick `b` as either the owner (if different) or the next bidder.
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
        # cross events (use min of proposed target cells as location key)
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            keyr = min(ca[0], cb[0])
            keyc = min(ca[1], cb[1])
            events.append((1, keyr, keyc, a, b, "cross"))
        # segment-cross events
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
                cell = (r, c)
                resolve_cell_collision(cell, a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision(a, b, prop)
            else:
                resolve_cross_collision(a, b, prop)
            continue

        # No collisions: apply all moves.
        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    for _col, br, bc in blocks:
        fr0, fc0 = br - 1, bc - 1
        for r in range(fr0, fr0 + 4):
            for c in range(fc0, fc0 + 4):
                if int(g[r, c]) != bg:
                    out[r, c] = bg

    return out

# --------------- generator (o5x) ---------------

# ---- CFG: tunable generation parameters ----
BLOCK_COUNT_RANGE = (1, 6)              # uniform draw [lo, hi)
GRID_SIZE_MAX = 31                      # exclusive upper bound for square grid side
GRID_SIZE_MIN_BY_BLOCKS = {1: 8, 2: 14, 5: 20}  # block_count >= threshold -> min grid side
BG_CHOICES = [0, 8]                     # possible background colors
BG_WEIGHTS = [0.25, 0.75]              # probability weights for BG_CHOICES
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}      # bg -> frame color
PALETTE_FOR_BG = {                      # bg -> available block colors (before frame exclusion)
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2                      # min cell gap between 4x4 frame bounding boxes
PLACEMENT_MARGIN = 2                    # min distance from grid edge to frame top-left
MIN_RAY_STEPS = 2                       # each gap beam must travel at least this many cells
MIN_BEAM_PIXELS_PER_GAP = 2            # solver output must retain >= this many beam pixels per gap
GEN_RETRIES = 300                       # outer retry limit in generate_input()

# ---- precomputed border geometry for a 4x4 frame ----

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    """Return all valid gap placements of `size` gaps on the 4x4 frame border."""
    combos = []
    for combo in itertools.combinations(_BORDER_ORDER, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        # Reject adjacent gaps (would merge into a wider opening).
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        # Reject the SE-corner decorative-gap pattern (see solver _frame_gaps_for_block).
        if size >= 3 and (3, 3) in combo and (0, 2) in combo:
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    """Outward beam direction for a gap at local (gr, gc) on the 4x4 frame."""
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if two 4x4 frames at top-left positions a and b have >= FRAME_BBOX_GAP clearance."""
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    """Pick `count` non-overlapping frame positions within an n x n grid, or None if impossible."""
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
    grid: np.ndarray, r0: int, c0: int, bg: int,
    frame_color: int, block_color: int, gaps: tuple[tuple[int, int], ...],
) -> None:
    """Stamp a 4x4 frame with 2x2 inner block and gap openings onto grid."""
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _rays_clear(grid: np.ndarray, bg: int,
                seeds: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
    """Check that every seed ray travels >= MIN_RAY_STEPS through bg-only cells."""
    h, w = grid.shape
    for (r, c), (dr, dc) in seeds:
        steps = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            steps += 1
            if int(grid[nr, nc]) != bg:
                return False
            nr += dr
            nc += dc
        if steps < MIN_RAY_STEPS:
            return False
    return True


def _clear_gap_combos(
    obstacles: np.ndarray, bg: int, r0: int, c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """For one block at (r0, c0), return {gap_count: [valid_combos]} against obstacle grid."""
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    """Choose bg, frame_color, and block_count distinct block colors."""
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    """Minimum grid side length for a given block count."""
    result = 8
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

    # Build obstacle grid (full frames, no gaps) to validate ray clearance per block.
    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    chosen_gaps = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        gap_count = int(rng.choice(sorted(available)))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    # Render final grid with gaps and collect beam seeds.
    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.any(out[grid != bg] != bg):
        return None

    if np.count_nonzero(out != bg) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 input")
