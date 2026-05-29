# Combo variant of e12f9a14 (index 11)
# combo_spec: D8/0+D9/0
# Name: 始终单像素宽+无间隙

import itertools

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


# Design analysis for combo D8/0 + D9/0
# a) Concrete behavior trace.
#    - 90-degree cell collision in the original solver: two cardinal rays try to enter the same cell,
#      that cell is erased into a 1-pixel hole, then both rays turn toward the bisector and continue as
#      two adjacent thin tracks. In this combo the turn stays the same, but the collision cell is painted.
#      So the visible change is "same bend, no central hole".
#    - 45-degree cell collision in the original solver: one cardinal ray claims the meeting cell, the
#      diagonal ray steps sideways, and both heads continue with the same zigzag pattern. That creates a
#      2-cell-thick staircase band. In this combo the diagonal ray merges instead of running in parallel:
#      only one head survives, and the outgoing zigzag is a single-pixel staircase.
#    - Small example: an upward ray and a north-east ray meeting at (r,c). Originally the output around
#      the collision becomes a two-lane stair. Here only the upward ray keeps the collision cell and then
#      follows the alternating (north-east, up) pattern, so the post-collision trace is one thin line.
# b) Complexity calibration.
#    - If the generator allowed no collisions, this combo would visually collapse back to the original
#      input-output family. To keep both fused changes observable, every generated instance must contain
#      at least one 90-degree cell collision and at least one 45-degree cell collision.
#    - The generator therefore uses two explicit gadgets rather than unconstrained random placement.
#      This keeps the task readable: humans see exactly two collision motifs, each repeated enough to
#      infer "all collision cells are filled" and "zigzags never widen".
# c) Hidden-assumption audit in main().
#    - The original `blocked` set encoded the 90-degree gap. D9/0 invalidates that assumption, so the
#      simulation can no longer treat those cells as permanently forbidden.
#    - The original 45-degree branch assumed two synchronized heads must survive to model a thick band.
#      D8/0 removes that assumption. The fused solver now intentionally deactivates the diagonal partner
#      and keeps only one outgoing zigzag head.
#    - Cross-collision and segment-cross logic do not depend on the removed gap/thickening cues, so they
#      remain intact.
# d) Placement legality.
#    - Every frame is still a legal 4x4 border around a 2x2 monochrome block.
#    - Frame top-left coordinates are derived from target collision cells and checked against the original
#      `FRAME_BBOX_GAP` non-overlap rule. The chosen gadget geometry leaves all rays inside background
#      space until the intended collision.
# e) Visual observability.
#    - Humans can read the rule from 2-4 examples because the same two collision motifs recur: a filled
#      90-degree elbow and a thin 45-degree staircase. No color-id ordering or other invisible metadata is
#      used to disambiguate outcomes.
# f) Visual cleanliness.
#    - The 90-degree gadget lives in the upper-right of the canvas and turns northwest.
#    - The 45-degree gadget lives in the lower-left and turns southeast.
#      This spatial split avoids clutter and makes the two modified outcomes easy to compare.
# g) Inferability of color behavior.
#    - Colors still only identify source blocks; the combo changes geometry, not recoloring. Each block
#      keeps a unique beam color, so the solver can visually track which emitted line participated in which
#      collision.
# 9) New assumptions introduced by the combo solver.
#    - The 45-degree merge now assumes the diagonal participant can be safely terminated at the collision;
#      the generator enforces isolated 45-degree gadgets so that no later logic needs that diagonal head.
#    - The 90-degree no-gap branch now paints the collision cell and lets one beam own it. The generator
#      keeps later traffic away from that exact cell, avoiding ambiguous re-entry cases while still letting
#      the outgoing turn be fully visible.


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


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    stats = {
        "cell_collision_45": 0,
        "cell_collision_90": 0,
        "cross_collision": 0,
        "segment_cross_collision": 0,
        "merged_beams": 0,
    }

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

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
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def append_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.pos = cell
        paint_cell(bid, cell, enter_dir)

    def claim_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        owner = cell_owner.get(cell)
        if owner == bid:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid].path):
                truncate_beam(bid, idx + 1)
            else:
                beams[bid].pos = cell
            return
        if owner is not None:
            idx = cell_index[cell]
            assert idx > 0, "unexpected attempt to steal a beam seed"
            truncate_beam(owner, idx)
            unpaint_cell(cell)
        append_cell(bid, cell, enter_dir)

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
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg

    def find_partner(bid: int) -> int | None:
        b = beams.get(bid)
        if b is None or not b.active:
            return None
        d = b.next_dir()
        partner: int | None = None
        for oid, ob in beams.items():
            if oid == bid or not ob.active:
                continue
            if ob.pattern != b.pattern or ob.phase != b.phase or ob.next_dir() != d:
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
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            stats["cell_collision_90"] += 1
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "same-cell 90-degree collision should be cardinal/cardinal"

            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                assert idx > 0, "unexpected occupied collision on a beam seed"
                truncate_beam(b, idx)
                unpaint_cell(cell)

            claim_cell(a, cell, da)
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

        stats["cell_collision_45"] += 1
        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must involve exactly one cardinal and one diagonal beam"

        if is_card_a:
            u, v = da, db
            bid_u, bid_v = a, b
        else:
            u, v = db, da
            bid_u, bid_v = b, a

        if cell_owner.get(cell) == bid_v:
            idx = cell_index[cell]
            assert idx > 0, "unexpected diagonal ownership of a seed during 45-degree merge"
            truncate_beam(bid_v, idx)
            unpaint_cell(cell)
        claim_cell(bid_u, cell, u)

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = False
        stats["merged_beams"] += 1

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
        stats["cross_collision"] += 1
        da = prop[a][1]
        db = prop[b][1]
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        beams[a].pattern = (u,)
        beams[a].phase = 0
        beams[b].pattern = (u,)
        beams[b].phase = 0
        beams[a].active = True
        beams[b].active = True

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        stats["segment_cross_collision"] += 1
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

    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or int(g[nxt[0], nxt[1]]) != bg, "beam must stop only at edge or frame"
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
            append_cell(bid, nxt, d)
            beams[bid].advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from framed 2x2 color blocks; collisions redirect the rays but every painted path stays thin.

    Input structure:
    - The background is the most common color.
    - Each object is a 2x2 monochrome block enclosed by a 4x4 frame of one shared wall color.
    - Missing wall pixels on that border are beam exits; corner exits launch diagonal rays and side exits
      launch cardinal rays.

    Rule:
    - Every gap emits one same-color ray that paints background cells until the ray reaches the edge,
      reaches a frame, or collides with another painted trajectory.
    - 90-degree same-cell collisions still turn both participants toward the bisector, but the collision
      cell is painted instead of left blank.
    - 45-degree cardinal/diagonal collisions still become a zigzag, but the outgoing path is a single
      one-pixel staircase rather than a thicker two-lane band.

    Key insight:
    - The original task distinguished collision types with width and with a hole. This combo removes both
      cues, so the only reliable clue is the change in direction at the collision.
    """
    out, _stats = _simulate(grid, collect_stats=False)
    return out


# --------------- generator (combo D8/0 + D9/0) ---------------

GRID_SIZE_RANGE = (24, 31)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
GEN_RETRIES = 200
REQUIRED_COLLISIONS_45 = 1
REQUIRED_COLLISIONS_90 = 1
MIN_OUTPUT_DELTA = 20

_TOP_GAPS = ((0, 1), (0, 2))
_LEFT_GAPS = ((1, 0), (2, 0))
_BOTTOM_GAPS = ((3, 1), (3, 2))
_SE_GAP = (3, 3)


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


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=4, replace=False))
    return bg, frame_color, block_colors


def _frame_pos_for_target(target: tuple[int, int], gap: tuple[int, int], steps: int) -> tuple[int, int]:
    dr, dc = _gap_dir(gap)
    seed = (target[0] - dr * steps, target[1] - dc * steps)
    return seed[0] - gap[0], seed[1] - gap[1]


def _layout_blocks(n: int, rng: np.random.Generator) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    gap_north = _TOP_GAPS[int(rng.integers(len(_TOP_GAPS)))]
    gap_west = _LEFT_GAPS[int(rng.integers(len(_LEFT_GAPS)))]
    gap_south = _BOTTOM_GAPS[int(rng.integers(len(_BOTTOM_GAPS)))]

    step_90_down = int(rng.integers(4, 6))
    step_90_right = int(rng.integers(4, 6))
    step_45_up = int(rng.integers(5, 7))
    step_45_diag = int(rng.integers(5, 7))

    t90 = (int(rng.integers(6, 9)), int(rng.integers(n - 12, n - 8)))
    t45 = (int(rng.integers(n - 10, n - 7)), int(rng.integers(10, 13)))

    blocks = [
        (_frame_pos_for_target(t90, gap_north, step_90_down), (gap_north,)),
        (_frame_pos_for_target(t90, gap_west, step_90_right), (gap_west,)),
        (_frame_pos_for_target(t45, gap_south, step_45_up), (gap_south,)),
        (_frame_pos_for_target(t45, _SE_GAP, step_45_diag), (_SE_GAP,)),
    ]

    positions = [pos for pos, _gaps in blocks]
    for r0, c0 in positions:
        if not (2 <= r0 and 2 <= c0 and r0 + 3 < n - 1 and c0 + 3 < n - 1):
            raise AssertionError("gadget frame out of bounds")
    for a, b in itertools.combinations(positions, 2):
        if not _frames_separated(a, b):
            raise AssertionError("gadget frames too close")
    return blocks


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(*GRID_SIZE_RANGE))
    bg, frame_color, block_colors = _pick_colors(rng)

    try:
        blocks = _layout_blocks(n, rng)
    except AssertionError:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), gaps, color in zip((pos for pos, _ in blocks), (g for _p, g in blocks), block_colors):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if stats["cell_collision_45"] < REQUIRED_COLLISIONS_45:
        return None
    if stats["cell_collision_90"] < REQUIRED_COLLISIONS_90:
        return None
    if np.count_nonzero(out != grid) < MIN_OUTPUT_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
