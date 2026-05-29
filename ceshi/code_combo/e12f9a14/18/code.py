# Combo variant of e12f9a14 (index 18)
# combo_spec: D11/1+D20/1
# Name: 面对面放置+固定两框架

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D11/1 + D20/1
#
# 1. Concrete behavior shift:
#    - Original task: 1-5 framed 2x2 blocks, each frame may have multiple gaps, and
#      collision may or may not happen depending on free placement.
#    - Combo task: exactly two frames, each with exactly one structural gap. The two
#      gaps are chosen on the inward-facing sides/corners of the pair, so the two
#      emitted beams must collide once.
#    - Small example:
#      * Original: a left frame could emit east while a distant top frame emits north;
#        if their paths miss, the output is just two unrelated lines.
#      * Combo 90-degree case: the left frame emits east and the lower frame emits
#        north into the same meeting cell. That cell stays blank, then both traces
#        turn into a shared diagonal direction.
#      * Combo 45-degree case: a left frame emits east and a lower-left frame emits
#        north-east. They meet in one cell, keep that cell filled, and continue as a
#        zigzag band.
#      The visible difference from the base task is that source counting and free
#      layout disappear; the collision aftermath becomes the only meaningful clue.
#
# 2. Complexity calibration:
#    - The input domain is intentionally reduced to two single-gap frames, but not to
#      one fixed collision outcome. We keep two collision families from the original
#      logic: 90-degree (blank junction) and 45-degree (gapless zigzag).
#    - This keeps the variant visually distinct and avoids trivializing it into
#      "always draw one fixed diagonal" while still preventing the generator from
#      producing messy many-beam tangles.
#
# 3. Implicit-assumption audit against the original solver:
#    - `_find_blocks_2x2` originally tolerated many blocks. Here `main()` now asserts
#      there are exactly two.
#    - `_frame_gaps_for_block` originally allowed several gaps per frame. Here we
#      require exactly one structural gap per frame, so one frame always means one
#      beam.
#    - The original simulation allowed zero collisions. Here we assert that at least
#      one collision event is resolved before the loop ends.
#    - The original layout was unconstrained. Here each beam direction must point into
#      a 90-degree sector aimed at the paired frame, which is the code-level form of
#      "face-to-face placement" used by this combo.
#    - The collision resolver still assumes only 45-degree and 90-degree events are
#      valid. The generator therefore never creates head-on 180-degree pairs.
#
# 4. Placement legality:
#    - The generator works from a chosen collision cell, beam directions, and travel
#      lengths, then algebraically solves each frame top-left from
#      `seed = collision - steps * direction`.
#    - This guarantees the pair is in-bounds, non-overlapping, single-gap, and aimed
#      at one collision point. We reject any candidate whose solved frames violate the
#      facing constraint or whose post-collision continuation is too short.
#
# 5. Visual observability / cleanliness:
#    - Every sample contains only two frames and one collision region, so there is no
#      ambiguity about which objects interact.
#    - 90-degree cases show an unmistakable 1-cell hole at the turning point.
#    - 45-degree cases show a continuous elbow that turns into a two-step zigzag.
#    - The generator keeps several pre-collision cells and several post-collision
#      cells visible, so the "before hit / after hit" structure can be inferred from
#      a few examples without relying on color IDs or hidden ordering.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_SIDE_GAPS = {
    (-1, 0): [(0, 1), (0, 2)],
    (0, 1): [(1, 3), (2, 3)],
    (1, 0): [(3, 1), (3, 2)],
    (0, -1): [(1, 0), (2, 0)],
}
_CORNER_GAPS = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 3),
    (1, 1): (3, 3),
    (1, -1): (3, 0),
}

_RIGHT_ANGLE_TEMPLATES = [
    ((0, 1), (-1, 0)),
    ((-1, 0), (0, -1)),
    ((0, -1), (1, 0)),
    ((1, 0), (0, 1)),
]
_FORTY_FIVE_TEMPLATES = [
    ((0, 1), (-1, 1)),
    ((-1, 0), (-1, -1)),
    ((0, -1), (1, -1)),
    ((1, 0), (1, 1)),
]


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
    assert len(frame_colors) == 1, "expected one shared frame color around each block"
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
        assert dr != 0 or dc != 0, "gap must lie on the border"
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


def _gap_dir_for_block(br: int, bc: int, gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
    dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
    d = _normalize_dir(dr, dc)
    assert d != (0, 0), "expected an outward-facing frame gap"
    return d


def _frame_centroid(br: int, bc: int) -> tuple[float, float]:
    return br + 0.5, bc + 0.5


def _beam_faces_other(br: int, bc: int, d: tuple[int, int], other_br: int, other_bc: int) -> bool:
    cr, cc = _frame_centroid(br, bc)
    orr, occ = _frame_centroid(other_br, other_bc)
    aim = _normalize_dir(int(np.sign(orr - cr)), int(np.sign(occ - cc)))
    return aim != (0, 0) and _dir_diff(d, aim) <= 2


def _extract_layout(g: np.ndarray) -> list[dict]:
    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    infos: list[dict] = []
    for color, br, bc in sorted(blocks, key=lambda x: (x[1], x[2], x[0])):
        frame_color = _infer_frame_color(g, bg, br, bc, color)
        gaps = _frame_gaps_for_block(g, bg, frame_color, br, bc)
        info = {
            "color": int(color),
            "br": br,
            "bc": bc,
            "frame_color": frame_color,
            "gaps": gaps,
        }
        if len(gaps) == 1:
            info["dir"] = _gap_dir_for_block(br, bc, gaps[0])
            info["gap"] = gaps[0]
        infos.append(info)
    return infos


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Two inward-facing framed blocks emit exactly two beams, and the output is determined by their forced collision.

    Input structure:
    - The background is the most common color. Exactly two 2x2 monochrome blocks sit inside 4x4 frames of one shared wall color.
    - Each frame has exactly one structural gap, and each gap lies on the side or corner pointing toward the other frame.

    Rule:
    - Each gap emits one beam in its outward direction, using the inner block color.
    - The two beams travel only through background cells and must collide once.
    - A 90-degree collision leaves a 1-cell blank junction and redirects both beams along the angle bisector.
    - A 45-degree collision keeps the meeting point filled and converts the pair into the original zigzag continuation.

    Key insight:
    - Because frame count, gap count, and pair layout are fixed, the puzzle reduces to recognizing which collision signature appears: a blank-corner redirect or a continuous zigzag merge.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    layout = _extract_layout(g)
    assert len(layout) == 2, "combo variant expects exactly two framed 2x2 blocks"
    for info in layout:
        assert len(info["gaps"]) == 1, "combo variant expects exactly one structural gap per frame"
    d0 = layout[0]["dir"]
    d1 = layout[1]["dir"]
    assert _dir_diff(d0, d1) in (1, 2), "generator must avoid head-on or obtuse pairs"
    assert _beam_faces_other(layout[0]["br"], layout[0]["bc"], d0, layout[1]["br"], layout[1]["bc"])
    assert _beam_faces_other(layout[1]["br"], layout[1]["bc"], d1, layout[0]["br"], layout[0]["bc"])

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    beams: dict[int, _Beam] = {}
    next_bid = 0
    collision_happened = False

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

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "beam seed must start at the gap cell"
        assert pos not in cell_owner, "gap cells must be unique"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for info in layout:
        spawn_beam(info["color"], info["gap"], (info["dir"],))

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
            assert partner is None, "expected at most one lockstep partner"
            partner = oid
        return partner

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        nonlocal collision_happened
        collision_happened = True
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "only 45-degree and 90-degree collisions are valid"

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "diagonal 90-degree pairs should cross geometrically"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
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
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                beams[gid].active = True
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must be cardinal+diagonal"

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
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free zigzag offset cell"
        bv.path.append(adj)
        bv.enter_dirs.append(u)
        bv.pos = adj
        paint_cell(bid_v, adj, u)

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True

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
        nonlocal collision_happened
        collision_happened = True
        da = prop[a][1]
        db = prop[b][1]
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "crossing diagonals should produce a forward direction"
        beams[a].pattern = (u,)
        beams[a].phase = 0
        beams[b].pattern = (u,)
        beams[b].phase = 0
        beams[a].active = True
        beams[b].active = True

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        nonlocal collision_happened
        collision_happened = True
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
            raise AssertionError("expected a partner-owned diagonal segment")

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert u != (0, 0), "segment cross should produce a forward direction"

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must stop only at border or blocked collision cell"
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
                resolve_cross_collision(a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free application step"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    assert collision_happened, "combo variant requires the two beams to collide"
    return out


# --------------- generator (combo D11/1 + D20/1) ---------------

BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
GRID_SIZE_RANGE = (16, 27)
PLACEMENT_MARGIN = 2
FRAME_GAP_CLEARANCE = 1
MIN_TO_COLLISION = 3
MAX_TO_COLLISION = 8
MIN_CHANGED_PIXELS = 12
MIN_COLLISION_EFFECT = 4
GEN_RETRIES = 400


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gap: tuple[int, int],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    patch[gap] = bg
    paste(grid, patch, r0, c0)


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=2, replace=False))
    return bg, frame_color, block_colors


def _frame_bbox(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    return r0, c0, r0 + 3, c0 + 3


def _frames_separated(a: tuple[int, int], b: tuple[int, int], gap: int = FRAME_GAP_CLEARANCE) -> bool:
    ar0, ac0, ar1, ac1 = _frame_bbox(a)
    br0, bc0, br1, bc1 = _frame_bbox(b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _gap_for_dir(rng: np.random.Generator, d: tuple[int, int]) -> tuple[int, int]:
    if d in _SIDE_GAPS:
        choices = _SIDE_GAPS[d]
        return choices[int(rng.integers(len(choices)))]
    return _CORNER_GAPS[d]


def _seed_from_collision(collision: tuple[int, int], d: tuple[int, int], steps: int) -> tuple[int, int]:
    return collision[0] - steps * d[0], collision[1] - steps * d[1]


def _top_left_from_collision(
    collision: tuple[int, int],
    d: tuple[int, int],
    gap_local: tuple[int, int],
    steps: int,
) -> tuple[int, int]:
    seed = _seed_from_collision(collision, d, steps)
    return seed[0] - gap_local[0], seed[1] - gap_local[1]


def _in_bounds_frame(top_left: tuple[int, int], n: int) -> bool:
    r0, c0 = top_left
    return PLACEMENT_MARGIN <= r0 and PLACEMENT_MARGIN <= c0 and r0 + 3 < n - PLACEMENT_MARGIN and c0 + 3 < n - PLACEMENT_MARGIN


def _forward_room(collision: tuple[int, int], d: tuple[int, int], n: int) -> int:
    r, c = collision
    steps = 0
    while True:
        r += d[0]
        c += d[1]
        if not (0 <= r < n and 0 <= c < n):
            return steps
        steps += 1


def _straight_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    layout = _extract_layout(g)
    for info in layout:
        r, c = info["gap"]
        dr, dc = info["dir"]
        color = info["color"]
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == bg:
            out[r, c] = color
            r += dr
            c += dc
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    collision_kind = "45" if float(rng.random()) < 0.5 else "90"
    dirs = _FORTY_FIVE_TEMPLATES if collision_kind == "45" else _RIGHT_ANGLE_TEMPLATES
    d1, d2 = dirs[int(rng.integers(len(dirs)))]
    g1 = _gap_for_dir(rng, d1)
    g2 = _gap_for_dir(rng, d2)
    t1 = int(rng.integers(MIN_TO_COLLISION, MAX_TO_COLLISION + 1))
    t2 = int(rng.integers(MIN_TO_COLLISION, MAX_TO_COLLISION + 1))
    collision = (
        int(rng.integers(4, n - 4)),
        int(rng.integers(4, n - 4)),
    )

    top_lefts = [
        _top_left_from_collision(collision, d1, g1, t1),
        _top_left_from_collision(collision, d2, g2, t2),
    ]
    if not all(_in_bounds_frame(p, n) for p in top_lefts):
        return None
    if not _frames_separated(top_lefts[0], top_lefts[1]):
        return None

    post_dir = _normalize_dir(d1[0] + d2[0], d1[1] + d2[1])
    if post_dir == (0, 0):
        return None
    if collision_kind == "90":
        if _forward_room(collision, post_dir, n) < 3:
            return None
    else:
        if _forward_room(collision, post_dir, n) < 4:
            return None

    bg, frame_color, block_colors = _pick_colors(rng)
    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gap in zip(top_lefts, block_colors, (g1, g2)):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gap)

    try:
        layout = _extract_layout(grid)
        if len(layout) != 2 or any(len(info["gaps"]) != 1 for info in layout):
            return None
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < MIN_CHANGED_PIXELS:
        return None
    straight = _straight_output(grid)
    if np.count_nonzero(out != straight) < MIN_COLLISION_EFFECT:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo D11/1+D20/1 input")
