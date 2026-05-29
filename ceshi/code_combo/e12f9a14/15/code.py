# Combo variant of e12f9a14 (index 15)
# combo_spec: D10/2+D16/0
# Name: 缺口仅在角落+角落多方向发射

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D10/2 + D16/0
# =========================================
# a) Concrete behavior and trace
#    - Original object: a 2x2 monochrome core inside a 4x4 frame, with one or more bg holes
#      anywhere on the border. Every hole launched exactly one beam in the outward direction
#      implied by that hole's side/corner position.
#    - Fused object: each frame has exactly one structural hole, and that hole must be one of
#      the four frame corners.
#    - Original trace for a NW-corner hole at absolute cell (r, c):
#        old launch set = {(-1, -1)}
#      Fused trace for the same visible corner hole:
#        new launch set = {(-1, -1), (-1, +1)}
#      So the corner gap paints a shared seed corner and then fans into a symmetric pair above
#      the frame. Rotating this rule gives:
#        NW -> {NW, NE}, NE -> {NE, SE}, SE -> {SE, SW}, SW -> {SW, NW}.
#    - Difference from the source task:
#      the input no longer asks the solver to infer "which border cell(s) emit"; it asks the
#      solver to infer "which corner is missing, then expand that corner into a 2-ray fan".
#      Downstream collision handling remains the original one, so the new part is truly fused
#      into the old dynamics rather than replacing them.
# b) Complexity calibration
#    - We keep only four corner categories and a fixed fan size of two. That is enough to make
#      the D10 change visually obvious without creating too many emission modes.
#    - Every frame always emits exactly two rays, so the new rule repeats within each sample and
#      does not rely on rare corner cases.
#    - The generator keeps 1-5 frames and unique block colors, so multiple symmetric fans can
#      coexist without becoming ambiguous.
# c) Hidden-assumption audit against the original solver
#    - Original assumption: every structural gap is an independent beam seed occupying a unique
#      cell in the beam-ownership map. Broken here because one corner gap must spawn two beams.
#      Fix: beams start from the same gap position but the shared seed cell is treated as a
#      painted launch marker, not as exclusive dynamic ownership.
#    - Original assumption: gaps may appear on sides or corners. Broken by D16/0.
#      Fix: `_corner_gap_for_block()` requires exactly one structural gap and it must be a corner.
#    - Original assumption in `spawn_beam()`: the seed cell is immediately owned by one beam.
#      Broken by the 2-beam fan. Fix: spawn heads at the gap position without painting it into
#      `cell_owner`; normal ownership starts on the first moved cell.
#    - Collision code assumes only 45-degree and 90-degree encounters after launch. This still
#      holds because all emitted directions are diagonal and the original simulator already
#      supports diagonal-diagonal, diagonal-cardinal, and post-turn interactions. The generator
#      validates accepted inputs by running the fused simulator.
# d) Placement legality
#    - Frames remain 4x4 shells around 2x2 cores, with the original non-overlap spacing.
#    - A chosen corner gap is legal only if both emitted rays have at least two clear bg steps
#      in the obstacle grid built from closed frames.
#    - The generator additionally rejects cases where the dynamic simulation ever paints back
#      onto a launch corner; that would make the shared-seed abstraction ambiguous.
# e) Visual observability
#    - A single frame already shows the fused rule clearly: one missing corner creates a 2-ray
#      fan rather than one diagonal line.
#    - Across a few samples, different missing corners rotate the fan direction in a directly
#      visible way. No hidden dependence on numeric color ids is used.
# f) Visual cleanliness
#    - Limiting each frame to one corner gap avoids clutter from 4-8 launches per object.
#    - The 2-cell bbox clearance between frames is preserved so the small frames stay readable
#      against the fan geometry.
#    - Samples that self-hit seed corners are rejected because those produce visually muddy
#      ownership around the shared launch cell.
# g) Generator/solver contract and new assumptions
#    - `main()` assumes exactly one structural corner gap per frame and uses the corner alone to
#      choose the two launch directions.
#    - `generate_input()` constructs only such frames, ensures both fan branches have runway, and
#      checks the fused simulator succeeds with `seed_hits == 0`.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))
_CORNER_EMIT_DIRS = {
    (0, 0): ((-1, -1), (-1, 1)),
    (0, 3): ((-1, 1), (1, 1)),
    (3, 3): ((1, 1), (1, -1)),
    (3, 0): ((1, -1), (-1, -1)),
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


def _corner_gap_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> tuple[int, int]:
    gaps = _frame_gaps_for_block(g, bg, wall, br, bc)
    fr0, fc0 = br - 1, bc - 1
    local = {(r - fr0, c - fc0): (r, c) for r, c in gaps}
    corner_hits = [local[p] for p in _CORNER_GAPS if p in local]
    assert len(gaps) == 1, "fused variant expects exactly one structural gap per frame"
    assert len(corner_hits) == 1, "fused variant expects the unique gap to be a corner"
    return corner_hits[0]


def _corner_emit_dirs(br: int, bc: int, gap: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    fr0, fc0 = br - 1, bc - 1
    local = (gap[0] - fr0, gap[1] - fc0)
    assert local in _CORNER_EMIT_DIRS, "expected a corner gap"
    return _CORNER_EMIT_DIRS[local]


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
        # path[0] is the shared launch corner; it is not dynamically owned.
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
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
    launch_cells: list[tuple[int, tuple[int, int]]] = []
    seed_positions: set[tuple[int, int]] = set()

    stats = {
        "frame_count": len(blocks),
        "beam_count": 0,
        "collision_events": 0,
        "seed_hits": 0,
    }

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        if cell in seed_positions:
            stats["seed_hits"] += 1
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
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gap = _corner_gap_for_block(g, bg, frame_col, br, bc)
        launch_cells.append((col, gap))
        seed_positions.add(gap)
        for d in _corner_emit_dirs(br, bc, gap):
            spawn_beam(col, gap, (d,))

    stats["beam_count"] = len(beams)
    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        """Return an active beam adjacent to bid that is moving in lockstep."""
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
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams should cross geometrically"
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
        assert is_card_a != is_card_b, "45-degree collision must involve one cardinal and one diagonal beam"

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit block or frame"
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
            stats["collision_events"] += 1
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
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    for color, pos in launch_cells:
        if pos in blocked:
            continue
        if cell_owner.get(pos) is None and int(out[pos[0], pos[1]]) == bg:
            out[pos[0], pos[1]] = int(color)

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit two-way diagonal fans from corner gaps on framed 2x2 blocks, then keep the original collision dynamics.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block surrounded by a shared 4x4 frame color.
    - Every frame has exactly one missing border cell, and that missing cell is always a corner of the 4x4 border.

    Rule:
    - A corner gap paints its own corner and launches two beams of the block color, one along each adjacent diagonal direction:
      NW -> {NW, NE}, NE -> {NE, SE}, SE -> {SE, SW}, SW -> {SW, NW}.
    - After launch, the beams follow the source task's motion rules: they keep extending through background, 90-degree meetings leave a 1-cell blank pivot and turn toward the bisector, and 45-degree meetings merge into the alternating zigzag continuation.

    Key insight:
    - The fused change is entirely in the launch decoding: a corner hole is no longer a single outward ray but a symmetric 2-ray fan. Once those two rays exist, all later interactions are resolved with the original beam physics.
    """
    out, _stats = _simulate(grid)
    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 8, 2: 14, 5: 20}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_FRAME = 4
GEN_RETRIES = 400


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
    gap: tuple[int, int] | None,
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    if gap is not None:
        patch[gap[0], gap[1]] = bg
    paste(grid, patch, r0, c0)


def _trace_ray_steps(grid: np.ndarray, bg: int, seed: tuple[int, int], d: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = seed
    steps = 0
    nr, nc = r + d[0], c + d[1]
    while 0 <= nr < h and 0 <= nc < w:
        steps += 1
        if int(grid[nr, nc]) != bg:
            return -1
        nr += d[0]
        nc += d[1]
    return steps


def _corner_gap_is_clear(obstacles: np.ndarray, bg: int, r0: int, c0: int, gap: tuple[int, int]) -> bool:
    seed = (r0 + gap[0], c0 + gap[1])
    for d in _CORNER_EMIT_DIRS[gap]:
        steps = _trace_ray_steps(obstacles, bg, seed, d)
        if steps < MIN_RAY_STEPS:
            return False
    return True


def _valid_corner_gaps(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> list[tuple[int, int]]:
    return [gap for gap in _CORNER_GAPS if _corner_gap_is_clear(obstacles, bg, r0, c0, gap)]


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
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

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), None)

    chosen_gaps: list[tuple[int, int]] = []
    for r0, c0 in positions:
        available = _valid_corner_gaps(obstacles, bg, r0, c0)
        if not available:
            return None
        chosen_gaps.append(available[int(rng.integers(len(available)))])

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gap in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gap)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid)
    except Exception:
        return None

    if stats["beam_count"] != 2 * block_count:
        return None
    if stats["seed_hits"] != 0:
        return None
    if np.count_nonzero(out != grid) < block_count * MIN_BEAM_PIXELS_PER_FRAME:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
