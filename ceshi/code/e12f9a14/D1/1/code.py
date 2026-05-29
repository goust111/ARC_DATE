# Variant of e12f9a14
# Dimension: D1 — 射线发射方向 (path)
# Change: 对角线射线（diagonal_ray）——彩色像素通过绿色框架缺口沿45度对角线方向向外发射，方向由缺口相对于内部像素的... → 反射弹跳路径

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D1 variant:
# design_analysis:
#   concrete_behavior:
#     - Keep the same visible objects: 2x2 monochrome color blocks inside a shared-color 4x4 frame.
#     - Keep the same exit encoding: each bg gap on the frame border is a launch point whose side/corner
#       determines the outward direction.
#     - Only the path law changes: when a beam would leave the grid, it reflects off that grid edge and
#       continues from the same head cell with the mirrored direction. Hitting a corner flips both axes.
#     - Small example:
#       original: a NE beam at row 1 moving to row -1 stops after painting the topmost reachable cell.
#       variant: the same NE beam reaches the top edge, flips vertical sign, then continues as SE and can
#       later hit the right edge, flip horizontal sign, and continue as SW. The output therefore shows a
#       clear V / zigzag bounce path instead of a truncated diagonal.
#   complexity_calibration:
#     - Reflection is the only new conceptual ingredient; the original collision rules remain unchanged.
#     - To make the change observable, generation prefers corner gaps and rejects inputs whose solved paths
#       do not include multiple reflections. This avoids degenerate samples that would look identical to the
#       non-bouncing task.
#     - Gap count is reduced to 1-2 per frame in generation so the outputs stay legible even though paths
#       are longer.
#   implicit_assumption_audit:
#     - Original proposal building assumed "next step out of bounds => beam dies". That assumption is no
#       longer valid, so proposal generation must mirror the direction before deciding whether a move exists.
#     - Original solver could safely ignore self-intersections because a straight beam terminated at the
#       boundary. With reflections, a beam can revisit its own trail, so the variant explicitly stops a beam
#       when its next reflected step would enter one of its own cells or cross one of its own diagonal
#       segments. This is a compatibility guard, not a new visible encoding.
#     - Collision resolvers depend on the current move direction. Therefore the reflected direction must be
#       visible to collision handling in the same iteration.
#   placement_legality:
#     - Frames still obey non-overlap and clearance constraints.
#     - Generator validates each candidate gap against a static obstacle grid with full frames, ensuring the
#       corresponding single beam has enough open runway and at least one bounce even before inter-beam
#       interactions are considered.
#     - The final candidate is accepted only if the actual solver run performs multiple reflections.
#   visual_observability:
#     - Reflection is human-visible because at least one beam must bounce multiple times in the solved grid.
#     - Corner-launched rays are favored so the reflected geometry is diagonal and easy to distinguish from
#       the static rectangular frames.
#   visual_cleanliness:
#     - Larger frame spacing and fewer simultaneous gaps keep long reflected trajectories readable.
#     - Colors are unchanged: beams still inherit the inner block color, so the only new cue is path shape.
#   color_inference:
#     - No color remapping is introduced. Beam color stays tied to the framed block color, exactly as in
#       the base task, so the variant remains shape-driven rather than id-driven.
#   solver_assumptions_enforced_by_generator:
#     - At least one solved beam reflects twice or more.
#     - The solver may stop a beam on self-reentry; generation therefore uses roomy layouts that usually
#       produce long fresh stretches before any self-contact.


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
        b.active = True

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        reflection_counts[bid] = 0
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide at one cell"
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

    max_iters = h * w * 400
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
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
            a, b = sorted((bid, obid))
            cross_pairs.add((a, b))

        seg_cross: set[tuple[int, int]] = set()
        dead_self_cross: set[int] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            p2 = (pr, pc + d[1])
            n2 = (pr + d[0], pc)
            obid = diag_segment_owner(p2, n2)
            if obid is None:
                continue
            if obid == bid:
                dead_self_cross.add(bid)
                continue
            seg_cross.add((bid, obid))

        if dead_self_cross:
            for bid in dead_self_cross:
                beams[bid].active = False
                prop.pop(bid, None)
            if not prop:
                break
            cross_pairs = {pair for pair in cross_pairs if pair[0] in prop and pair[1] in prop}
            seg_cross = {pair for pair in seg_cross if pair[0] in prop}

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
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    if not collect_stats:
        return out

    stats = {
        "beam_count": len(beams),
        "total_reflections": sum(reflection_counts.values()),
        "reflected_beams": sum(v > 0 for v in reflection_counts.values()),
        "max_reflections_on_beam": max(reflection_counts.values(), default=0),
        "reflection_counts": dict(reflection_counts),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from framed 2x2 blocks, letting the beams bounce off the grid boundary before applying the original collision rules.

    Input structure:
    - The background is the most frequent color, and each colored 2x2 block sits inside a 4x4 frame of one shared wall color.
    - Missing wall pixels on the frame border are beam exits; corner exits launch diagonal rays and side exits launch cardinal rays.

    Rule:
    - Every gap seeds a beam in the inner block's color, moving outward from the frame exactly as in the base task.
    - When a beam would step outside the grid, it reflects off that boundary and keeps traveling with the mirrored direction instead of stopping.
    - After any reflection, all pre-existing beam interaction rules still apply: 90-degree meetings create a blank collision gap and redirect to the bisector, while 45-degree meetings merge into the alternating zigzag path.

    Key insight:
    - The visible novelty is not a new object type but the longer mirrored trajectories: you must mentally continue a beam through one or more edge bounces before resolving the same collision mechanics as the original task.
    """
    return _solve(grid, collect_stats=False)


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 10, 2: 16, 4: 22}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 3
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 6
MIN_RAY_BOUNCES = 1
MIN_TOTAL_REFLECTIONS = 2
MIN_MAX_REFLECTIONS_ON_BEAM = 2
MIN_BEAM_PIXELS_PER_GAP = 5
GEN_RETRIES = 400

_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))
_GAP_COMBOS = {
    1: [combo for combo in itertools.combinations(_CORNER_GAPS, 1)],
    2: [combo for combo in itertools.combinations(_CORNER_GAPS, 2)],
}


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
    grid: np.ndarray, r0: int, c0: int, bg: int,
    frame_color: int, block_color: int, gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _single_ray_profile(
    grid: np.ndarray, bg: int, seed: tuple[int, int], d: tuple[int, int]
) -> tuple[int, int]:
    h, w = grid.shape
    pos = seed
    cur_d = d
    steps = 0
    bounces = 0
    visited_cells = {seed}
    seen_states = {(seed, cur_d)}

    for _ in range(h * w * 8):
        nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
        bounced = False
        if not (0 <= nr < h and 0 <= nc < w):
            flip_r = not (0 <= nr < h)
            flip_c = not (0 <= nc < w)
            cur_d = _reflect_dir(cur_d, flip_r, flip_c)
            nr, nc = pos[0] + cur_d[0], pos[1] + cur_d[1]
            bounced = True
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) != bg:
            break
        nxt = (nr, nc)
        if nxt in visited_cells:
            break
        pos = nxt
        visited_cells.add(nxt)
        steps += 1
        if bounced:
            bounces += 1
        state = (pos, cur_d)
        if state in seen_states:
            break
        seen_states.add(state)
    return steps, bounces


def _clear_gap_combos(
    obstacles: np.ndarray, bg: int, r0: int, c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            profiles = [
                _single_ray_profile(obstacles, bg, (r0 + gr, c0 + gc), _gap_dir((gr, gc)))
                for gr, gc in combo
            ]
            if all(steps >= MIN_RAY_STEPS and bounces >= MIN_RAY_BOUNCES for steps, bounces in profiles):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 10
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _choose_gap_combo(
    available: dict[int, list[tuple[tuple[int, int], ...]]], rng: np.random.Generator
) -> tuple[tuple[int, int], ...]:
    if 1 in available and (2 not in available or rng.random() < 0.75):
        combos = available[1]
    else:
        size = max(available)
        combos = available[size]
    return combos[int(rng.integers(len(combos)))]


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

    chosen_gaps = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        chosen_gaps.append(_choose_gap_combo(available, rng))

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, collect_stats=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["total_reflections"] < MIN_TOTAL_REFLECTIONS:
        return None
    if stats["max_reflections_on_beam"] < MIN_MAX_REFLECTIONS_ON_BEAM:
        return None
    if stats["reflected_beams"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D1 input")
