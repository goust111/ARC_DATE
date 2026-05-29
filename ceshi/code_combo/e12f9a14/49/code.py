# Combo variant of e12f9a14 (index 49)
# combo_spec: D1/1+D13/0
# Name: 射线重着色框架+反射弹跳

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# analysis:
#   concretization_and_trace:
#     - We keep the original visible objects: unique-color 2x2 blocks, each inside a shared-color 4x4 frame.
#       To make the fused rule legible, generation uses only corner gaps, so every launched beam is diagonal.
#     - The D1 part changes beam kinematics: a beam that would leave the grid reflects off the touched border
#       and continues with the mirrored diagonal direction instead of dying at the edge.
#     - The D13 part is fused into that reflected motion rather than left as an unrelated launch-side effect.
#       A frame side is recolored only when a reflected beam later comes back and strikes its own source-side
#       border on the 4x4 frame. The original gap determines which frame side(s) are eligible for recolor.
#     - Small trace: suppose a block has a NW corner gap. In the base task, the beam paints the NW diagonal
#       until the grid boundary and stops; the frame border stays green. In the combo task, the beam bounces
#       off the outer walls, eventually returns near the same frame, and its next step would hit the top or
#       left border of that frame. At that moment the top row plus left column recolor into the block color.
#       If another beam from the same frame later strikes a different source-side border, more sides recolor.
#   complexity_calibration:
#     - If recoloring happened immediately at launch, the combo would visually decompose into two independent
#       edits: "reflection" plus "launch tint". That is too weak a fusion, because the recolor would not
#       depend on the new reflected geometry. Delaying recolor until post-bounce frame contact forces the
#       solver to reason jointly about both changes.
#     - Generation restricts each sample to a single corner gap and increases frame spacing. This keeps long
#       reflected diagonals readable and removes non-essential multi-beam collisions from the training signal.
#     - Each accepted sample must show both signatures: at least one beam reflects multiple times, and at
#       least one reflected beam actually causes a frame recolor. That prevents degeneration to D1-only.
#   implicit_assumption_audit:
#     - Original proposal building assumed "out of bounds means beam dies"; that is replaced by boundary
#       mirroring before occupancy checks.
#     - Original D13 launch recolor could safely be implemented once in `spawn_beam()`. That assumption is
#       invalid here because the fused rule depends on later path history, so recolor is triggered only when
#       a reflected beam's next step would hit the relevant border cells of its own frame.
#     - Original D1 self-reentry guard could have blocked a literal gap revisit. The fused solver no longer
#       requires that exact revisit geometry; the visible event is the reflected return to the source-side
#       border, which avoids opening arbitrary self-overlap cases.
#     - Collision rollback must still erase only genuine beam trail cells. Recolored frame pixels are written
#       directly to `out` and never inserted into `cell_owner`, so truncation leaves them intact.
#     - `is_free()` must keep consulting immutable input `g`, not mutable `out`; otherwise recolored frame
#       borders would be mistaken for new obstacles or traversable path depending on update order.
#   placement_legality:
#     - Frames remain fully inside the grid and keep a widened bounding-box clearance so long reflected beams
#       do not immediately crash into neighboring structures.
#     - Candidate gap sets are screened with a single-beam reflection simulator on an obstacle grid whose own
#       frame already has those gaps opened. This guarantees enough runway, at least one bounce, and at least
#       one potential self-frame return before we even run the full multi-beam solver.
#     - Final acceptance still depends on the real solver so collision side effects cannot invalidate the
#       generator silently.
#   visual_observability:
#     - The output difference is human-visible: the same block color appears both in a reflected diagonal path
#       and on specific frame sides that only recolor once the beam returns.
#     - Restricting to corner gaps avoids mixing cardinal and diagonal launches, so the viewer can attribute
#       the recolor to the bounced diagonal trajectory rather than to a separate local launch rule.
#   visual_cleanliness:
#     - Wider inter-frame spacing and low gap counts reduce overdraw from long reflections.
#     - Recolor stays confined to the source frame border, so frame edits remain spatially local even when
#       the path itself is long.
#   color_inference:
#     - No hidden color mapping is introduced. Recolored border cells always take the same color as the inner
#       2x2 source block, and the generator keeps frame color distinct from both background and block colors.
#   new_solver_assumptions_enforced_by_generator:
#     - Every chosen combo has a single-beam reflected return route that strikes its own source-side border.
#     - Accepted full samples contain at least one actual post-bounce frame recolor and at least one beam
#       with two or more reflections.
#     - The solver still leaves launch gaps outside `cell_owner`, and the generator validates with the full
#       solver, so accidental foreign-gap interactions are rejected rather than relied on.


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


def _launch_frame_cells(fr0: int, fc0: int, gap: tuple[int, int]) -> set[tuple[int, int]]:
    """Border cells associated with a gap on a 4x4 frame."""
    gr, gc = gap
    lr, lc = gr - fr0, gc - fc0
    assert 0 <= lr < 4 and 0 <= lc < 4, "expected gap inside 4x4 frame coordinates"
    cells: set[tuple[int, int]] = set()
    if lr == 0:
        cells.update((fr0, fc0 + dc) for dc in range(4))
    if lr == 3:
        cells.update((fr0 + 3, fc0 + dc) for dc in range(4))
    if lc == 0:
        cells.update((fr0 + dr, fc0) for dr in range(4))
    if lc == 3:
        cells.update((fr0 + dr, fc0 + 3) for dr in range(4))
    cells.discard(gap)
    return cells


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
    self_frame_contacts: dict[int, int] = {}
    contacted_cells: dict[int, set[tuple[int, int]]] = {}
    beam_frame_color: dict[int, int] = {}
    beam_recolor_cells: dict[int, set[tuple[int, int]]] = {}
    recolored_frame_pixels: set[tuple[int, int]] = set()

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

    def paint_seed_cell(color: int, cell: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = int(color)

    def recolor_home_frame_contact(bid: int, hit_cell: tuple[int, int]) -> None:
        contacted_cells[bid].add(hit_cell)
        self_frame_contacts[bid] += 1
        for cell in beam_recolor_cells[bid]:
            r, c = cell
            if 0 <= r < h and 0 <= c < w and int(g[r, c]) == beam_frame_color[bid]:
                out[r, c] = beams[bid].color
                recolored_frame_pixels.add(cell)

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

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        frame_col: int,
        gap: tuple[int, int],
        frame_top_left: tuple[int, int],
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        reflection_counts[bid] = 0
        self_frame_contacts[bid] = 0
        contacted_cells[bid] = set()
        beam_frame_color[bid] = int(frame_col)
        fr0, fc0 = frame_top_left
        beam_recolor_cells[bid] = _launch_frame_cells(fr0, fc0, gap)
        paint_seed_cell(color, pos)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = tuple(_frame_gaps_for_block(g, bg, frame_col, br, bc))
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), frame_col, (gr, gc), (br - 1, bc - 1))

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
                if (
                    reflected or reflection_counts[bid] > 0
                ) and in_bounds(nxt) and nxt in beam_recolor_cells[bid] and int(g[nxt]) == beam_frame_color[bid]:
                    recolor_home_frame_contact(bid, nxt)
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
        "self_frame_contacts": sum(self_frame_contacts.values()),
        "beams_with_self_contact": sum(v > 0 for v in self_frame_contacts.values()),
        "unique_contact_cells": sum(len(v) for v in contacted_cells.values()),
        "recolored_frame_pixels": len(recolored_frame_pixels),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks, let them ricochet off the grid boundary, and recolor the source frame only when the bounced path comes back and hits it.

    Input structure:
    - The most common color is the background, and each monochrome 2x2 block is wrapped by a 4x4 frame of one shared wall color.
    - Frame border gaps are background cells. In this fused variant, generation uses corner gaps, so each gap launches a diagonal beam outward from its block.

    Rule:
    - Every gap seeds a beam in the inner block's color. The beam keeps moving diagonally, reflecting off any grid edge by mirroring the blocked axis instead of terminating there.
    - Away from frames, beam interactions still follow the original collision grammar: 90-degree meetings leave a blank pivot and turn toward the bisector, while 45-degree meetings merge into the alternating zigzag path.
    - A frame is no longer recolored at launch. Instead, once a reflected beam later tries to re-enter its own frame through the border side associated with its source gap, that side's frame pixels recolor into the beam color, so returns from different gaps can accumulate on the same frame.

    Key insight:
    - The important dependency is temporal: you must trace the bounced trajectory first, then apply frame recolor only at the moment the beam comes back to its own border. The frame edit is therefore evidence of the reflected path, not a separate local decoration.
    """
    return _solve(grid, collect_stats=False)


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (1, 2)
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
MIN_RECOLORED_FRAME_PIXELS = 6
GEN_RETRIES = 400

_CORNER_GAPS = ((0, 0), (0, 3), (3, 3), (3, 0))
_GAP_COMBOS = {
    1: [combo for combo in itertools.combinations(_CORNER_GAPS, 1)],
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


def _canon_seg(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _single_ray_profile(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    d: tuple[int, int],
    home_contact_cells: frozenset[tuple[int, int]],
) -> tuple[int, int, set[tuple[int, int]]]:
    h, w = grid.shape
    pos = seed
    cur_d = d
    steps = 0
    bounces = 0
    contact_cells: set[tuple[int, int]] = set()
    visited_cells = {seed}
    seen_states = {(seed, cur_d)}
    diag_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

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
            if bounces > 0 and (nr, nc) in home_contact_cells:
                contact_cells.add((nr, nc))
            break
        nxt = (nr, nc)
        if abs(cur_d[0]) + abs(cur_d[1]) == 2:
            p2 = (pos[0], pos[1] + cur_d[1])
            n2 = (pos[0] + cur_d[0], pos[1])
            if _canon_seg(p2, n2) in diag_segments:
                break
        steps += 1
        if bounced:
            bounces += 1
        if nxt in visited_cells:
            break
        if abs(cur_d[0]) + abs(cur_d[1]) == 2:
            diag_segments.add(_canon_seg(pos, nxt))
        pos = nxt
        visited_cells.add(nxt)
        state = (pos, cur_d)
        if state in seen_states:
            break
        seen_states.add(state)
    return steps, bounces, contact_cells


def _clear_gap_combos(
    obstacles: np.ndarray,
    bg: int,
    frame_color: int,
    block_color: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            probe = obstacles.copy()
            _draw_framed_block(probe, r0, c0, bg, frame_color, block_color, combo)
            profiles = []
            for gr, gc in combo:
                seed = (r0 + gr, c0 + gc)
                contact_cells = frozenset(_launch_frame_cells(r0, c0, seed))
                profiles.append(_single_ray_profile(probe, bg, seed, _gap_dir((gr, gc)), contact_cells))
            if not all(steps >= MIN_RAY_STEPS and bounces >= MIN_RAY_BOUNCES for steps, bounces, _ in profiles):
                continue
            if not any(contact_cells for _steps, _bounces, contact_cells in profiles):
                continue
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
    combos = available[1]
    return combos[int(rng.integers(len(combos)))]


def _count_recolored_frame_pixels(inp: np.ndarray, out: np.ndarray) -> int:
    bg = get_bg(inp)
    count = 0
    for col, br, bc in _find_blocks_2x2(inp, bg):
        frame_col = _infer_frame_color(inp, bg, br, bc, col)
        fr0, fc0 = br - 1, bc - 1
        fr1, fc1 = br + 2, bc + 2
        for r in range(fr0, fr1 + 1):
            for c in range(fc0, fc1 + 1):
                if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]):
                    continue
                if not (r in (fr0, fr1) or c in (fc0, fc1)):
                    continue
                if int(inp[r, c]) == frame_col and int(out[r, c]) == col:
                    count += 1
    return count


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
    for (r0, c0), color in zip(positions, block_colors):
        available = _clear_gap_combos(obstacles, bg, frame_color, int(color), r0, c0)
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
    if stats["beams_with_self_contact"] < 1:
        return None
    if _count_recolored_frame_pixels(grid, out) < MIN_RECOLORED_FRAME_PIXELS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
