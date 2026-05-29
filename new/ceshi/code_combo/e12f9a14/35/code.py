# Combo variant of e12f9a14 (index 35)
# combo_spec: D8/1+D9/1
# Name: 所有碰撞均有间隙+宽度递增

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D8/1 + D9/1
# a) Concrete behavior and trace:
#    - The source task already has framed 2x2 color blocks whose gaps emit 1-pixel beam spines.
#    - I keep collision detection on those 1-pixel spines, because the original topology logic
#      (`cell_owner`, segment-cross checks, truncation) depends on single ownership per spine cell.
#    - New fusion rule:
#      1. Every collision widens the participating future trajectory by +1.
#      2. Every collision also leaves a permanent blank gap at the interaction site.
#      3. Angle is encoded by gap length: 90-degree collisions leave one skipped collision unit,
#         while 45-degree collisions leave two skipped collision units.
#    - Concrete example:
#      a) Before fusion, an east-going beam and a south-going beam meet at one cell, that cell is
#         erased, and the outgoing SE continuation stays thin.
#      b) After fusion, the same pivot stays blank, but the post-collision SE continuation now uses
#         width 2. If that widened segment collides again later, the next continuation uses width 3,
#         so the later blank pivot is also carved with a larger visible hole.
#      c) Before fusion, a cardinal+diagonal 45-degree merge paints the collision cell immediately
#         and continues as a zigzag. After fusion, both immediate merge cells are skipped, producing
#         a 2-step break before the widened zigzag resumes.
# b) Complexity calibration:
#    - Width still has a tiny discrete state space: 1, 2, 3, ...
#    - Gap encoding also has only two observable categories: 1-unit gaps for 90-degree events and
#      2-unit gaps for 45-degree events. This is enough to be visible without becoming brittle.
#    - The generator therefore forces collisions, wide ribbons, and both gap categories so samples
#      do not collapse back to the original task.
# c) Hidden assumptions audit:
#    - The original solver assumes each simulated spine cell has one owner. That still holds because
#      widening is deferred to rendering and virtual 45-degree gap cells are never added to the path.
#    - `truncate_beam()` assumes stored path/tick/width histories can be cut together. That remains
#      valid because gap skips move only `pos`, not historical path ownership.
#    - The old 45-degree branch assumed the collision cell must be painted to seed the zigzag. This
#      fusion breaks that assumption, so the branch now places two virtual beam heads on skipped gap
#      cells and resumes the zigzag one beat later.
#    - Cross/segment-cross events previously relied on the geometric crossing itself to imply a gap.
#      Here I record an explicit 1-unit blank anchor as well, so every collision class participates
#      in the same "always leave a gap" rule.
# d) Placement legality:
#    - Frames still use non-overlapping 4x4 boxes and bg-only runways from every gap.
#    - The generator validates candidates by running this exact solver and rejecting any case without
#      sufficient space for widened paint and visible gap footprints.
# e) Visual observability:
#    - No invisible color ordering is used. The only new information comes from shape: trajectories
#      thicken after collisions, and 45-degree breaks are visibly longer than 90-degree breaks.
#    - Because each accepted sample contains at least one 45-degree and one 90-degree collision,
#      a human can compare short-vs-long breaks across just a few examples.
# f) Visual cleanliness:
#    - Wide paint is still clipped to original background cells, so frames and 2x2 source blocks stay
#      legible. Gap carving also only blanks original background cells, preventing accidental frame loss.
# g) Color inference:
#    - Colors do not change. The block color still controls the beam color, while collision count and
#      collision angle control only ribbon width and gap length.
# h) New solver assumptions introduced here:
#    - The widened blank areas are rendered as square footprints centered on skipped spine cells. The
#      generator therefore rejects cramped scenes where those footprints would almost entirely clip away.
#    - 45-degree gaps depend on two free skip cells (`cell` and `adj`) existing at the merge site; the
#      generator relies on solver validation so only legal topologies survive.


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


def _centered_offsets(width: int) -> range:
    lo = -((width - 1) // 2)
    hi = width // 2
    return range(lo, hi + 1)


def _footprint(cell: tuple[int, int], enter_dir: tuple[int, int], width: int) -> list[tuple[int, int]]:
    r, c = cell
    if width <= 1:
        return [cell]
    if abs(enter_dir[0]) + abs(enter_dir[1]) == 1:
        offs = _centered_offsets(width)
        if enter_dir[0] != 0:
            return [(r, c + dc) for dc in offs]
        return [(r + dr, c) for dr in offs]
    offs = list(_centered_offsets(width))
    return [(r + dr, c + dc) for dr in offs for dc in offs]


def _gap_footprint(cell: tuple[int, int], width: int) -> list[tuple[int, int]]:
    return _footprint(cell, (1, 1), width)


class _Beam:
    __slots__ = (
        "bid",
        "color",
        "pos",
        "pattern",
        "phase",
        "path",
        "enter_dirs",
        "width",
        "widths",
        "ticks",
        "active",
    )

    def __init__(
        self,
        bid: int,
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        tick: int,
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.width = 1
        self.widths = [1]
        self.ticks = [tick]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    spine = g.copy()
    blocked: set[tuple[int, int]] = set()
    gap_events: list[tuple[str, int, tuple[tuple[int, int], ...]]] = []

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0
    next_tick = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    stats: dict[str, int] = {
        "collision_events": 0,
        "collision_45": 0,
        "collision_90": 0,
        "collision_cross": 0,
        "collision_segcross": 0,
        "gap_events_45": 0,
        "gap_events_90": 0,
        "gap_cells_45": 0,
        "gap_cells_90": 0,
        "gap_pixels_45": 0,
        "gap_pixels_90": 0,
        "max_width": 1,
        "wide_spine_cells": 0,
        "wide_pixels": 0,
    }

    def paint_spine_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        spine[r, c] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_spine_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        spine[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_spine_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.widths = b.widths[:keep_len]
        b.ticks = b.ticks[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid, next_tick
        assert int(spine[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, next_tick)
        next_tick += 1
        paint_spine_cell(bid, pos, pattern[0])

    def append_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        nonlocal next_tick
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.widths.append(b.width)
        b.ticks.append(next_tick)
        next_tick += 1
        b.pos = cell
        paint_spine_cell(bid, cell, enter_dir)

    def set_virtual_head(bid: int, pos: tuple[int, int]) -> None:
        beams[bid].pos = pos
        beams[bid].active = True

    def clear_painted_cell(cell: tuple[int, int]) -> None:
        owner = cell_owner.get(cell)
        if owner is None:
            return
        idx = cell_index[cell]
        assert idx > 0, "expected collision cleanup only on non-seed spine cells"
        truncate_beam(owner, idx)
        unpaint_spine_cell(cell)

    def record_gap(kind: str, width: int, cells: tuple[tuple[int, int], ...]) -> None:
        for cell in cells:
            blocked.add(cell)
            if cell_owner.get(cell) is not None:
                clear_painted_cell(cell)
        gap_events.append((kind, width, cells))
        if kind == "45":
            stats["gap_events_45"] += 1
            stats["gap_cells_45"] += len(cells)
        else:
            stats["gap_events_90"] += 1
            stats["gap_cells_90"] += len(cells)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

    def is_lockstep_adjacent(aid: int, bid: int) -> bool:
        a = beams.get(aid)
        b = beams.get(bid)
        if a is None or b is None or not a.active or not b.active:
            return False
        if a.pattern != b.pattern or a.phase != b.phase:
            return False
        d = a.next_dir()
        if b.next_dir() != d:
            return False
        if abs(a.pos[0] - b.pos[0]) + abs(a.pos[1] - b.pos[1]) != 1:
            return False
        if (a.pos[0] + d[0], a.pos[1] + d[1]) == b.pos:
            return False
        if (b.pos[0] + d[0], b.pos[1] + d[1]) == a.pos:
            return False
        return True

    def lockstep_group(seed_ids: set[int]) -> set[int]:
        group = {bid for bid in seed_ids if bid in beams}
        changed = True
        while changed:
            changed = False
            for oid, ob in beams.items():
                if oid in group or not ob.active:
                    continue
                if any(is_lockstep_adjacent(oid, gid) for gid in group):
                    group.add(oid)
                    changed = True
        return group

    def widen(group: set[int]) -> int:
        width_after = 1
        for gid in group:
            beams[gid].width += 1
            width_after = max(width_after, beams[gid].width)
            stats["max_width"] = max(stats["max_width"], beams[gid].width)
        return width_after

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

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
        else:
            db = cell_enter[cell]

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not share one cell"
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = lockstep_group({a, b})
            gap_width = widen(group)
            record_gap("90", gap_width, (cell,))
            for gid in group:
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                beams[gid].active = True
            stats["collision_events"] += 1
            stats["collision_90"] += 1
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

        diag_head = beams[bid_v].pos
        cell_gap = cell
        adj_gap = (diag_head[0] + u[0], diag_head[1] + u[1])
        assert in_bounds(adj_gap) and is_free(adj_gap) and cell_owner.get(adj_gap) is None, "expected free second gap cell"

        gap_width = widen({bid_u, bid_v})
        record_gap("45", gap_width, (cell_gap, adj_gap))

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        set_virtual_head(bid_u, cell_gap)
        set_virtual_head(bid_v, adj_gap)
        stats["collision_events"] += 1
        stats["collision_45"] += 1

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        db = prop[b][1]
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        gap_anchor = min(prop[a][0], prop[b][0])
        group = lockstep_group({a, b})
        gap_width = widen(group)
        record_gap("90", gap_width, (gap_anchor,))
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            beams[gid].active = True
        stats["collision_events"] += 1
        stats["collision_cross"] += 1

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        if (
            cell_owner.get(n2) == b
            and cell_index.get(n2, 0) > 0
            and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1])
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            gap_anchor = n2
            truncate_beam(b, cell_index[n2])
        elif (
            cell_owner.get(p2) == b
            and cell_index.get(p2, 0) > 0
            and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1])
        ):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            gap_anchor = p2
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert u != (0, 0), "expected segment crossing to resolve to a forward direction"

        group = lockstep_group({a, b})
        gap_width = widen(group)
        record_gap("90", gap_width, (gap_anchor,))
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            beams[gid].active = True
        stats["collision_events"] += 1
        stats["collision_segcross"] += 1

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

    render = g.copy()
    draw_events: list[tuple[int, int, int, tuple[int, int], tuple[int, int], int]] = []
    for bid, b in beams.items():
        for idx, cell in enumerate(b.path):
            draw_events.append((b.ticks[idx], bid, idx, cell, b.enter_dirs[idx], b.widths[idx]))
    draw_events.sort()

    wide_pixels: set[tuple[int, int]] = set()
    for _tick, bid, _idx, cell, enter_dir, width in draw_events:
        for pix in _footprint(cell, enter_dir, width):
            if not in_bounds(pix):
                continue
            if int(g[pix[0], pix[1]]) != bg:
                continue
            if render[pix[0], pix[1]] != bg:
                continue
            render[pix[0], pix[1]] = beams[bid].color
            if width > 1:
                wide_pixels.add(pix)

    gap_pixels_45: set[tuple[int, int]] = set()
    gap_pixels_90: set[tuple[int, int]] = set()
    for kind, width, cells in gap_events:
        for cell in cells:
            for pix in _gap_footprint(cell, width):
                if not in_bounds(pix):
                    continue
                if int(g[pix[0], pix[1]]) != bg:
                    continue
                render[pix[0], pix[1]] = bg
                if kind == "45":
                    gap_pixels_45.add(pix)
                else:
                    gap_pixels_90.add(pix)

    stats["wide_spine_cells"] = sum(1 for b in beams.values() for width in b.widths if width > 1)
    stats["wide_pixels"] = len(wide_pixels)
    stats["gap_pixels_45"] = len(gap_pixels_45)
    stats["gap_pixels_90"] = len(gap_pixels_90)
    return render, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace framed-block beams with collision-driven widening and angle-coded permanent breaks.

    Input structure:
    - The most common color is background. Several monochrome 2x2 blocks sit inside 4x4 frames of one shared wall color.
    - Missing frame-border cells act as beam emitters; the side or corner of each missing cell determines the outward cardinal or diagonal launch direction.

    Rule:
    - Every gap emits one beam in its outward direction, and the invisible beam spine still follows the original redirection logic: 90-degree meetings turn toward the bisector, while 45-degree meetings become a zigzag.
    - New fusion rule 1: every collision increases the future beam width by 1, so later ribbon segments become progressively thicker.
    - New fusion rule 2: every collision leaves a permanent blank break. A 90-degree collision leaves a 1-unit break, while a 45-degree collision leaves a 2-unit break before the widened continuation resumes.
    - Those blank breaks are rendered with the current widened scale, so repeated collisions create both fatter outgoing ribbons and visibly larger holes.

    Key insight: The solver must track two coupled hidden states after each collision: angle decides whether the break is short or long, and collision count decides how wide both the next ribbon and that break footprint become.
    """
    out, _stats = _simulate(grid)
    return out


# --------------- generator (combo) ---------------

BLOCK_COUNT_RANGE = (4, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {4: 22, 5: 24}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 3
MIN_RAY_STEPS = 4
MIN_BEAM_PIXELS_PER_GAP = 5
MIN_WIDE_PIXELS = 16
MIN_WIDE_SPINE_CELLS = 6
MIN_GAP_PIXELS_45 = 6
MIN_GAP_PIXELS_90 = 3
GEN_RETRIES = 700

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_BORDER_ORDER, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        if size >= 3 and (3, 3) in combo and (0, 2) in combo:
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


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
    center = (n - 1) / 2
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        viable.sort(key=lambda p: abs(p[0] - center) + abs(p[1] - center))
        topk = viable[: max(18, len(viable) // 3)]
        selected.append(topk[int(rng.integers(len(topk)))])
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


def _rays_clear(
    grid: np.ndarray,
    bg: int,
    seeds: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
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
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
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
    result = 22
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_gap_combo(
    available: dict[int, list[tuple[tuple[int, int], ...]]],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    gap_counts = sorted(available)
    weights = np.array([count * count * count for count in gap_counts], dtype=float)
    weights /= weights.sum()
    gap_count = int(rng.choice(gap_counts, p=weights))
    combos = available[gap_count]
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
        chosen_gaps.append(_pick_gap_combo(available, rng))

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
        out, stats = _simulate(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["collision_45"] < 1:
        return None
    if stats["collision_90"] < 1:
        return None
    if stats["max_width"] < (3 if rng.random() < 0.45 else 2):
        return None
    if stats["wide_pixels"] < MIN_WIDE_PIXELS:
        return None
    if stats["wide_spine_cells"] < MIN_WIDE_SPINE_CELLS:
        return None
    if stats["gap_events_45"] < 1 or stats["gap_events_90"] < 1:
        return None
    if stats["gap_pixels_45"] < MIN_GAP_PIXELS_45:
        return None
    if stats["gap_pixels_90"] < MIN_GAP_PIXELS_90:
        return None
    if stats["gap_cells_45"] <= stats["gap_cells_90"]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
