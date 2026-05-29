# Variant of e12f9a14
# Dimension: D14 — 多步处理与射线顺序 (multi_step)
# Change: 所有射线同时发射并独立延伸，碰撞在轨迹交叉时自然发生（无显式顺序）... → 按框架距离排序处理

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior and where it changes
#    - The visible objects stay the same: square grids, one shared frame color, 2x2 solid inner blocks,
#      and border gaps that emit colored rays.
#    - The new behavior is not a new object type; it is a new execution schedule.
#      Each 4x4 frame is assigned a visible "centrality" key: the squared Euclidean distance between
#      the frame center and the grid center. Frames fire from near-center to far-from-center.
#    - All gaps of the same frame still fire together, so the original local collision rules remain
#      intact inside a frame. The change happens between frames: once one frame finishes, its colored
#      trail cells and drawn diagonal segments freeze and become hard obstacles for later frames.
#    - Small example:
#      * Original logic: frame A emits east, frame B emits south, and both beams reach the same bg cell
#        in the same global timestep. That is a 90-degree collision, so the meeting cell is erased and
#        both beams turn to the bisector.
#      * Variant logic: if frame A is closer to the grid center, A finishes first and paints that cell.
#        When frame B is processed later, B does not get a collision response there; it simply stops
#        before the occupied cell because A's earlier trail is now an obstacle.
#      * The output difference is therefore a truncated later beam instead of a symmetric joint turn.
# b) Complexity calibration
#    - The rule is intentionally limited to one ordering signal: center distance.
#      That is enough to change behavior, but still visually inferable from 2-4 examples.
#    - Generator samples use at least two frames and force the sequential result to differ from the
#      simultaneous baseline, so the D14 change is actually observable.
#    - To keep the cue clean, the generator uses single-gap frames. The solver still supports multiple
#      gaps per frame, so only the generation distribution is simplified, not the task semantics.
# c) Hidden assumptions in the original solver that must change
#    - Original `main()` treated every active beam as part of one global synchronous system.
#      That implicitly assumed any occupied beam cell could still participate in a future collision.
#      In the variant, occupied cells from earlier frames are no longer dynamic participants.
#    - Original segment-cross logic allowed a moving diagonal to "collide with" an already drawn
#      diagonal segment and redirect both trajectories. That is only correct when both segments belong
#      to the same simultaneous batch. In the variant, an earlier-frame diagonal segment is static, so
#      a later diagonal must stop instead of triggering a redirect.
#    - Original code used one `cell_owner/cell_enter` table for all beams. That becomes unsafe once
#      some paths are meant to be frozen obstacles, because later beams must not truncate or rewrite
#      older-frame paths. The variant therefore simulates one frame-batch at a time with local owners
#      and a separate static obstacle set.
# d) Placement legality
#    - Frames still obey the original 4x4 geometry and non-overlap spacing.
#    - Generator additionally enforces distinct center-distance keys for all frames, so the order does
#      not depend on an invisible tie. A visible reading-order tie-break remains in the solver for
#      completeness, but generation rejects tied-distance layouts.
#    - Candidate gaps are still validated against the obstacle grid containing intact frames, so rays
#      never begin by hitting another frame or leaving the grid immediately.
# e) Visual observability
#    - Sorting never depends on colors, ids, or randomness. The only control signal is visible position
#      relative to the grid center.
#    - Generator rejects any sample where sequential and simultaneous outputs match, and also requires
#      at least one later beam to be stopped by an earlier frozen trail.
# f) Visual neatness
#    - Single-gap frames reduce overdraw and make the order effect legible: one or more later beams end
#      abruptly against central trails instead of producing a symmetric global collision pattern.
#    - Frame spacing and runway checks keep the scene sparse enough that the trails remain readable.
# g) Color inference
#    - Colors do not change meaning. Every beam still inherits its inner 2x2 block color, so the only
#      new inference burden is the spatial ordering rule.
# New solver assumptions and generator guarantees
#    - Assumption: each generated frame has at least one structural gap. Guaranteed by construction.
#    - Assumption: frame center-distance keys are distinct in generator outputs. Guaranteed explicitly.
#    - Assumption: static obstacles only need to block later beams; they do not need to participate in
#      45-degree/90-degree resolution. Guaranteed by the sequential batching logic itself.


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


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _frame_center_dist2(br: int, bc: int, h: int, w: int) -> int:
    """Squared distance * 4 from frame/block center to grid center, kept integral."""
    dr2 = 2 * br + 1 - (h - 1)
    dc2 = 2 * bc + 1 - (w - 1)
    return int(dr2 * dr2 + dc2 * dc2)


def _diag_seg_key(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (p, q) if p <= q else (q, p)


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


def _build_frames(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    frames: list[dict[str, object]] = []
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected each frame to expose at least one structural gap"
        frames.append(
            {
                "color": int(col),
                "br": int(br),
                "bc": int(bc),
                "gaps": list(gaps),
                "dist2": _frame_center_dist2(br, bc, h, w),
            }
        )
    return frames


def _frame_seeds(frame: dict[str, object]) -> list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]]:
    br = int(frame["br"])
    bc = int(frame["bc"])
    color = int(frame["color"])
    seeds: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for gr, gc in frame["gaps"]:
        dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
        dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
        d = _normalize_dir(dr, dc)
        assert d != (0, 0), "expected each frame gap to face outward"
        seeds.append((color, (gr, gc), (d,)))
    return seeds


def _simulate_beam_group(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    seeds: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]],
    static_cells: set[tuple[int, int]] | None = None,
    static_segments: set[tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> dict[str, object]:
    h, w = g.shape
    blocked: set[tuple[int, int]] = set()
    beams: dict[int, _Beam] = {}
    next_bid = 0

    static_cells = set() if static_cells is None else static_cells
    static_segments = set() if static_segments is None else static_segments

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    blocked_by_static: set[int] = set()

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
        assert pos not in static_cells, "expected frame gap not to be occupied by earlier trails"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for color, pos, pattern in seeds:
        spawn_beam(color, pos, pattern)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked and cell not in static_cells

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically"
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
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free zigzag step"
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
        assert u != (0, 0), "expected crossing diagonals to resolve forward"
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
        assert u != (0, 0), "expected segment crossing to resolve forward"

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
            if not in_bounds(nxt):
                b.active = False
                continue
            if nxt in static_cells:
                blocked_by_static.add(bid)
                b.active = False
                continue
            if not is_free(nxt):
                assert nxt in blocked or int(g[nxt[0], nxt[1]]) != bg, "beam must stop on local obstacles or structure"
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        for bid, (_nxt, d) in list(prop.items()):
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            p2 = (pr, pc + d[1])
            n2 = (pr + d[0], pc)
            if _diag_seg_key(p2, n2) in static_segments:
                blocked_by_static.add(bid)
                beams[bid].active = False
                prop.pop(bid, None)

        if not prop:
            continue

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
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    diag_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for beam in beams.values():
        for p, q in zip(beam.path, beam.path[1:]):
            if abs(q[0] - p[0]) == 1 and abs(q[1] - p[1]) == 1:
                diag_segments.add(_diag_seg_key(p, q))

    return {
        "cells": set(cell_owner),
        "segments": diag_segments,
        "blocked_by_static": len(blocked_by_static),
        "beam_count": len(beams),
    }


def _solve(grid: np.ndarray, sequential: bool, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    out = g.copy()

    if not sequential:
        all_seeds: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
        for frame in frames:
            all_seeds.extend(_frame_seeds(frame))
        _simulate_beam_group(g, out, bg, all_seeds)
        if want_stats:
            return out, {
                "frame_count": len(frames),
                "order": list(range(len(frames))),
                "dist2": [int(frame["dist2"]) for frame in frames],
                "blocked_by_static": 0,
                "blocked_frames": 0,
            }
        return out

    order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["dist2"]), int(frames[i]["br"]), int(frames[i]["bc"])))
    static_cells: set[tuple[int, int]] = set()
    static_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    stats = {
        "frame_count": len(frames),
        "order": list(order),
        "dist2": [int(frames[i]["dist2"]) for i in order],
        "blocked_by_static": 0,
        "blocked_frames": 0,
    }

    for idx in order:
        group = _simulate_beam_group(g, out, bg, _frame_seeds(frames[idx]), static_cells, static_segments)
        static_cells.update(group["cells"])
        static_segments.update(group["segments"])
        stats["blocked_by_static"] += int(group["blocked_by_static"])
        if int(group["blocked_by_static"]) > 0:
            stats["blocked_frames"] += 1

    if want_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit framed-block beams in center-outward frame order instead of one global simultaneous batch.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block wrapped by a 4x4 frame.
    - Missing frame-border cells are gaps. Every gap is a beam source whose direction is determined by which side or corner is open.

    Rule:
    - Rank frames by distance from the grid center, nearest first; ties break top-to-bottom then left-to-right.
    - For one frame at a time, launch all of that frame's beams together and keep the original local collision rules:
      90-degree meetings create a blank pivot and turn to the bisector, while 45-degree meetings merge into a zigzag.
    - After a frame finishes, its painted trail cells and diagonal segments freeze. Later frames treat those trails as solid obstacles,
      so they stop there instead of triggering a new cross-frame collision response.

    Key insight:
    - The output is governed by hidden processing order, not just geometry. More central frames get priority, and their finished trails
      can cut off farther frames that would have collided differently under the old simultaneous interpretation.
    """
    return _solve(grid, sequential=True)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    seq, stats = _solve(grid, sequential=True, want_stats=True)
    sim = _solve(grid, sequential=False)
    return {
        "frame_count": int(stats["frame_count"]),
        "blocked_by_static": int(stats["blocked_by_static"]),
        "blocked_frames": int(stats["blocked_frames"]),
        "diff_vs_simultaneous": int(np.count_nonzero(seq != sim)),
        "painted_pixels": int(np.count_nonzero(seq != np.asarray(grid))),
    }


# --------------- generator (D14 variant) ---------------

BLOCK_COUNT_CHOICES = [2, 3]
BLOCK_COUNT_WEIGHTS = [0.72, 0.28]
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 3: 18}
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
MIN_BEAM_PIXELS_PER_GAP = 2
MIN_ORDER_DIFF_PIXELS = 4
GEN_RETRIES = 300

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
    all_positions = [(r, c) for r in range(margin, n - 5) for c in range(margin, n - 5)]
    selected: list[tuple[int, int]] = []
    used_dists: set[int] = set()
    for _ in range(count):
        viable = [
            p
            for p in all_positions
            if _frame_center_dist2(p[0] + 1, p[1] + 1, n, n) not in used_dists
            and all(_frames_separated(p, q) for q in selected)
        ]
        if not viable:
            return None
        p = viable[int(rng.integers(len(viable)))]
        selected.append(p)
        used_dists.add(_frame_center_dist2(p[0] + 1, p[1] + 1, n, n))
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
    block_colors = list(map(int, rng.choice(palette, size=block_count, replace=False)))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _render_candidate(
    n: int,
    bg: int,
    frame_color: int,
    positions: list[tuple[int, int]],
    block_colors: list[int],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
    return grid


def _find_interacting_grid(
    n: int,
    bg: int,
    frame_color: int,
    positions: list[tuple[int, int]],
    block_colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    single_gap_choices: list[list[tuple[tuple[int, int], ...]]] = []
    for r0, c0 in positions:
        clear = _clear_gap_combos(obstacles, bg, r0, c0)
        choices = list(clear.get(1, []))
        if not choices:
            return None
        rng.shuffle(choices)
        single_gap_choices.append(choices)

    for chosen_gaps in itertools.product(*single_gap_choices):
        grid = _render_candidate(n, bg, frame_color, positions, block_colors, list(chosen_gaps))
        if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
            continue
        try:
            seq, stats = _solve(grid, sequential=True, want_stats=True)
            sim = _solve(grid, sequential=False)
        except Exception:
            continue
        if stats["blocked_by_static"] <= 0 or stats["blocked_frames"] <= 0:
            continue
        if np.array_equal(seq, sim):
            continue
        if np.count_nonzero(seq != sim) < MIN_ORDER_DIFF_PIXELS:
            continue
        if np.count_nonzero(seq != grid) < len(chosen_gaps) * MIN_BEAM_PIXELS_PER_GAP:
            continue
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.choice(BLOCK_COUNT_CHOICES, p=BLOCK_COUNT_WEIGHTS))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    grid = _find_interacting_grid(n, bg, frame_color, positions, block_colors, rng)
    if grid is None:
        return None

    metrics = variant_metrics(grid)
    if metrics["blocked_by_static"] <= 0 or metrics["diff_vs_simultaneous"] < MIN_ORDER_DIFF_PIXELS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D14 e12f9a14 input")
