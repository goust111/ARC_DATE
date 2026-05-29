# Combo variant of e12f9a14 (index 38)
# combo_spec: D3/2+D14/0
# Name: 碰撞后颜色改变+按距离排序

from __future__ import annotations

import itertools

import numpy as np

try:
    from arc_utils import get_bg, paste
except ImportError:
    from arc2_solver_select.arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and worked trace
#    - The visible input objects stay close to the base task: square grids, one shared
#      frame color, 2x2 monochrome inner blocks, and structural frame gaps that emit rays.
#    - D14 contributes a visible ordering key: frames are processed from near the grid
#      center to far from it, using squared center distance and row-major tie-break only
#      as a deterministic fallback. Generation rejects equal distances so the hidden key
#      is still recoverable from position alone.
#    - D3 contributes mutable ray color, but in the fused setting the important collisions
#      are asymmetric: an earlier frame finishes first, so its trail is frozen; when a
#      later frame hits that frozen trail, the later continuation inherits the older trail
#      color while the older trail itself remains unchanged. This is the concrete meaning
#      of "the first-fired color transfers to the later-fired ray".
#    - Small trace:
#      * Base task: a central red frame and a farther blue frame would fire together; if
#        their rays meet, both continuations react symmetrically and each keeps its own
#        color.
#      * D14 alone: the red frame fires first, freezes, and the blue ray just stops when
#        it reaches red's trail; no cross-frame color cue appears.
#      * Fused combo: the red frame still fires first, but when the blue ray later reaches
#        the frozen red trail, the blue continuation inherits red and bends according to
#        the same local collision geometry. The frozen red trail stays in place, so order
#        decides the direction of color transfer.
# b) Complexity calibration
#    - The fusion is intentionally limited to one ordering signal and one transfer rule:
#      center distance controls who can donate color, and donor color is always the color
#      currently visible on the earlier trail cell/segment that was hit.
#    - The generator uses mostly 2-3 single-gap frames. That keeps examples sparse enough
#      that a human can see which frame is more central, which later ray touched it, and
#      where the trail color changes after that touch.
#    - We reject samples that only exhibit within-frame color swaps; at least one visible
#      cross-frame transfer must occur, otherwise the combo collapses back toward either
#      plain D3 or plain D14.
# c) Implicit assumption audit
#    - Base/D14 assumption broken by D3: `beam.color` cannot be constant. The solver must
#      track mutable current color plus immutable `source_color` for diagnostics.
#    - D14 assumption broken by the fusion: earlier trails cannot be mere solid blockers.
#      They must retain direction and visible color metadata so a later ray can collide
#      with them asymmetrically.
#    - Base 90-degree rule erased the meeting cell. That is impossible when the donor cell
#      belongs to a frozen earlier frame. The fused solver therefore keeps the older cell
#      intact and only redirects the later continuation; this is the key new asymmetry.
#    - Base ownership tables assumed every visited path cell is writable by the current
#      beam. That fails when a later beam's 45-degree collision point is an already-frozen
#      cell. The implementation therefore allows a beam path to include "traversed but not
#      owned" positions so future direction can continue from a frozen collision point.
# d) Placement legality
#    - Frames still obey the original 4x4 geometry, non-overlap spacing, and in-bounds
#      constraints.
#    - Candidate gaps are validated against intact full-frame obstacles so every emitted ray
#      has runway through background cells before any beam painting starts.
#    - Generation also enforces distinct frame-center distances and checks the solver on
#      every candidate, guaranteeing that all new collision assumptions used by `main()`
#      are physically realizable on produced inputs.
# e) Visual observability
#    - Order depends only on visible geometry relative to the grid center; no numeric color
#      ids or hidden randomness influence it.
#    - Color transfer is also structural rather than numeric: later segments take the color
#      of the earlier trail they visibly touch. Because block colors are distinct, the cue
#      can be inferred from just a few examples.
#    - The generator rejects candidates where the fused output matches either the D14
#      stop-on-static baseline or the original simultaneous baseline too closely; this keeps
#      the fusion effect visible rather than theoretical.
# f) Visual neatness
#    - Single-gap generation avoids dense self-intersections and makes the centrality cue
#      legible. The output is usually one earlier trail plus one or more later trails that
#      visibly change color after touching it.
#    - Frame spacing and runway checks prevent frames from crowding the same local area, so
#      the eye can separate the donor trail from the inherited continuation.
# g) New solver assumptions and generator guarantees
#    - Assumption: if a later ray hits a frozen trail at 45 or 90 degrees, the frozen trail
#      carries enough metadata (enter direction, segment direction, visible color) to define
#      the donor. Guaranteed because earlier batches record those maps explicitly.
#    - Assumption: generator outputs contain at least one visible cross-frame transfer and a
#      nontrivial difference from both simpler baselines. Guaranteed by `variant_metrics()`
#      filters in `_find_interacting_grid()`.
#    - Assumption: later rays never need to overwrite frozen cells. Guaranteed by the fused
#      collision semantics, which either continue from the frozen collision point without
#      repainting it or redirect before it.


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
    dr2 = 2 * br + 1 - (h - 1)
    dc2 = 2 * bc + 1 - (w - 1)
    return int(dr2 * dr2 + dc2 * dc2)


def _diag_seg_key(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (p, q) if p <= q else (q, p)


class _Beam:
    __slots__ = (
        "bid",
        "source_color",
        "color",
        "pos",
        "pattern",
        "phase",
        "path",
        "enter_dirs",
        "step_colors",
        "active",
    )

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.source_color = int(color)
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.step_colors = [int(color)]
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
    static_cell_color: dict[tuple[int, int], int] | None = None,
    static_cell_enter: dict[tuple[int, int], tuple[int, int]] | None = None,
    static_segments: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], int]] | None = None,
    cross_frame_inherit: bool = True,
    collect_stats: bool = False,
) -> dict[str, object]:
    h, w = g.shape
    blocked: set[tuple[int, int]] = set()
    beams: dict[int, _Beam] = {}
    next_bid = 0

    static_cell_color = {} if static_cell_color is None else dict(static_cell_color)
    static_cell_enter = {} if static_cell_enter is None else dict(static_cell_enter)
    static_segments = {} if static_segments is None else dict(static_segments)

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    ever_changed: set[int] = set()
    blocked_by_static: set[int] = set()
    stats = {
        "dynamic_collision_events": 0,
        "visible_dynamic_swap_events": 0,
        "cross_frame_collision_events": 0,
        "visible_static_transfer_events": 0,
    }

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.step_colors.append(int(b.color))
        b.pos = cell
        out[cell[0], cell[1]] = b.color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(b.path) - 1

    def occupy_static_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.step_colors.append(int(b.color))
        b.pos = cell

    def repaint_owned_cell(cell: tuple[int, int], bid: int) -> None:
        assert cell_owner.get(cell) == bid, "expected repaint target to stay owned by the beam"
        out[cell[0], cell[1]] = beams[bid].color

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        out[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one step"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.step_colors = b.step_colors[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def set_beam_color(bid: int, color: int) -> bool:
        color = int(color)
        if beams[bid].color == color:
            return False
        beams[bid].color = color
        ever_changed.add(bid)
        return True

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in static_cell_color, "expected frame gap not to be occupied by earlier trails"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        out[pos[0], pos[1]] = int(color)
        cell_owner[pos] = bid
        cell_enter[pos] = pattern[0]
        cell_index[pos] = 0

    for color, pos, pattern in seeds:
        spawn_beam(color, pos, pattern)

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    def can_paint_new(cell: tuple[int, int]) -> bool:
        return (
            in_bounds(cell)
            and int(g[cell[0], cell[1]]) == bg
            and cell not in blocked
            and cell not in static_cell_color
            and cell_owner.get(cell) is None
        )

    def is_lockstep_neighbor(a: int, b: int) -> bool:
        ba = beams[a]
        bb = beams[b]
        if not ba.active or not bb.active:
            return False
        if ba.pattern != bb.pattern or ba.phase != bb.phase or ba.color != bb.color:
            return False
        d = ba.next_dir()
        if bb.next_dir() != d:
            return False
        if abs(bb.pos[0] - ba.pos[0]) + abs(bb.pos[1] - ba.pos[1]) != 1:
            return False
        if (ba.pos[0] + d[0], ba.pos[1] + d[1]) == bb.pos:
            return False
        if (bb.pos[0] + d[0], bb.pos[1] + d[1]) == ba.pos:
            return False
        return True

    def lockstep_group(start: int) -> set[int]:
        group = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for oid in beams:
                if oid in group or oid == cur:
                    continue
                if is_lockstep_neighbor(cur, oid):
                    group.add(oid)
                    stack.append(oid)
        return group

    def apply_static_color(group: set[int], color: int) -> None:
        if any(beams[gid].color != int(color) for gid in group):
            stats["visible_static_transfer_events"] += 1
        for gid in group:
            set_beam_color(gid, int(color))

    def swap_groups(group_a: set[int], group_b: set[int]) -> None:
        assert group_a and group_b and group_a.isdisjoint(group_b), "expected disjoint collision groups"
        col_a = {beams[gid].color for gid in group_a}
        col_b = {beams[gid].color for gid in group_b}
        assert len(col_a) == 1 and len(col_b) == 1, "expected uniform lockstep colors"
        color_a = next(iter(col_a))
        color_b = next(iter(col_b))
        if color_a != color_b:
            stats["visible_dynamic_swap_events"] += 1
        for gid in group_a:
            set_beam_color(gid, color_b)
        for gid in group_b:
            set_beam_color(gid, color_a)

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        stats["dynamic_collision_events"] += 1
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
                truncate_beam(b, cell_index[cell])
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group_a = lockstep_group(a)
            group_b = lockstep_group(b)
            swap_groups(group_a, group_b)
            for gid in group_a | group_b:
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

        color_u = beams[bid_u].color
        color_v = beams[bid_v].color
        if color_u != color_v:
            stats["visible_dynamic_swap_events"] += 1
        set_beam_color(bid_u, color_v)
        set_beam_color(bid_v, color_u)

        if cell_owner.get(cell) != bid_u:
            if cell_owner.get(cell) == bid_v:
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                idx = cell_index[cell]
                truncate_beam(bid_u, idx + 1)
                repaint_owned_cell(cell, bid_u)
            else:
                paint_cell(bid_u, cell, u)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)
            repaint_owned_cell(cell, bid_u)

        adj = (beams[bid_v].pos[0] + u[0], beams[bid_v].pos[1] + u[1])
        assert can_paint_new(adj), "expected free adjacent zigzag step"
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
        stats["dynamic_collision_events"] += 1
        da = prop[a][1]
        db = prop[b][1]
        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        group_a = lockstep_group(a)
        group_b = lockstep_group(b)
        swap_groups(group_a, group_b)
        for gid in group_a | group_b:
            beams[gid].pattern = (nd,)
            beams[gid].phase = 0
            beams[gid].active = True

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        stats["dynamic_collision_events"] += 1
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        nd = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert nd != (0, 0), "expected segment crossing to resolve to a forward direction"
        group_a = lockstep_group(a)
        group_b = lockstep_group(b)
        swap_groups(group_a, group_b)
        for gid in group_a | group_b:
            beams[gid].pattern = (nd,)
            beams[gid].phase = 0
            beams[gid].active = True

    def resolve_static_cell_collision(cell: tuple[int, int], a: int, prop) -> None:
        stats["cross_frame_collision_events"] += 1
        da = prop[a][1]
        db = static_cell_enter[cell]
        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected eligible static collision to be 45 or 90 degrees"
        donor_color = static_cell_color[cell]

        if diff == 2:
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = lockstep_group(a)
            apply_static_color(group, donor_color)
            for gid in group:
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                beams[gid].active = True
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must involve one cardinal and one diagonal direction"
        if set_beam_color(a, donor_color):
            stats["visible_static_transfer_events"] += 1

        if is_card_a:
            u, v = da, db
            occupy_static_cell(a, cell, u)
        else:
            u, v = db, da
            adj = (beams[a].pos[0] + u[0], beams[a].pos[1] + u[1])
            assert can_paint_new(adj), "expected free adjacent step after static 45-degree collision"
            paint_cell(a, adj, u)

        beams[a].pattern = (v, u)
        beams[a].phase = 0
        beams[a].active = True

    def resolve_static_segment_collision(a: int, prop) -> None:
        stats["cross_frame_collision_events"] += 1
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        key = _diag_seg_key((pr, pc + dc), (pr + dr, pc))
        seg_dir, donor_color = static_segments[key]
        nd = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert nd != (0, 0), "expected static diagonal crossing to resolve forward"
        group = lockstep_group(a)
        apply_static_color(group, donor_color)
        for gid in group:
            beams[gid].pattern = (nd,)
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
            if nxt in blocked:
                b.active = False
                continue
            if int(g[nxt[0], nxt[1]]) != bg:
                b.active = False
                continue
            if nxt in static_cell_color:
                if not cross_frame_inherit or _dir_diff(d, static_cell_enter[nxt]) not in (1, 2):
                    blocked_by_static.add(bid)
                    b.active = False
                    continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        if cross_frame_inherit:
            static_seg_cross: set[int] = set()
            for bid, (_nxt, d) in prop.items():
                if abs(d[0]) + abs(d[1]) != 2:
                    continue
                pr, pc = beams[bid].pos
                key = _diag_seg_key((pr, pc + d[1]), (pr + d[0], pc))
                if key in static_segments:
                    static_seg_cross.add(bid)
        else:
            static_seg_cross = set()
            for bid, (_nxt, d) in list(prop.items()):
                if abs(d[0]) + abs(d[1]) != 2:
                    continue
                pr, pc = beams[bid].pos
                key = _diag_seg_key((pr, pc + d[1]), (pr + d[0], pc))
                if key in static_segments:
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
            obid = move_map.get(((pr, pc + dc), (pr + dr, pc)))
            if obid is None or obid == bid:
                continue
            cross_pairs.add(tuple(sorted((bid, obid))))

        seg_cross: set[tuple[int, int]] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            obid = diag_segment_owner((pr, pc + d[1]), (pr + d[0], pc))
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
            if owner is not None or len(bidders_sorted) >= 2:
                a = bidders_sorted[0]
                cand_bs: list[int] = []
                if owner is not None and owner != a:
                    cand_bs.append(owner)
                for bb in bidders_sorted[1:]:
                    if bb != a:
                        cand_bs.append(bb)
                        break
                if cand_bs:
                    events.append((0, cell[0], cell[1], a, cand_bs[0], "cell"))
            elif cell in static_cell_color:
                events.append((0, cell[0], cell[1], bidders_sorted[0], -1, "static_cell"))

        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            events.append((1, min(ca[0], cb[0]), min(ca[1], cb[1]), a, b, "cross"))

        for a, b in seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            events.append((1, min(pr, pr + dr), min(pc, pc + dc), a, b, "segcross"))

        for a in static_seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            events.append((1, min(pr, pr + dr), min(pc, pc + dc), a, -1, "static_segcross"))

        if events:
            events.sort()
            _, r, c, a, b, kind = events[0]
            if kind == "cell":
                resolve_cell_collision((r, c), a, b, prop)
            elif kind == "cross":
                resolve_cross_collision(a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision(a, b, prop)
            elif kind == "static_cell":
                resolve_static_cell_collision((r, c), a, prop)
            else:
                resolve_static_segment_collision(a, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None and nxt not in static_cell_color, "expected collision-free writable target"
            paint_cell(bid, nxt, d)
            beams[bid].advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    cell_colors = {cell: int(out[cell[0], cell[1]]) for cell in cell_owner}
    diag_segments: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], int]] = {}
    for beam in beams.values():
        for i in range(1, len(beam.path)):
            p = beam.path[i - 1]
            q = beam.path[i]
            dr = q[0] - p[0]
            dc = q[1] - p[1]
            if abs(dr) == 1 and abs(dc) == 1:
                diag_segments[_diag_seg_key(p, q)] = ((dr, dc), int(beam.step_colors[i]))

    result = {
        "cell_colors": cell_colors,
        "cell_enter": dict(cell_enter),
        "segments": diag_segments,
        "blocked_by_static": len(blocked_by_static),
        "beam_count": len(beams),
    }
    if collect_stats:
        result.update(
            {
                "dynamic_collision_events": int(stats["dynamic_collision_events"]),
                "visible_dynamic_swap_events": int(stats["visible_dynamic_swap_events"]),
                "cross_frame_collision_events": int(stats["cross_frame_collision_events"]),
                "visible_static_transfer_events": int(stats["visible_static_transfer_events"]),
                "foreign_color_cells": sum(
                    1 for cell, bid in cell_owner.items() if int(out[cell[0], cell[1]]) != beams[bid].source_color
                ),
                "beams_with_color_change": len(ever_changed),
            }
        )
    return result


def _solve(grid: np.ndarray, sequential: bool, cross_frame_inherit: bool, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frames = _build_frames(g, bg)
    out = g.copy()

    if not sequential:
        all_seeds: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
        for frame in frames:
            all_seeds.extend(_frame_seeds(frame))
        group = _simulate_beam_group(g, out, bg, all_seeds, cross_frame_inherit=False, collect_stats=want_stats)
        if want_stats:
            return out, {
                "frame_count": len(frames),
                "order": list(range(len(frames))),
                "dist2": [int(frame["dist2"]) for frame in frames],
                "visible_static_transfer_events": 0,
                "cross_frame_collision_events": 0,
                "frames_with_transfer": 0,
                "blocked_by_static": 0,
                "foreign_color_cells": int(group.get("foreign_color_cells", 0)),
                "visible_dynamic_swap_events": int(group.get("visible_dynamic_swap_events", 0)),
            }
        return out

    order = sorted(range(len(frames)), key=lambda i: (int(frames[i]["dist2"]), int(frames[i]["br"]), int(frames[i]["bc"])))
    static_cell_color: dict[tuple[int, int], int] = {}
    static_cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    static_segments: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], int]] = {}
    stats = {
        "frame_count": len(frames),
        "order": list(order),
        "dist2": [int(frames[i]["dist2"]) for i in order],
        "visible_static_transfer_events": 0,
        "cross_frame_collision_events": 0,
        "frames_with_transfer": 0,
        "blocked_by_static": 0,
        "foreign_color_cells": 0,
        "visible_dynamic_swap_events": 0,
    }

    for idx in order:
        group = _simulate_beam_group(
            g,
            out,
            bg,
            _frame_seeds(frames[idx]),
            static_cell_color,
            static_cell_enter,
            static_segments,
            cross_frame_inherit=cross_frame_inherit,
            collect_stats=want_stats,
        )
        static_cell_color.update(group["cell_colors"])
        static_cell_enter.update(group["cell_enter"])
        static_segments.update(group["segments"])
        stats["blocked_by_static"] += int(group["blocked_by_static"])
        if want_stats:
            stats["visible_static_transfer_events"] += int(group.get("visible_static_transfer_events", 0))
            stats["cross_frame_collision_events"] += int(group.get("cross_frame_collision_events", 0))
            stats["foreign_color_cells"] += int(group.get("foreign_color_cells", 0))
            stats["visible_dynamic_swap_events"] += int(group.get("visible_dynamic_swap_events", 0))
            if int(group.get("visible_static_transfer_events", 0)) > 0:
                stats["frames_with_transfer"] += 1

    if want_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire framed-block rays in center-distance order; later rays inherit color from earlier trails they hit.

    Input structure:
    - The grid is square with one dominant background color. Each object is a monochrome 2x2 block enclosed by a shared-color 4x4 frame.
    - Missing frame-border cells are structural gaps. Every gap launches one outward ray determined by its side or corner position.

    Rule:
    - Sort frames by distance from the grid center, nearest first; if a tie ever appears, break it top-to-bottom then left-to-right.
    - Process one frame at a time. Inside that frame, keep the original local ray geometry: 90-degree meetings redirect along the bisector, and 45-degree meetings merge into the same zigzag motif.
    - Finished trails from earlier frames stay frozen. When a later ray hits one of those frozen trail cells or diagonal segments at a valid collision angle, the later continuation adopts the older trail's color and bends accordingly, while the older trail remains unchanged.
    - Cells painted before a collision keep their old color. Only the continuation after the collision uses the inherited color.

    Key insight:
    - Solving now requires tracking one chronology that couples order and color: central frames act first, and their completed trails are not just blockers but color donors that redirect farther rays.
    """
    return _solve(grid, sequential=True, cross_frame_inherit=True)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    combo, stats = _solve(grid, sequential=True, cross_frame_inherit=True, want_stats=True)
    stop = _solve(grid, sequential=True, cross_frame_inherit=False)
    sim = _solve(grid, sequential=False, cross_frame_inherit=False)
    base = np.asarray(grid, dtype=int)
    return {
        "frame_count": int(stats["frame_count"]),
        "visible_static_transfer_events": int(stats["visible_static_transfer_events"]),
        "cross_frame_collision_events": int(stats["cross_frame_collision_events"]),
        "frames_with_transfer": int(stats["frames_with_transfer"]),
        "blocked_by_static": int(stats["blocked_by_static"]),
        "foreign_color_cells": int(stats["foreign_color_cells"]),
        "visible_dynamic_swap_events": int(stats["visible_dynamic_swap_events"]),
        "diff_vs_stop_baseline": int(np.count_nonzero(combo != stop)),
        "diff_vs_simultaneous": int(np.count_nonzero(combo != sim)),
        "painted_pixels": int(np.count_nonzero(combo != base)),
    }


# --------------- generator (combo variant) ---------------

BLOCK_COUNT_CHOICES = [2, 3]
BLOCK_COUNT_WEIGHTS = [0.82, 0.18]
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
MIN_FUSION_DIFF_PIXELS = 4
MIN_ORDER_DIFF_PIXELS = 4
MIN_FOREIGN_COLOR_CELLS = 3
GEN_RETRIES = 400

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


def _rays_clear(grid: np.ndarray, bg: int, seeds: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
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


def _clear_gap_combos(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
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
            out = main(grid)
            metrics = variant_metrics(grid)
        except Exception:
            continue
        if np.count_nonzero(out != grid) < len(chosen_gaps) * MIN_BEAM_PIXELS_PER_GAP:
            continue
        if metrics["visible_static_transfer_events"] < 1:
            continue
        if metrics["cross_frame_collision_events"] < 1:
            continue
        if metrics["frames_with_transfer"] < 1:
            continue
        if metrics["foreign_color_cells"] < MIN_FOREIGN_COLOR_CELLS:
            continue
        if metrics["diff_vs_stop_baseline"] < MIN_FUSION_DIFF_PIXELS:
            continue
        if metrics["diff_vs_simultaneous"] < MIN_ORDER_DIFF_PIXELS:
            continue
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.choice(BLOCK_COUNT_CHOICES, p=BLOCK_COUNT_WEIGHTS))
    min_n = _grid_size_min(block_count)
    max_n = min(GRID_SIZE_MAX, min_n + 8)
    n = int(rng.integers(min_n, max_n))
    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    grid = _find_interacting_grid(n, bg, frame_color, positions, block_colors, rng)
    if grid is None:
        return None

    metrics = variant_metrics(grid)
    if metrics["visible_static_transfer_events"] < 1:
        return None
    if metrics["diff_vs_stop_baseline"] < MIN_FUSION_DIFF_PIXELS:
        return None
    if metrics["diff_vs_simultaneous"] < MIN_ORDER_DIFF_PIXELS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo e12f9a14 input")
