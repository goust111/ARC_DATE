# Variant of e12f9a14
# Dimension: D13 — 框架对射线绘制的影响 (transform)
# Change: 射线经过框架自身的缺口位置时会在缺口处绘制源颜色像素（替换原来的背景色），框架的绿色边框像素保持不变... → 碰撞后射线改变自身框架边框颜色

import itertools
from collections import Counter

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for D13
# a) Concrete behavior:
#    - Keep the original object vocabulary: 2x2 colored cores inside 4x4 shared-color frames with 1-cell openings.
#    - The D13 change is local to how a frame is affected by its own beam. The opening itself now stays open in the
#      output; the first painted beam pixel is the first exterior cell after the opening.
#    - When two beams collide, the collision is still resolved by the original path rules, but each colliding beam
#      also sends the opponent color back to its source frame. Every surviving wall pixel on that source frame border
#      is recolored to the opponent beam's color, while the gap cell remains background.
#    - Small example: a red east-going beam and a yellow south-going beam meet. Originally the red frame kept its
#      green border and only the opening was filled red. In this variant, the meeting point still resolves as before,
#      but the entire red frame border turns yellow and the yellow frame border turns red.
# b) Complexity calibration:
#    - The solver still supports all original collision branches. The generator narrows the placement space to
#      one-gap frames and forces at least one inter-frame collision so the new effect is visible and not drowned out
#      by multi-gap ambiguity or collision-free samples.
# c) Old assumptions that no longer hold:
#    - `spawn_beam()` used to paint the gap immediately; that would erase the new "opening stays open" cue.
#    - Collision handling used to affect only local beam paths; now every collision must also update source-frame
#      border colors, so each beam needs a stable source-frame id.
#    - Because frame colors can change in the output, any future motion logic must continue to read occupancy from the
#      immutable input grid `g`, not from `out`. The original code already does this; the generator therefore only
#      needs to guarantee clear motion relative to the input layout.
# d) Placement legality:
#    - Frames still stay fully in-bounds and keep >=2 cells of bbox clearance.
#    - The generator first constructs one intentional collision pair, then places any extra frames in non-overlapping
#      positions and validates the full grid by running the solver.
# e) Visual observability:
#    - Every accepted sample has at least one collision and at least two recolored frames, so the D13 rule is visible
#      even from a small set of I/O examples.
#    - Colors are never ordered by id; the recolor rule is purely structural: "frame gets the other beam's color."
# f) Visual cleanliness:
#    - One opening per frame avoids ambiguous multi-message overwrites on the same frame.
#    - Extra frames use clear single-ray exits and are validated against the full obstacle layout.
# g) Color-change inferability:
#    - Each recolored frame is paired with a single beam color, making the swap-like remote transmission legible.
#    - Distinct block colors are still used, so the source/opponent correspondence is visually recoverable.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_DIR_TO_GAPS = {
    (-1, 0): ((0, 1), (0, 2)),
    (-1, 1): ((0, 3),),
    (0, 1): ((1, 3), (2, 3)),
    (1, 1): ((3, 3),),
    (1, 0): ((3, 1), (3, 2)),
    (1, -1): ((3, 0),),
    (0, -1): ((1, 0), (2, 0)),
    (-1, -1): ((0, 0),),
}
_CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]


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


def _frame_wall_cells(g: np.ndarray, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    cells: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == wall:
                cells.append((r, c))
    return cells


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


_RIGHT_ANGLE_COLLISIONS = [(a, b) for a, b in itertools.combinations(_CARDINAL_DIRS, 2) if _dir_diff(a, b) == 2]
_FORTYFIVE_COLLISIONS = [(a, b) for a in _CARDINAL_DIRS for b in _DIAGONAL_DIRS if _dir_diff(a, b) == 1]


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "source_frame")

    def __init__(
        self,
        bid: int,
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        source_frame: int,
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        # path[0] is the gap location; unlike the original task, it stays visually open.
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.source_frame = source_frame

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0

    frames: dict[int, dict[str, object]] = {}
    frame_mask = np.zeros_like(g, dtype=bool)

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    stats = {
        "collision_count": 0,
        "collision_kinds": Counter(),
        "recolor_events": 0,
        "recolored_frames": set(),
    }

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

    def recolor_frame(fid: int, color: int) -> None:
        wall_cells = frames[fid]["wall_cells"]
        for r, c in wall_cells:
            out[r, c] = int(color)
        stats["recolored_frames"].add(fid)

    def note_collision(a: int, b: int, kind: str) -> None:
        stats["collision_count"] += 1
        stats["collision_kinds"][kind] += 1
        fa = beams[a].source_frame
        fb = beams[b].source_frame
        recolor_frame(fa, beams[b].color)
        recolor_frame(fb, beams[a].color)
        stats["recolor_events"] += 2 if fa != fb else 1

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        source_frame: int,
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, source_frame)

    for fid, (col, br, bc) in enumerate(blocks):
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        wall_cells = _frame_wall_cells(g, frame_col, br, bc)
        for cell in wall_cells:
            frame_mask[cell] = True
        frames[fid] = {
            "block_color": int(col),
            "frame_color": int(frame_col),
            "wall_cells": wall_cells,
        }
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), fid)

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
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"
        note_collision(a, b, "cell90" if diff == 2 else "cell45")

        if diff == 2:
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not meet at one cell"
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

        assert beams[bid_u].pos == cell, "cardinal beam head must sit at the collision cell"

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
        note_collision(a, b, "cross")
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
        note_collision(a, b, "segcross")
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
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a solid frame or block"
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
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    if return_stats:
        stats["recolored_frames"] = len(stats["recolored_frames"])
        stats["frame_border_changes"] = int(sum(out[cell] != g[cell] for fid in frames for cell in frames[fid]["wall_cells"]))
        stats["beam_pixels"] = int(np.count_nonzero((out != g) & (~frame_mask)))
        stats["frame_count"] = len(frames)
        stats["beam_count"] = len(beams)
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace beams from small framed blocks and let collisions send color back to the frames.

    Input structure:
    - The most frequent color is the background. Each object is a monochrome 2x2 block wrapped by a shared-color 4x4 frame.
    - Every frame has one or more 1-cell openings on its border; these openings indicate the outward launch direction.

    Rule:
    - A beam of the inner block's color starts at each opening but leaves the opening itself blank; painting begins on the
      first exterior cell and continues until the beam reaches the boundary or interacts with another beam.
    - The local collision geometry stays the same as the original task: 90-degree meetings create a 1-cell hole and bend
      both beams toward the bisector, while 45-degree meetings form the same alternating zigzag continuation.
    - New D13 effect: every collision also transmits the opponent beam's color back to each source frame, recoloring that
      frame's existing border pixels while leaving the opening cell as background.

    Key insight:
    - A collision now has two coupled consequences: the beam paths change near the meeting point, and the same event
      remotely rewrites the colors of the originating frames. The output must satisfy both the local path rule and the
      non-local frame recoloring rule.
    """
    return _simulate(grid, return_stats=False)


# --------------- generator (D13 variant) ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 4: 18}
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
PAIR_DISTANCE_RANGE = (4, 7)
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
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _frame_in_bounds(n: int, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    return PLACEMENT_MARGIN <= r0 <= n - 4 - PLACEMENT_MARGIN and PLACEMENT_MARGIN <= c0 <= n - 4 - PLACEMENT_MARGIN


def _seed_to_frame_pos(target: tuple[int, int], gap: tuple[int, int], direction: tuple[int, int], steps: int) -> tuple[int, int]:
    seed = (target[0] - steps * direction[0], target[1] - steps * direction[1])
    return seed[0] - gap[0], seed[1] - gap[1]


def _choose_positions_with_seeded(
    n: int,
    count: int,
    seeded: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    all_positions = [(r, c) for r in range(PLACEMENT_MARGIN, n - 3 - PLACEMENT_MARGIN) for c in range(PLACEMENT_MARGIN, n - 3 - PLACEMENT_MARGIN)]
    selected = list(seeded)
    while len(selected) < count:
        viable = [p for p in all_positions if p not in selected and all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _try_collision_pair(n: int, rng: np.random.Generator):
    modes = [("right_angle", pair) for pair in _RIGHT_ANGLE_COLLISIONS] + [("fortyfive", pair) for pair in _FORTYFIVE_COLLISIONS]
    for _ in range(200):
        mode, (d1, d2) = modes[int(rng.integers(len(modes)))]
        steps1 = int(rng.integers(*PAIR_DISTANCE_RANGE))
        steps2 = int(rng.integers(*PAIR_DISTANCE_RANGE))
        gap1 = _DIR_TO_GAPS[d1][int(rng.integers(len(_DIR_TO_GAPS[d1])))]
        gap2 = _DIR_TO_GAPS[d2][int(rng.integers(len(_DIR_TO_GAPS[d2])))]
        target = (
            int(rng.integers(PLACEMENT_MARGIN + 1, n - PLACEMENT_MARGIN - 1)),
            int(rng.integers(PLACEMENT_MARGIN + 1, n - PLACEMENT_MARGIN - 1)),
        )
        pos1 = _seed_to_frame_pos(target, gap1, d1, steps1)
        pos2 = _seed_to_frame_pos(target, gap2, d2, steps2)
        if not _frame_in_bounds(n, pos1) or not _frame_in_bounds(n, pos2):
            continue
        if not _frames_separated(pos1, pos2):
            continue
        return [
            {"pos": pos1, "gaps": (gap1,), "dir": d1, "target": target},
            {"pos": pos2, "gaps": (gap2,), "dir": d2, "target": target},
        ], mode
    return None, None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    pair_specs, _mode = _try_collision_pair(n, rng)
    if pair_specs is None:
        return None

    pair_positions = [spec["pos"] for spec in pair_specs]
    positions = _choose_positions_with_seeded(n, block_count, pair_positions, rng)
    if positions is None:
        return None

    extra_positions = positions[len(pair_positions):]
    frame_specs = [
        {"pos": pair_specs[0]["pos"], "gaps": pair_specs[0]["gaps"]},
        {"pos": pair_specs[1]["pos"], "gaps": pair_specs[1]["gaps"]},
    ]

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    pair_seeds = []
    for spec in pair_specs:
        (gap,) = spec["gaps"]
        r0, c0 = spec["pos"]
        pair_seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))
    if not _rays_clear(obstacles, bg, pair_seeds):
        return None

    for r0, c0 in extra_positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        combos = available.get(1, [])
        if not combos:
            return None
        frame_specs.append({"pos": (r0, c0), "gaps": combos[int(rng.integers(len(combos)))]})

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for spec, color in zip(frame_specs, block_colors):
        r0, c0 = spec["pos"]
        gaps = spec["gaps"]
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(obstacles, bg, seeds):
        return None

    try:
        out, stats = _simulate(grid, return_stats=True)
    except Exception:
        return None

    if stats["collision_count"] < 1:
        return None
    if stats["recolored_frames"] < 2:
        return None
    if stats["frame_border_changes"] < 8:
        return None
    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D13 input")
