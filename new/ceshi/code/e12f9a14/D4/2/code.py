# Variant of e12f9a14
# Dimension: D4 — 源对象类型 (src)
# Change: 绿色框架内的非绿色彩色2x2像素块，每个框架包含一种内部颜色... → 可变大小的内部色块

import itertools
from collections import Counter
from functools import lru_cache

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   source_objects:
#     original: fixed 2x2 monochrome square inside a 4x4 frame
#     new: solid kxk monochrome square with k in {1,2,3}, inside a (k+2)x(k+2) frame
#   preserved_mechanics:
#     - frame is still one-cell thick
#     - gap cells are still single missing border pixels
#     - gap position still determines initial outward direction
#     - beam collision rules stay the same as the original task
#   new_size_mapping:
#     - size k maps to a beam travel cap of 2*k+1 forward steps after the gap seed
#     - examples: 1x1 -> 3 steps, 2x2 -> 5 steps, 3x3 -> 7 steps
#   concrete_trace:
#     - original: a top-edge gap on a 2x2 block emits a cardinal beam that keeps going until edge/collision
#     - variant: the same top-edge gap on a 2x2 block emits the same cardinal beam, but it stops after 5
#       forward cells unless a collision or grid edge stops it earlier; a 1x1 block under the same geometry
#       would stop after 3 cells, and a 3x3 block after 7
#     - difference: the ray engine is unchanged, but every source now carries an explicit size-derived budget
#   complexity_calibration:
#     - only three source sizes are used, so the new inference burden is visible but not combinatorially large
#     - every generated input contains at least two distinct sizes, so the mapping can be compared within one grid
#     - the mapping is monotone and geometric; it does not depend on color ids or hidden ordering
#   implicit_assumption_audit:
#     - original _find_blocks_2x2 assumed every source is exactly 2x2; replaced with filled-square component
#       detection after inferring the shared frame color
#     - original frame inference hard-coded a 4x4 border; generalized to a border around the detected kxk block
#     - original gap enumeration hard-coded frame corners/edges for size 4; generalized to any frame side length
#     - original generator placed only 4x4 boxes; now placement and clearance checks use each frame's true size
#     - the collision engine only depends on beam seeds and movement directions, so it remains valid unchanged
#   placement_legality:
#     - generator places full frames first, with at least a two-cell bbox gap between any pair of frames
#     - candidate gap sets are accepted only if each seed has enough runway and at least one seed per block has
#       more free runway than that block's size-derived cap, making the size effect observable
#   visual_observability:
#     - block size is directly visible in the input because the colored core changes from 1x1 to 2x2 to 3x3
#     - beam length differences are directly visible in the output because beams terminate at different distances
#       in otherwise open background regions
#   cleanliness:
#     - gap count is limited to 1..3 to avoid dense clutter once multiple differently sized framed blocks coexist
#   new_solver_assumptions:
#     - frame color is the most frequent non-background color in the input; generator enforces this by using
#       at least two framed blocks and a shared wall color
#     - non-frame, non-background color components are solid squares and correspond one-to-one with source blocks


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


def _beam_cap(block_size: int) -> int:
    return 2 * int(block_size) + 1


def _infer_frame_color(g: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in g.ravel() if int(v) != bg]
    assert vals, "expected at least one non-background color"
    return Counter(vals).most_common(1)[0][0]


def _find_blocks(g: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    seen: set[tuple[int, int]] = set()
    blocks: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v == bg or v == frame_color:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen or int(g[nr, nc]) != v:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            size = r1 - r0 + 1
            assert size == c1 - c0 + 1, "expected square inner block"
            assert len(cells) == size * size, "expected filled square inner block"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == v), "expected monochrome square block"
            assert r0 >= 1 and c0 >= 1 and r1 + 1 < h and c1 + 1 < w, "expected full frame inside grid"
            ring_vals: set[int] = set()
            fr0, fc0 = r0 - 1, c0 - 1
            fr1, fc1 = r1 + 1, c1 + 1
            for rr in range(fr0, fr1 + 1):
                for cc in range(fc0, fc1 + 1):
                    if rr in (fr0, fr1) or cc in (fc0, fc1):
                        vv = int(g[rr, cc])
                        if vv != bg and vv != v:
                            ring_vals.add(vv)
            assert ring_vals == {frame_color}, "expected a single shared frame color around each block"
            blocks.append((v, r0, c0, size))
    return sorted(blocks, key=lambda x: (x[1], x[2], x[0]))


def _frame_gaps_for_block(
    g: np.ndarray, bg: int, wall: int, br: int, bc: int, block_size: int
) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + block_size, bc + block_size
    out: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) != bg:
                continue
            dr = -1 if r == fr0 else (1 if r == fr1 else 0)
            dc = -1 if c == fc0 else (1 if c == fc1 else 0)
            assert dr != 0 or dc != 0, "expected candidate gap on frame border"
            if dr == 0:
                ok = int(g[r - 1, c]) == wall and int(g[r + 1, c]) == wall
            elif dc == 0:
                ok = int(g[r, c - 1]) == wall and int(g[r, c + 1]) == wall
            else:
                ok = int(g[r - dr, c]) == wall and int(g[r, c - dc]) == wall
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


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "max_steps")

    def __init__(
        self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...], max_steps: int
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.max_steps = int(max_steps)

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)

    def can_advance(self) -> bool:
        return len(self.path) - 1 < self.max_steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed monochrome squares of varying size; each size changes how far its beams can travel.

    Input structure:
    - bg is the most common color. A shared wall color forms one-cell-thick square frames around solid kxk color blocks.
    - The inner blocks may be 1x1, 2x2, 3x3, etc., and each frame still contains one or more single-cell gaps.

    Rule:
    - Every gap emits a beam of the enclosed block's color in the outward direction determined by that gap's side/corner.
    - The beam interaction logic is unchanged from the base task, but each beam now has a size-derived travel cap:
      a kxk source may advance at most 2k+1 cells beyond its gap seed.
    - 90-degree collisions leave a blank collision cell and redirect the participating beams to the bisector.
    - 45-degree collisions merge into the same zigzag continuation used in the original task.

    Key insight:
    - Source size is now part of the rule. Two blocks with the same kind of gap emit beams with the same directions and
      collision behavior, but larger inner squares sustain longer trajectories before stopping.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frame_color = _infer_frame_color(g, bg)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks(g, bg, frame_color)
    assert blocks, "expected at least one framed block in input"

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

    def spawn_beam(
        color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...], max_steps: int
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, max_steps)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc, block_size in blocks:
        for gr, gc in _frame_gaps_for_block(g, bg, frame_color, br, bc, block_size):
            fr0, fc0 = br - 1, bc - 1
            fr1, fc1 = br + block_size, bc + block_size
            dr = -1 if gr == fr0 else (1 if gr == fr1 else 0)
            dc = -1 if gc == fc0 else (1 if gc == fc1 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,), _beam_cap(block_size))

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
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
        bv.path.append(adj)
        bv.enter_dirs.append(u)
        bv.pos = adj
        paint_cell(bid_v, adj, u)

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell"

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
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (
            n2[0] - p2[0],
            n2[1] - p2[1],
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (
            p2[0] - n2[0],
            p2[1] - n2[1],
        ):
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
            if not b.can_advance():
                b.active = False
                continue
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
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    return out


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 5)
BLOCK_SIZE_VALUES = (1, 2, 3)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 3: 18, 4: 22}
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
MAX_GAP_COUNT = 3
SIZE_EFFECT_HEADROOM = 1
GEN_RETRIES = 400


def _grid_size_min(block_count: int) -> int:
    result = 12
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_block_sizes(block_count: int, rng: np.random.Generator) -> list[int]:
    active_count = 2 if block_count == 2 else int(rng.integers(2, 4))
    active_sizes = list(rng.choice(BLOCK_SIZE_VALUES, size=active_count, replace=False))
    sizes = active_sizes[:]
    while len(sizes) < block_count:
        sizes.append(int(rng.choice(active_sizes)))
    rng.shuffle(sizes)
    return sizes


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


@lru_cache(maxsize=None)
def _border_order(frame_size: int) -> tuple[tuple[int, int], ...]:
    s = int(frame_size)
    pts: list[tuple[int, int]] = []
    for c in range(s):
        pts.append((0, c))
    for r in range(1, s - 1):
        pts.append((r, s - 1))
    for c in range(s - 1, -1, -1):
        pts.append((s - 1, c))
    for r in range(s - 2, 0, -1):
        pts.append((r, 0))
    return tuple(pts)


@lru_cache(maxsize=None)
def _gap_combos(frame_size: int, gap_count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    border = _border_order(frame_size)
    index = {p: i for i, p in enumerate(border)}
    combos: list[tuple[tuple[int, int], ...]] = []
    border_len = len(border)
    for combo in itertools.combinations(border, gap_count):
        idxs = sorted(index[p] for p in combo)
        if any((idxs[(i + 1) % gap_count] - idxs[i]) % border_len == 1 for i in range(gap_count)):
            continue
        combos.append(combo)
    return tuple(combos)


def _gap_dir(frame_size: int, gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    last = frame_size - 1
    dr = -1 if gr == 0 else (1 if gr == last else 0)
    dc = -1 if gc == 0 else (1 if gc == last else 0)
    return dr, dc


def _frames_separated(a: tuple[int, int], sa: int, b: tuple[int, int], sb: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + sa - 1, ac0 + sa - 1
    br1, bc1 = br0 + sb - 1, bc0 + sb - 1
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, block_sizes: list[int], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    frame_sizes = [size + 2 for size in block_sizes]
    order = sorted(range(len(frame_sizes)), key=lambda i: frame_sizes[i], reverse=True)
    selected: dict[int, tuple[int, int]] = {}
    placed: list[tuple[tuple[int, int], int]] = []
    for idx in order:
        fs = frame_sizes[idx]
        coords = [
            (r, c)
            for r in range(PLACEMENT_MARGIN, n - PLACEMENT_MARGIN - fs + 1)
            for c in range(PLACEMENT_MARGIN, n - PLACEMENT_MARGIN - fs + 1)
        ]
        rng.shuffle(coords)
        found = None
        for pos in coords:
            if all(_frames_separated(pos, fs, other_pos, other_fs) for other_pos, other_fs in placed):
                found = pos
                break
        if found is None:
            return None
        selected[idx] = found
        placed.append((found, fs))
    return [selected[i] for i in range(len(block_sizes))]


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    block_size: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    frame_size = block_size + 2
    patch = np.full((frame_size, frame_size), frame_color, dtype=int)
    patch[1 : 1 + block_size, 1 : 1 + block_size] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _ray_runway(grid: np.ndarray, bg: int, seed: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = seed
    dr, dc = direction
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg:
        steps += 1
        nr += dr
        nc += dc
    return steps


def _clear_gap_combos(
    obstacles: np.ndarray, bg: int, r0: int, c0: int, block_size: int
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    frame_size = block_size + 2
    cap = _beam_cap(block_size)
    for gap_count in range(1, MAX_GAP_COUNT + 1):
        for combo in _gap_combos(frame_size, gap_count):
            runways = [
                _ray_runway(obstacles, bg, (r0 + gr, c0 + gc), _gap_dir(frame_size, (gr, gc))) for gr, gc in combo
            ]
            if min(runways) < MIN_RAY_STEPS:
                continue
            if max(runways) < cap + SIZE_EFFECT_HEADROOM:
                continue
            clear.setdefault(gap_count, []).append(combo)
    return clear


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    block_sizes = _pick_block_sizes(block_count, rng)
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, block_sizes, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, block_size in zip(positions, block_colors, block_sizes):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), block_size, ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    expressive_blocks = 0
    for (r0, c0), block_size in zip(positions, block_sizes):
        available = _clear_gap_combos(obstacles, bg, r0, c0, block_size)
        if not available:
            return None
        gap_count = int(rng.choice(sorted(available)))
        combos = available[gap_count]
        gaps = combos[int(rng.integers(len(combos)))]
        chosen_gaps.append(gaps)
        frame_size = block_size + 2
        runways = [
            _ray_runway(obstacles, bg, (r0 + gr, c0 + gc), _gap_dir(frame_size, (gr, gc))) for gr, gc in gaps
        ]
        if max(runways) >= _beam_cap(block_size) + SIZE_EFFECT_HEADROOM:
            expressive_blocks += 1

    if expressive_blocks < block_count:
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, block_size, gaps in zip(positions, block_colors, block_sizes, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), block_size, gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D4 input")
