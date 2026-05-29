# Combo variant of e12f9a14 (index 22)
# combo_spec: D5/0+D10/2+D16/0
# Name: 多缺口+角落多方向发射+缺口仅在角落

# Design analysis for the fused variant.
# 1. Concrete behavior:
#    - Original local rule: one missing border cell launches exactly one ray whose
#      outward direction is read directly from that gap's side/corner location.
#    - Fused local rule: every usable gap is a missing corner of the 4x4 frame,
#      and each such corner launches two diagonal rays instead of one. We use a
#      fixed corner-to-fan mapping that matches the prompt's example:
#        NW -> {NW, NE}, NE -> {NE, SE}, SE -> {SE, SW}, SW -> {SW, NW}.
#      Example: a frame with NW and SE corner holes used to emit two beams
#      (NW and SE). It now emits four beams: NW+NE from the top-left hole and
#      SE+SW from the bottom-right hole, creating two visible fans.
#    - Collision rules are intentionally unchanged. The fusion acts at launch
#      time; afterwards the same 90-degree gap rule and 45-degree zigzag rule
#      still govern the ray network. This creates genuine interaction instead of
#      three unrelated edits.
# 2. Complexity calibration:
#    - To make the change visible but still learnable from 2-4 examples, the
#      generator uses only four corner categories and only two gap-count tiers:
#      each frame has 2-4 corner gaps, never side gaps.
#    - The difficult part is not hidden color bookkeeping. Colors stay the same
#      as the base task: each beam uses its 2x2 block color.
# 3. Broken original assumptions:
#    - "one structural gap -> one beam" is false now; launch code must expand one
#      corner gap into two initial directions.
#    - "seed position is unique per beam" is false now because two beams share one
#      gap cell. The simulator therefore allows multiple beams to start from the
#      same seed while painting that seed cell only once.
#    - "at most one adjacent lockstep partner" is no longer safe in dense fan
#      layouts, so partner grouping is upgraded from a single optional beam to a
#      whole connected lockstep component.
#    - Edge-gap logic is obsolete. The solver now only accepts corner gaps, and
#      the generator only constructs such inputs.
# 4. Placement legality:
#    - Frames remain 4x4 around unique-color 2x2 blocks.
#    - The generator keeps frames well separated, away from borders, and validates
#      every emitted direction against an obstacle grid before committing gaps.
#    - It then runs the fused solver itself and rejects any sample that crashes or
#      produces too little beam growth.
# 5. Observability and tidiness:
#    - Missing corners are easy to see in the input, and the output's extra rays
#      are anchored exactly at those same corners, so the dual-launch rule is
#      visually local and human-readable.
#    - To avoid messy overlays, the generator biases toward larger boards, wider
#      frame spacing, and rejects degenerate samples whose fused output matches a
#      single-ray corner baseline.
# 6. New solver assumptions introduced here:
#    - Every true gap is one of the four frame corners.
#    - Every frame has at least two corner gaps.
#    - The two fan directions per corner are fixed and shared across samples.
#    generate_input() enforces all three.

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_ROOT = Path("/home/lixinyu/snap/new")
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_CORNER_OFFSETS = ((0, 0), (0, 3), (3, 3), (3, 0))
_CORNER_TO_SINGLE_DIR = {
    (0, 0): (-1, -1),
    (0, 3): (-1, 1),
    (3, 3): (1, 1),
    (3, 0): (1, -1),
}
_CORNER_TO_FAN_DIRS = {
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


def _frame_corner_gaps_for_block(
    g: np.ndarray, bg: int, wall: int, br: int, bc: int
) -> list[tuple[int, int]]:
    """Return structural corner gaps only; this fused variant never uses side gaps."""
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    out: list[tuple[int, int]] = []
    for gr, gc in _CORNER_OFFSETS:
        r = fr0 + gr
        c = fc0 + gc
        if not (0 <= r < h and 0 <= c < w):
            continue
        if int(g[r, c]) != bg:
            continue
        dr = -1 if gr == 0 else 1
        dc = -1 if gc == 0 else 1
        if not (0 <= r - dr < h and 0 <= c - dc < w):
            continue
        if int(g[r - dr, c]) == wall and int(g[r, c - dc]) == wall:
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


def _corner_local_from_abs(br: int, bc: int, gap: tuple[int, int]) -> tuple[int, int]:
    fr0, fc0 = br - 1, bc - 1
    return gap[0] - fr0, gap[1] - fc0


def _launch_dirs_for_corner(local_gap: tuple[int, int], dual_corner: bool) -> tuple[tuple[int, int], ...]:
    if dual_corner:
        return _CORNER_TO_FAN_DIRS[local_gap]
    return (_CORNER_TO_SINGLE_DIR[local_gap],)


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


def _solve(grid: np.ndarray, *, dual_corner: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0
    seed_claimed: set[tuple[int, int]] = set()

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
        assert int(g[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        if pos not in seed_claimed:
            seed_claimed.add(pos)
            paint_cell(bid, pos, pattern[0])
        else:
            out[pos[0], pos[1]] = int(color)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_corner_gaps_for_block(g, bg, frame_col, br, bc)
        assert len(gaps) >= 2, "fused variant expects at least two corner gaps per frame"
        for gap in gaps:
            local_gap = _corner_local_from_abs(br, bc, gap)
            assert local_gap in _CORNER_TO_SINGLE_DIR, "fused variant expects corner gaps only"
            for d in _launch_dirs_for_corner(local_gap, dual_corner=dual_corner):
                spawn_beam(col, gap, (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def is_lockstep_neighbor(a: int, b: int) -> bool:
        ba = beams[a]
        bb = beams[b]
        if not ba.active or not bb.active:
            return False
        if ba.pattern != bb.pattern or ba.phase != bb.phase:
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
        todo = [start]
        seen: set[int] = set()
        while todo:
            bid = todo.pop()
            if bid in seen or bid not in beams or not beams[bid].active:
                continue
            seen.add(bid)
            for oid in beams:
                if oid not in seen and oid != bid and is_lockstep_neighbor(bid, oid):
                    todo.append(oid)
        return seen

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams should cross geometrically, not share a cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = lockstep_group(a) | lockstep_group(b)
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
        for gid in (a, b):
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            beams[gid].active = True

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
        group = lockstep_group(a) | lockstep_group(b)
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            beams[gid].active = True

    max_iters = h * w * 240
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

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit dense diagonal beam fans from corner-gapped framed 2x2 blocks; the fans still interact under the original collision system.

    Input structure:
    - The background is the most common color. Each object is a 2x2 monochrome block enclosed by a 4x4 frame of one shared wall color.
    - Every usable gap is a missing frame corner, never a side hole, and every frame contains multiple such corner gaps.

    Rule:
    - Repaint each corner gap with the enclosed block color.
    - A corner gap emits two diagonal rays in a fixed adjacent fan determined by the corner itself:
      NW -> NW+NE, NE -> NE+SE, SE -> SE+SW, SW -> SW+NW.
    - Rays continue through background cells until they leave the grid or hit a collision. Afterwards the same collision mechanics as the source task apply:
      90-degree meetings create a permanent 1-cell blank gap and redirect both beams to the bisector, while 45-degree diagonal/cardinal meetings merge into a zigzag without a blank gap.

    Key insight:
    - The fusion does not change the downstream collision language; it changes how much structure a single visible corner gap encodes. Solving the task means first reading each missing corner as a 2-ray fan source, then following the unchanged collision rules on the denser network.
    """
    return _solve(grid, dual_corner=True)


def _single_corner_baseline(grid: np.ndarray) -> np.ndarray:
    """Corner-only baseline used to reject degenerate generator samples."""
    return _solve(grid, dual_corner=False)


# --------------- generator (combo D5/0 + D10/2 + D16/0) ---------------

BLOCK_COUNT_RANGE = (1, 6)  # uniform draw [lo, hi)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 10, 2: 16, 3: 20, 4: 24, 5: 27}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 3
PLACEMENT_MARGIN = 3
MIN_RAY_STEPS = 3
MIN_BEAM_PIXELS_PER_LAUNCH = 2
MIN_FUSION_DELTA = 4
GEN_RETRIES = 500

_CORNER_COMBOS = {
    k: list(itertools.combinations(_CORNER_OFFSETS, k))
    for k in range(2, 5)
}


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [
        (r, c)
        for r in range(margin, n - 3 - margin)
        for c in range(margin, n - 3 - margin)
    ]
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


def _launch_seeds_for_gaps(
    r0: int, c0: int, gaps: tuple[tuple[int, int], ...], *, dual_corner: bool
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for gap in gaps:
        seed = (r0 + gap[0], c0 + gap[1])
        for d in _launch_dirs_for_corner(gap, dual_corner=dual_corner):
            seeds.append((seed, d))
    return seeds


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
            if int(grid[nr, nc]) != bg:
                return False
            steps += 1
            nr += dr
            nc += dc
        if steps < MIN_RAY_STEPS:
            return False
    return True


def _clear_corner_combos(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _CORNER_COMBOS.items():
        for combo in combos:
            seeds = _launch_seeds_for_gaps(r0, c0, combo, dual_corner=True)
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
    result = 10
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def measure_fusion_delta(grid: np.ndarray) -> dict[str, int]:
    fused = main(grid)
    base = _single_corner_baseline(grid)
    gap_count = 0
    bg = get_bg(np.asarray(grid, dtype=int))
    for _col, br, bc in _find_blocks_2x2(np.asarray(grid, dtype=int), bg):
        frame_col = _infer_frame_color(np.asarray(grid, dtype=int), bg, br, bc, _col)
        gap_count += len(_frame_corner_gaps_for_block(np.asarray(grid, dtype=int), bg, frame_col, br, bc))
    return {
        "gap_count": int(gap_count),
        "launch_count": int(gap_count * 2),
        "fusion_delta_pixels": int(np.count_nonzero(fused != base)),
        "beam_pixels_added": int(np.count_nonzero(fused != np.asarray(grid, dtype=int))),
    }


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

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    for r0, c0 in positions:
        available = _clear_corner_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        gap_count_choices = sorted(available)
        gap_count = int(rng.choice(gap_count_choices))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    if max(len(gaps) for gaps in chosen_gaps) < 3 and sum(len(gaps) for gaps in chosen_gaps) < 6:
        return None

    grid = np.full((n, n), bg, dtype=int)
    launch_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        frame_seeds = _launch_seeds_for_gaps(r0, c0, gaps, dual_corner=True)
        seeds.extend(frame_seeds)
        launch_total += len(frame_seeds)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        fused = main(grid)
        base = _single_corner_baseline(grid)
    except Exception:
        return None

    if np.count_nonzero(fused != grid) < launch_total * MIN_BEAM_PIXELS_PER_LAUNCH:
        return None
    if np.count_nonzero(fused != base) < MIN_FUSION_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused e12f9a14 input")
