# Variant of e12f9a14
# Dimension: D5 — 框架结构与缺口检测 (src)
# Change: frame_gap_detection——绿色框架为4x4或4x3的矩形边框，恰好有一个像素位置缺失形成缺口，缺口位置决... → 开放式框架（半封闭结构）

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for this D5 variant
# a) Concrete object + behavior trace:
#    Each colored 2x2 block is still wrapped by a 4x4 footprint, but one whole side of the frame is absent.
#    Example: if the top side is missing, the input shows a 4-cell-wide mouth above the block. The two cells of
#    that mouth aligned with the 2x2 block (local positions (0,1) and (0,2)) become beam seeds and both shoot
#    north. In the original task a single missing border pixel was both the detector and the seed; here the
#    detector is the missing side, while the launch pads are the two middle cells on that side.
# b) Complexity calibration:
#    The opening is visually obvious, but each block now emits a 2-cell-wide strip instead of a single ray.
#    This keeps the puzzle distinguishable from the original and still leaves enough interaction structure for
#    90-degree turns and later diagonal / zigzag behavior after collisions.
# c) Old-assumption audit:
#    The original _frame_gaps_for_block assumed isolated structural holes on a closed border and inferred beam
#    direction from each hole coordinate. That breaks here because a whole side is blank, so none of those holes
#    should be treated as four independent exits. The replacement detector must validate "exactly one side fully
#    open, all remaining border cells closed" and emit only the two middle launch pads of that side.
# d) Placement legality:
#    The generator keeps frames fully inside the grid with a margin, uses non-overlapping 4x4 bounding boxes,
#    and validates runway against a conservative obstacle grid so a beam cannot punch through another frame.
# e) Visual observability:
#    A half-open frame is readable from 2-3 examples without any hidden color ordering. The two parallel launch
#    cells make the new rule visible even in one-block examples, and multi-block examples still show downstream
#    collision behavior.
# f) Visual cleanliness:
#    Only the opening side is blank; the rest of the frame remains a crisp rectangle fragment. The generator
#    rejects cramped placements so wide openings do not visually blend with neighboring objects.
# g) Color inference:
#    Colors keep the original roles: one shared frame color, unique block colors, beams inherit the block color.
#    No extra color mapping is introduced.
#
# New solver assumptions introduced by this variant, and how generate_input enforces them:
# - Every block has a full 4x4 surrounding footprint with exactly one fully open side.
# - The remaining eight border cells are all frame-colored.
# - The two launch pads on the open side are background in the input.
# - Frames do not touch the outer grid edge, so open-side detection never has to guess through truncation.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_SET = frozenset(_BORDER_ORDER)
_OPEN_SIDE_CELLS = {
    (-1, 0): ((0, 0), (0, 1), (0, 2), (0, 3)),
    (1, 0): ((3, 0), (3, 1), (3, 2), (3, 3)),
    (0, -1): ((0, 0), (1, 0), (2, 0), (3, 0)),
    (0, 1): ((0, 3), (1, 3), (2, 3), (3, 3)),
}
_SEED_OFFSETS = {
    (-1, 0): ((0, 1), (0, 2)),
    (1, 0): ((3, 1), (3, 2)),
    (0, -1): ((1, 0), (2, 0)),
    (0, 1): ((1, 3), (2, 3)),
}
_CLOSED_BORDER_CELLS = {
    d: tuple(p for p in _BORDER_ORDER if p not in _OPEN_SIDE_CELLS[d]) for d in _OPEN_SIDE_CELLS
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


def _open_side_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> tuple[int, int]:
    """Detect which full side of the surrounding 4x4 frame is absent."""
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    for lr, lc in _BORDER_ORDER:
        r, c = fr0 + lr, fc0 + lc
        assert 0 <= r < h and 0 <= c < w, "expected full 4x4 frame footprint inside grid"

    matches: list[tuple[int, int]] = []
    for d, open_cells in _OPEN_SIDE_CELLS.items():
        if not all(int(g[fr0 + lr, fc0 + lc]) == bg for lr, lc in open_cells):
            continue
        if not all(int(g[fr0 + lr, fc0 + lc]) == wall for lr, lc in _CLOSED_BORDER_CELLS[d]):
            continue
        matches.append(d)
    assert len(matches) == 1, "expected exactly one open frame side per block"
    return matches[0]


def _seed_specs_for_open_side(br: int, bc: int, open_dir: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    fr0, fc0 = br - 1, bc - 1
    return [((fr0 + lr, fc0 + lc), open_dir) for lr, lc in _SEED_OFFSETS[open_dir]]


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
        # enter_dirs[i] is the move used to enter path[i]; for start, store the first move dir.
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch paired beams from half-open 4x4 frames around 2x2 blocks; collisions reuse the original turning rules.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block surrounded by frame color on
      exactly three sides of its 4x4 footprint.
    - The missing side is a contiguous 4-cell opening. The two opening cells aligned with the inner 2x2 block are
      blank launch pads.

    Rule:
    - Each launch pad emits a beam in the outward normal direction of the opening, so every block starts as a
      2-cell-wide parallel strip in the block's color.
    - Beams paint background cells until they leave the grid, hit a blocked cell, or collide with another beam.
    - 90-degree collisions leave a 1-cell gap at the meeting point and redirect the affected strip toward the
      angular bisector.
    - 45-degree collisions between a cardinal beam and a diagonal beam merge into the same zigzag behavior as the
      original task, with no gap at the meeting point.

    Key insight:
    - The launch cue is no longer a single missing pixel. You must detect which whole side of the frame is open,
      then fire the two middle cells of that side as a matched beam pair.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    # Build initial beams from open frame sides.
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

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at an open frame side"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        open_dir = _open_side_for_block(g, bg, frame_col, br, bc)
        for pos, d in _seed_specs_for_open_side(br, bc, open_dir):
            spawn_beam(col, pos, (d,))

    assert beams, "expected at least one beam spawned from open frame sides"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        """Return an active beam adjacent to bid that is moving in lockstep (same next_dir/pattern/phase)."""
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not hit the same cell"
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

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell after 45-degree resolution"

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True

    def diag_segment_owner(p: tuple[int, int], q: tuple[int, int]) -> int | None:
        """Return the beam id that currently owns the diagonal segment connecting p<->q, if any."""
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
        for bid, (nxt, d) in prop.items():
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
MIN_BEAM_PIXELS_PER_SEED = 2
GEN_RETRIES = 300


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
    open_dir: tuple[int, int] | None,
) -> None:
    """Stamp a closed or half-open 4x4 frame with a 2x2 inner block."""
    patch = np.full((4, 4), bg, dtype=int)
    for lr, lc in _BORDER_ORDER:
        patch[lr, lc] = frame_color
    if open_dir is not None:
        for lr, lc in _OPEN_SIDE_CELLS[open_dir]:
            patch[lr, lc] = bg
    patch[1:3, 1:3] = block_color
    paste(grid, patch, r0, c0)


def _seed_specs_for_frame_top_left(
    r0: int, c0: int, open_dir: tuple[int, int]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((r0 + lr, c0 + lc), open_dir) for lr, lc in _SEED_OFFSETS[open_dir]]


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


def _available_open_dirs(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for open_dir in _OPEN_SIDE_CELLS:
        seeds = _seed_specs_for_frame_top_left(r0, c0, open_dir)
        if _rays_clear(obstacles, bg, seeds):
            dirs.append(open_dir)
    return dirs


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

    chosen_dirs: list[tuple[int, int]] = []
    for r0, c0 in positions:
        available = _available_open_dirs(obstacles, bg, r0, c0)
        if not available:
            return None
        chosen_dirs.append(available[int(rng.integers(len(available)))])

    grid = np.full((n, n), bg, dtype=int)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, open_dir in zip(positions, block_colors, chosen_dirs):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), open_dir)
        seeds.extend(_seed_specs_for_frame_top_left(r0, c0, open_dir))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < len(seeds) * MIN_BEAM_PIXELS_PER_SEED:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D5-open-frame input")
