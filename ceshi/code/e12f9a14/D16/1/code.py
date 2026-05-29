# Variant of e12f9a14
# Dimension: D16 — 缺口位置的约束 (meta)
# Change: 缺口可以出现在框架的任意一条边的任意位置（角落或边的中间），方向从输入中推断... → 缺口位置由内部色块的偏移编码
#
# Design analysis
# ===============
# spec:
#   source_task: none
#   concrete_form:
#     original: each 2x2 monochrome block sat inside a 4x4 frame whose border contained 1-4 explicit bg holes;
#       each hole was also the first painted beam cell.
#     variant: each object is now a fully closed 5x5 frame. The 2x2 colored block is placed inside the 3x3 cavity
#       at exactly one of four corner offsets: NW, NE, SW, SE. That offset encodes a single diagonal launch direction.
#       The beam "pierces" the matching frame corner, so the output repaints that one border corner with the block color
#       and continues diagonally outward from there.
#   behavior_trace:
#     example:
#       input: a 5x5 green frame contains a red 2x2 block in the cavity's top-left corner.
#       original logic at this stage: solver searched the frame border for bg gaps and found none, so no beam would exist.
#       variant logic: solver reads the top-left offset, maps it to direction (-1,-1), seeds a red beam on the frame's
#         top-left corner cell, then extends it outward using the unchanged collision simulator.
#       difference: direction is now inferred from the block's placement, not from a missing wall pixel.
#   complexity_calibration:
#     why 5x5: a 4x4 frame has only a 2x2 cavity, so the inner 2x2 block has no positional freedom. Enlarging the frame
#       to 5x5 is the minimum structural change that creates four visually distinct offset states while preserving the
#       same "frame around a 2x2 block" motif.
#     why only four offsets: they match the four diagonal exits, keep the mapping small enough to infer from 2-4 samples,
#       and avoid ambiguous free-form displacement magnitudes.
#     collision richness: each frame now emits one diagonal beam instead of up to four mixed beams, so the generator
#       biases directions toward the grid center to keep crossings and redirects observable without changing the solver's
#       downstream collision rules.
#   hidden_assumptions_in_original_main:
#     - _frame_gaps_for_block assumed border bg cells were the only launch evidence. This is invalid once frames are closed.
#     - spawn_beam asserted the seed cell was bg. In the variant the seed is a frame wall corner, so the output is allowed
#       to overwrite that border cell.
#     - the old 4x4 geometry hard-coded block-at-center. With offset encoding, frame recognition must validate a 5x5 border
#       and verify that the non-bg interior cells form exactly one 2x2 block at one of four allowed cavity corners.
#     - is_free still assumes beams may travel only through bg cells. This remains valid because only the seed corner is
#       pierced; afterward beams stay outside frames and may still collide only in bg space.
#   placement_legality:
#     - generator places non-overlapping 5x5 frames with at least a 2-cell bbox gap, preserving the original cleanliness.
#     - every chosen direction must have at least MIN_RAY_STEPS empty cells beyond the pierced corner.
#     - solver-side frame detection expects the 3x3 cavity cells outside the 2x2 block to be bg, so the generator enforces
#       that exact interior pattern.
#   visual_observability:
#     - the key signal is visible in the input itself: the 2x2 block is visibly shoved into one corner of the cavity.
#     - the corresponding output beam always exits through the matching frame corner, making the offset-to-direction rule
#       legible without any hidden use of color ids or ordering.
#   neatness:
#     - only one border corner per frame changes in the output, so the frame stays readable as a frame rather than dissolving
#       into many openings.
#     - the 5x5 object preserves clear separation between frame color, block color, background, and beam traces.

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_FRAME_SIZE = 5
_FRAME_LAST = _FRAME_SIZE - 1
_BLOCK_OFFSETS = ((1, 1), (1, 2), (2, 1), (2, 2))
_OFFSET_TO_DIR = {
    (1, 1): (-1, -1),
    (1, 2): (-1, 1),
    (2, 1): (1, -1),
    (2, 2): (1, 1),
}
_DIAG_DIRS = tuple(_OFFSET_TO_DIR.values())


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


def _seed_cell(frame_r0: int, frame_c0: int, direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (frame_r0 if dr < 0 else frame_r0 + _FRAME_LAST, frame_c0 if dc < 0 else frame_c0 + _FRAME_LAST)


def _find_offset_frames(
    g: np.ndarray, bg: int
) -> list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]]:
    """
    Find fully closed 5x5 frames whose 3x3 cavity contains exactly one offset 2x2 block.

    Returns tuples:
      (block_color, block_r, block_c, frame_color, launch_dir, seed_cell)
    """
    h, w = g.shape
    frames: list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]] = []
    for r0 in range(h - _FRAME_SIZE + 1):
        for c0 in range(w - _FRAME_SIZE + 1):
            patch = g[r0 : r0 + _FRAME_SIZE, c0 : c0 + _FRAME_SIZE]
            border_vals = []
            border_ok = True
            for rr in range(_FRAME_SIZE):
                for cc in range(_FRAME_SIZE):
                    if rr not in (0, _FRAME_LAST) and cc not in (0, _FRAME_LAST):
                        continue
                    v = int(patch[rr, cc])
                    if v == bg:
                        border_ok = False
                        break
                    border_vals.append(v)
                if not border_ok:
                    break
            if not border_ok or len(set(border_vals)) != 1:
                continue

            frame_color = border_vals[0]
            candidates: list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]] = []
            for off_r, off_c in _BLOCK_OFFSETS:
                block_color = int(patch[off_r, off_c])
                if block_color in (bg, frame_color):
                    continue
                if not np.all(patch[off_r : off_r + 2, off_c : off_c + 2] == block_color):
                    continue
                interior_ok = True
                for rr in range(1, _FRAME_LAST):
                    for cc in range(1, _FRAME_LAST):
                        inside_block = off_r <= rr < off_r + 2 and off_c <= cc < off_c + 2
                        if inside_block:
                            continue
                        if int(patch[rr, cc]) != bg:
                            interior_ok = False
                            break
                    if not interior_ok:
                        break
                if not interior_ok:
                    continue
                direction = _OFFSET_TO_DIR[(off_r, off_c)]
                candidates.append(
                    (
                        block_color,
                        r0 + off_r,
                        c0 + off_c,
                        frame_color,
                        direction,
                        _seed_cell(r0, c0, direction),
                    )
                )
            assert len(candidates) <= 1, "expected a unique offset-coded block per closed frame"
            if len(candidates) == 1:
                frames.append(candidates[0])
    return frames


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from closed frames whose inner 2x2 blocks are offset within the cavity; beams still interact on collision.

    Input structure:
    - bg is the most common color. Each object is a fully closed 5x5 frame of one shared wall color.
    - Inside each 5x5 frame, a monochrome 2x2 block sits in one corner of the 3x3 cavity, leaving the rest of the cavity as bg.

    Rule:
    - The block's cavity offset encodes one diagonal launch direction: NW, NE, SW, or SE.
    - A beam of the block's color pierces the matching frame corner, repaints that corner cell, then continues through bg cells until edge or collision.
    - Beam-beam interactions are unchanged from the original task: 90-degree meetings create a 1-cell gap and redirect to the bisector; 45-degree meetings merge into a zigzag with no gap.

    Key insight: the visible cue is no longer a missing wall pixel; the solver must read the block's displacement inside a sealed frame and convert that offset into the exit corner and direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _find_offset_frames(g, bg)
    assert frames, "expected at least one closed offset-coded frame in input"

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
        assert int(g[pos[0], pos[1]]) != bg, "expected beam seed on a closed frame corner"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for color, _br, _bc, _frame_color, direction, seed in frames:
        spawn_beam(color, seed, (direction,))

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not meet in one cell"
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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a closed frame"
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


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 9, 2: 16, 5: 23}
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
MIN_BEAM_PIXELS_PER_BLOCK = 2
GEN_RETRIES = 300


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + _FRAME_LAST, ac0 + _FRAME_LAST
    br1, bc1 = br0 + _FRAME_LAST, bc0 + _FRAME_LAST
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    hi = n - _FRAME_SIZE - margin + 1
    all_positions = [(r, c) for r in range(margin, hi) for c in range(margin, hi)]
    selected: list[tuple[int, int]] = []
    for _ in range(count):
        viable = [p for p in all_positions if all(_frames_separated(p, q) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])
    return selected


def _draw_framed_block(
    grid: np.ndarray, r0: int, c0: int, bg: int, frame_color: int, block_color: int, direction: tuple[int, int]
) -> None:
    patch = np.full((_FRAME_SIZE, _FRAME_SIZE), frame_color, dtype=int)
    patch[1:_FRAME_LAST, 1:_FRAME_LAST] = bg
    off_r, off_c = {v: k for k, v in _OFFSET_TO_DIR.items()}[direction]
    patch[off_r : off_r + 2, off_c : off_c + 2] = block_color
    paste(grid, patch, r0, c0)


def _rays_clear(
    grid: np.ndarray, bg: int, seeds: list[tuple[tuple[int, int], tuple[int, int]]]
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


def _clear_dirs(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> list[tuple[int, int]]:
    clear: list[tuple[int, int]] = []
    for direction in _DIAG_DIRS:
        seed = _seed_cell(r0, c0, direction)
        if _rays_clear(obstacles, bg, [(seed, direction)]):
            clear.append(direction)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 9
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _preferred_dirs(n: int, r0: int, c0: int) -> list[tuple[int, int]]:
    center = (n - 1) / 2.0
    scored = []
    for direction in _DIAG_DIRS:
        seed = _seed_cell(r0, c0, direction)
        nxt = (seed[0] + direction[0], seed[1] + direction[1])
        dist = abs(nxt[0] - center) + abs(nxt[1] - center)
        scored.append((dist, direction))
    scored.sort()
    return [direction for _dist, direction in scored]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    placeholder_dir = (-1, -1)
    for (r0, c0), color in zip(positions, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), placeholder_dir)

    chosen_dirs: list[tuple[int, int]] = []
    for r0, c0 in positions:
        available = _clear_dirs(obstacles, bg, r0, c0)
        if not available:
            return None
        preferred = [d for d in _preferred_dirs(n, r0, c0) if d in available]
        assert preferred, "expected at least one direction after filtering"
        if len(preferred) > 1 and rng.random() < 0.25:
            chosen = preferred[int(rng.integers(len(preferred)))]
        else:
            chosen = preferred[0]
        chosen_dirs.append(chosen)

    grid = np.full((n, n), bg, dtype=int)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, direction in zip(positions, block_colors, chosen_dirs):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), direction)
        seeds.append((_seed_cell(r0, c0, direction), direction))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < block_count * MIN_BEAM_PIXELS_PER_BLOCK:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D16 input")
