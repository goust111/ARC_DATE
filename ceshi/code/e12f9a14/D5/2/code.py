# Variant of e12f9a14
# Dimension: D5 — 框架结构与缺口检测 (src)
# Change: frame_gap_detection——绿色框架为4x4或4x3的矩形边框，恰好有一个像素位置缺失形成缺口，缺口位置决... → 缺口大小可变

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D5 variant
# ==================================
# a) Concretization and behavior trace
#    - Keep the original 2x2 colored core and 4x4 frame.
#    - Change only the frame opening model: each frame now has exactly one contiguous
#      corner-anchored gap segment of length 2 or 3 on the 12-cell perimeter cycle.
#    - Length-2 opening: a corner cell plus one adjacent side cell are missing. This
#      emits one diagonal beam; the emitting seed is the non-corner missing cell.
#    - Length-3 opening: the corner cell plus both adjacent side cells are missing.
#      This emits two parallel diagonal beams, one from each non-corner missing cell.
#      The output therefore contains a visibly wider diagonal ribbon.
#    - Example: local NW opening [(1,0),(0,0),(0,1)] used to be impossible under the
#      single-gap assumption. In the variant, it is parsed as one 3-cell opening with
#      NW direction and two seeds at (1,0) and (0,1), so the solver spawns a paired
#      beam group instead of a single line.
# b) Complexity calibration
#    - Only two opening sizes are used: 2 -> narrow, 3 -> wide. This is enough to
#      make the D5 change obvious without adding many cases.
#    - The gap must stay corner-anchored. That keeps the diagonal direction legible
#      from the input and avoids ambiguous side-only openings.
# c) Hidden-assumption review
#    - The original gap detector treated each bg border pixel independently; that
#      breaks here because a 3-cell opening must be interpreted as one object, not
#      three separate exits.
#    - The original beam grouping logic only considered orthogonally adjacent heads.
#      Wide openings create paired beams that can be diagonally adjacent, so partner
#      detection is widened to 8-neighborhood and restricted to same-color beams to
#      avoid accidental grouping across different blocks.
#    - Diagonal crossing resolution previously updated only the two colliding beams.
#      With wide openings, the paired companion beams must turn together, so partner
#      grouping is applied there too.
# d) Placement legality
#    - The generator still places non-overlapping 4x4 frames with margin from the
#      grid edge and validates every emitted seed path against the obstacle grid.
#    - A generated frame is accepted only if its opening pattern is one of the valid
#      contiguous corner runs and at least one sample-wide 3-cell opening exists.
# e) Visual observability
#    - Human-solvable cue: a 2-cell corner bite produces a single diagonal line;
#      a 3-cell L-shaped corner bite produces a 2-lane diagonal band.
#    - The generator biases toward including at least one wide opening per sample,
#      and often mixes narrow + wide openings in the same grid for contrast.
# f) Visual neatness
#    - Colors are unchanged: frame color stays uniform, each core block keeps a unique
#      color, and beam colors still match their own block.
#    - Object spacing is preserved so wide ribbons remain separable from frames.
# g) Generator/solver contract
#    - main() assumes one contiguous opening run per frame, length 2 or 3, containing
#      exactly one corner. generate_input() constructs only such frames.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_CORNER_TO_DIR = {0: (-1, -1), 3: (-1, 1), 6: (1, 1), 9: (1, -1)}
_CORNER_NEIGHBORS = {0: {11, 1}, 3: {2, 4}, 6: {5, 7}, 9: {8, 10}}


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


def _gap_run_for_block(g: np.ndarray, bg: int, br: int, bc: int) -> tuple[int, ...]:
    """Return the unique contiguous bg run on the 4x4 border as border-order indices."""
    fr0, fc0 = br - 1, bc - 1
    present = [
        _BORDER_INDEX[(lr, lc)]
        for lr, lc in _BORDER_ORDER
        if int(g[fr0 + lr, fc0 + lc]) == bg
    ]
    assert present, "expected at least one missing frame cell"

    present_set = set(present)
    runs: list[tuple[int, ...]] = []
    for idx in sorted(present_set):
        if (idx - 1) % len(_BORDER_ORDER) in present_set:
            continue
        run = [idx]
        nxt = (idx + 1) % len(_BORDER_ORDER)
        while nxt in present_set and nxt != idx:
            run.append(nxt)
            nxt = (nxt + 1) % len(_BORDER_ORDER)
        runs.append(tuple(run))

    assert len(runs) == 1, "expected a single contiguous opening per frame"
    return runs[0]


def _opening_info(run: tuple[int, ...]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Map a perimeter run to (direction, emitting seed cells in local 4x4 coords)."""
    assert len(run) in (2, 3), "opening width must be 2 or 3 cells"
    corners = [idx for idx in run if idx in _CORNER_TO_DIR]
    assert len(corners) == 1, "opening must be anchored on exactly one corner"
    corner = corners[0]
    arm_idxs = tuple(idx for idx in run if idx != corner)
    assert set(arm_idxs).issubset(_CORNER_NEIGHBORS[corner]), "opening must hug the corner"
    if len(run) == 3:
        assert set(arm_idxs) == _CORNER_NEIGHBORS[corner], "3-cell opening must use both corner arms"
    return _CORNER_TO_DIR[corner], tuple(_BORDER_ORDER[idx] for idx in arm_idxs)


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
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks through corner openings; some openings emit paired beams.

    Input structure:
    - The background is the most common color. Each object is a 2x2 monochrome block enclosed by a 4x4 frame.
    - Every frame has exactly one contiguous corner opening made of 2 or 3 missing border pixels.
    - A 2-cell opening removes the corner plus one adjacent border cell; a 3-cell opening removes the corner plus both adjacent border cells.

    Rule:
    - The opening corner sets the outward diagonal direction (NW, NE, SW, or SE).
    - Each non-corner missing cell in the opening becomes an emitting seed. So a 2-cell opening emits one beam, while a 3-cell opening emits two synchronized diagonal beams.
    - Beams keep the original collision physics: 90-degree meetings blank the intersection and turn toward the bisector; 45-degree meetings merge into the zigzag pattern.

    Key insight:
    - The D5 change is not "more independent gaps"; it is "one wider opening object". The solver must first recover the single contiguous corner opening, then convert its width into one or two parallel beam seeds.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

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
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at an opening cell"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        _infer_frame_color(g, bg, br, bc, col)
        run = _gap_run_for_block(g, bg, br, bc)
        d, seeds_local = _opening_info(run)
        assert seeds_local, "each opening must emit at least one beam"
        for lr, lc in seeds_local:
            spawn_beam(col, (br - 1 + lr, bc - 1 + lc), (d,))

    assert beams, "expected at least one beam spawned from frame openings"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        """Return an active same-color beam adjacent in 8-neighborhood and moving in lockstep."""
        b = beams.get(bid)
        if b is None or not b.active:
            return None
        d = b.next_dir()
        partner: int | None = None
        for oid, ob in beams.items():
            if oid == bid or not ob.active:
                continue
            if ob.color != b.color:
                continue
            if ob.pattern != b.pattern or ob.phase != b.phase:
                continue
            if ob.next_dir() != d:
                continue
            dr = abs(ob.pos[0] - b.pos[0])
            dc = abs(ob.pos[1] - b.pos[1])
            if max(dr, dc) != 1 or (dr == 0 and dc == 0):
                continue
            if (b.pos[0] + d[0], b.pos[1] + d[1]) == ob.pos:
                continue
            if (ob.pos[0] + d[0], ob.pos[1] + d[1]) == b.pos:
                continue
            assert partner is None, "expected at most one adjacent lockstep partner"
            partner = oid
        return partner

    def lockstep_group(*bids: int) -> set[int]:
        group = set(bids)
        queue = list(bids)
        while queue:
            cur = queue.pop()
            partner = find_partner(cur)
            if partner is not None and partner not in group:
                group.add(partner)
                queue.append(partner)
        return group

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
            assert not (is_diag_a and is_diag_b), "diagonal beams should cross geometrically, not share a cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            for gid in lockstep_group(a, b):
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                beams[gid].active = True
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        assert is_card_a != is_card_b, "45-degree collision must involve exactly one cardinal beam"

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

        assert beams[bid_u].pos == cell, "cardinal beam head must end at the collision cell"

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
        for gid in lockstep_group(a, b):
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

        for gid in lockstep_group(a, b):
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
            nr, nc = b.pos[0] + d[0], b.pos[1] + d[1]
            nxt = (nr, nc)
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a frame or block"
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
            keyr = min(ca[0], cb[0])
            keyc = min(ca[1], cb[1])
            events.append((1, keyr, keyc, a, b, "cross"))

        for a, b in seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            keyr = min(pr, pr + dr)
            keyc = min(pc, pc + dc)
            events.append((1, keyr, keyc, a, b, "segcross"))

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
GEN_RETRIES = 400

_OPENING_RUNS = {
    2: (
        (11, 0), (0, 1),
        (2, 3), (3, 4),
        (5, 6), (6, 7),
        (8, 9), (9, 10),
    ),
    3: (
        (11, 0, 1),
        (2, 3, 4),
        (5, 6, 7),
        (8, 9, 10),
    ),
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
    gap_run: tuple[int, ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for idx in gap_run:
        gr, gc = _BORDER_ORDER[idx]
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


def _opening_seeds(r0: int, c0: int, run: tuple[int, ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    direction, seed_cells = _opening_info(run)
    return [((r0 + gr, c0 + gc), direction) for gr, gc in seed_cells]


def _clear_openings(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> dict[int, list[tuple[int, ...]]]:
    clear: dict[int, list[tuple[int, ...]]] = {}
    for size, runs in _OPENING_RUNS.items():
        for run in runs:
            if _rays_clear(obstacles, bg, _opening_seeds(r0, c0, run)):
                clear.setdefault(size, []).append(run)
    return clear


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
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), ())

    available_per_block: list[dict[int, list[tuple[int, ...]]]] = []
    for r0, c0 in positions:
        available = _clear_openings(obstacles, bg, r0, c0)
        if not available:
            return None
        available_per_block.append(available)

    wide_candidates = [i for i, available in enumerate(available_per_block) if 3 in available]
    if not wide_candidates:
        return None
    wide_idx = wide_candidates[int(rng.integers(len(wide_candidates)))]

    narrow_idx: int | None = None
    narrow_candidates = [i for i, available in enumerate(available_per_block) if i != wide_idx and 2 in available]
    if block_count >= 2 and narrow_candidates and rng.random() < 0.8:
        narrow_idx = narrow_candidates[int(rng.integers(len(narrow_candidates)))]

    chosen_runs: list[tuple[int, ...]] = []
    for i, available in enumerate(available_per_block):
        if i == wide_idx:
            size = 3
        elif i == narrow_idx:
            size = 2
        else:
            sizes = sorted(available)
            if 2 in available and 3 in available:
                size = 3 if rng.random() < 0.6 else 2
            else:
                size = sizes[int(rng.integers(len(sizes)))]
        runs = available[size]
        chosen_runs.append(runs[int(rng.integers(len(runs)))])

    grid = np.full((n, n), bg, dtype=int)
    seed_total = 0
    all_seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, run in zip(positions, block_colors, chosen_runs):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), run)
        seeds = _opening_seeds(r0, c0, run)
        seed_total += len(seeds)
        all_seeds.extend(seeds)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, all_seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < seed_total * MIN_BEAM_PIXELS_PER_SEED:
        return None
    if not any(len(run) == 3 for run in chosen_runs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D5 variant input")
