# Combo variant of e12f9a14 (index 7)
# combo_spec: D4/1+D10/1
# Name: 多色内部像素+双信息源方向决定

import itertools

import numpy as np

from arc_utils import get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and behavior trace
#    The source object is still a 4x4 frame around a 2x2 chamber, but the chamber
#    now contains a fixed checkerboard of two colors:
#      [[A, B],
#       [B, A]]
#    Gaps are restricted to side border cells:
#      top    -> (0,1) or (0,2)
#      right  -> (1,3) or (2,3)
#      bottom -> (3,2) or (3,1)
#      left   -> (2,0) or (1,0)
#    For each gap, the unique inward-adjacent inner pixel does two jobs at once:
#    - its color becomes the beam color
#    - its row/column supplies the orthogonal sign missing from the gap side
#    Example:
#    - old task, local gap (0,1): beam color = one frame-wide block color,
#      launch = straight up/cardinal from the top side logic.
#    - D4 alone would change only the color source: local gap (0,1) would use the
#      top-left pixel color A, but keep the old direction rule.
#    - D10 alone would change only the direction source: local gap (0,1) would use
#      the inward pixel position to choose left vs right, but keep one beam color.
#    - combo rule here: local gap (0,1) reads the top-left inner pixel, so the
#      beam launches (-1,-1) and uses color A; local gap (0,2) launches (-1,+1)
#      and uses color B. The same visible local cue determines both properties.
#
# b) Complexity calibration
#    The variant keeps only two inner colors and only eight legal side-gap slots.
#    That is enough to make the fusion visible without exploding the case count.
#    Restricting to side gaps also guarantees every initial beam is diagonal, so
#    the path rule is visibly different from the original side-gap cardinal beams.
#
# c) Implicit assumption audit
#    - Original `_find_blocks_2x2` assumed a solid monochrome 2x2; that breaks.
#      The combo solver replaces it with checkerboard detection.
#    - Original `_infer_frame_color` excluded only one block color; here both
#      inner colors must be excluded.
#    - Original spawning used one frame-wide beam color and let the gap fully
#      determine direction. Both assumptions are invalid now: the spawned beam
#      must read one specific inward-adjacent pixel for both color and the second
#      direction component.
#    - Collision handling still assumes beams only meet at 45 or 90 degrees. That
#      remains valid because every accepted input is validated by running `main()`.
#
# d) Placement legality
#    The generator still uses separated 4x4 frame footprints and checks that each
#    chosen beam has at least two clear steps through background. Gap sets are
#    picked only from the side-gap catalog, so no corner ambiguity is generated.
#
# e) Visual observability
#    A human can see the relevant evidence directly in the pixels: the beam comes
#    from a side opening, and the touched inner checker cell explains both which
#    color leaves the frame and whether the diagonal leans left/right or up/down.
#    The generator additionally enforces sample-level support:
#    - every frame emits both inner colors
#    - across the grid, at least one side family uses both of its two slots, so
#      the solver must notice the slot-specific direction change
#
# f) Visual cleanliness
#    The frame, the 2x2 checkerboard, and the outgoing beams remain spatially
#    separated. No extra markers are added, and the changed rule is encoded only
#    in already-visible local geometry.
#
# g) Color-change inferability
#    There are exactly two colors per frame and the color mapping is structural,
#    not numeric: the beam copies the inward-adjacent checker cell. Because every
#    frame uses both colors, the puzzle does not collapse back into a visually
#    monochrome source where the D4 part would be invisible.
#
# 9) New solver assumptions and generator guarantees
#    - main() assumes every valid source is a checkerboard 2x2 framed by one
#      border color with at least one structural side gap.
#    - main() assumes side gaps only. generate_input() never produces corner gaps.
#    - main() assumes each emitted beam's color/direction can be read from the
#      unique inner pixel adjacent to that gap. The checkerboard layout and side
#      gap positions make that adjacency unique in every generated sample.
# -----------------------------------------------------------------------------


Inner2x2 = tuple[tuple[int, int], tuple[int, int]]

_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_SIDE_GAPS = (
    (0, 1), (0, 2),
    (1, 3), (2, 3),
    (3, 2), (3, 1),
    (2, 0), (1, 0),
)


def _is_checker_inner(inner: Inner2x2, bg: int) -> bool:
    a, b = inner[0]
    c, d = inner[1]
    return bg not in (a, b, c, d) and a == d and b == c and a != b


def _checker_inner(a: int, b: int) -> Inner2x2:
    return ((int(a), int(b)), (int(b), int(a)))


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


def _old_gap_dir(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, inner_cols: set[int]) -> int | None:
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
            if v == bg:
                continue
            if v in inner_cols:
                return None
            vals.append(v)
    frame_colors = sorted(set(vals))
    if len(frame_colors) != 1:
        return None
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    gaps: list[tuple[int, int]] = []
    for lr, lc in _SIDE_GAPS:
        r, c = fr0 + lr, fc0 + lc
        if not (0 <= r < h and 0 <= c < w):
            continue
        if int(g[r, c]) != bg:
            continue
        if lr in (0, 3):
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        if ok:
            gaps.append((r, c))
    return gaps


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, Inner2x2, int, tuple[tuple[int, int], ...]]]:
    h, w = g.shape
    blocks: list[tuple[int, int, Inner2x2, int, tuple[tuple[int, int], ...]]] = []
    for br in range(h - 1):
        for bc in range(w - 1):
            inner: Inner2x2 = (
                (int(g[br, bc]), int(g[br, bc + 1])),
                (int(g[br + 1, bc]), int(g[br + 1, bc + 1])),
            )
            if not _is_checker_inner(inner, bg):
                continue
            frame_col = _infer_frame_color(g, bg, br, bc, {inner[0][0], inner[0][1]})
            if frame_col is None:
                continue
            gaps = tuple(_frame_gaps_for_block(g, bg, frame_col, br, bc))
            if not gaps:
                continue
            blocks.append((br, bc, inner, frame_col, gaps))
    return blocks


def _gap_source_color_and_dir(inner: Inner2x2, br: int, bc: int, gap: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    fr0, fc0 = br - 1, bc - 1
    lr = gap[0] - fr0
    lc = gap[1] - fc0
    assert (lr, lc) in _SIDE_GAPS, "expected a side gap on the 4x4 frame"
    ir = 0 if lr <= 1 else 1
    ic = 0 if lc <= 1 else 1
    color = int(inner[ir][ic])
    if lr == 0:
        return color, (-1, -1 if ic == 0 else 1)
    if lr == 3:
        return color, (1, -1 if ic == 0 else 1)
    if lc == 0:
        return color, (-1 if ir == 0 else 1, -1)
    return color, (-1 if ir == 0 else 1, 1)


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
    Emit beams from framed checkerboards where one inward-adjacent pixel jointly determines each beam's color and diagonal lean.

    Input structure:
    - The background is the most common color, and one shared wall color forms 4x4 frames.
    - Inside each frame is a 2x2 checkerboard using exactly two colors on opposite corners.
    - Frame openings appear only on side border cells, never on the four corners.

    Rule:
    - Each side gap launches one diagonal beam.
    - The gap side fixes the outward axis, while the unique inward-adjacent checker cell fixes the orthogonal sign and also donates the beam color.
    - Beams keep painting through background until they hit the grid boundary or collide; 90-degree meetings leave a blank pivot and bisect, while 45-degree meetings merge into the same zigzag behavior as the base task.

    Key insight:
    - A solver must fuse two visible cues per beam: the opening tells which axis points outward, and the touched inner checker cell tells both which color exits and which way the diagonal tilts.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one framed checkerboard 2x2 block in input"

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
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for br, bc, inner, _frame_col, gaps in blocks:
        for gap in gaps:
            color, d = _gap_source_color_and_dir(inner, br, bc, gap)
            spawn_beam(color, gap, (d,))

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not share a cell"
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
        if (
            cell_owner.get(n2) == b
            and cell_index.get(n2, 0) > 0
            and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1])
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif (
            cell_owner.get(p2) == b
            and cell_index.get(p2, 0) > 0
            and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1])
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


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 5: 20}
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
GEN_RETRIES = 500


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_SIDE_GAPS, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    all_positions = [(r, c) for r in range(PLACEMENT_MARGIN, n - 5) for c in range(PLACEMENT_MARGIN, n - 5)]
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
    inner: Inner2x2,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), bg, dtype=int)
    for lr, lc in _BORDER_ORDER:
        patch[lr, lc] = frame_color
    patch[1, 1] = inner[0][0]
    patch[1, 2] = inner[0][1]
    patch[2, 1] = inner[1][0]
    patch[2, 2] = inner[1][1]
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _launch_dir_from_gap_and_inner(local_gap: tuple[int, int], inner: Inner2x2) -> tuple[int, int]:
    color, d = _gap_source_color_and_dir(inner, 1, 1, (local_gap[0], local_gap[1]))
    _ = color
    return d


def _gap_local_color(inner: Inner2x2, local_gap: tuple[int, int]) -> int:
    ir = 0 if local_gap[0] <= 1 else 1
    ic = 0 if local_gap[1] <= 1 else 1
    return int(inner[ir][ic])


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
    inner: Inner2x2,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            colors = {_gap_local_color(inner, gap) for gap in combo}
            dirs = {_launch_dir_from_gap_and_inner(gap, inner) for gap in combo}
            if len(colors) < 2 or len(dirs) < 2:
                continue
            seeds = [((r0 + gr, c0 + gc), _launch_dir_from_gap_and_inner((gr, gc), inner)) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[Inner2x2]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    ordered_pairs = [(a, b) for a in palette for b in palette if a != b]
    idxs = rng.choice(len(ordered_pairs), size=block_count, replace=False)
    inners = [_checker_inner(*ordered_pairs[int(i)]) for i in np.atleast_1d(idxs)]
    return bg, frame_color, inners


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    changed_dirs = 0
    total_gaps = 0
    mixed_frames = 0
    side_slot_support = 0
    family_slots = {"top": set(), "right": set(), "bottom": set(), "left": set()}
    for br, bc, inner, _frame_col, gaps in blocks:
        gap_colors = set()
        for gap in gaps:
            local_gap = (gap[0] - (br - 1), gap[1] - (bc - 1))
            color, new_dir = _gap_source_color_and_dir(inner, br, bc, gap)
            old_dir = _old_gap_dir(local_gap)
            total_gaps += 1
            gap_colors.add(color)
            if new_dir != old_dir:
                changed_dirs += 1
            if local_gap[0] == 0:
                family_slots["top"].add(local_gap[1])
            elif local_gap[0] == 3:
                family_slots["bottom"].add(local_gap[1])
            elif local_gap[1] == 0:
                family_slots["left"].add(local_gap[0])
            else:
                family_slots["right"].add(local_gap[0])
        if len(gap_colors) >= 2:
            mixed_frames += 1
    for slots in family_slots.values():
        if len(slots) >= 2:
            side_slot_support += 1
    return {
        "frame_count": len(blocks),
        "total_gaps": total_gaps,
        "changed_dirs": changed_dirs,
        "mixed_frames": mixed_frames,
        "side_slot_support": side_slot_support,
    }


def _variant_signal(grid: np.ndarray, expected_blocks: int) -> bool:
    metrics = _variant_metrics(grid)
    return (
        metrics["frame_count"] == expected_blocks
        and metrics["mixed_frames"] == expected_blocks
        and metrics["total_gaps"] >= expected_blocks * 2
        and metrics["changed_dirs"] == metrics["total_gaps"]
        and metrics["side_slot_support"] >= 1
    )


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, inners = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), inner in zip(positions, inners):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, inner, ())

    chosen_gaps = []
    for (r0, c0), inner in zip(positions, inners):
        available = _clear_gap_combos(obstacles, bg, r0, c0, inner)
        if not available:
            return None
        gap_count = int(rng.choice(sorted(available)))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), inner, gaps in zip(positions, inners, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, inner, gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _launch_dir_from_gap_and_inner(gap, inner)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None
    if not _variant_signal(grid, block_count):
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
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
