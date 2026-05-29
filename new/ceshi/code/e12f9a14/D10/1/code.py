# Variant of e12f9a14
# Dimension: D10 — 射线发射方向与缺口位置的映射 (path)
# Change: 缺口在框架的某条边上时，射线向该方向的对角线发射——缺口方向与射线方向严格对应... → 缺口方向决定水平/垂直分量，内部色块位置决定另一分量
#
# Design analysis
# - Concrete object change:
#   - The old task used a solid 2x2 monochrome block inside a 4x4 frame.
#   - This variant replaces that with a single colored marker pixel placed in one
#     of the four interior cells of the 2x2 chamber.
#   - The marker now visibly carries two signs at once:
#     - top vs bottom row = vertical bias
#     - left vs right column = horizontal bias
#   - Gaps are restricted to side cells of the frame border (never corners), so
#     each gap contributes exactly one axis of motion and the marker contributes
#     the orthogonal axis.
# - Behavior trace:
#   - Old logic for a top-edge gap at local (0,1): launch direction was (-1,0),
#     because the gap itself fully determined the beam.
#   - New logic for the same top-edge gap:
#     - the gap still fixes the outward vertical component to -1
#     - the interior marker column fixes the horizontal component:
#       left-column marker -> -1, right-column marker -> +1
#     - example: marker at local (1,2) yields launch (-1,+1)
#   - The downstream collision rules are intentionally kept unchanged; only the
#     initial encoding of direction is altered.
# - Complexity calibration:
#   - Using a single marker keeps the added reasoning degree small and visible:
#     there are only four marker quadrants.
#   - Side gaps only means every emitted beam is diagonal from the start, so the
#     new D10 change is always visually different from the original cardinal side
#     beams.
#   - The generator also rejects samples that fail to show both values of at
#     least one marker-controlled sign across same-family gaps, so the rule is
#     inferable from a small set of examples.
# - Solver assumption audit:
#   - Old `_find_blocks_2x2` depended on a solid 2x2 blob. That is invalid now,
#     so frame detection is replaced with a direct scan for 4x4 framed chambers
#     containing exactly one interior marker.
#   - Old spawn code treated `(dr,dc)` from the gap as the full launch vector.
#     That is replaced with `_launch_dir_from_gap_and_marker`, which combines the
#     side-gap axis with the marker's row/column signs.
#   - Collision code assumes only 45-degree or 90-degree meetings. That remains
#     valid for accepted samples because the generator validates every input by
#     running `main()` and rejecting unsupported geometries.
# - Placement legality:
#   - Frames still use a 4x4 footprint and keep a gap between bounding boxes.
#   - The generator computes launch directions before finalizing gaps and checks
#     that each beam has at least two clear diagonal steps through background.
#   - Inputs are accepted only if `main()` succeeds and paints enough output.
# - Visual observability / neatness:
#   - The marker is the only colored interior cell, so its quadrant is easy to
#     read and never obscured by frame color.
#   - Side-gap-only frames avoid ambiguous corner semantics and keep beam exits
#     visually clean.
#   - Colors are not used for ordering or hidden control; only visible geometry
#     (gap side + marker quadrant) determines direction.

import itertools

import numpy as np

from arc_utils import get_bg, paste


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
_INTERIOR_CELLS = ((1, 1), (1, 2), (2, 1), (2, 2))


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


def _marker_bias(local_marker: tuple[int, int]) -> tuple[int, int]:
    mr, mc = local_marker
    assert mr in (1, 2) and mc in (1, 2), "expected marker inside the 2x2 chamber"
    vr = -1 if mr == 1 else 1
    hc = -1 if mc == 1 else 1
    return vr, hc


def _launch_dir_from_gap_and_marker(
    local_gap: tuple[int, int], local_marker: tuple[int, int]
) -> tuple[int, int]:
    gr, gc = local_gap
    vr, hc = _marker_bias(local_marker)
    if gr == 0:
        return (-1, hc)
    if gr == 3:
        return (1, hc)
    if gc == 0:
        return (vr, -1)
    if gc == 3:
        return (vr, 1)
    raise AssertionError("expected a side gap on the 4x4 frame border")


def _old_gap_dir(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _frame_gaps_for_frame(g: np.ndarray, bg: int, wall: int, r0: int, c0: int) -> list[tuple[int, int]]:
    h, w = g.shape
    gaps: list[tuple[int, int]] = []
    for lr, lc in _SIDE_GAPS:
        r, c = r0 + lr, c0 + lc
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


def _find_marked_frames(g: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[int, int], int]]:
    h, w = g.shape
    frames: list[tuple[int, int, int, tuple[int, int], int]] = []
    seen: set[tuple[int, int]] = set()
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            patch = g[r0:r0 + 4, c0:c0 + 4]
            interior_vals = [(ir, ic, int(patch[ir, ic])) for ir, ic in _INTERIOR_CELLS]
            markers = [(ir, ic, v) for ir, ic, v in interior_vals if v != bg]
            if len(markers) != 1:
                continue
            mr, mc, block_col = markers[0]
            if any(v == block_col for lr, lc in _BORDER_ORDER for v in [int(patch[lr, lc])]):
                continue
            border_nonbg = [int(patch[lr, lc]) for lr, lc in _BORDER_ORDER if int(patch[lr, lc]) != bg]
            if len(border_nonbg) < 8:
                continue
            frame_colors = sorted(set(border_nonbg))
            if len(frame_colors) != 1:
                continue
            frame_col = frame_colors[0]
            if frame_col == block_col:
                continue
            if any(int(patch[ir, ic]) == frame_col for ir, ic in _INTERIOR_CELLS):
                continue
            gaps = _frame_gaps_for_frame(g, bg, frame_col, r0, c0)
            if not gaps:
                continue
            key = (r0, c0)
            if key in seen:
                continue
            seen.add(key)
            frames.append((block_col, r0, c0, (mr, mc), frame_col))
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
    Emit beams from framed marker pixels; each side gap and marker quadrant jointly decide the diagonal launch.

    Input structure:
    - The background is the most common color.
    - Each object is a 4x4 frame of one shared wall color with one colored marker pixel inside the 2x2 chamber.
    - Frame openings appear only on side border cells, and the marker's quadrant encodes vertical and horizontal bias.

    Rule:
    - Every side gap emits a beam in the marker color.
    - The gap fixes the outward axis, while the marker fixes the orthogonal axis, so each launch is diagonal.
    - Beams keep extending until they hit the boundary or interact; 90-degree meetings create a blank pivot and bisect, while 45-degree meetings merge into the existing zigzag rule.

    Key insight:
    - A beam direction is no longer read from the gap alone: you must combine two visible cues, the gap side and the marker's offset inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _find_marked_frames(g, bg)
    assert frames, "expected at least one marked frame in input"

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

    for col, r0, c0, marker_local, frame_col in frames:
        gaps = _frame_gaps_for_frame(g, bg, frame_col, r0, c0)
        assert gaps, "expected each frame to have at least one side gap"
        for gr, gc in gaps:
            local_gap = (gr - r0, gc - c0)
            d = _launch_dir_from_gap_and_marker(local_gap, marker_local)
            spawn_beam(col, (gr, gc), (d,))

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


def _draw_marked_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    marker_local: tuple[int, int],
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), bg, dtype=int)
    for lr, lc in _BORDER_ORDER:
        patch[lr, lc] = frame_color
    mr, mc = marker_local
    patch[mr, mc] = block_color
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
    marker_local: tuple[int, int],
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [
                (
                    (r0 + gr, c0 + gc),
                    _launch_dir_from_gap_and_marker((gr, gc), marker_local),
                )
                for gr, gc in combo
            ]
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
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _variant_signal(
    markers: list[tuple[int, int]],
    chosen_gaps: list[tuple[tuple[int, int], ...]],
) -> bool:
    tb_horiz: set[int] = set()
    lr_vert: set[int] = set()
    total_gaps = 0
    changed = 0
    for marker_local, gaps in zip(markers, chosen_gaps):
        for gap in gaps:
            total_gaps += 1
            new_dir = _launch_dir_from_gap_and_marker(gap, marker_local)
            old_dir = _old_gap_dir(gap)
            if new_dir != old_dir:
                changed += 1
            if gap[0] in (0, 3):
                tb_horiz.add(new_dir[1])
            else:
                lr_vert.add(new_dir[0])
    return total_gaps >= 2 and changed == total_gaps and (len(tb_horiz) == 2 or len(lr_vert) == 2)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    markers = [_INTERIOR_CELLS[int(rng.integers(len(_INTERIOR_CELLS)))] for _ in range(block_count)]

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, marker_local in zip(positions, block_colors, markers):
        _draw_marked_frame(obstacles, r0, c0, bg, frame_color, int(color), marker_local, ())

    chosen_gaps = []
    for (r0, c0), marker_local in zip(positions, markers):
        available = _clear_gap_combos(obstacles, bg, r0, c0, marker_local)
        if not available:
            return None
        gap_count = int(rng.choice(sorted(available)))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    if not _variant_signal(markers, chosen_gaps):
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, marker_local, gaps in zip(positions, block_colors, markers, chosen_gaps):
        gap_total += len(gaps)
        _draw_marked_frame(grid, r0, c0, bg, frame_color, int(color), marker_local, gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _launch_dir_from_gap_and_marker(gap, marker_local)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
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
    raise RuntimeError("failed to generate a valid e12f9a14 D10 input")
