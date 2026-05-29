# Combo variant of e12f9a14 (index 34)
# combo_spec: D5/3+D12/2
# Name: 开放式框架+输出扩展

import numpy as np

from arc_utils import get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for combo D5/3 + D12/2
# ---------------------------------------------------------------------------
# a) Concrete behavior trace:
#    Base task:
#    - A 2x2 color block sits inside a 4x4 border.
#    - One or more isolated missing border cells emit beams.
#    - Beams interact in-grid, and the final canvas stays the same size.
#    Fused task:
#    - The 4x4 frame is only closed on three sides; one whole side is open.
#    - The two opening cells aligned with the inner 2x2 block act as a parallel
#      launch row, both firing outward in the open-side normal direction.
#    - Example: if the top side is missing, local cells (0,1) and (0,2) are
#      seeds and both shoot north. If they travel 5 visible cells before
#      clipping at the top border, the output is expanded upward and each beam
#      continues 5 more cells outside the original crop.
#    Difference:
#    - D5 changes launch detection from "find isolated holes" to "find the one
#      fully absent side and emit a matched beam pair".
#    - D12 changes the stopping rule from "edge terminates the beam" to "edge
#      reveals only half the final segment, so the canvas grows to show the rest".
#
# b) Complexity calibration:
#    The opening is visually stronger than a single missing pixel, but we do not
#    let all four open-side cells emit because that would create 4-wide bands and
#    add unnecessary partner-tracking ambiguity. Restricting emission to the two
#    middle cells preserves a clear "row of parallel rays" that is still easy to
#    infer from 2-3 examples and keeps the original collision mechanics legible.
#    Output expansion is tied to observed segment length, not to arbitrary side
#    metadata, so the new freedom remains human-readable.
#
# c) Hidden-assumption audit:
#    Original assumptions that no longer hold:
#    - `_frame_gaps_for_block()` assumed isolated structural holes on a closed
#      frame. In the combo, a whole side is background, so we replace it with
#      `_open_side_for_block()` plus `_seed_specs_for_open_side()`.
#    - `out = g.copy()` assumed fixed output shape. The combo must allow beam
#      cells at coordinates outside `[0,h) x [0,w)`, so painting is tracked in a
#      sparse map and rendered onto an expanded canvas afterward.
#    - `in_bounds()` used to mean "beam can continue". Now leaving bounds starts
#      clipped-tail completion instead of terminating the segment.
#    - Collision handlers used to change direction only. With output expansion,
#      they also reset the current segment start so the mirrored outside tail is
#      measured from the most recent launch/turn point.
#    New combo assumptions:
#    - Each block has exactly one fully open side and the other eight border
#      cells are intact frame pixels.
#    - The 4x4 footprint stays fully inside the input, so open-side detection is
#      unambiguous.
#    - Outside continuations never overlap each other; generator validation uses
#      the solver itself to reject violating samples.
#
# d) Placement legality:
#    The generator keeps the full 4x4 frame footprint inside the grid, enforces a
#    gap between frame bounding boxes, and checks that every initial launch row
#    has at least two in-grid background cells of runway. It then runs the fused
#    solver and rejects any sample where expansion is absent or outside tails
#    collide.
#
# e) Visual observability:
#    A human can see three persistent cues across examples:
#    - the frame is open on exactly one side,
#    - two parallel rays originate from the middle of that opening,
#    - the output is larger whenever those rays reach the crop boundary.
#    No color ordering or numeric color identity is used; colors only label which
#    inner block owns which beams.
#
# f) Visual cleanliness:
#    Bounding-box spacing prevents open sides from visually merging. The solver
#    rejects outside overlaps, so the expanded border region stays readable
#    instead of becoming a pile-up of different colors.
#
# g) Change inferability:
#    Colors are unchanged from the base task: each beam inherits its block color.
#    The fused novelty is entirely structural and geometric, which is easier to
#    infer from few-shot ARC examples than introducing a new recoloring scheme.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
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
    """Find the unique non-bg color on the surrounding 4x4 border."""
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
    assert len(frame_colors) == 1, "expected a unique frame color around each block"
    return frame_colors[0]


def _open_side_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> tuple[int, int]:
    """Detect which whole side of the surrounding 4x4 frame is open."""
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
    assert len(matches) == 1, "expected exactly one fully open frame side per block"
    return matches[0]


def _seed_specs_for_open_side(
    br: int, bc: int, open_dir: tuple[int, int]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "segment_start")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.segment_start = 0

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)

    def segment_cells(self) -> int:
        return len(self.path) - self.segment_start


def _solve_variant(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    blocked: set[tuple[int, int]] = set()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    open_dirs: list[tuple[int, int]] = []

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def grid_value(cell: tuple[int, int]) -> int:
        if in_bounds(cell):
            return int(g[cell[0], cell[1]])
        return bg

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        assert grid_value(cell) == bg, "expected beam to paint only background"
        assert cell not in blocked, "expected beam not to paint a blocked cell"
        prev = cell_owner.get(cell)
        assert prev is None or prev == bid, "expected painted cells to have a unique owner"
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
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
        if b.segment_start >= keep_len:
            b.segment_start = keep_len - 1
        b.pos = b.path[-1]
        b.active = True

    def reset_segment(bid: int) -> None:
        beams[bid].segment_start = len(beams[bid].path) - 1

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert grid_value(pos) == bg, "expected beam seed on the open frame side"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        open_dir = _open_side_for_block(g, bg, frame_col, br, bc)
        open_dirs.append(open_dir)
        for pos, d in _seed_specs_for_open_side(br, bc, open_dir):
            spawn_beam(col, pos, (d,))

    assert beams, "expected at least one beam spawned from open frame sides"

    def is_free(cell: tuple[int, int]) -> bool:
        return grid_value(cell) == bg and cell not in blocked

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

    def finish_clipped_segment(bid: int) -> None:
        b = beams[bid]
        steps = b.segment_cells()
        assert steps >= 1, "expected a visible segment before outside completion"
        for _ in range(steps):
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            assert not in_bounds(nxt), "expected mirrored tail to stay outside the input crop"
            assert is_free(nxt), "expected outside continuation to remain on background"
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
        b.active = False

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
                truncate_beam(b, cell_index[cell])
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
                reset_segment(gid)
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
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                truncate_beam(bid_u, cell_index[cell] + 1)
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
        reset_segment(bid_u)
        reset_segment(bid_v)

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
        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        beams[a].pattern = (nd,)
        beams[a].phase = 0
        beams[b].pattern = (nd,)
        beams[b].phase = 0
        beams[a].active = True
        beams[b].active = True
        reset_segment(a)
        reset_segment(b)

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

        nd = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert nd != (0, 0), "expected segment crossing to resolve to a forward direction"

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
            reset_segment(gid)

    max_iters = (h + w + len(beams) + 4) * (h * w + 20)
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        exiting: list[int] = []
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt):
                exiting.append(bid)
                continue
            if not is_free(nxt):
                assert nxt in blocked, "beam must not run into a non-background obstacle"
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop and exiting:
            for bid in sorted(exiting):
                if beams[bid].active:
                    finish_clipped_segment(bid)
            continue

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

        for bid in sorted(exiting):
            if beams[bid].active:
                finish_clipped_segment(bid)
    else:
        raise AssertionError("beam simulation did not converge")

    painted_cells = list(cell_owner)
    min_r = min([0] + [r for r, _c in painted_cells])
    max_r = max([h - 1] + [r for r, _c in painted_cells])
    min_c = min([0] + [c for _r, c in painted_cells])
    max_c = max([w - 1] + [c for _r, c in painted_cells])

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    out[-min_r : -min_r + h, -min_c : -min_c + w] = g
    for cell, bid in cell_owner.items():
        out[cell[0] - min_r, cell[1] - min_c] = beams[bid].color

    stats = {
        "pad_top": max(0, -min_r),
        "pad_bottom": max(0, max_r - (h - 1)),
        "pad_left": max(0, -min_c),
        "pad_right": max(0, max_c - (w - 1)),
        "outside_pixels": sum(1 for r, c in painted_cells if not (0 <= r < h and 0 <= c < w)),
        "beam_pixels": len(painted_cells),
        "beam_count": len(beams),
        "block_count": len(blocks),
        "seed_count": len(beams),
        "open_dirs": tuple(open_dirs),
        "blocked_count": len(blocked),
        "output_shape": out.shape,
    }
    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch paired rays from half-open frames, then enlarge the canvas to reveal the full emitted segments.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block inside a 4x4 frame footprint.
    - Exactly one frame side is completely open, while the other three sides remain in the shared frame color.
    - The two open-side cells aligned with the inner 2x2 block are blank launch pads.

    Rule:
    - Detect the open side of each frame and fire both launch pads outward in parallel, using the block's color.
    - Inside the original crop, beam motion and beam-beam interactions follow the original task's collision rules:
      90-degree meetings create a blank gap and redirect both beams to the bisector, while 45-degree cardinal/diagonal
      meetings merge into the same zigzag continuation.
    - When a current segment reaches the input boundary, do not truncate the final result there. Expand the output grid
      and continue that same segment outside the crop for exactly as many cells as were already visible in that segment.

    Key insight:
    - The main cue is the missing whole side, not an isolated hole. Solve by reading frame orientation first, emitting a
      matched 2-ray strip from the opening, and then inferring the larger output size from how far each visible segment
      had already traveled before it hit the crop boundary.
    """
    return _solve_variant(grid, return_stats=False)


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
MIN_OUTSIDE_PIXELS = 4
MIN_TOTAL_PADDING = 2
GEN_RETRIES = 500


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    margin = PLACEMENT_MARGIN
    all_positions = [(r, c) for r in range(margin, n - 3 - margin) for c in range(margin, n - 3 - margin)]
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


def _available_open_dirs(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> list[tuple[int, int]]:
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
        _out, stats = _solve_variant(grid, return_stats=True)
    except Exception:
        return None

    total_padding = stats["pad_top"] + stats["pad_bottom"] + stats["pad_left"] + stats["pad_right"]
    if stats["seed_count"] != 2 * block_count:
        return None
    if stats["beam_pixels"] < len(seeds) * MIN_BEAM_PIXELS_PER_SEED:
        return None
    if stats["outside_pixels"] < MIN_OUTSIDE_PIXELS:
        return None
    if total_padding < MIN_TOTAL_PADDING:
        return None
    if stats["output_shape"] == grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo D5/3+D12/2 input")
