# Combo variant of e12f9a14 (index 16)
# combo_spec: D17/2+D18/0
# Name: 碰撞规则图例+三种碰撞角度
#
# Design analysis
# ---------------
# concretization:
# - The source task already has framed 2x2 color blocks that emit beams from
#   frame gaps, plus two collision families: 90-degree corner meetings and
#   45-degree diagonal/cardinal meetings.
# - This combo adds an explicit corner legend and upgrades the collision system
#   from two branches to three. The legend sits in the top-left corner, outside
#   the beam field, with three rows:
#   90° input glyph -> "leave a hole, then turn to the bisector",
#   45° input glyph -> "merge into a zigzag",
#   180° input glyph -> "head-on stop".
# - Concrete example:
#   Original logic: an east-going beam that reaches a west-going beam has no
#   dedicated rule; the old solver would fall outside its supported angle set.
#   Combo logic: the bottom legend row shows a head-on pair and an output icon
#   with two separated endpoints, so the same meeting now means both beams stop
#   before occupying the same continuation path.
# - The generator uses one clean five-block layout. One pair realizes a 90°
#   collision, one realizes a 45° collision, and one horizontal pair realizes a
#   180° collision. One block supplies two non-adjacent gaps so the full combo
#   fits within five unique block colors.
#
# complexity calibration:
# - The legend encodes exactly three collision classes. That is enough to create
#   a genuine new branch structure without introducing many free parameters.
# - Every sample contains all three collision types, so each legend row has a
#   visible counterpart in the output. This prevents the legend from becoming
#   decorative or the 180° rule from appearing only in rare cases.
# - The legend uses dedicated colors 4/5/6 while the generator restricts block
#   colors to five other colors. That keeps instruction pixels visually separate
#   from beam sources and avoids accidental color-based ambiguity.
#
# hidden-assumption audit:
# - _find_blocks_2x2 scans the whole grid, so the legend must not contain any
#   solid 2x2 non-bg square. All legend icons are sparse 3x3 motifs for that
#   reason.
# - The original solver treated every background cell as traversable. With a
#   corner legend, that assumption breaks; is_free() must also reject reserved
#   legend cells so beams do not overwrite instruction pixels.
# - The original collision code only supported 45° and 90° meetings. The combo
#   adds diff==4 handling plus swap-style head-on detection for adjacent beams
#   moving directly into each other.
# - The original generator could use bg 8 and up to six distinct colors. This
#   combo reserves 4/5/6 for the legend, so the generator fixes bg=0 and frame=8
#   to keep five unique non-legend block colors available.
# - The 45° rule remains the original zigzag phase. The legend still has to be
#   parsed, but now its role is to expose the full three-rule table instead of
#   selecting among multiple 45° submodes.
#
# placement legality:
# - The legend occupies a fixed 13x9 top-left box; all frames are placed outside
#   that box and then globally jittered together, so they never overlap it.
# - Frame bounding boxes stay separated by at least the original 2-cell margin.
# - The curated layout is validated by running the solver. Samples are accepted
#   only when the oracle records exactly one 45°, one 90°, and one 180° event.
#
# observability / cleanliness:
# - Human solvers can see the same beam vocabulary as in the source task, but
#   the legend now makes the collision table explicit. The three rows are easy
#   to compare because they share the same left/right slot structure.
# - The 180° branch is visually obvious in output: two approaching beams terminate
#   around a blank meeting point instead of merging, crossing, or turning.
# - The collision zones are spatially separated: the 90° event is upper-middle,
#   the 45° event is lower-right, and the 180° event is lower-middle. This keeps
#   the output readable despite the extra rule family.
#
# new solver assumptions guaranteed by the generator:
# - There is exactly one legend, always in the top-left corner, and it always
#   uses the fixed three-row glyph inventory defined below.
# - The beam field contains exactly five framed blocks with unique colors.
# - The layout is jittered but not reconfigured, so the intended three collision
#   events always occur and do not get masked by unrelated secondary collisions.

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

LEGEND_H = 13
LEGEND_W = 9
LEGEND_OUTLINE = 5
LEGEND_INPUT = 4
LEGEND_OUTPUT = 6

_SLOT_90_IN = (1, 1)
_SLOT_90_OUT = (1, 5)
_SLOT_45_IN = (5, 1)
_SLOT_45_OUT = (5, 5)
_SLOT_180_IN = (9, 1)
_SLOT_180_OUT = (9, 5)

_ICON_90 = frozenset({(0, 1), (1, 0), (1, 1)})
_ICON_45 = frozenset({(0, 2), (1, 1), (2, 0)})
_ICON_180 = frozenset({(1, 0), (1, 1), (1, 2)})

_ICON_GAP_TURN = frozenset({(0, 2), (2, 0), (2, 1)})
_ICON_ZIG = frozenset({(0, 0), (1, 1), (1, 2), (2, 2)})
_ICON_STOP = frozenset({(1, 0), (1, 2)})

_LEGEND_RESERVED = {(r, c) for r in range(LEGEND_H) for c in range(LEGEND_W)}


def _slot_cells(anchor: tuple[int, int], icon: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in icon}


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


def _extract_icon(g: np.ndarray, anchor: tuple[int, int], color: int) -> frozenset[tuple[int, int]]:
    ar, ac = anchor
    pts: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if int(g[ar + dr, ac + dc]) == color:
                pts.add((dr, dc))
    return frozenset(pts)


def _parse_corner_legend(g: np.ndarray) -> dict[str, bool]:
    assert g.shape[0] >= LEGEND_H and g.shape[1] >= LEGEND_W, "legend box must fit in the corner"
    assert _extract_icon(g, _SLOT_90_IN, LEGEND_INPUT) == _ICON_90, "90-degree legend glyph missing"
    assert _extract_icon(g, _SLOT_45_IN, LEGEND_INPUT) == _ICON_45, "45-degree legend glyph missing"
    assert _extract_icon(g, _SLOT_180_IN, LEGEND_INPUT) == _ICON_180, "180-degree legend glyph missing"
    assert _extract_icon(g, _SLOT_90_OUT, LEGEND_OUTPUT) == _ICON_GAP_TURN, "90-degree output glyph missing"
    assert _extract_icon(g, _SLOT_45_OUT, LEGEND_OUTPUT) == _ICON_ZIG, "45-degree output glyph missing"
    assert _extract_icon(g, _SLOT_180_OUT, LEGEND_OUTPUT) == _ICON_STOP, "180-degree output glyph missing"
    return {"has_90": True, "has_45": True, "has_180": True}


def _draw_legend(grid: np.ndarray, bg: int) -> None:
    grid[:LEGEND_H, :LEGEND_W] = bg
    grid[0, :LEGEND_W] = LEGEND_OUTLINE
    grid[LEGEND_H - 1, :LEGEND_W] = LEGEND_OUTLINE
    grid[:LEGEND_H, 0] = LEGEND_OUTLINE
    grid[:LEGEND_H, LEGEND_W - 1] = LEGEND_OUTLINE
    grid[1 : LEGEND_H - 1, 4] = LEGEND_OUTLINE
    grid[4, 1 : LEGEND_W - 1] = LEGEND_OUTLINE
    grid[8, 1 : LEGEND_W - 1] = LEGEND_OUTLINE

    for anchor, icon, color in (
        (_SLOT_90_IN, _ICON_90, LEGEND_INPUT),
        (_SLOT_45_IN, _ICON_45, LEGEND_INPUT),
        (_SLOT_180_IN, _ICON_180, LEGEND_INPUT),
        (_SLOT_90_OUT, _ICON_GAP_TURN, LEGEND_OUTPUT),
        (_SLOT_45_OUT, _ICON_ZIG, LEGEND_OUTPUT),
        (_SLOT_180_OUT, _ICON_STOP, LEGEND_OUTPUT),
    ):
        for r, c in _slot_cells(anchor, icon):
            grid[r, c] = color


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


def _solve(grid: np.ndarray, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend = _parse_corner_legend(g)
    assert legend["has_90"] and legend["has_45"] and legend["has_180"]

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    legend_cells = _LEGEND_RESERVED

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    stats = {
        "cell_45": 0,
        "cell_90": 0,
        "head_on_180_cell": 0,
        "head_on_180_swap": 0,
        "cross": 0,
        "segcross": 0,
    }

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
        assert pos not in legend_cells, "frame gap must not overlap legend"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked and cell not in legend_cells

    def stop_beams(ids: set[int]) -> None:
        for bid in ids:
            beams[bid].active = False

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
        assert diff in (1, 2, 4), "expected 45-degree, 90-degree, or 180-degree collisions"

        if diff == 4:
            stats["head_on_180_cell"] += 1
            if b_owns_cell:
                stop_beams({a, b})
            else:
                stop_beams({a, b})
                unpaint_cell(cell)
            return

        if diff == 2:
            stats["cell_90"] += 1
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

        stats["cell_45"] += 1
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
        stats["cross"] += 1
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
        stats["segcross"] += 1
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

    def resolve_swap_head_on(a: int, b: int) -> None:
        stats["head_on_180_swap"] += 1
        stop_beams({a, b})

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
                assert not in_bounds(nxt) or nxt in blocked or nxt in legend_cells, "beam must not hit frame pixels"
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

        swap_pairs: set[tuple[int, int]] = set()
        for a, (nxt_a, da) in prop.items():
            oa = cell_owner.get(nxt_a)
            if oa is None or oa == a or oa not in prop:
                continue
            nxt_b, db = prop[oa]
            if nxt_b != beams[a].pos:
                continue
            if _dir_diff(da, db) != 4:
                continue
            swap_pairs.add(tuple(sorted((a, oa))))

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int, int, str]] = []
        for a, b in swap_pairs:
            r = min(beams[a].pos[0], beams[b].pos[0])
            c = min(beams[a].pos[1], beams[b].pos[1])
            events.append((-1, r, c, a, b, "swap"))
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
            if kind == "swap":
                resolve_swap_head_on(a, b)
            elif kind == "cell":
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

    stats["head_on_180"] = stats["head_on_180_cell"] + stats["head_on_180_swap"]
    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend beams from framed color blocks, using a corner legend that spells out three collision rules.

    Input structure:
    - The main field contains the original framed 2x2 monochrome blocks on a dominant background, with each missing frame pixel acting as a beam exit.
    - Side gaps launch cardinal beams and corner gaps launch diagonal beams, using the inner block color.
    - The top-left corner is a reserved legend box with three rows, one each for 90°, 45°, and 180° collisions.

    Rule:
    - Emit one beam from every structural frame gap and keep painting background cells until the beam hits the boundary, a blocked collision hole, or the legend box.
    - At a 90° meeting, remove the meeting cell and redirect both beams along the bisector shown in the legend.
    - At a 45° meeting, merge the beams into the zigzag pattern shown in the legend, with no gap at the collision point.
    - At a 180° head-on meeting, stop both beams as indicated by the legend's bottom row; the beams do not continue through each other.

    Key insight: The beam geometry still comes from the gapped frames, but the puzzle now has an explicit three-case collision table. Solving requires matching each observed meeting angle to the correct legend row before extending the beams.
    """
    return _solve(grid, want_stats=False)


# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (33, 35)
BG = 0
FRAME_COLOR = 8
BLOCK_COLORS = [1, 2, 3, 7, 9]
GEN_RETRIES = 200

BASE_LAYOUT = [
    ((13, 10), ((1, 3),)),
    ((8, 20), ((3, 1),)),
    ((22, 16), ((1, 3), (2, 0))),
    ((16, 16), ((3, 3),)),
    ((22, 9), ((2, 3),)),
]


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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    dr = int(rng.integers(0, n - 26))
    dc = int(rng.integers(0, n - 24))

    colors = list(rng.permutation(BLOCK_COLORS))
    grid = np.full((n, n), BG, dtype=int)
    _draw_legend(grid, BG)

    for ((base_r, base_c), gaps), color in zip(BASE_LAYOUT, colors, strict=False):
        _draw_framed_block(grid, base_r + dr, base_c + dc, BG, FRAME_COLOR, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != BG:
        return None

    try:
        out, stats = _solve(grid, want_stats=True)
    except Exception:
        return None

    if stats["cell_45"] != 1 or stats["cell_90"] != 1 or stats["head_on_180"] != 1:
        return None
    if stats["cross"] != 0 or stats["segcross"] != 0:
        return None
    if np.count_nonzero(out != grid) < 25:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
