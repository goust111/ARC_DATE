# Variant of e12f9a14
# Dimension: D18 — 符号/图例编码 (symbol_instructions)
# Change: 无显式图例——所有规则从输入输出对中隐式推断... → 网格角落放置碰撞规则图例
#
# Design analysis
# ---------------
# concretization:
# - The original task already has two visually different collision families:
#   90-degree meetings create a 1-cell hole and then continue along a bisector,
#   while 45-degree meetings merge into a two-lane zigzag.
# - This variant adds a small top-left legend box. The legend is a reserved
#   corner region: beams never enter it. Inside the box, the left column shows
#   the collision type with a tiny input glyph (orthogonal corner for 90°,
#   slash for 45°). The right column shows the post-collision pattern glyph.
# - The 90° row always shows the original "gap then turn" motif, so humans get
#   an explicit key for that branch. The 45° row is the actual control signal:
#   it selects one of two zigzag phases, "diagonal-first" or "cardinal-first".
#   The same beam layout therefore yields different outputs depending on the
#   bottom legend icon.
# - Small example:
#   Original logic: a right-going beam and a down-right diagonal beam meet,
#   the diagonal beam first sidesteps by one cardinal move, and the merged band
#   repeats (diag, card, diag, card, ...).
#   Variant logic: the legend may instead demand the phase (card, diag, ...),
#   so the merged band starts with a straight cardinal step after the meeting.
#
# complexity calibration:
# - Only one binary instruction is introduced; this keeps the search space small
#   enough for ARC while still making the legend materially necessary.
# - The generator forces both a 90° and a 45° collision into every sample, so
#   the legend is not decorative and the input/output pairs repeatedly support
#   the same symbolic mechanism.
# - Colors 4/5/6 are reserved for the legend and excluded from beam colors, so
#   the legend is visually isolated from the block/ray system.
#
# hidden-assumption audit:
# - _find_blocks_2x2 previously scanned the whole grid and would misread any
#   solid 2x2 patch in a legend as a block. The legend therefore uses sparse
#   glyphs plus a 1-pixel outline, never any solid 2x2 non-bg square.
# - The original solver assumed all bg cells are traversable. That no longer
#   holds because the legend contains bg-colored holes. is_free() now also
#   checks that a cell is outside the reserved legend bbox.
# - The original 45° branch hard-coded zig = (diag, card). That assumption is
#   exactly what changes here; the phase now comes from the parsed legend.
# - Frame-color inference still assumes a unique non-bg non-block color around
#   each 2x2 block. The generator keeps all frames well away from the legend so
#   the legend cannot contaminate those local neighborhoods.
#
# placement legality:
# - The generator uses a curated 4-frame layout with global offset jitter.
#   This layout guarantees one orthogonal collision and one 45° collision while
#   keeping frames separated by at least the original 2-cell bbox gap.
# - The legend occupies a reserved 8x9 corner box. Frames are never placed
#   inside or adjacent to that box, and solver traversal also treats it as
#   blocked, so legend pixels cannot be overwritten.
#
# observability / cleanliness:
# - Every sample contains the same object vocabulary as the source task plus a
#   single, tidy corner legend. The beam field stays in the large open area,
#   which avoids cluttering the collision zone with instruction pixels.
# - The controlled branch is visible: the lower legend row changes, and the
#   resulting 45° merged band changes phase in a way that can be compared across
#   training pairs.
#
# new solver assumptions guaranteed by the generator:
# - grid is at least 30x30, so the fixed legend box and the curated frame layout
#   both fit comfortably.
# - there is exactly one legend in the top-left corner.
# - at least one 45° collision occurs with enough runway for the two zigzag
#   phases to produce different outputs.

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

LEGEND_H = 9
LEGEND_W = 9
LEGEND_OUTLINE = 5
LEGEND_INPUT = 4
LEGEND_OUTPUT = 6

_SLOT_90_IN = (1, 1)
_SLOT_90_OUT = (1, 5)
_SLOT_45_IN = (5, 1)
_SLOT_45_OUT = (5, 5)

_ICON_90 = frozenset({(0, 1), (1, 0), (1, 1)})
_ICON_45 = frozenset({(0, 2), (1, 1), (2, 0)})
_ICON_GAP = frozenset({(0, 2), (2, 0), (2, 1)})
_ICON_ZIG_DIAG = frozenset({(0, 0), (1, 1), (1, 2), (2, 2)})
_ICON_ZIG_CARD = frozenset({(0, 0), (0, 1), (1, 1), (2, 2)})

ZIG_DIAG_FIRST = "diag_first"
ZIG_CARD_FIRST = "card_first"


def _slot_cells(anchor: tuple[int, int], icon: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in icon}


_LEGEND_RESERVED = {(r, c) for r in range(LEGEND_H) for c in range(LEGEND_W)}


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


def _legend_bbox_cells(shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    assert h >= LEGEND_H and w >= LEGEND_W, "grid too small for legend"
    return _LEGEND_RESERVED


def _extract_icon(g: np.ndarray, anchor: tuple[int, int], color: int) -> frozenset[tuple[int, int]]:
    ar, ac = anchor
    pts: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if int(g[ar + dr, ac + dc]) == color:
                pts.add((dr, dc))
    return frozenset(pts)


def _parse_corner_legend(g: np.ndarray) -> dict[str, str]:
    assert g.shape[0] >= LEGEND_H and g.shape[1] >= LEGEND_W, "legend box must fit in the corner"
    assert _extract_icon(g, _SLOT_90_IN, LEGEND_INPUT) == _ICON_90, "90-degree legend glyph missing"
    assert _extract_icon(g, _SLOT_45_IN, LEGEND_INPUT) == _ICON_45, "45-degree legend glyph missing"
    assert _extract_icon(g, _SLOT_90_OUT, LEGEND_OUTPUT) == _ICON_GAP, "90-degree output glyph must be gap-turn"
    zig_icon = _extract_icon(g, _SLOT_45_OUT, LEGEND_OUTPUT)
    if zig_icon == _ICON_ZIG_DIAG:
        zig_mode = ZIG_DIAG_FIRST
    elif zig_icon == _ICON_ZIG_CARD:
        zig_mode = ZIG_CARD_FIRST
    else:
        raise AssertionError("45-degree output glyph must encode one of the supported zigzag phases")
    return {"zig_mode": zig_mode}


def _draw_legend(grid: np.ndarray, bg: int, zig_mode: str) -> None:
    grid[:LEGEND_H, :LEGEND_W] = bg
    grid[0, :LEGEND_W] = LEGEND_OUTLINE
    grid[LEGEND_H - 1, :LEGEND_W] = LEGEND_OUTLINE
    grid[:LEGEND_H, 0] = LEGEND_OUTLINE
    grid[:LEGEND_H, LEGEND_W - 1] = LEGEND_OUTLINE
    grid[1 : LEGEND_H - 1, 4] = LEGEND_OUTLINE

    for anchor, icon, color in (
        (_SLOT_90_IN, _ICON_90, LEGEND_INPUT),
        (_SLOT_45_IN, _ICON_45, LEGEND_INPUT),
        (_SLOT_90_OUT, _ICON_GAP, LEGEND_OUTPUT),
        (_SLOT_45_OUT, _ICON_ZIG_DIAG if zig_mode == ZIG_DIAG_FIRST else _ICON_ZIG_CARD, LEGEND_OUTPUT),
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
    legend_cells = _legend_bbox_cells(g.shape)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    stats = {
        "cell_45": 0,
        "cell_90": 0,
        "cross": 0,
        "segcross": 0,
        "zig_mode": legend["zig_mode"],
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

        zig = (v, u) if legend["zig_mode"] == ZIG_DIAG_FIRST else (u, v)
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

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend beams from gapped framed blocks, using a corner legend to decode collision behavior.

    Input structure:
    - The main field matches the source task: a dominant background, several 4x4 frames, and monochrome 2x2 blocks inside those frames.
    - Missing border pixels on a frame are beam exits; side gaps emit cardinal rays and corner gaps emit diagonal rays.
    - The top-left corner also contains a small reserved legend box with one row for 90° meetings and one row for 45° meetings.

    Rule:
    - Emit one beam from every structural frame gap, with the block color and outward direction determined by the gap location.
    - 90° collisions follow the legend's upper row: the meeting cell becomes a permanent hole and both beams continue along the bisector.
    - 45° collisions follow the legend's lower row: beams merge without a hole, but the legend chooses whether the resulting zigzag repeats diagonal-then-cardinal or cardinal-then-diagonal.
    - The legend itself is explanatory only; it remains unchanged in the output and is not traversed by beams.

    Key insight: The geometry of the puzzle is still read from the framed blocks, but the 45° merge phase is no longer hard-coded; the same beam layout can produce different outputs depending on the symbol shown in the corner legend.
    """
    return _solve(grid, want_stats=False)


# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (30, 32)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
GEN_RETRIES = 300

BASE_LAYOUT = [
    ((12, 9), ((1, 3),)),
    ((7, 19), ((3, 1),)),
    ((21, 8), ((1, 3),)),
    ((15, 15), ((3, 3),)),
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


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in {frame_color, LEGEND_INPUT, LEGEND_OUTPUT, LEGEND_OUTLINE}]
    block_colors = list(map(int, rng.choice(palette, size=4, replace=False)))
    return bg, frame_color, block_colors


def _legend_override(grid: np.ndarray, bg: int, zig_mode: str) -> np.ndarray:
    alt = grid.copy()
    _draw_legend(alt, bg, zig_mode)
    return alt


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    dr = int(rng.integers(0, n - 30 + 1))
    dc = int(rng.integers(0, n - 30 + 1))

    bg, frame_color, block_colors = _pick_colors(rng)
    zig_mode = ZIG_DIAG_FIRST if int(rng.integers(2)) == 0 else ZIG_CARD_FIRST

    grid = np.full((n, n), bg, dtype=int)
    _draw_legend(grid, bg, zig_mode)

    for ((base_r, base_c), gaps), color in zip(BASE_LAYOUT, block_colors, strict=False):
        _draw_framed_block(grid, base_r + dr, base_c + dc, bg, frame_color, color, gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, want_stats=True)
    except Exception:
        return None

    if stats["cell_45"] < 1 or stats["cell_90"] < 1:
        return None

    alt_mode = ZIG_CARD_FIRST if zig_mode == ZIG_DIAG_FIRST else ZIG_DIAG_FIRST
    alt_grid = _legend_override(grid, bg, alt_mode)
    try:
        alt_out, alt_stats = _solve(alt_grid, want_stats=True)
    except Exception:
        return None

    if alt_stats["cell_45"] != stats["cell_45"] or alt_stats["cell_90"] != stats["cell_90"]:
        return None
    if np.array_equal(out, alt_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D18 input")
