# Variant of e12f9a14
# Dimension: D15 — 背景色与网格环境 (meta)
# Change: 背景色为天蓝色(8)或黑色(0)，网格中除框架外无其他前景对象... → 存在干扰项框架

"""
design_analysis:
  concrete_behavior:
    active_source:
      shape: 4x4 frame of one wall color around a centered 2x2 monochrome block
      openings: 1-4 structural border holes; each hole emits exactly as in the original task
      output: only these frames create colored beam traces
    distractors:
      sealed_filled:
        shape: same 4x4 frame and same 2x2 colored interior, but the border is fully closed
        role: looks source-like, but there is no exit so it remains unchanged
      hollow_open:
        shape: same 4x4 frame, but the 2x2 interior is background; the frame may still have 1-4 holes
        role: looks even more deceptive because holes exist, yet no beam starts because there is no colored core
      shattered_filled:
        shape: same 4x4 colored-core frame, but with 5-6 valid holes so the border is obviously too broken
        role: this keeps the object frame-like while making the variant non-degenerate, because the old solver
          would still fire from the surviving holes whereas the new solver rejects over-open frames
    small_trace:
      old_logic: scan every 2x2 monochrome block and emit from every structural gap around it
      new_logic: scan every 4x4 frame candidate first, classify it by interior+gap pattern, then emit only from
        frames that are filled and moderately open
      difference_point: closed colored frames, hollow frames, and over-open colored frames are all explicit
        distractors now instead of being silently conflated with the original launcher pattern
  complexity_check:
    - The variant adds only three distractor archetypes, and all three reuse the same 4x4 frame template.
    - At least one distractor is generated in every sample, and at least one of them is over-open so the new
      filtering rule affects actual solver behavior rather than only the input distribution.
    - Beam physics, collision rules, color palette, and frame size remain unchanged, so the task does not drift
      into a different ARC mechanism.
  implicit_assumptions_review:
    - The original _find_blocks_2x2 path assumes every relevant object is discoverable from a colored 2x2 core.
      That is no longer sufficient because the input now intentionally contains empty frames.
    - The original source-building loop assumes "block found" implies "candidate source". In this variant a
      filled frame can still be inert when it is fully sealed or too broken, so the solver must check frame
      classification before spawning beams.
    - Collision code assumes every non-background cell in the input is an obstacle. This still holds, so beams
      must be validated against source frames and distractor frames alike.
  placement_legality:
    - Every object still occupies one 4x4 bounding box.
    - Bounding boxes keep a minimum empty moat, so frames stay visually separable and frame-scan matching remains
      unambiguous even when some distractors carry many holes.
    - Ray-clearance checks treat every object as a full obstacle frame, including hollow-open distractors, to
      prevent beams from visually tunneling through distractor holes.
  visual_observability:
    - The difference is visible without using hidden color ids: some frames with colored cores remain inert
      because they are closed or too broken, and some frames with visible holes remain inert because they are hollow.
    - Across 2-4 examples a human can infer that "not every frame is a launcher"; the launcher must be both
      colored inside and actually open.
  cleanliness:
    - Total object count is capped, and larger grids are used when more frames appear.
    - Distractors never overlap emitted paths at generation time, so output remains legible.
  new_solver_assumptions_to_guarantee:
    - Every deliberate frame uses only wall or background on its border.
    - Every border background cell is a true structural gap, not an accidental broken edge.
    - At least one classified frame is a true source; generator validation enforces this.
"""

import itertools

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


class _FrameCandidate:
    __slots__ = ("r0", "c0", "wall", "block_color", "gaps")

    def __init__(
        self,
        r0: int,
        c0: int,
        wall: int,
        block_color: int | None,
        gaps: tuple[tuple[int, int], ...],
    ) -> None:
        self.r0 = r0
        self.c0 = c0
        self.wall = int(wall)
        self.block_color = None if block_color is None else int(block_color)
        self.gaps = gaps

    @property
    def is_source(self) -> bool:
        return self.block_color is not None and 1 <= len(self.gaps) <= 4


def _frame_gaps_for_frame(g: np.ndarray, bg: int, wall: int, r0: int, c0: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = r0, c0
    fr1, fc1 = r0 + 3, c0 + 3
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
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
            ok = int(g[r - 1, c]) == wall and int(g[r + 1, c]) == wall
        elif dc == 0:
            ok = int(g[r, c - 1]) == wall and int(g[r, c + 1]) == wall
        else:
            ok = int(g[r - dr, c]) == wall and int(g[r, c - dc]) == wall
        if ok:
            out.append((r, c))
    out_set = set(out)
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, c0 + 2) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _classify_frames(g: np.ndarray, bg: int) -> list[_FrameCandidate]:
    h, w = g.shape
    frames: list[_FrameCandidate] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            border_vals: list[int] = []
            border_bg: list[tuple[int, int]] = []
            for dr, dc in _BORDER_ORDER:
                v = int(g[r0 + dr, c0 + dc])
                border_vals.append(v)
                if v == bg:
                    border_bg.append((r0 + dr, c0 + dc))
            wall_colors = sorted({v for v in border_vals if v != bg})
            if len(wall_colors) != 1:
                continue
            wall = wall_colors[0]
            wall_count = sum(v == wall for v in border_vals)
            if not (6 <= wall_count <= 12):
                continue

            inner = g[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
            inner_vals = {int(v) for v in inner.ravel()}
            block_color: int | None
            if inner_vals == {bg}:
                block_color = None
            elif len(inner_vals) == 1:
                only = next(iter(inner_vals))
                if only in (bg, wall):
                    continue
                block_color = only
            else:
                continue

            gaps = tuple(_frame_gaps_for_frame(g, bg, wall, r0, c0))
            if set(border_bg) != set(gaps):
                continue
            frames.append(_FrameCandidate(r0, c0, wall, block_color, gaps))
    return frames


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
    Emit beam trails only from genuine framed launchers while ignoring frame-shaped distractors.

    Input structure:
    - The background is still the most common color, and every visible object is a 4x4 frame built from one wall color.
    - Some frames contain a centered 2x2 monochrome block, some are hollow, and some borders have holes while others are sealed.

    Rule:
    - A frame launches beams only when it has both a colored 2x2 core and a moderate opening count: 1-4 structural border holes.
    - Closed colored frames, hollow frames, and over-open colored frames are distractors: they stay unchanged even if they resemble launchers.
    - Once launched, beams follow the original motion system: they extend through background, stop at obstacles/edges, and
      resolve 90-degree and 45-degree collisions with the same turn and zigzag rules as before.

    Key insight:
    - The new work is not in beam physics but in object filtering: identify which frame candidates are actual sources before
      simulating collisions, because visually similar decoy frames now coexist in the same grid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _classify_frames(g, bg)
    sources = [fr for fr in frames if fr.is_source]
    assert sources, "expected at least one true source frame in input"

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

    for fr in sources:
        assert fr.block_color is not None
        for gr, gc in fr.gaps:
            dr = -1 if gr == fr.r0 else (1 if gr == fr.r0 + 3 else 0)
            dc = -1 if gc == fr.c0 else (1 if gc == fr.c0 + 3 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each source gap to face outward"
            spawn_beam(fr.block_color, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from source frames"

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide on-cell"
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
        assert u != (0, 0), "expected crossing diagonals to resolve forward"
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
        assert u != (0, 0), "expected segment crossing to resolve forward"

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a frame or colored core"
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


# --------------- generator (o5x) ---------------

SOURCE_COUNT_RANGE = (1, 6)
DISTRACTOR_COUNT_RANGE = (1, 4)
MAX_TOTAL_FRAMES = 6
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_FRAMES = {1: 8, 2: 14, 4: 20, 6: 24}
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

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


def _gap_combos(size: int) -> list[tuple[tuple[int, int], ...]]:
    combos = []
    for combo in itertools.combinations(_BORDER_ORDER, size):
        idxs = sorted(_BORDER_INDEX[p] for p in combo)
        if any((idxs[(i + 1) % size] - idxs[i]) % 12 == 1 for i in range(size)):
            continue
        if size >= 3 and (3, 3) in combo and (0, 2) in combo:
            continue
        combos.append(combo)
    return combos


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 7)}


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


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


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int | None,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = bg if block_color is None else int(block_color)
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _rays_clear(grid: np.ndarray, bg: int, seeds: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
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


def _clear_gap_combos(obstacles: np.ndarray, bg: int, r0: int, c0: int) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(
    rng: np.random.Generator,
    source_count: int,
    filled_distractor_count: int,
) -> tuple[int, int, list[int], list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    filled_count = source_count + filled_distractor_count
    colors = list(rng.choice(palette, size=filled_count, replace=False))
    return bg, frame_color, colors[:source_count], colors[source_count:]


def _grid_size_min(frame_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_FRAMES.items()):
        if frame_count >= threshold:
            result = min_n
    return result


def _choose_distractor_types(rng: np.random.Generator, distractor_count: int) -> list[str]:
    types = ["shattered_filled"]
    if distractor_count >= 2:
        types.append("hollow_open")
    for _ in range(distractor_count - len(types)):
        types.append(str(rng.choice(["sealed_filled", "hollow_open", "shattered_filled"], p=[0.25, 0.4, 0.35])))
    rng.shuffle(types)
    return types


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    source_count = int(rng.integers(*SOURCE_COUNT_RANGE))
    max_distractors = min(DISTRACTOR_COUNT_RANGE[1] - 1, MAX_TOTAL_FRAMES - source_count)
    if max_distractors < DISTRACTOR_COUNT_RANGE[0]:
        return None
    distractor_count = int(rng.integers(DISTRACTOR_COUNT_RANGE[0], max_distractors + 1))
    distractor_types = _choose_distractor_types(rng, distractor_count)
    total_frames = source_count + distractor_count

    n = int(rng.integers(_grid_size_min(total_frames), GRID_SIZE_MAX))
    positions = _choose_positions(n, total_frames, rng)
    if positions is None:
        return None

    filled_distractor_count = sum(t in ("sealed_filled", "shattered_filled") for t in distractor_types)
    bg, frame_color, source_colors, distractor_colors = _pick_colors(rng, source_count, filled_distractor_count)

    objects: list[dict[str, object]] = [{"kind": "source"} for _ in range(source_count)]
    objects.extend({"kind": t} for t in distractor_types)
    rng.shuffle(objects)
    for obj, pos in zip(objects, positions):
        obj["pos"] = pos

    src_iter = iter(source_colors)
    filled_iter = iter(distractor_colors)
    for obj in objects:
        kind = obj["kind"]
        if kind == "source":
            obj["block_color"] = int(next(src_iter))
        elif kind in ("sealed_filled", "shattered_filled"):
            obj["block_color"] = int(next(filled_iter))
        else:
            obj["block_color"] = None

    obstacles = np.full((n, n), bg, dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        _draw_frame(obstacles, r0, c0, bg, frame_color, obj["block_color"], ())

    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for obj in objects:
        r0, c0 = obj["pos"]
        kind = obj["kind"]
        if kind == "source":
            available = _clear_gap_combos(obstacles, bg, r0, c0)
            small_counts = [count for count in sorted(available) if 1 <= count <= 4]
            if not small_counts:
                return None
            gap_count = int(rng.choice(small_counts))
            gaps = available[gap_count][int(rng.integers(len(available[gap_count])))]
            obj["gaps"] = gaps
            gap_total += len(gaps)
            for gap in gaps:
                seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))
        elif kind == "hollow_open":
            gap_count = int(rng.integers(1, 5))
            combos = _GAP_COMBOS[gap_count]
            obj["gaps"] = combos[int(rng.integers(len(combos)))]
        elif kind == "shattered_filled":
            available = _clear_gap_combos(obstacles, bg, r0, c0)
            big_counts = [count for count in sorted(available) if count >= 5]
            if not big_counts:
                return None
            gap_count = int(rng.choice(big_counts))
            gaps = available[gap_count][int(rng.integers(len(available[gap_count])))]
            obj["gaps"] = gaps
        else:
            obj["gaps"] = ()

    grid = np.full((n, n), bg, dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        _draw_frame(grid, r0, c0, bg, frame_color, obj["block_color"], obj["gaps"])

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    frames = _classify_frames(grid, bg)
    source_frames = [fr for fr in frames if fr.is_source]
    hollow_open = [fr for fr in frames if fr.block_color is None and len(fr.gaps) > 0]
    sealed_filled = [fr for fr in frames if fr.block_color is not None and len(fr.gaps) == 0]
    shattered_filled = [fr for fr in frames if fr.block_color is not None and len(fr.gaps) >= 5]
    if len(frames) != total_frames:
        return None
    if len(source_frames) != source_count:
        return None
    if len(shattered_filled) < 1:
        return None
    if len(hollow_open) != sum(t == "hollow_open" for t in distractor_types):
        return None
    if len(sealed_filled) != sum(t == "sealed_filled" for t in distractor_types):
        return None
    if len(shattered_filled) != sum(t == "shattered_filled" for t in distractor_types):
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
    raise RuntimeError("failed to generate a valid e12f9a14 D15 input")
