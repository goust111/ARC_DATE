# Combo variant of e12f9a14 (index 6)
# combo_spec: D4/0+D5/2
# Name: 单像素源+可变缺口大小

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concretization and behavior trace
#    - Keep the original collision engine and the 4x4 green frame footprint, but
#      replace the 2x2 colored core with a single colored pixel somewhere inside
#      the 2x2 interior. The other three interior cells stay background-colored.
#    - Change the frame opening from "independent gap pixels" to "one contiguous
#      corner opening object" of size 1, 2, or 3 on the 12-cell border cycle.
#    - The opening direction is still the corner it touches: NW / NE / SE / SW.
#      The difference is width: every missing cell in that corner opening now
#      emits one diagonal lane. So width-1 emits one beam from the corner hole,
#      width-2 emits a paired ribbon, and width-3 emits a 3-lane diagonal band.
#    - Example: a NW frame with only (0,0) missing emits one lane from the corner.
#      A NW frame with (0,0),(0,1),(1,0) missing emits three lockstep NW lanes.
#      The source pixel color gives the beam color, while the opening width gives
#      the ribbon thickness.
# b) Complexity calibration
#    - Only three opening sizes are used, which is enough to make the width cue
#      visible without creating a large lookup table.
#    - The source pixel position is allowed to vary among the four interior cells,
#      so the solver cannot rely on a fixed center, but the variation does not
#      encode behavior; it is purely the color carrier.
#    - The opening is always corner-anchored. That keeps the diagonal direction
#      legible and prevents ambiguous side-only missing runs.
# c) Hidden-assumption review
#    - The original solver assumed a detectable 2x2 monochrome block. That no
#      longer exists, so frame discovery must scan for 4x4 windows with exactly
#      one interior non-bg non-frame pixel instead.
#    - The original gap detector treated each border bg cell as an independent
#      emitter. That breaks here because a width-3 opening is one object whose
#      three cells must stay synchronized as a band.
#    - Wide bands can form an L-shaped front (size 3), so lockstep grouping must
#      use same-color 8-neighborhood connectivity rather than only orthogonal
#      adjacency.
#    - The collision engine still assumes all beams move on the same discrete
#      lattice and that only 45-degree / 90-degree interactions occur. Because
#      every emitted lane is still diagonal at birth, those assumptions remain
#      valid under the fused rule.
# d) Placement legality
#    - The generator places non-overlapping 4x4 frames with >=2 cells between
#      frame bounding boxes and a margin from the grid edge.
#    - Every chosen opening is validated against an obstacle grid so each emitted
#      lane has at least two free steps to the boundary.
#    - To keep the wide-band behavior readable and highly reliable, the generator
#      additionally rejects samples where any emitted diagonal lanes would collide
#      or geometrically cross before leaving the grid.
# e) Visual observability
#    - The key visible cue is a direct width mapping: one missing border cell ->
#      one diagonal trace, two contiguous missing cells -> two parallel lanes,
#      three missing cells -> a visibly thicker 3-lane ribbon.
#    - The single source pixel is also visually obvious: compared with the base
#      task, each frame now contains one colored dot rather than a solid 2x2 core.
#    - Samples are forced to include at least one width-3 opening and at least
#      one narrower opening so the contrast is present within each grid.
# f) Visual neatness
#    - Colors remain simple: one shared frame color, unique source colors, and
#      beams inherit their own source color.
#    - Collision-heavy clutter is avoided in generated samples, so the emitted
#      widths remain easy to count by eye.
# g) Generator/solver contract
#    - main() assumes each valid object is a 4x4 frame with exactly one contiguous
#      corner opening of length 1..3 and exactly one colored interior source cell.
#    - generate_input() constructs only such objects and guarantees enough runway
#      for every emitted lane.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_INTERIOR_CELLS = ((1, 1), (1, 2), (2, 1), (2, 2))
_CORNER_TO_DIR = {0: (-1, -1), 3: (-1, 1), 6: (1, 1), 9: (1, -1)}
_CORNER_NEIGHBORS = {0: {11, 1}, 3: {2, 4}, 6: {5, 7}, 9: {8, 10}}


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


def _missing_runs(missing: list[int]) -> list[tuple[int, ...]]:
    if not missing:
        return []
    present = set(missing)
    runs: list[tuple[int, ...]] = []
    n = len(_BORDER_ORDER)
    for idx in sorted(present):
        if (idx - 1) % n in present:
            continue
        run = [idx]
        nxt = (idx + 1) % n
        while nxt in present and nxt != idx:
            run.append(nxt)
            nxt = (nxt + 1) % n
        runs.append(tuple(run))
    return runs


def _opening_info(run: tuple[int, ...]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    """Map a corner-anchored border run to (direction, emitting seed cells)."""
    assert len(run) in (1, 2, 3), "opening width must be 1, 2, or 3 cells"
    corners = [idx for idx in run if idx in _CORNER_TO_DIR]
    assert len(corners) == 1, "opening must contain exactly one corner"
    corner = corners[0]
    arm_idxs = tuple(idx for idx in run if idx != corner)
    assert set(arm_idxs).issubset(_CORNER_NEIGHBORS[corner]), "opening must hug its corner"
    if len(run) == 3:
        assert set(arm_idxs) == _CORNER_NEIGHBORS[corner], "3-cell opening must use both corner arms"
    return _CORNER_TO_DIR[corner], tuple(_BORDER_ORDER[idx] for idx in run)


def _find_frames(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, tuple[int, int], tuple[int, ...]]]:
    """Return (src_color, r0, c0, frame_color, src_local, run) for each valid 4x4 frame."""
    h, w = g.shape
    found: list[tuple[int, int, int, int, tuple[int, int], tuple[int, ...]]] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            border_nonbg: list[int] = []
            missing: list[int] = []
            ok = True
            for idx, (lr, lc) in enumerate(_BORDER_ORDER):
                v = int(g[r0 + lr, c0 + lc])
                if v == bg:
                    missing.append(idx)
                else:
                    border_nonbg.append(v)
            if not border_nonbg:
                continue
            frame_colors = sorted(set(border_nonbg))
            if len(frame_colors) != 1:
                continue
            frame_color = frame_colors[0]

            srcs: list[tuple[int, int, int]] = []
            for lr, lc in _INTERIOR_CELLS:
                v = int(g[r0 + lr, c0 + lc])
                if v == bg:
                    continue
                if v == frame_color:
                    ok = False
                    break
                srcs.append((lr, lc, v))
            if not ok or len(srcs) != 1:
                continue

            runs = _missing_runs(missing)
            if len(runs) != 1:
                continue
            run = runs[0]
            try:
                _opening_info(run)
            except AssertionError:
                continue

            src_lr, src_lc, src_color = srcs[0]
            found.append((src_color, r0, c0, frame_color, (src_lr, src_lc), run))

    # The generator guarantees disjoint frames, so the recognized windows must be disjoint too.
    seen_toplefts = {(r0, c0) for _color, r0, c0, _frame, _src, _run in found}
    assert len(seen_toplefts) == len(found), "expected each frame to be recognized once"
    return found


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


# Emit diagonal ribbons from single-pixel framed sources; opening width controls ribbon width.
# Input structure:
# - The background is the most common color. Each object is a 4x4 frame of one shared wall color.
# - Inside each frame, exactly one of the four interior cells is a colored source pixel; the other
#   three interior cells are background.
# - Each frame has exactly one contiguous corner opening of length 1, 2, or 3 on the border cycle.
#
# Rule:
# - The touched corner fixes the outward diagonal direction.
# - Every missing cell in that opening emits one beam lane in the source color, so opening length and
#   beam width are the same quantity.
# - Once emitted, lanes follow the original collision physics: 90-degree meetings leave a blank
#   intersection and turn toward the bisector; 45-degree meetings merge into the zigzag pattern.
#
# Key insight:
# - The source no longer encodes width by being a 2x2 block. Width is now read entirely from the
#   shape of the frame opening, while the single interior pixel only supplies the beam color.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _find_frames(g, bg)
    assert frames, "expected at least one valid framed source"

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

    for src_color, r0, c0, _frame_color, _src_local, run in frames:
        direction, seeds_local = _opening_info(run)
        for lr, lc in seeds_local:
            spawn_beam(src_color, (r0 + lr, c0 + lc), (direction,))

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a frame or source"
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


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 12, 3: 16, 5: 21}
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
GEN_RETRIES = 500

_OPENING_RUNS = {
    1: (
        (0,),
        (3,),
        (6,),
        (9,),
    ),
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


def _draw_framed_source(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    src_color: int,
    src_local: tuple[int, int],
    gap_run: tuple[int, ...],
) -> None:
    patch = np.full((4, 4), bg, dtype=int)
    for lr, lc in _BORDER_ORDER:
        patch[lr, lc] = frame_color
    for idx in gap_run:
        gr, gc = _BORDER_ORDER[idx]
        patch[gr, gc] = bg
    patch[src_local[0], src_local[1]] = src_color
    paste(grid, patch, r0, c0)


def _opening_seeds(r0: int, c0: int, run: tuple[int, ...]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    direction, seed_cells = _opening_info(run)
    return [((r0 + lr, c0 + lc), direction) for lr, lc in seed_cells]


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


def _seed_paths_noninteracting(
    shape: tuple[int, int],
    seeds: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    h, w = shape
    occupied: dict[tuple[int, int], int] = {}
    segments: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for sid, (start, direction) in enumerate(seeds):
        r, c = start
        dr, dc = direction
        prev = start
        while True:
            if prev in occupied:
                return False
            occupied[prev] = sid
            nxt = (prev[0] + dr, prev[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                break
            cross = ((prev[0], prev[1] + dc), (prev[0] + dr, prev[1]))
            if cross in segments and segments[cross] != sid:
                return False
            segments[(prev, nxt)] = sid
            prev = nxt
    return True


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
    result = 10
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, source_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    src_cells = [_INTERIOR_CELLS[int(rng.integers(len(_INTERIOR_CELLS)))] for _ in range(block_count)]

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, src_local in zip(positions, source_colors, src_cells):
        _draw_framed_source(obstacles, r0, c0, bg, frame_color, int(color), src_local, ())

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

    narrow_candidates = [i for i, available in enumerate(available_per_block) if i != wide_idx and (1 in available or 2 in available)]
    if not narrow_candidates:
        return None
    narrow_idx = narrow_candidates[int(rng.integers(len(narrow_candidates)))]

    chosen_runs: list[tuple[int, ...]] = []
    for i, available in enumerate(available_per_block):
        if i == wide_idx:
            size = 3
        elif i == narrow_idx:
            narrow_sizes = [size for size in (1, 2) if size in available]
            size = narrow_sizes[int(rng.integers(len(narrow_sizes)))]
        else:
            sizes = sorted(available)
            weights = np.array([1.3 if s == 3 else 1.0 for s in sizes], dtype=float)
            weights /= weights.sum()
            size = int(rng.choice(sizes, p=weights))
        runs = available[size]
        chosen_runs.append(runs[int(rng.integers(len(runs)))])

    grid = np.full((n, n), bg, dtype=int)
    seed_total = 0
    all_seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, src_local, run in zip(positions, source_colors, src_cells, chosen_runs):
        _draw_framed_source(grid, r0, c0, bg, frame_color, int(color), src_local, run)
        seeds = _opening_seeds(r0, c0, run)
        seed_total += len(seeds)
        all_seeds.extend(seeds)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, all_seeds):
        return None
    if not _seed_paths_noninteracting(grid.shape, all_seeds):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < seed_total * MIN_BEAM_PIXELS_PER_SEED:
        return None
    widths = {len(run) for run in chosen_runs}
    if 3 not in widths or not ({1, 2} & widths):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
