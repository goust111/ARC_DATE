# Combo variant of e12f9a14 (index 24)
# combo_spec: D7/0+D13/0
# Name: 射线重着色框架+多色框架

import itertools

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


# Design analysis for combo D7/0 + D13/0
# analysis:
#   concretization_and_trace:
#     - Base task objects stay the same: each source is still a monochrome 2x2 block inside a 4x4 frame with
#       1-4 border gaps, and each gap still launches one outward beam whose later collisions follow the original
#       geometry grammar.
#     - D7 changes the input encoding from "one global frame color" to "each frame has its own border color".
#       Example: one frame may be red and another yellow even before any beam moves.
#     - D13 changes launch-time rendering. If a frame with block color blue has a top gap, the output paints the
#       gap cell blue as before, and also repaints the rest of that top frame side blue. If the gap is a corner,
#       both touched sides are repainted blue.
#     - Combined behavior on a concrete pair: suppose frame A has red border and blue block, frame B has yellow
#       border and green block. In the output, A keeps some red pixels but the side(s) touched by blue rays become
#       blue; B analogously keeps some yellow while touched side(s) become green. Downstream beam collisions are
#       unchanged from the source task. The visible difference from the original puzzle is therefore local frame
#       color diversity plus local overwriting by the emitting beam color.
#   complexity_calibration:
#     - The fusion adds exactly one new inference burden beyond D13: the solver can no longer assume one global
#       wall color and must infer each frame's border color locally around its own 2x2 block.
#     - To keep the puzzle readable, the generator always creates at least two frames, but caps the count at four
#       so frame colors and block colors can remain pairwise distinct in a 10-color palette without hidden reuse.
#     - To avoid visually collapsing a frame into a single color, the generator rejects gap sets that would let
#       launch recoloring cover the entire remaining border. This keeps both the original frame hue and the new
#       beam-colored overwrite simultaneously visible.
#   hidden_assumptions_review:
#     - `_infer_frame_color()` is still valid because each frame border is locally monochrome in the input even
#       though different frames now use different colors globally.
#     - `is_free()` must continue checking the immutable input `g`, not `out`; recolored border pixels are a
#       rendering effect, not traversable background.
#     - Launch-side recolors must stay out of `cell_owner` bookkeeping. Collision rollback may erase beam trail
#       cells, but must never erase the permanent frame repaint.
#     - `spawn_beam()` still seeds only at genuine gap cells. Recoloring the rest of a touched side does not widen
#       the beam footprint or alter collision geometry.
#     - Because `spawn_beam()` asserts the seed cell is background, `_launch_frame_cells()` must never include the
#       gap cell itself. The helper below enforces that explicitly.
#   placement_legality:
#     - Frames remain 4x4 objects with the original non-overlap and runway constraints, so the original spacing
#       logic still works after replacing the single frame color with per-frame colors.
#     - The new generator additionally guarantees: at least two frames exist, their frame colors are all distinct,
#       their block colors are all distinct and disjoint from frame colors, and each chosen gap set leaves some
#       untouched border pixels so the mixed-color frame effect is visible.
#   observability_and_cleanliness:
#     - Human-visible cues now come from structure, not numeric color IDs: a 2x2 block color always matches its
#       emitted beams and the recolored portion of its own frame, while the untouched border pixels preserve that
#       frame's original input color.
#     - The generator requires multiple distinct frame colors and a minimum number of recolored border pixels, so
#       2-4 examples are enough to show both halves of the combo instead of silently degenerating to the original.
#   color_inference:
#     - No arbitrary color mapping is introduced. The recolored side always copies the local source block color,
#       and the frame's initial color is read directly from the input frame itself.
#   new_solver_assumptions_and_generator_guards:
#     - The solver now assumes each detected 2x2 block has exactly one surrounding frame color on its own border.
#       `_pick_colors()` and `_draw_framed_block()` guarantee that construction.
#     - The solver also assumes every frame has at least one true structural gap and that each gap lies on the
#       4x4 border. `_gap_combos()` and `_frame_gaps_for_block()` keep these conditions aligned.
#     - The generator validates non-degeneracy by requiring at least two distinct input frame colors and by
#       checking that `main()` recolors a minimum number of frame-border pixels in the produced output.


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


def _launch_frame_cells(fr0: int, fc0: int, gap: tuple[int, int]) -> set[tuple[int, int]]:
    """Border cells recolored when a beam exits through `gap` on a 4x4 frame."""
    gr, gc = gap
    lr, lc = gr - fr0, gc - fc0
    assert 0 <= lr < 4 and 0 <= lc < 4, "expected gap inside 4x4 frame coordinates"
    cells: set[tuple[int, int]] = set()
    if lr == 0:
        cells.update((fr0, fc0 + dc) for dc in range(4))
    if lr == 3:
        cells.update((fr0 + 3, fc0 + dc) for dc in range(4))
    if lc == 0:
        cells.update((fr0 + dr, fc0) for dr in range(4))
    if lc == 3:
        cells.update((fr0 + dr, fc0 + 3) for dr in range(4))
    cells.discard(gap)
    return cells


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
    Emit beams from differently colored 4x4 frames and repaint the launch side of each frame.

    Input structure:
    - The most common color is background. Each monochrome 2x2 block is surrounded by its own locally uniform
      4x4 frame, and different frames may use different border colors.
    - One or more border cells on each frame are background-colored gaps. Edge gaps launch cardinal beams and
      corner gaps launch diagonal beams, determined by the gap's outward-facing position on that frame.

    Rule:
    - Every gap emits a beam in the outward direction given by its frame position, using the inner 2x2 block's
      color. The gap cell itself becomes part of the beam.
    - At launch, the beam also repaints the border side(s) it exits through on its own frame: a side gap recolors
      one side, while a corner gap recolors both incident sides. Only border cells that originally had that frame's
      color are overwritten; the beam does not recolor other frames.
    - Once outside the frame, beam travel and collision handling match the source task: free motion through
      background, 90-degree meetings leave a blank pivot and turn to the bisector, and 45-degree meetings merge
      into a zigzag.

    Key insight: The global "green wall" assumption is gone. Solve each frame locally, then apply a local
    launch-side recolor before running the unchanged beam-collision geometry.
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

    def recolor_launch_frame(color: int, frame_color: int, br: int, bc: int, gap: tuple[int, int]) -> None:
        fr0, fc0 = br - 1, bc - 1
        for cell in _launch_frame_cells(fr0, fc0, gap):
            r, c = cell
            if 0 <= r < h and 0 <= c < w and int(g[r, c]) == frame_color:
                out[r, c] = int(color)

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

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            recolor_launch_frame(col, frame_col, br, bc, (gr, gc))
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically (odd parity), not collide at same cell"
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
        assert is_card_a != is_card_b, "45° collision must involve exactly one cardinal and one diagonal beam"

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

        assert beams[bid_u].pos == cell, "cardinal beam head must be at collision cell after 45-deg resolution"

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
            nr, nc = b.pos[0] + d[0], b.pos[1] + d[1]
            nxt = (nr, nc)
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

BLOCK_COUNT_RANGE = (2, 5)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 4: 18}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
ALL_COLORS = list(range(10))
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
MIN_RECOLORED_FRAME_PIXELS = 6
MIN_RECOLORED_PER_FRAME = 3
MIN_UNTOUCHED_FRAME_PIXELS = 2
GEN_RETRIES = 300

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


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 5)}


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
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
            if _rays_clear(obstacles, bg, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, list[int], list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    usable = [c for c in ALL_COLORS if c != bg]
    chosen = list(rng.choice(usable, size=block_count * 2, replace=False))
    frame_colors = chosen[:block_count]
    block_colors = chosen[block_count:]
    return bg, frame_colors, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 14
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _frame_recolor_offsets_for_gap(gap: tuple[int, int]) -> set[tuple[int, int]]:
    return {(r, c) for r, c in _launch_frame_cells(0, 0, gap)}


def _count_recolored_frame_pixels(inp: np.ndarray, out: np.ndarray) -> int:
    bg = get_bg(inp)
    count = 0
    for col, br, bc in _find_blocks_2x2(inp, bg):
        frame_col = _infer_frame_color(inp, bg, br, bc, col)
        fr0, fc0 = br - 1, bc - 1
        fr1, fc1 = br + 2, bc + 2
        for r in range(fr0, fr1 + 1):
            for c in range(fc0, fc1 + 1):
                if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]):
                    continue
                if not (r in (fr0, fr1) or c in (fc0, fc1)):
                    continue
                if int(inp[r, c]) == frame_col and int(out[r, c]) == col:
                    count += 1
    return count


def _count_distinct_frame_colors(inp: np.ndarray) -> int:
    bg = get_bg(inp)
    frame_colors = set()
    for col, br, bc in _find_blocks_2x2(inp, bg):
        frame_colors.add(_infer_frame_color(inp, bg, br, bc, col))
    return len(frame_colors)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_colors, block_colors = _pick_colors(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), frame_color, block_color in zip(positions, frame_colors, block_colors):
        _draw_framed_block(obstacles, r0, c0, bg, int(frame_color), int(block_color), ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    planned_recolor_total = 0
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        viable: list[tuple[tuple[int, int], ...]] = []
        for gap_count, combos in available.items():
            for combo in combos:
                recolor_cells: set[tuple[int, int]] = set()
                for gap in combo:
                    recolor_cells.update(_frame_recolor_offsets_for_gap(gap))
                frame_border_pixels = 12 - gap_count
                untouched = frame_border_pixels - len(recolor_cells)
                if len(recolor_cells) < MIN_RECOLORED_PER_FRAME:
                    continue
                if untouched < MIN_UNTOUCHED_FRAME_PIXELS:
                    continue
                viable.append(combo)
        if not viable:
            return None
        combo = viable[int(rng.integers(len(viable)))]
        chosen_gaps.append(combo)
        block_recolor_cells: set[tuple[int, int]] = set()
        for gap in combo:
            block_recolor_cells.update(_frame_recolor_offsets_for_gap(gap))
        planned_recolor_total += len(block_recolor_cells)

    if planned_recolor_total < MIN_RECOLORED_FRAME_PIXELS:
        return None

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), frame_color, block_color, gaps in zip(positions, frame_colors, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, int(frame_color), int(block_color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None
    if _count_distinct_frame_colors(grid) != block_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if _count_recolored_frame_pixels(grid, out) < MIN_RECOLORED_FRAME_PIXELS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
