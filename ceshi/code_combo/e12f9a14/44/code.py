# Combo variant of e12f9a14 (index 44)
# combo_spec: D3/1+D16/2
# Name: 缺口大小编码速度+间歇跳跃轨迹

from __future__ import annotations

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Keep the original object family: square grid, shared background, a shared frame color,
#      and several uniquely colored 2x2 blocks inside 4x4 one-cell-thick frames.
#    - Apply D16/2 first at the frame parser level: an "exit" is no longer an isolated border
#      pixel. It is a connected slit of length 1/2/3 that lies entirely on one frame side.
#      The side gives the initial outward direction, and the slit length gives the interval:
#        len 1 -> color every step
#        len 2 -> move every step but only color every 2nd visited cell
#        len 3 -> move every step but only color every 3rd visited cell
#    - Apply D3/1 at the trajectory level: skipped cells are not empty motion in the physics.
#      Beams still occupy and collide on every traversed cell; only their visible trail becomes
#      sparse. Example:
#        original top opening at anchor (r,c): color r-1, r-2, r-3, r-4, ...
#        combo top slit of length 3: fill the slit itself, then the beam internally visits
#        r-1, r-2, r-3, r-4, ... but only paints r-3, r-6, r-9, ...
#      A 45-degree collision at an unpainted internal step is still real; after the turn, the
#      beam keeps its own interval and continues painting every Nth step along the new pattern.
#
# b) Complexity calibration
#    - The only new discrete parameter is slit length in {1,2,3}. That is small enough to infer
#      from a few examples, but rich enough to create a visible density contrast.
#    - Generator requires at least one non-unit slit and usually mixes multiple slit lengths in
#      the same grid, so the mapping "wider opening -> sparser trail" is repeatedly supported.
#    - We keep the original collision family instead of introducing a second independent code,
#      which prevents the variant from degenerating into a trivial dotted-line drawing task.
#
# c) Implicit-assumption audit against the original main()
#    - Original `_frame_gaps_for_block()` assumed every background border cell was an independent
#      exit. That breaks once two adjacent border cells may represent one wider slit. We replace
#      it with connected-component parsing constrained to a single frame side.
#    - Original occupancy and visibility were the same thing: a cell was part of a beam iff it was
#      painted. D3/1 breaks that assumption. We therefore track every visited cell for collision
#      logic and keep a separate paint flag for whether that visit is visible.
#    - Original proposal logic assumed "beam blocked" when the next cell is non-background or a
#      permanent gap. That remains valid because internal skipped cells are still unit moves on
#      background cells; only painting frequency changes.
#    - Original 45-degree handling relied on the collision cell being visibly filled. With dotted
#      trails that may no longer happen automatically, so the combo solver force-paints the 45deg
#      meeting cell to preserve the original no-gap semantic, while 90deg still force-clears it.
#    - Original diagonal-cross detection was written for unit diagonal motion. That remains valid
#      here because the combo keeps unit movement and changes only paint cadence, not geometric
#      step length.
#
# d) Placement legality
#    - Frames still sit fully in bounds and keep a clearance gap between 4x4 bounding boxes.
#    - Each slit stays on one frame side, has length 1-3, and different slits on one frame cannot
#      touch edge-adjacently; otherwise they would visually merge into a larger ambiguous slit.
#    - Generator checks that every chosen slit has enough unobstructed runway to produce at least
#      two visible beam marks at its own interval before reaching the boundary.
#
# e) Visual observability
#    - Human-solvable cue: the input already shows the controlling variable because slit width is
#      visible directly on the frame. The output shows the consequence as different trail density.
#    - The rule does not depend on numeric color ids or hidden ordering. The same block color can
#      emit dense or sparse trails depending only on the geometry of its slit.
#    - Because the whole slit itself is colored in the output, the solver can align each dotted
#      trail back to the opening that generated it.
#
# f) Clean visualization
#    - Generator biases toward 1-2 slits per frame and uses extra spacing between frames, which
#      keeps dotted trails readable and reduces accidental clutter from many invisible crossings.
#    - Samples that collapse to the dense baseline or add too few visible pixels are rejected.
#      This keeps the fusion visually distinct from the original task.
#
# g) Change inferability
#    - No new color mapping is introduced. Beam color stays equal to the inner 2x2 block color.
#    - All new information is structural: slit width controls paint cadence, while the original
#      collision geometry still explains turning behavior.

import itertools

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


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


def _frame_border_cells(fr0: int, fc0: int) -> list[tuple[int, int]]:
    fr1, fc1 = fr0 + 3, fc0 + 3
    return [
        (fr0, fc0),
        (fr0, fc0 + 1),
        (fr0, fc0 + 2),
        (fr0, fc1),
        (fr0 + 1, fc1),
        (fr0 + 2, fc1),
        (fr1, fc1),
        (fr1, fc0 + 2),
        (fr1, fc0 + 1),
        (fr1, fc0),
        (fr0 + 2, fc0),
        (fr0 + 1, fc0),
    ]


def _frame_openings_for_block(
    g: np.ndarray, bg: int, wall: int, br: int, bc: int
) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int], int]]:
    """Return openings as (cells, anchor, outward_dir, interval)."""
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    assert 0 <= fr0 and 0 <= fc0 and fr1 < h and fc1 < w, "expected full 4x4 frame in bounds"

    border = _frame_border_cells(fr0, fc0)
    border_set = set(border)
    bg_border = {cell for cell in border if int(g[cell[0], cell[1]]) == bg}

    seen: set[tuple[int, int]] = set()
    openings = []
    for cell in border:
        if cell in seen or cell not in bg_border:
            continue
        stack = [cell]
        comp: list[tuple[int, int]] = []
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                ncell = (nr, nc)
                if ncell in bg_border and ncell in border_set and ncell not in seen:
                    seen.add(ncell)
                    stack.append(ncell)

        assert 1 <= len(comp) <= 3, "opening size must encode intervals 1..3"
        if len(comp) == 1:
            r, c = comp[0]
            assert not (
                r in (fr0, fr1) and c in (fc0, fc1)
            ), "single-cell corner openings are ambiguous once size encodes cadence"

        if all(r == fr0 for r, _ in comp):
            ordered = tuple(sorted(comp, key=lambda p: p[1]))
            d = (-1, 0)
        elif all(r == fr1 for r, _ in comp):
            ordered = tuple(sorted(comp, key=lambda p: p[1]))
            d = (1, 0)
        elif all(c == fc0 for _, c in comp):
            ordered = tuple(sorted(comp, key=lambda p: p[0]))
            d = (0, -1)
        elif all(c == fc1 for _, c in comp):
            ordered = tuple(sorted(comp, key=lambda p: p[0]))
            d = (0, 1)
        else:
            raise AssertionError("each opening must lie on exactly one frame side")

        interval = len(ordered)
        anchor = ordered[(interval - 1) // 2]

        if d[0] != 0:
            row = ordered[0][0]
            cols = [c for _, c in ordered]
            if min(cols) > fc0:
                assert int(g[row, min(cols) - 1]) == wall
            if max(cols) < fc1:
                assert int(g[row, max(cols) + 1]) == wall
        else:
            col = ordered[0][1]
            rows = [r for r, _ in ordered]
            if min(rows) > fr0:
                assert int(g[min(rows) - 1, col]) == wall
            if max(rows) < fr1:
                assert int(g[max(rows) + 1, col]) == wall

        openings.append((ordered, anchor, d, interval))

    return openings


class _Beam:
    __slots__ = (
        "bid",
        "color",
        "pos",
        "pattern",
        "phase",
        "path",
        "enter_dirs",
        "paint_flags",
        "active",
        "interval",
        "seed_cells",
        "steps_taken",
    )

    def __init__(
        self,
        bid: int,
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        interval: int,
        seed_cells: tuple[tuple[int, int], ...],
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.paint_flags = [True]
        self.active = True
        self.interval = int(interval)
        self.seed_cells = seed_cells
        self.steps_taken = 0

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _solve(grid: np.ndarray, interval_override: int | None = None) -> np.ndarray:
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

    def occupy_start(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = 0
        out[cell[0], cell[1]] = beams[bid].color

    def set_visible(bid: int, idx: int, visible: bool) -> None:
        b = beams[bid]
        if b.paint_flags[idx] == visible:
            return
        b.paint_flags[idx] = visible
        cell = b.path[idx]
        out[cell[0], cell[1]] = b.color if visible else bg

    def vacate_cell(cell: tuple[int, int]) -> None:
        owner = cell_owner.pop(cell, None)
        if owner is None:
            return
        cell_enter.pop(cell, None)
        idx = cell_index.pop(cell)
        if beams[owner].paint_flags[idx]:
            out[cell[0], cell[1]] = bg

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                vacate_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.paint_flags = b.paint_flags[:keep_len]
        b.pos = b.path[-1]
        b.steps_taken = len(b.path) - 1
        b.active = True

    def append_step(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int], force_paint: bool | None = None) -> None:
        b = beams[bid]
        b.steps_taken += 1
        paint = (b.steps_taken % b.interval == 0) if force_paint is None else bool(force_paint)
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.paint_flags.append(paint)
        b.pos = cell
        idx = len(b.path) - 1
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = idx
        if paint:
            out[cell[0], cell[1]] = b.color

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        interval: int,
        seed_cells: tuple[tuple[int, int], ...],
    ) -> None:
        nonlocal next_bid
        assert int(g[pos[0], pos[1]]) == bg, "expected beam anchor at a frame opening"
        assert pos not in cell_owner, "expected unique beam anchors"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, interval, seed_cells)
        for sr, sc in seed_cells:
            out[sr, sc] = int(color)
        occupy_start(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        openings = _frame_openings_for_block(g, bg, frame_col, br, bc)
        assert openings, "expected at least one opening per frame"
        for cells, anchor, d, interval in openings:
            beam_interval = int(interval_override) if interval_override is not None else interval
            spawn_beam(col, anchor, (d,), beam_interval, cells)

    assert beams, "expected at least one beam"

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
            if ob.interval != b.interval or ob.steps_taken % ob.interval != b.steps_taken % b.interval:
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
            assert not (
                is_diag_a and is_diag_b
            ), "two diagonal beams should cross geometrically, not land on the same cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            vacate_cell(cell)
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
        assert is_card_a != is_card_b, "45-degree collision must involve one cardinal and one diagonal beam"

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
                vacate_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                idx = cell_index[cell]
                truncate_beam(bid_u, idx + 1)
                set_visible(bid_u, idx, True)
            else:
                append_step(bid_u, cell, u, force_paint=True)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)
            set_visible(bid_u, idx, True)

        bv = beams[bid_v]
        adj = (bv.pos[0] + u[0], bv.pos[1] + u[1])
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free zigzag side-step"
        append_step(bid_v, adj, u)

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
            raise AssertionError("expected crossing segment to belong to the partner beam")

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
                b.active = False
                continue
            if cell_owner.get(nxt) == bid:
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
            assert cell_owner.get(nxt) is None, "expected collision-free next cell"
            append_step(bid, nxt, d)
            beams[bid].advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    for b in beams.values():
        for r, c in b.seed_cells:
            out[r, c] = b.color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed 2x2 blocks through variable-width side slits; slit width controls dotted cadence.

    Input structure:
    - The background is the most common color. Each object is a monochrome 2x2 block wrapped by a
      shared-color 4x4 frame.
    - Every frame has one or two connected openings, and each opening is a 1-3 cell slit that lies
      on exactly one frame side. The slit width is meaningful rather than decorative.

    Rule:
    - Each slit emits one beam in that side's outward normal direction using the inner block color,
      and the slit cells themselves are filled in the output.
    - The beam still advances one grid cell at a time for geometry and collision detection, but it
      paints only every Nth visited cell, where N is the slit width. Width 1 is dense, width 2 is
      every other cell, and width 3 is every third cell.
    - Collisions keep the original turning semantics: 90-degree impacts leave a blank collision
      cell and redirect both beams to the bisector, while 45-degree impacts keep the meeting cell
      filled and merge the motion into an alternating diagonal-cardinal zigzag.

    Key insight:
    - The controlling variable is encoded geometrically in the opening itself. Wider slit means a
      sparser visible trail, even though the hidden path still participates in the same collision
      logic as the original task.
    """
    return _solve(grid)


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 10, 2: 16, 5: 22}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 3
PLACEMENT_MARGIN = 2
MIN_VISIBLE_MARKS_PER_OPENING = 2
MIN_BEAM_PIXELS_PER_OPENING = 3
GEN_RETRIES = 500


def _opening_segment_catalog() -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    corners = {(0, 0), (0, 3), (3, 0), (3, 3)}
    side_dirs = {
        "N": (-1, 0),
        "S": (1, 0),
        "W": (0, -1),
        "E": (0, 1),
    }
    for side in ("N", "S", "W", "E"):
        for length in (1, 2, 3):
            for start in range(4 - length + 1):
                if side == "N":
                    cells = tuple((0, start + k) for k in range(length))
                elif side == "S":
                    cells = tuple((3, start + k) for k in range(length))
                elif side == "W":
                    cells = tuple((start + k, 0) for k in range(length))
                else:
                    cells = tuple((start + k, 3) for k in range(length))
                if length == 1 and cells[0] in corners:
                    continue
                anchor = cells[(length - 1) // 2]
                segments.append(
                    {
                        "side": side,
                        "cells": cells,
                        "anchor": anchor,
                        "dir": side_dirs[side],
                        "interval": length,
                    }
                )
    return segments


_OPENING_SEGMENTS = _opening_segment_catalog()


def _segments_compatible(segments: tuple[dict[str, object], ...]) -> bool:
    used: set[tuple[int, int]] = set()
    for seg in segments:
        cells = seg["cells"]
        assert isinstance(cells, tuple)
        for cell in cells:
            if cell in used:
                return False
        for a in cells:
            for b in used:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                    return False
        used.update(cells)
    return True


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
    openings: tuple[dict[str, object], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for seg in openings:
        cells = seg["cells"]
        assert isinstance(cells, tuple)
        for gr, gc in cells:
            patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _ray_mark_capacity(
    grid: np.ndarray,
    bg: int,
    start: tuple[int, int],
    d: tuple[int, int],
    interval: int,
) -> int:
    h, w = grid.shape
    r, c = start
    steps = 0
    while True:
        r += d[0]
        c += d[1]
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != bg:
            return 0
        steps += 1
    return steps // interval


def _valid_opening_sets(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> list[tuple[dict[str, object], ...]]:
    usable = []
    for seg in _OPENING_SEGMENTS:
        anchor = seg["anchor"]
        assert isinstance(anchor, tuple)
        ar = r0 + anchor[0]
        ac = c0 + anchor[1]
        dr, dc = seg["dir"]
        interval = int(seg["interval"])
        marks = _ray_mark_capacity(obstacles, bg, (ar, ac), (dr, dc), interval)
        if marks >= MIN_VISIBLE_MARKS_PER_OPENING:
            usable.append(seg)

    combos: list[tuple[dict[str, object], ...]] = []
    for k in (1, 2):
        for combo in itertools.combinations(usable, k):
            tup = tuple(combo)
            if _segments_compatible(tup):
                combos.append(tup)
    return combos


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


def _collect_opening_intervals(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    intervals: list[int] = []
    for col, br, bc in _find_blocks_2x2(g, bg):
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for cells, _anchor, _d, interval in _frame_openings_for_block(g, bg, frame_col, br, bc):
            assert len(cells) == interval
            intervals.append(interval)
    return intervals


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

    chosen_openings: list[tuple[dict[str, object], ...]] = []
    for r0, c0 in positions:
        options = _valid_opening_sets(obstacles, bg, r0, c0)
        if not options:
            return None
        weights = np.array(
            [
                2
                + sum(int(seg["interval"]) > 1 for seg in combo) * 3
                + len({int(seg["interval"]) for seg in combo}) * 2
                + len(combo)
                for combo in options
            ],
            dtype=float,
        )
        weights /= weights.sum()
        chosen_openings.append(options[int(rng.choice(len(options), p=weights))])

    all_intervals = [int(seg["interval"]) for combo in chosen_openings for seg in combo]
    if len(all_intervals) < 2:
        return None
    if not any(interval > 1 for interval in all_intervals):
        return None
    if 1 not in all_intervals and rng.random() < 0.6:
        return None
    if len(set(all_intervals)) < 2 and rng.random() < 0.75:
        return None

    grid = np.full((n, n), bg, dtype=int)
    opening_total = 0
    for (r0, c0), color, openings in zip(positions, block_colors, chosen_openings):
        opening_total += len(openings)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), openings)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    intervals = _collect_opening_intervals(grid)
    if sorted(intervals) != sorted(all_intervals):
        return None

    try:
        out = main(grid)
        dense = _solve(grid, interval_override=1)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < opening_total * MIN_BEAM_PIXELS_PER_OPENING:
        return None
    if np.array_equal(out, dense):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
