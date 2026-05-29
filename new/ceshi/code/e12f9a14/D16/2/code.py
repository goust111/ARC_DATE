# Variant of e12f9a14
# Dimension: D16 — 缺口位置的约束 (meta)
# Change: 缺口可以出现在框架的任意一条边的任意位置（角落或边的中间），方向从输入中推断... → 缺口大小编码射线速度

from __future__ import annotations

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Keep the original object family: square grid, shared background, 2x2 monochrome blocks,
#      and a one-pixel-thick 4x4 frame around each block.
#    - The visible change is at the frame opening. Instead of isolated 1-cell exits, each exit is
#      a connected slit of length 1-3 lying entirely on one frame side. Its side still determines
#      the outward direction, but its length now determines the beam stride:
#        len 1 -> paint every 1st cell
#        len 2 -> paint every 2nd cell
#        len 3 -> paint every 3rd cell
#    - Example trace:
#      original top-side gap at one cell: seed on that cell, then color r-1, r-2, r-3, ...
#      variant top-side slit of length 3: fill the 3 opening cells in the output, pick a fixed
#      anchor on that slit, then color r-3, r-6, r-9, ... above the anchor.
#      The rule change lives in two places:
#        1. input parsing: isolated border pixels -> side-wise connected opening components
#        2. propagation: unit-step painting -> stride-based dotted painting
#
# b) Complexity calibration
#    - The new degrees of freedom are intentionally small: only slit lengths 1/2/3.
#    - Generator forces at least one non-unit slit per instance and prefers multiple speeds in the
#      same grid so the mapping "wider opening => sparser beam" is visually supported.
#    - We do not add extra marker colors or detached numerals; the speed cue is purely geometric.
#
# c) Implicit-assumption audit against the original main()
#    - Old _frame_gaps_for_block assumed every bg border cell was an independent exit.
#      That becomes wrong once two neighboring bg border cells belong to one wider slit.
#      Fix: parse connected border components and require each component to stay on one side.
#    - Old spawn_beam assumed the gap cell itself was the unique seed/anchor.
#      Now an opening can have multiple cells, so we separate:
#        seed_cells: all opening pixels recolored in the output
#        anchor: one deterministic cell used as the beam head for simulation
#    - Old motion assumed "next position = pos + dir" and collisions happened on painted cells.
#      Now next position is pos + speed * dir and only landing cells get painted.
#      The generator therefore avoids ambiguous cases where two fast diagonal leaps would cross
#      only through skipped cells.
#    - Old diagonal-cross detection was written for unit diagonals. It is kept only for speed-1
#      diagonal steps; all speed>1 geometric-cross ambiguities are excluded by generation.
#
# d) Placement legality
#    - Frames still use non-overlapping 4x4 bounding boxes with extra clearance.
#    - Each slit must be a connected component on one side only; different slits on the same frame
#      are forbidden from touching by 4-neighborhood, otherwise they would visually merge.
#    - For every slit, the obstacle check verifies that repeated stride moves stay inside the grid,
#      pass only through background cells, and have at least two painted landings.
#
# e) Visual observability
#    - The rule is visible from structure alone:
#        narrow slit -> dense beam
#        medium slit -> every-other-cell beam
#        wide slit -> every-third-cell beam
#    - Because the same colored block emits through different slit widths across examples, the
#      solver does not depend on color id semantics.
#
# f) Clean visualization
#    - Generator limits each frame to 1-2 slits and keeps slits from merging around corners.
#    - Faster beams are dotted, so excessive overlap would become noisy; generation rejects cases
#      with poor runway or broken solver assumptions.
#
# g) Color-change inferability
#    - No new color mapping is introduced. Beam color remains the inner block color.
#    - The only new parameter is slit length, carried by shape rather than color identity.

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
    """Return slit components as (cells, anchor, outward_dir, speed)."""
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

        assert 1 <= len(comp) <= 3, "opening size must encode speed 1..3"
        if len(comp) == 1:
            r, c = comp[0]
            assert not (
                r in (fr0, fr1) and c in (fc0, fc1)
            ), "single-pixel corner openings are directionally ambiguous in this variant"

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

        anchor = ordered[(len(ordered) - 1) // 2]
        speed = len(ordered)

        # Structural sanity: cells just outside the component on the same side, if any, must be wall.
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

        openings.append((ordered, anchor, d, speed))

    return openings


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "speed", "seed_cells")

    def __init__(
        self,
        bid: int,
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        speed: int,
        seed_cells: tuple[tuple[int, int], ...],
    ):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.speed = int(speed)
        self.seed_cells = seed_cells

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _solve(grid: np.ndarray, speed_override: int | None = None) -> np.ndarray:
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

    def spawn_beam(
        color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        speed: int,
        seed_cells: tuple[tuple[int, int], ...],
    ) -> None:
        nonlocal next_bid
        assert int(g[pos[0], pos[1]]) == bg, "expected beam anchor at a frame opening"
        assert pos not in cell_owner, "expected unique beam anchors"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, speed, seed_cells)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        openings = _frame_openings_for_block(g, bg, frame_col, br, bc)
        assert openings, "expected at least one opening per frame"
        for cells, anchor, d, speed in openings:
            beam_speed = speed if speed_override is None else int(speed_override)
            spawn_beam(col, anchor, (d,), beam_speed, cells)

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def stride_cells(start: tuple[int, int], d: tuple[int, int], speed: int) -> list[tuple[int, int]]:
        return [(start[0] + step * d[0], start[1] + step * d[1]) for step in range(1, speed + 1)]

    def traverse(start: tuple[int, int], d: tuple[int, int], speed: int) -> list[tuple[int, int]] | None:
        cells = stride_cells(start, d, speed)
        for i, cell in enumerate(cells):
            if not in_bounds(cell):
                return None
            if int(g[cell[0], cell[1]]) != bg or cell in blocked:
                return None
            if i < len(cells) - 1 and cell in cell_owner:
                return None
        return cells

    def find_partner(bid: int) -> int | None:
        b = beams.get(bid)
        if b is None or not b.active:
            return None
        d = b.next_dir()
        partner: int | None = None
        for oid, ob in beams.items():
            if oid == bid or not ob.active:
                continue
            if ob.pattern != b.pattern or ob.phase != b.phase or ob.speed != b.speed:
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
            ), "two diagonal beams must cross geometrically, not collide by land on the same cell"
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
        adj_cells = traverse(bv.pos, u, bv.speed)
        assert adj_cells is not None, "expected free stride for the zigzag side-step"
        adj = adj_cells[-1]
        assert cell_owner.get(adj) is None, "expected free adjacent zigzag landing"
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

        prop: dict[int, tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            cells = traverse(b.pos, d, b.speed)
            if cells is None:
                b.active = False
                continue
            nxt = cells[-1]
            if cell_owner.get(nxt) == bid:
                b.active = False
                continue
            prop[bid] = (nxt, d, cells)

        if not prop:
            break

        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d, _cells) in prop.items():
            if beams[bid].speed != 1:
                continue
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
        for bid, (_nxt, d, _cells) in prop.items():
            if beams[bid].speed != 1:
                continue
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
        for bid, (nxt, _d, _cells) in prop.items():
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
            nxt, d, _cells = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free landing cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    for b in beams.values():
        for cell in b.seed_cells:
            out[cell[0], cell[1]] = b.color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed 2x2 blocks through variable-width side slits; slit width controls stride.

    Input structure:
    - The background is the most common color. Each object is still a monochrome 2x2 block wrapped
      by a shared-color 4x4 frame.
    - Every frame has one or two connected openings, and each opening is a 1-3 pixel slit that
      stays on a single frame side instead of being treated as separate independent holes.

    Rule:
    - Each slit emits one beam in the outward normal direction of that side, using the inner block's
      color. The whole slit gets filled in the output.
    - Slit length is the beam speed: length 1 paints every step, length 2 paints every other cell,
      and length 3 paints every third cell while the skipped cells stay background.
    - When two beams land into the same interaction geometry, they still use the original turning
      rules: 90-degree impacts leave a blank collision cell and bend both beams to the bisector;
      45-degree impacts merge into an alternating diagonal/cardinal zigzag.

    Key insight:
    - The new information is not encoded by color or by a detached marker. It is encoded directly
      in the opening geometry: wider slit means the same-colored beam advances in larger dotted hops.
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
MIN_RAY_LANDINGS = 2
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
                        "speed": length,
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


def _path_cells(start: tuple[int, int], d: tuple[int, int], speed: int) -> list[tuple[int, int]]:
    return [(start[0] + k * d[0], start[1] + k * d[1]) for k in range(1, speed + 1)]


def _ray_stride_clear(
    grid: np.ndarray,
    bg: int,
    start: tuple[int, int],
    d: tuple[int, int],
    speed: int,
) -> bool:
    h, w = grid.shape
    pos = start
    landings = 0
    while True:
        cells = _path_cells(pos, d, speed)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            break
        if any(int(grid[r, c]) != bg for r, c in cells):
            return False
        pos = cells[-1]
        landings += 1
    return landings >= MIN_RAY_LANDINGS


def _valid_opening_sets(
    obstacles: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
) -> list[tuple[dict[str, object], ...]]:
    usable = []
    for seg in _OPENING_SEGMENTS:
        ar = r0 + seg["anchor"][0]
        ac = c0 + seg["anchor"][1]
        dr, dc = seg["dir"]
        speed = int(seg["speed"])
        if _ray_stride_clear(obstacles, bg, (ar, ac), (dr, dc), speed):
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


def _collect_opening_speeds(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    speeds: list[int] = []
    for col, br, bc in _find_blocks_2x2(g, bg):
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for cells, _anchor, _d, speed in _frame_openings_for_block(g, bg, frame_col, br, bc):
            assert len(cells) == speed
            speeds.append(speed)
    return speeds


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
                + sum(int(seg["speed"]) > 1 for seg in combo) * 2
                + len({int(seg["speed"]) for seg in combo})
                for combo in options
            ],
            dtype=float,
        )
        weights /= weights.sum()
        choice = int(rng.choice(len(options), p=weights))
        chosen_openings.append(options[choice])

    all_speeds = [int(seg["speed"]) for combo in chosen_openings for seg in combo]
    if not any(speed > 1 for speed in all_speeds):
        return None
    if len(all_speeds) < 2:
        return None
    if len(set(all_speeds)) < 2 and rng.random() < 0.75:
        return None

    grid = np.full((n, n), bg, dtype=int)
    opening_total = 0
    for (r0, c0), color, openings in zip(positions, block_colors, chosen_openings):
        opening_total += len(openings)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), openings)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    speeds = _collect_opening_speeds(grid)
    if sorted(speeds) != sorted(all_speeds):
        return None

    try:
        out = main(grid)
        baseline = _solve(grid, speed_override=1)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < opening_total * MIN_BEAM_PIXELS_PER_OPENING:
        return None
    if not any(speed > 1 for speed in speeds):
        return None
    if np.array_equal(out, baseline):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D16 input")
