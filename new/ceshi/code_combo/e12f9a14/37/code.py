# Combo variant of e12f9a14 (index 37)
# combo_spec: D18/2+D19/1
# Name: 框架标记编码优先级+不对称碰撞

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the fused variant:
# - Concrete behavior: each 4x4 frame still wraps a 2x2 monochrome block, but some border-wall
#   pixels are recolored into a single marker color. The count of those marker pixels is the
#   collision priority for all beams emitted by that frame. Example: a frame with 3 markers and
#   one east-facing gap meets a frame with 1 marker and one north-facing gap. In the original
#   task both beams would redirect symmetrically; here the 3-marker beam keeps going while the
#   1-marker beam alone bends.
# - Complexity calibration: priorities are limited to 1/2/3 markers so humans can count them
#   quickly, and all markers share one color so the input does not hide a second legend in hue.
#   Generator samples distinct priorities for the beams that actually collide, preventing the
#   variant from collapsing back to the symmetric original.
# - Broken assumptions from the base task:
#   1. Border pixels are no longer a single frame color, so frame inference must identify the
#      dominant wall color and treat other non-bg/non-block border colors as markers.
#   2. Structural-gap checks can no longer require exact frame-color neighbors; they must accept
#      either frame-color or marker-color walls.
#   3. Collision code can no longer assume both participants change state together. The winner
#      keeps its path, while only the lower-priority beam is repositioned and retargeted.
# - Placement legality: the generator uses explicit 90-degree or 45-degree collision templates,
#   keeps frame bounding boxes separated, keeps every frame inside the grid, and validates the
#   finished grid by rerunning the solver. Extra randomness only affects marker placement, colors,
#   distances, and which beam wins.
# - Visual observability and cleanliness: marker count is a visible, local cue on each frame;
#   the winning ray is the one whose marker count is larger. Output remains legible because each
#   training-style sample has a single intended priority clash and uses one gap per frame, so the
#   asymmetric bend is easy to isolate with the eye.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


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


def _frame_border_cells(br: int, bc: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((br - 1 + gr, bc - 1 + gc), (gr, gc)) for gr, gc in _BORDER_ORDER]


def _infer_frame_features(
    g: np.ndarray, bg: int, br: int, bc: int, block_col: int
) -> tuple[int, list[tuple[int, int]]]:
    """Return dominant frame color and global coords of marker pixels on the border."""
    color_counts: dict[int, int] = {}
    marker_candidates: list[tuple[tuple[int, int], int]] = []
    for (r, c), _local in _frame_border_cells(br, bc):
        v = int(g[r, c])
        if v == bg or v == block_col:
            continue
        color_counts[v] = color_counts.get(v, 0) + 1
        marker_candidates.append(((r, c), v))
    assert color_counts, "expected framed border colors around each 2x2 block"
    max_count = max(color_counts.values())
    frame_colors = [col for col, cnt in color_counts.items() if cnt == max_count]
    assert len(frame_colors) == 1, "expected a unique dominant frame color"
    frame_col = frame_colors[0]
    markers = [pos for pos, col in marker_candidates if col != frame_col]
    return frame_col, markers


def _is_wall_like(v: int, bg: int, block_col: int) -> bool:
    return v != bg and v != block_col


def _frame_gaps_for_block(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for (r, c), _local in _frame_border_cells(br, bc):
        if not (0 <= r < h and 0 <= c < w):
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
                and _is_wall_like(int(g[r - 1, c]), bg, block_col)
                and _is_wall_like(int(g[r + 1, c]), bg, block_col)
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and _is_wall_like(int(g[r, c - 1]), bg, block_col)
                and _is_wall_like(int(g[r, c + 1]), bg, block_col)
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c < w
                and 0 <= r < h
                and 0 <= c - dc < w
                and _is_wall_like(int(g[r - dr, c]), bg, block_col)
                and _is_wall_like(int(g[r, c - dc]), bg, block_col)
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


class _Beam:
    __slots__ = ("bid", "color", "priority", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(
        self,
        bid: int,
        color: int,
        priority: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
    ):
        self.bid = bid
        self.color = int(color)
        self.priority = int(priority)
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


def _simulate(grid: np.ndarray, collect_metrics: bool = False, priority_override: int | None = None):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    metrics = {
        "priority_collision_count": 0,
        "priority_90_count": 0,
        "priority_45_count": 0,
        "priority_cross_count": 0,
        "tie_collision_count": 0,
        "frame_priorities": [],
    }

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

    def spawn_beam(color: int, priority: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, priority, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col, markers = _infer_frame_features(g, bg, br, bc, col)
        del frame_col
        priority = len(markers) if priority_override is None else int(priority_override)
        metrics["frame_priorities"].append(priority)
        for gr, gc in _frame_gaps_for_block(g, bg, br, bc, col):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, priority, (gr, gc), (d,))

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
            if ob.color != b.color or ob.priority != b.priority:
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

    def append_step(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int], advance_phase: bool) -> None:
        assert cell_owner.get(cell) is None, "expected target cell to be free before append"
        b = beams[bid]
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.pos = cell
        paint_cell(bid, cell, enter_dir)
        if advance_phase:
            b.advance_phase()
        b.active = True

    def apply_proposed_move(bid: int, prop) -> None:
        nxt, d = prop[bid]
        append_step(bid, nxt, d, advance_phase=True)

    def priority_order(a: int, b: int) -> tuple[int, int] | None:
        pa = beams[a].priority
        pb = beams[b].priority
        if pa == pb:
            return None
        return (a, b) if pa > pb else (b, a)

    def remove_loser_ownership_if_needed(loser: int, cell: tuple[int, int]) -> None:
        if cell_owner.get(cell) == loser:
            idx = cell_index[cell]
            truncate_beam(loser, idx)

    def redirect_loser_straight(loser: int, turn_dir: tuple[int, int]) -> None:
        assert turn_dir != (0, 0), "expected non-zero turn direction"
        turn_cell = (beams[loser].pos[0] + turn_dir[0], beams[loser].pos[1] + turn_dir[1])
        assert in_bounds(turn_cell) and is_free(turn_cell) and cell_owner.get(turn_cell) is None, "expected free redirected step"
        append_step(loser, turn_cell, turn_dir, advance_phase=False)
        beams[loser].pattern = (turn_dir,)
        beams[loser].phase = 0

    def redirect_loser_zigzag(loser: int, loser_dir: tuple[int, int], other_dir: tuple[int, int]) -> None:
        sidestep = other_dir
        turn_cell = (beams[loser].pos[0] + sidestep[0], beams[loser].pos[1] + sidestep[1])
        assert in_bounds(turn_cell) and is_free(turn_cell) and cell_owner.get(turn_cell) is None, "expected free zigzag sidestep"
        append_step(loser, turn_cell, sidestep, advance_phase=False)
        beams[loser].pattern = (loser_dir, sidestep)
        beams[loser].phase = 0

    def resolve_cell_collision_symmetric(cell: tuple[int, int], a: int, b: int, prop) -> None:
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide in one cell"
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

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True

    def resolve_cell_collision_priority(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
        else:
            db = cell_enter[cell]

        order = priority_order(a, b)
        if order is None:
            metrics["tie_collision_count"] += 1
            resolve_cell_collision_symmetric(cell, a, b, prop)
            return

        winner, loser = order
        diff = _dir_diff(da, db)
        assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"

        remove_loser_ownership_if_needed(loser, cell)
        if winner in prop and prop[winner][0] == cell:
            apply_proposed_move(winner, prop)

        metrics["priority_collision_count"] += 1
        if diff == 2:
            metrics["priority_90_count"] += 1
            turn_dir = _normalize_dir(da[0] + db[0], da[1] + db[1])
            redirect_loser_straight(loser, turn_dir)
            return

        metrics["priority_45_count"] += 1
        winner_dir = da if winner == a else db
        loser_dir = db if winner == a else da
        is_card_w = abs(winner_dir[0]) + abs(winner_dir[1]) == 1
        is_card_l = abs(loser_dir[0]) + abs(loser_dir[1]) == 1
        assert is_card_w != is_card_l, "45-degree collision must involve one cardinal and one diagonal direction"
        redirect_loser_zigzag(loser, loser_dir, winner_dir)

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

    def resolve_cross_collision_symmetric(a: int, b: int, prop) -> None:
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

    def resolve_cross_collision_priority(a: int, b: int, prop) -> None:
        order = priority_order(a, b)
        if order is None:
            metrics["tie_collision_count"] += 1
            resolve_cross_collision_symmetric(a, b, prop)
            return
        winner, loser = order
        winner_dir = prop[winner][1]
        loser_dir = prop[loser][1]
        turn_dir = _normalize_dir(winner_dir[0] + loser_dir[0], winner_dir[1] + loser_dir[1])
        metrics["priority_collision_count"] += 1
        metrics["priority_90_count"] += 1
        metrics["priority_cross_count"] += 1
        apply_proposed_move(winner, prop)
        redirect_loser_straight(loser, turn_dir)

    def resolve_segment_cross_collision_symmetric(a: int, b: int, prop) -> None:
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

    def resolve_segment_cross_collision_priority(a: int, b: int, prop) -> None:
        pr, pc = beams[a].pos
        dr, dc = prop[a][1]
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        seg_dir: tuple[int, int] | None = None
        keep_to: int | None = None
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            keep_to = cell_index[n2]
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            keep_to = cell_index[p2]
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        order = priority_order(a, b)
        if order is None:
            metrics["tie_collision_count"] += 1
            resolve_segment_cross_collision_symmetric(a, b, prop)
            return

        winner, loser = order
        assert seg_dir is not None and keep_to is not None
        if loser == b:
            truncate_beam(b, keep_to)
            turn_dir = _normalize_dir(prop[a][1][0] + seg_dir[0], prop[a][1][1] + seg_dir[1])
            apply_proposed_move(a, prop)
            redirect_loser_straight(b, turn_dir)
        else:
            turn_dir = _normalize_dir(prop[a][1][0] + seg_dir[0], prop[a][1][1] + seg_dir[1])
            redirect_loser_straight(a, turn_dir)
        metrics["priority_collision_count"] += 1
        metrics["priority_90_count"] += 1
        metrics["priority_cross_count"] += 1

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
                assert not in_bounds(nxt) or nxt in blocked or cell_owner.get(nxt) is not None, "beam must stop only at boundary or occupied path"
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
                resolve_cell_collision_priority((r, c), a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision_priority(a, b, prop)
            else:
                resolve_cross_collision_priority(a, b, prop)
            continue

        for bid in sorted(prop):
            apply_proposed_move(bid, prop)
    else:
        raise AssertionError("beam simulation did not converge")

    if collect_metrics:
        return out, metrics
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed color blocks, with border markers encoding which beam wins collisions.

    Input structure:
    - The background is the most common color. Each object is still a 2x2 monochrome block wrapped
      by a 4x4 border frame.
    - A frame border now uses two wall-like colors: one dominant frame color plus a smaller number
      of marker pixels. Background-valued border cells are emission gaps.

    Rule:
    - Every gap emits a beam of the inner block's color in the outward direction implied by the gap.
    - Count the marker pixels on that frame border to get the beam priority. When two beams collide,
      the higher-priority beam keeps its current trajectory while the lower-priority beam alone
      changes course. For 90-degree encounters the loser turns onto the bisector ray; for 45-degree
      encounters the loser becomes a solo zigzag. If priorities tie, fall back to the original
      symmetric behavior where both beams react together.

    Key insight:
    - The output is not determined by color ranking anymore. The crucial hidden variable is a visible
      local count on each frame: more border markers means the ray has precedence, so only the
      lower-count frame's ray bends at the meeting point.
    """
    return _simulate(grid, collect_metrics=False)


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (2, 4)  # uniform draw [lo, hi): 2 or 3 blocks
GRID_SIZE_MAX = 31
GRID_SIZE_MIN = 18
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 5, 6, 7, 9],
    0: [1, 2, 3, 4, 5, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
MIN_BEAM_PIXELS_PER_GAP = 4
GEN_RETRIES = 400
PRIORITY_LEVELS = (1, 2, 3)


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


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    remaining = [c for c in palette if c not in block_colors]
    marker_color = int(rng.choice(remaining))
    return bg, frame_color, marker_color, block_colors


def _sample_markers(
    gaps: tuple[tuple[int, int], ...],
    count: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    candidates = [p for p in _BORDER_ORDER if p not in gaps]
    picks = rng.choice(len(candidates), size=count, replace=False)
    chosen = tuple(candidates[int(i)] for i in np.atleast_1d(picks))
    return chosen


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    marker_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], ...],
    markers: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for mr, mc in markers:
        patch[mr, mc] = marker_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _rays_clear(grid: np.ndarray, bg: int, seeds: list[tuple[tuple[int, int], tuple[int, int]]]) -> bool:
    h, w = grid.shape
    for (r, c), (dr, dc) in seeds:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        while 0 <= nr < h and 0 <= nc < w:
            if int(grid[nr, nc]) != bg:
                return False
            nr += dr
            nc += dc
    return True


def _place_template_90(rng: np.random.Generator, n: int):
    y_left = int(rng.choice([1, 2]))
    x_bottom = int(rng.choice([1, 2]))
    dx = int(rng.integers(4, 8))
    dy = int(rng.integers(4, 8))
    kr_lo = max(4, y_left + 1)
    kr_hi = n - dy - 4
    kc_lo = max(5, dx + 4)
    kc_hi = n - 5
    if kr_lo > kr_hi or kc_lo > kc_hi:
        return None
    kr = int(rng.integers(kr_lo, kr_hi + 1))
    kc = int(rng.integers(kc_lo, kc_hi + 1))
    left = (kr - y_left, kc - dx - 3)
    bottom = (kr + dy, kc - x_bottom)
    if min(left) < 1 or min(bottom) < 1:
        return None
    if left[0] + 3 >= n - 1 or left[1] + 3 >= n - 1 or bottom[0] + 3 >= n - 1 or bottom[1] + 3 >= n - 1:
        return None
    if not _frames_separated(left, bottom):
        return None
    return {
        "kind": "90",
        "frames": [
            {"pos": left, "gap": ((y_left, 3),), "dir": (0, 1)},
            {"pos": bottom, "gap": ((0, x_bottom),), "dir": (-1, 0)},
        ],
    }


def _place_template_45(rng: np.random.Generator, n: int):
    y_left = int(rng.choice([1, 2]))
    dx = int(rng.integers(4, 8))
    t = int(rng.integers(4, 8))
    kr_lo = max(6, t + 4, y_left + 1)
    kr_hi = n - 5
    kc_lo = max(6, dx + 4, t + 4)
    kc_hi = n - 5
    if kr_lo > kr_hi or kc_lo > kc_hi:
        return None
    kr = int(rng.integers(kr_lo, kr_hi + 1))
    kc = int(rng.integers(kc_lo, kc_hi + 1))
    left = (kr - y_left, kc - dx - 3)
    upper_left = (kr - t - 3, kc - t - 3)
    if min(left) < 1 or min(upper_left) < 1:
        return None
    if left[0] + 3 >= n - 1 or left[1] + 3 >= n - 1 or upper_left[0] + 3 >= n - 1 or upper_left[1] + 3 >= n - 1:
        return None
    if not _frames_separated(left, upper_left):
        return None
    return {
        "kind": "45",
        "frames": [
            {"pos": left, "gap": ((y_left, 3),), "dir": (0, 1)},
            {"pos": upper_left, "gap": ((3, 3),), "dir": (1, 1)},
        ],
    }


def _candidate_extra_positions(n: int, placed: list[tuple[int, int]]) -> list[tuple[int, int]]:
    all_positions = [(r, c) for r in range(1, n - 4) for c in range(1, n - 4)]
    return [p for p in all_positions if all(_frames_separated(p, q) for q in placed)]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, frame_color, marker_color, block_colors = _pick_colors(rng, block_count)

    template = None
    for _ in range(80):
        if int(rng.integers(2)) == 0:
            template = _place_template_90(rng, n)
        else:
            template = _place_template_45(rng, n)
        if template is not None:
            break
    if template is None:
        return None

    priorities = list(rng.choice(PRIORITY_LEVELS, size=2, replace=False))
    if int(rng.integers(2)) == 0:
        priorities.sort()
    else:
        priorities.sort(reverse=True)

    grid = np.full((n, n), bg, dtype=int)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    placed = []
    gap_total = 0

    for idx, frame_spec in enumerate(template["frames"]):
        r0, c0 = frame_spec["pos"]
        gaps = tuple(frame_spec["gap"])
        markers = _sample_markers(gaps, priorities[idx], rng)
        _draw_framed_block(
            grid,
            r0,
            c0,
            bg,
            frame_color,
            marker_color,
            int(block_colors[idx]),
            gaps,
            markers,
        )
        placed.append((r0, c0))
        gap_total += len(gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    for idx in range(2, block_count):
        candidates = _candidate_extra_positions(n, placed)
        if not candidates:
            return None
        pos = candidates[int(rng.integers(len(candidates)))]
        gap = (tuple(_BORDER_ORDER[int(rng.integers(len(_BORDER_ORDER)))]),)
        markers = _sample_markers(gap, int(rng.choice(PRIORITY_LEVELS)), rng)
        r0, c0 = pos
        patch_grid = grid.copy()
        _draw_framed_block(
            patch_grid,
            r0,
            c0,
            bg,
            frame_color,
            marker_color,
            int(block_colors[idx]),
            gap,
            markers,
        )
        extra_seed = ((r0 + gap[0][0], c0 + gap[0][1]), _gap_dir(gap[0]))
        if not _rays_clear(patch_grid, bg, [extra_seed]):
            continue
        grid = patch_grid
        placed.append(pos)
        gap_total += 1
        seeds.append(extra_seed)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out, metrics = _simulate(grid, collect_metrics=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if metrics["priority_collision_count"] < 1:
        return None
    if metrics["priority_90_count"] + metrics["priority_45_count"] < 1:
        return None
    if min(metrics["frame_priorities"]) < 1 or max(metrics["frame_priorities"]) > 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid priority-marked e12f9a14 input")
