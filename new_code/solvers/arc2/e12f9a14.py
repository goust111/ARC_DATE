import numpy as np
from collections import Counter

from arc_utils import get_bg as _bg_color


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


def _wall_color(g: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    best = None
    best_cnt = -1
    for v, c in zip(vals.tolist(), cnts.tolist(), strict=False):
        v = int(v)
        if v == bg:
            continue
        if c > best_cnt:
            best_cnt = c
            best = v
    return int(best) if best is not None else 3


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


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int | None:
    """Pick the dominant non-bg color on the 4x4 border around a 2x2 block."""
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
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


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
                # It's a missing wall pixel on the would-be 4x4 frame border.
                gaps.append((r, c))
    # Keep only "structural" gaps: those that have wall pixels on the two inward-adjacent border cells.
    # This filters out arbitrary background on the frame perimeter when the frame touches the grid edge.
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        if dr == 0 and dc == 0:
            continue
        ok = True
        if dr == 0:  # side gap: require walls above/below on the frame border
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:  # top/bottom gap: require walls left/right on the border
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:  # corner gap: require the two inward orthogonal border cells to be walls
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
    # Some frames can include an extra decorative corner hole in addition to multiple true exits.
    # We only suppress the SE corner when there are at least 3 structural gaps (so 2-gap frames still emit both).
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
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        # enter_dirs[i] is the move used to enter path[i]; for start, store the first move dir.
        self.enter_dirs = [pattern[0]]
        self.active = True

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

    Input structure:
    - bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
    - Each frame has one or more gap pixels where the wall is missing.

    Rule:
    - Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
    - Beams paint every cell they cross until they hit the grid edge or collide.
    - 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
    - 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

    Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    if not blocks:
        return out

    # Build initial beams from frame gaps.
    beams: dict[int, _Beam] = {}
    seen_seed: set[tuple[int, int, int, tuple[tuple[int, int], ...]]] = set()
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
        if keep_len >= len(b.path):
            return
        for cell in b.path[keep_len:]:
            # Only remove if still owned by this beam.
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> int:
        nonlocal next_bid
        key = (color, pos[0], pos[1], pattern)
        if key in seen_seed:
            return -1
        seen_seed.add(key)
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        # Paint the start cell if it's background and unowned.
        if out[pos[0], pos[1]] == bg and pos not in cell_owner:
            paint_cell(bid, pos, pattern[0])
        else:
            # If already occupied (should be rare), mark inactive; it will be involved in collisions later.
            beams[bid].active = True
        return bid

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        if frame_col is None:
            continue
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = (dr, dc)
            if d == (0, 0):
                continue
            d = _normalize_dir(*d)
            bid = spawn_beam(col, (gr, gc), (d,))
            if bid >= 0:
                # Force-paint start (gaps are bg in input); if already painted, collision will handle.
                if (gr, gc) not in cell_owner:
                    paint_cell(bid, (gr, gc), d)

    if not beams:
        return out

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

    def find_partner(bid: int) -> int | None:
        """Return an active beam adjacent to bid that is moving in lockstep (same next_dir/pattern/phase)."""
        b = beams.get(bid)
        if b is None or not b.active:
            return None
        d = b.next_dir()
        for oid, ob in beams.items():
            if oid == bid or not ob.active:
                continue
            if ob.pattern != b.pattern or ob.phase != b.phase:
                continue
            if ob.next_dir() != d:
                continue
            if abs(ob.pos[0] - b.pos[0]) + abs(ob.pos[1] - b.pos[1]) != 1:
                continue
            # Avoid \"chasing\" pairs where one would step into the other's current cell.
            if (b.pos[0] + d[0], b.pos[1] + d[1]) == ob.pos:
                continue
            if (ob.pos[0] + d[0], ob.pos[1] + d[1]) == b.pos:
                continue
            return oid
        return None

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        # prop: dict bid -> (next_cell, move_dir)
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        diff = _dir_diff(da, db)
        if diff not in (1, 2):
            # Unhandled angle: stop the moving beam.
            beams[a].active = False
            return

        if diff == 2:
            # 90°: remove intersection cell and turn both to the bisector direction.
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)  # keep up to pre-cell (idx is collision cell index)
            # Ensure the intersection cell is permanently blank (gap).
            blocked.add(cell)
            unpaint_cell(cell)
            # Turning direction: normalized sum.
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

        # 45°: merge and turn to a 22.5° zigzag (alternate diag+card), no gap.
        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1
        if is_card_a == is_card_b:
            # If both are cardinal/diagonal but diff==1, fallback to stop.
            beams[a].active = False
            return

        if is_card_a:
            u, v = da, db
            bid_u, bid_v = a, b
        else:
            u, v = db, da
            bid_u, bid_v = b, a

        # Put the cardinal beam head at the collision cell (it "wins" the cell).
        if cell_owner.get(cell) != bid_u:
            # If the diagonal beam owned it, truncate it away.
            if cell_owner.get(cell) == bid_v:
                idx = cell_index[cell]
                truncate_beam(bid_v, idx)  # keep up to pre-cell (idx is collision cell index)
                unpaint_cell(cell)
            # Truncate cardinal beam to its current head, then append collision cell.
            # (If it already had the cell earlier, truncate to it.)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                idx = cell_index[cell]
                truncate_beam(bid_u, idx + 1)
            else:
                bu = beams[bid_u]
                bu.path.append(cell)
                bu.enter_dirs.append(u)
                bu.pos = cell
                paint_cell(bid_u, cell, u)

        # Diagonal beam moves one step in the cardinal direction instead of into the collision cell.
        bv = beams[bid_v]
        adj = (bv.pos[0] + u[0], bv.pos[1] + u[1])
        if in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None:
            bv.path.append(adj)
            bv.enter_dirs.append(u)
            bv.pos = adj
            paint_cell(bid_v, adj, u)
        else:
            bv.active = False

        zig = (v, u)  # after the resolving cardinal step, next step is the diagonal component.
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True

    def diag_segment_owner(p: tuple[int, int], q: tuple[int, int]) -> int | None:
        """Return the beam id that currently owns the diagonal segment connecting p<->q, if any."""
        dr, dc = q[0] - p[0], q[1] - p[1]
        if abs(dr) != 1 or abs(dc) != 1:
            return None
        # Segment p->q exists if q was entered from p.
        oq = cell_owner.get(q)
        if oq is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return oq
        # Or segment q->p exists if p was entered from q.
        op = cell_owner.get(p)
        if op is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return op
        return None

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        db = prop[b][1]
        # Expect both diagonal.
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        if u == (0, 0):
            beams[a].active = False
            beams[b].active = False
            return
        beams[a].pattern = (u,)
        beams[a].phase = 0
        beams[b].pattern = (u,)
        beams[b].phase = 0
        beams[a].active = True
        beams[b].active = True
        # Do not spawn opposite-direction beams: the band continues only forward (matches train0 mid-gap).

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        """Handle a diagonal move that would cross an existing diagonal segment (geometric crossing)."""
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        # Determine the existing segment direction on b.
        seg_dir: tuple[int, int] | None = None
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            # Truncate b back to p2 (remove n2 and everything after it).
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            # Truncate b back to n2 (remove p2 and everything after it).
            truncate_beam(b, cell_index[p2])
        else:
            # If we can't localize the segment orientation, stop the mover.
            beams[a].active = False
            return

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        if u == (0, 0):
            beams[a].active = False
            beams[b].active = False
            return

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

    # Main loop: iterate until no motion or collision.
    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        # Proposals for active beams.
        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            nr, nc = b.pos[0] + d[0], b.pos[1] + d[1]
            nxt = (nr, nc)
            if not in_bounds(nxt) or not is_free(nxt):
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        # Detect crossing collisions among diagonal moves.
        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            move_map[(beams[bid].pos, nxt)] = bid

        # Check for opposite-diagonal crossing in a 2x2 square:
        # (r,c)->(r+sgn, c+sgn) crosses with (r, c+sgn)->(r+sgn, c)
        for (p, n), bid in list(move_map.items()):
            pr, pc = p
            nr, nc = n
            dr = nr - pr
            dc = nc - pc
            # other diagonal endpoints
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            obid = move_map.get((p2, n2))
            if obid is None or obid == bid:
                continue
            a, b = sorted((bid, obid))
            cross_pairs.add((a, b))

        # Detect geometric crossings: a diagonal move crossing an already-drawn diagonal segment.
        seg_cross: set[tuple[int, int]] = set()  # (moving_bid, other_bid)
        for bid, (nxt, d) in prop.items():
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

        # Detect cell collisions (including moving into an occupied cell).
        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int, int, str]] = []
        # cell events
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            # Prefer a moving beam as `a` (must exist in `prop`).
            bidders_sorted = sorted(bidders)
            if not bidders_sorted:
                continue
            a = bidders_sorted[0]
            # Pick `b` as either the owner (if different) or the next bidder.
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
        # cross events (use min of proposed target cells as location key)
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            keyr = min(ca[0], cb[0])
            keyc = min(ca[1], cb[1])
            events.append((1, keyr, keyc, a, b, "cross"))
        # segment-cross events
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
                cell = (r, c)
                resolve_cell_collision(cell, a, b, prop)
            elif kind == "segcross":
                resolve_segment_cross_collision(a, b, prop)
            else:
                resolve_cross_collision(a, b, prop)
            continue

        # No collisions: apply all moves.
        for bid in sorted(prop):
            nxt, d = prop[bid]
            if cell_owner.get(nxt) is not None:
                continue
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()

    return out
