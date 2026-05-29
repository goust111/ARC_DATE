# Variant of e12f9a14
# Dimension: D15 — 背景色与网格环境 (meta)
# Change: 背景色为天蓝色(8)或黑色(0)，网格中除框架外无其他前景对象... → 网格中存在额外障碍物像素

import itertools

import numpy as np

from arc_utils import get_bg, paste


OBSTACLE_COLOR = 5

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


def _rotate_dir_cw(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS8[(_DIR_TO_I[d] + 2) % 8]


def _rotate_pattern_cw(pattern: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(_rotate_dir_cw(d) for d in pattern)


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


# Design analysis for D15
# - Concrete change: keep the original 4x4 frames and 2x2 emitters, then add isolated gray singleton pixels
#   as explicit obstacles in the background. They are visually distinct from both the frame color and the
#   colored 2x2 sources, so a human can spot "special steering points" immediately.
# - Behavior trace: originally, a beam from a gap continued straight until edge/collision. In this variant,
#   if the next cell is a gray obstacle, the beam does not paint that gray cell; instead it turns 90 degrees
#   clockwise at the previous cell and moves into the turned side cell. After that detour, the whole current
#   movement pattern rotates as well, so straight beams stay straight and zigzags stay zigzags, only rotated.
# - Complexity calibration: one fixed obstacle color and one fixed detour rule avoids hidden case splits, but
#   obstacle placements still create visible route planning because the same beam/collision logic now runs on
#   bent trajectories. The generator requires actual obstacle hits, so samples do not collapse back to the
#   obstacle-free original task.
# - Old assumptions that had to be audited:
#   * `_find_blocks_2x2` assumes every meaningful non-bg 2x2 square is a block. The generator keeps obstacles
#     isolated, so they never fake a block.
#   * `_infer_frame_color` and `_frame_gaps_for_block` assume no extra non-bg pixels touch the frame border.
#     The generator enforces obstacle isolation, so border inference stays valid.
#   * The original proposal step treated any non-bg next cell as an invalid frame/block hit. The variant adds
#     a deliberate non-bg exception, so obstacle interception must happen before that old assertion.
# - Placement legality: obstacles are chosen only on otherwise-clear beam runways, with at least two cells of
#   straight travel before the turn and at least two cells of clear runway after the clockwise turn. Obstacles
#   are also kept away from frames and from each other by an 8-neighborhood isolation check.
# - Visual observability and cleanliness: the obstacle color is fixed gray(5), unused elsewhere. In the output
#   the gray pixel remains unchanged while the nearby colored beam bends around it, making the new rule legible
#   from a few examples without relying on color ids, ordering tricks, or overlapping clutter.


def _simulate(grid: np.ndarray, collect_stats: bool = False) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    frame_colors = {_infer_frame_color(g, bg, br, bc, col) for col, br, bc in blocks}
    assert len(frame_colors) == 1, "expected a single shared frame color"
    frame_color = next(iter(frame_colors))
    block_colors = {col for col, _br, _bc in blocks}
    assert bg != OBSTACLE_COLOR and frame_color != OBSTACLE_COLOR and OBSTACLE_COLOR not in block_colors, (
        "gray(5) is reserved for singleton obstacles in this variant"
    )
    obstacle_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE_COLOR)}

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    stats = {
        "obstacle_turns": 0,
        "obstacle_stops": 0,
        "collision_cell": 0,
        "collision_cross": 0,
        "collision_segcross": 0,
    }

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

    for col, br, bc in blocks:
        for gr, gc in _frame_gaps_for_block(g, bg, frame_color, br, bc):
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
            assert not (is_diag_a and is_diag_b), (
                "two diagonal beams must cross geometrically, not collide at the same cell"
            )
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

        prop: dict[int, tuple[tuple[int, int], tuple[int, int], bool]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            raw = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(raw):
                b.active = False
                continue
            if raw in obstacle_cells:
                nd = _rotate_dir_cw(d)
                alt = (b.pos[0] + nd[0], b.pos[1] + nd[1])
                if not in_bounds(alt):
                    stats["obstacle_stops"] += 1
                    b.active = False
                    continue
                if int(g[alt[0], alt[1]]) != bg:
                    raise AssertionError("obstacle turn must not enter a frame, block, or obstacle")
                if alt in blocked:
                    stats["obstacle_stops"] += 1
                    b.active = False
                    continue
                stats["obstacle_turns"] += 1
                prop[bid] = (alt, nd, True)
                continue
            if int(g[raw[0], raw[1]]) != bg:
                raise AssertionError("beam must not hit a frame, block, or obstacle directly")
            if raw in blocked:
                b.active = False
                continue
            prop[bid] = (raw, d, False)

        if not prop:
            break

        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d, _turned) in prop.items():
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
        for bid, (nxt, d, _turned) in prop.items():
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
        for bid, (nxt, _d, _turned) in prop.items():
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
                stats["collision_cell"] += 1
                resolve_cell_collision((r, c), a, b, prop)
            elif kind == "segcross":
                stats["collision_segcross"] += 1
                resolve_segment_cross_collision(a, b, prop)
            else:
                stats["collision_cross"] += 1
                resolve_cross_collision(a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d, turned = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            if turned:
                b.pattern = _rotate_pattern_cw(b.pattern)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored beams from framed 2x2 blocks, but let gray obstacle pixels deflect the beams before normal beam collisions apply.

    Input structure:
    - The most common color is the background (black or azure), and each colored 2x2 block is wrapped by a shared 4x4 frame with one or more border gaps.
    - Additional gray singleton pixels are scattered in the background; these are obstacles, not emitters and not part of any frame.

    Rule:
    - Every frame gap emits a beam in its outward border direction, painting background cells with the inner 2x2 block color.
    - If a beam would enter a gray obstacle, the obstacle stays gray and the beam turns 90 degrees clockwise at the previous cell; its current movement pattern rotates the same way.
    - After any obstacle detour, beam-beam interactions use the original collision logic: 90-degree meetings create a blank gap and redirect to the bisector, while 45-degree meetings merge into a zigzag.

    Key insight:
    - The new gray pixels are visible steering points: they do not get recolored, but they force nearby rays to peel clockwise around them, so the final drawing depends on both static obstacle placement and the old collision mechanics.
    """
    out, _stats = _simulate(grid, collect_stats=False)
    return out


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
MIN_BEAM_PIXELS_PER_GAP = 2
GEN_RETRIES = 400
MAX_OBSTACLES_PER_GRID = 3
MIN_PRETURN_CELLS = 2
MIN_POSTTURN_CELLS = 2

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


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, OBSTACLE_COLOR)]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _line_cells_until_edge(
    grid: np.ndarray,
    bg: int,
    start: tuple[int, int],
    d: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += d[0]
        c += d[1]
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cell = (r, c)
        if int(grid[r, c]) != bg or cell in reserved:
            return None
        cells.append(cell)


def _isolated_obstacle_site(
    grid: np.ndarray,
    bg: int,
    cell: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != bg or cell in reserved:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            here = (rr, cc)
            if here == cell:
                continue
            if int(grid[rr, cc]) != bg or here in reserved:
                return False
    return True


def _obstacle_candidates_for_seed(
    grid: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    d: tuple[int, int],
    raw_path: list[tuple[int, int]],
    usage: dict[tuple[int, int], int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    turn_dir = _rotate_dir_cw(d)
    candidates: list[tuple[int, int]] = []
    for obs_idx in range(MIN_PRETURN_CELLS, len(raw_path)):
        obs = raw_path[obs_idx]
        if usage.get(obs, 0) != 1:
            continue
        if not _isolated_obstacle_site(grid, bg, obs, reserved):
            continue
        prev = raw_path[obs_idx - 1] if obs_idx > 0 else seed
        turn_path = _line_cells_until_edge(grid, bg, prev, turn_dir, reserved | {obs})
        if turn_path is None or len(turn_path) < MIN_POSTTURN_CELLS:
            continue
        candidates.append(obs)
    return candidates


def _plan_obstacles(
    grid: np.ndarray,
    bg: int,
    seeds: list[tuple[tuple[int, int], tuple[int, int]]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    raw_paths: dict[int, list[tuple[int, int]]] = {}
    usage: dict[tuple[int, int], int] = {}
    for sid, (seed, d) in enumerate(seeds):
        raw_path = _line_cells_until_edge(grid, bg, seed, d, set())
        if raw_path is None or len(raw_path) <= MIN_PRETURN_CELLS:
            continue
        raw_paths[sid] = raw_path
        for cell in raw_path:
            usage[cell] = usage.get(cell, 0) + 1

    if not raw_paths:
        return None

    eligible = [
        sid
        for sid, (seed, d) in enumerate(seeds)
        if sid in raw_paths and _obstacle_candidates_for_seed(grid, bg, seed, d, raw_paths[sid], usage, set())
    ]
    if not eligible:
        return None

    seed_order = eligible[:]
    rng.shuffle(seed_order)
    target = int(rng.integers(1, min(MAX_OBSTACLES_PER_GRID, len(eligible)) + 1))

    chosen: list[tuple[int, int]] = []
    reserved: set[tuple[int, int]] = set()
    for sid in seed_order:
        seed, d = seeds[sid]
        candidates = _obstacle_candidates_for_seed(grid, bg, seed, d, raw_paths[sid], usage, reserved)
        if not candidates:
            continue
        obs = candidates[int(rng.integers(len(candidates)))]
        chosen.append(obs)
        reserved.add(obs)
        if len(chosen) == target:
            break

    if not chosen:
        return None
    return chosen


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

    chosen_gaps = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        gap_count = int(rng.choice(sorted(available)))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    obstacle_cells = _plan_obstacles(grid, bg, seeds, rng)
    if not obstacle_cells:
        return None
    for r, c in obstacle_cells:
        grid[r, c] = OBSTACLE_COLOR

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if stats["obstacle_turns"] < len(obstacle_cells):
        return None
    if stats["obstacle_stops"] != 0:
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
