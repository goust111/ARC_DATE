# Combo variant of e12f9a14 (index 8)
# combo_spec: D5/0+D11/1
# Name: 多缺口框架+固定两框架

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D5/0 + D11/1
# a) Concrete behavior trace:
#    The base task allowed 1-5 framed 2x2 blocks and each frame could expose 1-4 structural gaps.
#    Here the arena is tightened to exactly two 4x4 frames, and each frame must expose exactly two
#    gaps, so the input always launches four beams. A concrete template used by the generator is:
#    - left frame gaps at local (0,3) and (3,3), launching NE and SE;
#    - right frame gaps at local (0,0) and (3,1), launching NW and S.
#    In the original task one block could act almost independently. In this fused version the two
#    diagonal beams first cross, then the descendant cardinal/diagonal traces collide again, so the
#    solver has to follow a 4-beam interaction network instead of isolated single exits.
# b) Complexity calibration:
#    Fixing the frame count to two removes source-count uncertainty, but the added second gap per
#    frame more than compensates: four simultaneous beams guarantee a dense middle interaction zone.
#    The generator keeps to a small family of parity-controlled templates and rotates them, so the
#    pattern is visibly different from the base task without exploding into too many case splits.
# c) Hidden-assumption audit:
#    - The original `main()` only assumed "at least one 2x2 block" and "each block has >=1 gap".
#      Those assumptions are now invalid: the fused rule depends on exactly two blocks and exactly
#      two structural gaps per block, so the solver asserts both explicitly.
#    - The collision code assumes beams only hit background or previously blocked collision cells.
#      Multi-gap frames increase the risk of a beam punching into the other frame, so the generator
#      constructs layouts with even frame separation and then rejects any sample whose simulation
#      still violates the free-runway assumption.
#    - `find_partner()` groups adjacent lockstep beams. Under this combo that grouping becomes common
#      rather than incidental, because every sample starts with two 2-beam source families.
# d) Placement legality:
#    The generator uses only pre-validated two-frame templates. Each template already satisfies the
#    4x4-frame non-overlap rule and is then translated inside a larger square canvas before optional
#    0/90/180/270-degree rotation, so all placements stay in bounds with the original >=2-cell clearance.
# e) Visual observability:
#    Human solvers can now rely on three visible invariants across examples: there are always exactly
#    two frames, each frame has exactly two missing border pixels, and the four resulting beams always
#    create at least two distinct collisions. No invisible color ordering is used.
# f) Visual cleanliness:
#    Only two objects are present, both use the same wall color, and the generator rejects weak cases
#    with too few painted pixels or only one collision site. This keeps the interaction zone readable
#    instead of filling the board with unrelated beams.
# g) Color inference:
#    Colors keep the original semantics: background is the most frequent color, one shared frame color
#    encloses the blocks, and each beam inherits the color of its own inner 2x2 block. No extra color
#    mapping is introduced.
#
# New solver assumptions introduced here, and how generate_input() guarantees them:
# - exactly two framed 2x2 blocks are present;
# - the two blocks share one frame color and have distinct payload colors;
# - each frame has exactly two structural border gaps;
# - the chosen gap geometry forces at least one diagonal crossing and at least one later cell collision;
# - frames are fully inside the grid, so gap detection never has to infer through edge truncation.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


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


def _collect_frame_specs(g: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[tuple[int, int], tuple[int, int]]]]:
    blocks = sorted(_find_blocks_2x2(g, bg), key=lambda x: (x[1], x[2], x[0]))
    assert len(blocks) == 2, "expected exactly two 2x2 blocks"
    assert len({col for col, _, _ in blocks}) == 2, "expected the two inner blocks to use distinct colors"

    specs: list[tuple[int, int, int, tuple[tuple[int, int], tuple[int, int]]]] = []
    frame_colors: set[int] = set()
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        frame_colors.add(frame_col)
        gaps = tuple(sorted(set(_frame_gaps_for_block(g, bg, frame_col, br, bc))))
        assert len(gaps) == 2, "expected exactly two structural gaps per frame"
        specs.append((col, br, bc, gaps))
    assert len(frame_colors) == 1, "expected the two frames to share one wall color"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate the original beam-collision physics on a stricter two-frame, two-gap arena.

    Input structure:
    - The background is the most common color. Exactly two monochrome 2x2 blocks appear, and each is enclosed by
      a 4x4 frame border of one shared wall color.
    - Each frame contains exactly two structural border gaps. Together they define four beam seeds, each seed using
      the inner block's color and pointing outward from its own gap location.

    Rule:
    - Launch all four beams simultaneously from the two frames. A corner gap emits a diagonal ray; a side gap emits
      a cardinal ray, exactly as in the base task.
    - Beams keep painting background cells until they leave the grid or enter a collision event.
    - 90-degree collisions leave the meeting cell blank and turn the affected beam family toward the bisector; 45-degree
      cardinal+diagonal collisions create the same alternating zigzag continuation as the base task.

    Key insight:
    - The hard part is no longer "how many sources are there?" but "which two gaps belong to each of the two frames,
      and how do four simultaneous beams interact after repeated collisions?" The fused variant is solved by detecting
      the four exits first, then tracing the original collision rules through a denser interaction network.
    """
    return _simulate(grid, collect_stats=False)[0]


def _simulate(grid: np.ndarray, collect_stats: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    frame_specs = _collect_frame_specs(g, bg)

    stats: dict[str, object] = {
        "block_count": len(frame_specs),
        "gap_counts": [2, 2],
        "beam_count": 0,
        "collision_events": 0,
        "cell_collisions": 0,
        "cross_collisions": 0,
        "segment_cross_collisions": 0,
        "collision_sites": set(),
        "colliding_beams": set(),
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
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

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
            if ob.pattern != b.pattern or ob.phase != b.phase or ob.next_dir() != d:
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

    def record_collision(kind: str, site: tuple[int, int], participants: set[int]) -> None:
        stats["collision_events"] = int(stats["collision_events"]) + 1
        stats["collision_sites"].add(site)
        stats["colliding_beams"].update(participants)
        if kind == "cell":
            stats["cell_collisions"] = int(stats["cell_collisions"]) + 1
        elif kind == "cross":
            stats["cross_collisions"] = int(stats["cross_collisions"]) + 1
        else:
            stats["segment_cross_collisions"] = int(stats["segment_cross_collisions"]) + 1

    for col, br, bc, gaps in frame_specs:
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    stats["beam_count"] = len(beams)
    assert len(beams) == 4, "expected exactly four spawned beams"

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
            assert not (is_diag_a and is_diag_b), "two diagonal beams should collide geometrically, not on one cell"
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
            record_collision("cell", cell, group)
            for gid in group:
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

        zig = (v, u)
        beams[bid_u].pattern = zig
        beams[bid_u].phase = 0
        beams[bid_v].pattern = zig
        beams[bid_v].phase = 0
        beams[bid_u].active = True
        beams[bid_v].active = True
        record_collision("cell", cell, {bid_u, bid_v})

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

    def resolve_cross_collision(a: int, b: int, prop, site: tuple[int, int]) -> None:
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
        record_collision("cross", site, {a, b})

    def resolve_segment_cross_collision(a: int, b: int, prop, site: tuple[int, int]) -> None:
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
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
        record_collision("segment", site, group)

    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            beam = beams[bid]
            d = beam.next_dir()
            nxt = (beam.pos[0] + d[0], beam.pos[1] + d[1])
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit a frame or block"
                beam.active = False
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
            other = move_map.get(((pr, pc + dc), (pr + dr, pc)))
            if other is None or other == bid:
                continue
            cross_pairs.add(tuple(sorted((bid, other))))

        seg_cross: set[tuple[int, int]] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            dr, dc = d
            other = diag_segment_owner((pr, pc + dc), (pr + dr, pc))
            if other is None or other == bid:
                continue
            seg_cross.add((bid, other))

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
            if cand_bs:
                events.append((0, cell[0], cell[1], a, cand_bs[0], "cell"))
        for a, b in cross_pairs:
            ca = prop[a][0]
            cb = prop[b][0]
            events.append((1, min(ca[0], cb[0]), min(ca[1], cb[1]), a, b, "cross"))
        for a, b in seg_cross:
            pr, pc = beams[a].pos
            dr, dc = prop[a][1]
            events.append((1, min(pr, pr + dr), min(pc, pc + dc), a, b, "seg"))

        if events:
            events.sort()
            _, r, c, a, b, kind = events[0]
            site = (r, c)
            if kind == "cell":
                resolve_cell_collision(site, a, b, prop)
            elif kind == "cross":
                resolve_cross_collision(a, b, prop, site)
            else:
                resolve_segment_cross_collision(a, b, prop, site)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            beam = beams[bid]
            beam.path.append(nxt)
            beam.enter_dirs.append(d)
            beam.pos = nxt
            paint_cell(bid, nxt, d)
            beam.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    stats["distinct_collision_sites"] = len(stats["collision_sites"])
    stats["painted_pixels"] = int(np.count_nonzero(out != g))
    return out, stats if collect_stats else {}


# --------------- generator (combo D5/0 + D11/1) ---------------

GRID_SIZE_RANGE = (18, 29)
GEN_RETRIES = 200
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
MIN_PAINTED_PIXELS = 24
MIN_COLLISION_EVENTS = 2
MIN_DISTINCT_COLLISION_SITES = 2

_BASE_LAYOUTS = (
    {
        "name": "template_a",
        "min_n": 18,
        "offsets": ((8, 0), (0, 4)),
        "gaps": (((0, 1), (0, 3)), ((3, 3), (2, 0))),
    },
    {
        "name": "template_b",
        "min_n": 22,
        "offsets": ((14, 6), (0, 0)),
        "gaps": (((0, 0), (2, 3)), ((3, 3), (3, 0))),
    },
)


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=2, replace=False))
    return bg, frame_color, block_colors


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gaps: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _layout_bbox(layout: dict[str, object]) -> tuple[int, int]:
    offsets = layout["offsets"]
    assert isinstance(offsets, tuple)
    max_r = max(int(r) + 3 for r, _c in offsets)
    max_c = max(int(c) + 3 for _r, c in offsets)
    return max_r, max_c


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    layout = _BASE_LAYOUTS[int(rng.integers(len(_BASE_LAYOUTS)))]
    n_lo = max(GRID_SIZE_RANGE[0], int(layout["min_n"]))
    if n_lo >= GRID_SIZE_RANGE[1]:
        return None
    n = int(rng.integers(n_lo, GRID_SIZE_RANGE[1]))
    bg, frame_color, block_colors = _pick_colors(rng)

    max_r, max_c = _layout_bbox(layout)
    max_dr = n - max_r - 2
    max_dc = n - max_c - 2
    if max_dr < 2 or max_dc < 2:
        return None
    dr = int(rng.integers(2, max_dr + 1))
    dc = int(rng.integers(2, max_dc + 1))

    offsets = layout["offsets"]
    gaps = layout["gaps"]
    assert isinstance(offsets, tuple) and isinstance(gaps, tuple)

    grid = np.full((n, n), bg, dtype=int)
    for (r_off, c_off), gap_pair, block_color in zip(offsets, gaps, block_colors):
        _draw_framed_block(grid, int(r_off) + dr, int(c_off) + dc, bg, frame_color, int(block_color), gap_pair)

    rot_k = int(rng.integers(4))
    if rot_k:
        grid = np.rot90(grid, rot_k).copy()

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if stats["beam_count"] != 4:
        return None
    if stats["collision_events"] < MIN_COLLISION_EVENTS:
        return None
    if stats["cell_collisions"] < 1:
        return None
    if int(stats["cross_collisions"]) + int(stats["segment_cross_collisions"]) < 1:
        return None
    if stats["distinct_collision_sites"] < MIN_DISTINCT_COLLISION_SITES:
        return None
    if len(stats["colliding_beams"]) != 4:
        return None
    if stats["painted_pixels"] < MIN_PAINTED_PIXELS:
        return None
    if np.count_nonzero(out != grid) < MIN_PAINTED_PIXELS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
