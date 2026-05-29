# Variant of e12f9a14
# Dimension: D4 — 源对象类型 (src)
# Change: 绿色框架内的非绿色彩色2x2像素块，每个框架包含一种内部颜色... → 边缘单像素标记作为源

import numpy as np

from arc_utils import get_bg


# Design analysis for the D4 variant
# ---
# variant_spec:
#   input_appearance:
#     background: most-common color, still a plain square field
#     sources:
#       object_type: isolated single pixels on the outer border only
#       color_role: the marker color is also the beam color
#       direction_rule: every source fires diagonally toward the geometric center
#       direction_values: one of NE/NW/SE/SW, derived only from visible position
#     removed_structure:
#       - no 2x2 inner blocks
#       - no 4x4 green frames
#       - no gap detection stage
#   concrete_trace:
#     old:
#       detect a 2x2 monochrome block, infer its surrounding frame color, find missing border cells,
#       and launch a beam from each gap using the gap's outward normal.
#     new:
#       detect every non-background border pixel, convert its location to a centerward diagonal
#       direction, and launch a same-colored beam from that border pixel.
#     difference:
#       source recognition changes completely; beam simulation after spawning stays the same.
#   complexity_calibration:
#     kept:
#       - multi-beam interaction
#       - 90-degree collisions creating a blank hinge cell and cardinal continuation
#       - 45-degree collisions creating zigzags
#     changed:
#       - source taxonomy is simplified to one visible object class, but direction still has
#         four repeated spatial cases so the task is not trivialized into arbitrary line drawing.
#   implicit_assumptions_review:
#     obsolete_from_original:
#       - _find_blocks_2x2 depended on interior 2x2 objects
#       - _infer_frame_color depended on a unique frame ring
#       - _frame_gaps_for_block depended on missing wall pixels on that ring
#       - spawn_beam asserting the seed cell is background is no longer valid because source cells
#         are now occupied colored pixels on the border
#     still_valid:
#       - traversable cells must be background in the original input
#       - blocked collision gaps stay permanently impassable
#       - collision resolution logic can remain unchanged once initial beam directions are defined
#   placement_legality:
#     generator_guarantees:
#       - every non-background input pixel lies on the border
#       - no source lies on the exact center row/column of an odd-sized grid, so the
#         centerward diagonal is never ambiguous
#       - four sources are drawn from opposite border sectors, which keeps the input sparse
#         while repeatedly producing inward beam interactions
#       - solver validation is run on every generated input before acceptance
#   visual_observability:
#     why_humans_can_see_it:
#       - frames disappear entirely, replaced by a few colored border markers
#       - all output strokes clearly emanate from those border markers
#       - repeated examples support the rule "markers point diagonally inward toward the center"
#   visual_cleanliness:
#     choices:
#       - only 3-4 source markers are used
#       - colors are distinct
#       - the canvas is otherwise empty, so ray interactions remain readable
#   color_inference:
#     mapping: identity; each beam keeps its source pixel's color
#     safety: no invisible ordering or color-id arithmetic is used
#   new_main_assumptions_and_generator_support:
#     assumptions:
#       - all non-background pixels are sources on the border
#       - source positions are unique
#       - each source has a nonzero centerward row and column component
#     generator_support:
#       - it constructs exactly four border markers
#       - it excludes ambiguous center-line placements
#       - it validates the interaction pattern with the same simulation core


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}


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


def _source_dir(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    """Diagonal direction pointing from a border source toward the grid center."""
    assert r in (0, h - 1) or c in (0, w - 1), "source must lie on the border"
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    dr = int(np.sign(cr - r))
    dc = int(np.sign(cc - c))
    assert dr != 0 and dc != 0, "source on exact center line would make direction ambiguous"
    return (dr, dc)


def _find_edge_sources(g: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = g.shape
    sources: list[tuple[int, int, int, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == bg:
                continue
            assert r in (0, h - 1) or c in (0, w - 1), "all non-background input pixels must be border sources"
            sources.append((v, r, c, _source_dir(h, w, r, c)))
    sources.sort(key=lambda x: (x[1], x[2], x[0]))
    return sources


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


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    sources = _find_edge_sources(g, bg)
    assert sources, "expected at least one border source"

    stats = {
        "source_count": len(sources),
        "cell_collision_90": 0,
        "cell_collision_45": 0,
        "cross_collision": 0,
        "segment_cross_collision": 0,
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
        assert pos not in cell_owner, "expected unique source positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for color, r, c, direction in sources:
        spawn_beam(color, (r, c), (direction,))

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
            stats["cell_collision_90"] += 1
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not collide at one cell"
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

        stats["cell_collision_45"] += 1
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
        stats["cross_collision"] += 1
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
        stats["segment_cross_collision"] += 1
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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit another source"
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

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit centerward diagonal beams from border markers and let them interact.

    Input structure:
    - bg = most common color. Every non-bg pixel is a single-cell source placed directly on the grid border.
    - There are no enclosing frames or inner 2x2 blocks; source direction is inferred from border position.

    Rule:
    - Each border marker launches a beam of its own color along the diagonal pointing toward the grid center.
    - Beams paint background cells until they leave the grid or collide with other beams.
    - 90° collision: the meeting cell becomes a permanent 1-pixel gap, and both beams redirect along the bisector.
    - 45° collision: the beams merge into a zigzag that alternates diagonal and cardinal steps without leaving a gap.

    Key insight:
    - The only conceptual change is source discovery: border markers replace framed emitters, while the original
      collision system still explains the different post-collision trajectories.
    """
    out, _stats = _simulate(grid)
    return out


# --------------- generator ---------------

GRID_SIZE_RANGE = (15, 31)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.3, 0.7]
PALETTE_FOR_BG = {
    8: [1, 2, 3, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 8, 9],
}
MIN_BEAM_PIXELS_PER_SOURCE = 4
GEN_RETRIES = 400


def _pick_bg_and_colors(rng: np.random.Generator, count: int) -> tuple[int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    palette = PALETTE_FOR_BG[bg]
    colors = list(map(int, rng.choice(palette, size=count, replace=False)))
    return bg, colors


def _sample_sector_sources(n: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    mid = n // 2
    top_left = [(0, c) for c in range(1, mid)]
    top_right = [(0, c) for c in range(mid + 1, n - 1)]
    right_top = [(r, n - 1) for r in range(1, mid)]
    right_bottom = [(r, n - 1) for r in range(mid + 1, n - 1)]
    bottom_left = [(n - 1, c) for c in range(1, mid)]
    bottom_right = [(n - 1, c) for c in range(mid + 1, n - 1)]
    left_top = [(r, 0) for r in range(1, mid)]
    left_bottom = [(r, 0) for r in range(mid + 1, n - 1)]

    families = [
        (top_left, right_top, bottom_right, left_bottom),
        (top_right, right_bottom, bottom_left, left_top),
    ]
    family = families[int(rng.integers(len(families)))]
    if any(not band for band in family):
        return None

    positions = [band[int(rng.integers(len(band)))] for band in family]
    if len(set(positions)) != 4:
        return None
    return positions


def _render_sources(n: int, bg: int, colors: list[int], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((n, n), bg, dtype=int)
    for color, (r, c) in zip(colors, positions):
        grid[r, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))

    positions = _sample_sector_sources(n, rng)
    if positions is None:
        return None

    source_count = 4
    bg, colors = _pick_bg_and_colors(rng, source_count)
    grid = _render_sources(n, bg, colors, positions)

    try:
        out, stats = _simulate(grid)
    except Exception:
        return None

    if stats["source_count"] != source_count:
        return None
    if sum(stats[k] for k in stats if k != "source_count") < 1:
        return None
    if np.count_nonzero(out != grid) < source_count * MIN_BEAM_PIXELS_PER_SOURCE:
        return None

    nonbg = np.argwhere(grid != bg)
    if any(0 < r < n - 1 and 0 < c < n - 1 for r, c in nonbg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D4 input")
