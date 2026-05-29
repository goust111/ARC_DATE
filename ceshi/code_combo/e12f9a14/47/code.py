# Combo variant of e12f9a14 (index 47)
# combo_spec: D3/0+D15/1
# Name: 网格分区+周期颜色图案

from __future__ import annotations

import itertools

import numpy as np

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# variant:
#   core_change:
#     from:
#       - the input had one uniform background color
#       - each beam painted a single global AABB-style cycle in the D3 variant
#     to:
#       - the input is split into two large rectangular background regions
#       - the beam keeps the original geometry, but its color cycle depends on
#         the region it is currently traversing
#   concrete_appearance:
#     input:
#       - one straight full-grid partition line divides the board into two
#         background colors
#       - framed 2x2 sources sit entirely inside one side of the partition
#       - gaps inherit the local region color, so the exit visibly belongs to a
#         specific region before the beam starts
#     output:
#       - in the top-left region, trails use AABB
#       - in the opposite region, trails use ABAB
#       - when a beam crosses the partition, the new region restarts its own
#         cycle from phase 0 at the first painted cell after the crossing
#   behavior_trace:
#     example:
#       - suppose A=red block color and B=green frame color
#       - before the partition: red, red, green, green, ...
#       - after the partition: red, green, red, green, ...
#       - if a 90-degree collision blanks a cell, that missing cell does not
#         advance the cycle; if a 45-degree collision paints resolving cells,
#         those cells do advance the cycle exactly once
#   complexity_calibration:
#     - only two region behaviors are used, keeping the task learnable
#     - both region cycles reuse colors already visible in each source object:
#       A is the inner block and B is the shared frame
#     - the generator requires at least one cross-region beam, so the fused
#       behavior is visible and not reducible to the plain D3 variant
#   implicit_assumptions_review:
#     - original `_find_blocks_2x2()` is no longer valid because large region
#       rectangles contain many accidental monochrome 2x2 patches
#     - original `is_free()` assumed exactly one traversable background color;
#       now any region-color cell is traversable
#     - original `unpaint_cell()` could restore one global background color;
#       now it must restore each cell's original region color
#     - D3-style path truncation now also has to restore the beam's local
#       region-cycle state, because the future color phase depends on the last
#       surviving painted cell and on which region it belongs to
#   placement_legality:
#     - frames are still non-overlapping 4x4 objects with original gap rules
#     - every frame is placed wholly inside one partition side
#     - the generator checks ray clearance against full frames before opening
#       gaps, then validates the result with the solver
#   visual_observability:
#     - the partition is a large visible structure in the input, so the switch
#       condition is directly observable
#     - the output shows both periodicity and the reset point at the boundary,
#       which makes the fusion stronger than "D3 plus an unrelated background"
#   cleanliness:
#     - only one straight partition is used, avoiding noisy mosaic backgrounds
#     - all frames share one frame color and regions are large simple blocks,
#       so the output remains easy to parse
#   color_inference:
#     - no branch depends on raw color ids; the special region is identified by
#       visible position: the color occupying the top-left corner uses AABB
#     - the opposite region uses ABAB, giving a repeated positional cue rather
#       than an arbitrary color lookup


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}


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


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _structural_gaps(window: np.ndarray, frame_color: int) -> list[tuple[int, int]]:
    gaps = [(r, c) for r, c in _BORDER_ORDER if int(window[r, c]) != frame_color]
    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == 0 else (1 if r == 3 else 0)
        dc = -1 if c == 0 else (1 if c == 3 else 0)
        assert dr != 0 or dc != 0, "expected candidate gap on frame border"
        ok = True
        if dr == 0:
            ok = int(window[r - 1, c]) == frame_color and int(window[r + 1, c]) == frame_color
        elif dc == 0:
            ok = int(window[r, c - 1]) == frame_color and int(window[r, c + 1]) == frame_color
        else:
            ok = int(window[r - dr, c]) == frame_color and int(window[r, c - dc]) == frame_color
        if ok:
            out.append((r, c))
    out_set = set(out)
    if len(out_set) >= 3 and (3, 3) in out_set and (0, 2) in out_set:
        out = [p for p in out if p != (3, 3)]
    return out


def _find_sources_and_regions(
    g: np.ndarray,
) -> tuple[list[tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]], tuple[int, int], int]:
    h, w = g.shape
    sources: list[tuple[int, int, int, int, int, tuple[tuple[int, int], ...]]] = []
    seen_centers: set[tuple[int, int]] = set()

    for r0 in range(h - 3):
        for c0 in range(w - 3):
            window = g[r0:r0 + 4, c0:c0 + 4]
            center = window[1:3, 1:3]
            block_color = int(center[0, 0])
            if not np.all(center == block_color):
                continue

            border_vals = [int(window[r, c]) for r, c in _BORDER_ORDER]
            border_colors = sorted(set(border_vals))
            if len(border_colors) != 2 or block_color in border_colors:
                continue

            counts = {color: border_vals.count(color) for color in border_colors}
            frame_candidates = [color for color, cnt in counts.items() if 8 <= cnt <= 11]
            gap_candidates = [color for color, cnt in counts.items() if 1 <= cnt <= 4]
            if len(frame_candidates) != 1 or len(gap_candidates) != 1:
                continue
            frame_color = frame_candidates[0]
            gap_region = gap_candidates[0]
            if frame_color == gap_region:
                continue

            local_gaps = _structural_gaps(window, frame_color)
            if not local_gaps:
                continue
            if any(int(window[gr, gc]) != gap_region for gr, gc in local_gaps):
                continue
            if sum(int(window[r, c]) != frame_color for r, c in _BORDER_ORDER) != len(local_gaps):
                continue

            br, bc = r0 + 1, c0 + 1
            if (br, bc) in seen_centers:
                continue
            seen_centers.add((br, bc))
            gaps_global = tuple((r0 + gr, c0 + gc) for gr, gc in local_gaps)
            sources.append((block_color, br, bc, frame_color, gap_region, gaps_global))

    assert sources, "expected at least one framed source in input"

    frame_colors = {frame_color for _block, _br, _bc, frame_color, _gap_region, _gaps in sources}
    assert len(frame_colors) == 1, "expected one shared frame color across sources"
    frame_color = next(iter(frame_colors))

    block_colors = {block for block, _br, _bc, _frame_color, _gap_region, _gaps in sources}
    region_colors = tuple(sorted(int(v) for v in set(np.unique(g)) - block_colors - {frame_color}))
    assert len(region_colors) == 2, "expected exactly two background-region colors"

    for _block, _br, _bc, _frame_color, gap_region, _gaps in sources:
        assert gap_region in region_colors, "frame gaps must expose one of the region colors"

    return sources, (int(region_colors[0]), int(region_colors[1])), frame_color


def _default_region_patterns(g: np.ndarray, region_colors: tuple[int, int]) -> dict[int, tuple[int, ...]]:
    corner_region = int(g[0, 0])
    assert corner_region in region_colors, "top-left corner must belong to a region"
    other_region = next(color for color in region_colors if color != corner_region)
    return {
        corner_region: (0, 0, 1, 1),  # AABB
        other_region: (0, 1, 0, 1),   # ABAB
    }


class _Beam:
    __slots__ = (
        "bid",
        "source_color",
        "frame_color",
        "pos",
        "pattern",
        "phase",
        "path",
        "enter_dirs",
        "active",
        "region_color",
        "color_phase",
        "region_hist",
        "color_phase_hist",
    )

    def __init__(
        self,
        bid: int,
        source_color: int,
        frame_color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
    ):
        self.bid = bid
        self.source_color = int(source_color)
        self.frame_color = int(frame_color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.region_color: int | None = None
        self.color_phase = 0
        self.region_hist: list[int] = []
        self.color_phase_hist: list[int] = []

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _simulate(
    grid: np.ndarray,
    collect_stats: bool = False,
    region_patterns_override: dict[int, tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sources, region_colors, frame_color_global = _find_sources_and_regions(g)
    region_colors_set = set(region_colors)
    region_patterns = (
        dict(region_patterns_override)
        if region_patterns_override is not None
        else _default_region_patterns(g, region_colors)
    )
    assert set(region_patterns) == region_colors_set, "expected one pattern per region color"
    assert all(len(pattern) == 4 for pattern in region_patterns.values()), "expected 4-step region patterns"

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    stats = {
        "beam_count": 0,
        "collision_45": 0,
        "collision_90": 0,
        "collision_cross": 0,
        "collision_segment_cross": 0,
        "beam_pixels": 0,
        "frame_trail_pixels": 0,
        "source_trail_pixels": 0,
        "region_switches": 0,
        "beams_that_switch": 0,
        "corner_region_pixels": 0,
        "other_region_pixels": 0,
    }

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}
    switched_beams: set[int] = set()
    source_colors = {block for block, _br, _bc, _frame, _gap_region, _gaps in sources}

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        r, c = cell
        beam = beams[bid]
        region = int(g[r, c])
        assert region in region_patterns, "beam must stay on region cells"
        if beam.region_color != region:
            if beam.region_color is not None:
                stats["region_switches"] += 1
                switched_beams.add(bid)
            beam.region_color = region
            beam.color_phase = 0
        token = region_patterns[region][beam.color_phase]
        out[r, c] = beam.source_color if token == 0 else beam.frame_color
        beam.color_phase = (beam.color_phase + 1) % len(region_patterns[region])
        beam.region_hist.append(region)
        beam.color_phase_hist.append(beam.color_phase)
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beam.path) - 1

    def unpaint_cell(cell: tuple[int, int]) -> None:
        if cell not in cell_owner:
            return
        out[cell[0], cell[1]] = int(g[cell[0], cell[1]])
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
        b.region_hist = b.region_hist[:keep_len]
        b.color_phase_hist = b.color_phase_hist[:keep_len]
        b.pos = b.path[-1]
        b.region_color = b.region_hist[-1]
        b.color_phase = b.color_phase_hist[-1]
        b.active = True

    def spawn_beam(
        source_color: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
    ) -> None:
        nonlocal next_bid
        assert int(g[pos[0], pos[1]]) in region_colors_set, "expected beam seed on a region-colored gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, source_color, frame_color_global, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc, _frame_col, _gap_region, gaps in sources:
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    stats["beam_count"] = len(beams)
    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_free(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) in region_colors_set and cell not in blocked

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
            stats["collision_90"] += 1
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not land on one cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                truncate_beam(b, cell_index[cell])
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

        stats["collision_45"] += 1
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
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                truncate_beam(bid_u, cell_index[cell] + 1)
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
        stats["collision_cross"] += 1
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
        stats["collision_segment_cross"] += 1
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)
        seg_dir: tuple[int, int] | None = None
        if (
            cell_owner.get(n2) == b
            and cell_index.get(n2, 0) > 0
            and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1])
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif (
            cell_owner.get(p2) == b
            and cell_index.get(p2, 0) > 0
            and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1])
        ):
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
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked, "beam must stop only at edge or a blocked collision gap"
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
            cross_pairs.add(tuple(sorted((bid, obid))))

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
            events.append((0, cell[0], cell[1], a, cand_bs[0], "cell"))
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

    if collect_stats:
        painted_cells = list(cell_owner)
        stats["beam_pixels"] = len(painted_cells)
        stats["frame_trail_pixels"] = sum(int(out[r, c]) == frame_color_global for r, c in painted_cells)
        stats["source_trail_pixels"] = sum(int(out[r, c]) in source_colors for r, c in painted_cells)
        corner_region = int(g[0, 0])
        other_region = next(color for color in region_colors if color != corner_region)
        stats["corner_region_pixels"] = sum(int(g[r, c]) == corner_region for r, c in painted_cells)
        stats["other_region_pixels"] = sum(int(g[r, c]) == other_region for r, c in painted_cells)
        stats["beams_that_switch"] = len(switched_beams)

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed 2x2 sources exactly as in the original task, but let the trail's repeating color rhythm depend on which background partition the beam is currently inside.

    Input structure:
    - The board is split into two large rectangular background-color regions by one straight horizontal or vertical boundary.
    - Each source is still a monochrome 2x2 block inside a 4x4 frame of one shared frame color, and every gap sits inside one of the visible regions.

    Rule:
    - Gap positions still determine the emitted beam directions, and all beam-collision geometry stays the same as the original task.
    - The trail is periodic rather than solid. In the region touching the top-left corner, the beam paints `AABB`; in the opposite region, it paints `ABAB`, where `A` is the source-block color and `B` is the shared frame color.
    - Crossing the partition is a color-mode switch: the first painted cell in the new region restarts that region's pattern from its own first phase. Collision gaps still consume no phase, while painted collision-resolution cells do.

    Key insight: The puzzle is now about combining two cues at once — source objects provide the two trail colors, and the large background partition tells you which periodic template controls the next stretch of the same beam.
    """
    return _simulate(grid, collect_stats=False)[0]


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (1, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {1: 12, 2: 14, 5: 20}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
MIN_FRAME_TRAIL_PIXELS = 4
MIN_PRE_SWITCH_CELLS = 3
MIN_POST_SWITCH_CELLS = 4
MIN_OTHER_REGION_PIXELS = 4
MIN_PATTERN_DIFF_PIXELS = 2
GEN_RETRIES = 400


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


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _grid_size_min(block_count: int) -> int:
    result = 12
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, int, list[int]]:
    colors = list(int(v) for v in rng.choice(np.arange(10), size=3 + block_count, replace=False))
    region_a, region_b, frame_color = colors[:3]
    block_colors = colors[3:]
    return region_a, region_b, frame_color, block_colors


def _partition_grid(
    n: int,
    region_a: int,
    region_b: int,
    orientation: str,
    split: int,
) -> np.ndarray:
    grid = np.full((n, n), region_a, dtype=int)
    if orientation == "v":
        grid[:, split:] = region_b
    else:
        grid[split:, :] = region_b
    return grid


def _region_at(
    cell: tuple[int, int],
    orientation: str,
    split: int,
    region_a: int,
    region_b: int,
) -> int:
    r, c = cell
    if orientation == "v":
        return region_a if c < split else region_b
    return region_a if r < split else region_b


def _choose_positions(
    n: int,
    count: int,
    orientation: str,
    split: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]] | None:
    margin = PLACEMENT_MARGIN
    row_positions = list(range(margin, n - 5))
    left: list[tuple[int, int, int]] = []
    right: list[tuple[int, int, int]] = []

    if orientation == "v":
        left_cols = list(range(margin, split - 3))
        right_cols = list(range(split, n - 5))
        if not left_cols or not right_cols:
            return None
        left = [(r, c, 0) for r in row_positions for c in left_cols]
        right = [(r, c, 1) for r in row_positions for c in right_cols]
    else:
        top_rows = list(range(margin, split - 3))
        bottom_rows = list(range(split, n - 5))
        if not top_rows or not bottom_rows:
            return None
        left = [(r, c, 0) for r in top_rows for c in row_positions]
        right = [(r, c, 1) for r in bottom_rows for c in row_positions]

    selected: list[tuple[int, int, int]] = []
    requirements = [int(rng.integers(2))] if count == 1 else [0, 1]
    pools = {0: left, 1: right}

    for side in requirements:
        viable = [p for p in pools[side] if all(_frames_separated((p[0], p[1]), (q[0], q[1])) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])

    all_positions = left + right
    for _ in range(count - len(selected)):
        viable = [p for p in all_positions if all(_frames_separated((p[0], p[1]), (q[0], q[1])) for q in selected)]
        if not viable:
            return None
        selected.append(viable[int(rng.integers(len(viable)))])

    return selected


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_color: int,
    block_color: int,
    gap_color: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = gap_color
    paste(grid, patch, r0, c0)


def _rays_clear(
    grid: np.ndarray,
    region_colors: set[int],
    seeds: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    for (r, c), (dr, dc) in seeds:
        steps = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            steps += 1
            if int(grid[nr, nc]) not in region_colors:
                return False
            nr += dr
            nc += dc
        if steps < MIN_RAY_STEPS:
            return False
    return True


def _seed_switch_profile(
    n: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
    orientation: str,
    split: int,
    region_a: int,
    region_b: int,
) -> tuple[bool, int, int]:
    start_region = _region_at(seed, orientation, split, region_a, region_b)
    before_cells = 1
    after_cells = 0
    crossed = False

    r, c = seed
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < n and 0 <= c < n:
        region = _region_at((r, c), orientation, split, region_a, region_b)
        if not crossed and region == start_region:
            before_cells += 1
        else:
            crossed = True
            after_cells += 1
        r += dr
        c += dc
    return crossed, before_cells, after_cells


def _combo_has_good_crossing(
    n: int,
    r0: int,
    c0: int,
    combo: tuple[tuple[int, int], ...],
    orientation: str,
    split: int,
    region_a: int,
    region_b: int,
) -> bool:
    for gap in combo:
        seed = (r0 + gap[0], c0 + gap[1])
        crossed, before_cells, after_cells = _seed_switch_profile(
            n,
            seed,
            _gap_dir(gap),
            orientation,
            split,
            region_a,
            region_b,
        )
        if crossed and before_cells >= MIN_PRE_SWITCH_CELLS and after_cells >= MIN_POST_SWITCH_CELLS:
            return True
    return False


def _clear_gap_combos(
    obstacles: np.ndarray,
    region_colors: set[int],
    r0: int,
    c0: int,
) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    clear: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for gap_count, combos in _GAP_COMBOS.items():
        for combo in combos:
            seeds = [((r0 + gr, c0 + gc), _gap_dir((gr, gc))) for gr, gc in combo]
            if _rays_clear(obstacles, region_colors, seeds):
                clear.setdefault(gap_count, []).append(combo)
    return clear


def _pick_gap_count(available_counts: list[int], rng: np.random.Generator) -> int:
    weighted: list[int] = []
    for gap_count in available_counts:
        weighted.extend([gap_count] * gap_count)
    return int(weighted[int(rng.integers(len(weighted)))])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    orientation = "v" if int(rng.integers(2)) == 0 else "h"
    split_lo = max(6, n // 3)
    split_hi = min(n - 6, (2 * n) // 3)
    if split_lo > split_hi:
        return None
    split = int(rng.integers(split_lo, split_hi + 1))

    region_a, region_b, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, block_count, orientation, split, rng)
    if positions is None:
        return None

    base = _partition_grid(n, region_a, region_b, orientation, split)
    obstacles = base.copy()
    for (r0, c0, side), color in zip(positions, block_colors):
        gap_color = region_a if side == 0 else region_b
        _draw_framed_block(obstacles, r0, c0, frame_color, int(color), gap_color, ())

    available_per_frame: list[dict[int, list[tuple[tuple[int, int], ...]]]] = []
    crossing_choices: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    region_colors_set = {region_a, region_b}
    for idx, (r0, c0, _side) in enumerate(positions):
        available = _clear_gap_combos(obstacles, region_colors_set, r0, c0)
        if not available:
            return None
        available_per_frame.append(available)
        for combos in available.values():
            for combo in combos:
                if _combo_has_good_crossing(n, r0, c0, combo, orientation, split, region_a, region_b):
                    crossing_choices.append((idx, combo))

    if not crossing_choices:
        return None

    chosen_gaps: list[tuple[tuple[int, int], ...] | None] = [None] * len(positions)
    anchor_idx, anchor_combo = crossing_choices[int(rng.integers(len(crossing_choices)))]
    chosen_gaps[anchor_idx] = anchor_combo

    for idx, available in enumerate(available_per_frame):
        if chosen_gaps[idx] is not None:
            continue
        gap_count = _pick_gap_count(sorted(available), rng)
        combos = available[gap_count]
        chosen_gaps[idx] = combos[int(rng.integers(len(combos)))]

    assert all(combo is not None for combo in chosen_gaps)

    grid = base.copy()
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0, side), color, gaps in zip(positions, block_colors, chosen_gaps):
        gap_color = region_a if side == 0 else region_b
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, frame_color, int(color), gap_color, gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if not _rays_clear(grid, region_colors_set, seeds):
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
        uniform_out, _ = _simulate(
            grid,
            collect_stats=False,
            region_patterns_override={region_a: (0, 0, 1, 1), region_b: (0, 0, 1, 1)},
        )
    except Exception:
        return None

    if stats["beam_pixels"] < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["frame_trail_pixels"] < MIN_FRAME_TRAIL_PIXELS:
        return None
    if stats["region_switches"] < 1 or stats["beams_that_switch"] < 1:
        return None
    if stats["other_region_pixels"] < MIN_OTHER_REGION_PIXELS:
        return None
    if np.count_nonzero(out != uniform_out) < MIN_PATTERN_DIFF_PIXELS:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
