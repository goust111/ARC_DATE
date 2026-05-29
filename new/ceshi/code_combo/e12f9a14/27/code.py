# Combo variant of e12f9a14 (index 27)
# combo_spec: D12/0+D13/1
# Name: 裁剪到射线区域+框架消失

import itertools

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concretization and behavior trace:
#    The input remains the base-task scene: a square background canvas containing
#    several 4x4 frames, each wrapped around a monochrome 2x2 block, with 1-4
#    structural gaps on the frame border. Each gap launches a beam exactly as in
#    the source task, and collisions still use the same 90-degree turn rule and
#    45-degree zigzag rule.
#    The fused output changes in two linked stages:
#    - First solve the full beam physics on the original canvas while the intact
#      frames still block motion, because gap detection and collision geometry
#      depend on the original layout.
#    - Then erase every original non-background source pixel belonging to each
#      4x4 framed object, so only the emitted beam trace survives.
#    - Finally crop to the minimal bounding box of the surviving beam pixels.
#    Small trace: if a frame with a NE gap emits a red diagonal to the top edge,
#    the base task outputs the original frame plus that red ray on the full grid.
#    In this combo, the same ray is simulated, then the green border and red 2x2
#    core vanish, and the final answer keeps only the tight rectangle covering the
#    ray itself. The crop boundary is therefore inferred from the rays, not from
#    the source object.
#
# b) Complexity calibration:
#    Applying only erasure could still leave the original canvas size, and applying
#    only cropping could still leave the source frame visible. The fusion is made
#    visually strong by requiring both effects simultaneously: the source must be
#    consumed, and the remaining ray-only footprint must trim multiple sides. The
#    generator mildly biases toward fewer gaps so rays do not saturate all borders.
#
# c) Implicit-assumption audit:
#    - _find_blocks_2x2, _infer_frame_color, and _frame_gaps_for_block must read
#      the intact input, so erasure cannot happen before detection.
#    - Beam motion uses int(g[r, c]) == bg to decide free space. That premise must
#      continue to reference the input grid; otherwise erased frames would become
#      tunnels and silently change the physics.
#    - Cropping must happen after erasure, not before. D12 alone would crop around
#      frames plus rays, but this combo specification says the output should contain
#      only rays, so the bounding box must be computed on the ray-only rendering.
#    - The crop helper assumes at least one non-background output pixel. The combo
#      therefore introduces a new requirement that at least one beam cell survive;
#      the generator guarantees this with the original gap requirement plus a beam
#      pixel threshold on the erased full-size ray layer.
#
# d) Placement legality:
#    Generator placement keeps the original non-overlap constraints for 4x4 frame
#    bounding boxes, ensures every seed ray has runway through background cells,
#    and validates by executing the fused solver pipeline on the sampled grid.
#
# e) Visual observability:
#    Across 2-4 examples a human can see two stable cues: framed sources present in
#    the input are gone in the output, and the output canvas now hugs just the ray
#    pattern. Because rays keep the block colors, no hidden color remapping must be
#    inferred; only geometry and post-emission consumption matter.
#
# f) Visual cleanliness:
#    The final output removes clutter instead of adding it. Only beam trajectories
#    remain, so the crop yields compact patterns with no extra legends or markers.
#    Generator rejection of weak crops avoids examples that still look nearly full
#    canvas and would obscure the new rule.
#
# g) Change inferability:
#    No new color mapping is introduced. Colors still identify beam provenance, and
#    the only new deductions are: source objects disappear after emission, and the
#    answer is the tight ray-only rectangle.


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


def _non_bg_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    box = bbox(grid != bg)
    assert box is not None, "expected solved grid to contain non-background structure"
    return box


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    return crop_bbox(grid, _non_bg_bbox(grid, bg))


def _trim_margins(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _non_bg_bbox(grid, bg)
    h, w = grid.shape
    return r0, h - 1 - r1, c0, w - 1 - c1


def _source_mask(g: np.ndarray, bg: int, blocks: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros(g.shape, dtype=bool)
    for _col, br, bc in blocks:
        fr0, fc0 = br - 1, bc - 1
        for r in range(fr0, fr0 + 4):
            for c in range(fc0, fc0 + 4):
                if int(g[r, c]) != bg:
                    mask[r, c] = True
    return mask


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


def _solve_ray_overlay(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"
    src_mask = _source_mask(g, bg, blocks)

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
        beam = beams[bid]
        assert 0 < keep_len < len(beam.path), "expected truncate to remove at least one cell"
        for cell in beam.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        beam.path = beam.path[:keep_len]
        beam.enter_dirs = beam.enter_dirs[:keep_len]
        beam.pos = beam.path[-1]
        beam.active = True

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
        beam = beams.get(bid)
        if beam is None or not beam.active:
            return None
        d = beam.next_dir()
        partner: int | None = None
        for oid, other in beams.items():
            if oid == bid or not other.active:
                continue
            if other.pattern != beam.pattern or other.phase != beam.phase:
                continue
            if other.next_dir() != d:
                continue
            if abs(other.pos[0] - beam.pos[0]) + abs(other.pos[1] - beam.pos[1]) != 1:
                continue
            if (beam.pos[0] + d[0], beam.pos[1] + d[1]) == other.pos:
                continue
            if (other.pos[0] + d[0], other.pos[1] + d[1]) == beam.pos:
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
                "two diagonal beams must cross geometrically (odd parity), not collide at same cell"
            )
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
                truncate_beam(bid_v, cell_index[cell])
                unpaint_cell(cell)
            if cell in cell_owner and cell_owner[cell] == bid_u:
                truncate_beam(bid_u, cell_index[cell] + 1)
            else:
                beam_u = beams[bid_u]
                beam_u.path.append(cell)
                beam_u.enter_dirs.append(u)
                beam_u.pos = cell
                paint_cell(bid_u, cell, u)
        else:
            idx = cell_index[cell]
            if idx + 1 < len(beams[bid_u].path):
                truncate_beam(bid_u, idx + 1)

        beam_v = beams[bid_v]
        adj = (beam_v.pos[0] + u[0], beam_v.pos[1] + u[1])
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
        beam_v.path.append(adj)
        beam_v.enter_dirs.append(u)
        beam_v.pos = adj
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
        owner_q = cell_owner.get(q)
        if owner_q is not None and cell_index.get(q, 0) > 0 and cell_enter.get(q) == (dr, dc):
            return owner_q
        owner_p = cell_owner.get(p)
        if owner_p is not None and cell_index.get(p, 0) > 0 and cell_enter.get(p) == (-dr, -dc):
            return owner_p
        return None

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        db = prop[b][1]
        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        beams[a].pattern = (nd,)
        beams[a].phase = 0
        beams[b].pattern = (nd,)
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
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (
            n2[0] - p2[0],
            n2[1] - p2[1],
        ):
            seg_dir = (n2[0] - p2[0], n2[1] - p2[1])
            truncate_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (
            p2[0] - n2[0],
            p2[1] - n2[1],
        ):
            seg_dir = (p2[0] - n2[0], p2[1] - n2[1])
            truncate_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        nd = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert nd != (0, 0), "expected segment crossing to resolve to a forward direction"

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit block/frame"
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
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            other_bid = move_map.get((p2, n2))
            if other_bid is None or other_bid == bid:
                continue
            cross_pairs.add(tuple(sorted((bid, other_bid))))

        seg_cross: set[tuple[int, int]] = set()
        for bid, (_nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            pr, pc = beams[bid].pos
            dr, dc = d
            p2 = (pr, pc + dc)
            n2 = (pr + dr, pc)
            other_bid = diag_segment_owner(p2, n2)
            if other_bid is None or other_bid == bid:
                continue
            seg_cross.add((bid, other_bid))

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
            beam = beams[bid]
            beam.path.append(nxt)
            beam.enter_dirs.append(d)
            beam.pos = nxt
            paint_cell(bid, nxt, d)
            beam.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    out[src_mask] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate the frame-launched beam system, consume the launching frames, and return only the compact ray footprint.

    Input structure:
    - The input is the same square scene family as the base task: a dominant background, several shared-color 4x4 frames, and a solid 2x2 colored block inside each frame.
    - Each frame has one or more structural border gaps. Those gaps are the launch sites that determine the outward beam directions.

    Rule:
    - Reconstruct the beam propagation and collision behavior on the full input canvas exactly as in the original task, using the intact input frames as obstacles while beams travel.
    - After the rays settle, erase every original non-background source pixel from each 4x4 framed object, so the frame border and inner 2x2 block disappear but the painted gap seed and later beam cells remain.
    - Then remove every all-background outer row and column, returning the smallest rectangle that contains the remaining ray pattern and nothing else.

    Key insight:
    - The answer is not a full solved overlay. You must first infer the original beam physics, then realize the source objects are consumed, and only after that infer the crop boundary from the surviving rays alone.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    ray_overlay = _solve_ray_overlay(g)
    return _tight_crop(ray_overlay, bg)


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
MIN_TRIMMED_SIDES = 2
MIN_CROP_AREA_REDUCTION = 8
GEN_RETRIES = 300

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}
_GAP_COUNT_WEIGHTS = {1: 5, 2: 3, 3: 2, 4: 1}


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


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _choose_gap_pattern(
    available: dict[int, list[tuple[tuple[int, int], ...]]],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    counts = sorted(available)
    weights = np.array([_GAP_COUNT_WEIGHTS.get(count, 1) for count in counts], dtype=float)
    weights /= weights.sum()
    gap_count = int(rng.choice(counts, p=weights))
    combos = available[gap_count]
    return combos[int(rng.integers(len(combos)))]


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
        chosen_gaps.append(_choose_gap_pattern(available, rng))

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

    try:
        ray_overlay = _solve_ray_overlay(grid)
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(ray_overlay != bg) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None

    blocks = _find_blocks_2x2(grid, bg)
    src_mask = _source_mask(grid, bg, blocks)
    if not np.all(ray_overlay[src_mask] == bg):
        return None

    trimmed = _trim_margins(ray_overlay, bg)
    trimmed_sides = sum(margin > 0 for margin in trimmed)
    area_reduction = grid.size - out.size
    if trimmed_sides < MIN_TRIMMED_SIDES:
        return None
    if area_reduction < MIN_CROP_AREA_REDUCTION:
        return None
    if out.shape == grid.shape:
        return None
    if not np.array_equal(out, _tight_crop(ray_overlay, bg)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
