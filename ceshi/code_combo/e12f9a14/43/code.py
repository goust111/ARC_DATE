# Combo variant of e12f9a14 (index 43)
# combo_spec: D4/2+D7/2
# Name: 可变大小色块+可变大小框架

from collections import Counter
from functools import lru_cache

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   source_objects:
#     original: a solid 2x2 color block touches a one-cell-thick 4x4 frame, so source size and frame size are tied.
#     new: every source is a solid kxk block with k in {1,2,3}, enclosed by a square frame of side f in {3,4,5,6}.
#       The frame stays one cell thick, but the block no longer has to touch it; any extra cavity is deterministic
#       interior padding. This is necessary so block size and frame size can vary independently instead of collapsing
#       to the fixed relation f = k + 2.
#   concrete_trace:
#     - Base task: a 2x2 block inside a 4x4 frame emits a one-cell beam that keeps going until edge/collision.
#     - Combo task: take two inputs with the same 1x1 red block but put one in a 3x3 frame and one in a 5x5 frame.
#       Both emit the same color and direction, but the 5x5 frame's beam survives longer because frame size sets
#       the travel cap. Now compare a 1x1 block and a 3x3 block inside equally sized 5x5 frames: they stop after the
#       same number of centerline steps, but the 3x3 block paints a visibly wider band because block size sets width.
#     - Difference: the solver must recover two visible measurements from the input and route them to two different
#       beam attributes: frame side -> length cap, block side -> rendered width.
#   new_size_mapping:
#     - block size k maps to rendered beam width = k cells.
#     - frame size f maps to centerline travel cap = 2*f - 3 forward steps after the gap seed.
#       Examples: 3x3 -> 3 steps, 4x4 -> 5, 5x5 -> 7, 6x6 -> 9.
#   complexity_calibration:
#     - Only three block widths and four frame sizes are used, so the rule is visible but still learnable.
#     - Every generated grid contains at least two block sizes and at least two frame sizes.
#     - Generation also guarantees cross-support: one block size appears in two frame sizes, and one frame size
#       appears with two block sizes, so a human can separate the two mappings from a small number of examples.
#   implicit_assumption_audit:
#     - Original _find_blocks_2x2 assumed all sources are exactly 2x2. Replaced with connected-component discovery
#       of solid monochrome squares of size 1..3.
#     - Original frame inference assumed the block directly touches the inner side of a 4x4 border. Replaced with
#       an explicit search over legal frame sizes and deterministic cavity padding.
#     - Original gap scanning was hard-coded for size 4. Replaced with frame-size-generic border enumeration.
#     - Original beam painting treated beam width as always one cell. The collision engine still runs on a one-cell
#       centerline, but the final rendering widens each path segment according to block size.
#     - New assumption introduced by the combo: every non-frame colored component belongs to exactly one legal frame
#       under the padding convention. The generator enforces this by placing full frames first and keeping interiors
#       otherwise background-only.
#   placement_legality:
#     - Frames are placed by true outer bbox with a fixed clearance gap, never touching the grid edge.
#     - Each frame is first stamped closed on an obstacle grid; only gap positions with enough straight runway are
#       considered. This ensures beams do not crash into other frames immediately.
#     - The generator keeps just one gap per frame. This makes the width/length effects visually clean while still
#       using the original collision-capable engine.
#   visual_observability:
#     - Block size is readable directly from the colored square in the frame interior.
#     - Frame size is readable directly from the outer border, including the extra empty cavity on some examples.
#     - Width and length differences show up independently in the output, so the fused rule is not hidden in color ids
#       or ordering.
#   cleanliness:
#     - One gap per source, larger object spacing, and rejection of color-overlap in widened rendering keep outputs
#       legible. The result looks like separated colored bands rather than tangled paint.
#   non_degeneracy_check:
#     - On generated samples, the combo must differ from the base task in two measurable ways:
#       widened beams for k > 1 and capped beams whenever runway exceeds 2*f - 3.
#     - The diagnostic script in the workflow checks both that samples contain independent size evidence and that
#       combo outputs differ from a narrow unbounded baseline on the same inputs.
#   new_solver_assumptions:
#     - Frame color is the most common non-background color; generator uses multiple shared frames so this holds.
#     - Non-frame, non-background components are solid squares and the rest of each frame interior is background.
#     - The block sits at deterministic top-left-biased centered padding within the frame cavity:
#       offset = floor((f - 2 - k) / 2) from the frame's inner top-left corner.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

BLOCK_SIZE_VALUES = (1, 2, 3)
FRAME_SIZE_VALUES = (3, 4, 5, 6)


def _beam_cap(frame_size: int) -> int:
    return 2 * int(frame_size) - 3


def _inner_offset(frame_size: int, block_size: int) -> int:
    return (int(frame_size) - 2 - int(block_size)) // 2


def _infer_frame_color(g: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in g.ravel() if int(v) != bg]
    assert vals, "expected at least one non-background color"
    return Counter(vals).most_common(1)[0][0]


def _find_blocks(g: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    seen: set[tuple[int, int]] = set()
    blocks: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v == bg or v == frame_color:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen or int(g[nr, nc]) != v:
                        continue
                    seen.add((nr, nc))
                    stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            size = r1 - r0 + 1
            assert size == c1 - c0 + 1, "expected square inner block"
            assert size in BLOCK_SIZE_VALUES, "expected allowed block size"
            assert len(cells) == size * size, "expected filled square block"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == v), "expected solid block"
            blocks.append((v, r0, c0, size))
    blocks.sort(key=lambda x: (x[1], x[2], x[0]))
    assert blocks, "expected at least one source block"
    return blocks


def _frame_gaps_for_frame(
    g: np.ndarray, bg: int, wall: int, fr0: int, fc0: int, frame_size: int
) -> list[tuple[int, int]]:
    fr1 = fr0 + frame_size - 1
    fc1 = fc0 + frame_size - 1
    out: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) != bg:
                continue
            dr = -1 if r == fr0 else (1 if r == fr1 else 0)
            dc = -1 if c == fc0 else (1 if c == fc1 else 0)
            assert dr != 0 or dc != 0, "expected candidate gap on border"
            if dr == 0:
                ok = int(g[r - 1, c]) == wall and int(g[r + 1, c]) == wall
            elif dc == 0:
                ok = int(g[r, c - 1]) == wall and int(g[r, c + 1]) == wall
            else:
                ok = int(g[r - dr, c]) == wall and int(g[r, c - dc]) == wall
            if ok:
                out.append((r, c))
    return out


def _match_frame(
    g: np.ndarray, bg: int, frame_color: int, block: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    color, br, bc, block_size = block
    matches: list[tuple[int, int, int]] = []
    for frame_size in FRAME_SIZE_VALUES:
        if frame_size < block_size + 2:
            continue
        offset = _inner_offset(frame_size, block_size)
        fr0 = br - 1 - offset
        fc0 = bc - 1 - offset
        fr1 = fr0 + frame_size - 1
        fc1 = fc0 + frame_size - 1
        if fr0 < 0 or fc0 < 0 or fr1 >= g.shape[0] or fc1 >= g.shape[1]:
            continue
        exp_br = fr0 + 1 + offset
        exp_bc = fc0 + 1 + offset
        if (exp_br, exp_bc) != (br, bc):
            continue
        ok = True
        for r in range(fr0, fr1 + 1):
            for c in range(fc0, fc1 + 1):
                v = int(g[r, c])
                on_border = r in (fr0, fr1) or c in (fc0, fc1)
                in_block = exp_br <= r < exp_br + block_size and exp_bc <= c < exp_bc + block_size
                if in_block:
                    if v != color:
                        ok = False
                        break
                elif on_border:
                    if v not in (bg, frame_color):
                        ok = False
                        break
                else:
                    if v != bg:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue
        gaps = _frame_gaps_for_frame(g, bg, frame_color, fr0, fc0, frame_size)
        if not gaps:
            continue
        matches.append((fr0, fc0, frame_size))
    assert len(matches) == 1, "expected a unique enclosing frame for each block"
    return matches[0]


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


@lru_cache(maxsize=None)
def _width_offsets(width: int) -> tuple[int, ...]:
    start = -(int(width) // 2)
    return tuple(start + i for i in range(int(width)))


def _brush_cells(cell: tuple[int, int], move_dir: tuple[int, int], width: int) -> list[tuple[int, int]]:
    r, c = cell
    dr, dc = move_dir
    tr, tc = -dc, dr
    return [(r + off * tr, c + off * tc) for off in _width_offsets(width)]


class _Beam:
    __slots__ = ("bid", "color", "width", "pos", "pattern", "phase", "path", "enter_dirs", "active", "max_steps")

    def __init__(
        self,
        bid: int,
        color: int,
        width: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        max_steps: int,
    ):
        self.bid = bid
        self.color = int(color)
        self.width = int(width)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.max_steps = int(max_steps)

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)

    def can_advance(self) -> bool:
        return len(self.path) - 1 < self.max_steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit collision-aware beams from variable-sized framed blocks; frame size controls range and block size controls width.

    Input structure:
    - The background is the most common color. A shared frame color forms one-cell-thick square borders of side 3..6.
    - Each frame encloses one monochrome square block of side 1..3, possibly with extra background padding inside the frame.
    - Every frame has one or more single-cell border gaps, and each non-frame colored component is the unique block for one frame.

    Rule:
    - Each gap launches a beam in the outward direction implied by its border position. The beam keeps the original collision
      mechanics, but its centerline can advance only 2*f-3 steps, where f is that frame's side length.
    - After centerlines are simulated, every visited beam cell is widened into a band whose thickness equals the source block size.
    - 90-degree collisions still leave a blank blocked cell and redirect both beams to the bisector direction.
    - 45-degree collisions still splice a diagonal and a cardinal beam into the same alternating zigzag continuation.

    Key insight:
    - Two independent measurements must be extracted from the input. The outer frame size determines how long a ray can persist,
      while the inner block size determines how thick the painted band becomes, even when two rays share the same color-free geometry.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    frame_color = _infer_frame_color(g, bg)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks(g, bg, frame_color)
    framed_blocks: list[tuple[int, int, int, int, int, int, int]] = []
    for block in blocks:
        fr0, fc0, frame_size = _match_frame(g, bg, frame_color, block)
        color, br, bc, block_size = block
        framed_blocks.append((color, br, bc, block_size, fr0, fc0, frame_size))
        assert frame_size in FRAME_SIZE_VALUES, "expected allowed frame size"

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
        width: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
        max_steps: int,
    ) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, width, pos, pattern, max_steps)
        paint_cell(bid, pos, pattern[0])

    for color, br, bc, block_size, fr0, fc0, frame_size in framed_blocks:
        for gr, gc in _frame_gaps_for_frame(g, bg, frame_color, fr0, fc0, frame_size):
            fr1 = fr0 + frame_size - 1
            fc1 = fc0 + frame_size - 1
            dr = -1 if gr == fr0 else (1 if gr == fr1 else 0)
            dc = -1 if gc == fc0 else (1 if gc == fc1 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected outward-facing gap"
            spawn_beam(color, block_size, (gr, gc), (d,), _beam_cap(frame_size))

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
            if ob.next_dir() != d:
                continue
            if abs(ob.pos[0] - b.pos[0]) + abs(ob.pos[1] - b.pos[1]) != 1:
                continue
            if (b.pos[0] + d[0], b.pos[1] + d[1]) == ob.pos:
                continue
            if (ob.pos[0] + d[0], ob.pos[1] + d[1]) == b.pos:
                continue
            assert partner is None, "expected at most one lockstep partner"
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams should cross geometrically, not share a cell"
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
        assert in_bounds(adj) and is_free(adj) and cell_owner.get(adj) is None, "expected free zigzag-adjacent cell"
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
        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected crossing diagonals to resolve forward"
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
        assert nd != (0, 0), "expected segment crossing to resolve forward"
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
        active = [bid for bid, b in beams.items() if b.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            if not b.can_advance():
                b.active = False
                continue
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
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

    final = g.copy()
    for bid in sorted(beams):
        beam = beams[bid]
        for cell, enter_dir in zip(beam.path, beam.enter_dirs):
            for rr, cc in _brush_cells(cell, enter_dir, beam.width):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in blocked or int(g[rr, cc]) != bg:
                    continue
                final[rr, cc] = beam.color
    return final


# --------------- generator ---------------

BLOCK_COUNT_RANGE = (3, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {3: 18, 4: 22, 5: 25}
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 3
PLACEMENT_MARGIN = 3
MIN_RAY_STEPS = 3
RAY_HEADROOM = 1
MIN_BEAM_PIXELS_PER_GAP = 3
GEN_RETRIES = 500


def _grid_size_min(block_count: int) -> int:
    result = 16
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _allowed_frame_sizes(block_size: int) -> tuple[int, ...]:
    return tuple(f for f in FRAME_SIZE_VALUES if f >= block_size + 2)


def _allowed_block_sizes(frame_size: int) -> tuple[int, ...]:
    return tuple(k for k in BLOCK_SIZE_VALUES if k <= frame_size - 2)


def _pick_object_specs(block_count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    shared_block = int(rng.choice(BLOCK_SIZE_VALUES))
    block_frames = list(_allowed_frame_sizes(shared_block))
    if len(block_frames) < 2:
        raise AssertionError("expected at least two frame sizes for shared block")
    rng.shuffle(block_frames)
    f1, f2 = sorted(block_frames[:2])

    shared_frame_candidates = [f for f in _allowed_frame_sizes(shared_block) if len(_allowed_block_sizes(f)) >= 2]
    shared_frame = int(rng.choice(shared_frame_candidates))
    other_blocks = [k for k in _allowed_block_sizes(shared_frame) if k != shared_block]
    other_block = int(rng.choice(other_blocks))

    specs = [(shared_block, f1), (shared_block, f2), (other_block, shared_frame)]

    support_blocks = sorted({shared_block, other_block})
    support_frames = sorted({f1, f2, shared_frame})
    pool = [
        (k, f)
        for k in support_blocks
        for f in support_frames
        if f >= k + 2
    ]
    while len(specs) < block_count:
        specs.append(pool[int(rng.integers(len(pool)))])
    rng.shuffle(specs)
    assert len({k for k, _f in specs}) >= 2, "expected multiple block sizes"
    assert len({f for _k, f in specs}) >= 2, "expected multiple frame sizes"
    return specs


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


@lru_cache(maxsize=None)
def _border_order(frame_size: int) -> tuple[tuple[int, int], ...]:
    s = int(frame_size)
    pts: list[tuple[int, int]] = []
    for c in range(s):
        pts.append((0, c))
    for r in range(1, s - 1):
        pts.append((r, s - 1))
    for c in range(s - 1, -1, -1):
        pts.append((s - 1, c))
    for r in range(s - 2, 0, -1):
        pts.append((r, 0))
    return tuple(pts)


def _gap_dir(frame_size: int, gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    last = frame_size - 1
    dr = -1 if gr == 0 else (1 if gr == last else 0)
    dc = -1 if gc == 0 else (1 if gc == last else 0)
    return dr, dc


def _frames_separated(a: tuple[int, int], sa: int, b: tuple[int, int], sb: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + sa - 1, ac0 + sa - 1
    br1, bc1 = br0 + sb - 1, bc0 + sb - 1
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_positions(
    n: int, specs: list[tuple[int, int]], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(specs)), key=lambda i: specs[i][1], reverse=True)
    selected: dict[int, tuple[int, int]] = {}
    placed: list[tuple[tuple[int, int], int]] = []
    for idx in order:
        _block_size, frame_size = specs[idx]
        coords = [
            (r, c)
            for r in range(PLACEMENT_MARGIN, n - PLACEMENT_MARGIN - frame_size + 1)
            for c in range(PLACEMENT_MARGIN, n - PLACEMENT_MARGIN - frame_size + 1)
        ]
        rng.shuffle(coords)
        found = None
        for pos in coords:
            if all(_frames_separated(pos, frame_size, other_pos, other_size) for other_pos, other_size in placed):
                found = pos
                break
        if found is None:
            return None
        selected[idx] = found
        placed.append((found, frame_size))
    return [selected[i] for i in range(len(specs))]


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    block_size: int,
    frame_size: int,
    gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((frame_size, frame_size), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    offset = _inner_offset(frame_size, block_size)
    br = 1 + offset
    bc = 1 + offset
    patch[br : br + block_size, bc : bc + block_size] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _ray_runway(grid: np.ndarray, bg: int, seed: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = grid.shape
    r, c = seed
    dr, dc = direction
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg:
        steps += 1
        nr += dr
        nc += dc
    return steps


def _valid_single_gap_positions(
    obstacles: np.ndarray, bg: int, r0: int, c0: int, frame_size: int
) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    cap = _beam_cap(frame_size)
    for gap in _border_order(frame_size):
        runway = _ray_runway(obstacles, bg, (r0 + gap[0], c0 + gap[1]), _gap_dir(frame_size, gap))
        if runway < max(MIN_RAY_STEPS, cap + RAY_HEADROOM):
            continue
        valid.append(gap)
    return valid


def _combo_evidence(specs: list[tuple[int, int]]) -> bool:
    by_block: dict[int, set[int]] = {}
    by_frame: dict[int, set[int]] = {}
    for block_size, frame_size in specs:
        by_block.setdefault(block_size, set()).add(frame_size)
        by_frame.setdefault(frame_size, set()).add(block_size)
    return any(len(v) >= 2 for v in by_block.values()) and any(len(v) >= 2 for v in by_frame.values())


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    specs = _pick_object_specs(block_count, rng)
    if not _combo_evidence(specs):
        return None
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, block_colors = _pick_colors(rng, block_count)
    positions = _choose_positions(n, specs, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), color, (block_size, frame_size) in zip(positions, block_colors, specs):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(color), block_size, frame_size, ())

    chosen_gaps: list[tuple[tuple[int, int], ...]] = []
    for (r0, c0), (_block_size, frame_size) in zip(positions, specs):
        valid = _valid_single_gap_positions(obstacles, bg, r0, c0, frame_size)
        if not valid:
            return None
        gap = valid[int(rng.integers(len(valid)))]
        chosen_gaps.append((gap,))

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    for (r0, c0), color, (block_size, frame_size), gaps in zip(positions, block_colors, specs, chosen_gaps):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), block_size, frame_size, gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if changed == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
