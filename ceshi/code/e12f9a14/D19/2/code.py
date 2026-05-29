# Variant of e12f9a14
# Dimension: D19 — 射线碰撞后的转向方向 (path)
# Change: 碰撞后两条射线各转45度——具体转向方向由碰撞的几何关系决定... → 碰撞后沿墙壁方向转弯

import itertools

import numpy as np

from arc_utils import get_bg, paste


# Design analysis:
# variant:
#   core_change:
#     original: 90-degree collisions redirected both beams to the bisector; 45-degree
#       collisions created a diagonal+cardinal zigzag. Collision geometry picked the new
#       path family.
#     new: after a collision, a beam continues along the other beam's direction. For
#       active-active collisions this is an exact direction swap; for a moving beam
#       entering an already-painted ray, the newcomer follows the encountered ray's local
#       path direction. This is the visible D19 change.
#   concretization:
#     objects: keep the original framed 2x2 blocks, shared frame color, background choices,
#       and beam colors. A frame gap still determines the initial outward cardinal or
#       diagonal direction.
#     pixel_effect:
#       - same-cell collision: both beams reach the meeting cell, the cell stays blank, and
#         both continue from that junction using each other's incoming direction.
#       - diagonal geometric crossing: neither beam paints a new collision cell; they swap
#         diagonals and continue along the opposite arm of the X.
#       - moving into an older painted ray: the newcomer merges into that ray and follows
#         the encountered local path direction, producing the "takeover" effect.
#     tracked_example:
#       - original: two inward diagonals from top-left and top-right cross, then both turn
#         into a shared downward cardinal direction.
#       - variant: the same two diagonals exchange directions at the crossing, so the left
#         ray continues down-left and the right ray continues down-right. The post-collision
#         shape is a bounce / split instead of a bisector funnel.
#   complexity_calibration:
#     solver: keep the original beam simulator, cell ownership tables, geometric crossing
#       detection, and frame parsing. Only collision resolution semantics are replaced.
#     generator: tighten block count to one decisive collision pair. This is a deliberate
#       observability adjustment: D19 is about post-collision routing, so every generated
#       sample should contain a collision whose result differs strongly from the original
#       task. Using exactly two blocks avoids distractors and keeps the new rule legible.
#   implicit_assumptions_review:
#     - original resolve_cell_collision relied on angle classes {45, 90}; that assumption
#       is removed. The variant accepts any pair of directions and swaps them instead.
#     - original code used pattern lengths >1 for zigzags and partner propagation for
#       thick bands. Those structures no longer carry meaning; beams now always use a
#       single-direction pattern after collisions.
#     - when following an already-painted beam, using only the stored entering direction is
#       insufficient if the old beam later turned. The variant therefore derives the local
#       outgoing direction from the owning beam's path when available.
#   placement_legality:
#     generator places two 4x4 frames with corner gaps chosen so their diagonals cross
#       geometrically. Frame bounding boxes still keep >=2 cells of clearance and remain
#       inside the grid. The pair orientation is randomly rotated so the dataset is not
#       locked to one absolute direction.
#   visual_observability:
#     each sample contains one guaranteed inward diagonal crossing. Under the original rule
#       this would become a bisector line; under the variant it becomes a bounce into the
#       opposite diagonals. The difference is visible from shape alone, with no use of
#       hidden color ordering.
#   neatness:
#     only two beams participate, so the output stays sparse and readable. The collision is
#       centered enough that there are visible pre-collision and post-collision segments.
#   color_reasoning:
#     colors are unchanged by the rule. Each beam keeps its source block color before and
#       after the collision, so the viewer can track which ray inherited which direction.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}

_PAIR_LAYOUTS = {
    "down": (((3, 3),), ((3, 0),)),
    "up": (((0, 3),), ((0, 0),)),
    "right": (((3, 3),), ((0, 3),)),
    "left": (((3, 0),), ((0, 0),)),
}


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored beams from framed 2x2 blocks; collisions hand off direction instead of forming bisectors.

    Input structure:
    - The background is the most common color. Each monochrome 2x2 block sits inside a shared-color 4x4 frame.
    - Missing border cells on a frame are beam exits; corner gaps launch diagonal beams and edge gaps launch cardinal beams.

    Rule:
    - Each gap paints a beam in the block's color, moving outward until it reaches the border, a blocked collision node, or another beam.
    - When two moving beams collide, each continues using the other beam's incoming direction.
    - When a moving beam enters an already-painted ray, it merges onto that ray and follows the encountered local path direction.
    - Same-cell collision nodes stay blank; geometric diagonal crossings swap the diagonals without adding a painted junction cell.

    Key insight: Collision cells no longer choose a new direction from geometry classes; they act like direction handoff points where motion transfers onto the other ray's path.
    """
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

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    def set_dir(bid: int, d: tuple[int, int]) -> None:
        beams[bid].pattern = (d,)
        beams[bid].phase = 0
        beams[bid].active = True

    def append_virtual_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        b = beams[bid]
        if b.pos == cell:
            return
        b.path.append(cell)
        b.enter_dirs.append(enter_dir)
        b.pos = cell
        b.active = True

    def beam_dir_through_cell(bid: int, cell: tuple[int, int]) -> tuple[int, int]:
        assert cell_owner.get(cell) == bid, "expected queried path cell to belong to the given beam"
        idx = cell_index[cell]
        b = beams[bid]
        if idx + 1 < len(b.path):
            nr, nc = b.path[idx + 1]
            return _normalize_dir(nr - cell[0], nc - cell[1])
        if b.pos == cell and b.active:
            return b.next_dir()
        return cell_enter[cell]

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

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            blocked.add(cell)
            if cell_owner.get(cell) in {a, b}:
                unpaint_cell(cell)
            append_virtual_cell(a, cell, da)
            append_virtual_cell(b, cell, db)
            set_dir(a, db)
            set_dir(b, da)
            return

        assert cell_owner.get(cell) == b, "expected occupied-cell collision partner to own the cell"
        db = beam_dir_through_cell(b, cell)
        append_virtual_cell(a, cell, da)
        set_dir(a, db)

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
        set_dir(a, db)
        set_dir(b, da)

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

        assert seg_dir is not None
        set_dir(a, seg_dir)
        set_dir(b, da)

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
                assert not in_bounds(nxt) or nxt in blocked, "beam must not hit block/frame"
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            break

        cross_pairs: set[tuple[int, int]] = set()
        box_map: dict[tuple[int, int, int, int], dict[int, int]] = {}
        for bid, (nxt, d) in prop.items():
            if abs(d[0]) + abs(d[1]) != 2:
                continue
            p = beams[bid].pos
            key = (
                min(p[0], nxt[0]),
                min(p[1], nxt[1]),
                max(p[0], nxt[0]),
                max(p[1], nxt[1]),
            )
            orient = 1 if (nxt[0] - p[0]) == (nxt[1] - p[1]) else -1
            prev = box_map.setdefault(key, {})
            if -orient in prev and prev[-orient] != bid:
                cross_pairs.add(tuple(sorted((bid, prev[-orient]))))
            prev[orient] = bid

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

    return out


# --------------- generator (o5x) ---------------

BLOCK_COUNT = 2
GRID_SIZE_RANGE = (16, 30)
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
MIN_BEAM_PIXELS = 8
GEN_RETRIES = 300


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


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


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=BLOCK_COUNT, replace=False))
    return bg, frame_color, block_colors


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _choose_collision_pair(
    n: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], ...]]] | None:
    orient = str(rng.choice(sorted(_PAIR_LAYOUTS)))

    if orient in {"down", "up"}:
        deltas = [d for d in range(6, n - 7, 2)]
        if not deltas:
            return None
        delta = int(rng.choice(deltas))
        c_max = n - 6 - delta
        if c_max < PLACEMENT_MARGIN:
            return None
        c_left = int(rng.integers(PLACEMENT_MARGIN, c_max + 1))
        c_right = c_left + delta
        if orient == "down":
            r_min = PLACEMENT_MARGIN
            r_max = min(5, n - 11)
        else:
            r_min = max(PLACEMENT_MARGIN, n - 10)
            r_max = n - 6
        if r_max < r_min:
            return None
        r0 = int(rng.integers(r_min, r_max + 1))
        positions = [(r0, c_left), (r0, c_right)]
    else:
        deltas = [d for d in range(6, n - 7, 2)]
        if not deltas:
            return None
        delta = int(rng.choice(deltas))
        r_max = n - 6 - delta
        if r_max < PLACEMENT_MARGIN:
            return None
        r_top = int(rng.integers(PLACEMENT_MARGIN, r_max + 1))
        r_bottom = r_top + delta
        if orient == "right":
            c_min = PLACEMENT_MARGIN
            c_max = min(5, n - 11)
        else:
            c_min = max(PLACEMENT_MARGIN, n - 10)
            c_max = n - 6
        if c_max < c_min:
            return None
        c0 = int(rng.integers(c_min, c_max + 1))
        positions = [(r_top, c0), (r_bottom, c0)]

    assert _frames_separated(*positions), "collision pair must preserve frame clearance"
    gaps = list(_PAIR_LAYOUTS[orient])
    return positions, gaps


def _count_diagonal_crossings(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    blocks = _find_blocks_2x2(g, bg)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            if abs(d[0]) + abs(d[1]) == 2:
                seeds.append(((gr, gc), d))

    count = 0
    for (p0, d0), (p1, d1) in itertools.combinations(seeds, 2):
        if p0[0] == p1[0]:
            if d0[0] != d1[0] or d0[1] == d1[1]:
                continue
            sep = abs(p0[1] - p1[1])
            if sep >= 3 and sep % 2 == 1:
                count += 1
            continue
        if p0[1] == p1[1]:
            if d0[1] != d1[1] or d0[0] == d1[0]:
                continue
            sep = abs(p0[0] - p1[0])
            if sep >= 3 and sep % 2 == 1:
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))

    bg, frame_color, block_colors = _pick_colors(rng)
    chosen = _choose_collision_pair(n, rng)
    if chosen is None:
        return None
    positions, chosen_gaps = chosen

    grid = np.full((n, n), bg, dtype=int)
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), color, gaps in zip(positions, block_colors, chosen_gaps):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None
    if _count_diagonal_crossings(grid) < 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < MIN_BEAM_PIXELS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D19 input")
