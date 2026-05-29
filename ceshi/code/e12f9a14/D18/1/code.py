# Variant of e12f9a14
# Dimension: D18 — 符号/图例编码 (symbol_instructions)
# Change: 无显式图例——所有规则从输入输出对中隐式推断... → 框架内部的颜色图案编码射线行为

import itertools

import numpy as np

from arc_utils import get_bg, paste

# Design analysis for the D18 variant
# ----------------------------------
# concretization:
# - Each object is still a 4x4 frame with 1-4 border gaps, but the 2x2 interior is no longer monochrome.
# - The interior is a two-color diagonal symbol. Grey (color 5) is shared across all blocks and acts as
#   instruction ink; the other diagonal color is the emitted beam color.
# - Grey on the NW-SE diagonal (`\`) means "turn left by 45 degrees at every collision".
# - Grey on the NE-SW diagonal (`/`) means "turn right by 45 degrees at every collision".
# - Gaps, outward launch directions, beam painting, event ordering, and geometric-cross detection stay close
#   to the original task so the D18 change is isolated to symbol decoding and collision response.
#
# behavior trace:
# - Original 90-degree collision example: an upward beam and a rightward beam hit the same cell, that cell
#   becomes blank, and both continue together along the single bisector direction NE.
# - Variant example: if the upward beam came from a `\` symbol and the rightward beam came from a `/` symbol,
#   the collision cell is still blank, but the first beam turns to NW while the second turns to SE. The
#   outgoing directions now depend on the source symbols, not only on the collision geometry.
# - Diagonal geometric crossings follow the same idea: there is no painted intersection cell, but the involved
#   beams pivot according to their own symbol rather than collapsing to one universal forward direction.
#
# complexity calibration:
# - There are only two instruction types, both visually strong and easy to compare across examples.
# - The shared grey marker color keeps the "symbol ink" role obvious, while the emitted beam color remains the
#   non-grey diagonal and therefore readable without using hidden color-id logic.
# - The generator forces both symbol types to appear and requires at least one collision, so the behavior is
#   repeatedly supported instead of hiding inside rare edge cases.
#
# hidden-assumption audit from the original main():
# - `_find_blocks_2x2` assumed the entire 2x2 interior had one color. That is no longer true, so the solver
#   must detect framed 2x2 symbols instead of monochrome blocks.
# - `_infer_frame_color(..., block_col)` assumed the interior color could simply be excluded from the border
#   scan. With bicolor symbols, border inference must ignore all interior colors and look only for a unique
#   non-background border color.
# - `spawn_beam(col, ...)` used the interior color as both identity and emission color. The variant separates
#   "beam color" from "instruction ink", so the solver must decode the beam color from the symbol pattern.
# - The original collision handlers used geometry-only redirection (`bisector` or fixed zigzag). Those lines
#   are precisely where D18 applies: the outgoing direction now depends on the symbol attached to each beam.
#
# placement / legality:
# - Frames remain separated by at least two cells and stay away from the boundary, matching the original
#   spatial regime and avoiding overlap ambiguity.
# - Gap choices are still validated against full-frame obstacles so initial rays never pass through another
#   frame. The solver therefore only needs to handle beam-beam interactions, not beam-frame penetration.
# - The generator validates every sample by running the new solver, requiring enough added beam pixels,
#   at least one collision, and a different output from the "all-left" and "all-right" counterfactual runs.
#
# visual observability:
# - Human solvers can see a direct repeated cue: the same grey diagonal orientation reappears in multiple
#   frames and consistently predicts how a beam bends after contact.
# - No ordering or color rewrite depends on numeric ids. All relevant distinctions are visible as geometry:
#   gap position sets launch direction, grey diagonal orientation sets turn sense.


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

INSTRUCTION_COLOR = 5
TURN_LEFT = -1
TURN_RIGHT = 1


def _parse_symbol(inner: np.ndarray, bg: int, frame_color: int) -> tuple[int, int] | None:
    vals = [int(v) for v in inner.ravel()]
    if any(v in (bg, frame_color) for v in vals):
        return None
    a, b, c, d = vals
    if a == INSTRUCTION_COLOR and d == INSTRUCTION_COLOR and b == c and b != INSTRUCTION_COLOR:
        return int(b), TURN_LEFT
    if b == INSTRUCTION_COLOR and c == INSTRUCTION_COLOR and a == d and a != INSTRUCTION_COLOR:
        return int(a), TURN_RIGHT
    return None


def _find_symbol_blocks(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int, int, int]] = []
    for br in range(1, h - 2):
        for bc in range(1, w - 2):
            fr0, fc0 = br - 1, bc - 1
            fr1, fc1 = br + 2, bc + 2
            border_vals: list[int] = []
            border_ok = True
            for r in range(fr0, fr1 + 1):
                for c in range(fc0, fc1 + 1):
                    if not (r in (fr0, fr1) or c in (fc0, fc1)):
                        continue
                    v = int(g[r, c])
                    if v != bg:
                        border_vals.append(v)
            if not border_vals:
                continue
            frame_colors = sorted(set(border_vals))
            if len(frame_colors) != 1:
                continue
            frame_color = frame_colors[0]
            for r in range(fr0, fr1 + 1):
                for c in range(fc0, fc1 + 1):
                    if not (r in (fr0, fr1) or c in (fc0, fc1)):
                        continue
                    if int(g[r, c]) not in (bg, frame_color):
                        border_ok = False
                        break
                if not border_ok:
                    break
            if not border_ok:
                continue
            parsed = _parse_symbol(g[br : br + 2, bc : bc + 2], bg, frame_color)
            if parsed is None:
                continue
            beam_color, turn = parsed
            blocks.append((beam_color, turn, br, bc, frame_color))
    return blocks


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


def _rotate45(d: tuple[int, int], turn: int) -> tuple[int, int]:
    return _DIRS8[(_DIR_TO_I[d] + turn) % 8]


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "turn")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...], turn: int):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.turn = int(turn)

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _solve(grid: np.ndarray, turn_override: int | None = None, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_symbol_blocks(g, bg)
    assert blocks, "expected at least one framed symbol in input"

    stats = {
        "block_count": len(blocks),
        "left_blocks": sum(1 for _col, turn, *_rest in blocks if turn == TURN_LEFT),
        "right_blocks": sum(1 for _col, turn, *_rest in blocks if turn == TURN_RIGHT),
        "gap_total": 0,
        "collisions": 0,
        "cell_collisions": 0,
        "cross_collisions": 0,
        "segment_cross_collisions": 0,
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

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...], turn: int) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern, turn)
        paint_cell(bid, pos, pattern[0])

    for beam_color, turn, br, bc, frame_color in blocks:
        for gr, gc in _frame_gaps_for_block(g, bg, frame_color, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            stats["gap_total"] += 1
            spawn_beam(beam_color, (gr, gc), (d,), turn if turn_override is None else turn_override)

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

    def retarget(gid: int, move_dir: tuple[int, int]) -> None:
        beams[gid].pattern = (_rotate45(move_dir, beams[gid].turn),)
        beams[gid].phase = 0
        beams[gid].active = True

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        b_can_continue = True
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            b_owns_cell = False
        else:
            db = cell_enter[cell]
            b_owns_cell = True

        if b_owns_cell and cell_owner.get(cell) == b:
            idx = cell_index[cell]
            if idx == 0:
                unpaint_cell(cell)
                beams[b].active = False
                b_can_continue = False
            else:
                truncate_beam(b, idx)

        blocked.add(cell)
        unpaint_cell(cell)

        group = {a}
        if b_can_continue:
            group.add(b)
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if b_can_continue and pb is not None:
            group.add(pb)

        base_dirs = {a: da, b: db}
        for gid in group:
            retarget(gid, base_dirs.get(gid, beams[gid].next_dir()))

        stats["collisions"] += 1
        stats["cell_collisions"] += 1

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
        group = {a, b}
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if pb is not None:
            group.add(pb)
        for gid in group:
            retarget(gid, prop[gid][1] if gid in prop else beams[gid].next_dir())
        stats["collisions"] += 1
        stats["cross_collisions"] += 1

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
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

        retarget(a, da)
        retarget(b, seg_dir)
        stats["collisions"] += 1
        stats["segment_cross_collisions"] += 1

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
            if cell_owner.get(nxt) == bid:
                b.active = False
                continue
            if not in_bounds(nxt) or not is_free(nxt):
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

    return (out, stats) if collect_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed 2x2 symbols; each symbol's internal diagonal marks how its beams turn on impact.

    Input structure:
    - bg is the dominant background. A shared frame color forms 4x4 borders around 2x2 bicolor symbols.
    - Inside each frame, grey cells (color 5) occupy one diagonal and the other diagonal is the beam color.
      The grey diagonal orientation is the hidden legend: `\` means left-turn, `/` means right-turn.

    Rule:
    - Every structural gap in a frame border launches a beam of the symbol's non-grey diagonal color, with the
      outward direction still determined by the gap position.
    - Beams paint background cells until they reach the boundary or hit another beam interaction.
    - If two beams contest the same cell, that cell is cleared to background and each involved beam pivots
      45 degrees according to its own symbol instruction.
    - If two diagonal segments cross geometrically, no new cell is painted there; instead the involved beams
      redirect in place using their own symbol instruction and continue.

    Key insight: The collision geometry still matters for detecting when a redirect happens, but the outgoing
    direction is no longer universal. It is encoded locally by the symbol inside the beam's source frame.
    """
    return _solve(grid)


# --------------- generator (o5x) ---------------

BLOCK_COUNT_RANGE = (2, 6)
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_BLOCKS = {2: 14, 5: 20}
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
MIN_COLLISIONS = 1

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
    beam_color: int,
    gaps: tuple[tuple[int, int], ...],
    turn: int,
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    if turn == TURN_LEFT:
        patch[1:3, 1:3] = np.array(
            [[INSTRUCTION_COLOR, beam_color], [beam_color, INSTRUCTION_COLOR]],
            dtype=int,
        )
    else:
        patch[1:3, 1:3] = np.array(
            [[beam_color, INSTRUCTION_COLOR], [INSTRUCTION_COLOR, beam_color]],
            dtype=int,
        )
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
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, INSTRUCTION_COLOR)]
    beam_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, beam_colors


def _pick_turns(rng: np.random.Generator, block_count: int) -> list[int]:
    turns = [TURN_LEFT, TURN_RIGHT]
    while len(turns) < block_count:
        turns.append(TURN_LEFT if int(rng.integers(2)) == 0 else TURN_RIGHT)
    rng.shuffle(turns)
    return turns


def _grid_size_min(block_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_BLOCKS.items()):
        if block_count >= threshold:
            result = min_n
    return result


def _observable_variant(grid: np.ndarray, out: np.ndarray, stats: dict) -> bool:
    if stats["left_blocks"] == 0 or stats["right_blocks"] == 0:
        return False
    if stats["collisions"] < MIN_COLLISIONS:
        return False
    try:
        out_left = _solve(grid, turn_override=TURN_LEFT)
        left_diff = bool(np.any(out != out_left))
    except Exception:
        left_diff = True
    try:
        out_right = _solve(grid, turn_override=TURN_RIGHT)
        right_diff = bool(np.any(out != out_right))
    except Exception:
        right_diff = True
    return left_diff and right_diff


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    block_count = int(rng.integers(*BLOCK_COUNT_RANGE))
    n = int(rng.integers(_grid_size_min(block_count), GRID_SIZE_MAX))

    bg, frame_color, beam_colors = _pick_colors(rng, block_count)
    turns = _pick_turns(rng, block_count)

    positions = _choose_positions(n, block_count, rng)
    if positions is None:
        return None

    obstacles = np.full((n, n), bg, dtype=int)
    for (r0, c0), beam_color, turn in zip(positions, beam_colors, turns):
        _draw_framed_block(obstacles, r0, c0, bg, frame_color, int(beam_color), (), int(turn))

    chosen_gaps = []
    for r0, c0 in positions:
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if not available:
            return None
        candidate_counts = [k for k in sorted(available) if k >= 2] or sorted(available)
        weights = np.array([k * k for k in candidate_counts], dtype=float)
        weights /= weights.sum()
        gap_count = int(rng.choice(candidate_counts, p=weights))
        combos = available[gap_count]
        chosen_gaps.append(combos[int(rng.integers(len(combos)))])

    grid = np.full((n, n), bg, dtype=int)
    gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), beam_color, gaps, turn in zip(positions, beam_colors, chosen_gaps, turns):
        gap_total += len(gaps)
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(beam_color), gaps, int(turn))
        for gap in gaps:
            seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    try:
        out, stats = _solve(grid, collect_stats=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if not _observable_variant(grid, out, stats):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D18 input")
