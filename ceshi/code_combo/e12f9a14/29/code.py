# Combo variant of e12f9a14 (index 29)
# combo_spec: D15/2+D18/1
# Name: 干扰项框架+颜色图案编码行为

"""
design_analysis:
  concretization_and_trace:
    object_family:
      common_shell:
        shape: every deliberate object is still a 4x4 border of one shared wall color
        spacing: all 4x4 boxes remain separated by at least two blank cells
      true_source:
        interior: a 2x2 two-color diagonal symbol
        symbol_code: grey color 5 marks the instruction diagonal; the other diagonal color is the beam color
        launch_condition: the frame must have 1-4 structural gaps
      distractor_hollow_open:
        interior: 2x2 background
        border: 1-4 structural gaps, so it visually looks launch-ready
        effect: no beams because there is no valid symbol to decode
      distractor_shattered_symbol:
        interior: same valid diagonal symbol as a true source
        border: 5-6 structural gaps
        effect: inert because the fusion rule only treats moderately open symbol frames as sources
    behavior_trace:
      old_task_step:
        - find a monochrome 2x2 core
        - infer its frame and launch from every structural gap
        - collisions are resolved by geometry alone
      fused_task_step:
        - scan 4x4 frame candidates first, because some open frames are hollow and some symbol frames are too broken
        - keep only frames whose interior is a valid diagonal symbol and whose gap count is between 1 and 4
        - decode beam color and turn sense from that symbol, then launch
        - when a collision event happens, each involved beam rotates 45 degrees according to its own source symbol
      concrete_example:
        - frame A has a `\\` grey diagonal and 3 gaps, so it is active and all its beams turn left after impact
        - frame B has a `/` grey diagonal and 2 gaps, so it is active and all its beams turn right after impact
        - frame C is hollow with 2 gaps, so it looks open but remains unchanged
        - frame D has the same symbol style as A or B but 5 gaps, so it is rejected as over-open and also remains unchanged
  complexity_check:
    - Only two instruction types are used: left-turn and right-turn. This keeps the symbolic branch small enough to infer.
    - Distractors reuse the same 4x4 visual template instead of introducing new shapes, so the extra difficulty comes from filtering, not from object proliferation.
    - At least one hollow-open distractor and one shattered-symbol distractor are generated in each sample, so both filtering rules matter behaviorally.
  implicit_assumptions_review:
    - The original `_find_blocks_2x2` assumption is invalid because interiors are no longer monochrome and some frame-like objects are empty.
    - The original source-building loop assumed that "colored interior implies emitter". That breaks here because a valid symbol with 5-6 gaps is a distractor.
    - The D18-style collision logic assumes each beam carries a persistent instruction. The generator therefore must ensure every emitted beam comes from a decodable symbol.
    - Frame scanning assumes every border background cell on a deliberate object is a true structural gap. The generator enforces this by drawing only legal gap patterns.
  placement_legality:
    - Objects are placed on disjoint 4x4 bounding boxes with a two-cell moat.
    - Clearance checks are run against closed-frame obstacles, including distractors, so real emitted beams never tunnel through another object after gaps are opened.
    - The shattered distractor is only accepted when its 5-6 gaps would also have legal ray runway; this makes the over-open filter behaviorally meaningful.
  visual_observability:
    - Humans can separate the two subproblems from examples: first not every open frame emits, then among the emitting ones the grey diagonal orientation predicts turning direction.
    - The cue does not rely on hidden numeric ids. Gap count, empty-vs-symbol interior, and diagonal orientation are all directly visible.
    - The generator forces both symbol orientations to appear among true sources and requires at least one collision, so the encoded behavior is actually witnessed.
  cleanliness:
    - Total frame count stays between five and six, which is enough to show repeated evidence without overcrowding the grid.
    - Hollow distractors and shattered distractors never overlap beam spawn cells; their role stays legible in the input and unchanged in the output.
  inferability_guardrails:
    - Beam colors come from the non-grey diagonal of the source symbol, giving a direct visible link between symbol and trace color.
    - The generator rejects samples whose output matches an all-left or all-right counterfactual, and also rejects samples whose output matches the counterfactual that wrongly lets shattered symbol frames emit.
"""

import itertools

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}

INSTRUCTION_COLOR = 5
TURN_LEFT = -1
TURN_RIGHT = 1


class _FrameCandidate:
    __slots__ = ("r0", "c0", "wall", "gaps", "kind", "beam_color", "turn")

    def __init__(
        self,
        r0: int,
        c0: int,
        wall: int,
        gaps: tuple[tuple[int, int], ...],
        kind: str,
        beam_color: int | None = None,
        turn: int | None = None,
    ) -> None:
        self.r0 = r0
        self.c0 = c0
        self.wall = int(wall)
        self.gaps = gaps
        self.kind = kind
        self.beam_color = None if beam_color is None else int(beam_color)
        self.turn = None if turn is None else int(turn)

    @property
    def is_source(self) -> bool:
        return self.kind == "source"


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


def _frame_gaps_for_frame(g: np.ndarray, bg: int, wall: int, r0: int, c0: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = r0, c0
    fr1, fc1 = r0 + 3, c0 + 3
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
    if len(out_set) >= 3 and (fr1, fc1) in out_set and (fr0, c0 + 2) in out_set:
        out = [p for p in out if p != (fr1, fc1)]
    return out


def _classify_frames(g: np.ndarray, bg: int) -> list[_FrameCandidate]:
    h, w = g.shape
    frames: list[_FrameCandidate] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            border_vals: list[int] = []
            border_bg: list[tuple[int, int]] = []
            border_ok = True
            for dr, dc in _BORDER_ORDER:
                v = int(g[r0 + dr, c0 + dc])
                if v == bg:
                    border_bg.append((r0 + dr, c0 + dc))
                else:
                    border_vals.append(v)
            if not border_vals:
                continue
            wall_colors = sorted(set(border_vals))
            if len(wall_colors) != 1:
                continue
            wall = wall_colors[0]
            for dr, dc in _BORDER_ORDER:
                if int(g[r0 + dr, c0 + dc]) not in (bg, wall):
                    border_ok = False
                    break
            if not border_ok:
                continue
            wall_count = sum(int(g[r0 + dr, c0 + dc]) == wall for dr, dc in _BORDER_ORDER)
            if not (6 <= wall_count <= 12):
                continue

            gaps = tuple(_frame_gaps_for_frame(g, bg, wall, r0, c0))
            if set(border_bg) != set(gaps):
                continue

            inner = g[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
            parsed = _parse_symbol(inner, bg, wall)
            if parsed is not None:
                beam_color, turn = parsed
                if 1 <= len(gaps) <= 4:
                    kind = "source"
                elif len(gaps) == 0:
                    kind = "sealed_symbol"
                elif 5 <= len(gaps) <= 6:
                    kind = "shattered_symbol"
                else:
                    continue
                frames.append(_FrameCandidate(r0, c0, wall, gaps, kind, beam_color, turn))
                continue

            inner_vals = {int(v) for v in inner.ravel()}
            if inner_vals == {bg} and 1 <= len(gaps) <= 4:
                frames.append(_FrameCandidate(r0, c0, wall, gaps, "hollow_open"))
    return frames


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


def _solve(
    grid: np.ndarray,
    turn_override: int | None = None,
    accept_over_open: bool = False,
    collect_stats: bool = False,
):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    frames = _classify_frames(g, bg)
    sources = [fr for fr in frames if fr.is_source or (accept_over_open and fr.kind == "shattered_symbol")]
    assert sources, "expected at least one true source frame in input"

    stats = {
        "frame_count": len(frames),
        "source_count": sum(fr.kind == "source" for fr in frames),
        "hollow_open": sum(fr.kind == "hollow_open" for fr in frames),
        "sealed_symbol": sum(fr.kind == "sealed_symbol" for fr in frames),
        "shattered_symbol": sum(fr.kind == "shattered_symbol" for fr in frames),
        "source_left": sum(fr.kind == "source" and fr.turn == TURN_LEFT for fr in frames),
        "source_right": sum(fr.kind == "source" and fr.turn == TURN_RIGHT for fr in frames),
        "emitting_gap_total": 0,
        "collisions": 0,
        "cell_collisions": 0,
        "cross_collisions": 0,
        "segment_cross_collisions": 0,
        "accept_over_open": bool(accept_over_open),
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

    for fr in sources:
        assert fr.beam_color is not None and fr.turn is not None
        for gr, gc in fr.gaps:
            dr = -1 if gr == fr.r0 else (1 if gr == fr.r0 + 3 else 0)
            dc = -1 if gc == fr.c0 else (1 if gc == fr.c0 + 3 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each source gap to face outward"
            stats["emitting_gap_total"] += 1
            spawn_beam(fr.beam_color, (gr, gc), (d,), fr.turn if turn_override is None else turn_override)

    assert beams, "expected at least one beam spawned from source frames"

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
    Emit beams only from genuine symbol-bearing launch frames while ignoring frame-shaped decoys.

    Input structure:
    - The dominant background is still color 0 or 8, and every deliberate object is a 4x4 frame of one wall color.
    - True launchers contain a 2x2 diagonal symbol: grey color 5 occupies one diagonal and the other diagonal color
      is the beam color. Hollow open frames and over-open symbol frames act as distractors.

    Rule:
    - First classify frames: only symbol frames with 1-4 structural gaps are active sources.
    - Hollow frames with gaps emit nothing, and symbol frames with 5-6 gaps are also inert even though their symbol is valid.
    - Each active gap launches a beam in the outward direction set by the gap position.
    - Whenever beams collide on a cell or cross diagonally, the interaction point is handled as in the symbol variant:
      the contested cell is cleared when applicable, and each involved beam turns 45 degrees according to its own source symbol.

    Key insight:
    - The task is a two-stage inference chain. You must first filter out fake frames, then read the surviving frames'
      internal diagonal symbols to know how their beams will bend at collisions.
    """
    return _solve(grid)


# --------------- generator (o5x) ---------------

SOURCE_COUNT_RANGE = (3, 5)
DISTRACTOR_TYPES = ("hollow_open", "shattered_symbol")
MAX_TOTAL_FRAMES = 6
GRID_SIZE_MAX = 31
GRID_SIZE_MIN_BY_FRAMES = {5: 22, 6: 24}
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
GEN_RETRIES = 700
MIN_COLLISIONS = 1


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


_GAP_COMBOS = {n: _gap_combos(n) for n in range(1, 7)}


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


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    gaps: tuple[tuple[int, int], ...],
    kind: str,
    beam_color: int | None = None,
    turn: int | None = None,
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    if kind == "hollow_open":
        patch[1:3, 1:3] = bg
    else:
        assert beam_color is not None and turn is not None
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


def _pick_symbol_colors(rng: np.random.Generator, source_count: int) -> tuple[int, int, list[int], int]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c not in (frame_color, INSTRUCTION_COLOR)]
    colors = list(rng.choice(palette, size=source_count + 1, replace=False))
    return bg, frame_color, colors[:source_count], int(colors[-1])


def _pick_source_turns(rng: np.random.Generator, source_count: int) -> list[int]:
    turns = [TURN_LEFT, TURN_RIGHT]
    while len(turns) < source_count:
        turns.append(TURN_LEFT if int(rng.integers(2)) == 0 else TURN_RIGHT)
    rng.shuffle(turns)
    return turns


def _grid_size_min(frame_count: int) -> int:
    result = 8
    for threshold, min_n in sorted(GRID_SIZE_MIN_BY_FRAMES.items()):
        if frame_count >= threshold:
            result = min_n
    return result


def _observable_combo_variant(grid: np.ndarray, out: np.ndarray, stats: dict) -> bool:
    if stats["source_left"] == 0 or stats["source_right"] == 0:
        return False
    if stats["hollow_open"] < 1 or stats["shattered_symbol"] < 1:
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
    try:
        out_over_open = _solve(grid, accept_over_open=True)
        over_open_diff = bool(np.any(out != out_over_open))
    except Exception:
        over_open_diff = True
    return left_diff and right_diff and over_open_diff


def _frame_bbox_clean(grid: np.ndarray, out: np.ndarray, stats: dict) -> bool:
    bg = get_bg(grid)
    extra_painted = 0
    for fr in _classify_frames(grid, bg):
        patch_in = grid[fr.r0 : fr.r0 + 4, fr.c0 : fr.c0 + 4]
        patch_out = out[fr.r0 : fr.r0 + 4, fr.c0 : fr.c0 + 4]
        extra_painted += int(np.count_nonzero((patch_in == bg) & (patch_out != bg)))
    return extra_painted <= stats["emitting_gap_total"]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    source_count = int(rng.integers(*SOURCE_COUNT_RANGE))
    distractor_count = len(DISTRACTOR_TYPES)
    total_frames = source_count + distractor_count
    if total_frames > MAX_TOTAL_FRAMES:
        return None

    n = int(rng.integers(_grid_size_min(total_frames), GRID_SIZE_MAX))
    positions = _choose_positions(n, total_frames, rng)
    if positions is None:
        return None

    bg, frame_color, source_colors, shattered_color = _pick_symbol_colors(rng, source_count)
    source_turns = _pick_source_turns(rng, source_count)
    shattered_turn = TURN_LEFT if int(rng.integers(2)) == 0 else TURN_RIGHT

    objects: list[dict[str, object]] = [{"kind": "source"} for _ in range(source_count)]
    objects.extend({"kind": kind} for kind in DISTRACTOR_TYPES)
    rng.shuffle(objects)
    for obj, pos in zip(objects, positions):
        obj["pos"] = pos

    source_color_iter = iter(source_colors)
    source_turn_iter = iter(source_turns)
    for obj in objects:
        kind = obj["kind"]
        if kind == "source":
            obj["beam_color"] = int(next(source_color_iter))
            obj["turn"] = int(next(source_turn_iter))
        elif kind == "shattered_symbol":
            obj["beam_color"] = shattered_color
            obj["turn"] = shattered_turn
        else:
            obj["beam_color"] = None
            obj["turn"] = None

    obstacles = np.full((n, n), bg, dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        _draw_frame(
            obstacles,
            r0,
            c0,
            bg,
            frame_color,
            (),
            str(obj["kind"]),
            obj["beam_color"],
            obj["turn"],
        )

    source_gap_total = 0
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for obj in objects:
        r0, c0 = obj["pos"]
        kind = str(obj["kind"])
        available = _clear_gap_combos(obstacles, bg, r0, c0)
        if kind == "source":
            candidate_counts = [k for k in sorted(available) if 2 <= k <= 4]
            if not candidate_counts:
                candidate_counts = [k for k in sorted(available) if 1 <= k <= 4]
            if not candidate_counts:
                return None
            weights = np.array([k * k for k in candidate_counts], dtype=float)
            weights /= weights.sum()
            gap_count = int(rng.choice(candidate_counts, p=weights))
            gaps = available[gap_count][int(rng.integers(len(available[gap_count])))]
            obj["gaps"] = gaps
            source_gap_total += len(gaps)
            for gap in gaps:
                seeds.append(((r0 + gap[0], c0 + gap[1]), _gap_dir(gap)))
        elif kind == "shattered_symbol":
            candidate_counts = [k for k in sorted(available) if 5 <= k <= 6]
            if not candidate_counts:
                return None
            gap_count = int(rng.choice(candidate_counts))
            obj["gaps"] = available[gap_count][int(rng.integers(len(available[gap_count])))]
        else:
            candidate_counts = [k for k in sorted(_GAP_COMBOS) if 1 <= k <= 4]
            weights = np.array([1.0, 2.0, 2.0, 1.0], dtype=float)
            idx = int(rng.choice(len(candidate_counts), p=weights / weights.sum()))
            gap_count = candidate_counts[idx]
            combos = _GAP_COMBOS[gap_count]
            obj["gaps"] = combos[int(rng.integers(len(combos)))]

    grid = np.full((n, n), bg, dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        _draw_frame(
            grid,
            r0,
            c0,
            bg,
            frame_color,
            obj["gaps"],
            str(obj["kind"]),
            obj["beam_color"],
            obj["turn"],
        )

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None
    if not _rays_clear(grid, bg, seeds):
        return None

    frames = _classify_frames(grid, bg)
    if len(frames) != total_frames:
        return None
    if sum(fr.kind == "source" for fr in frames) != source_count:
        return None
    if sum(fr.kind == "hollow_open" for fr in frames) != 1:
        return None
    if sum(fr.kind == "shattered_symbol" for fr in frames) != 1:
        return None

    try:
        out, stats = _solve(grid, collect_stats=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < source_gap_total * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if not _frame_bbox_clean(grid, out, stats):
        return None
    if not _observable_combo_variant(grid, out, stats):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
