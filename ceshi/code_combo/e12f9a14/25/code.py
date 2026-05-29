# Combo variant of e12f9a14 (index 25)
# combo_spec: D2/3+D8/0+D19/0
# Name: 统一碰撞规则+始终单像素+转90度

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo:
# a) Concrete behavior trace:
#    Original e12f9a14 split collisions into two families. A cardinal beam meeting a diagonal
#    beam at 45 degrees created a thicker zigzag, while a 90-degree collision created a blank
#    meeting cell and then sent both beams along the bisector. In this combo, those branches
#    disappear. Every collision is rewritten the same way: roll both beams back to the last
#    uncontested cells, keep the contested location empty, then choose a legal 90-degree turn
#    for each beam. Example: an east-going beam from a right-edge gap meets a NE beam from a
#    top-right gap. The original task would turn them into a thick zigzag. Here the east beam
#    bends upward, the NE beam bends rightward, and both remain single-pixel lines.
# b) Complexity calibration:
#    The rule is simpler, but not degenerate. The generator always includes at least one legacy
#    45-degree collision, so the new output is visibly different from the original thick-zigzag
#    behavior. Rotation, grid size, separation, and colors vary, while the collision grammar
#    stays small enough for a human to infer from a few examples.
# c) Hidden-assumption audit:
#    The old solver assumed pattern/phase switching, 2x2 thickening, and angle-specific
#    collision buckets. None of those remain valid. The new solver removes pattern cycling,
#    never paints multi-cell-wide trajectories, and handles cell / diagonal-cross / segment-cross
#    events through the same redirection primitive. The new assumption is local free space around
#    the collision so two turned continuations can be chosen; the generator guarantees this by
#    using a motif whose forced turn cells are clear.
# d) Placement legality:
#    The generator only stamps 4x4 framed blocks fully inside the grid, keeps them separated,
#    and validates the finished grid by running the new solver. The collision motif computes
#    its own runway so beams do not hit another frame before colliding or turning.
# e) Visual observability:
#    Each sample contains a deliberate old-45-degree collision. In the original puzzle family,
#    that would create a thick zigzag. In this combo, the same setup yields two crisp single-pixel
#    orthogonal deflections around an empty meeting point, which is easy to spot by eye.
# f) Visual cleanliness:
#    The generator uses only two framed blocks and one forced collision motif, so there is little
#    overdraw. The blank collision cell keeps colors from competing for one pixel and preserves a
#    readable bend for both beams.
# g) Inferability of change:
#    No hidden numeric ordering is used. Colors only identify which frame emitted which beam.
#    The key clue is geometric: diagonal/cardinal beams that used to merge into thick paths now
#    remain thin and peel off into orthogonal polylines.


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


def _is_diag(d: tuple[int, int]) -> bool:
    return abs(d[0]) + abs(d[1]) == 2


def _turn_options(d: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = d
    if _is_diag(d):
        return ((dr, 0), (0, dc))
    if dr == 0:
        return ((-1, 0), (1, 0))
    return ((0, -1), (0, 1))


class _Beam:
    __slots__ = ("bid", "color", "pos", "dir", "path", "enter_dirs", "active", "turned")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.dir = direction
        self.path = [pos]
        self.enter_dirs = [direction]
        self.active = True
        self.turned = False


def _solve(grid: np.ndarray, *, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    stats = {
        "collision_count": 0,
        "legacy45_collision_count": 0,
        "legacy90_collision_count": 0,
        "diag_cross_collision_count": 0,
        "segment_cross_collision_count": 0,
        "blocked_cell_count": 0,
        "post_turn_steps": 0,
    }

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

    def clear_beam(bid: int) -> None:
        b = beams[bid]
        for cell in b.path:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = []
        b.enter_dirs = []
        b.active = False

    def rewind_beam(bid: int, keep_len: int) -> tuple[int, int] | None:
        b = beams[bid]
        if keep_len <= 0:
            clear_beam(bid)
            return None
        assert keep_len <= len(b.path), "invalid rewind length"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.active = True
        return b.pos

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, direction)
        paint_cell(bid, pos, direction)

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = _normalize_dir(dr, dc)
            assert direction != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), direction)

    assert beams, "expected at least one beam spawned from frame gaps"

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def is_open_bg(cell: tuple[int, int]) -> bool:
        r, c = cell
        return int(g[r, c]) == bg and cell not in blocked

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

    def choose_turn_pair(
        src_a: tuple[int, int] | None,
        dir_a: tuple[int, int],
        src_b: tuple[int, int] | None,
        dir_b: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if src_a is None or src_b is None:
            return None
        best: tuple[int, int, tuple[int, int], tuple[int, int]] | None = None
        for out_a in _turn_options(dir_a):
            ta = (src_a[0] + out_a[0], src_a[1] + out_a[1])
            if not in_bounds(ta) or not is_open_bg(ta) or cell_owner.get(ta) is not None:
                continue
            for out_b in _turn_options(dir_b):
                tb = (src_b[0] + out_b[0], src_b[1] + out_b[1])
                if not in_bounds(tb) or not is_open_bg(tb) or cell_owner.get(tb) is not None:
                    continue
                if ta == tb:
                    continue
                score = abs(ta[0] - tb[0]) + abs(ta[1] - tb[1])
                tie = int(out_a != out_b)
                cand = (score, tie, out_a, out_b)
                if best is None or cand > best:
                    best = cand
        if best is None:
            return None
        return best[2], best[3]

    def apply_redirect(
        bid_a: int,
        src_a: tuple[int, int] | None,
        dir_a: tuple[int, int],
        bid_b: int,
        src_b: tuple[int, int] | None,
        dir_b: tuple[int, int],
    ) -> None:
        choice = choose_turn_pair(src_a, dir_a, src_b, dir_b)
        if choice is None:
            beams[bid_a].active = False
            beams[bid_b].active = False
            return
        out_a, out_b = choice
        beams[bid_a].dir = out_a
        beams[bid_b].dir = out_b
        beams[bid_a].active = True
        beams[bid_b].active = True
        beams[bid_a].turned = True
        beams[bid_b].turned = True

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        src_a = beams[a].pos
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            src_b = beams[b].pos
        else:
            db = cell_enter[cell]
            idx = cell_index[cell]
            src_b = rewind_beam(b, idx)

        stats["collision_count"] += 1
        diff = _dir_diff(da, db)
        if diff == 1:
            stats["legacy45_collision_count"] += 1
        else:
            stats["legacy90_collision_count"] += 1

        blocked.add(cell)
        unpaint_cell(cell)
        stats["blocked_cell_count"] = len(blocked)
        apply_redirect(a, src_a, da, b, src_b, db)

    def resolve_cross_collision(a: int, b: int, prop) -> None:
        stats["collision_count"] += 1
        stats["legacy90_collision_count"] += 1
        stats["diag_cross_collision_count"] += 1
        apply_redirect(a, beams[a].pos, prop[a][1], b, beams[b].pos, prop[b][1])

    def resolve_segment_cross_collision(a: int, b: int, prop) -> None:
        da = prop[a][1]
        pr, pc = beams[a].pos
        dr, dc = da
        p2 = (pr, pc + dc)
        n2 = (pr + dr, pc)

        src_b: tuple[int, int] | None = None
        db: tuple[int, int] | None = None
        if cell_owner.get(n2) == b and cell_index.get(n2, 0) > 0 and cell_enter.get(n2) == (n2[0] - p2[0], n2[1] - p2[1]):
            db = (n2[0] - p2[0], n2[1] - p2[1])
            src_b = rewind_beam(b, cell_index[n2])
        elif cell_owner.get(p2) == b and cell_index.get(p2, 0) > 0 and cell_enter.get(p2) == (p2[0] - n2[0], p2[1] - n2[1]):
            db = (p2[0] - n2[0], p2[1] - n2[1])
            src_b = rewind_beam(b, cell_index[p2])
        else:
            raise AssertionError("expected crossing segment to be owned by the partner beam")

        assert db is not None
        stats["collision_count"] += 1
        stats["legacy90_collision_count"] += 1
        stats["segment_cross_collision_count"] += 1
        apply_redirect(a, beams[a].pos, da, b, src_b, db)

    max_iters = h * w * 200
    for _ in range(max_iters):
        active = [bid for bid, b in beams.items() if b.active and b.path]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            nxt = (b.pos[0] + b.dir[0], b.pos[1] + b.dir[1])
            if not in_bounds(nxt) or not is_open_bg(nxt):
                b.active = False
                continue
            if cell_owner.get(nxt) == bid:
                b.active = False
                continue
            prop[bid] = (nxt, b.dir)

        if not prop:
            break

        cross_pairs: set[tuple[int, int]] = set()
        move_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for bid, (nxt, d) in prop.items():
            if not _is_diag(d):
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
            if not _is_diag(d):
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
            b = None
            if owner is not None and owner != a:
                b = owner
            elif len(bidders_sorted) >= 2:
                b = bidders_sorted[1]
            if b is not None:
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
            elif kind == "cross":
                resolve_cross_collision(a, b, prop)
            else:
                resolve_segment_cross_collision(a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            b = beams[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            if b.turned:
                stats["post_turn_steps"] += 1
    else:
        raise AssertionError("beam simulation did not converge")

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from framed 2x2 blocks, but collapse every collision into the same right-angle deflection rule.

    Input structure:
    - The background is the most frequent color.
    - Each colored 2x2 block sits inside a 4x4 frame of one shared wall color, and one or more wall pixels are missing.
    - Every missing frame pixel is an exit that seeds a beam in the outward-facing cardinal or diagonal direction.

    Rule:
    - Beams paint one cell at a time in their current direction and always stay single-pixel wide.
    - When a beam would hit another beam, or two diagonal steps would geometrically cross, the contested meeting point is not painted.
    - Both participating beams instead restart from the last uncontested cells and each takes one legal 90-degree turn, after which they continue as thin orthogonal polylines until another collision or the grid boundary stops them.

    Key insight:
    - The original angle-dependent system is gone. There is only one collision vocabulary now: keep the collision site empty, reroute both beams by a quarter-turn, and never thicken the trajectory.
    """
    return _solve(grid, want_stats=False)


# --------------- generator (combo) ---------------

GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 30
GEN_RETRIES = 300
MIN_POST_TURN_STEPS = 3
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}


def _pick_colors(rng: np.random.Generator, block_count: int) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(rng.choice(palette, size=block_count, replace=False))
    return bg, frame_color, block_colors


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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    bg, frame_color, block_colors = _pick_colors(rng, 2)
    grid = np.full((n, n), bg, dtype=int)

    east_row = int(rng.choice([1, 2]))
    layouts: list[tuple[int, int]] = []
    for sep in range(6, 11):
        t = sep - east_row
        max_r0 = n - sep - 4
        max_c0 = n - t - 6
        if t >= 2 and max_r0 >= 2 and max_c0 >= 2:
            layouts.append((sep, t))
    if not layouts:
        return None

    sep, t = layouts[int(rng.integers(len(layouts)))]
    max_r0 = n - sep - 4
    max_c0 = n - t - 6
    r0 = int(rng.integers(2, max_r0 + 1))
    c0 = int(rng.integers(2, max_c0 + 1))
    r1 = r0 + sep

    _draw_framed_block(grid, r0, c0, bg, frame_color, int(block_colors[0]), ((east_row, 3),))
    _draw_framed_block(grid, r1, c0, bg, frame_color, int(block_colors[1]), ((0, 3),))

    k = int(rng.integers(4))
    if k:
        grid = np.rot90(grid, k).copy()

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, want_stats=True)
    except Exception:
        return None

    if stats["collision_count"] < 1:
        return None
    if stats["legacy45_collision_count"] < 1:
        return None
    if stats["post_turn_steps"] < MIN_POST_TURN_STEPS:
        return None
    if np.count_nonzero(out != grid) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
