# Variant of e12f9a14
# Dimension: D14 — 多步处理与射线顺序 (multi_step)
# Change: 所有射线同时发射并独立延伸，碰撞在轨迹交叉时自然发生（无显式顺序）... → 迭代碰撞——碰撞后的新方向射线可再次碰撞

import importlib.util
from pathlib import Path

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D14 variant
# analysis:
#   concretization:
#     - 输入外观仍是原题的 4x4 框 + 内部 2x2 单色块；缺口继续决定射线起点和初始方向。
#     - 输出里最重要的新现象不是新颜色，而是“同一束射线的多次折向”：
#       射线不再全局同步起跑，而是按框/缺口的读取顺序逐束启动；
#       一次碰撞后形成的新方向会获得局部优先级，先连续推进，直到撞上第三束射线或到边界。
#     - 具体小例子：左侧先后发射的两条竖线会成为“已存在轨道”，右侧较晚启动的一条水平线先撞上右竖线，
#       改向后再撞上左竖线。同一束水平线因此连续经历两次转向。
#       原逻辑更接近同时展开，而本变体里“后发射的线先撞谁、再撞谁”完全由顺序决定。
#   complexity_calibration:
#     - 只改处理顺序，不改颜色规则与角度规则，避免把题目变成全新机制。
#     - 为保证 D14 可观察，生成器固定构造一个三框级联母题：右侧水平线先撞右竖线，再撞左竖线。
#       这样 2-3 个样例里都能重复看到“二次折向”，而不是靠低概率随机碰撞。
#     - 仍保留随机背景色、块颜色、间距、第二次碰撞所在行，避免样例过于死板。
#   implicit_assumptions_review:
#     - 原 main() 的主循环默认“所有 active beam 同步提案”；这里改成“顺序启动 + 碰撞组独占推进”。
#     - 原 resolve_* 在碰撞后只需改 pattern；这里额外要求返回碰撞组，以便后续调度知道谁继续先走。
#     - 原 find_partner 只在同步推进语境下用于双线并行；这里仍可用，但生成器需保证并行双线之间不会产生
#       180 度对撞等未定义角度。
#     - 原 cell collision 依赖 owner cell 代表“历史上已画出的射线路径”；新变体仍成立，因为暂停射线会保留
#       已画轨迹，正好作为后续级联碰撞的目标。
#   placement_legality:
#     - 生成器让三个核心框保持 >=2 格框间距，且所有射线初段都只穿过背景。
#     - 左上两框使用底边缺口生成两条竖线，右侧框使用左边缺口生成一条水平线；位置参数保证后发水平线能连续触发两次碰撞。
#     - 参数选择会预留第二次碰撞后继续向东南移动的 runway，避免刚拐弯就撞边界导致现象不可见。
#   observability:
#     - 新规则完全靠可见几何关系推断，不使用颜色 id、编号、隐式排序等不可见信息。
#     - 每个样本都至少出现一次“同一碰撞组的第二次碰撞”，因此与原题的“碰后直接一路走完”在视觉上可区分。
#   cleanliness:
#     - 每个核心框只开一个缺口，减少杂线叠加；级联发生在中部空白区，不遮住框体。
#     - 额外随机块默认关闭，避免把级联区域变成难以辨认的杂乱交叉网。
#   color_inference:
#     - 颜色语义不变：射线颜色仍等于内块颜色。变体只改时序，所以无需额外颜色映射学习负担。


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


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs", "active", "collisions")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]
        self.active = True
        self.collisions = 0

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
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

    stats = {
        "collision_events": 0,
        "cascade_events": 0,
        "max_collision_depth": 0,
        "priority_bursts": 0,
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
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"
    launch_order = sorted(beams)
    unlaunched = set(launch_order)
    next_launch_idx = 0
    for beam in beams.values():
        beam.active = False

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

    def mark_collision(group: set[int]) -> None:
        stats["collision_events"] += 1
        if any(beams[gid].collisions > 0 for gid in group):
            stats["cascade_events"] += 1
        for gid in group:
            beams[gid].collisions += 1
            stats["max_collision_depth"] = max(stats["max_collision_depth"], beams[gid].collisions)

    def activate_bid(bid: int) -> None:
        unlaunched.discard(bid)
        beams[bid].active = True

    def launch_next() -> int | None:
        nonlocal next_launch_idx
        while next_launch_idx < len(launch_order):
            bid = launch_order[next_launch_idx]
            next_launch_idx += 1
            if bid in unlaunched:
                activate_bid(bid)
                return bid
        return None

    def band_group(*base_bids: int) -> set[int]:
        group = {bid for bid in base_bids if bid in beams}
        for bid in tuple(group):
            partner = find_partner(bid)
            if partner is not None:
                group.add(partner)
        return group

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> set[int]:
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
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not share a cell"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            group = band_group(a, b)
            mark_collision(group)
            for gid in group:
                beams[gid].pattern = (nd,)
                beams[gid].phase = 0
                activate_bid(gid)
            return group

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
        group = {bid_u, bid_v}
        mark_collision(group)
        for gid in group:
            beams[gid].pattern = zig
            beams[gid].phase = 0
            activate_bid(gid)
        return group

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

    def resolve_cross_collision(a: int, b: int, prop) -> set[int]:
        da = prop[a][1]
        db = prop[b][1]
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        group = band_group(a, b)
        mark_collision(group)
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            activate_bid(gid)
        return group

    def resolve_segment_cross_collision(a: int, b: int, prop) -> set[int]:
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
        assert u != (0, 0), "expected segment crossing to resolve to a forward direction"

        group = band_group(a, b)
        mark_collision(group)
        for gid in group:
            beams[gid].pattern = (u,)
            beams[gid].phase = 0
            activate_bid(gid)
        return group

    moving_group: set[int] = set()
    max_iters = h * w * 300
    for _ in range(max_iters):
        if not moving_group:
            bid = launch_next()
            if bid is None:
                break
            moving_group = {bid}

        active = sorted(bid for bid in moving_group if beams[bid].active)
        if not active:
            moving_group = set()
            continue

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt) or not is_free(nxt):
                assert not in_bounds(nxt) or nxt in blocked or cell_owner.get(nxt) is not None, "beam must not hit block/frame"
                b.active = False
                continue
            prop[bid] = (nxt, d)

        if not prop:
            moving_group = set()
            continue

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
                group = resolve_cell_collision((r, c), a, b, prop)
            elif kind == "segcross":
                group = resolve_segment_cross_collision(a, b, prop)
            else:
                group = resolve_cross_collision(a, b, prop)
            moving_group = {gid for gid in group if beams[gid].active}
            stats["priority_bursts"] += 1
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
        moving_group = {gid for gid in moving_group if beams[gid].active}

    else:
        raise AssertionError("beam simulation did not converge")

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from gapped 4x4 frames around 2x2 blocks, with collisions resolved as local cascades.

    Input structure:
    - The background is the most common color; every colored 2x2 block sits inside a shared-color 4x4 frame.
    - Missing border pixels on that frame are exits, and each exit launches one ray in the outward cardinal or diagonal direction.

    Rule:
    - Rays are launched in reading order of the discovered gaps, and each moving ray (or collision-created ray group) paints background cells until it hits the boundary or a collision event.
    - The collision geometry is unchanged from the base task: 90-degree meetings leave a blank gap and redirect to the bisector, while 45-degree meetings merge into a zigzag pair.
    - The D14 change is temporal: launch order matters, and after a collision the newly redirected collision group keeps moving first, so it may collide again with a third paused ray before the next untouched gap gets its turn.

    Key insight:
    - The puzzle is no longer “draw every ray independently once”; it is “follow the launch order and the collision bursts it creates”.
    - A first collision creates a priority sub-system whose redirected heads can cause a second visible turn, so output shape depends on iterative collision order rather than only on static intersection geometry.
    """
    return _solve_with_stats(grid)[0]


def collect_variant_metrics(grid: np.ndarray) -> dict[str, int]:
    return dict(_solve_with_stats(grid)[1])


# --------------- generator (o5x) ---------------

GRID_SIZE_MIN = 20
GRID_SIZE_MAX = 31
GEN_RETRIES = 300
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
FRAME_BBOX_GAP = 2
MIN_CASCADE_DEPTH = 2
MIN_BEAM_PIXELS = 12

_CORE_GAPS = {
    "A": ((3, 1),),  # south
    "B": ((3, 1),),  # south
    "C": ((2, 0),),  # west
}
_ORIGINAL_MAIN = None


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


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


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    frame_color = FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in PALETTE_FOR_BG[bg] if c != frame_color]
    block_colors = list(map(int, rng.choice(palette, size=3, replace=False)))
    return bg, frame_color, block_colors


def _load_original_main():
    global _ORIGINAL_MAIN
    if _ORIGINAL_MAIN is not None:
        return _ORIGINAL_MAIN
    path = Path("/home/lixinyu/snap/new/arc2_solver_select/e12f9a14.py")
    if not path.exists():
        _ORIGINAL_MAIN = False
        return None
    spec = importlib.util.spec_from_file_location("_orig_e12f9a14", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ORIGINAL_MAIN = mod.main
    return _ORIGINAL_MAIN


def _choose_core_layout(n: int, rng: np.random.Generator):
    sep_choices = [sep for sep in (6, 8) if sep + 10 < n]
    if not sep_choices:
        return None
    sep = int(rng.choice(sep_choices))
    c_offset = int(rng.choice([6, 7]))

    c_left_max = n - (sep + c_offset + 4)
    r_top_max = n - (sep + 8)
    if c_left_max < 2 or r_top_max < 2:
        return None

    c_left = int(rng.integers(2, c_left_max + 1))
    r_top = int(rng.integers(2, r_top_max + 1))
    pos_a = (r_top, c_left)
    pos_b = (r_top, c_left + sep)
    pos_c = (r_top + 2, c_left + sep + c_offset)
    if not (_frames_separated(pos_a, pos_b) and _frames_separated(pos_a, pos_c) and _frames_separated(pos_b, pos_c)):
        return None

    second_hit_row = r_top + 3 + sep
    second_hit_col = c_left + 1
    if second_hit_row >= n - 3 or second_hit_col <= 1:
        return None

    return [
        (pos_a, _CORE_GAPS["A"]),
        (pos_b, _CORE_GAPS["B"]),
        (pos_c, _CORE_GAPS["C"]),
    ]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, frame_color, block_colors = _pick_colors(rng)
    layout = _choose_core_layout(n, rng)
    if layout is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for (r0, c0), gaps, color in zip((item[0] for item in layout), (item[1] for item in layout), block_colors):
        _draw_framed_block(grid, r0, c0, bg, frame_color, color, gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve_with_stats(grid)
    except Exception:
        return None

    if stats["max_collision_depth"] < MIN_CASCADE_DEPTH:
        return None
    if stats["cascade_events"] < 1:
        return None
    if stats["priority_bursts"] < 2:
        return None
    if np.count_nonzero(out != grid) < MIN_BEAM_PIXELS:
        return None

    orig_main = _load_original_main()
    if orig_main is not None:
        try:
            orig_out = orig_main(grid)
        except Exception:
            return None
        if np.array_equal(orig_out, out):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D14 e12f9a14 input")
