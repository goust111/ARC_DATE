# Combo variant of e12f9a14 (index 10)
# combo_spec: D7/0+D17/0
# Name: 多色框架+按颜色区分碰撞
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题: 所有 4x4 框都同色, 框色只负责“这是框”; 碰撞分支完全由几何角度决定.
#      同一套正交 90° 相撞在任何颜色下都留 1 格空洞并一起转向角平分线.
#    - 本融合: 框色不再只是装饰, 而是 beam 类别编码. 这里把框色分成两类:
#        cool = {1, 8}, warm = {2, 4}.
#      beam 仍用内层 2x2 方块颜色绘制, 但碰撞类型改由“发出它们的框色家族关系”决定:
#        * 同家族: 走规则 A, 保留原题 90° 的“留空 + 一起转向”.
#        * 异家族: 走规则 B, 不留空; 对 90° 正交相撞, 由竖直 beam 吃掉交点并一起沿角平分线继续.
#    - 生成器刻意构造两处几何完全同构的 90° 场景:
#        * scene A: 同家族框色相撞 -> 输出交点为空.
#        * scene B: 异家族框色相撞 -> 输出交点被保留.
#      两处场景的 gap 形状、相对位置、碰撞角度都一样, 唯一变的是框色家族关系, 因而肉眼能直接看出
#      D17/0 的控制变量已经从角度切换到颜色类别.
#
# b) 复杂度校准
#    - 只用两大家族而不是每个框一个独立碰撞规则, 避免自由度过多.
#    - 每个样例至少有 3 种框色, 其中同家族场景使用两种不同的实际颜色(例如红+黄或蓝+青),
#      防止解题者误以为规则是“完全同色”而不是“同色系”.
#    - 同时保留 beam 的原始绘制色(内块色), 让“框色负责类别、beam 色负责轨迹外观”的双层编码明显可见.
#
# c) 隐式假设审查
#    - 原 `_infer_frame_color()` 假设“每个框局部只有一种边框色”; 融合后仍成立, 只是不同框之间可不同色.
#    - 原 `resolve_cell_collision()` 把 `diff==2` 直接等价为 90° 规则, `diff==1` 直接等价为 45° 规则.
#      这在融合后失效, 所以现在先按 beam.family 分流, 再在分支内处理具体几何子型.
#    - 新 solver 额外依赖“框色必须能映射到家族”. 生成器只使用 {1,8,2,4}, 因而这一前提被构造保证.
#    - 同家族 45° 留空在离散同步模型里会让一侧 beam 立即撞回被封锁的交点, 视觉也不整洁.
#      因此生成器避免该子型; solver 仍支持异家族 45° zigzag, 但不主动生成同家族 45°.
#
# d) 放置合法性
#    - 所有框是显式坐标构造的 4x4 patch, 不会越界.
#    - 两个碰撞场景相距较远, 后续转向后的轨迹也不会再撞上别的框.
#    - 每个框只开一个 gap, 减少多射线互相干扰, 保证 solver 中的新分支只在目标场景被触发.
#
# e) 视觉可观察性
#    - 两处场景使用完全相同的正交碰撞几何, 只换颜色家族关系, 所以“同角度却不同结果”非常醒目.
#    - scene A 交点空, scene B 交点实, 是单像素级、可直接比对的显著差别.
#
# f) 可视化整洁度
#    - 每个框只发一条 beam, 输出没有大面积交叠带状结构.
#    - 两个场景上下分离, 留出充足背景, 人眼能独立跟踪每处局部因果链.
#
# g) 变化的可推断性
#    - 类别映射限定为两大家族, 且同家族场景使用“不同实际颜色但同结果”的对照.
#    - 异家族场景复用其中一个框色再换到另一家族, 提示“不是某个具体颜色特殊, 而是分组关系特殊”.

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_COOL_FAMILY = {1, 8}
_WARM_FAMILY = {2, 4}
_FAMILY_BY_FRAME = {c: 0 for c in _COOL_FAMILY} | {c: 1 for c in _WARM_FAMILY}

_BASE_N = 28
_BG = 0
_BLOCK_PALETTE = [3, 6, 7, 9]

_SCENE_SAME = {
    "positions": ((4, 3), (10, 10)),
    "gaps": ((1, 3), (0, 1)),
    "collision": (5, 11),
}
_SCENE_DIFF = {
    "positions": ((16, 3), (22, 10)),
    "gaps": ((1, 3), (0, 1)),
    "collision": (17, 11),
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
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
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


def _frame_family(frame_color: int) -> int:
    assert frame_color in _FAMILY_BY_FRAME, f"unexpected frame color {frame_color}"
    return _FAMILY_BY_FRAME[frame_color]


class _Beam:
    __slots__ = ("bid", "color", "frame_color", "family", "pos", "pattern", "phase", "path", "enter_dirs", "active")

    def __init__(
        self,
        bid: int,
        color: int,
        frame_color: int,
        family: int,
        pos: tuple[int, int],
        pattern: tuple[tuple[int, int], ...],
    ):
        self.bid = bid
        self.color = int(color)
        self.frame_color = int(frame_color)
        self.family = int(family)
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
    Emit rays from gapped framed 2x2 blocks, but classify collisions by the emitting frame colors.

    Input structure:
    - The background is the most common color.
    - Each 2x2 monochrome block is surrounded by a 4x4 frame whose border color may differ from other frames.
    - Every frame has one or more border gaps. A gap still determines the outgoing beam direction, while the frame
      color assigns that beam to a warm/cool collision family.

    Rule:
    - Each gap emits a beam painted with the inner block color.
    - Beams move exactly as in the seed task until they meet another beam.
    - Collision handling no longer keys off angle first. Instead:
      * same-family collision: use the thin, gap-producing redirect rule;
      * different-family collision: use the no-gap merge rule.
    - For mixed cardinal/diagonal different-family collisions, the result is the original zigzag merge.
      For orthogonal cardinal-vs-cardinal different-family collisions, the vertical beam keeps the meeting cell
      and both beams continue along the bisector without creating a blank hole.

    Key insight:
    Two collision sites can have identical geometry yet produce different outputs because the frame border colors
    reveal the beam classes. The solver must read frame colors as semantic labels, not as decorative wall pigments.
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

    def spawn_beam(frame_color: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, frame_color, _frame_family(frame_color), pos, pattern)
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

    def set_group_pattern(group: set[int], pattern: tuple[tuple[int, int], ...]) -> None:
        for gid in group:
            beams[gid].pattern = pattern
            beams[gid].phase = 0
            beams[gid].active = True

    def collision_group(a: int, b: int) -> set[int]:
        group = {a, b}
        pa = find_partner(a)
        pb = find_partner(b)
        if pa is not None:
            group.add(pa)
        if pb is not None:
            group.add(pb)
        return group

    def family_match(a: int, b: int) -> bool:
        return beams[a].family == beams[b].family

    def place_winner_at_cell(winner: int, loser: int, cell: tuple[int, int], winner_dir: tuple[int, int]) -> None:
        if cell_owner.get(cell) == loser:
            idx = cell_index[cell]
            truncate_beam(loser, idx)
            unpaint_cell(cell)
        if cell_owner.get(cell) == winner:
            idx = cell_index[cell]
            if idx + 1 < len(beams[winner].path):
                truncate_beam(winner, idx + 1)
        else:
            bw = beams[winner]
            bw.path.append(cell)
            bw.enter_dirs.append(winner_dir)
            bw.pos = cell
            paint_cell(winner, cell, winner_dir)

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

        if family_match(a, b):
            assert diff == 2, "generator avoids same-family 45-degree collisions"
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "unexpected same-family diagonal cell collision"
            if b_owns_cell and cell_owner.get(cell) == b:
                idx = cell_index[cell]
                truncate_beam(b, idx)
            blocked.add(cell)
            unpaint_cell(cell)
            nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
            set_group_pattern(collision_group(a, b), (nd,))
            return

        is_card_a = abs(da[0]) + abs(da[1]) == 1
        is_card_b = abs(db[0]) + abs(db[1]) == 1

        if is_card_a != is_card_b:
            assert diff == 1, "cardinal/diagonal collisions should be 45-degree"
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
            set_group_pattern({bid_u, bid_v}, zig)
            return

        nd = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert nd != (0, 0), "expected 90-degree merge to have a bisector"

        if is_card_a and is_card_b:
            winner = a if da[0] != 0 else b
            loser = b if winner == a else a
            winner_dir = da if winner == a else db
            place_winner_at_cell(winner, loser, cell, winner_dir)
            set_group_pattern(collision_group(a, b), (nd,))
            return

        winner = b if b_owns_cell else a
        loser = a if winner == b else b
        winner_dir = db if winner == b else da
        place_winner_at_cell(winner, loser, cell, winner_dir)
        set_group_pattern(collision_group(a, b), (nd,))

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
        u = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert u != (0, 0), "expected crossing diagonals to resolve to a forward direction"
        set_group_pattern(collision_group(a, b), (u,))

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

        u = _normalize_dir(da[0] + seg_dir[0], da[1] + seg_dir[1])
        assert u != (0, 0), "expected segment crossing to resolve to a forward direction"
        set_group_pattern(collision_group(a, b), (u,))

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(frame_col, col, (gr, gc), (d,))

    assert beams, "expected at least one beam spawned from frame gaps"

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

    return out


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_color: int,
    block_color: int,
    gap: tuple[int, int],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    patch[gap[0], gap[1]] = _BG
    paste(grid, patch, r0, c0)


def _transform_grid(grid: np.ndarray, transform: str) -> np.ndarray:
    if transform == "id":
        return grid
    if transform == "flip_h":
        return grid[:, ::-1]
    if transform == "flip_v":
        return grid[::-1, :]
    if transform == "flip_both":
        return grid[::-1, ::-1]
    raise AssertionError(f"unknown transform {transform}")


def _transform_cell(cell: tuple[int, int], n: int, transform: str) -> tuple[int, int]:
    r, c = cell
    if transform == "id":
        return cell
    if transform == "flip_h":
        return (r, n - 1 - c)
    if transform == "flip_v":
        return (n - 1 - r, c)
    if transform == "flip_both":
        return (n - 1 - r, n - 1 - c)
    raise AssertionError(f"unknown transform {transform}")


def _build_base_grid(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    grid = np.full((_BASE_N, _BASE_N), _BG, dtype=int)
    block_colors = list(map(int, rng.choice(_BLOCK_PALETTE, size=4, replace=False)))

    same_family_name = "warm" if int(rng.integers(2)) == 0 else "cool"
    same_colors = [2, 4] if same_family_name == "warm" else [1, 8]
    other_colors = [1, 8] if same_family_name == "warm" else [2, 4]
    rng.shuffle(same_colors)
    reused_same = same_colors[int(rng.integers(2))]
    outsider = other_colors[int(rng.integers(2))]

    frames = [
        (_SCENE_SAME["positions"][0], same_colors[0], block_colors[0], _SCENE_SAME["gaps"][0]),
        (_SCENE_SAME["positions"][1], same_colors[1], block_colors[1], _SCENE_SAME["gaps"][1]),
        (_SCENE_DIFF["positions"][0], reused_same, block_colors[2], _SCENE_DIFF["gaps"][0]),
        (_SCENE_DIFF["positions"][1], outsider, block_colors[3], _SCENE_DIFF["gaps"][1]),
    ]
    for (r0, c0), frame_color, block_color, gap in frames:
        _draw_framed_block(grid, r0, c0, frame_color, block_color, gap)

    assert len({frame for _, frame, _, _ in frames}) >= 3, "expected at least three distinct frame colors"
    return grid, {"same": _SCENE_SAME["collision"], "diff": _SCENE_DIFF["collision"]}


def _verify_fused_behavior(out: np.ndarray, collisions: dict[str, tuple[int, int]]) -> bool:
    same_cell = collisions["same"]
    diff_cell = collisions["diff"]
    return int(out[same_cell]) == _BG and int(out[diff_cell]) != _BG


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    base, collisions = _build_base_grid(rng)
    transform = ["id", "flip_h", "flip_v", "flip_both"][int(rng.integers(4))]
    grid = _transform_grid(base, transform)
    transformed_collisions = {
        name: _transform_cell(cell, _BASE_N, transform) for name, cell in collisions.items()
    }

    out = main(grid)
    assert get_bg(grid) == _BG, "generator expects black background to remain dominant"
    assert _verify_fused_behavior(out, transformed_collisions), "constructed sample failed fused-behavior check"
    assert np.count_nonzero(out != grid) >= 18, "constructed sample should produce non-trivial beams"
    return grid
