# Variant of e12f9a14
# Dimension: D7 — 框架的颜色与功能 (obstacle)
# Change: 所有框架使用统一的绿色(3)作为边框颜色，框架本身不阻挡其他射线... → 框架作为射线障碍物

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D7 variant
# ----------------------------------
# variant_spec:
#   visual_form:
#     - 每个对象仍然是一个 2x2 纯色小块，外面包一圈 4x4 绿色(3)框。
#     - 缺口仍然是边框上的背景色像素，仍然负责发射射线。
#     - 与原题不同的是：别的框不再只是“背景里的装饰”，而是绿色障碍物。
#     - 射线撞到别的框边时不再触发原版的 assert/非法布局；它会依据入射方式停止或反弹。
#   concrete_behavior:
#     - 斜向步(四个对角方向)撞到某个框的水平/垂直边时，会在当前格反弹到镜像对角方向。
#     - 直向步(上下左右)撞到框边时直接停止；撞到框角时也停止。
#     - 框的“缺口”不是障碍，所以生成器必须避免把目标命中点本身做成缺口，否则会退化回原逻辑。
#   pipeline_change_trace:
#     - 原始流程：proposal 阶段若下一步不是背景，代码默认它只能是越界或 blocked-gap；若是别的框边会触发
#       `assert not in_bounds(nxt) or nxt in blocked`，因此生成器只能回避“撞框”。
#     - D7 流程：proposal 阶段若下一步是框边，则先识别被撞的是水平边/垂直边/角，再把这次尝试改写成
#       “反弹后的一步”或“停止”，并把这次撞框记为有效事件。
#     - 差异本质：框从单纯射线源变成“源 + 障碍物”，影响点正是在 proposal 里对非背景下一步的解释。
#   complexity_calibration:
#     - 只引入两档新行为：`diagonal -> bounce` 与 `cardinal/corner -> stop`，避免出现过多分支。
#     - 生成器固定造出三个框：一个被撞的目标框、一个制造反弹的源框、一个制造停止的源框。
#       这样 2-3 组样例里人眼就能看到“绿色边框会挡住/折返射线”，不会被随机性淹没。
#     - 仍保留原始射线-射线碰撞引擎，因此 D7 之外的核心机制未被删改。
#   hidden_assumptions_review:
#     - 原 `is_free()` 把“背景且未 blocked”当成唯一可走单元；这隐含了“任何框边都不能被碰到”。
#       变体下这个前提失效，所以 proposal 逻辑必须先区分“框障碍”与“其它实心物”。
#     - 原代码允许走进已经上色但输入里是背景的格子，用于处理射线间碰撞；这个前提仍成立。
#       因此新逻辑不能把 `cell_owner` 当成 obstacle，只需要阻止射线走回自己已画过的轨迹。
#     - 原 collision resolver 假定 proposal 中的方向就是本轮真实移动方向。D7 下反弹方向必须先写进
#       proposal，才能让后续碰撞处理继续复用原逻辑。
#   placement_legality:
#     - 生成器先放置 3 个 4x4 框，保证 bbox 之间至少隔 2 格，不重叠、不越界。
#     - 目标框上预留两个“保留墙格”：一个给反弹源撞、一个给停止源撞；目标框自己的缺口不能开在这些格子上。
#     - 反弹源与停止源的射线都要求在撞框前至少穿过 2 个背景格，保证事件可见。
#   observability_and_cleanliness:
#     - 每个样例都至少包含一次反弹和一次停止，避免新机制只偶尔出现。
#     - 其他额外射线被压到最低：每框只开一个缺口，减少覆盖层次，保持画面整洁。
#     - 颜色不参与任何排序或映射；可推断规则完全来自几何关系与绿色边框的阻挡作用。


FRAME_COLOR = 3

_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_DIAG_GAP_FOR_DIR = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 3),
    (1, -1): (3, 0),
    (1, 1): (3, 3),
}

_BORDER_ORDER = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (2, 3), (3, 3), (3, 2),
    (3, 1), (3, 0), (2, 0), (1, 0),
)
_BORDER_INDEX = {p: i for i, p in enumerate(_BORDER_ORDER)}

_TARGET_BOUNCE_CHOICES = [
    ((1, 0), (1, 1)),
    ((2, 0), (-1, 1)),
    ((1, 3), (1, -1)),
    ((2, 3), (-1, -1)),
    ((0, 1), (1, 1)),
    ((0, 2), (1, -1)),
    ((3, 1), (-1, 1)),
    ((3, 2), (-1, -1)),
]

_TARGET_STOP_CHOICES = [
    ((1, 0), (0, 1), (1, 3)),
    ((2, 0), (0, 1), (2, 3)),
    ((1, 3), (0, -1), (1, 0)),
    ((2, 3), (0, -1), (2, 0)),
    ((0, 1), (1, 0), (3, 1)),
    ((0, 2), (1, 0), (3, 2)),
    ((3, 1), (-1, 0), (0, 1)),
    ((3, 2), (-1, 0), (0, 2)),
]


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


def _reflect_dir(d: tuple[int, int], border_kind: str) -> tuple[int, int] | None:
    dr, dc = d
    if border_kind == "corner":
        return None
    if abs(dr) + abs(dc) != 2:
        return None
    if border_kind == "h":
        return (-dr, dc)
    if border_kind == "v":
        return (dr, -dc)
    raise AssertionError("unknown border kind")


def _reflect_pattern(pattern: tuple[tuple[int, int], ...], border_kind: str) -> tuple[tuple[int, int], ...] | None:
    if border_kind == "corner":
        return None
    out = []
    for dr, dc in pattern:
        if border_kind == "h":
            out.append((-dr, dc))
        else:
            out.append((dr, -dc))
    return tuple(out)


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


def _build_frame_obstacles(
    g: np.ndarray, bg: int, blocks: list[tuple[int, int, int]]
) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    for col, br, bc in blocks:
        wall = _infer_frame_color(g, bg, br, bc, col)
        gaps = set(_frame_gaps_for_block(g, bg, wall, br, bc))
        fr0, fc0 = br - 1, bc - 1
        for lr in range(4):
            for lc in range(4):
                if not (lr in (0, 3) or lc in (0, 3)):
                    continue
                cell = (fr0 + lr, fc0 + lc)
                if cell in gaps:
                    continue
                if int(g[cell]) != wall:
                    continue
                if (lr, lc) in ((0, 0), (0, 3), (3, 0), (3, 3)):
                    kind = "corner"
                elif lr in (0, 3):
                    kind = "h"
                else:
                    kind = "v"
                lookup[cell] = kind
    return lookup


def _simulate(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"
    frame_lookup = _build_frame_obstacles(g, bg, blocks)

    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    stats = None
    if collect_stats:
        stats = {
            "frame_hits": 0,
            "frame_bounces": 0,
            "frame_stops": 0,
            "collision_events": 0,
            "beams_spawned": 0,
            "beams_finished": 0,
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
        if stats is not None:
            stats["beams_spawned"] += 1

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

    def is_free_bg(cell: tuple[int, int]) -> bool:
        return int(g[cell]) == bg and cell not in blocked

    def stop_beam(bid: int) -> None:
        beams[bid].active = False
        if stats is not None:
            stats["beams_finished"] += 1

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
            is_diag_a = abs(da[0]) + abs(da[1]) == 2
            is_diag_b = abs(db[0]) + abs(db[1]) == 2
            assert not (is_diag_a and is_diag_b), "two diagonal beams must cross geometrically, not share a cell"
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
        assert in_bounds(adj) and is_free_bg(adj) and cell_owner.get(adj) is None, "expected free adjacent zigzag step"
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
        pending_patterns: dict[int, tuple[tuple[int, int], ...]] = {}
        for bid in active:
            b = beams[bid]
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])

            if not in_bounds(nxt) or nxt in blocked or cell_owner.get(nxt) == bid:
                stop_beam(bid)
                continue

            if is_free_bg(nxt):
                prop[bid] = (nxt, d)
                continue

            border_kind = frame_lookup.get(nxt)
            if border_kind is None:
                stop_beam(bid)
                continue

            if stats is not None:
                stats["frame_hits"] += 1

            reflected = _reflect_pattern(b.pattern, border_kind)
            rd = _reflect_dir(d, border_kind)
            if reflected is None or rd is None:
                if stats is not None:
                    stats["frame_stops"] += 1
                stop_beam(bid)
                continue

            bounce = (b.pos[0] + rd[0], b.pos[1] + rd[1])
            if not in_bounds(bounce) or bounce in blocked or int(g[bounce]) != bg or cell_owner.get(bounce) == bid:
                if stats is not None:
                    stats["frame_stops"] += 1
                stop_beam(bid)
                continue

            prop[bid] = (bounce, rd)
            pending_patterns[bid] = reflected
            if stats is not None:
                stats["frame_bounces"] += 1

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
            if stats is not None:
                stats["collision_events"] += 1
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
            if bid in pending_patterns:
                b.pattern = pending_patterns[bid]
            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
    else:
        raise AssertionError("beam simulation did not converge")

    if stats is None:
        return out
    stats["painted_pixels"] = int(np.count_nonzero(out != g))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from gapped green frames around 2x2 color blocks; rays still collide, but frames now obstruct them.

    Input structure:
    - The background is the most common color, and every 2x2 monochrome block is surrounded by a 4x4 green frame.
    - Each frame contains one or more background-colored holes on its border; those holes are the beam exits.

    Rule:
    - Every gap emits a same-colored beam outward according to its border position, painting background cells as it travels.
    - Beams keep the original beam-vs-beam collision rules, but a green frame border from any block is now a real obstacle.
    - A diagonal step that strikes a horizontal or vertical frame edge reflects immediately to the mirrored diagonal step; a cardinal step, or any hit on a frame corner, stops at that border.
    - Reflected beams never paint the struck frame cell itself; they continue from the last free cell, and then remain subject to the usual 45° / 90° beam collisions.

    Key insight:
    - The new information is encoded by where beams terminate or ricochet against green frames, so the same framed objects act as both emitters and blockers.
    """
    return _simulate(grid, collect_stats=False)


# --------------- generator (o5x) ---------------

BLOCK_COUNT = 3
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 31
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 4, 6, 7, 8, 9],
}
FRAME_BBOX_GAP = 2
PLACEMENT_MARGIN = 2
MIN_RAY_STEPS = 2
MIN_BEAM_PIXELS_PER_GAP = 2
GEN_RETRIES = 400
LAYOUT_RETRIES = 80
DIST_CHOICES = [4, 5, 6, 7]


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _frame_fits(n: int, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    return PLACEMENT_MARGIN <= r0 <= n - 4 - PLACEMENT_MARGIN and PLACEMENT_MARGIN <= c0 <= n - 4 - PLACEMENT_MARGIN


def _draw_framed_block(
    grid: np.ndarray, r0: int, c0: int, bg: int,
    frame_color: int, block_color: int, gaps: tuple[tuple[int, int], ...],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    for gr, gc in gaps:
        patch[gr, gc] = bg
    paste(grid, patch, r0, c0)


def _trace_until_obstacle(
    grid: np.ndarray, bg: int, seed: tuple[int, int], d: tuple[int, int]
) -> tuple[int, tuple[int, int] | None]:
    h, w = grid.shape
    r, c = seed[0] + d[0], seed[1] + d[1]
    bg_steps = 0
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != bg:
            return bg_steps, (r, c)
        bg_steps += 1
        r += d[0]
        c += d[1]
    return bg_steps, None


def _pick_colors(rng: np.random.Generator) -> tuple[int, list[int]]:
    bg = int(rng.choice(BG_CHOICES, p=BG_WEIGHTS))
    palette = [c for c in PALETTE_FOR_BG[bg] if c != FRAME_COLOR]
    block_colors = list(rng.choice(palette, size=BLOCK_COUNT, replace=False))
    return bg, block_colors


def _choose_target_gap(
    obstacles: np.ndarray,
    bg: int,
    target_pos: tuple[int, int],
    reserved_locals: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    r0, c0 = target_pos
    choices = []
    for gap in _BORDER_ORDER:
        if gap in reserved_locals:
            continue
        d = _gap_dir(gap)
        steps, hit = _trace_until_obstacle(obstacles, bg, (r0 + gap[0], c0 + gap[1]), d)
        if steps >= MIN_RAY_STEPS and hit is None:
            choices.append(gap)
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _build_layout(n: int, rng: np.random.Generator):
    for _ in range(LAYOUT_RETRIES):
        target = (
            int(rng.integers(PLACEMENT_MARGIN + 4, n - 4 - PLACEMENT_MARGIN - 4)),
            int(rng.integers(PLACEMENT_MARGIN + 4, n - 4 - PLACEMENT_MARGIN - 4)),
        )

        bounce_candidates = []
        for hit_local, d in _TARGET_BOUNCE_CHOICES:
            gap_local = _DIAG_GAP_FOR_DIR[d]
            for k in DIST_CHOICES:
                hit_abs = (target[0] + hit_local[0], target[1] + hit_local[1])
                gap_abs = (hit_abs[0] - k * d[0], hit_abs[1] - k * d[1])
                src = (gap_abs[0] - gap_local[0], gap_abs[1] - gap_local[1])
                if _frame_fits(n, src) and _frames_separated(src, target):
                    bounce_candidates.append((src, gap_local, hit_local, hit_abs, d))
        if not bounce_candidates:
            continue
        bounce_src, bounce_gap, bounce_hit_local, bounce_hit_abs, bounce_dir = bounce_candidates[int(rng.integers(len(bounce_candidates)))]

        stop_candidates = []
        for hit_local, d, gap_local in _TARGET_STOP_CHOICES:
            if hit_local == bounce_hit_local:
                continue
            for k in DIST_CHOICES:
                hit_abs = (target[0] + hit_local[0], target[1] + hit_local[1])
                gap_abs = (hit_abs[0] - k * d[0], hit_abs[1] - k * d[1])
                src = (gap_abs[0] - gap_local[0], gap_abs[1] - gap_local[1])
                if not _frame_fits(n, src):
                    continue
                if not _frames_separated(src, target) or not _frames_separated(src, bounce_src):
                    continue
                stop_candidates.append((src, gap_local, hit_local, hit_abs, d))
        if not stop_candidates:
            continue
        stop_src, stop_gap, stop_hit_local, stop_hit_abs, stop_dir = stop_candidates[int(rng.integers(len(stop_candidates)))]

        obstacles = np.full((n, n), 8, dtype=int)
        positions = [target, bounce_src, stop_src]
        for pos in positions:
            _draw_framed_block(obstacles, pos[0], pos[1], 8, FRAME_COLOR, 1, ())
        bg_mask = obstacles == 8
        obstacles[bg_mask] = 0
        obstacles[~bg_mask] = FRAME_COLOR

        bounce_seed = (bounce_src[0] + bounce_gap[0], bounce_src[1] + bounce_gap[1])
        stop_seed = (stop_src[0] + stop_gap[0], stop_src[1] + stop_gap[1])
        bounce_steps, bounce_hit = _trace_until_obstacle(obstacles, 0, bounce_seed, bounce_dir)
        stop_steps, stop_hit = _trace_until_obstacle(obstacles, 0, stop_seed, stop_dir)
        if bounce_steps < MIN_RAY_STEPS or stop_steps < MIN_RAY_STEPS:
            continue
        if bounce_hit != bounce_hit_abs or stop_hit != stop_hit_abs:
            continue

        bounce_kind = "h" if bounce_hit_local[0] in (0, 3) else "v"
        reflected_dir = _reflect_dir(bounce_dir, bounce_kind)
        assert reflected_dir is not None
        before_hit = (bounce_hit_abs[0] - bounce_dir[0], bounce_hit_abs[1] - bounce_dir[1])
        bounce_target = (before_hit[0] + reflected_dir[0], before_hit[1] + reflected_dir[1])
        if not (0 <= bounce_target[0] < n and 0 <= bounce_target[1] < n):
            continue
        if obstacles[bounce_target] != 0:
            continue

        reserved = {bounce_hit_local, stop_hit_local}
        target_gap = _choose_target_gap(obstacles, 0, target, reserved, rng)
        if target_gap is None:
            continue

        return {
            "positions": positions,
            "gaps": [target_gap, bounce_gap, stop_gap],
            "reserved": reserved,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(GRID_SIZE_MIN, GRID_SIZE_MAX))
    bg, block_colors = _pick_colors(rng)

    layout = _build_layout(n, rng)
    if layout is None:
        return None

    grid = np.full((n, n), bg, dtype=int)
    for pos, color, gap in zip(layout["positions"], block_colors, layout["gaps"]):
        _draw_framed_block(grid, pos[0], pos[1], bg, FRAME_COLOR, int(color), (gap,))

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _simulate(grid, collect_stats=True)
    except Exception:
        return None

    if np.count_nonzero(out != grid) < len(layout["gaps"]) * MIN_BEAM_PIXELS_PER_GAP:
        return None
    if stats["frame_hits"] < 2:
        return None
    if stats["frame_bounces"] < 1 or stats["frame_stops"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 D7 input")
