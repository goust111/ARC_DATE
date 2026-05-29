# Combo variant of e12f9a14 (index 50)
# combo_spec: D2/3+D9/1+D17/2
# Name: 三种碰撞角度+统一碰撞规则+均有间隙
#
# Design analysis for the combo variant
# -------------------------------------
# concretization:
# - 原题里有两类碰撞：
#   - 90°: 交点留 1 格空白，然后两束光改走角平分线方向。
#   - 45°: 不留空白，而是并成一条 cardinal/diagonal 交替的 zigzag 带。
# - 融合后改成三类角度，但只保留一套“变向”逻辑：
#   - 90° 与 45° 都改为：两束光在碰撞后都沿 `normalize(d1+d2)` 的统一方向继续。
#   - 180° 正面对撞新增为第三类：两束光都停止。
#   - 三类碰撞都留下永久空白带，且空白带长度直接编码角度：
#     90° -> 1 格，45° -> 2 格，180° -> 3 格。
# - 一个具体小例子：
#   - 原题：一条向右的射线和一条右下对角射线在同一格相遇，会并成 zigzag，交点不空。
#   - 现在：同样两束光仍统一改向到右下，但不会画 zigzag；它们会在碰撞后各自的出射道上先跳过 2 格空白，再继续画细单像素射线。
#   - 因而 45° 与 90° 的唯一视觉差异不再是“形状分支”，而只剩空隙长短。
#
# complexity calibration:
# - 这里不能把问题简化成“所有碰撞都一样”。如果 45° 和 90° 连空隙长度也相同，
#   D9/1 与 D17/2 就会退化成不可见。
# - 也不能保留原题的 zigzag；那样 45° 与 90° 仍会被轨迹形状区分，违背“三级间隙大小成为唯一视觉区分”。
# - 所以最稳定的折中是：45°/90° 共用同一出射方向公式，但用 1 格 vs 2 格永久空白做唯一标签；
#   180° 则用停止 + 3 格断裂，形成清晰的第三档。
#
# hidden-assumption audit:
# - 原 `resolve_cell_collision()` 假设 `diff in (1, 2)`，即只有 45°/90°。这里必须放宽到 `diff in (1, 2, 4)`，
#   其中 `4` 表示 180° 对撞。
# - 原 45° 分支依赖“恰好一条 cardinal、一条 diagonal”以及后续 zigzag 相位；这些前提在融合后全部失效，
#   因为 45° 现在不再进入独立 zigzag 分支。
# - 原 `resolve_cross_collision()` / `resolve_segment_cross_collision()` 处理几何对角交叉。为了让“空隙长度是唯一视觉差异”
#   保持干净，生成器不再制造这类无显式同格交点的情况；solver 也因此只需要稳定处理同格碰撞。
# - 原代码默认碰撞只会抹去一个交点像素。这里 45°/180° 会抹去更长的永久空白段，所以 `blocked`
#   不能只记录一个点，而要允许一次加入一整段 gap cells。
#
# placement legality:
# - 生成器不再随机搜索任意碰撞，而是构造 3 个彼此分离的碰撞区：一个 90°、一个 45°、一个 180°。
# - 每个区只放两条会相遇的射线，所有框架都保持至少原题要求的 2-cell bbox clearance。
# - 每条射线都先有足够长的无碰撞跑道，碰撞后的统一出射方向也预留了额外空间，避免 gap 段越界或撞到别的框。
#
# observability / cleanliness:
# - 每个样例都同时包含三种角度，并且它们分散在网格的三个区域，避免多束光叠在一起看不清。
# - 由于所有射线都保持 1 像素宽，视觉上最显眼的变化就是空隙长度 1/2/3；解题者在 2-4 个样例里能直接比对出来。
# - 颜色不参与任何隐藏排序；角度信息完全由几何关系和空白长度给出。
#
# new solver assumptions guaranteed by the generator:
# - 只生成“同一目标格”的碰撞，不生成纯几何 X 形 diagonal crossing。
# - 三类角度至少各出现一次，因此 gap-size 编码不是偶发装饰。
# - 每个碰撞后的 gap 段都有足够 runway，可按 solver 的统一规则安全插入。

import numpy as np

from arc_utils import get_bg, paste


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}
_DEFAULT_GAP_BY_DIFF = {1: 2, 2: 1, 4: 3}


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


class _Beam:
    __slots__ = ("bid", "color", "pos", "direction", "path", "enter_dirs", "active")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], direction: tuple[int, int]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.direction = direction
        self.path = [pos]
        self.enter_dirs = [direction]
        self.active = True


def _solve(grid: np.ndarray, want_stats: bool = False, gap_by_diff: dict[int, int] | None = None):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    gap_by_diff = dict(_DEFAULT_GAP_BY_DIFF if gap_by_diff is None else gap_by_diff)

    out = g.copy()
    blocked: set[tuple[int, int]] = set()
    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    stats = {
        "collision_45": 0,
        "collision_90": 0,
        "collision_180": 0,
        "gap_cells_45": 0,
        "gap_cells_90": 0,
        "gap_cells_180": 0,
    }

    beams: dict[int, _Beam] = {}
    next_bid = 0
    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    def in_bounds(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < h and 0 <= c < w

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = beams[bid].color
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
        assert 0 < keep_len < len(b.path), "expected truncate to remove at least one painted cell"
        for cell in b.path[keep_len:]:
            if cell_owner.get(cell) == bid:
                unpaint_cell(cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]
        b.active = True

    def block_cell(cell: tuple[int, int]) -> None:
        assert in_bounds(cell), "gap cell must stay inside the grid"
        assert int(g[cell[0], cell[1]]) == bg, "gaps may only occupy background cells"
        unpaint_cell(cell)
        blocked.add(cell)
        out[cell[0], cell[1]] = bg

    def spawn_beam(color: int, pos: tuple[int, int], direction: tuple[int, int]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        assert pos not in cell_owner, "expected unique beam seed positions"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, direction)
        paint_cell(bid, pos, direction)

    def is_free(cell: tuple[int, int]) -> bool:
        return int(g[cell[0], cell[1]]) == bg and cell not in blocked

    def insert_gap_after_turn(bids: tuple[int, int], out_dir: tuple[int, int], gap_len: int) -> int:
        inserted = 0
        for bid in bids:
            b = beams[bid]
            last = b.pos
            for step in range(1, gap_len + 1):
                cell = (b.pos[0] + out_dir[0] * step, b.pos[1] + out_dir[1] * step)
                block_cell(cell)
                last = cell
                inserted += 1
            b.pos = last
            b.direction = out_dir
            b.active = True
        return inserted

    def resolve_head_on(cell: tuple[int, int], a: int, b: int, db: tuple[int, int], both_moving: bool) -> None:
        if both_moving:
            cells = [beams[a].pos, cell, beams[b].pos]
        else:
            cells = [beams[a].pos, cell, (cell[0] - db[0], cell[1] - db[1])]
        unique_cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for gc in cells:
            if not in_bounds(gc) or gc in seen:
                continue
            seen.add(gc)
            unique_cells.append(gc)
        for gc in unique_cells:
            block_cell(gc)
        beams[a].active = False
        beams[b].active = False
        stats["collision_180"] += 1
        stats["gap_cells_180"] += len(unique_cells)

    def resolve_cell_collision(cell: tuple[int, int], a: int, b: int, prop) -> None:
        da = prop[a][1]
        if b in prop and prop[b][0] == cell:
            db = prop[b][1]
            both_moving = True
        else:
            db = cell_enter[cell]
            both_moving = False

        diff = _dir_diff(da, db)
        assert diff in gap_by_diff, "unexpected collision angle"

        if diff == 4:
            resolve_head_on(cell, a, b, db, both_moving)
            return

        if not both_moving and cell_owner.get(cell) == b:
            idx = cell_index[cell]
            truncate_beam(b, idx)

        block_cell(cell)
        out_dir = _normalize_dir(da[0] + db[0], da[1] + db[1])
        assert out_dir != (0, 0), "45°/90° collision must yield a shared outgoing direction"
        gap_len = gap_by_diff[diff]
        inserted = insert_gap_after_turn((a, b), out_dir, gap_len)
        if diff == 1:
            stats["collision_45"] += 1
            stats["gap_cells_45"] += inserted
        else:
            stats["collision_90"] += 1
            stats["gap_cells_90"] += inserted

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        gaps = _frame_gaps_for_block(g, bg, frame_col, br, bc)
        assert gaps, "expected every block to have at least one structural frame gap"
        for gr, gc in gaps:
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            direction = _normalize_dir(dr, dc)
            assert direction != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), direction)

    max_iters = h * w * 100
    for _ in range(max_iters):
        active = [bid for bid, beam in beams.items() if beam.active]
        if not active:
            break

        prop: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        for bid in active:
            beam = beams[bid]
            nxt = (beam.pos[0] + beam.direction[0], beam.pos[1] + beam.direction[1])
            if not in_bounds(nxt) or not is_free(nxt):
                beam.active = False
                continue
            prop[bid] = (nxt, beam.direction)

        if not prop:
            break

        cell_to_bids: dict[tuple[int, int], list[int]] = {}
        for bid, (nxt, _d) in prop.items():
            cell_to_bids.setdefault(nxt, []).append(bid)

        events: list[tuple[int, int, int, int]] = []
        for cell, bidders in cell_to_bids.items():
            owner = cell_owner.get(cell)
            if len(bidders) >= 2:
                ordered = sorted(bidders)
                events.append((cell[0], cell[1], ordered[0], ordered[1]))
            elif owner is not None and owner != bidders[0]:
                events.append((cell[0], cell[1], bidders[0], owner))

        if events:
            events.sort()
            r, c, a, b = events[0]
            resolve_cell_collision((r, c), a, b, prop)
            continue

        for bid in sorted(prop):
            nxt, d = prop[bid]
            assert cell_owner.get(nxt) is None, "expected collision-free target cell"
            beam = beams[bid]
            beam.path.append(nxt)
            beam.enter_dirs.append(d)
            beam.pos = nxt
            paint_cell(bid, nxt, d)
    else:
        raise AssertionError("beam simulation did not converge")

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit beams from gapped framed 2x2 blocks, then resolve collisions with angle-coded permanent gaps.

    Input structure:
    - The background is the dominant color, and each object is a monochrome 2x2 block wrapped by a 4x4 frame of one shared wall color.
    - Missing wall pixels on that frame border are beam exits; edge gaps launch cardinal beams and corner gaps launch diagonal beams.
    - Every beam keeps the color of its source 2x2 block and starts directly on the visible gap cell.

    Rule:
    - Extend each beam through background cells until it reaches the grid edge or collides with another beam.
    - 90° and 45° collisions now share one turn rule: both beams redirect to the normalized sum of their incoming directions.
    - All collisions leave a permanent blank gap, and the gap length encodes the angle: 90° uses 1 cell, 45° uses 2 cells, and 180° head-on collisions use 3 cells and stop both beams.
    - Aside from these beam traces and collision gaps, the input drawing stays unchanged.

    Key insight: The old puzzle required recognizing different post-collision trajectory families; the fused variant removes that branch and makes the angle readable only from the size of the blank interruption left at the collision.
    """
    return _solve(grid, want_stats=False)


# --------------- generator (o5x) ---------------

GRID_SIZE_RANGE = (33, 36)
BG_CHOICES = [0, 8]
BG_WEIGHTS = [0.25, 0.75]
FRAME_COLOR_FOR_BG = {8: 3, 0: 8}
PALETTE_FOR_BG = {
    8: [1, 2, 4, 6, 7, 9],
    0: [1, 2, 3, 4, 6, 7, 9],
}
GEN_RETRIES = 300
MIN_CHANGED_PIXELS = 18

BASE_LAYOUT = (
    ((3, 16), ((1, 3),)),   # 180° pair, left beam goes east
    ((3, 27), ((1, 0),)),   # 180° pair, right beam goes west
    ((12, 2), ((1, 3),)),   # 90° pair, left beam goes east
    ((5, 9), ((3, 1),)),    # 90° pair, top beam goes south
    ((18, 20), ((3, 3),)),  # 45° pair, diagonal beam goes SE
    ((25, 20), ((1, 3),)),  # 45° pair, cardinal beam goes east
)


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
    block_colors = list(map(int, rng.choice(palette, size=6, replace=False)))
    return bg, frame_color, block_colors


def _shifted_layout(n: int, rng: np.random.Generator):
    max_r0 = max(r0 for (r0, _c0), _g in BASE_LAYOUT)
    max_c0 = max(c0 for (_r0, c0), _g in BASE_LAYOUT)
    dr = int(rng.integers(0, n - (max_r0 + 4) + 1))
    dc = int(rng.integers(0, n - (max_c0 + 4) + 1))
    return [((r0 + dr, c0 + dc), gaps) for (r0, c0), gaps in BASE_LAYOUT]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(*GRID_SIZE_RANGE))
    layout = _shifted_layout(n, rng)

    bg, frame_color, block_colors = _pick_colors(rng)
    grid = np.full((n, n), bg, dtype=int)
    for ((r0, c0), gaps), color in zip(layout, block_colors, strict=False):
        _draw_framed_block(grid, r0, c0, bg, frame_color, int(color), gaps)

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != bg:
        return None

    try:
        out, stats = _solve(grid, want_stats=True)
        collapsed, collapsed_stats = _solve(grid, want_stats=True, gap_by_diff={1: 1, 2: 1, 4: 1})
    except Exception:
        return None

    if stats["collision_90"] < 1 or stats["collision_45"] < 1 or stats["collision_180"] < 1:
        return None
    if np.array_equal(out, collapsed):
        return None
    if collapsed_stats["collision_90"] != stats["collision_90"]:
        return None
    if collapsed_stats["collision_45"] != stats["collision_45"]:
        return None
    if collapsed_stats["collision_180"] != stats["collision_180"]:
        return None
    if np.count_nonzero(out != grid) < MIN_CHANGED_PIXELS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e12f9a14 combo input")
