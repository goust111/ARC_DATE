# Combo variant of db695cfb (index 28)
# combo_spec: D6/0+D11/1
# Name: 多对蓝色像素+最近邻配对
#
# Design analysis
# a) 具体化与行为追踪:
#    - 原题默认每条相关对角线上只出现一对蓝点，因此 solver 会把同一条对角线上的最外侧蓝点直接连起来。
#    - 本融合版固定构造 6 个蓝点: 其中 4 个蓝点落在同一条 45 度对角线上，按最近邻形成两段短蓝线；另 2 个蓝点
#      落在垂直对角线上，与前者共享一个品红触发点。
#    - 一个典型小例子是主对角线上蓝点偏移为 0,4,7,11。原逻辑会把 0 和 11 连成长线；融合后必须先做最近邻完美配对，
#      得到 (0,4) 与 (7,11) 两段短线，中间 5..6 保持非蓝。若共享的品红点落在左侧短线上，同时也落在竖交的另一条
#      对角线上，那么它会累计两组“垂直对角方向”，从而向四个斜向边界继续发射。
# b) 复杂度校准:
#    - 最近邻配对只允许蓝点总数为 6，避免配对自由度过高；同时主对角线上始终有两对蓝点，保证和原题视觉上稳定不同。
#    - 交互只引入一个共享品红点，不再叠加更多共享交点，避免画面乱成网格。
#    - 最近邻歧义通过“总长度最小，再按坐标字典序最小”消歧，规则是几何可见的，不依赖颜色编号之类不可视因素。
# c) 隐式假设审查:
#    - 原 main() 的“按颜色找最长对角对，再按整条对角线取极值端点”前提已失效；这里改成先对全部蓝点做完美匹配。
#    - draw_segment() 原先默认路径上遇到的非蓝像素都只是单线触发器；融合后同一品红点可能被多条线命中，所以必须把
#      触发方向收集成集合并求并集，再统一发射。
#    - 新 solver 的新假设是: 蓝点数为偶数，且存在由对角边组成的完美匹配。generator 用构造和 oracle 检查保证它成立。
# d) 放置合法性:
#    - 生成器先放一条包含两段短线的主对角 cluster，再在其中一个内部品红点上穿过一条反向对角短线。
#    - 所有端点、触发点都在网格内且互不重合；共享交点只允许发生在品红触发点上。
#    - 输出期望值按“先画所有蓝段，再把共享触发点累计到的所有垂直方向统一发射”构建，并用 solver 反验。
# e) 视觉可观察性:
#    - 人类能直接看到“同一条对角线上有两段短蓝线，中间断开”，这迫使解题者放弃原题的极值连线思路。
#    - 共享品红点还会在两条线的交叉处生成四向斜射线，样例里能稳定展示“一个点可被多条线同时触发”的现象。
# f) 可视化整洁度:
#    - 只保留 3 条有效蓝段和 1 个共享触发点，避免出现多组交叉共享导致难以分辨。
#    - 干扰品红点只放在活动对角线和所有输出射线之外，避免看起来像“应该触发却没触发”的伪规律。
# g) 变化的可推断性:
#    - 颜色语义仍然只有三类: 背景色、蓝色成段、品红射线；没有新增颜色映射。
#    - 融合变化完全由几何关系决定: 先最近邻配对，再根据共享触发点所在的每一条线分别贡献垂直射线方向。

from functools import lru_cache
import random

import numpy as np

from arc_utils import get_bg


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]
Segment = tuple[Coord, Coord, tuple[Coord, ...], tuple[Coord, ...], tuple[tuple[int, int], ...]]


def _is_diag_pair(p: Coord, q: Coord) -> bool:
    return p != q and p[0] != q[0] and abs(p[0] - q[0]) == abs(p[1] - q[1])


def _pair_distance(p: Coord, q: Coord) -> int:
    return abs(p[0] - q[0])


def _canonical_pair(p: Coord, q: Coord) -> Pair:
    return (p, q) if p <= q else (q, p)


def _diag_path(p: Coord, q: Coord) -> list[Coord]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _best_blue_matching(points: list[Coord]) -> list[Pair]:
    pts = tuple(sorted(points))
    assert pts and len(pts) % 2 == 0, "blue endpoints must be a non-empty even-sized set"

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[Coord, ...]) -> tuple[Pair, ...] | None:
        if not remaining:
            return ()

        first = remaining[0]
        best_key = None
        best_pairs = None
        for i in range(1, len(remaining)):
            other = remaining[i]
            if not _is_diag_pair(first, other):
                continue
            rest = solve(remaining[1:i] + remaining[i + 1 :])
            if rest is None:
                continue
            pairs = tuple(sorted((_canonical_pair(first, other),) + rest))
            lengths = tuple(sorted(_pair_distance(a, b) for a, b in pairs))
            key = (sum(lengths), lengths, pairs)
            if best_key is None or key < best_key:
                best_key = key
                best_pairs = pairs
        return best_pairs

    result = solve(pts)
    assert result is not None, "blue endpoints must admit a diagonal perfect matching"
    return list(result)


def _segment_dirs(p: Coord, q: Coord) -> tuple[int, int]:
    return (1 if q[0] > p[0] else -1, 1 if q[1] > p[1] else -1)


def _collect_trigger_dirs(segments: list[Segment]) -> dict[Coord, set[tuple[int, int]]]:
    trigger_dirs: dict[Coord, set[tuple[int, int]]] = {}
    for p, q, trigger_positions, _, _ in segments:
        dr, dc = _segment_dirs(p, q)
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for pos in trigger_positions:
            trigger_dirs.setdefault(pos, set()).update(perp_dirs)
    return trigger_dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match multiple blue endpoints by nearest diagonal pairing, then let every magenta point on those matched segments emit all perpendicular rays implied by the segments that pass through it.

    Input structure:
    - The dominant color is background. Blue pixels form an even-sized endpoint set, typically with two short pairs on one diagonal plus another pair on the perpendicular diagonal. Magenta pixels may sit on one matched segment or at an intersection shared by multiple matched segments.

    Rule:
    - Compute the perfect matching of blue pixels using only diagonal-aligned candidate pairs; choose the one with minimum total diagonal length, breaking ties by coordinate order. Draw every matched pair as its own blue diagonal segment.
    - Whenever a non-blue pixel lies on a drawn segment, treat it as a trigger for that segment's perpendicular diagonal directions. If the same magenta pixel lies on several matched segments, union all contributed directions and extend magenta rays to the border, filling only background cells.

    Key insight:
    - The decisive reasoning step is global pairing, not extreme-to-extreme connection. Once the correct short segments are chosen, a shared trigger can inherit perpendicular directions from more than one segment, so the final magenta pattern reflects both the pairing choice and the segment interaction.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    line_color = 1

    blue_points = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]
    pairs = _best_blue_matching(blue_points)

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: Coord, q: Coord) -> None:
        path = _diag_path(p, q)
        dr, dc = _segment_dirs(p, q)
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for r, c in path:
            value = int(out[r, c])
            if value == bg:
                out[r, c] = line_color
            elif value != line_color:
                key = (r, c, value)
                triggers.setdefault(key, set()).update(perp_dirs)

    for p, q in pairs:
        draw_segment(p, q)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_PAIR_LEN_RANGE = (3, 5)
_PAIR_GAP_RANGE = (1, 3)
_CROSS_ARM_RANGE = (1, 3)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_MOTIF_RETRIES = 200
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _max_steps_from(start: Coord, dr: int, dc: int, H: int, W: int) -> int:
    r, c = start
    steps = 0
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
        steps += 1
    return steps


def _point_on_diag(start: Coord, dc: int, offset: int) -> Coord:
    return (start[0] + offset, start[1] + offset * dc)


def _segment_record(p: Coord, q: Coord, triggers: list[Coord]) -> Segment:
    path = tuple(_diag_path(p, q))
    dr, dc = _segment_dirs(p, q)
    perp_dirs = ((dr, -dc), (-dr, dc))
    return (p, q, tuple(triggers), path, perp_dirs)


def _compute_ray_cells(origin: Coord, dirs: set[tuple[int, int]], H: int, W: int) -> set[Coord]:
    cells: set[Coord] = set()
    for ddr, ddc in dirs:
        r, c = origin[0] + ddr, origin[1] + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[Segment]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()

    for _, _, _, path, _ in segments:
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR

    trigger_dirs = _collect_trigger_dirs(segments)
    for (r0, c0), dirs in trigger_dirs.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < H and 0 <= c < W:
                if int(expected[r, c]) == bg:
                    expected[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return expected


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    by_color: dict[int, list[Coord]] = {}
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value != bg:
                by_color.setdefault(value, []).append((r, c))

    best = None
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                p, q = pts[i], pts[j]
                if not _is_diag_pair(p, q):
                    continue
                cand = (_pair_distance(p, q), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None
    _, line_color = best
    pts = by_color[line_color]

    groups_a: dict[int, list[Coord]] = {}
    groups_b: dict[int, list[Coord]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: Coord, q: Coord) -> None:
        dr, dc = _segment_dirs(p, q)
        perp_dirs = {(dr, -dc), (-dr, dc)}
        for r, c in _diag_path(p, q):
            value = int(out[r, c])
            if value == bg:
                out[r, c] = line_color
            elif value != line_color:
                triggers.setdefault((r, c, value), set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])
    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


def _try_make_motif(H: int, W: int) -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    for _ in range(_MOTIF_RETRIES):
        main_dc = random.choice((1, -1))
        len_a = random.randint(*_PAIR_LEN_RANGE)
        len_b = random.randint(*_PAIR_LEN_RANGE)
        gap = random.randint(*_PAIR_GAP_RANGE)

        start_b = len_a + gap + 1
        end_b = start_b + len_b
        max_offset = end_b

        if main_dc == 1:
            if H - 1 - max_offset < 0 or W - 1 - max_offset < 0:
                continue
            base = (random.randint(0, H - 1 - max_offset), random.randint(0, W - 1 - max_offset))
            active_backslash = {base[0] - base[1]}
            active_slash: set[int] = set()
        else:
            if H - 1 - max_offset < 0 or max_offset > W - 1:
                continue
            base = (random.randint(0, H - 1 - max_offset), random.randint(max_offset, W - 1))
            active_backslash = set()
            active_slash = {base[0] + base[1]}

        seg_a = (_point_on_diag(base, main_dc, 0), _point_on_diag(base, main_dc, len_a))
        seg_b = (_point_on_diag(base, main_dc, start_b), _point_on_diag(base, main_dc, end_b))

        shared_source = random.choice((0, 1))
        shared_lo = 1 if shared_source == 0 else start_b + 1
        shared_hi = len_a - 1 if shared_source == 0 else end_b - 1
        if shared_lo > shared_hi:
            continue
        shared_offset = random.randint(shared_lo, shared_hi)
        shared_trigger = _point_on_diag(base, main_dc, shared_offset)

        cross_dc = -main_dc
        max_neg = _max_steps_from(shared_trigger, -1, -cross_dc, H, W)
        max_pos = _max_steps_from(shared_trigger, 1, cross_dc, H, W)
        if min(max_neg, max_pos) < 1:
            continue

        cross_choices = []
        for neg in range(_CROSS_ARM_RANGE[0], min(_CROSS_ARM_RANGE[1], max_neg) + 1):
            for pos in range(_CROSS_ARM_RANGE[0], min(_CROSS_ARM_RANGE[1], max_pos) + 1):
                if 3 <= neg + pos <= 6:
                    cross_choices.append((neg, pos))
        if not cross_choices:
            continue
        cross_neg, cross_pos = random.choice(cross_choices)
        cross_seg = (
            (shared_trigger[0] - cross_neg, shared_trigger[1] - cross_neg * cross_dc),
            (shared_trigger[0] + cross_pos, shared_trigger[1] + cross_pos * cross_dc),
        )
        if cross_dc == 1:
            active_backslash.add(cross_seg[0][0] - cross_seg[0][1])
        else:
            active_slash.add(cross_seg[0][0] + cross_seg[0][1])

        segments = [
            _segment_record(seg_a[0], seg_a[1], [shared_trigger] if shared_source == 0 else []),
            _segment_record(seg_b[0], seg_b[1], [shared_trigger] if shared_source == 1 else []),
            _segment_record(cross_seg[0], cross_seg[1], [shared_trigger]),
        ]

        blue_points = [seg_a[0], seg_a[1], seg_b[0], seg_b[1], cross_seg[0], cross_seg[1]]
        intended_pairs = sorted(_canonical_pair(*pair) for pair in (seg_a, seg_b, cross_seg))
        if sorted(_best_blue_matching(blue_points)) != intended_pairs:
            continue

        inp = np.full((H, W), bg, dtype=int)
        for r, c in blue_points:
            inp[r, c] = _LINE_COLOR
        inp[shared_trigger] = _TRIGGER_COLOR

        expected = _build_expected(inp, bg, segments)
        if np.array_equal(expected, _legacy_main(inp)):
            continue

        trigger_dirs = _collect_trigger_dirs(segments)
        blocked = set()
        for _, _, trigger_positions, path, _ in segments:
            blocked.update(path)
            for pos in trigger_positions:
                blocked.add(pos)
        for pos, dirs in trigger_dirs.items():
            blocked.update(_compute_ray_cells(pos, dirs, H, W))

        for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
            for _ in range(_DISTRACTOR_RETRIES):
                r = random.randint(0, H - 1)
                c = random.randint(0, W - 1)
                if (r, c) in blocked or int(inp[r, c]) != bg:
                    continue
                if (r - c) in active_backslash or (r + c) in active_slash:
                    continue
                inp[r, c] = _TRIGGER_COLOR
                blocked.add((r, c))
                break

        expected = _build_expected(inp, bg, segments)
        try:
            result = main(inp)
        except Exception:
            continue
        if np.array_equal(result, expected):
            return inp, expected

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    built = _try_make_motif(H, W)
    if built is None:
        return None
    inp, _ = built
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
