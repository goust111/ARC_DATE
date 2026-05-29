# Variant of 5545f144
# Dimension: D11 — 路径复杂度约束 (meta)
# Change: total_steps 4-12，turn_ratio 0.4-0.9，至少使用3个基数方向，至少2次转弯，步距2-7且... → 回文路径
"""
设计分析
========

a) 具体化与行为追踪
-
原任务的像素元素不变: 输入仍然是若干横向拼接的快照帧, 每帧里有一个三角形精灵和若干与精灵同色的单像素标记点; 输出仍然是最终一帧的精灵位置。

本变体只改“路径如何被生成”。原任务里 future-head 序列只要求满足若干复杂度统计约束, 因而方向序列可能是不规则的:
`U, R, R, D, L, ...`

现在改为: future-head 的方向序列必须是逐步级别的回文。例如:
`U, R, D, R, U`
对应的头部位置可以是:
`h0 -> h1(上) -> h2(右) -> h3(下) -> h4(右) -> h5(上)`
输入帧里看到的所有标记点仍然只是这些未来 head 的落点, 但它们对应的行进方向满足
`dir(h0,h1) == dir(h4,h5)`、
`dir(h1,h2) == dir(h3,h4)`。

流程差异:
- 原始逻辑: generator 先采样“高转折、多方向、非单调”的 step_dirs, 再把每一步落点渲染成标记。
- 变体逻辑: generator 先采样一个前半段 prefix, 再镜像成完整的回文 step_dirs, 其余 rollout / 渲染 / solver 消费规则保持不变。
- solver 侧的局部行为没有变化: 它仍然从最右帧出发, 每次沿唯一可见的半线前往最近标记; 回文性质是全路径分布约束, 不是单步更新规则。

b) 复杂度校准
-
如果只要求“是回文”, 最容易退化成 `R,R,R,R` 这种肉眼几乎看不出新规律的路径。为避免视觉退化, 这里做两个与回文直接相关的轻量约束:
- prefix 至少包含 2 个方向段, 因而完整回文路径不会退化成整条直线。
- 方向段之间只允许直行或 90 度转弯, 不允许 180 度反转, 这样既保留原题“不会倒车”的局部动力学, 也让镜像后的折返更整洁。

没有保留旧 D11 的 turn-ratio / 三方向 / 非单调硬约束, 以免把“回文”又变回原先那组统计规则。

c) 隐式假设审查
-
与变体相关的 main()/solver 前提如下:
- `_split_frames()` 假设只有分隔列会形成“整列同色且非背景”。回文路径可能更容易出现长竖直轨迹, 所以 generator 额外拒绝任何伪分隔整列。
- `_find_agent()` 假设最后一帧里, 精灵模板仍能从“精灵像素 + 标记像素”的并集中唯一找回。generator 继续保留原来的 oracle 检查。
- `_choose_target()` 假设从当前 head 出发, 剩余标记至多只会落在一个基数方向的半线上; generator 仍通过逐步兼容性检查强制保证这一点。
- `_simulate_to_end()` 原版默认“最后一帧一定还有标记”, 否则 `agent` 变量未初始化。这里顺手把初始精灵体保存下来, 消除这个原本只是在旧分布下恰好无害的前提。
- solver 仍然依赖“不会出现 180 度反向移动”。新 generator 在采样回文序列时显式排除了这种相邻反向。

d) 放置合法性
-
generator 必须同时保证:
- 每个未来 head 都在同一帧内可放下对应方向的完整模板。
- 标记不会压到任何一步的精灵体上, 也不会与旧 head 重合或紧贴到旧精灵体造成误检。
- 对任一历史 head, 新放下的标记若恰好与其同行/同列, 必须正好位于该步的出发方向上, 且不能比真实下一步更近, 否则 solver 会提前吃错。
- 拼接后的整张输入图里, 除了 divider 位置外, 不允许再出现整列纯色的非背景列。

e) 视觉可观察性
-
人类可观察到的变化是“转向模式成镜像”: 路径不是随意折线, 而是前半段怎么拐, 后半段按相反时间顺序再拐一遍。这个规律直接体现在位置关系上, 不依赖颜色编号、随机种子或不可见元数据。

f) 可视化整洁度
-
对象种类仍只有三类视觉元素: 背景、单色精灵、单色标记、以及可选的单色分隔列。为了避免回文折返把路径挤成一团, 生成时保留原有“不重叠、不贴身、不制造二义性”的几何筛选, 并优先使用 2..7 的步长把折线拉开。

g) 颜色变化可推断性
-
本变体不引入新的颜色映射。背景、精灵/标记、分隔列三类颜色的职责与原题相同, 因而不存在额外的颜色归纳负担。
"""

import importlib.util
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left
DISTRACTOR_COLOR = 8

_ORIGINAL_5545F144 = None


def _matches_original(inp: np.ndarray, out: np.ndarray) -> bool:
    global _ORIGINAL_5545F144
    if _ORIGINAL_5545F144 is None:
        spec = importlib.util.spec_from_file_location(
            "orig_5545f144", "/home/lixinyu/snap/new/arc2_solver_select/5545f144.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ORIGINAL_5545F144 = module
    try:
        original_out = np.asarray(_ORIGINAL_5545F144.main(np.asarray(inp, dtype=int)), dtype=int)
    except Exception:
        return False
    return np.array_equal(original_out, np.asarray(out, dtype=int))


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns of a non-background color. Return list of frames and bg.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) != bg]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
    return frames, bg


# Templates are defined as offsets relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    # Hollow triangle (3 pixels)
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],  # Up
        1: [(0, 0), (-1, -1), (1, -1)],  # Right
        2: [(0, 0), (-1, -1), (-1, 1)],  # Down
        3: [(0, 0), (-1, 1), (1, 1)],  # Left
    },
    # Filled triangle (4 pixels: tip + base of 3)
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    # Arrow-like triangle (4 pixels: tip + stem + two base corners)
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
    # 3-row triangle (6 pixels): tip + filled middle row + base corners
    "midfilled6": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    },
}


def _find_agent(
    frame: np.ndarray, actor: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == actor)))
    if not pts:
        return None

    best = None  # (score, template_name, dir, head, placed)
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed.issubset(pts):
                    leftover = len(pts - placed)
                    score = (len(placed), -leftover)
                    if best is None or score > best[0]:
                        best = (score, tname, d, (hy, hx), placed)

    if best is None:
        return None
    _, tname, d, head, placed = best
    markers = pts - placed
    return tname, d, head, set(map(tuple, placed)), set(map(tuple, markers))


def _detect_actor(frames: list[np.ndarray], bg: int, div_colors: set[int]) -> int:
    colors = set(map(int, np.concatenate([frame.ravel() for frame in frames])))
    best: tuple[tuple[int, int], int] | None = None
    for color in sorted(colors - {bg} - div_colors):
        detections = [_find_agent(frame, color) for frame in frames]
        count = sum(found is not None for found in detections)
        if count == 0:
            continue
        marker_total = sum(len(found[4]) for found in detections if found is not None)
        score = (count, marker_total)
        if best is None or score > best[0]:
            best = (score, color)
    assert best is not None, "No actor-colored sprite detected"
    return best[1]


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    offs: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, tuple[int, int]] | None:
    hy, hx = head

    def candidates(dir_idx: int) -> list[tuple[int, tuple[int, int]]]:
        out: list[tuple[int, tuple[int, int]]] = []
        for y, x in markers:
            if dir_idx == 0 and x == hx and y < hy:
                out.append((hy - y, (y, x)))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
        out.sort()
        return out

    def fits(target: tuple[int, int]) -> bool:
        ty, tx = target
        dy, dx = ty - hy, tx - hx
        for oy, ox in offs:
            ny = hy + oy + dy
            nx = hx + ox + dx
            if not (0 <= ny < h and 0 <= nx < w):
                return False
        return True

    # Collect all reachable markers (on any half-line, fitting in grid).
    reachable: list[tuple[int, int, tuple[int, int]]] = []  # (dist, dir, target)
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    # At most one direction should have candidates.
    dirs_with_candidates = set(dd for _, dd, _ in reachable)
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    # Pick the nearest on that direction.
    reachable.sort()
    _, dd, pt = reachable[0]

    # The task family never uses immediate 180 reversals.
    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}

        # Markers are only consumed when the head lands on them.
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through its remaining aligned markers and return the final frame.

    Input structure:
    - Frames are concatenated horizontally and optional full-height divider columns separate them.
    - Each frame contains one triangle sprite plus same-colored single-pixel future landing points; across the full history, the move directions form a palindrome.

    Rule:
    - Read only the rightmost snapshot as the current state.
    - From the sprite head, exactly one cardinal half-line contains reachable markers; move to the nearest one, consume it, and rotate the sprite to that travel direction.
    - Continue until no markers remain, then output a clean single frame containing only the final sprite.

    Key insight:
    - The mirrored outbound-and-return route is encoded globally across frames, but each local move is still determined by the triangle tip and the unique aligned marker direction.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    # Actor color is the non-bg, non-divider color.
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    actor = _detect_actor(frames, bg, div_colors)

    # Use the latest frame (rightmost) as the starting state.
    start = frames[-1]
    return _simulate_to_end(start, actor)


# --------------- generator (palindromic direction-sequence variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)  # rng.integers low/high+1 -> 1..4 frames
FRAME_MAX = 16  # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15  # clamps for computed H lower bound
W_FLOOR, W_CAP = 6, 15  # clamps for computed W lower bound

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)  # candidate total_steps (4..12 inclusive)
STEP_DIST_RANGE = (2, 8)  # per-step manhattan distance choices (2..7 inclusive)
PREFIX_SEGMENTS_RANGE = (2, 5)  # 2..4 segments before mirroring, capped by prefix length
MIN_TURNS_ACCEPT = 2  # rejects degenerate straight-line palindromes

# --- CFG: template selection ---
# Weights applied when n_steps >= TEMPLATE_HEAVY_THRESHOLD, in TEMPLATES order:
# (hollow3, filled4, stem4, midfilled6). Otherwise uniform.
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: retry budget ---
MAX_ATTEMPTS = 3000
DIR_ATTEMPTS = 80

# Precomputed: how many divisors in 1..4 each step count has (used to reweight n_steps
# so that step counts compatible with fewer frame counts aren't over-represented).
_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(tname: str, d: int, actor: int, bg: int) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = actor
    return patch, -r0, -c0


def _add_distractors(
    rng: np.random.Generator, frame: np.ndarray, bg: int, color: int, count: int = 1
) -> np.ndarray | None:
    out = frame.copy()
    occupied = set(map(tuple, np.argwhere(frame != bg)))
    candidates = [
        (r, c)
        for r in range(frame.shape[0])
        for c in range(frame.shape[1])
        if frame[r, c] == bg and not any(_touching((r, c), cell) for cell in occupied)
    ]
    if len(candidates) < count:
        return None
    picks = rng.choice(len(candidates), size=count, replace=False)
    for idx in np.atleast_1d(picks):
        r, c = candidates[int(idx)]
        out[r, c] = color
    return out


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """Cardinal direction from a to b, or None if not axis-aligned."""
    ar, ac = a
    br, bc = b
    if ac == bc and br < ar:
        return 0
    if ar == br and bc > ac:
        return 1
    if ac == bc and br > ar:
        return 2
    if ar == br and bc < ac:
        return 3
    return None


def _dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _touching(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _shift_interval(
    placements: list[tuple[tuple[int, int], int]], h: int, w: int, tname: str
) -> tuple[int, int, int, int] | None:
    """Interval of (tr, tc) translations that keep every placement's template in-grid."""
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d in placements:
        r0, r1, c0, c1 = _template_bounds(tname, d)
        tr_lo = max(tr_lo, -hr - r0)
        tr_hi = min(tr_hi, h - 1 - hr - r1)
        tc_lo = max(tc_lo, -hc - c0)
        tc_hi = min(tc_hi, w - 1 - hc - c1)
        if tr_lo > tr_hi or tc_lo > tc_hi:
            return None
    return tr_lo, tr_hi, tc_lo, tc_hi


def _sample_params(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, actor, divider, tname)."""
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n**3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = [int(c) for c in rng.permutation(10) if int(c) != DISTRACTOR_COLOR]
    bg, actor = palette[0], palette[1]
    divider = palette[2] if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


def _count_turns(step_dirs: list[int]) -> int:
    return sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs)))


def _is_palindrome(step_dirs: list[int]) -> bool:
    return step_dirs == step_dirs[::-1]


def _mirror_prefix(prefix: list[int], n_steps: int) -> list[int]:
    if n_steps % 2 == 0:
        return prefix + prefix[::-1]
    return prefix + prefix[-2::-1]


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int] | None:
    """Produce a palindromic length-n direction sequence with no 180-degree adjacencies."""
    prefix_len = (n_steps + 1) // 2
    max_segments = min(PREFIX_SEGMENTS_RANGE[1] - 1, prefix_len)
    min_segments = min(PREFIX_SEGMENTS_RANGE[0], max_segments)
    if max_segments < 2:
        return None

    for _ in range(DIR_ATTEMPTS):
        n_segments = int(rng.integers(min_segments, max_segments + 1))
        seg_dirs = [int(rng.integers(0, 4))]
        for _ in range(1, n_segments):
            seg_dirs.append(int((seg_dirs[-1] + rng.choice([-1, 1])) % 4))

        lengths = np.ones(n_segments, dtype=int)
        for _ in range(prefix_len - n_segments):
            lengths[int(rng.integers(0, n_segments))] += 1

        prefix: list[int] = []
        for d, ln in zip(seg_dirs, lengths):
            prefix.extend([int(d)] * int(ln))

        step_dirs = _mirror_prefix(prefix, n_steps)
        if not _is_palindrome(step_dirs):
            continue
        if any(step_dirs[i] == (step_dirs[i - 1] + 2) % 4 for i in range(1, n_steps)):
            continue
        if len(set(step_dirs)) < 2:
            continue
        if _count_turns(step_dirs) < MIN_TURNS_ACCEPT:
            continue
        return step_dirs
    return None


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
    """Greedy path rollout. Returns (heads, state_dirs, fit_placements) or None on stuck step."""
    n_steps = len(step_dirs)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    fit_placements: list[tuple[tuple[int, int], int]] = [((0, 0), d0)]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        if next_dir == (cur_dir + 2) % 4:
            return None

        dr, dc = DIRS4[next_dir]
        dist_order = list(rng.permutation(np.arange(*STEP_DIST_RANGE)))
        chosen = None
        for dist in dist_order:
            nxt = (cur[0] + int(dist) * dr, cur[1] + int(dist) * dc)
            if nxt in heads:
                continue
            if any(_touching(nxt, m) for m in heads[1:]):
                continue
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, step_dirs, tname):
                continue
            trial_fit = fit_placements + [(nxt, cur_dir), (nxt, next_dir)]
            if _shift_interval(trial_fit, h, w, tname) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        fit_placements.extend([(chosen, cur_dir), (chosen, next_dir)])

    return heads, state_dirs, fit_placements


def _marker_compatible_with_existing(
    nxt: tuple[int, int],
    heads: list[tuple[int, int]],
    state_dirs: list[int],
    step_dirs: list[int],
    tname: str,
) -> bool:
    """Validate candidate marker against all previously-placed heads.

    Ensures: no sprite-body adjacency, no alternative half-line from a past head
    (would create ambiguity at that step), and no nearer marker on that half-line
    (would be consumed first).
    """
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, tname, state_dirs[j])
        if any(_touching(nxt, cell) for cell in sprite):
            return False
        rel_dir = _direction(old_head, nxt)
        if rel_dir is None:
            continue
        if rel_dir != step_dirs[j]:
            return False
        if j < len(heads) - 1 and _dist(old_head, nxt) <= _dist(old_head, heads[j + 1]):
            return False
    return True


def _validate_path(step_dirs: list[int], heads: list[tuple[int, int]]) -> tuple[bool, int, list[int]]:
    """Check variant-specific path constraints. Returns (ok, turns, step_distances)."""
    n_steps = len(step_dirs)
    if not _is_palindrome(step_dirs):
        return False, 0, []
    if any(step_dirs[i] == (step_dirs[i - 1] + 2) % 4 for i in range(1, n_steps)):
        return False, 0, []
    if len(set(step_dirs)) < 2:
        return False, 0, []
    turns = _count_turns(step_dirs)
    if turns < MIN_TURNS_ACCEPT:
        return False, turns, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if not all(STEP_DIST_RANGE[0] <= d < STEP_DIST_RANGE[1] for d in dists):
        return False, turns, dists
    return True, turns, dists


def _render_frame(
    h: int,
    w: int,
    bg: int,
    actor: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = actor
    return frame


def _assemble_input(frames: list[np.ndarray], bg: int, divider: int | None) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if divider is not None and i < n_frames - 1:
            inp[:, c0 + w] = divider
    return inp


def _divider_layout_ok(inp: np.ndarray, bg: int, divider: int | None, frame_w: int, n_frames: int) -> bool:
    expected = {i * (frame_w + 1) + frame_w for i in range(n_frames - 1)}
    for c in range(inp.shape[1]):
        if len(set(map(int, inp[:, c]))) != 1:
            continue
        color = int(inp[0, c])
        if color == bg:
            continue
        if divider is None:
            return False
        if c not in expected or color != divider:
            return False
    return True


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)
    if step_dirs is None:
        return None

    rollout = _rollout_path(rng, tname, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, fit_placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(fit_placements, h, w, tname)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]
    placements = [(heads[i], state_dirs[i]) for i in range(n_steps + 1)]

    # Snapshots at steps 0, g, 2g, ..., (F-1)*g.
    frames = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))
    noisy_frames = []
    for frame in frames:
        noisy = _add_distractors(rng, frame, bg, DISTRACTOR_COLOR, count=1)
        if noisy is None:
            return None
        noisy_frames.append(noisy)
    frames = noisy_frames

    # Final output: just the sprite body, no markers.
    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    if not _divider_layout_ok(inp, bg, divider, w, n_frames):
        return None

    # Oracle checks: solver's agent detector must recover the last-frame sprite
    # exactly, the remaining markers must match, and end-to-end solve must match.
    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None
    if f_markers != set(heads[last_idx + 1 :]):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    mirror_match_ratio = sum(
        step_dirs[i] == step_dirs[-1 - i] for i in range(n_steps)
    ) / n_steps

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "palindrome_ok": _is_palindrome(step_dirs),
            "mirror_match_ratio": mirror_match_ratio,
            "prefix_length": (n_steps + 1) // 2,
            "parity": "odd" if n_steps % 2 else "even",
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
