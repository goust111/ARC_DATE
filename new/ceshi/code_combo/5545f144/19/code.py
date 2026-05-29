# Combo variant of 5545f144 (index 19)
# combo_spec: D10/0+D21/0
# Name: 精灵与标记不同色+干扰标记
#
# 设计分析
# - source_task: 原题 5545f144，加上两个 meta 维度改动：D10 让精灵与标记分色，D21 再加入不参与模拟的干扰点。
# - a) 具体化与行为追踪
#   - 原题最后一帧里，精灵和待吃标记共用一种颜色，solver 需要先从同色像素里拆出模板，再把剩余同色点当标记。
#   - 融合后最后一帧里至少有三类可见前景：精灵色、标记色、干扰色。比如蓝色三角头朝右，红色点在它右侧半线上，黄色点散落在别处。现在 solver 先用“哪种颜色能精确组成一个三角模板”识别精灵，再在剩余颜色里逐个尝试：只有把红色当标记时，才能沿唯一半线一路吃完；把黄色当标记会立刻卡住，因此黄色应被忽略。
#   - 差异不只是“多了一种颜色”，而是角色推断链变成两段：先按形状找精灵，再按可模拟性找真实标记色。
# - b) 复杂度校准
#   - 保留原题路径复杂度：4..12 步、至少 2 次转向、至少 3 个方向、步长有变化、禁止 180 度反转。
#   - 颜色自由度仍严格受控：背景、精灵、标记、干扰、可选分隔线。干扰只引入一种额外角色色，视觉差异明显但不会把颜色关系做得过散。
# - c) 隐式假设审查（main 相关）
#   - 原 `_split_frames` 把任意非背景纯色整列都当 divider；融合后前景颜色更多，这个假设不再安全，所以改成“只有能把整图切成等宽帧的一组整列”才算 divider。
#   - 原 `actor = next(iter(colors - {bg} - div_colors))` 假设只有一种角色色；融合后失效，因为最后一帧会出现精灵色、标记色和干扰色。
#   - 原 `_find_agent` 允许模板只是同色集合的子集；D10 后精灵色只属于精灵本体，所以这里改成“模板必须精确等于该颜色全部像素”，避免把同色噪声误并入主体。
#   - 新的 `_infer_roles` 引入一条新假设：除精灵色外，只有一种颜色在“作为标记参与模拟”时能完整吃完。生成器必须主动构造这种唯一性。
# - d) 放置合法性
#   - 继续 path-first：先采样头部轨迹，再把未来落点渲染成标记。
#   - 生成器保证：精灵始终在界内；真实标记不与任何时刻的精灵体接触；干扰点落在背景上、与所有非背景像素保持 8 邻域分离；干扰点不与当前帧头部同行同列，从而不会伪装成可达标记；帧内不存在整列纯前景，避免和 divider 混淆。
# - e) 视觉可观察性
#   - 人眼能直接看到三种职责：某一种颜色的三角在移动，另一种颜色的点被逐步吃掉，第三种颜色的点从头到尾都没有参与。
#   - 推理不依赖颜色编号大小规律；颜色只承担“角色标签”，角色由形状和动态可达性共同决定。
# - f) 可视化整洁度
#   - 标记与干扰都保持单像素且不贴近主体，避免多种前景糊成一团。
#   - 干扰数量控制在每帧 2..4 个，足够显眼，但不会盖过真实路径。
# - g) 变化的可推断性
#   - 每个样例中，精灵色会重复出现在所有帧的同一模板上；标记色会沿路径逐帧减少；干扰色保持为孤立噪声点。解题者在 2..3 组例子里可以稳定归纳出“不是所有彩色点都要被吃”。
# - 8) 变体行为量化指标
#   - 指标 1：最后输入帧（除背景、分隔线外）固定有 3 种角色色。
#   - 指标 2：恰有 1 种颜色能被精确模板检测为精灵色。
#   - 指标 3：在剩余颜色中，恰有 1 种颜色作为标记时能完成整段模拟；其它颜色都会失败。这同时验证了 D10 的分色和 D21 的干扰确实同时生效。
# - 9) 新 main 的隐式假设
#   - 起始帧里精灵色不会和其它噪声同色，且能唯一匹配一个模板。
#   - 真实标记色在起始帧中至少还有 1 个像素保留；干扰色虽然可见，但不会从起始头部构成合法的第一步。
#   - 若存在 divider，它们是一组等间距的全高纯色列；生成器用整洁列约束保证这一点。

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    """
    Input is one or more equal-width frames concatenated horizontally.
    Divider columns, when present, are full-height solid columns of a dedicated
    non-background color.
    """
    _h, w = g.shape
    bg = _bg_color(g)

    solid_by_color: dict[int, list[int]] = {}
    for c in range(w):
        vals = set(map(int, g[:, c]))
        if len(vals) != 1:
            continue
        color = next(iter(vals))
        if color == bg:
            continue
        solid_by_color.setdefault(int(color), []).append(c)

    candidates: list[list[int]] = []
    for cols in solid_by_color.values():
        if not (1 <= len(cols) <= 3):
            continue
        cuts = [-1] + cols + [w]
        widths = [cuts[i + 1] - cuts[i] - 1 for i in range(len(cuts) - 1)]
        if min(widths) <= 0:
            continue
        if len(set(widths)) != 1:
            continue
        candidates.append(cols)

    if len(candidates) != 1:
        return [g.copy()], bg

    div_cols = candidates[0]
    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
    return frames, bg


# Templates are defined as offsets relative to the head pixel at (0, 0).
TEMPLATES: dict[str, dict[int, list[tuple[int, int]]]] = {
    "hollow3": {
        0: [(0, 0), (1, -1), (1, 1)],
        1: [(0, 0), (-1, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 1)],
        3: [(0, 0), (-1, 1), (1, 1)],
    },
    "filled4": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    "stem4": {
        0: [(0, 0), (1, 0), (2, -1), (2, 1)],
        1: [(0, 0), (0, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, 0), (-2, -1), (-2, 1)],
        3: [(0, 0), (0, 1), (-1, 2), (1, 2)],
    },
    "midfilled6": {
        0: [(0, 0), (1, -1), (1, 0), (1, 1), (2, -1), (2, 1)],
        1: [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, -2), (1, -2)],
        2: [(0, 0), (-1, -1), (-1, 0), (-1, 1), (-2, -1), (-2, 1)],
        3: [(0, 0), (-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2)],
    },
}


def _find_agent(
    frame: np.ndarray, sprite_color: int
) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]] | None:
    pts = set(map(tuple, np.argwhere(frame == sprite_color)))
    if not pts:
        return None

    matches: list[tuple[str, int, tuple[int, int], set[tuple[int, int]]]] = []
    for head in pts:
        hy, hx = int(head[0]), int(head[1])
        for tname, dirs in TEMPLATES.items():
            for d, offs in dirs.items():
                placed = {(hy + dy, hx + dx) for dy, dx in offs}
                if placed == pts:
                    matches.append((tname, d, (hy, hx), placed))

    if len(matches) != 1:
        return None
    return matches[0]


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
            elif dir_idx == 1 and y == hy and x > hx:
                out.append((x - hx, (y, x)))
            elif dir_idx == 2 and x == hx and y > hy:
                out.append((y - hy, (y, x)))
            elif dir_idx == 3 and y == hy and x < hx:
                out.append((hx - x, (y, x)))
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

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    if len(dirs_with_candidates) != 1:
        return None

    reachable.sort()
    _, dd, pt = reachable[0]
    if dd == (d + 2) % 4:
        return None
    return dd, pt


def _simulate_with_roles(
    frame: np.ndarray, sprite_color: int, marker_color: int
) -> tuple[np.ndarray, int] | None:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, sprite_color)
    if found is None:
        return None

    tname, d, head, agent = found
    markers = set(map(tuple, np.argwhere(frame == marker_color)))
    markers = {(int(y), int(x)) for y, x in markers}
    if not markers:
        return None

    agent = {(int(y), int(x)) for y, x in agent}
    consumed = 0
    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        choice = _choose_target(head, d, markers, TEMPLATES[tname][d], h, w)
        if choice is None:
            return None
        d, head = choice
        agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)
        consumed += 1

    if markers:
        return None

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = sprite_color
    return out, consumed


def _marker_candidate_report(
    frame: np.ndarray, sprite_color: int
) -> dict[int, int]:
    bg = _bg_color(frame)
    report: dict[int, int] = {}
    for color in sorted(set(map(int, frame.ravel())) - {bg, sprite_color}):
        sim = _simulate_with_roles(frame, sprite_color, color)
        report[int(color)] = -1 if sim is None else int(sim[1])
    return report


def _infer_roles(
    frame: np.ndarray,
) -> tuple[int, int, tuple[int, ...], tuple[str, int, tuple[int, int], set[tuple[int, int]]]]:
    bg = _bg_color(frame)
    colors = sorted(set(map(int, frame.ravel())) - {bg})

    sprite_hits: list[tuple[int, tuple[str, int, tuple[int, int], set[tuple[int, int]]]]] = []
    for color in colors:
        found = _find_agent(frame, color)
        if found is not None:
            sprite_hits.append((int(color), found))

    assert len(sprite_hits) == 1, f"expected exactly 1 sprite color, got {sprite_hits}"
    sprite_color, found = sprite_hits[0]

    report = _marker_candidate_report(frame, sprite_color)
    marker_hits = [color for color, consumed in report.items() if consumed > 0]
    assert len(marker_hits) == 1, f"expected exactly 1 marker color, got report={report}"
    marker_color = marker_hits[0]
    distractor_colors = tuple(sorted(color for color in report if color != marker_color))
    assert distractor_colors, "expected at least one distractor color"
    return sprite_color, marker_color, distractor_colors, found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the triangular sprite simulation while ignoring unrelated colored noise.

    Input structure:
    - One or more equal-width frames are concatenated horizontally; if dividers
      exist, they are full-height solid columns that split the frames evenly.
    - In each frame, the triangle sprite has one dedicated color, the consumable
      single-pixel markers have a different color, and extra single-pixel
      distractors use at least one further non-background color.

    Rule:
    - Read the rightmost frame as the current state.
    - Identify the sprite by the color whose pixels exactly match one rotated
      triangle template. Among the remaining colors, only one behaves like real
      markers: from the current head, its pixels can be consumed one by one by
      repeatedly moving to the nearest aligned pixel on the unique valid
      cardinal half-line, without 180-degree reversals.
    - Simulate that chase until those markers are exhausted, ignore the other
      colored dots, and output only the final sprite on a clean background.

    Key insight: shape identifies the moving object, while successful path
    simulation separates true targets from same-sized distractor dots.
    """
    g = np.asarray(grid, dtype=int)
    frames, _bg = _split_frames(g)

    start = frames[-1]
    sprite_color, marker_color, _distractor_colors, _found = _infer_roles(start)
    sim = _simulate_with_roles(start, sprite_color, marker_color)
    assert sim is not None
    out, _consumed = sim
    return out


# --------------- generator (D10/0 + D21/0 fused variant) ---------------

# --- CFG: canvas / frame bounds ---
# Keep at least 2 frames so the original temporal cue remains observable.
N_FRAMES_RANGE = (2, 5)
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: template selection ---
TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

# --- CFG: distractors ---
DISTRACTOR_RANGE = (2, 5)  # 2..4 per frame

# --- CFG: retry budget ---
MAX_ATTEMPTS = 1200

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(1, 5)) for n in range(*STEPS_RANGE)}


def _template_bounds(tname: str, d: int) -> tuple[int, int, int, int]:
    offs = TEMPLATES[tname][d]
    rs = [dr for dr, _ in offs]
    cs = [dc for _, dc in offs]
    return min(rs), max(rs), min(cs), max(cs)


def _template_patch(
    tname: str, d: int, sprite_color: int, bg: int
) -> tuple[np.ndarray, int, int]:
    r0, r1, c0, c1 = _template_bounds(tname, d)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for dr, dc in TEMPLATES[tname][d]:
        patch[dr - r0, dc - c0] = sprite_color
    return patch, -r0, -c0


def _sprite_cells(head: tuple[int, int], tname: str, d: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[tname][d]}


def _direction(a: tuple[int, int], b: tuple[int, int]) -> int | None:
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
) -> tuple[int, int, int, int, int, int, int, int, int, int | None, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, sprite, marker, distractor, divider, tname)."""
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

    palette = rng.permutation(10)
    bg = int(palette[0])
    sprite_color = int(palette[1])
    marker_color = int(palette[2])
    distractor_color = int(palette[3])
    divider = int(palette[4]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return (
        n_frames,
        n_steps,
        gap,
        h,
        w,
        bg,
        sprite_color,
        marker_color,
        distractor_color,
        divider,
        tname,
    )


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    n_turns = int(rng.integers(turn_min, turn_max + 1))
    n_segments = n_turns + 1

    segments = [int(rng.integers(0, 4))]
    for _ in range(n_segments - 1):
        segments.append(int((segments[-1] + rng.choice([1, -1])) % 4))

    lengths = np.ones(n_segments, dtype=int)
    for _ in range(n_steps - n_segments):
        lengths[int(rng.integers(0, n_segments))] += 1

    out: list[int] = []
    for d, ln in zip(segments, lengths):
        out.extend([d] * int(ln))
    return out


def _rollout_path(
    rng: np.random.Generator, tname: str, step_dirs: list[int], h: int, w: int
) -> tuple[list[tuple[int, int]], list[int], list[tuple[tuple[int, int], int]]] | None:
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
    n_steps = len(step_dirs)
    dir_set = set(step_dirs)
    if len(dir_set) < MIN_DIR_DIVERSITY:
        return False, 0, []
    if not (dir_set & {0, 2} and dir_set & {1, 3}):
        return False, 0, []
    turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, n_steps))
    if turns < MIN_TURNS_ACCEPT or not (TURN_RATIO_LO <= turns / n_steps <= TURN_RATIO_HI):
        return False, turns, []
    dists = [_dist(heads[i], heads[i + 1]) for i in range(n_steps)]
    if len(set(dists)) < MIN_DIST_VARIETY:
        return False, turns, dists
    return True, turns, dists


def _render_frame(
    h: int,
    w: int,
    bg: int,
    sprite_color: int,
    marker_color: int,
    head: tuple[int, int],
    d: int,
    tname: str,
    markers: list[tuple[int, int]],
) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, sprite_color, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        frame[m] = marker_color
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


def _has_internal_solid_nonbg_column(frame: np.ndarray, bg: int) -> bool:
    for c in range(frame.shape[1]):
        vals = set(map(int, frame[:, c]))
        if len(vals) == 1 and next(iter(vals)) != bg:
            return True
    return False


def _add_distractors(
    rng: np.random.Generator,
    frame: np.ndarray,
    bg: int,
    color: int,
    count: int,
    avoid_row: int,
    avoid_col: int,
) -> np.ndarray | None:
    out = frame.copy()
    occupied = set(map(tuple, np.argwhere(out != bg)))
    candidates = [
        (r, c)
        for r in range(out.shape[0])
        for c in range(out.shape[1])
        if out[r, c] == bg and r != avoid_row and c != avoid_col
    ]
    rng.shuffle(candidates)

    placed = 0
    for r, c in candidates:
        cell = (int(r), int(c))
        if any(_touching(cell, other) for other in occupied):
            continue
        out[cell] = color
        occupied.add(cell)
        placed += 1
        if placed == count:
            return out
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        n_frames,
        n_steps,
        gap,
        h,
        w,
        bg,
        sprite_color,
        marker_color,
        distractor_color,
        divider,
        tname,
    ) = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

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

    frames: list[np.ndarray] = []
    distractor_counts: list[int] = []
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        frame = _render_frame(
            h,
            w,
            bg,
            sprite_color,
            marker_color,
            head,
            d,
            tname,
            heads[si + 1 :],
        )
        distractor_count = int(rng.integers(*DISTRACTOR_RANGE))
        frame = _add_distractors(
            rng,
            frame,
            bg,
            distractor_color,
            distractor_count,
            avoid_row=head[0],
            avoid_col=head[1],
        )
        if frame is None:
            return None
        if _has_internal_solid_nonbg_column(frame, bg):
            return None
        frames.append(frame)
        distractor_counts.append(distractor_count)

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, sprite_color, marker_color, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    last_idx = (n_frames - 1) * gap
    found = _find_agent(frames[-1], sprite_color)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent = found
    if (f_tname, f_dir, f_head) != (tname, placements[last_idx][1], placements[last_idx][0]):
        return None

    last_markers = set(map(tuple, np.argwhere(frames[-1] == marker_color)))
    if last_markers != set(heads[last_idx + 1 :]):
        return None

    sprite2, marker2, distractors2, _found2 = _infer_roles(frames[-1])
    if (sprite2, marker2) != (sprite_color, marker_color):
        return None
    if distractors2 != (distractor_color,):
        return None

    report = _marker_candidate_report(frames[-1], sprite_color)
    if report.get(marker_color, -1) != gap:
        return None
    if any(consumed != -1 for color, consumed in report.items() if color != marker_color):
        return None

    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": n_frames,
            "template": tname,
            "bg": bg,
            "sprite_color": sprite_color,
            "marker_color": marker_color,
            "distractor_color": distractor_color,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "distractor_counts": distractor_counts,
            "role_colors_in_last_frame": 3,
            "marker_candidate_report": report,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
