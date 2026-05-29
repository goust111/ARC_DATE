# Variant of 5545f144
# Dimension: D10 — 颜色使用策略 (meta)
# Change: 只有三种颜色角色：背景色、actor色（精灵+标记共用）、分隔线色；精灵和标记不可色彩区分。... → 精灵与标记不同色
#
# 设计分析
# - source_task: None。本变体只围绕 D10 改颜色角色，不引入额外来源任务机制。
# - a) 具体化与行为追踪
#   - 输入仍是若干帧横向拼接，帧间仍用整列分隔线隔开；输出仍是单帧终态。
#   - 每一帧里现在固定有两类前景：
#     - 精灵色：一个三角/箭头模板，颜色在整题内保持一致。
#     - 标记色：若干单像素点，颜色与精灵色不同。
#   - 小例子：若最后一帧里蓝色精灵头朝右，红点只出现在它右边同一行的半线上，那么旧逻辑会先把“所有蓝色像素”拆成精灵模板 + 剩余蓝点；新逻辑则直接从蓝色里识别模板，再从红色像素集合里取标记。移动、转向、只在头落点处消耗标记，这些动态规则不变。
#   - 差异集中在“状态解码”环节：旧题靠同色前景内部再分离精灵/标记；新题靠颜色先分角色，再做原来的路径模拟。
# - b) 复杂度校准
#   - 不能因为颜色分离就把路径做得太简单，所以保留原题的复杂路径约束：4..12 步、至少 2 次转向、至少 3 个方向、步长要有变化、禁止 180 度反转。
#   - 颜色自由度控制在非常少的角色色：背景、精灵、标记、以及可选分隔线。人眼可稳定看出“谁在动、谁被吃掉”。
# - c) 隐式假设审查（原 main 相关）
#   - 原 `_split_frames` 默认“非背景整列”就是分隔线。变体里多了一种前景色，所以生成器必须避免帧内部出现整列纯前景；同时 solver 也改成只把能切成等宽帧的整列颜色当 divider。
#   - 原 `actor = next(iter(colors - {bg} - div_colors))` 假设只有一种非背景非分隔线颜色；变体失效，因为现在应有两种角色色。
#   - 原 `_simulate_to_end(start, actor)` 假设 `_find_agent` 返回的 leftover 就是标记；变体失效，因为标记不再和精灵同色，必须单独取 `frame == marker_color`。
#   - 原 `_find_agent` 允许“模板是同色点集的子集”。旧题需要这样做，因为同色集中还混着标记；新题里精灵色只属于精灵，所以改成“模板必须精确等于该颜色所有像素”，避免把标记色误识别成精灵。
# - d) 放置合法性
#   - 继续沿用 path-first 生成：先采样头部轨迹，再把未来落点渲染成标记。
#   - 生成器仍保证：每一步整只精灵都在边界内；标记不与任何时刻的精灵体接触；未来标记不会给过去状态制造多方向歧义；分隔线颜色独立；每帧内部不存在整列纯前景，避免与 divider 混淆。
# - e) 视觉可观察性
#   - 变体的可见变化必须直接体现在像素上：输入帧里“会移动的精灵”和“会被消耗的点”始终是两种颜色。
#   - 颜色 id 不参与排序或变换规则；颜色只承担角色标签。
# - f) 可视化整洁度
#   - 标记仍是孤立单像素，不与精灵体重叠，也不贴边接触，避免两种前景糊成一团。
#   - 输出只保留精灵色，不保留标记色，终态一眼可见。
# - g) 颜色变化可推断性
#   - 每个样例中只有一个精灵色和一个标记色，且它们在所有输入帧中反复出现。人类从 2-3 个例子即可归纳“红点/绿点是目标，蓝形状/黄形状是主体”。
# - 8) 变体行为量化指标
#   - 指标 1：最后输入帧的非背景角色色数量应恒为 2。
#   - 指标 2：恰有一种角色色能被模板检测为精灵，另一种只能解释为标记。
#   - 指标 3：最后输入帧保留的标记数 `gap` 仍为正，因此颜色分离在每个样例中都实际可见，而不是只在早期帧出现。
# - 9) 新 main 的隐式假设
#   - 右侧起始帧中恰有两种非背景角色色。
#   - 其中仅一种颜色形成完整模板；生成器用“标记孤立且不接触”保证这一点。
#   - 若存在分隔线，它们能把总图切成等宽帧；生成器保证 divider 仍是一列宽、全高纯色。

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


def _infer_role_colors(
    frame: np.ndarray,
) -> tuple[int, int, tuple[str, int, tuple[int, int], set[tuple[int, int]]]]:
    bg = _bg_color(frame)
    colors = sorted(set(map(int, frame.ravel())) - {bg})
    assert len(colors) == 2, f"expected 2 role colors in start frame, got {colors}"

    found: list[tuple[int, tuple[str, int, tuple[int, int], set[tuple[int, int]]]]] = []
    for color in colors:
        agent = _find_agent(frame, color)
        if agent is not None:
            found.append((color, agent))

    assert len(found) == 1, f"expected exactly 1 sprite color, got {found}"
    sprite_color, agent = found[0]
    marker_color = next(color for color in colors if color != sprite_color)
    return sprite_color, marker_color, agent


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

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if fits(pt):
                reachable.append((dist, dd, pt))

    if not reachable:
        return None

    dirs_with_candidates = {dd for _, dd, _ in reachable}
    assert len(dirs_with_candidates) == 1, (
        f"head={head} dir={d}: markers reachable in multiple directions "
        f"{dirs_with_candidates}, expected exactly one"
    )

    reachable.sort()
    _, dd, pt = reachable[0]

    assert dd != (d + 2) % 4, (
        f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    )

    return dd, pt


def _simulate_to_end(frame: np.ndarray, sprite_color: int, marker_color: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, sprite_color)
    if found is None:
        return frame.copy()

    tname, d, head, agent = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set(map(tuple, np.argwhere(frame == marker_color)))
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
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = sprite_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance a triangular sprite through a sequence of colored waypoint dots.

    Input structure:
    - One or more equal-width frames are concatenated horizontally; optional
      divider columns are full-height solid stripes.
    - Inside each frame, the sprite uses one color and the remaining waypoint
      dots use another color, so the two roles are visually distinct.

    Rule:
    - Start from the rightmost snapshot.
    - From the sprite head, look for waypoint dots on cardinal half-lines
      (same row or column, only on one side). Exactly one direction is valid;
      move to the nearest dot on that direction.
    - When the head lands on that dot, remove the dot and rotate the sprite to
      face the movement direction. Continue until no waypoint dots remain.

    Key insight: color now separates object roles, so the task is to read the
    sprite orientation from its shape and then simulate its dot-chasing path.
    """
    g = np.asarray(grid, dtype=int)
    frames, _bg = _split_frames(g)

    start = frames[-1]
    sprite_color, marker_color, _found = _infer_role_colors(start)
    return _simulate_to_end(start, sprite_color, marker_color)


# --------------- generator (D10 color-role split variant) ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (1, 5)
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

# --- CFG: retry budget ---
MAX_ATTEMPTS = 500

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
) -> tuple[int, int, int, int, int, int, int, int, int | None, str]:
    """Sample (n_frames, n_steps, gap, h, w, bg, sprite, marker, divider, tname)."""
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
    divider = int(palette[3]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, sprite_color, marker_color, divider, tname


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
        if _has_internal_solid_nonbg_column(frame, bg):
            return None
        frames.append(frame)

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
    inferred_sprite, inferred_marker, _ = _infer_role_colors(frames[-1])
    if (inferred_sprite, inferred_marker) != (sprite_color, marker_color):
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
            "role_colors_in_last_frame": 2,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
