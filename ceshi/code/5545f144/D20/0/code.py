# Variant of 5545f144
# Dimension: D20 — 分隔线属性 (grid_shape)
# Change: 帧间分隔线为单列、全高、单一颜色（divider色），当只有1帧时无分隔线。...
#         → 分隔线颜色编码帧间关系；不同颜色表示相邻快照之间消耗的标记数量。

# Design analysis
# ---------------
# spec:
#   concretization:
#     - 帧仍然是横向拼接的单张局部网格，分隔线仍然保持“单列、全高”这个几何外观。
#     - 关键变化是分隔线不再只是装饰：固定使用三个可见颜色档位，分别表示
#       相邻两帧之间精灵吃掉了 1/2/3 个标记。
#     - 每一帧内部仍然画出当前朝向的三角精灵，以及从该时刻往后的全部单像素标记。
#       因此，相邻帧的标记数差，正好就是分隔线颜色所编码的 consumed count。
#   behavior_trace:
#     - 原题: 若三帧分别是 step 0 / 3 / 6，则右侧最后一帧还剩 3 个标记；solver
#       只看最右帧，直接模拟到所有标记耗尽，得到输出。
#     - 本变体: 例如三帧是 step 0 / 2 / 5，两个分隔线分别是“消耗2个”“消耗3个”。
#       输出不是一路跑到终点，而是把最右帧再向前推进“最近一次观测到的步长”3 步，
#       也就是预测下一张快照。若后面还藏着额外标记，原题 solver 会继续跑过头，
#       与本变体输出不同，因而 divider 颜色真正参与求解。
#   complexity_calibration:
#     - 若只把分隔线换成多种颜色，但输出仍是“最右帧一路跑到终点”，原 solver
#       的拆帧与 actor 检测代码几乎不用改，divider 颜色会退化成无关噪声。
#     - 为了让 D20 可观察且不引入过多自由度，颜色映射固定为 3 档，且每个样例
#       至少出现两种不同 divider 颜色；这样人能直接通过相邻帧的标记数差看出映射。
#   implicit_assumptions_review:
#     - `_split_frames` 原本只依赖“非背景且整列纯色”，不依赖 divider 同色；
#       所以这里只需要把 divider 颜色列表保留下来，而不是单个 divider 色。
#     - `main()` 原本默认“最右帧就是距离输出还差固定终局 gap 的状态”，这个前提
#       在变体里失效；现在必须读取最后一个 divider 的颜色，得到要外推的步数。
#     - actor 颜色不能与 divider 调色板混淆。生成器因此把 actor/bg 限制在
#       非-divider 颜色里，并在 oracle 检查时再次验证拆帧结果与 divider 序列一致。
#   placement_legality:
#     - 路径生成仍沿用原题的“先造整条合法路径，再挑时间点截帧”。
#     - 每一步都保持：目标标记落在唯一半线上、不会 180 度回头、精灵旋转后完整落在网格内。
#     - 由于输出只前进一步长而非到终点，生成器还额外要求输出之后至少剩 1 个隐藏标记，
#       并拒绝“继续跑到终点后恰好回到同一姿态”的退化样例。
#   visual_observability:
#     - 颜色映射不是靠颜色 id 排序，而是靠结构线索推断：两帧标记数差 = divider 颜色含义。
#     - 每个样例固定 3 帧，保证恰好有两个可比较的 divider；并要求这两种颜色不同，
#       这样单个样例内部就能支撑“颜色 -> 步数”这个规律。
#   cleanliness:
#     - divider 仍是一列纯色，不会与帧内对象重叠。
#     - actor 与 markers 继续共色，divider 颜色单独占据列状区域，视觉层次清晰。
#   main_assumptions_after_change:
#     - `main()` 现在假设: 至少两帧时，最后一个 divider 颜色属于 1/2/3 这三个档位。
#     - 还假设: 最右帧内的剩余标记数不少于该档位对应的步数。
#     - `generate_input()` 通过固定 divider 调色板、设置 hidden tail > 0、并用
#       `main(input) == expected` + “原终局解 != 变体输出” 双重验证来保证这些前提。

from __future__ import annotations

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left

# Divider palette is deliberately small and fixed so humans can infer the mapping
# from marker-count differences between adjacent frames.
DIVIDER_COUNT_BY_COLOR = {
    1: 1,  # blue  -> consume 1 marker between snapshots
    2: 2,  # red   -> consume 2 markers between snapshots
    4: 3,  # yellow-> consume 3 markers between snapshots
}
DIVIDER_COLOR_BY_COUNT = {count: color for color, count in DIVIDER_COUNT_BY_COLOR.items()}
DIVIDER_COLORS = tuple(sorted(DIVIDER_COUNT_BY_COLOR))


def _split_frames(g: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    """
    Input is multiple frames concatenated horizontally, separated by full-height
    divider columns. Return frames, divider colors (left-to-right), and bg.
    """
    h, w = g.shape
    bg = _bg_color(g)

    solid_cols = [c for c in range(w) if len(set(map(int, g[:, c]))) == 1]
    div_cols = [c for c in solid_cols if int(g[0, c]) != bg]

    cuts = [-1] + div_cols + [w]
    frames: list[np.ndarray] = []
    divider_colors: list[int] = []
    for i in range(len(cuts) - 1):
        a = cuts[i] + 1
        b = cuts[i + 1]
        frames.append(g[:, a:b])
        if i < len(div_cols):
            divider_colors.append(int(g[0, div_cols[i]]))
    return frames, divider_colors, bg


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

    dirs_with_candidates = set(dd for _, dd, _ in reachable)
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


def _simulate_steps(frame: np.ndarray, actor: int, steps: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    agent = set((int(y), int(x)) for y, x in agent)
    markers = set((int(y), int(x)) for y, x in markers)

    assert steps <= len(markers), (
        f"divider requests {steps} steps, but only {len(markers)} markers remain"
    )

    for _ in range(steps):
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        offs = TEMPLATES[tname][d]
        agent = {(ty + oy, tx + ox) for oy, ox in offs}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _simulate_to_end(frame: np.ndarray, actor: int) -> np.ndarray:
    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()
    return _simulate_steps(frame, actor, len(found[4]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next triangular-sprite snapshot from color-coded frame dividers.

    Input structure:
    - Frames are concatenated horizontally, with full-height one-column dividers.
    - Each divider color encodes how many markers were consumed between the frame
      on its left and the frame on its right; each frame itself shows one actor-colored
      triangle plus all still-unconsumed actor-colored single-pixel markers.

    Rule:
    - Read the frames left-to-right as chronological snapshots of the same motion.
    - The divider immediately before the rightmost frame gives the step count for
      the next extrapolation interval (blue=1, red=2, yellow=3).
    - Starting from the rightmost frame, advance the sprite by exactly that many
      marker-consumption moves, turning toward the unique reachable half-line marker
      at each move, and output only the resulting sprite body on clean background.

    Key insight: the divider is no longer decorative; its color tells how far past the
    latest visible frame to simulate, while the triangle tip still determines the path.
    """
    g = np.asarray(grid, dtype=int)
    frames, divider_colors, bg = _split_frames(g)

    colors = set(map(int, g.ravel()))
    actor_colors = colors - {bg} - set(divider_colors)
    assert len(actor_colors) == 1, f"expected exactly one actor color, got {actor_colors}"
    actor = next(iter(actor_colors))

    start = frames[-1]
    if not divider_colors:
        return _simulate_to_end(start, actor)

    assert divider_colors[-1] in DIVIDER_COUNT_BY_COLOR, (
        f"unknown divider color {divider_colors[-1]}"
    )
    last_gap = DIVIDER_COUNT_BY_COLOR[int(divider_colors[-1])]
    return _simulate_steps(start, actor, last_gap)


# --------------- generator ---------------

N_FRAMES = 3
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 2
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

GAP_COUNTS = (1, 2, 3)
HIDDEN_TAIL_RANGE = (1, 4)  # 1..3 extra hidden markers after the predicted output
MAX_ATTEMPTS = 400


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
) -> tuple[int, int, int, int, int, int, str, list[int], list[int], int, int]:
    gap1, gap2 = map(int, rng.choice(np.array([(a, b) for a in GAP_COUNTS for b in GAP_COUNTS if a != b])))
    hidden_tail = int(rng.integers(*HIDDEN_TAIL_RANGE))
    output_step = gap1 + 2 * gap2
    n_steps = output_step + hidden_tail

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    free_palette = [c for c in range(10) if c not in DIVIDER_COLORS]
    palette = list(map(int, rng.permutation(free_palette)))
    bg, actor = palette[0], palette[1]

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))

    frame_steps = [0, gap1, gap1 + gap2]
    divider_counts = [gap1, gap2]
    divider_colors = [DIVIDER_COLOR_BY_COUNT[count] for count in divider_counts]
    return n_steps, h, w, bg, actor, hidden_tail, tname, frame_steps, divider_colors, gap2, output_step


def _make_step_dirs(rng: np.random.Generator, n_steps: int) -> list[int]:
    turn_min = max(TURN_MIN_ABS, int(np.ceil(TURN_RATIO_LO * n_steps)))
    turn_max = min(n_steps - 1, int(np.floor(TURN_RATIO_HI * n_steps)))
    if turn_min > turn_max:
        return []
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
    if n_steps == 0:
        return None

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


def _assemble_input(frames: list[np.ndarray], bg: int, divider_colors: list[int]) -> np.ndarray:
    n_frames = len(frames)
    h, w = frames[0].shape
    total_w = n_frames * w + (n_frames - 1)
    inp = np.full((h, total_w), bg, dtype=int)
    for i, fr in enumerate(frames):
        c0 = i * (w + 1)
        paste(inp, fr, 0, c0)
        if i < n_frames - 1:
            inp[:, c0 + w] = divider_colors[i]
    return inp


def _build_case(rng: np.random.Generator) -> dict | None:
    (
        n_steps,
        h,
        w,
        bg,
        actor,
        hidden_tail,
        tname,
        frame_steps,
        divider_colors,
        last_gap,
        output_step,
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
    for si in frame_steps:
        head, d = placements[si]
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, heads[si + 1 :]))

    out_head, out_dir = placements[output_step]
    out_frame = _render_frame(h, w, bg, actor, out_head, out_dir, tname, [])
    inp = _assemble_input(frames, bg, divider_colors)

    parsed_frames, parsed_dividers, parsed_bg = _split_frames(inp)
    if parsed_bg != bg or parsed_dividers != divider_colors or len(parsed_frames) != N_FRAMES:
        return None

    found = _find_agent(frames[-1], actor)
    if found is None:
        return None
    f_tname, f_dir, f_head, _f_agent, f_markers = found
    last_frame_step = frame_steps[-1]
    if (f_tname, f_dir, f_head) != (tname, placements[last_frame_step][1], placements[last_frame_step][0]):
        return None
    if f_markers != set(heads[last_frame_step + 1 :]):
        return None
    if len(f_markers) != last_gap + hidden_tail:
        return None

    terminal_from_last = _simulate_to_end(frames[-1], actor)
    if np.array_equal(terminal_from_last, out_frame):
        return None
    if not np.array_equal(main(inp), out_frame):
        return None

    return {
        "input": inp,
        "output": out_frame,
        "meta": {
            "frame_height": h,
            "frame_width": w,
            "n_frames": N_FRAMES,
            "template": tname,
            "bg": bg,
            "actor": actor,
            "divider_colors": list(divider_colors),
            "divider_counts": [DIVIDER_COUNT_BY_COLOR[c] for c in divider_colors],
            "total_steps": n_steps,
            "frame_steps": list(frame_steps),
            "last_gap": last_gap,
            "output_step": output_step,
            "hidden_tail": hidden_tail,
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "heads": list(heads),
            "step_distances": dists,
            "markers_in_last_frame": len(f_markers),
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
