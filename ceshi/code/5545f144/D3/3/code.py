# Variant of 5545f144
# Dimension: D3 — 目标对象：标记类型 (dst)
# Change: 散布的单像素标记点，与精灵共用同一颜色（actor色），被消耗后消失。... → 标记在消耗前不可见（隐藏标记）

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import paste

# Design analysis for the D3 variant
# a) 具体化与行为追踪:
#    仍然是“多帧横向拼接 + 三角精灵 + 单像素目标点”的画面；核心变化只发生在后续帧，
#    尤其最后一帧：仍未被吃掉的一部分目标点被画成背景色，因此最后一帧不再完整暴露剩余目标。
#    原逻辑是“从最后一帧读出所有剩余点并继续模拟”；新逻辑改为“从第一帧恢复完整目标序列，
#    再用最后一帧的精灵状态决定当前处于序列的哪个阶段，然后补回被隐藏的尾部目标继续模拟”。
#    小例子：若第一帧路径尾部还有 A->B，而最后一帧只显示精灵且把 A、B 都藏掉，
#    原版从最后一帧会误判“已经没有目标”，新逻辑则能从第一帧知道尾部仍有 A、B。
# b) 复杂度校准:
#    只增加一个隐藏维度，不引入新颜色、新对象、新排序。隐藏只发生在最后一帧，且总是至少隐藏“下一步要吃的点”，
#    这样视觉上一定和原题不同，同时不会把推理分叉扩展得过多。
# c) 隐式假设审查:
#    原版 main() 隐含依赖“最后一帧里所有剩余 marker 都可见”，体现在 _find_agent(frame)->markers
#    后直接 _simulate_to_end(start, actor)。该前提在本变体失效，因此 solver 不能把最后一帧 leftover 当作完整真相。
#    另一方面，_find_agent 对“精灵形状本身可见且与 marker 同色但不重叠”的假设仍成立，所以该部分保留。
# d) 放置合法性:
#    生成器仍沿用原来的 path-first 约束，保证每一步头部与整块模板都在框内、marker 不与精灵相撞、
#    且从任一状态出发只有一个合法半线方向。隐藏只是不渲染部分 marker，不改变其真实坐标。
# e) 视觉可观察性:
#    训练样例中，人可以看到前面帧里有一些将来会被隐藏的点，而最后一帧这些点突然“凭空少了”，
#    但输出仍要求精灵继续沿同一路径前进，因此“跨帧补全隐藏目标”是可观察的核心规律。
# f) 可视化整洁度:
#    不增加叠加层；隐藏 marker 直接退回背景色，画面只会更干净，不会引入新杂讯。
# g) 颜色变化可推断性:
#    本变体不引入任何颜色映射。背景、精灵、分隔线的角色保持原题语义，仅 marker 的可见性发生变化。

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L


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


def _infer_actor(g: np.ndarray, bg: int) -> int:
    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    return next(iter(colors - {bg} - div_colors))


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

    best = None
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


def _trace_visible_path(
    frame: np.ndarray, actor: int
) -> tuple[str, list[tuple[tuple[int, int], int]], list[tuple[int, int]]] | None:
    h, w = frame.shape
    found = _find_agent(frame, actor)
    if found is None:
        return None

    tname, d, head, _agent, markers = found
    markers = {tuple(map(int, pt)) for pt in markers}
    placements: list[tuple[tuple[int, int], int]] = [(head, d)]
    ordered_heads: list[tuple[int, int]] = []

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            return tname, placements, ordered_heads

        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d}"
        )
        nd, target = choice
        head = target
        d = nd
        ordered_heads.append(head)
        placements.append((head, d))
        markers.discard(head)

    raise AssertionError("path tracing exceeded max_steps")


def _simulate_from_state(
    tname: str,
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    h: int,
    w: int,
    bg: int,
    actor: int,
) -> np.ndarray:
    agent = {(head[0] + oy, head[1] + ox) for oy, ox in TEMPLATES[tname][d]}
    markers = {tuple(map(int, pt)) for pt in markers}

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break
        offs = TEMPLATES[tname][d]
        choice = _choose_target(head, d, markers, offs, h, w)
        assert choice is not None, (
            f"Stuck with {len(markers)} hidden markers at head={head} dir={d}"
        )
        nd, (ty, tx) = choice
        head = (ty, tx)
        d = nd
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def _simulate_visible_only(frame: np.ndarray, actor: int) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)
    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()
    tname, d, head, _agent, markers = found
    return _simulate_from_state(tname, head, d, markers, h, w, bg, actor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a triangular sprite's path even when some future markers are hidden in later frames.

    Input structure:
    - The input is a horizontal strip of equally sized frames, optionally separated by solid divider columns.
    - The first frame still shows the full path markers, while later frames may hide some still-unconsumed markers by painting them as background.

    Rule:
    - Recover the sprite template and the complete marker-consumption order from the earliest frame, where every target is visible.
    - Frames are evenly spaced snapshots of that same run, so the last frame gives the current sprite state but not necessarily the full remaining marker set.
    - Reconstruct the markers that must still remain after the last snapshot, then keep advancing to the nearest aligned target until the path is exhausted.

    Key insight: hidden dots are not gone; they are future head positions already revealed by earlier frames, so cross-frame comparison restores the missing tail of the route.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)
    actor = _infer_actor(g, bg)

    traced = _trace_visible_path(frames[0], actor)
    if traced is None:
        return frames[-1].copy()
    tname, placements, ordered_heads = traced

    n_frames = len(frames)
    total_steps = len(ordered_heads)
    assert total_steps % n_frames == 0, (
        f"total_steps={total_steps} is not divisible by n_frames={n_frames}"
    )
    gap = total_steps // n_frames

    for fi, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        assert found is not None, f"missing actor in frame {fi}"
        exp_head, exp_dir = placements[fi * gap]
        if (found[0], found[1], found[2]) != (tname, exp_dir, exp_head):
            raise AssertionError(
                f"frame {fi}: sprite mismatch, got {(found[0], found[1], found[2])}, "
                f"expected {(tname, exp_dir, exp_head)}"
            )

    last_step = (n_frames - 1) * gap
    last_found = _find_agent(frames[-1], actor)
    assert last_found is not None
    remaining = set(ordered_heads[last_step:])
    h, w = frames[-1].shape
    return _simulate_from_state(
        tname,
        last_found[2],
        last_found[1],
        remaining,
        h,
        w,
        bg,
        actor,
    )


# --------------- generator (D3 hidden-tail variant) ---------------

N_FRAMES_RANGE = (2, 5)          # 2..4 frames; hiding is only meaningful with later frames
FRAME_MAX = 16
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

STEPS_RANGE = (4, 13)
STEP_DIST_RANGE = (2, 8)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

TEMPLATE_HEAVY_THRESHOLD = 9
TEMPLATE_WEIGHTS_HEAVY = (4, 5, 3, 2)

MAX_ATTEMPTS = 400
_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(2, 5)) for n in range(*STEPS_RANGE)}


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
) -> tuple[int, int, int, int, int, int, int, int | None, str]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [n for n in range(*STEPS_RANGE) if n % n_frames == 0]
    weights = np.array([(n ** 3) / (_STEP_AVAILABILITY[n] ** 2) for n in multiples], dtype=float)
    weights /= weights.sum()
    n_steps = int(rng.choice(multiples, p=weights))
    gap = n_steps // n_frames

    h_lo = min(H_CAP, max(H_FLOOR, 6 + (n_steps + 1) // 2))
    w_lo = min(W_CAP, max(W_FLOOR, 4 + (n_steps + 2) // 2))
    h = int(rng.integers(h_lo, FRAME_MAX))
    w = int(rng.integers(w_lo, FRAME_MAX))

    palette = rng.permutation(10)
    bg, actor = int(palette[0]), int(palette[1])
    divider = int(palette[2]) if n_frames > 1 else None

    tw = np.array(
        TEMPLATE_WEIGHTS_HEAVY if n_steps >= TEMPLATE_HEAVY_THRESHOLD else (1, 1, 1, 1),
        dtype=float,
    )
    tw /= tw.sum()
    tname = str(rng.choice(list(TEMPLATES), p=tw))
    return n_frames, n_steps, gap, h, w, bg, actor, divider, tname


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
    hidden: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    hidden = hidden or set()
    frame = np.full((h, w), bg, dtype=int)
    patch, ar, ac = _template_patch(tname, d, actor, bg)
    paste(frame, patch, head[0] - ar, head[1] - ac, transparent=bg)
    for m in markers:
        if m not in hidden:
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


def _choose_hidden_tail(
    rng: np.random.Generator, remaining: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    assert remaining
    hide_count = int(rng.integers(1, len(remaining) + 1))
    hidden = {remaining[0]}
    if hide_count > 1:
        idxs = rng.choice(len(remaining) - 1, size=hide_count - 1, replace=False)
        for idx in np.atleast_1d(idxs):
            hidden.add(remaining[int(idx) + 1])
    return hidden


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, tname = _sample_params(rng)
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
    hidden_last: set[tuple[int, int]] = set()
    visible_last_remaining = 0
    for fi in range(n_frames):
        si = fi * gap
        head, d = placements[si]
        remaining = list(heads[si + 1 :])
        hidden = set()
        if fi == n_frames - 1:
            hidden = _choose_hidden_tail(rng, remaining[:gap])
            hidden_last = set(hidden)
            visible_last_remaining = len(remaining[:gap]) - len(hidden)
        frames.append(_render_frame(h, w, bg, actor, head, d, tname, remaining, hidden))

    fhead, fdir = placements[-1]
    out_frame = _render_frame(h, w, bg, actor, fhead, fdir, tname, [])
    inp = _assemble_input(frames, bg, divider)

    try:
        found = _find_agent(frames[-1], actor)
        if found is None:
            return None
        last_idx = (n_frames - 1) * gap
        if (found[0], found[1], found[2]) != (tname, placements[last_idx][1], placements[last_idx][0]):
            return None

        traced = _trace_visible_path(frames[0], actor)
        if traced is None:
            return None
        if traced[2] != heads[1:]:
            return None

        baseline_ok = False
        try:
            baseline = _simulate_visible_only(frames[-1], actor)
            baseline_ok = not np.array_equal(baseline, out_frame)
        except Exception:
            baseline_ok = True
        if not baseline_ok:
            return None

        solved = main(inp)
        if not np.array_equal(solved, out_frame):
            return None
    except Exception:
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
            "actor": actor,
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
            "hidden_last": sorted(hidden_last),
            "hidden_last_count": len(hidden_last),
            "visible_last_remaining": visible_last_remaining,
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
