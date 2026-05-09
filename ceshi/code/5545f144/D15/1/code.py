# Variant of 5545f144
# Dimension: D15 — 帧间信息关系 (multi_step)
# Change: 帧是同一模拟在不同时间点的快照，后续帧的标记是前续帧标记的子集（被消耗的标记消失）。同一模拟仍然保留，但现在精灵在连续帧之间不仅位置变化，模板类型也会按帧间可见的交替规律切换，solver 需要先从多帧差异中恢复该模板切换规律，再继续模拟到终局。
#
# Design analysis
# a) 具体化与行为追踪:
#    - 仍然只有一种 actor 颜色，精灵和单像素标记点共用该颜色；分隔列和背景规则不变。
#    - 变体只改“精灵模板是否恒定”。这里固定为: 每吃掉一个标记，精灵就在一对模板 A/B 之间切换一次。
#      因为输入帧仍是等间隔快照，所以当 gap 为奇数时，连续帧会清楚地呈现 A,B,A,B... 的交替。
#    - 小例子: 若三帧模板依次为 hollow3 -> stem4 -> hollow3，最后一帧还剩 3 个标记，则之后还会切换 3 次，
#      终局模板必须是 stem4。原始 solver 会错误地把最后一帧的 hollow3 一直沿用到结束。
# b) 复杂度校准:
#    - 只允许“两种模板交替”这一种切换机制，避免四模板任意映射带来的推断分支爆炸。
#    - 为了让人眼能看出规律，生成器只产生 n_frames >= 3 且 gap 为奇数的样本，使模板重复出现而不是偶然变化一次。
# c) 隐式假设审查:
#    - 原 main() 的 `_simulate_to_end(start, actor)` 隐含假设“最后一帧识别出的模板就是剩余所有步骤的模板”。
#      该前提在变体中失效，所以 solver 必须先读取全部帧，恢复最后一帧所处的 A/B 交替状态。
#    - 原 `_choose_target()` 用当前模板检查落点是否入界；现在落地后模板会切换，因此必须按“下一步模板+新朝向”检查。
# d) 放置合法性:
#    - 生成器对每个时间状态都记录 `(head, dir, template)`，平移可行域和标记邻接检查都基于真实模板，而不是单一模板。
#    - 每个未来标记都不能贴住任何过去状态的精灵实体，且对任一过去 head 仍只允许唯一半线方向可达。
# e) 视觉可观察性:
#    - 模板差异完全由形状体现，不使用颜色 id、排序或其他不可见元信息。
#    - 三到四帧内会至少出现 A,B,A，因此“模板在快照间交替”能直接被看出来。
# f) 可视化整洁度:
#    - 仍沿用原题的单色精灵+单色标记布局；通过邻接约束避免标记把精灵误看成更大模板。
# g) 新 solver 的输入假设与生成保证:
#    - solver 假设每帧都能唯一识别出一个模板，且所有帧只出现两种模板并按奇偶帧交替。
#    - generator 用逐帧 `_find_agent()` 回验这些条件，再调用 `main(input)` 验证整题闭环。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg as _bg_color
from arc_utils import paste


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Direction indices: 0=Up, 1=Right, 2=Down, 3=Left


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

TEMPLATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("hollow3", "filled4"),
    ("hollow3", "stem4"),
    ("hollow3", "midfilled6"),
    ("filled4", "stem4"),
    ("filled4", "midfilled6"),
    ("stem4", "midfilled6"),
)


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


def _template_fits(head: tuple[int, int], tname: str, d: int, h: int, w: int) -> bool:
    hr, hc = head
    for dr, dc in TEMPLATES[tname][d]:
        nr = hr + dr
        nc = hc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
    return True


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


def _toggle_template(tname: str, pair: tuple[str, str]) -> str:
    a, b = pair
    assert a != b and tname in pair
    return b if tname == a else a


def _infer_template_pair(
    frame_states: list[tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]
) -> tuple[str, str]:
    seq = [state[0] for state in frame_states]
    uniq: list[str] = []
    for tname in seq:
        if tname not in uniq:
            uniq.append(tname)
    assert len(uniq) == 2, f"expected exactly two alternating templates, got {seq}"
    for i, tname in enumerate(seq):
        assert tname == uniq[i % 2], f"template sequence is not alternating: {seq}"
    return uniq[0], uniq[1]


def _choose_target(
    head: tuple[int, int],
    d: int,
    markers: set[tuple[int, int]],
    h: int,
    w: int,
    next_tname: str,
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

    reachable: list[tuple[int, int, tuple[int, int]]] = []
    for dd in range(4):
        for dist, pt in candidates(dd):
            if _template_fits(pt, next_tname, dd, h, w):
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
    assert dd != (d + 2) % 4, f"head={head} dir={d}: would reverse to dir={dd}, not allowed"
    return dd, pt


def _simulate_to_end(frame: np.ndarray, actor: int, pair: tuple[str, str]) -> np.ndarray:
    h, w = frame.shape
    bg = _bg_color(frame)

    found = _find_agent(frame, actor)
    if found is None:
        return frame.copy()

    tname, d, head, agent, markers = found
    assert tname in pair, f"last frame template {tname} not in inferred pair {pair}"
    markers = set((int(y), int(x)) for y, x in markers)

    max_steps = h * w * 4
    for _ in range(max_steps):
        if not markers:
            break

        next_tname = _toggle_template(tname, pair)
        choice = _choose_target(head, d, markers, h, w, next_tname)
        assert choice is not None, (
            f"Stuck with {len(markers)} unconsumed markers at head={head} dir={d} "
            f"template={tname}"
        )
        nd, (ty, tx) = choice

        head = (ty, tx)
        d = nd
        tname = next_tname
        agent = {(ty + oy, tx + ox) for oy, ox in TEMPLATES[tname][d]}
        markers.discard(head)

    out = np.full((h, w), bg, dtype=int)
    for y, x in agent:
        if 0 <= y < h and 0 <= x < w:
            out[y, x] = actor
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue a marker-consuming arrow simulation whose sprite template alternates across snapshots.

    Input structure:
    - Frames are concatenated horizontally and optionally split by solid divider columns; background is the most common color.
    - Each frame contains one actor-colored arrow sprite plus actor-colored single-pixel markers.
    - Unlike the base task, the sprite's template type may differ from frame to frame while its head direction still encodes movement.

    Rule:
    - The frames are equally spaced checkpoints from one simulation, so the rightmost frame is the latest visible state.
    - Movement is unchanged: from the current head, exactly one cardinal half-line has reachable markers; move to the nearest one and forbid 180-degree reversals.
    - After every consumed marker, the sprite flips to the other template in the alternating pair revealed by the input frames; continue until all markers are gone.
    - Return only the final sprite on a clean background.

    Key insight: the earlier frames are needed to recover the alternating template pair; the last frame alone no longer tells you the final sprite shape.
    """
    g = np.asarray(grid, dtype=int)
    frames, bg = _split_frames(g)

    solid_cols = [c for c in range(g.shape[1]) if len(set(map(int, g[:, c]))) == 1]
    div_colors = {int(g[0, c]) for c in solid_cols if int(g[0, c]) != bg}
    colors = set(map(int, g.ravel()))
    actor = next(iter(colors - {bg} - div_colors))

    frame_states: list[
        tuple[str, int, tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]
    ] = []
    for frame in frames:
        found = _find_agent(frame, actor)
        assert found is not None, "every frame must contain a detectable sprite"
        frame_states.append(found)

    pair = _infer_template_pair(frame_states)
    return _simulate_to_end(frames[-1], actor, pair)


# --------------- generator ---------------

# --- CFG: canvas / frame bounds ---
N_FRAMES_RANGE = (3, 5)          # rng.integers low/high+1 -> 3..4 frames
FRAME_MAX = 16                   # rng.integers high for H, W -> 8..15 / 6..15
H_FLOOR, H_CAP = 8, 15
W_FLOOR, W_CAP = 6, 15

# --- CFG: path complexity ---
STEPS_RANGE = (4, 13)            # candidate total_steps (4..12 inclusive)
STEP_DIST_RANGE = (2, 8)         # per-step manhattan distance choices (2..7 inclusive)
TURN_RATIO_LO, TURN_RATIO_HI = 0.4, 0.9
TURN_MIN_ABS = 3
MIN_TURNS_ACCEPT = 2
MIN_DIR_DIVERSITY = 3
MIN_DIST_VARIETY = 2

# --- CFG: retry budget ---
MAX_ATTEMPTS = 400

_STEP_AVAILABILITY = {n: sum(n % f == 0 for f in range(3, 5)) for n in range(*STEPS_RANGE)}


def _state_templates(pair: tuple[str, str], n_steps: int) -> list[str]:
    return [pair[i % 2] for i in range(n_steps + 1)]


def _shift_interval(
    placements: list[tuple[tuple[int, int], int, str]], h: int, w: int
) -> tuple[int, int, int, int] | None:
    tr_lo, tr_hi = -10**9, 10**9
    tc_lo, tc_hi = -10**9, 10**9
    for (hr, hc), d, tname in placements:
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
) -> tuple[int, int, int, int, int, int, int, int | None, tuple[str, str]]:
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    multiples = [
        n for n in range(*STEPS_RANGE)
        if n % n_frames == 0 and (n // n_frames) % 2 == 1
    ]
    weights = np.array([(n ** 3) / max(1, _STEP_AVAILABILITY[n]) for n in multiples], dtype=float)
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
    pair = _sample_pair(rng)
    return n_frames, n_steps, gap, h, w, bg, actor, divider, pair


def _sample_pair(rng: np.random.Generator) -> tuple[str, str]:
    pair = TEMPLATE_PAIRS[int(rng.integers(0, len(TEMPLATE_PAIRS)))]
    if int(rng.integers(0, 2)) == 1:
        return pair[1], pair[0]
    return pair


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
    state_templates: list[str],
    step_dirs: list[int],
) -> bool:
    for j, old_head in enumerate(heads):
        sprite = _sprite_cells(old_head, state_templates[j], state_dirs[j])
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
    rng: np.random.Generator, pair: tuple[str, str], step_dirs: list[int], h: int, w: int
) -> tuple[
    list[tuple[int, int]],
    list[int],
    list[str],
    list[tuple[tuple[int, int], int, str]],
] | None:
    n_steps = len(step_dirs)
    templates = _state_templates(pair, n_steps)
    d0 = step_dirs[0]
    heads: list[tuple[int, int]] = [(0, 0)]
    state_dirs: list[int] = [d0]
    placements: list[tuple[tuple[int, int], int, str]] = [((0, 0), d0, templates[0])]

    for i in range(n_steps):
        cur = heads[-1]
        cur_dir = state_dirs[-1]
        next_dir = step_dirs[i]
        next_tname = templates[i + 1]
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
            if not _marker_compatible_with_existing(nxt, heads, state_dirs, templates, step_dirs):
                continue
            trial_fit = placements + [(nxt, next_dir, next_tname)]
            if _shift_interval(trial_fit, h, w) is None:
                continue
            chosen = nxt
            break

        if chosen is None:
            return None

        heads.append(chosen)
        state_dirs.append(next_dir)
        placements.append((chosen, next_dir, next_tname))

    return heads, state_dirs, templates, placements


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


def _build_case(rng: np.random.Generator) -> dict | None:
    n_frames, n_steps, gap, h, w, bg, actor, divider, pair = _sample_params(rng)
    step_dirs = _make_step_dirs(rng, n_steps)

    rollout = _rollout_path(rng, pair, step_dirs, h, w)
    if rollout is None:
        return None
    heads, state_dirs, state_templates, placements = rollout

    ok, turns, dists = _validate_path(step_dirs, heads)
    if not ok:
        return None

    shift = _shift_interval(placements, h, w)
    if shift is None:
        return None
    tr_lo, tr_hi, tc_lo, tc_hi = shift
    tr = int(rng.integers(tr_lo, tr_hi + 1))
    tc = int(rng.integers(tc_lo, tc_hi + 1))
    heads = [(r + tr, c + tc) for r, c in heads]

    frames = []
    snapshot_templates: list[str] = []
    for fi in range(n_frames):
        si = fi * gap
        snapshot_templates.append(state_templates[si])
        frames.append(
            _render_frame(
                h,
                w,
                bg,
                actor,
                heads[si],
                state_dirs[si],
                state_templates[si],
                heads[si + 1:],
            )
        )

    out_frame = _render_frame(
        h, w, bg, actor, heads[-1], state_dirs[-1], state_templates[-1], []
    )
    inp = _assemble_input(frames, bg, divider)

    if len(set(snapshot_templates)) != 2:
        return None
    for i, tname in enumerate(snapshot_templates):
        if tname != snapshot_templates[i % 2]:
            return None

    for fi, frame in enumerate(frames):
        found = _find_agent(frame, actor)
        if found is None:
            return None
        f_tname, f_dir, f_head, _f_agent, f_markers = found
        si = fi * gap
        if (f_tname, f_dir, f_head) != (state_templates[si], state_dirs[si], heads[si]):
            return None
        if f_markers != set(heads[si + 1:]):
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
            "template_pair": pair,
            "snapshot_templates": list(snapshot_templates),
            "bg": bg,
            "actor": actor,
            "divider": divider,
            "total_steps": n_steps,
            "gap": gap,
            "gap_is_odd": bool(gap % 2 == 1),
            "turns": turns,
            "turn_ratio": turns / n_steps,
            "direction_diversity": len(set(step_dirs)),
            "step_dirs": list(step_dirs),
            "state_dirs": list(state_dirs),
            "state_templates": list(state_templates),
            "heads": list(heads),
            "step_distances": dists,
            "count_in_first_frame": n_steps,
            "count_in_last_input_frame": gap,
            "final_template_differs_from_last_frame": state_templates[-1] != snapshot_templates[-1],
        },
    }


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
