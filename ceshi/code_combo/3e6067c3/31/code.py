# Combo variant of 3e6067c3 (index 31)
# combo_spec: D12/0+D5/0
# Name: 星形一对多连接+终点色桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
# - 画面仍然是规则格点上的同尺寸带框方块，底部控制行给出 trunk 访问顺序。
# - 原任务在一步 `a -> b` 中，只会从当前 a 方块挑一个 b 方块，并用 a 的颜色画一条桥。
# - 本融合变体把两个改动叠加到同一步：
#   1. 若颜色 b 对应多个可桥接方块，则从当前 a 方块同时向这些 b 方块扇出多条桥；
#   2. 每条桥不再取 a 的颜色，而取各自终点方块的内部色。
# - 具体小例子：控制序列为 [2, 4, 7]，当前 2 方块右侧有一个 4-trunk，左下远处还有一个 4-leaf。
#   原任务输出只会画一条 2 色桥到最近的 4-trunk；
#   本变体会同时画到两个 4 方块，且两条桥都画成终点色 4。之后继续从最近的 4-trunk 走向 7。
# - 差异点不是“另起一条独立链”，而是“同一控制步出现星形拓扑，同时颜色归属改到终点”。
#
# b) 复杂度校准
# - 每个样例只制造一次星形分叉，额外 leaf 数量限制为 1 或 2，避免场景变成大树导致混乱。
# - 除星形目标色外，其余 trunk 颜色保持唯一，这样人类能稳定看出：只有某一步存在一对多，其余步骤仍是单链。
# - 虽然 D12 的原始含义要求重复的是同一个目标色，因此星形各射线在该一步会同色，但 D5 仍通过整条 trunk
#   的“桥色跟终点走”保持清晰可见；generator 也保证每条 trunk 边的起终点颜色不同，避免退化回原题。
#
# c) 隐式假设审查
# - 原 main() 的 `by[color]` 允许重复，但后续总是压成单个 `bestf`。融合后这个假设失效：
#   solver 必须保留一个 trunk continuation，同时对所有可桥接重复目标都绘制桥。
# - 原 `draw_bridge(cur, nxt, a)` 假设桥色来自起点。融合后失效，必须改成对每个 target 用 `target["inner"]`。
# - 原 `cur = nxt` 的单 continuation 假设仍需保留，否则控制序列无法继续；generator 因而只让“最近且可桥接的
#   那个重复目标”承担后续 trunk，其余同色方块只作为 leaf，不再继续。
# - `_bridge_cells()` 仍要求两个方块共享同一内带行或列。generator 必须把 star leaves 放在当前节点的清晰射线上。
#
# d) 放置合法性
# - 继续采用 output-first：先构造带桥的目标输出，再擦掉桥得到输入。
# - 所有方块都放在规则格点上，trunk 是自避免路径，star leaves 只从某个 trunk 节点向空射线额外伸出。
# - 每条桥在写入前都检查：桥像素集非空、不越界、不压到任何方块、且不与已有桥重叠；否则整次尝试作废。
#
# e) 视觉可观察性
# - 输入里可直接看到：控制行只出现一次的某个颜色，在主区域里却有多个同色方块。
# - 输出里该控制步会从同一个源方块扇出多条桥，形成明显的星形，而不是单链。
# - 同时，整条路径上桥的颜色始终贴着终点方块走；比较 2-3 组样例，人类能同时观察到“星形一步”和“终点着色”。
#
# f) 可视化整洁度
# - 只允许一个星形事件，且 leaf 位于彼此不同的水平/垂直射线上，避免桥梁互相盖住。
# - 桥宽仍与内填充带同宽，不引入额外符号层，视觉语言与原题保持一致。
#
# g) 变化的可推断性
# - 不引入数值排序、隐藏标签或额外颜色映射。所有判断都由可见的颜色相等关系和几何对齐关系驱动。
# - D5 的颜色变化规则不是任意换色，而是“桥总是涂成它到达的方块颜色”；D12 的数量变化则由“一个控制色对应多个
#   同色方块”提供直接证据。
#
# h) 防退化指标
# - 指标 1：每个样本至少有 1 个控制步满足“可桥接目标数 >= 2”，否则没有星形行为。
# - 指标 2：每个样本至少有 3 条桥满足“桥色 = 终点色 且 桥色 != 起点色”，否则 D5 视觉信号太弱。
# - 测试后会对生成样本统计这两个指标，确保不是碰巧通过 `main(generate_input())`。
#
# i) main() 新增假设与生成器保障
# - 假设 1：若某个颜色重复出现，则至少有一个重复目标是当前节点的合法 trunk continuation。
#   generator 通过先选 trunk，再复制 trunk-next 的颜色给额外 leaves 来保证。
# - 假设 2：其余重复目标与当前节点之间存在可画桥的清晰空带。
#   generator 只从 `_find_star_choices()` 返回的空射线上选 leaves。
# - 假设 3：bridgeable 重复目标之间不会产生桥重叠。
#   generator 在 output-first 阶段逐条 reserve，出现重叠立即丢弃样本。


def _extract_frames(g: np.ndarray, border: int) -> list[dict]:
    lab, n = label_mask(g == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        t = None
        inner_col = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue

            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            t = int(tt)
            inner_col = int(next(iter(u)))
            break

        if t is None or inner_col is None:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "inner": int(inner_col),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            }
        )
    return frames


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _score_candidate(src: dict, dst: dict) -> tuple[int, int]:
    sr, sc = _aligned(src, dst)
    aligned = 1 if (sr or sc) else 0
    r1, c1 = _center(src)
    r2, c2 = _center(dst)
    dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(aligned), -int(dist)


def _best_candidate(src: dict, cands: list[dict]) -> dict | None:
    best = None
    bestf = None
    for f in cands:
        cand = _score_candidate(src, f)
        if best is None or cand > best:
            best = cand
            bestf = f
    return bestf


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        else:
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Traverse a single trunk according to the control row.
    - If the next control color appears on multiple bridgeable squares, draw from the current
      square to all of them at once.
    - Keep only the best same-color target as the continuation point.
    - Every bridge uses its destination square's inner color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    cnt = Counter(map(int, g.flatten()))
    border = None
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            border = int(col)
            break
    assert not (border is None)

    frames = _extract_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
        best = None
        bestf = None
        for f in start_cands:
            anchors = nxt_cands if nxt_cands else [f]
            for g2 in anchors:
                cand = _score_candidate(f, g2)
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf
    assert not (cur is None)

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        for r, c in _bridge_cells(fa, fb):
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue

        all_cands = by[b]
        nxt = _best_candidate(cur, all_cands)
        if nxt is None:
            continue

        bridgeable = [f for f in all_cands if _bridge_cells(cur, f)]
        targets = bridgeable if bridgeable else [nxt]
        for target in targets:
            draw_bridge(cur, target, int(target["inner"]))
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _sample_path(R: int, C: int, n_nodes: int) -> list[tuple[int, int]] | None:
    for _ in range(500):
        start = (int(np.random.randint(0, R)), int(np.random.randint(0, C)))
        path = [start]
        seen = {start}
        ok = True
        for _step in range(int(n_nodes) - 1):
            r, c = map(int, path[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            path.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in path}) < 2 or len({c for _r, c in path}) < 2:
            continue
        return path
    return None


def _find_star_choices(path: list[tuple[int, int]], R: int, C: int) -> list[tuple[int, list[tuple[int, int]]]]:
    occupied = set(path)
    choices: list[tuple[int, list[tuple[int, int]]]] = []
    for idx in range(int(len(path)) - 1):
        src = path[idx]
        trunk_next = path[idx + 1]
        rays: list[tuple[int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            blocked = False
            for dist in range(1, 4):
                rr = int(src[0] + dr * dist)
                cc = int(src[1] + dc * dist)
                if not in_bounds((int(R), int(C)), int(rr), int(cc)):
                    break
                pos = (int(rr), int(cc))
                if pos in occupied:
                    blocked = True
                if blocked:
                    break
                if dist >= 2 and pos != trunk_next:
                    rays.append(pos)
        if rays:
            choices.append((int(idx), rays))
    return choices


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 6:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(4, 7))
    C = int(np.random.randint(4, 7))
    n_path = int(np.random.randint(4, 7))
    if n_path > len(inner_palette):
        return None

    path = _sample_path(int(R), int(C), int(n_path))
    if path is None:
        return None

    star_choices = _find_star_choices(path, int(R), int(C))
    if not star_choices:
        return None

    star_step, star_rays = star_choices[int(np.random.randint(0, len(star_choices)))]
    extra_count = int(np.random.randint(1, min(2, len(star_rays)) + 1))
    pick_idx = np.random.choice(np.arange(len(star_rays), dtype=int), size=int(extra_count), replace=False)
    star_leaves = [star_rays[int(i)] for i in np.array(pick_idx).tolist()]

    used_slots = set(path) | set(star_leaves)

    row_gap = [int(np.random.randint(1, 6)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(1, 6)) for _ in range(int(C) - 1)]
    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    r0s = [int(top_margin)]
    for k in range(int(R) - 1):
        r0s.append(int(r0s[-1] + outer + row_gap[int(k)]))
    c0s = [int(left_margin)]
    for k in range(int(C) - 1):
        c0s.append(int(c0s[-1] + outer + col_gap[int(k)]))

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in used_slots:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    trunk_colors = np.random.choice(np.array(inner_palette, dtype=int), size=int(n_path), replace=False)
    colors_by_slot: dict[tuple[int, int], int] = {}
    for idx, slot in enumerate(path):
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(trunk_colors[int(idx)])

    star_color = int(colors_by_slot[path[int(star_step + 1)]])
    for slot in star_leaves:
        colors_by_slot[(int(slot[0]), int(slot[1]))] = int(star_color)

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()

    def reserve_bridge(fa: dict, fb: dict, color: int) -> bool:
        nonlocal bridge_cells_all
        cells = _bridge_cells(fa, fb)
        if not cells:
            return False
        if cells & bridge_cells_all:
            return False
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return False
            if int(out[int(r), int(c)]) != int(bg):
                return False
        for r, c in cells:
            out[int(r), int(c)] = int(color)
        bridge_cells_all |= set(cells)
        return True

    ok = True
    for idx, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        paint = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        if not reserve_bridge(fa, fb, int(paint)):
            ok = False
            break
        if int(idx) == int(star_step):
            for leaf in star_leaves:
                lf = slot_to_frame[(int(leaf[0]), int(leaf[1]))]
                if not reserve_bridge(fa, lf, int(colors_by_slot[(int(leaf[0]), int(leaf[1]))])):
                    ok = False
                    break
        if not ok:
            break
    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if len(ctrl_colors) < 2 or int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
