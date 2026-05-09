# Combo variant of 3e6067c3 (index 28)
# combo_spec: D4/1+D1/0+D5/0
# Name: 嵌套双层框+边框匹配+终点色桥梁

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

# Design analysis for combo D4/1 + D1/0 + D5/0
# =================================================
# a) 具体化与行为追踪
# - 原题对象是“单层边框 + 单色内部”，控制行按内部色匹配，桥段颜色也取起点内部色。
# - 本融合版对象改成“外框 A + 内框 B + 核心填充 C”的双层嵌套实心方块；控制行读的是外框 A；
#   当路径从 A 方块走到 D 方块时，桥段要画成终点方块的外框色 D，而且桥宽只对齐到核心填充区。
# - 小例子：若有 3 个 6x6 方块，结构分别为
#   2/5/7、4/5/7、8/5/7（外框/内框/填充），控制序列是 [2, 4, 8]。
#   原逻辑会失败三次：它会把 7 当识别色、把 5x5 内域误作通道、并把 2->4 段画成 2。
#   融合后应当识别三层结构，按外框 2->4->8 走，并把 2->4 段画成 4、4->8 段画成 8。
# - 三项变化不是独立拼接：因为控制色和桥色都改到“外层框色”，双层对象的最外圈才成为真正的语义承载层。
#
# b) 复杂度校准
# - 若让内框色或填充色也各自唯一，会削弱“必须看外框”的必要性；因此生成器固定所有方块共享同一个内框色与填充色。
# - 若让双层框厚度可变，会把 D4 的变化扩散成额外自由度；这里固定为 1 像素外框 + 1 像素内框。
# - 外框色唯一、内层重复，可以同时避免匹配歧义并放大视觉差异：控制行只能对应外框，桥段颜色也总能肉眼对到终点外框。
#
# c) 隐式假设审查
# - 原 `most common non-bg == border` 失效：每个方块外框色不同，且内框/填充可能更常见，所以不能按单色连通块找框。
# - 原 `label_mask(g == border)` 失效：双层框对象在输入中是一个完整非背景实心正方形，必须按 `g != bg` 找组件再解析三层。
# - 原 `by[f["inner"]]`、`assert cur["inner"] == a` 失效：控制色现在对应 `outer_border`。
# - 原 `inner_bbox` 同时承担“桥宽”和“对齐带”含义也失效：双层框下真正可通行的是核心填充 `fill_bbox`。
# - 原 `draw_bridge(cur, nxt, a)` 的颜色归属失效：现在必须是 `nxt["outer_border"]`，也就是控制序列中的下一个颜色。
#
# d) 放置合法性
# - 生成器仍用规则格点放方块，行列间至少 1 个背景像素，保证方块不重叠。
# - 路径是格点 4 邻自避免路径，因此任意相邻方块必共享同一行带或列带，桥段只落在两者之间的背景间隙里。
# - output-first 先画完整输出，再擦桥得到输入；每次画桥前都检查桥格仍是背景，防止与别的方块、桥或控制行冲突。
#
# e) 视觉可观察性
# - 人类从 2-4 个样例中能同时看到三点：
#   1. 方块变成双层框，桥明显只与更内层核心对齐；
#   2. 控制行颜色只出现在最外层边框，不出现在填充区；
#   3. 每段桥颜色与它“连向的终点方块外框”一致，而不是与起点一致。
# - 由于所有方块共享同一内框色和填充色，观察者几乎不可能把规则误解为“按内部色匹配”。
#
# f) 可视化整洁度
# - 每个对象只含 3 个严格同心的纯色层，桥只填在背景带里，不覆盖框体。
# - 控制行仍单独放在底部一行，且只用稀疏单像素编码，不与主场景粘连，整体保持原题的整洁布局。
#
# g) 变化的可推断性
# - 颜色角色被压缩到 4 类：背景、各方块唯一外框色、全局共享内框色、全局共享填充色。
# - 桥色规则也没有引入隐藏映射表，只是“桥段颜色 = 终点方块外框色”，终点就在桥的另一端，结构线索足够强。
#
# h) 防退化量化指标
# - 统计每个样本中 `(桥段颜色 == 终点外框色)` 的桥段数，以及 `(桥段颜色 != 起点外框色)` 的桥段数。
# - 在本生成器里两项都应等于总桥段数，因为外框色全局唯一，任一相邻路径对的起点色与终点色必不同。
# - 这保证了同一输入下，融合规则输出不会退化成原题的“起点色桥”或“内部色匹配”输出。
#
# i) 新 solver 假设与 generator 保证
# - 假设有效对象是完整非背景正方形组件，内部恰好满足 1 像素外框 + 1 像素内框 + 单色核心；`_draw_double_frame()` 保证。
# - 假设控制行颜色都能在某个对象的外框上找到，且外框色唯一；生成器直接用路径方块的外框色生成控制行。
# - 假设桥只在共享行带或列带的相邻方块间绘制；生成器的路径与 `_bridge_cells()` 共同保证这一点。


def _parse_double_frame(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, bbox)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 5:
        return None

    patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if bool((patch == int(bg)).any()):
        return None
    if int(patch.shape[0]) <= 4 or int(patch.shape[1]) <= 4:
        return None

    outer_border = int(patch[0, 0])
    if outer_border == int(bg):
        return None
    if not bool((patch[0, :] == int(outer_border)).all()):
        return None
    if not bool((patch[-1, :] == int(outer_border)).all()):
        return None
    if not bool((patch[:, 0] == int(outer_border)).all()):
        return None
    if not bool((patch[:, -1] == int(outer_border)).all()):
        return None

    inner_patch = patch[1:-1, 1:-1]
    inner_border = int(inner_patch[0, 0])
    if int(inner_border) in (int(bg), int(outer_border)):
        return None
    if not bool((inner_patch[0, :] == int(inner_border)).all()):
        return None
    if not bool((inner_patch[-1, :] == int(inner_border)).all()):
        return None
    if not bool((inner_patch[:, 0] == int(inner_border)).all()):
        return None
    if not bool((inner_patch[:, -1] == int(inner_border)).all()):
        return None

    core = inner_patch[1:-1, 1:-1]
    if core.size == 0:
        return None
    fill = int(core[0, 0])
    if int(fill) in (int(bg), int(outer_border), int(inner_border)):
        return None
    if not bool((core == int(fill)).all()):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "outer_border": int(outer_border),
        "inner_border": int(inner_border),
        "inner": int(fill),
        "fill_bbox": (int(r0 + 2), int(c0 + 2), int(r1 - 2), int(c1 - 2)),
    }


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 25:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if int(pts.shape[0]) != int((r1 - r0 + 1) * (c1 - c0 + 1)):
            continue
        parsed = _parse_double_frame(g, (int(r0), int(c0), int(r1), int(c1)), int(bg))
        if parsed is None:
            continue
        frames.append(parsed)
    return frames


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["fill_bbox"]
    b = frame_b["fill_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pick_best_target(cur: dict, cands: list[dict]) -> dict | None:
    best = None
    best_frame = None
    cur_center = _center(cur)
    for frame in cands:
        same_row, same_col = _aligned(cur, frame)
        aligned_score = 1 if (same_row or same_col) else 0
        frame_center = _center(frame)
        dist = abs(int(cur_center[0]) - int(frame_center[0])) + abs(int(cur_center[1]) - int(frame_center[1]))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            best_frame = frame
    return best_frame


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, frame_a["bbox"])
    r0b, c0b, r1b, c1b = map(int, frame_b["bbox"])
    fill_a = frame_a["fill_bbox"]
    fill_b = frame_b["fill_bbox"]

    if int(fill_a[0]) == int(fill_b[0]) and int(fill_a[2]) == int(fill_b[2]):
        rr0, rr1 = int(fill_a[0]), int(fill_a[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}

    if int(fill_a[1]) == int(fill_b[1]) and int(fill_a[3]) == int(fill_b[3]):
        cc0, cc1 = int(fill_a[1]), int(fill_a[3])
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
    Connect nested double-frame squares using outer-border matching and destination-colored bridges.

    Variant rule:
    - Each square is a solid non-bg component with outer border color A, inner border color B, and fill color C.
    - Control pixels match the outer border colors A, not the fill C.
    - Each bridge segment is drawn with the destination square's outer border color.
    - Bridge thickness is determined by the core fill region enclosed by the inner border.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert not (not frames)

    by_outer: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_outer[int(frame["outer_border"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)

    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    start_color = int(ctrl[0])
    assert int(start_color) in by_outer
    start_cands = by_outer[int(start_color)]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by_outer.get(int(nxt_color), [])
        best = None
        best_frame = None
        for frame in start_cands:
            probe_cands = nxt_cands if nxt_cands else [frame]
            probe = _pick_best_target(frame, list(probe_cands))
            if probe is None:
                continue
            same_row, same_col = _aligned(frame, probe)
            aligned_score = 1 if (same_row or same_col) else 0
            c1 = _center(frame)
            c2 = _center(probe)
            dist = abs(int(c1[0]) - int(c2[0])) + abs(int(c1[1]) - int(c2[1]))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                best_frame = frame
        cur = best_frame
    assert not (cur is None)

    out = g.copy()

    def draw_bridge(frame_a: dict, frame_b: dict, color: int) -> None:
        cells = _bridge_cells(frame_a, frame_b)
        for r, c in cells:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["outer_border"]) == int(a)
        assert int(b) in by_outer
        nxt = _pick_best_target(cur, by_outer[int(b)])
        assert not (nxt is None)
        draw_bridge(cur, nxt, int(b))
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _draw_double_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    outer_border: int,
    inner_border: int,
    fill_color: int,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(outer_border)
    g[int(r0 + 1) : int(r0 + outer - 1), int(c0 + 1) : int(c0 + outer - 1)] = int(inner_border)
    g[int(r0 + 2) : int(r0 + outer - 2), int(c0 + 2) : int(c0 + outer - 2)] = int(fill_color)


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    avail = [int(d) for d in _DIGITS if int(d) != int(bg)]
    if len(avail) < 4:
        return None

    inner_border = int(np.random.choice(np.array(avail, dtype=int)))
    fill_choices = [int(d) for d in avail if int(d) != int(inner_border)]
    if not fill_choices:
        return None
    fill_color = int(np.random.choice(np.array(fill_choices, dtype=int)))
    outer_choices = [int(d) for d in avail if int(d) not in (int(inner_border), int(fill_color))]
    if len(outer_choices) < 4:
        return None

    core = int(np.random.randint(2, 4))
    outer = int(core + 4)

    R = int(np.random.randint(2, 5))
    C = int(np.random.randint(2, 5))
    max_frames = int(R * C)
    if int(max_frames) < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(len(outer_choices), max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

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

    outer_colors = [
        int(x) for x in np.random.choice(np.array(outer_choices, dtype=int), size=int(n_frames), replace=False)
    ]

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    outer_by_slot: dict[tuple[int, int], int] = {}
    for idx, (rr, cc) in enumerate(path):
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        outer_border = int(outer_colors[int(idx)])
        frame = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "fill_bbox": (int(r0 + 2), int(c0 + 2), int(r1 - 2), int(c1 - 2)),
            "slot": (int(rr), int(cc)),
            "outer_border": int(outer_border),
            "inner_border": int(inner_border),
            "inner": int(fill_color),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame
        outer_by_slot[(int(rr), int(cc))] = int(outer_border)

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    ctrl_width = int(2 * n_frames)
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_width + 1))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_double_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            outer_border=int(frame["outer_border"]),
            inner_border=int(inner_border),
            fill_color=int(fill_color),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        frame_a = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        frame_b = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(outer_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(outer_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for color, col in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None

    frames_found = _detect_frames(inp, int(bg))
    if int(len(frames_found)) != int(len(frames)):
        return None
    if len({int(frame["outer_border"]) for frame in frames_found}) != int(len(frames)):
        return None
    if {int(frame["inner_border"]) for frame in frames_found} != {int(inner_border)}:
        return None
    if {int(frame["inner"]) for frame in frames_found} != {int(fill_color)}:
        return None
    if set(ctrl_colors) != {int(frame["outer_border"]) for frame in frames_found}:
        return None
    if int(fill_color) in set(ctrl_colors) or int(inner_border) in set(ctrl_colors):
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
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
