# Combo variant of 3e6067c3 (index 8)
# combo_spec: D2/3+D4/3
# Name: 位置顺序消歧+自由排列方块

"""
Design analysis for the combo variant

a) 具体化与行为追踪
- 画面仍由若干等尺寸带框方块、底部控制色序列、以及输出里补出的彩色桥组成。
- 原任务的两条关键假设分别是：
  1. 方块落在规则晶格上，所以“同一行带/同一列带”可由内区边界完全相等来判断。
  2. 如果某种内部颜色出现多个候选，则从当前方块出发，选对齐且最近的那个。
- 融合后两条都变：
  1. 方块自由散布，只要求真正需要连接的两块在某一轴的内区有可见重叠；桥的厚度取这个重叠条带。
  2. 当控制色命中多个方块时，不再看距离，而是把全部方块按位置排序
     `(r0, c0, r1, c1)`，选择该颜色里第一个尚未访问的方块。
- 一个具体小例子：
  起点 A 在左上，控制序列第二色是红色。真正路径目标 B 在 A 右侧，和 A 的内区只部分重叠，
  因而输出应画一条较薄的横桥。另有同色干扰块 X 在 A 下方，更近，而且与 A 还形成一条更厚的竖桥。
  原任务/距离策略会优先连向更近的 X；融合规则必须先按全局顺序看到更靠上的 B，再按自由排列重叠条带画到 B。

b) 复杂度校准
- 如果所有颜色都唯一，则 D2 消歧完全不可见；如果大量重复，则控制行和访问状态会变得杂乱。
- 因此生成器只固定制造一个关键分叉：路径第 2 个颜色复制出 1 个 decoy。
- 自由排列也做限幅：正交偏移只允许小于内区尺寸，使桥仍是清晰的矩形条带，而不是难以读图的锯齿或斜线。
- 每个样例至少包含 1 条部分重叠桥，避免 D4 退化回原始等带连接。

c) 隐式假设审查
- 原 `aligned()` 依赖“内区上下边界完全相等”或“左右边界完全相等”；D4 后该前提失效，
  必须改成“区间有交叠 + 另一轴有正间隔”。
- 原 `draw_bridge()` 默认桥厚度直接继承某一端内区宽/高；D4 后必须改成真实重叠条带。
- 原候选选择逻辑依赖当前方块与候选的局部几何关系（对齐、距离）；D2 后这一点不再决定“选谁”，
  只能由全局位置顺序决定。
- 原代码没有 `visited`；D2 明确要求“第一个未访问的”，所以 solver 必须记录访问集合。
- 两个改动叠加后的新约束是：虽然 solver 选择目标时不看几何，但 generator 必须保证
  “按全局顺序选中的那个同色方块”恰好也是可桥接的，否则输出会出现无桥空转。

d) 放置合法性
- 生成器使用 output-first：先构造完整输出，再擦掉路径桥得到输入。
- 主路径固定为 4 个方块，按“水平 -> 垂直 -> 水平”顺序自由排布；每一步都检查桥非空、桥之间不重叠，
  且桥不会撞到其它方块。
- 额外 decoy 放在起点下方，与起点也可桥接，但不属于真实路径。这样输入中存在真实歧义，
  又不会破坏输出布局。
- 最终会验证边框色仍是最常见非背景色、`main(input)` 精确还原输出。

e) 视觉可观察性
- 人类从 2-4 个样例里应能同时看到两件事：
  1. 有些桥比方块内区更薄，说明不是规则晶格上的整带连接。
  2. 遇到重复颜色时，输出没有连向最近的同色块，而是连向更靠上/更靠左的那个。
- 排序只依赖可见位置，不依赖颜色编号大小、连通块扫描顺序等人类不可见因素。

f) 可视化整洁度
- 只引入一个 decoy，避免场景过载。
- 所有对象仍是矩形带框块，桥仍是轴对齐矩形条带；控制行仍保持底部单行像素序列。
- 布局参数受限在 30x30 内，并通过重叠检测拒绝拥挤样本。

g) 变化的可推断性
- 桥颜色仍保持“从当前出发方块继承”的原始规则，不引入新的颜色映射。
- 颜色重复只出现在一个关键目标色上，且该色在输入/输出对里有明确的结构线索：
  一块更近但更靠后，一块更远但更靠前，便于归纳“按位置而非按距离”的新规则。

h) 量化退化检查思路
- 不能只检查 `main(input) == output`，因为那无法证明融合行为真的被触发。
- 生成器同时要求：
  1. 至少 1 条真实路径桥是部分重叠桥，证明 D4 活跃。
  2. 同一输入送入“自由排列 + 距离优先”的 legacy solver 时，输出必须不同，证明 D2 活跃。
  3. 同一输入送入“位置顺序 + 原始精确等带桥”的 legacy solver 时，输出也必须不同，证明 D4 活跃。

i) 新 solver 的输入假设与生成器保障
- 假设 1：首个控制色对应的起点全局唯一。生成器保证起点颜色不重复。
- 假设 2：每次按全局顺序选中的下一个同色块都可通过重叠条带桥接。生成器按路径构造保证。
- 假设 3：关键 decoy 比真实目标更近，但在全局顺序里更靠后。生成器通过“目标在右上、decoy 在正下方”
  以及“水平间隔大于竖直间隔”同时保证。
- 假设 4：至少一条真实路径边的重叠厚度小于完整内区厚度。生成器通过非零自由偏移保证。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, label_mask


_DIGITS = [int(i) for i in range(10)]


def _frame_order_key(f: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    t: int,
    border: int,
    inner_col: int,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _detect_frames(g: np.ndarray, border: int) -> list[dict]:
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
            vals = set(map(int, inner.flatten()))
            if len(vals) != 1:
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
            inner_col = int(next(iter(vals)))
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

    frames.sort(key=_frame_order_key)
    for idx, frame in enumerate(frames):
        frame["order"] = int(idx)
    return frames


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if int(lo) > int(hi):
        return None
    return int(lo), int(hi)


def _bridge_info(fa: dict, fb: dict) -> dict | None:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    row_overlap = _interval_overlap(int(ia[0]), int(ia[2]), int(ib[0]), int(ib[2]))
    if row_overlap is not None:
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        elif int(c1b) < int(c0a):
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        else:
            cc0, cc1 = 1, 0
        if int(cc0) <= int(cc1):
            rr0, rr1 = row_overlap
            return {
                "axis": "h",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "overlap": int(rr1 - rr0 + 1),
                "gap": int(cc1 - cc0 + 1),
            }

    col_overlap = _interval_overlap(int(ia[1]), int(ia[3]), int(ib[1]), int(ib[3]))
    if col_overlap is not None:
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        elif int(r1b) < int(r0a):
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        else:
            rr0, rr1 = 1, 0
        if int(rr0) <= int(rr1):
            cc0, cc1 = col_overlap
            return {
                "axis": "v",
                "rr0": int(rr0),
                "rr1": int(rr1),
                "cc0": int(cc0),
                "cc1": int(cc1),
                "overlap": int(cc1 - cc0 + 1),
                "gap": int(rr1 - rr0 + 1),
            }

    return None


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    info = _bridge_info(fa, fb)
    if info is None:
        return set()
    return {
        (int(r), int(c))
        for r in range(int(info["rr0"]), int(info["rr1"]) + 1)
        for c in range(int(info["cc0"]), int(info["cc1"]) + 1)
    }


def _bridge_cells_exact(fa: dict, fb: dict) -> set[tuple[int, int]]:
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])

    same_row = bool(int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]))
    if same_row:
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        elif int(c1b) < int(c0a):
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        else:
            cc0, cc1 = 1, 0
        if int(cc0) <= int(cc1):
            return {
                (int(r), int(c))
                for r in range(int(rr0), int(rr1) + 1)
                for c in range(int(cc0), int(cc1) + 1)
            }

    same_col = bool(int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]))
    if same_col:
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(r1a) < int(r0b):
            rr0, rr1 = int(r1a + 1), int(r0b - 1)
        elif int(r1b) < int(r0a):
            rr0, rr1 = int(r1b + 1), int(r0a - 1)
        else:
            rr0, rr1 = 1, 0
        if int(rr0) <= int(rr1):
            return {
                (int(r), int(c))
                for r in range(int(rr0), int(rr1) + 1)
                for c in range(int(cc0), int(cc1) + 1)
            }

    return set()


def _paint_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells(fa, fb):
        out[int(r), int(c)] = int(color)


def _paint_bridge_exact(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    for r, c in _bridge_cells_exact(fa, fb):
        out[int(r), int(c)] = int(color)


def _center(f: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, f["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _distance_score(src: dict, dst: dict) -> tuple[int, int, int]:
    info = _bridge_info(src, dst)
    if info is None:
        return (0, 0, -10**6)
    r1_, c1_ = _center(src)
    r2_, c2_ = _center(dst)
    d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
    return (1, int(info["overlap"]), -int(d))


def _boxes_overlap(a: dict, b: dict) -> bool:
    r0a, c0a, r1a, c1a = map(int, a["bbox"])
    r0b, c0b, r1b, c1b = map(int, b["bbox"])
    return not (
        int(r1a) < int(r0b)
        or int(r1b) < int(r0a)
        or int(c1a) < int(c0b)
        or int(c1b) < int(c0a)
    )


def _make_frame(r0: int, c0: int, outer: int, t: int, inner_col: int) -> dict:
    r1 = int(r0 + outer - 1)
    c1 = int(c0 + outer - 1)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "t": int(t),
        "inner": int(inner_col),
        "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
    }


def _parse_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, int, int, int, list[dict], dict[int, list[dict]], list[int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    border = _most_common_nonbg(g, int(bg))
    assert not (border is None)

    frames = _detect_frames(g, int(border))
    assert not (not frames)

    by: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by[int(frame["inner"])].append(frame)
    for col in by:
        by[int(col)].sort(key=_frame_order_key)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert not (ctrl_r is None)

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert not (len(ctrl) < 2)

    return g, int(H), int(W), int(bg), int(border), frames, by, ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect freely scattered framed squares via colored bridges following the control sequence.

    Combo logic:
    - Geometric bridgeability uses D4-style inner-strip overlap, not exact lattice bands.
    - When a control color matches multiple squares, D2-style global position order
      (top-to-bottom, then left-to-right) chooses the first unvisited one.
    """
    g, _H, _W, _bg, _border, _frames, by, ctrl = _parse_scene(grid)
    out = g.copy()

    start_color = int(ctrl[0])
    assert not (start_color not in by)

    visited: set[int] = set()
    cur = None
    for f in by[int(start_color)]:
        order = int(f["order"])
        if order not in visited:
            cur = f
            visited.add(order)
            break
    assert not (cur is None)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)

        nxt = None
        for f in by.get(int(b), []):
            order = int(f["order"])
            if order not in visited:
                nxt = f
                break
        if nxt is None:
            continue

        _paint_bridge(out, cur, nxt, int(a))
        visited.add(int(nxt["order"]))
        cur = nxt

    return out


def _legacy_distance_main(grid: np.ndarray) -> np.ndarray:
    g, _H, _W, _bg, _border, _frames, by, ctrl = _parse_scene(grid)
    out = g.copy()

    start_color = int(ctrl[0])
    assert not (start_color not in by)
    cur = by[int(start_color)][0]

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        bridgeable = [f for f in by.get(int(b), []) if _bridge_info(cur, f) is not None]
        if not bridgeable:
            continue
        nxt = max(bridgeable, key=lambda f: _distance_score(cur, f))
        _paint_bridge(out, cur, nxt, int(a))
        cur = nxt

    return out


def _legacy_order_exact_main(grid: np.ndarray) -> np.ndarray:
    g, _H, _W, _bg, _border, _frames, by, ctrl = _parse_scene(grid)
    out = g.copy()

    start_color = int(ctrl[0])
    assert not (start_color not in by)

    visited: set[int] = set()
    cur = None
    for f in by[int(start_color)]:
        order = int(f["order"])
        if order not in visited:
            cur = f
            visited.add(order)
            break
    assert not (cur is None)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)

        nxt = None
        for f in by.get(int(b), []):
            order = int(f["order"])
            if order not in visited:
                nxt = f
                break
        if nxt is None:
            continue

        _paint_bridge_exact(out, cur, nxt, int(a))
        visited.add(int(nxt["order"]))
        cur = nxt

    return out


def _place_scene(
    *,
    outer: int,
    t: int,
    path_colors: list[int],
    bg: int,
    border: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    inner = int(outer - 2 * t)
    if int(inner) < 2:
        return None

    s1 = int(np.random.randint(1, int(inner)))
    gh1 = int(np.random.randint(3, 5))
    gv_decoy = int(np.random.randint(1, 3))
    gv2 = int(np.random.randint(2, 5))
    s3 = int(np.random.randint(1, int(inner)))
    gh2 = int(np.random.randint(1, 4))
    s4 = int(np.random.randint(0, int(inner)))

    start_col, second_col, third_col, fourth_col = map(int, path_colors)

    frames = {
        "start": _make_frame(0, 0, int(outer), int(t), int(start_col)),
        "target": _make_frame(int(s1), int(outer + gh1), int(outer), int(t), int(second_col)),
        "third": _make_frame(int(s1 + outer + gv2), int(outer + gh1 + s3), int(outer), int(t), int(third_col)),
        "fourth": _make_frame(
            int(s1 + outer + gv2 + s4),
            int(outer + gh1 + s3 + outer + gh2),
            int(outer),
            int(t),
            int(fourth_col),
        ),
        "decoy": _make_frame(int(outer + gv_decoy), 0, int(outer), int(t), int(second_col)),
    }

    ordered_names = ["start", "target", "third", "fourth", "decoy"]
    frame_list = [frames[name] for name in ordered_names]
    for i in range(len(frame_list)):
        for j in range(i + 1, len(frame_list)):
            if _boxes_overlap(frame_list[i], frame_list[j]):
                return None

    path_pairs = [("start", "target"), ("target", "third"), ("third", "fourth")]
    bridges: list[tuple[str, str, set[tuple[int, int]]]] = []
    all_bridge_cells: set[tuple[int, int]] = set()
    partial_edges = 0

    for a_name, b_name in path_pairs:
        fa = frames[a_name]
        fb = frames[b_name]
        info = _bridge_info(fa, fb)
        if info is None:
            return None
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & all_bridge_cells:
            return None
        if info["axis"] == "h":
            full = int(fa["inner_bbox"][2] - fa["inner_bbox"][0] + 1)
        else:
            full = int(fa["inner_bbox"][3] - fa["inner_bbox"][1] + 1)
        if int(info["overlap"]) < int(full):
            partial_edges += 1
        bridges.append((a_name, b_name, cells))
        all_bridge_cells |= set(cells)

    decoy_info = _bridge_info(frames["start"], frames["decoy"])
    if decoy_info is None:
        return None

    for name, frame in frames.items():
        r0, c0, r1, c1 = map(int, frame["bbox"])
        for r, c in all_bridge_cells:
            if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1):
                return None
        if name == "decoy":
            continue
        for r, c in _bridge_cells(frames["start"], frames["decoy"]):
            if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1):
                return None

    target_order = sorted([frames["target"], frames["decoy"]], key=_frame_order_key)
    if target_order[0] is not frames["target"]:
        return None

    if _distance_score(frames["start"], frames["decoy"]) <= _distance_score(frames["start"], frames["target"]):
        return None

    min_r0 = min(int(f["bbox"][0]) for f in frame_list)
    min_c0 = min(int(f["bbox"][1]) for f in frame_list)
    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    dr = int(top_margin - min_r0)
    dc = int(left_margin - min_c0)

    shifted_frames: dict[str, dict] = {}
    for name, frame in frames.items():
        r0, c0, r1, c1 = map(int, frame["bbox"])
        shifted_frames[name] = {
            "bbox": (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc)),
            "t": int(frame["t"]),
            "inner": int(frame["inner"]),
            "inner_bbox": (
                int(frame["inner_bbox"][0] + dr),
                int(frame["inner_bbox"][1] + dc),
                int(frame["inner_bbox"][2] + dr),
                int(frame["inner_bbox"][3] + dc),
            ),
        }

    if bool(int(np.random.randint(0, 2))):
        max_c1 = max(int(f["bbox"][3]) for f in shifted_frames.values())
        mirrored: dict[str, dict] = {}
        for name, frame in shifted_frames.items():
            r0, c0, r1, c1 = map(int, frame["bbox"])
            width = int(c1 - c0 + 1)
            new_c0 = int(max_c1 - c1)
            new_c1 = int(new_c0 + width - 1)
            mirrored[name] = {
                "bbox": (int(r0), int(new_c0), int(r1), int(new_c1)),
                "t": int(frame["t"]),
                "inner": int(frame["inner"]),
                "inner_bbox": (
                    int(frame["inner_bbox"][0]),
                    int(max_c1 - frame["inner_bbox"][3]),
                    int(frame["inner_bbox"][2]),
                    int(max_c1 - frame["inner_bbox"][1]),
                ),
            }
        shifted_frames = mirrored

    max_r1 = max(int(f["bbox"][2]) for f in shifted_frames.values())
    max_c1 = max(int(f["bbox"][3]) for f in shifted_frames.values())
    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    ctrl_cols = [int(1 + 2 * i) for i in range(4)]
    right_margin = int(np.random.randint(1, 3))
    bottom_margin = 1
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_cols[-1] + 2))
    if not (8 <= int(H) <= 30 and 8 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for name in ordered_names:
        frame = shifted_frames[name]
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(frame["inner"]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    for a_name, b_name in path_pairs:
        fa = shifted_frames[a_name]
        fb = shifted_frames[b_name]
        cells = _bridge_cells(fa, fb)
        if not cells:
            return None
        if cells & bridge_cells_all:
            return None
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                return None
            if int(out[int(r), int(c)]) != int(bg):
                return None
        for r, c in cells:
            out[int(r), int(c)] = int(shifted_frames[a_name]["inner"])
        bridge_cells_all |= set(cells)

    ctrl_colors = [int(start_col), int(second_col), int(third_col), int(fourth_col)]
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
        legacy_d2 = _legacy_distance_main(inp)
        legacy_d4 = _legacy_order_exact_main(inp)
    except Exception:
        return None

    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None
    if bool(np.array_equal(legacy_d2, out)):
        return None
    if bool(np.array_equal(legacy_d4, out)):
        return None
    if int(partial_edges) < 1:
        return None

    metrics = {
        "partial_edges": int(partial_edges),
        "ctrl_len": 4,
    }
    return inp, out, metrics


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    t = 1
    inner = int(np.random.randint(2, 5))
    outer = int(inner + 2 * t)
    if not (4 <= int(outer) <= 6):
        return None

    path_colors = list(map(int, np.random.choice(np.array(inner_palette, dtype=int), size=4, replace=False)))
    placed = _place_scene(
        outer=int(outer),
        t=int(t),
        path_colors=path_colors,
        bg=int(bg),
        border=int(border),
    )
    if placed is None:
        return None
    inp, _out, _metrics = placed
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
