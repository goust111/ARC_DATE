# Combo variant of 3e6067c3 (index 16)
# combo_spec: D6/2+D5/0
# Name: 桥梁覆盖边框+终点色桥梁

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4


# Design analysis for combo D6/2 + D5/0
# ======================================
# a) 具体化与行为追踪
# - 原任务输入仍是“同尺寸带框方块 + 底部控制行”，控制色按访问顺序指定要经过的内部色方块。
# - 原任务输出在相邻访问方块之间画矩形桥，桥宽等于内部填充区宽度，且只填两框之间的纯背景带，
#   颜色取起点方块色。若一段是 2 -> 5，则桥画在 gap 上，颜色是 2。
# - 本融合版同时改两点，而且两点作用在同一桥段上：
#   1. 像素覆盖范围从“只填 gap”变成“从一侧 inner_bbox 外缘一直涂到另一侧 inner_bbox 外缘”，
#      因此会覆盖两端朝向通道的边框条带，把两个方块视觉上打通。
#   2. 这整段打通带的颜色不再来自起点，而来自终点。于是 2 -> 5 段会覆盖 source 边框、gap、
#      target 边框，并全部涂成 5。
# - 具体小例子：水平一对 thickness=1 的方块可抽象成
#   old: [inner 2][border][bg bg][border][inner 5]
#   combo: [inner 2][5][5 5][5][inner 5]
#   差异不是“多一条桥”和“换个颜色”各自独立发生，而是同一条桥既改了覆盖范围，又改了颜色归属。
#
# b) 复杂度校准
# - 不改变对象识别、控制行读取、起点选择、候选选择和几何对齐标准，避免引入与 combo 无关的新推理。
# - 生成器继续让路径为 4 邻自避免路径，因此每一段都真的是同一行带或列带上的相邻方块。
# - 为了让终点着色肉眼可见，生成器保留“同一行/列颜色唯一”的约束，使每条桥都满足 source != dest。
# - 为了让覆盖边框肉眼可见，生成器要求每条桥至少覆盖一些原边框像素，拒绝退化成只填背景的样本。
#
# c) 隐式假设审查
# - 方块检测仍在 INPUT 上进行；因此 INPUT 里的框必须保持完整，generator 只能在虚拟 OUTPUT 中覆写边框，
#   回推到 INPUT 时必须恢复原边框。
# - `aligned()` 与候选最近选择依赖的是 inner_bbox 的行带/列带，而不是桥梁颜色或最终输出形状；
#   这些前提在融合后仍成立。
# - 原 solver 中真正绑定旧语义的关键点有两处：
#   1. 桥像素集合默认只来自 gap；融合后必须改成 `_bridge_cells()` 覆盖端点边框条带。
#   2. `draw_bridge(cur, nxt, a)` 默认桥色来自起点；融合后必须改成 `draw_bridge(cur, nxt, b)`。
# - generator 的旧检查 `out[r, c] == bg` 也失效，因为 bridge cells 现在合法地落在 border 上；
#   必须放宽为 `bg` 或 `border`，同时要求确实覆盖到了 border。
# - 多项改动叠加后没有新的方向冲突：匹配目标仍由 `ctrl[i+1]` 指向，桥色也取 `ctrl[i+1]`，
#   两者一致地指向“下一块”，不会和 `assert cur["inner"] == a` 冲突。
#
# d) 放置合法性
# - 仍使用 output-first：先画完整方块和控制行，再按路径画融合桥，最后把桥抹掉得到输入。
# - 方块放在带背景间隔的格点上，路径仅连接 4 邻格点，`_bridge_cells()` 只会覆盖两端方块的朝向边框
#   和中间通道，不会穿过无关方块。
# - generator 显式检查：桥像素不越界、不与已有桥重叠、桥只落在 `bg/border` 上。
#
# e) 视觉可观察性
# - 人类看 2-3 组样例应能同时看到两件事：桥不再停在框外，而是“切开”框边；桥色又总与终点方块一致。
# - 由于控制序列直接给出访问顺序，而相邻方块颜色不同，解题者能比较“桥朝向哪一块”与“桥染成哪一色”，
#   从而归纳出终点着色。
#
# f) 可视化整洁度
# - 输出仍只有三层信息：完整方块结构、桥、控制行，没有额外标记系统。
# - 桥虽然覆盖边框，但覆盖范围严格限制在与 inner_bbox 同宽的走廊内；未接触的三条边仍保留原 border，
#   所以对象整体仍容易识别，不会变成杂乱色块。
#
# g) 变化的可推断性
# - 颜色不做隐藏映射，桥直接复用终点方块的内部色。
# - 退化量化指标可以同时看两项：
#   1. `dest_diff_segments`: 桥段中 source_color != dest_color 的段数，应等于总桥段数；
#   2. `border_override_segments`: 至少覆盖 1 个边框像素的桥段数，也应等于总桥段数。
# - 生成器把这两个条件都作为样本合法性的硬约束，因此不会生成与原任务同输出的退化样本。
#
# 9) main() 新增假设与 generator 保证
# - solver 假设所有有效对象都是单色边框包住单色内部的规则方块；generator 仅生成这种对象。
# - solver 假设相邻控制色对应的两个目标方块在 inner_bbox 上共享行带或列带；generator 的格点路径保证此事。
# - solver 假设桥覆盖边框后不会影响后续匹配；因为匹配总在 INPUT 上完成，而 INPUT 不含桥，故成立。
# - solver 假设目标色可直接由 `ctrl[i+1]` / `nxt["inner"]` 读取；generator 直接用路径方块内部色编码控制行。


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(frame_a: dict, frame_b: dict) -> tuple[bool, bool]:
    a = frame_a["inner_bbox"]
    b = frame_b["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _pick_start(start_cands: list[dict], nxt_cands: list[dict]) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]
    best = None
    bestf = None
    for frame in start_cands:
        for g2 in (nxt_cands if nxt_cands else [frame]):
            sr, sc = _aligned(frame, g2)
            al = 1 if (sr or sc) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(g2)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(al), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = frame
    assert bestf is not None
    return bestf


def _pick_best(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    r1, c1 = _center(cur)
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        al = 1 if (sr or sc) else 0
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(al), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
    assert bestf is not None
    return bestf


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

        found = None
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

            found = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(tt),
                "inner": int(next(iter(vals))),
                "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
            }
            break

        if found is not None:
            frames.append(found)
    return frames


def _find_border(g: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    raise AssertionError("missing border color")


def _find_control_row(g: np.ndarray, bg: int, frames: list[dict]) -> int:
    h = int(g.shape[0])
    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)
    for r in range(int(h - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            return int(r)
    raise AssertionError("missing control row")


def _control_colors(g: np.ndarray, bg: int, ctrl_r: int) -> list[int]:
    return [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    ia = frame_a["inner_bbox"]
    ib = frame_b["inner_bbox"]
    if int(ia[0]) == int(ib[0]) and int(ia[2]) == int(ib[2]):
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(frame_a["bbox"][1]) < int(frame_b["bbox"][1]):
            left, right = frame_a, frame_b
        else:
            left, right = frame_b, frame_a
        cc0 = int(left["inner_bbox"][3] + 1)
        cc1 = int(right["inner_bbox"][1] - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    if int(ia[1]) == int(ib[1]) and int(ia[3]) == int(ib[3]):
        cc0, cc1 = int(ia[1]), int(ia[3])
        if int(frame_a["bbox"][0]) < int(frame_b["bbox"][0]):
            top, bottom = frame_a, frame_b
        else:
            top, bottom = frame_b, frame_a
        rr0 = int(top["inner_bbox"][2] + 1)
        rr1 = int(bottom["inner_bbox"][0] - 1)
        if int(rr0) > int(rr1):
            return set()
        return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}
    return set()


def _draw_bridge(out: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> None:
    h, w = out.shape
    for r, c in _bridge_cells(frame_a, frame_b):
        if 0 <= int(r) < int(h) and 0 <= int(c) < int(w):
            out[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Consecutive control colors still specify the visitation path through inner-colored frames.
    - Each bridge segment spans from one inner cavity to the next, overwriting traversed border strips.
    - The whole segment uses the destination frame's inner color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    border = _find_border(g, int(bg))
    frames = _detect_frames(g, int(border))
    assert frames

    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)

    ctrl_r = _find_control_row(g, int(bg), frames)
    ctrl = _control_colors(g, int(bg), int(ctrl_r))
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert int(start_color) in by_inner
    cur = _pick_start(by_inner[int(start_color)], by_inner.get(int(ctrl[1]), []))

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        src_color = int(ctrl[i])
        dst_color = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(src_color)
        if int(dst_color) not in by_inner:
            continue
        nxt = _pick_best(cur, by_inner[int(dst_color)])
        _draw_bridge(out, cur, nxt, int(dst_color))
        cur = nxt

    return out


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _trace_path(inp: np.ndarray) -> tuple[list[dict], list[int], int, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    border = _find_border(g, int(bg))
    frames = _detect_frames(g, int(border))
    assert frames
    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)

    ctrl_r = _find_control_row(g, int(bg), frames)
    ctrl = _control_colors(g, int(bg), int(ctrl_r))
    assert len(ctrl) >= 2
    cur = _pick_start(by_inner[int(ctrl[0])], by_inner.get(int(ctrl[1]), []))

    path_frames = [cur]
    for i in range(int(len(ctrl)) - 1):
        dst_color = int(ctrl[i + 1])
        nxt = _pick_best(cur, by_inner[int(dst_color)])
        path_frames.append(nxt)
        cur = nxt

    return path_frames, ctrl, int(bg), int(border)


def fused_behavior_stats(inp: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a sample actually exhibits the combo behavior.

    Metrics:
    - segments: total bridge segments.
    - dest_diff_segments: segments where destination color differs from source color.
    - border_override_segments: segments whose bridge crosses at least one original border pixel.
    - border_override_cells: total number of overridden border pixels across all segments.
    """
    g = np.array(inp, dtype=int)
    path_frames, ctrl, _bg, border = _trace_path(g)
    stats = {
        "segments": 0,
        "dest_diff_segments": 0,
        "border_override_segments": 0,
        "border_override_cells": 0,
    }
    for frame_a, frame_b, src_color, dst_color in zip(path_frames, path_frames[1:], ctrl, ctrl[1:]):
        stats["segments"] += 1
        if int(src_color) != int(dst_color):
            stats["dest_diff_segments"] += 1
        cells = _bridge_cells(frame_a, frame_b)
        border_hits = sum(1 for r, c in cells if int(g[int(r), int(c)]) == int(border))
        if int(border_hits) > 0:
            stats["border_override_segments"] += 1
            stats["border_override_cells"] += int(border_hits)
    return stats


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        cur_path = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, cur_path[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            cur_path.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in cur_path}) < 2 or len({c for _r, c in cur_path}) < 2:
            continue
        path = cur_path
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

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
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

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_r = int(max_r1 + int(np.random.randint(2, 6)))
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)
    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        col = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(col)
        used_in_row[int(rr)].add(int(col))
        used_in_col[int(cc)].add(int(col))
    if len(colors_by_slot) != len(path):
        return None

    base = np.full((int(H), int(W)), int(bg), dtype=int)
    for frame in frames:
        rr, cc = map(int, frame["slot"])
        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        _draw_frame(
            base,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            t=int(t),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        base[int(ctrl_r), int(cc)] = int(col)

    out = base.copy()
    bridge_cells_all: set[tuple[int, int]] = set()
    border_override_segments = 0
    border_override_cells = 0
    dest_diff_segments = 0
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        frame_a = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        frame_b = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(frame_a, frame_b)
        if not cells or cells & bridge_cells_all:
            ok = False
            break
        src_color = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        dst_color = int(colors_by_slot[(int(b_slot[0]), int(b_slot[1]))])
        if int(src_color) == int(dst_color):
            ok = False
            break
        edge_border_override = 0
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(base[int(r), int(c)]) not in (int(bg), int(border)):
                ok = False
                break
            if int(base[int(r), int(c)]) == int(border):
                edge_border_override += 1
        if not ok or int(edge_border_override) == 0:
            ok = False
            break
        for r, c in cells:
            out[int(r), int(c)] = int(dst_color)
        bridge_cells_all |= set(cells)
        border_override_segments += 1
        border_override_cells += int(edge_border_override)
        dest_diff_segments += 1
    if not ok:
        return None

    inp = base.copy()

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

    stats = fused_behavior_stats(inp)
    if int(stats["segments"]) != int(len(path) - 1):
        return None
    if int(stats["dest_diff_segments"]) != int(dest_diff_segments):
        return None
    if int(stats["border_override_segments"]) != int(border_override_segments):
        return None
    if int(stats["border_override_cells"]) != int(border_override_cells):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
