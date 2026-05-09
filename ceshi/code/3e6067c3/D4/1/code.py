# Variant of 3e6067c3
# Dimension: D4 — 目标对象：带框方块的结构 (dst)
# Change: 等尺寸的带框方块（外层边框色 + 内层纯色填充），排列在网格状的格点上... → 嵌套双层框架

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

# Design analysis for the D4 variant
# a) Concrete pixel behavior:
#    原题的单个对象是“单层边框 + 纯色内部”，桥的厚度直接等于这块纯色内部的高度/宽度。
#    本变体把对象改成 3 个同心矩形带：外边框(颜色 A)、内边框(颜色 B)、中心填充(颜色 C)。
#    输入里没有桥，输出里仍按控制行顺序连桥，但桥的横截面要与“被内边框包住的中心填充区”对齐。
#    例子：若对象是 7x7，则像素结构是 1 格外框 + 1 格内框 + 3x3 核心；原逻辑会把 5x5 的
#    单层框内部误认为可通行区域，而变体要求真正可通行的带宽只有 3。
# b) Complexity calibration:
#    参考任务 45a5af55 说明“嵌套框架”本身已经足够成为可观察变化，因此这里把两层边框都固定为 1 像素，
#    不再额外引入每个对象不同的边框厚度或颜色映射。这样视觉上稳定、差异明显，同时不会让识别自由度过高。
# c) Implicit assumptions audit against the original main():
#    - 原代码假设“最常见非背景色”就是唯一边框色；双层框架后这个假设失效，因此改成先找所有非背景连通块。
#    - 原代码假设 frame component 只包含边框色；现在单个对象是整块非背景实心正方形，需要按 bbox 内的三层结构解析。
#    - 原代码用 inner_bbox 既做对齐判断又决定桥厚；现在必须改成 core/fill bbox，否则桥会比内层框允许的厚。
# d) Placement legality:
#    生成器仍把对象放在规则格点上，行列间至少留 1 格背景，因此对象不重叠、桥段在输出里只填背景间隙。
#    由于对象仍全部等尺寸，沿路径相邻的两个对象必然共享同一行带或列带，solver 的最近对齐策略仍然可用。
# e) Visual observability:
#    2~4 组样例中，人类可以直接看到每个方块从“单层框”变为“外框套内框”，且输出桥明显更窄，
#    因为桥只和中心填充带对齐，而不覆盖两层边框对应的厚度。
# f) Visual cleanliness:
#    同一对象只出现 3 种颜色且是严格同心分层；桥只画在背景间隙里，不穿过框体，因此画面仍整洁。
# g) Color inference:
#    控制行仍只编码中心填充色，不引入新的隐藏颜色逻辑；两层边框颜色在所有对象上全局共享，
#    解题者只需识别“目标对象结构变了”，不需要学习新的颜色对应表。


def _parse_double_frame(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, bbox)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 5:
        return None

    patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if bool((patch == int(bg)).any()):
        return None

    if patch.shape[0] <= 4 or patch.shape[1] <= 4:
        return None

    outer = int(patch[0, 0])
    if outer == int(bg):
        return None
    if not bool((patch[0, :] == int(outer)).all()):
        return None
    if not bool((patch[-1, :] == int(outer)).all()):
        return None
    if not bool((patch[:, 0] == int(outer)).all()):
        return None
    if not bool((patch[:, -1] == int(outer)).all()):
        return None

    inner_patch = patch[1:-1, 1:-1]
    inner_border = int(inner_patch[0, 0])
    if inner_border in (int(bg), int(outer)):
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
    if fill in (int(bg), int(outer), int(inner_border)):
        return None
    if not bool((core == int(fill)).all()):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "outer_border": int(outer),
        "inner_border": int(inner_border),
        "inner": int(fill),
        "fill_bbox": (int(r0 + 2), int(c0 + 2), int(r1 - 2), int(c1 - 2)),
    }


def _detect_frames(g: np.ndarray) -> list[dict]:
    bg = int(get_bg(g))
    lab, n = label_mask(g != int(bg), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 25:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        parsed = _parse_double_frame(g, (int(r0), int(c0), int(r1), int(c1)), bg)
        if parsed is None:
            continue
        if int(pts.shape[0]) != int((r1 - r0 + 1) * (c1 - c0 + 1)):
            continue
        frames.append(parsed)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect nested double-frame squares via colored bridges following a control-pixel sequence.

    Input structure:
    - Non-bg objects are equal-sized solid square components.
    - Each object has outer border color A, inner border color B, and a solid center fill color C.
    - Control row below the lattice lists center-fill colors in visit order.

    Rule:
    - Start at the square whose center fill matches the first control color.
    - For each consecutive pair (ci, ci+1), choose the nearest ci+1 square aligned in the same row/column band.
    - Draw a bridge colored ci across the gap, with thickness determined by the center region enclosed by the inner border.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g)
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

    def center(f: dict) -> tuple[int, int]:
        r0, c0, r1, c1 = map(int, f["bbox"])
        return int((r0 + r1) // 2), int((c0 + c1) // 2)

    def aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
        a = f1["fill_bbox"]
        b = f2["fill_bbox"]
        same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
        same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
        return same_row, same_col

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
            for g2 in (nxt_cands if nxt_cands else [f]):
                sr, sc = aligned(f, g2)
                al = 1 if (sr or sc) else 0
                r1_, c1_ = center(f)
                r2_, c2_ = center(g2)
                d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
                cand = (int(al), -int(d))
                if best is None or cand > best:
                    best = cand
                    bestf = f
        cur = bestf
    assert not (cur is None)

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict, color: int) -> None:
        r0a, c0a, r1a, c1a = map(int, fa["bbox"])
        r0b, c0b, r1b, c1b = map(int, fb["bbox"])
        fill_a = fa["fill_bbox"]
        sr, sc = aligned(fa, fb)
        if sr:
            rr0, rr1 = int(fill_a[0]), int(fill_a[2])
            if int(c1a) < int(c0b):
                cc0, cc1 = int(c1a + 1), int(c0b - 1)
            else:
                cc0, cc1 = int(c1b + 1), int(c0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)
            return
        if sc:
            cc0, cc1 = int(fill_a[1]), int(fill_a[3])
            if int(r1a) < int(r0b):
                rr0, rr1 = int(r1a + 1), int(r0b - 1)
            else:
                rr0, rr1 = int(r1b + 1), int(r0a - 1)
            for r in range(int(rr0), int(rr1) + 1):
                for c in range(int(cc0), int(cc1) + 1):
                    if 0 <= r < H and 0 <= c < W:
                        out[int(r), int(c)] = int(color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        cands = by[b]
        best = None
        bestf = None
        for f in cands:
            sr, sc = aligned(cur, f)
            al = 1 if (sr or sc) else 0
            r1_, c1_ = center(cur)
            r2_, c2_ = center(f)
            d = abs(int(r1_) - int(r2_)) + abs(int(c1_) - int(c2_))
            cand = (int(al), -int(d))
            if best is None or cand > best:
                best = cand
                bestf = f
        nxt = bestf
        assert not (nxt is None)
        draw_bridge(cur, nxt, int(a))
        cur = nxt

    return out


def _draw_double_frame(
    g: np.ndarray,
    *,
    r0: int,
    c0: int,
    outer: int,
    outer_border: int,
    inner_border: int,
    inner_col: int,
) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(outer_border)
    g[int(r0 + 1) : int(r0 + outer - 1), int(c0 + 1) : int(c0 + outer - 1)] = int(inner_border)
    g[int(r0 + 2) : int(r0 + outer - 2), int(c0 + 2) : int(c0 + outer - 2)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    r0a, c0a, r1a, c1a = map(int, fa["bbox"])
    r0b, c0b, r1b, c1b = map(int, fb["bbox"])
    fill_a = fa["fill_bbox"]
    fill_b = fb["fill_bbox"]
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


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    outer_border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_border = int(
        np.random.choice(np.array([d for d in _DIGITS if int(d) not in (int(bg), int(outer_border))], dtype=int))
    )
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(outer_border), int(inner_border))]
    if len(inner_palette) < 3:
        return None

    core = int(np.random.randint(1, 4))
    outer = int(core + 4)

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
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
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
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

    slot_to_frame: dict[tuple[int, int], dict] = {}
    frames: list[dict] = []
    for rr, cc in path:
        r0 = int(r0s[int(rr)])
        c0 = int(c0s[int(cc)])
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "fill_bbox": (int(r0 + 2), int(c0 + 2), int(r1 - 2), int(c1 - 2)),
            "slot": (int(rr), int(cc)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    ctrl_width = int(2 * n_frames)
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(max_c1 + 1 + right_margin, ctrl_width + 1))

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

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_double_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(outer),
            outer_border=int(outer_border),
            inner_border=int(inner_border),
            inner_col=int(colors_by_slot[(rr, cc)]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)

    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
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
