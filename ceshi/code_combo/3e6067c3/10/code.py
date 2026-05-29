# Combo variant of 3e6067c3 (index 10)
# combo_spec: D4/1+D1/0
# Name: 嵌套双层框架+边框匹配

from __future__ import annotations

from collections import defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, label_mask, neighbors4

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    原题里每个对象是“单层边框 + 内部纯色”，控制行按内部色指向目标，桥宽等于单层框内部带宽。
#    现在每个对象变成 3 层同心结构：外层边框 A、内层边框 B、中心填充 C。
#    控制行读取的是 A，而桥只能沿 B 包围出的核心区域延伸。
#    例如某个对象是 7x7，像素层次为 A-ring / B-ring / 3x3 core。原题若看到控制色 4，会找 inner=4 的方块，
#    也会把桥画成 5 格厚；本变体看到控制色 4 时要找 outer_border=4 的对象，并把桥画成仅 3 格厚。
#    因此“识别标签”和“可通行带宽”同时换了参照层，二者是联动而不是独立叠加。
# b) Complexity calibration:
#    为了让视觉差异稳定可见，但不引入过多自由度，生成器固定“每个对象 outer 唯一、inner border 全局共享、
#    core fill 也全局共享”。这样解题者能明确看到：控制色只出现在最外框，而桥又明显比外框内部更窄。
# c) Implicit assumptions broken from the original solver:
#    - “most common non-bg == 唯一边框色”失效，因为外层边框因对象而异。
#    - “边框组件只包含边框色”失效，因为双层框对象在输入里是一个完整非背景实心方块组件。
#    - “按 inner 色分组即可匹配控制行”失效，必须改为按 outer_border 分组。
#    - “inner_bbox 同时代表匹配带和桥宽”失效，双层框下必须使用更内层的 fill_bbox 作为桥宽。
# d) Placement legality:
#    生成器仍在规则格点上放置等尺寸方块，行列间至少留 1 格背景，路径是格点 4 邻自避免路径。
#    因此对象不会重叠，路径相邻对象必共享同一行带或列带，桥只填背景间隙。
# e) Visual observability:
#    少量样例里，人类应能同时观察到两点：
#    1. 控制行颜色出现在外框，不出现在内框或中心；
#    2. 输出桥宽与中心核心区对齐，而不会吞没内层边框。
#    只要看 2~3 组样例，便能排除“按中心色匹配”与“按单层框宽度连桥”的旧解释。
# f) Visual cleanliness:
#    每个对象只用 3 种颜色且严格同心，桥只画在对象间背景通道里。共享的内框色与填充色让版面整齐，
#    唯一外框色承担识别职责，避免多重编码混在同一区域。
# g) Inferability of changes:
#    控制行颜色集合与 outer border 集合完全一致，并与 inner border / fill 颜色集合分离。
#    同时 outer-2 的旧单层通道宽度严格大于 core 宽度，因此桥宽变化也有可见结构证据。
# 9) New solver assumptions and generator guarantees:
#    - solver 假设有效对象是非背景实心正方形组件，且其像素结构恰为 outer-ring / inner-ring / core 三层。
#      generator 仅生成这种对象。
#    - solver 假设控制色都能唯一匹配到某个 outer border。generator 为每个对象分配唯一 outer 色并直接编码到控制行。
#    - solver 假设相邻访问对象在 fill_bbox 上共享同一行带或列带。generator 的 4 邻格点路径保证这一点。
#    - solver 假设桥段不会与其他对象或桥重叠。generator 在 output-first 阶段显式检查并拒绝冲突样本。


def _parse_double_frame(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, bbox)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or h < 5:
        return None

    patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if bool((patch == int(bg)).any()):
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
    if inner_patch.size == 0:
        return None
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
        "fill": int(fill),
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
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if int(pts.shape[0]) != int(area):
            continue
        parsed = _parse_double_frame(g, (int(r0), int(c0), int(r1), int(c1)), int(bg))
        if parsed is not None:
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


def _pick_best(src: dict, candidates: list[dict]) -> dict | None:
    best = None
    best_frame = None
    r1, c1 = _center(src)
    for cand in candidates:
        sr, sc = _aligned(src, cand)
        aligned_score = 1 if (sr or sc) else 0
        r2, c2 = _center(cand)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        key = (int(aligned_score), -int(dist))
        if best is None or key > best:
            best = key
            best_frame = cand
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


def _draw_bridge(out: np.ndarray, frame_a: dict, frame_b: dict, color: int) -> None:
    for r, c in _bridge_cells(frame_a, frame_b):
        out[int(r), int(c)] = int(color)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect nested double-frame squares via bridges.

    Fused rule:
    - Each object is a solid non-bg square whose outer ring color identifies the object.
    - The next target is chosen by matching the control color to that outer ring.
    - The bridge is painted with the current object's outer-ring color.
    - Bridge thickness is limited by the core enclosed by the inner border.
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
    start_candidates = by_outer[int(start_color)]
    if len(start_candidates) == 1:
        cur = start_candidates[0]
    else:
        nxt_color = int(ctrl[1])
        lookahead = by_outer.get(int(nxt_color), [])
        best = None
        cur = None
        for cand in start_candidates:
            targets = lookahead if lookahead else [cand]
            for target in targets:
                sr, sc = _aligned(cand, target)
                aligned_score = 1 if (sr or sc) else 0
                r1, c1 = _center(cand)
                r2, c2 = _center(target)
                dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
                key = (int(aligned_score), -int(dist))
                if best is None or key > best:
                    best = key
                    cur = cand
        assert not (cur is None)

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        src_color = int(ctrl[i])
        dst_color = int(ctrl[i + 1])
        assert int(cur["outer_border"]) == int(src_color)
        assert int(dst_color) in by_outer
        nxt = _pick_best(cur, by_outer[int(dst_color)])
        assert not (nxt is None)
        _draw_bridge(out, cur, nxt, int(src_color))
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    avail = [int(d) for d in _DIGITS if int(d) != int(bg)]

    core = int(np.random.randint(1, 4))
    outer = int(core + 4)

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(7, max_frames)) + 1))
    if len(avail) < int(n_frames + 2):
        return None

    path: list[tuple[int, int]] | None = None
    for _ in range(400):
        p = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for _step in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in seen
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            seen.add((int(nxt[0]), int(nxt[1])))
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

    inner_border = int(np.random.choice(np.array(avail, dtype=int)))
    remain = [int(d) for d in avail if int(d) != int(inner_border)]
    fill_color = int(np.random.choice(np.array(remain, dtype=int)))
    outer_pool = [int(d) for d in remain if int(d) != int(fill_color)]
    if len(outer_pool) < int(n_frames):
        return None
    outer_colors = [int(x) for x in np.random.choice(np.array(outer_pool, dtype=int), size=int(n_frames), replace=False)]

    slot_to_frame: dict[tuple[int, int], dict] = {}
    outer_by_slot: dict[tuple[int, int], int] = {}
    frames: list[dict] = []
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
            "fill": int(fill_color),
        }
        frames.append(frame)
        slot_to_frame[(int(rr), int(cc))] = frame
        outer_by_slot[(int(rr), int(cc))] = int(outer_border)

    max_r1 = max(int(frame["bbox"][2]) for frame in frames)
    max_c1 = max(int(frame["bbox"][3]) for frame in frames)
    ctrl_r = int(max_r1 + int(np.random.randint(2, 6)))
    H = int(ctrl_r + 2)
    W = int(max(max_c1 + 1 + int(np.random.randint(1, 5)), 2 * int(n_frames) + 1))
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
        if not cells or cells & bridge_cells_all:
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
        paint = int(outer_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(outer_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, color in zip(ctrl_cols, ctrl_colors):
        out[int(ctrl_r), int(col)] = int(color)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None

    found = _detect_frames(inp, int(bg))
    if int(len(found)) != int(len(frames)):
        return None
    if len({int(frame["outer_border"]) for frame in found}) != int(len(frames)):
        return None
    if {int(frame["inner_border"]) for frame in found} != {int(inner_border)}:
        return None
    if {int(frame["fill"]) for frame in found} != {int(fill_color)}:
        return None
    if set(ctrl_colors) != {int(frame["outer_border"]) for frame in found}:
        return None
    if not set(ctrl_colors).isdisjoint({int(inner_border), int(fill_color)}):
        return None
    if int(core) >= int(outer - 2):
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
