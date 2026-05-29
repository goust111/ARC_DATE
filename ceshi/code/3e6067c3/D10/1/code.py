# Variant of 3e6067c3
# Dimension: D10 — 对象属性：边框颜色的角色 (meta)
# Change: 所有方块共享同一种边框颜色（最常见的非背景色），边框仅作结构分隔用
#   → 每个方块边框颜色不同，桥梁颜色取起点方块的边框色；内部颜色只负责和控制像素匹配。
#
# Design analysis
# a) 具体化与行为追踪:
#    - 输入仍是若干等尺寸带框方块 + 底部控制像素序列。不同的是，每个方块的边框色都不同，
#      而控制像素仍然只复述各方块的内部填充色。
#    - 输出仍是在路径上补出桥梁；原任务把桥涂成“当前内部色”，现在改成“当前方块的边框色”。
#    - 例子: 若控制序列前两项是 [red_inner, blue_inner]，并且 red_inner 所在方块边框为 yellow，
#      则从该方块连向 blue_inner 方块时，桥应涂 yellow；原任务则会涂 red_inner。
# b) 复杂度校准:
#    - 若仍让所有边框同色，则 D10 在视觉上不可见，所以生成器强制每个方块边框互异。
#    - 若边框色和内部色混用过多，人类难以看出“控制看内部、桥看边框”的分工，所以生成器把两组颜色分离。
#    - 路径长度限制到 4..5，使 10 色调色板下既能给每个方块唯一边框，又保留 4..5 个内部候选色。
# c) 隐式假设审查:
#    - 原 `main()` 用“最常见非背景色”认定统一边框，这在变体中失效；必须改成遍历所有非背景色连通块。
#    - 原 `draw_bridge(..., color=a)` 默认桥色等于当前内部色，这在变体中失效；必须读 `fa["border"]`。
#    - 其余路径判定仍依赖“内部色用于匹配、同行同列唯一性帮助消歧”，这一点保持不变。
# d) 放置合法性:
#    - 仍在规则晶格上放置等尺寸方块，行列间至少隔 1 个背景像素。
#    - 路径只走 4 邻格自避免步，因此任意相邻方块都严格同行或同列，桥梁矩形可合法落在空隙里。
#    - 生成器先构造输出，再抹去桥梁得到输入，并用 `main(input)` 回验。
# e) 视觉可观察性:
#    - 控制行只出现内部色；输出桥梁改成另一组边框色，因此两三组样例就能看出“匹配色”和“绘制色”脱钩。
#    - 若桥色仍偶尔等于内部色，会弱化 D10，可通过调色板分离彻底避免。
# f) 可视化整洁度:
#    - 仍采用规则方块和纯矩形桥梁，不引入额外纹理或叠层。
#    - 桥梁只填两框之间的背景空隙，不覆盖框和控制行。
# g) 颜色变化可推断性:
#    - 颜色变化不是无提示映射，而是由每个方块可见的边框色直接决定。
#    - 因为边框色在输入中已经显示，解题者可以从输出桥色回溯到“桥跟起点边框走，而不是跟控制像素走”。

from __future__ import annotations

from collections import defaultdict
import numpy as np

from arc_utils import colors_present, get_bg, in_bounds, label_mask, neighbors4


def _detect_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for border in colors_present(g, exclude={bg}):
        lab, n = label_mask(g == int(border), connectivity=4)
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

            comp_area = int(pts.shape[0])
            found = None
            for tt in range(1, int(min(h, w) // 2) + 1):
                inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
                if inner.size == 0:
                    continue
                if bool((inner == int(border)).any()):
                    continue
                u = set(map(int, inner.flatten()))
                if len(u) != 1:
                    continue

                ring_ok = True
                if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                    ring_ok = False
                if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                    ring_ok = False
                if not ring_ok:
                    continue

                inner_h = int(inner.shape[0])
                inner_w = int(inner.shape[1])
                ring_area = int(h * w - inner_h * inner_w)
                if int(comp_area) != int(ring_area):
                    continue

                found = {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "t": int(tt),
                    "border": int(border),
                    "inner": int(next(iter(u))),
                    "inner_bbox": (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt)),
                }
                break

            if found is not None:
                frames.append(found)
    return frames


def _center(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return int((r0 + r1) // 2), int((c0 + c1) // 2)


def _aligned(f1: dict, f2: dict) -> tuple[bool, bool]:
    a = f1["inner_bbox"]
    b = f2["inner_bbox"]
    same_row = bool(int(a[0]) == int(b[0]) and int(a[2]) == int(b[2]))
    same_col = bool(int(a[1]) == int(b[1]) and int(a[3]) == int(b[3]))
    return same_row, same_col


def _choose_start(start_cands: list[dict], nxt_cands: list[dict]) -> dict:
    if len(start_cands) == 1:
        return start_cands[0]

    best = None
    bestf = None
    for frame in start_cands:
        for g2 in (nxt_cands if nxt_cands else [frame]):
            sr, sc = _aligned(frame, g2)
            aligned_score = 1 if (sr or sc) else 0
            r1, c1 = _center(frame)
            r2, c2 = _center(g2)
            dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
            cand = (int(aligned_score), -int(dist))
            if best is None or cand > best:
                best = cand
                bestf = frame
    assert bestf is not None
    return bestf


def _pick_next(cur: dict, cands: list[dict]) -> dict:
    best = None
    bestf = None
    for frame in cands:
        sr, sc = _aligned(cur, frame)
        aligned_score = 1 if (sr or sc) else 0
        r1, c1 = _center(cur)
        r2, c2 = _center(frame)
        dist = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
        cand = (int(aligned_score), -int(dist))
        if best is None or cand > best:
            best = cand
            bestf = frame
    assert bestf is not None
    return bestf


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


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    frames = _detect_frames(g, int(bg))
    assert frames

    by_inner: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_inner[int(frame["inner"])].append(frame)

    max_frame_r = max(int(frame["bbox"][2]) for frame in frames)

    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by_inner
    nxt_color = int(ctrl[1])
    cur = _choose_start(by_inner[start_color], by_inner.get(nxt_color, []))

    out = g.copy()

    def draw_bridge(fa: dict, fb: dict) -> None:
        cells = _bridge_cells(fa, fb)
        bridge_color = int(fa["border"])
        for r, c in cells:
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(bridge_color)

    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by_inner:
            continue
        nxt = _pick_next(cur, by_inner[b])
        draw_bridge(cur, nxt)
        cur = nxt

    return out


_DIGITS = [int(i) for i in range(10)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    non_bg = [int(d) for d in _DIGITS if int(d) != int(bg)]

    n_frames = int(np.random.randint(4, 6))
    if len(non_bg) - int(n_frames) < 3:
        return None

    shuffled = [int(x) for x in np.random.permutation(np.array(non_bg, dtype=int)).tolist()]
    border_palette = [int(x) for x in shuffled[:n_frames]]
    inner_palette = [int(x) for x in shuffled[n_frames:]]
    if len(inner_palette) < 3:
        return None

    t = int(np.random.choice(np.array([1, 1, 2], dtype=int)))
    inner = int(np.random.randint(1, 4))
    outer = int(inner + 2 * t)
    if not (3 <= int(outer) <= 7):
        return None

    for _shape_try in range(200):
        R = int(np.random.randint(2, 5))
        C = int(np.random.randint(2, 5))
        if int(R * C) < int(n_frames):
            continue

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
            if len({int(r) for r, _c in p}) < 2 or len({int(c) for _r, c in p}) < 2:
                continue
            path = p
            break
        if path is None:
            continue

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

        frames: list[dict] = []
        slot_to_frame: dict[tuple[int, int], dict] = {}
        for idx, (rr, cc) in enumerate(path):
            r0 = int(r0s[int(rr)])
            c0 = int(c0s[int(cc)])
            r1 = int(r0 + outer - 1)
            c1 = int(c0 + outer - 1)
            frame = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
                "slot": (int(rr), int(cc)),
                "border": int(border_palette[int(idx)]),
            }
            frames.append(frame)
            slot_to_frame[(int(rr), int(cc))] = frame

        max_r1 = max(int(frame["bbox"][2]) for frame in frames)
        max_c1 = max(int(frame["bbox"][3]) for frame in frames)
        ctrl_gap = int(np.random.randint(2, 6))
        ctrl_r = int(max_r1 + ctrl_gap)
        bottom_margin = 1
        right_margin = int(np.random.randint(1, 5))
        H = int(ctrl_r + 1 + bottom_margin)
        W = int(max_c1 + 1 + right_margin)
        if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
            continue

        used_in_row: dict[int, set[int]] = defaultdict(set)
        used_in_col: dict[int, set[int]] = defaultdict(set)

        start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
        colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
        used_in_row[0].add(int(start_color))
        used_in_col[0].add(int(start_color))

        color_ok = True
        for rr, cc in path[1:]:
            rr = int(rr)
            cc = int(cc)
            forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
            choices = [int(x) for x in inner_palette if int(x) not in forbid]
            if not choices:
                color_ok = False
                break
            col = int(np.random.choice(np.array(choices, dtype=int)))
            colors_by_slot[(int(rr), int(cc))] = int(col)
            used_in_row[int(rr)].add(int(col))
            used_in_col[int(cc)].add(int(col))
        if not color_ok:
            continue

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
                border=int(frame["border"]),
                inner_col=int(colors_by_slot[(int(rr), int(cc))]),
            )

        bridge_cells_all: set[tuple[int, int]] = set()
        ok = True
        for a_slot, b_slot in zip(path, path[1:]):
            fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
            fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
            cells = _bridge_cells(fa, fb)
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
            paint = int(fa["border"])
            for r, c in cells:
                out[int(r), int(c)] = int(paint)
            bridge_cells_all |= set(cells)
        if not ok:
            continue

        ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
        ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
        if len(ctrl_colors) < 2 or int(ctrl_cols[-1]) >= int(W):
            continue
        for col, cc in zip(ctrl_colors, ctrl_cols):
            out[int(ctrl_r), int(cc)] = int(col)

        inp = out.copy()
        for r, c in bridge_cells_all:
            inp[int(r), int(c)] = int(bg)

        if int(get_bg(inp)) != int(bg):
            continue
        if len({int(frame["border"]) for frame in frames}) != len(frames):
            continue
        if not all(int(frame["border"]) != int(colors_by_slot[frame["slot"]]) for frame in frames):
            continue

        try:
            pred = main(inp)
        except Exception:
            continue
        if pred.shape != out.shape:
            continue
        if not bool(np.array_equal(pred, out)):
            continue

        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
