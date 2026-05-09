# Variant of 7ed72f31
# Dimension: D9 — 原始像素的保留与删除 (transform)
# Change: 反射后原始payload像素和红色标记都保留在输出中... → 删除红色标记
#
# Design analysis
# ---------------
# spec:
#   core_visual_change:
#     input:
#       - 仍然沿用原题的三类“红色指令”：
#         1) 与单色 payload 4 连通的水平红线，表示按该行做镜像；
#         2) 与单色 payload 4 连通的垂直红线，表示按该列做镜像；
#         3) 孤立单个红点，只与一个 8 连通彩色精灵对角接触，表示按该点做中心对称。
#       - 输入里能直接看到红色标记，因此“反射轴/反射中心”仍然可读。
#     output:
#       - 反射生成的镜像 payload 仍然出现，但所有红色标记都消失。
#       - 水平/垂直线型标记消失后，会在原 sprite 和镜像 sprite 之间留下 1 格背景缝；
#         这个空带正是“曾经有红线、现在被擦除”的可观察证据。
#       - 孤立红点消失后，输出只剩对角成对的原 sprite / 镜像 sprite，中间中心点回到背景色。
#   concrete_trace:
#     original_logic_example:
#       - 输入: 上方有 3 格绿色 payload，下面紧贴一条 4 格红横线。
#       - 原 solver: 复制绿色到红线另一侧，并保留原绿色与红线。
#       - 原输出: 绿色 + 红线 + 镜像绿色。
#     variant_logic:
#       - 仍先用红线决定镜像，再把该红线像素擦成背景。
#       - 新输出: 上方绿色 + 中间背景缝 + 下方镜像绿色。
#     delta:
#       - 变化只发生在“输出保留哪些原始像素”这一步，反射几何本身不变。
#   complexity_calibration:
#     - 不引入新颜色映射、不引入新几何类型，仍保持原题三类标记。
#     - 每个样例至少混合 row / col / isolated 三类中的多类，避免“只看一类就像普通复制”。
#     - 保持 3-7 个组、18-28 的网格、单色 payload、唯一 payload 颜色，避免颜色复用导致 solver
#       的颜色筛选把不同组意外串联。
#   implicit_assumption_audit_main:
#     - `lab = label_mask(g != bg, connectivity=4)` 依赖“输入里红标记仍然存在”，本变体保持成立。
#     - 原代码 `out = g.copy()` 默认把红标记保留到输出里；这是旧 D9 假设，变体中必须改。
#     - 新实现改成 `out = g.copy(); out[g == MARK] = bg`：
#       solver 仍然用 `g` 检测标记，但输出底板不再含红色。
#     - 所有镜像写入仍以 `g` 为判定依据，因此不会因为先擦掉 `out` 里的红色而丢失检测信息。
#     - 这也比“最后统一删红色”更稳妥：若某些异常输入把镜像写回原红点位置，后续 payload 仍可覆盖。
#     - `eligible(...)` 的旧假设仍然存在：它按颜色 + bbox/条带筛选像素。为避免串组，generator 保证
#       各组 payload 颜色唯一，且组间留出至少 1 格缓冲。
#     - 孤立红点分支假设红点 4 邻域没有非背景像素；generator 显式保证 sprite 只对角接触红点。
#   placement_legality:
#     - generator 先构造“无红输出”的完整 payload+mirror，再反推“有红输入”。
#     - 每个候选组在放置时同时检查 payload、mirror、marker 三部分的完整 footprint：
#       不能越界，不能与已有组重叠，且与已有 footprint 至少保持 1 格 8 邻域留白。
#     - 因为 input 是 output 的子集再加 marker，所以 output footprint 合法即可保证 solver 不串组。
#   observability:
#     - 多个样例中都会出现“输入有红，输出没红”的全局规律，且镜像位置仍围绕原红位置展开。
#     - 线型标记的消失会留下整条空带，点型标记的消失会留下单点空心中心，这两种视觉信号都很强。
#   visual_cleanliness:
#     - 组间强制留白，避免多个彩色体互相贴边造成哪一组属于哪条红线不清楚。
#     - 每组只用一种 payload 颜色，红色只扮演指令角色，因此层次清楚。
#   color_inference:
#     - 颜色本身不发生变换，只复制几何位置；因此无需额外颜色线索。
#     - 为避免误判，generator 限制每个主组 payload 颜色互不相同。
#   new_main_assumptions_after_edit:
#     - 假设“所有应被删除的红色都来自输入 marker”；generator 中没有其它红色用途。
#     - 假设“镜像不会依赖与别组共享颜色的全局筛选”；generator 用唯一颜色和间隔保障。
#     - 假设“镜像落点不是别组的 marker/payload”；generator 在 output-first 放置时已验证。

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red (2) markers as reflection instructions,
    then erase the red markers from the output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()
    out[g == int(MARK)] = int(bg)

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def _grow_connected(
    h: int,
    w: int,
    area: int,
    seed: tuple[int, int],
    *,
    connectivity: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    cells = {seed}
    while len(cells) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            if int(connectivity) == 4:
                nbrs = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            else:
                nbrs = (
                    (r - 1, c - 1),
                    (r - 1, c),
                    (r - 1, c + 1),
                    (r, c - 1),
                    (r, c + 1),
                    (r + 1, c - 1),
                    (r + 1, c),
                    (r + 1, c + 1),
                )
            for rr, cc in nbrs:
                if 0 <= rr < int(h) and 0 <= cc < int(w) and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def _make_row_group(color: int, rng: random.Random) -> dict[str, set[tuple[int, int, int]]]:
    while True:
        line_len = rng.randint(2, 6)
        depth = rng.randint(2, 5)
        cap = line_len * depth
        if cap >= 3:
            break
    area = rng.randint(3, min(11, cap))
    side = rng.choice(["above", "below"])
    seed_row = depth - 1 if side == "above" else 0
    seed = (seed_row, rng.randrange(line_len))
    shape = _grow_connected(depth, line_len, area, seed, connectivity=4, rng=rng)

    payload = set()
    mirror = set()
    for u, v in shape:
        dr = u - depth if side == "above" else u + 1
        payload.add((dr, v, color))
        mirror.add((-dr, v, color))

    markers = {(0, c, MARK) for c in range(line_len)}
    return {"payload": payload, "mirror": mirror, "markers": markers}


def _make_col_group(color: int, rng: random.Random) -> dict[str, set[tuple[int, int, int]]]:
    while True:
        line_len = rng.randint(2, 6)
        depth = rng.randint(2, 5)
        cap = line_len * depth
        if cap >= 3:
            break
    area = rng.randint(3, min(11, cap))
    side = rng.choice(["left", "right"])
    seed_col = depth - 1 if side == "left" else 0
    seed = (rng.randrange(line_len), seed_col)
    shape = _grow_connected(line_len, depth, area, seed, connectivity=4, rng=rng)

    payload = set()
    mirror = set()
    for u, v in shape:
        dc = v - depth if side == "left" else v + 1
        payload.add((u, dc, color))
        mirror.add((u, -dc, color))

    markers = {(r, 0, MARK) for r in range(line_len)}
    return {"payload": payload, "mirror": mirror, "markers": markers}


def _make_iso_group(color: int, rng: random.Random) -> dict[str, set[tuple[int, int, int]]]:
    h = rng.randint(1, 4)
    w = rng.randint(1, 4)
    area = rng.randint(1, min(8, h * w))
    sign_r, sign_c = rng.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    shape = _grow_connected(h, w, area, (0, 0), connectivity=8, rng=rng)

    payload = set()
    mirror = set()
    for u, v in shape:
        dr = sign_r * (u + 1)
        dc = sign_c * (v + 1)
        payload.add((dr, dc, color))
        mirror.add((-dr, -dc, color))

    markers = {(0, 0, MARK)}
    return {"payload": payload, "mirror": mirror, "markers": markers}


def _translate(cells: Iterable[tuple[int, int, int]], ar: int, ac: int) -> set[tuple[int, int, int]]:
    return {(int(ar + r), int(ac + c), int(v)) for r, c, v in cells}


def generate_input(seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)

    for _attempt in range(200):
        H = rng.randint(18, 28)
        W = rng.randint(18, 28)
        bg = rng.choice([c for c in range(10) if c != MARK])
        out = np.full((H, W), bg, dtype=int)
        inp = np.full((H, W), bg, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        n_groups = rng.randint(3, 7)
        types = ["row", "col", "iso"]
        while len(types) < n_groups:
            types.append(rng.choice(["row", "col", "iso"]))
        rng.shuffle(types)

        palette = [c for c in range(10) if c not in {bg, MARK}]
        if n_groups > len(palette):
            continue
        colors = rng.sample(palette, n_groups)

        placed = True
        total_markers = 0
        for kind, color in zip(types, colors):
            if kind == "row":
                proto = _make_row_group(color, rng)
            elif kind == "col":
                proto = _make_col_group(color, rng)
            else:
                proto = _make_iso_group(color, rng)

            footprint_proto = proto["payload"] | proto["mirror"] | proto["markers"]
            rs = [r for r, _c, _v in footprint_proto]
            cs = [c for _r, c, _v in footprint_proto]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)

            row_lo = -min_r
            row_hi = H - 1 - max_r
            col_lo = -min_c
            col_hi = W - 1 - max_c
            if row_lo > row_hi or col_lo > col_hi:
                placed = False
                break

            ok_here = False
            for _ in range(200):
                ar = rng.randint(row_lo, row_hi)
                ac = rng.randint(col_lo, col_hi)
                payload = _translate(proto["payload"], ar, ac)
                mirror = _translate(proto["mirror"], ar, ac)
                markers = _translate(proto["markers"], ar, ac)
                footprint = payload | mirror | markers

                conflict = False
                for r, c, _v in footprint:
                    if not (0 <= r < H and 0 <= c < W):
                        conflict = True
                        break
                    r0 = max(0, r - 1)
                    r1 = min(H - 1, r + 1)
                    c0 = max(0, c - 1)
                    c1 = min(W - 1, c + 1)
                    if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                        conflict = True
                        break
                if conflict:
                    continue

                for r, c, v in payload:
                    out[r, c] = v
                    inp[r, c] = v
                for r, c, v in mirror:
                    out[r, c] = v
                for r, c, _v in markers:
                    inp[r, c] = MARK
                for r, c, _v in footprint:
                    occupied[r, c] = True
                total_markers += len(markers)
                ok_here = True
                break

            if not ok_here:
                placed = False
                break

        if not placed or total_markers == 0:
            continue
        if np.count_nonzero(inp == MARK) != total_markers:
            continue
        if np.count_nonzero(out == MARK) != 0:
            continue
        if np.array_equal(inp, out):
            continue
        pred = main(inp)
        if not np.array_equal(pred, out):
            continue
        return inp

    raise RuntimeError("generate_input failed to build a valid D9 variant instance")
