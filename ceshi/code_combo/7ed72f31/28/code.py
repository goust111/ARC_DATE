# Combo variant of 7ed72f31 (index 28)
# combo_spec: D9/1+D12/1
# Name: 标记消耗与同色干扰
#
# Design analysis
# ---------------
# spec:
#   fusion_rule:
#     - 仍沿用原题的三类真标记：
#       1) 与 payload 4 连通的水平红线，表示按该行镜像；
#       2) 与 payload 4 连通的垂直红线，表示按该列镜像；
#       3) 与 payload 4 连通的单个红点，或与恰好一个 8 连通彩色精灵对角接触的孤立红点，
#          表示按该点做中心对称。
#     - 真标记在完成反射后会被擦除。
#     - 额外加入若干“假红色”装饰：它们和真标记颜色相同，但不满足原题的几何/邻接条件，
#       所以不会触发反射，也不会被擦除，输出中仍保留红色。
#   concrete_trace:
#     - 原题/纯 D9: 某条贴着绿色 payload 的红横线会在输出里消失，但若“把所有红都删掉”，
#       连远处纯装饰的红线也会一起被误删。
#     - 融合后: 只有真正参与反射的那条红横线消失；不接触 payload 的红装饰线/红点仍留在输出。
#     - 差异不在反射几何，而在“哪些红色被解释为指令像素”。
#   complexity_calibration:
#     - 每个样例仍至少混合 row / col / iso 三类真标记，避免退化成单一反射模式。
#     - 每个样例至少放 2 个假红装饰，且既可能是单点，也可能是短线/L 形，保证“同色干扰”
#       在肉眼上明显，而不是偶然一个远处红点。
#     - 主 payload 颜色保持各组唯一，避免 solver 需要全局拆分同色 payload。
#   implicit_assumption_audit_main:
#     - 原代码假设 `out = g.copy()` 后所有红都应保留；D9 已打破这一点。
#     - 单独做 D9 时可直接 `out[g==2]=bg`，但融合后这个假设又被 D12 打破：不是所有红都是真标记。
#     - 因此新 solver 先识别“实际触发了反射的红像素集合 consumed_markers”，最后只擦除这一集合。
#     - 对于 line / embedded-point，`label_mask(g != bg, 4)` 的前提仍成立：真标记与其 payload
#       在同一 4 连通分量内；假红装饰若不连到 payload，就不会误入该分支。
#     - 对于 isolated-point，仍要求红点没有 4 邻非背景像素，且仅与一个 8 连通彩色分量相邻；
#       生成器显式保证真孤立标记满足此条件，假红单点则故意不满足“恰有一个相邻精灵”。
#   placement_legality:
#     - 继续采用 output-first：先放完整输出中的 payload / mirror / 假红，再反推输入时把 mirror 去掉、
#       把真标记加回去。
#     - 所有真组与假红装饰都做整块 footprint 检查：不能越界，不能重叠，并与既有对象保留 1 格 8 邻域留白。
#     - 假红装饰额外要求和任一真标记保持 2-5 格切比雪夫距离，做到“看起来在同一片区域里干扰”，
#       但又不会因邻接而被 solver 当成真标记。
#   observability:
#     - 训练样例中会稳定出现两种红色行为：靠着彩色精灵的红会消失；游离/装饰性的红会保留。
#     - 线型真标记消失后留下背景缝，点型真标记消失后留下空心中心；而假红仍是红色实物。
#     - 解题者可以通过 input/output 对比直接归纳“只有承担轴/中心角色的红才会被消耗”。
#   visual_cleanliness:
#     - 假红不与 payload 贴边，不与真标记混成同一分量，避免局部一团红无法分辨。
#     - 每个主组只用一种非红颜色，组间留白，输出中剩余的红装饰也不会遮住镜像结果。
#   new_main_assumptions_after_edit:
#     - 假设真线标记的 payload 只在红线单侧；生成器保证这一点。
#     - 假设孤立点标记只接触一个 8 连通彩色分量；生成器保证这一点。
#     - 假设假红装饰不与非红 payload 4/8 邻接到会触发真标记判定的位置；生成器保证这一点。

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask

MARK = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror payloads around true red markers, erase only those consumed markers,
    and leave decorative red pixels untouched.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()
    consumed_markers: set[tuple[int, int]] = set()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    def color_neighbors4(r: int, c: int) -> list[tuple[int, int, int]]:
        nbrs: list[tuple[int, int, int]] = []
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not inb(rr, cc):
                continue
            v = int(g[rr, cc])
            if v == int(bg) or v == int(MARK):
                continue
            nbrs.append((int(rr), int(cc), int(v)))
        return nbrs

    # Pass 1: marker + payload are in one 4-connected non-bg component.
    lab_mix, n_mix = label_mask(g != int(bg), connectivity=4)
    for k in range(1, int(n_mix) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab_mix == int(k))]
        if not coords:
            continue

        reds = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(MARK)]
        payload = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) not in {int(bg), int(MARK)}]
        if not reds or not payload:
            continue

        rs = {int(r) for r, _c in reds}
        cs = {int(c) for _r, c in reds}
        axis: tuple[str, int | tuple[int, int]] | None = None
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            if any(int(r) == int(next(iter(rs))) for r, _c, _v in payload):
                continue
            sides = {int(r) < int(next(iter(rs))) for r, _c, _v in payload}
            if len(sides) != 1:
                continue
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            if any(int(c) == int(next(iter(cs))) for _r, c, _v in payload):
                continue
            sides = {int(c) < int(next(iter(cs))) for _r, c, _v in payload}
            if len(sides) != 1:
                continue
            axis = ("col", int(next(iter(cs))))
        if axis is None:
            continue

        for r, c, v in payload:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)
        consumed_markers.update(reds)

    # Pass 2: isolated single red point with one adjacent 8-connected color component.
    lab_red, n_red = label_mask(g == int(MARK), connectivity=4)
    for k in range(1, int(n_red) + 1):
        pts = [(int(r), int(c)) for r, c in np.argwhere(lab_red == int(k))]
        if len(pts) != 1:
            continue
        pr, pc = pts[0]
        if (int(pr), int(pc)) in consumed_markers:
            continue
        if color_neighbors4(int(pr), int(pc)):
            continue

        neigh_pixels: list[tuple[int, int, int]] = []
        neigh_colors: set[int] = set()
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
                neigh_pixels.append((int(rr), int(cc), int(v)))
                neigh_colors.add(int(v))
        if not neigh_pixels:
            continue

        touched_components: list[tuple[int, int, np.ndarray]] = []
        for v in sorted(neigh_colors):
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labels = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in sorted(labels):
                touched_components.append((int(v), int(lbl), labv))

        if len(touched_components) != 1:
            continue

        color, lbl, labv = touched_components[0]
        for r, c in np.argwhere(labv == int(lbl)):
            rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
            if inb(int(rr), int(cc)):
                out[int(rr), int(cc)] = int(color)
        consumed_markers.add((int(pr), int(pc)))

    for r, c in consumed_markers:
        out[int(r), int(c)] = int(bg)
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


def _make_fake_red_decoration(rng: random.Random) -> set[tuple[int, int, int]]:
    kind = rng.choice(["dot", "hline", "vline", "lshape"])
    if kind == "dot":
        return {(0, 0, MARK)}
    if kind == "hline":
        length = rng.randint(2, 4)
        return {(0, c, MARK) for c in range(length)}
    if kind == "vline":
        length = rng.randint(2, 4)
        return {(r, 0, MARK) for r in range(length)}
    arm_r = rng.randint(2, 3)
    arm_c = rng.randint(2, 3)
    cells = {(0, 0, MARK)}
    cells.update((r, 0, MARK) for r in range(1, arm_r))
    cells.update((0, c, MARK) for c in range(1, arm_c))
    return cells


def _translate(cells: Iterable[tuple[int, int, int]], ar: int, ac: int) -> set[tuple[int, int, int]]:
    return {(int(ar + r), int(ac + c), int(v)) for r, c, v in cells}


def _can_place(
    cells: set[tuple[int, int, int]],
    occupied: np.ndarray,
    H: int,
    W: int,
    *,
    gap: int = 1,
) -> bool:
    for r, c, _v in cells:
        if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
            return False
        r0 = max(0, int(r) - int(gap))
        r1 = min(int(H) - 1, int(r) + int(gap))
        c0 = max(0, int(c) - int(gap))
        c1 = min(int(W) - 1, int(c) + int(gap))
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
    return True


def generate_input(seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)

    for _attempt in range(300):
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

        true_marker_cells: list[tuple[int, int]] = []
        placed = True
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
                if not _can_place(footprint, occupied, H, W, gap=1):
                    continue

                for r, c, v in payload:
                    out[r, c] = v
                    inp[r, c] = v
                for r, c, v in mirror:
                    out[r, c] = v
                for r, c, _v in markers:
                    inp[r, c] = MARK
                    true_marker_cells.append((int(r), int(c)))
                for r, c, _v in footprint:
                    occupied[r, c] = True
                ok_here = True
                break

            if not ok_here:
                placed = False
                break

        if not placed or not true_marker_cells:
            continue

        # Add decorative red shapes that survive in the output.
        n_fake = rng.randint(2, 5)
        fake_cells_total = 0
        for _ in range(n_fake):
            proto = _make_fake_red_decoration(rng)
            rs = [r for r, _c, _v in proto]
            cs = [c for _r, c, _v in proto]
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
            for _ in range(300):
                ar = rng.randint(row_lo, row_hi)
                ac = rng.randint(col_lo, col_hi)
                deco = _translate(proto, ar, ac)
                if not _can_place(deco, occupied, H, W, gap=1):
                    continue

                # Keep decorations visually near the real action but not adjacent to it.
                dmin = min(
                    max(abs(int(r) - int(tr)), abs(int(c) - int(tc)))
                    for r, c, _v in deco
                    for tr, tc in true_marker_cells
                )
                if dmin < 2 or dmin > 5:
                    continue

                for r, c, v in deco:
                    inp[r, c] = v
                    out[r, c] = v
                    occupied[r, c] = True
                    fake_cells_total += 1
                ok_here = True
                break

            if not ok_here:
                placed = False
                break

        if not placed or fake_cells_total == 0:
            continue

        # Fusion sanity: output must keep some red, but fewer red pixels than input.
        inp_red = int(np.count_nonzero(inp == MARK))
        out_red = int(np.count_nonzero(out == MARK))
        if out_red <= 0:
            continue
        if inp_red <= out_red:
            continue
        if np.array_equal(inp, out):
            continue

        pred = main(inp)
        if not np.array_equal(pred, out):
            continue
        return inp

    raise RuntimeError("generate_input failed to build a valid combo instance")
