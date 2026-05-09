# Combo variant of 446ef5d2 (index 3)
# combo_spec: D3/2+D13/2
# Name: 多目标矩形配合多候选标记

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import paste, random_free_location


"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 原题只有 1 个 L 标记和 1 组碎片。solver 的隐式流程是：
     找唯一 marker -> 去掉 marker -> 把全部碎片当成一组 -> 找唯一矩形并装配。
   - 这个融合版固定为 2 个真实目标矩形。每个目标有自己的一组碎片，
     且每组碎片都只含一种强调色 accent，所有组共享同一个 rect_col 边框/通配色。
   - 同时，每组周围不止 1 个 L 候选，而是 2 个：1 个真 marker + 1 个假 marker。
     所有候选 marker 都贴在某块碎片的某个角上，所以局部看起来都像可能的锚点。
   - 新流程变成：
     找到全部候选 marker -> 去掉 marker 像素后提取所有矩形碎片 ->
     用碎片里的唯一 accent 先分成两组 ->
     对每组，收集“缺口落在本组哪块碎片里”的 marker 候选 ->
     逐个尝试候选，并强制把该候选贴住的碎片预放到最终矩形对应角位 ->
     只有真候选能成功，假候选会因为那块碎片并不具备该角位要求的两条外框边而失败。
   - 小例子：
     第一组是红色碎片，第二组是青色碎片。红组里有两个 L，一个贴在真正的左上角块上，
     另一个贴在一块内部块的右下角。若按第二个 L 去拼，旧逻辑可能仍会尝试装配；
     新逻辑会先把那块“被 L 指向的块”强制放到大矩形 BR 角，但它的下边/右边并非全 rect_col，
     因此立即判它是假 marker。之后只剩真候选。

b) 复杂度校准
   - 真实目标数固定为 2，候选数固定为“每组 2 个”。这样视觉上稳定地体现
     “多目标 + 候选消歧”两层变化，但搜索空间仍受控。
   - 分组线索只使用 2 种 accent 色，每块碎片都恰好只含一种 accent。
     这让分组可见，但仍需先完成分组，不能直接把全体碎片一起装配。
   - marker 颜色与碎片颜色无直接映射。候选归属来自“L 缺口贴着哪块碎片”，
     有助于保持 D3 和 D13 两层挑战相互独立。

c) 隐式假设审查
   - 原 main 假设“只有 1 个 marker”。这里不成立，必须找出全部 3 像素 L。
   - 原 main 假设“去掉 1 种 marker 颜色即可”。这里不成立，必须去掉全部 marker 像素。
   - 原 main 假设“所有碎片属于同一个总面积”。这里不成立，面积、候选尺寸、回溯装配
     都必须按组分别计算。
   - 原 main 里 anchor piece 只是可选预放置。这里不够，因为假 marker 若不强制预放，
     回溯可能绕过它照样拼出某个矩形。新 solver 把“候选缺口贴住的那块碎片”升级成强制锚点。
   - 新增假设 1：每个候选 marker 的缺口都落在某块碎片内部，generator 会显式构造并验证。
   - 新增假设 2：每块碎片只含一种 accent，generator 的分割逻辑会保证。
   - 新增假设 3：每组恰好一个候选可成功，generator 用 solver 自检这一唯一性。

d) 放置合法性
   - 先在输出坐标系中摆好两个最终矩形以及两个真 marker 的位置，并确保它们之间至少隔 1 格。
   - 输入中把每组的真锚块直接放回它的最终位置，让真 marker 明确贴住它。
   - 假 marker 则和对应组里的别的碎片一起做成 piece+marker patch，再随机散放。
   - 其余碎片也随机散放，全部对象之间保持至少 1 格背景间隔；只有 marker 与自己的锚定块允许贴边。

e) 视觉可观察性
   - 人类能直接看到：有两种碎片家族、每个家族旁边都有两个 L 候选。
   - 输出里会出现两个独立矩形，各自回到某一个候选 L 指示的位置。
   - 真/假 marker 的区别不是颜色编号，而是“它指着的那块能不能真的当该角位外框块”。

f) 可视化整洁度
   - 候选 marker 只占 3 格，且总是贴在某块碎片外侧，不覆盖碎片内部。
   - 两个最终目标先在布局层面隔开，输入阶段的散块也用 padding=1 放置，避免对象挤成一团。

g) 变化的可推断性
   - 颜色规则只有两层：rect_col 是全局共享边框/通配色；两组碎片各有一种 accent。
   - marker 颜色只表示“这是一个候选 L”，不参与隐藏排序。
   - 因此解题者能从少量样例观察到：
     先按碎片家族分两堆，再在每堆里从多个候选 L 里挑出那个能真实充当角块的候选。

h) main() 逐段审查后的实现策略
   - marker 检测：继续使用“颜色全局出现 3 次且形成 2x2 缺 1 的 L”，
     但必须收集全部候选，而不是拿到第一个就结束。generator 会避免其他颜色也恰好出现 3 次。
   - 碎片提取：从非背景且非 marker-mask 的区域里提取矩形块。
   - 分组：每块碎片去掉 rect_col 后应只剩 1 个 accent；按该 accent 分组。
   - 候选归属：看 marker 缺口落在哪块碎片里，再用那块碎片的 accent 把 marker 归到对应组。
   - 组内求解：只允许使用本组碎片；并把候选贴住的那块强制放到该候选声明的矩形角上。
   - 唯一性：每组必须恰好 1 个候选成功。generator 依赖这一点，并在生成时用 solver 复验。
"""


CORNERS = ("TL", "TR", "BL", "BR")


def _corner_from_missing(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _top_left_from_marker(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(mr), int(mc)
    if corner == "TR":
        return int(mr), int(mc - W + 1)
    if corner == "BL":
        return int(mr - H + 1), int(mc)
    return int(mr - H + 1), int(mc - W + 1)


def _missing_from_top_left(tr: int, tc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return int(tr), int(tc)
    if corner == "TR":
        return int(tr), int(tc + W - 1)
    if corner == "BL":
        return int(tr + H - 1), int(tc)
    return int(tr + H - 1), int(tc + W - 1)


def _marker_cells_from_missing(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _make_piece_marker_patch(piece: np.ndarray, corner: str, marker_col: int, bg: int) -> np.ndarray:
    ph, pw = piece.shape
    marker_coords = {
        "TL": [(-1, -1), (-1, 0), (0, -1)],
        "TR": [(-1, pw - 1), (-1, pw), (0, pw)],
        "BL": [(ph - 1, -1), (ph, -1), (ph, 0)],
        "BR": [(ph - 1, pw), (ph, pw - 1), (ph, pw)],
    }[corner]
    coords = [(0, 0), (ph - 1, pw - 1)] + marker_coords
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    pr0 = -min_r
    pc0 = -min_c
    patch[pr0 : pr0 + ph, pc0 : pc0 + pw] = piece
    for rr, cc in marker_coords:
        patch[rr - min_r, cc - min_c] = int(marker_col)
    return patch


def _find_candidate_markers(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    cnt = Counter(map(int, g.flatten()))
    markers: list[dict] = []
    mask = np.zeros_like(g, dtype=bool)
    for col in sorted(int(c) for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = tuple(map(int, miss[0]))
        corner = _corner_from_missing(missing, (int(r0), int(c0), int(r1), int(c1)))
        markers.append(
            {
                "color": int(col),
                "cells": pts,
                "missing": missing,
                "corner": corner,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
        for r, c in pts:
            mask[int(r), int(c)] = True
    return markers, mask


def _extract_rectangles(g: np.ndarray, bg: int, marker_mask: np.ndarray) -> list[dict]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (~marker_mask)
    used = np.zeros_like(occ, dtype=bool)
    rects: list[dict] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append(
            {
                "r": int(rr),
                "c": int(cc),
                "h": int(h),
                "w": int(w),
                "arr": g[rr : rr + h, cc : cc + w].copy(),
            }
        )
    return rects


def _piece_accent(piece: np.ndarray, rect_col: int) -> int:
    cols = {int(v) for v in piece.flatten()} - {int(rect_col)}
    assert len(cols) == 1
    return int(next(iter(cols)))


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    vv1 = np.array(v1, dtype=int)
    vv2 = np.array(v2, dtype=int)
    m = (vv1 != int(rect_col)) & (vv2 != int(rect_col))
    return bool((vv1[m] == vv2[m]).all())


def _assemble_for_candidate(
    group_rects: list[dict],
    rect_col: int,
    marker: dict,
    grid_shape: tuple[int, int],
) -> dict | None:
    GH, GW = map(int, grid_shape)
    mr, mc = map(int, marker["missing"])
    corner = str(marker["corner"])

    anchor_idx = None
    for i, rect in enumerate(group_rects):
        rr = int(rect["r"])
        cc = int(rect["c"])
        hh = int(rect["h"])
        ww = int(rect["w"])
        if rr <= mr < rr + hh and cc <= mc < cc + ww:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    rect = group_rects[anchor_idx]
    rr = int(rect["r"])
    cc = int(rect["c"])
    hh = int(rect["h"])
    ww = int(rect["w"])
    corners = {
        (rr, cc): "TL",
        (rr, cc + ww - 1): "TR",
        (rr + hh - 1, cc): "BL",
        (rr + hh - 1, cc + ww - 1): "BR",
    }
    if corners.get((mr, mc)) != corner:
        return None

    pieces = [rect["arr"] for rect in group_rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]
    total_area = int(sum(int(p.shape[0]) * int(p.shape[1]) for p in pieces))

    dims: list[tuple[int, int]] = []
    for hh2 in range(1, int(total_area**0.5) + 1):
        if total_area % hh2 == 0:
            dims.append((hh2, total_area // hh2))
            if hh2 * hh2 != total_area:
                dims.append((total_area // hh2, hh2))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_marker(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def assemble(H: int, W: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if (r + h) == H and not np.all(a[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if (c + w) == W and not np.all(a[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False
            if not boundary_ok(a, r, c):
                return False
            if not neighbor_ok(a, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

        ah = ph[anchor_idx]
        aw = pw[anchor_idx]
        if corner == "TL":
            ar0, ac0 = 0, 0
        elif corner == "TR":
            ar0, ac0 = 0, W - aw
        elif corner == "BL":
            ar0, ac0 = H - ah, 0
        else:
            ar0, ac0 = H - ah, W - aw
        if not can_place(anchor_idx, ar0, ac0):
            return None
        place(anchor_idx, ar0, ac0)

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

        solved = False

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                solved = True
                return
            r, c = nxt
            for i in order:
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, r, c)

        bt()
        return occ2 if solved else None

    for H, W in dims:
        ass = assemble(int(H), int(W))
        if ass is None:
            continue
        tr, tc = _top_left_from_marker(mr, mc, int(H), int(W), corner)
        return {
            "marker": marker,
            "assembly": ass,
            "dims": (int(H), int(W)),
            "top_left": (int(tr), int(tc)),
        }
    return None


def _solve_groups(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    markers, marker_mask = _find_candidate_markers(g, bg)
    assert len(markers) >= 4

    rects = _extract_rectangles(g, bg, marker_mask)
    assert rects

    allpix: list[int] = []
    for rect in rects:
        allpix.extend(map(int, rect["arr"].flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    groups: dict[int, list[dict]] = defaultdict(list)
    for rect in rects:
        accent = _piece_accent(rect["arr"], rect_col)
        rect["accent"] = int(accent)
        groups[int(accent)].append(rect)

    marker_groups: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        mr, mc = map(int, marker["missing"])
        owner = None
        for rect in rects:
            rr = int(rect["r"])
            cc = int(rect["c"])
            hh = int(rect["h"])
            ww = int(rect["w"])
            if rr <= mr < rr + hh and cc <= mc < cc + ww:
                owner = rect
                break
        assert owner is not None
        marker["accent"] = int(owner["accent"])
        marker_groups[int(owner["accent"])].append(marker)

    assert set(groups.keys()) == set(marker_groups.keys())

    out = np.full((GH, GW), int(bg), dtype=int)
    filled = np.zeros((GH, GW), dtype=bool)
    group_results = []
    for accent in sorted(groups):
        successes = []
        for marker in marker_groups[int(accent)]:
            sol = _assemble_for_candidate(groups[int(accent)], rect_col, marker, (GH, GW))
            if sol is not None:
                successes.append(sol)
        assert len(successes) == 1
        sol = successes[0]
        tr, tc = map(int, sol["top_left"])
        H, W = map(int, sol["dims"])
        assert not filled[tr : tr + H, tc : tc + W].any()
        out[tr : tr + H, tc : tc + W] = sol["assembly"]
        filled[tr : tr + H, tc : tc + W] = True
        group_results.append(
            {
                "accent": int(accent),
                "candidate_count": len(marker_groups[int(accent)]),
                "success_count": len(successes),
                "valid_marker_color": int(sol["marker"]["color"]),
                "top_left": (int(tr), int(tc)),
                "dims": (int(H), int(W)),
            }
        )

    return {
        "bg": int(bg),
        "rect_col": int(rect_col),
        "markers": markers,
        "groups": groups,
        "marker_groups": marker_groups,
        "group_results": group_results,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_groups(grid)["output"]


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 5) -> list[int] | None:
    segs: list[int] = []
    rem = int(total)
    eff_max = min(int(max_s), max(int(min_s), rng.randint(min_s, max_s)))
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(int(rem))
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = rng.randint(min_s, hi)
            segs.append(int(s))
            rem -= int(s)
    return segs


def _build_target_rect(RH: int, RW: int, rect_col: int, accent: int, rng: random.Random) -> np.ndarray | None:
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    for _ in range(120):
        interior = np.full((iH, iW), int(accent), dtype=int)
        hole_count = rng.randint(1, 4)
        for _hole in range(hole_count):
            hh = rng.randint(1, max(1, iH // 2))
            ww = rng.randint(1, max(1, iW // 2))
            rr = rng.randint(0, iH - hh)
            cc = rng.randint(0, iW - ww)
            interior[rr : rr + hh, cc : cc + ww] = int(rect_col)
        if rng.random() < 0.7 and iH >= 2:
            rr = rng.randint(0, iH - 1)
            span = rng.randint(max(1, iW // 3), iW)
            cc = rng.randint(0, iW - span)
            interior[rr, cc : cc + span] = int(rect_col)
        if rng.random() < 0.7 and iW >= 2:
            cc = rng.randint(0, iW - 1)
            span = rng.randint(max(1, iH // 3), iH)
            rr = rng.randint(0, iH - span)
            interior[rr : rr + span, cc] = int(rect_col)
        fill = float(np.mean(interior == int(accent)))
        if not (0.35 <= fill <= 0.85):
            continue
        if not np.any(interior == int(rect_col)):
            continue
        rect = np.full((RH, RW), int(rect_col), dtype=int)
        rect[1:-1, 1:-1] = interior
        return rect
    return None


def _piece_supports_corner(piece_info: tuple[int, int, int, int], corner: str, RH: int, RW: int) -> bool:
    pr, pc, ph, pw = piece_info
    if corner == "TL":
        return pr == 0 and pc == 0
    if corner == "TR":
        return pr == 0 and (pc + pw) == RW
    if corner == "BL":
        return (pr + ph) == RH and pc == 0
    return (pr + ph) == RH and (pc + pw) == RW


def _partition_rect(rect: np.ndarray, rect_col: int, accent: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    RH, RW = rect.shape
    for _ in range(220):
        row_segs = _rand_splits(RH, rng, min_s=2, max_s=min(5, RH))
        col_segs = _rand_splits(RW, rng, min_s=2, max_s=min(5, RW))
        if row_segs is None or col_segs is None:
            continue

        row_starts: list[int] = []
        acc = 0
        for rs in row_segs:
            row_starts.append(acc)
            acc += int(rs)
        col_starts: list[int] = []
        acc = 0
        for cs in col_segs:
            col_starts.append(acc)
            acc += int(cs)

        nr = len(row_segs)
        nc = len(col_segs)
        cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
        n_cells = nr * nc
        if n_cells < 4:
            continue
        target_pieces = rng.randint(4, min(6, n_cells))

        def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
            cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
            if not cells:
                return None
            ris = [c[0] for c in cells]
            cis = [c[1] for c in cells]
            return min(ris), min(cis), max(ris), max(cis)

        def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
            h = sum(int(row_segs[i]) for i in range(r0i, r1i + 1))
            w = sum(int(col_segs[j]) for j in range(c0i, c1i + 1))
            return int(h), int(w)

        merges: list[tuple[str, int, int]] = []
        for i in range(nr):
            for j in range(nc):
                if j + 1 < nc:
                    merges.append(("h", i, j))
                if i + 1 < nr:
                    merges.append(("v", i, j))
        rng.shuffle(merges)

        def count_pieces() -> int:
            return len({cell_id[i][j] for i in range(nr) for j in range(nc)})

        for direction, mi, mj in merges:
            if count_pieces() <= target_pieces:
                break
            if direction == "h":
                pid1 = cell_id[mi][mj]
                pid2 = cell_id[mi][mj + 1]
            else:
                pid1 = cell_id[mi][mj]
                pid2 = cell_id[mi + 1][mj]
            if pid1 == pid2:
                continue
            bb1 = get_piece_bbox(pid1)
            bb2 = get_piece_bbox(pid2)
            if bb1 is None or bb2 is None:
                continue
            mr0 = min(bb1[0], bb2[0])
            mc0 = min(bb1[1], bb2[1])
            mr1 = max(bb1[2], bb2[2])
            mc1 = max(bb1[3], bb2[3])
            ph, pw = pixel_size(mr0, mc0, mr1, mc1)
            if ph > 7 or pw > 7:
                continue
            ok = True
            for ii in range(mr0, mr1 + 1):
                for jj in range(mc0, mc1 + 1):
                    if cell_id[ii][jj] not in (pid1, pid2):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for ii in range(nr):
                for jj in range(nc):
                    if cell_id[ii][jj] == pid2:
                        cell_id[ii][jj] = pid1

        pieces: list[tuple[int, int, int, int]] = []
        valid = True
        seen: set[int] = set()
        for i in range(nr):
            for j in range(nc):
                pid = cell_id[i][j]
                if pid in seen:
                    continue
                seen.add(pid)
                bb = get_piece_bbox(pid)
                assert bb is not None
                r0i, c0i, r1i, c1i = bb
                pr = row_starts[r0i]
                pc = col_starts[c0i]
                ph = sum(int(row_segs[ii]) for ii in range(r0i, r1i + 1))
                pw = sum(int(col_segs[jj]) for jj in range(c0i, c1i + 1))
                piece = rect[pr : pr + ph, pc : pc + pw]

                if ph < 2 or pw < 2 or ph > 7 or pw > 7:
                    valid = False
                    break
                cols = {int(v) for v in piece.flatten()}
                if cols - {int(rect_col), int(accent)}:
                    valid = False
                    break
                n_accent = int(np.sum(piece == int(accent)))
                if n_accent == 0:
                    valid = False
                    break
                fill = n_accent / float(ph * pw)
                if fill < 0.10 or fill > 0.85:
                    valid = False
                    break
                if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
                    valid = False
                    break
                if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
                    valid = False
                    break
                if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
                    valid = False
                    break
                if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
                    valid = False
                    break
                if not any(np.any(edge == int(accent)) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
                    valid = False
                    break
                pieces.append((int(pr), int(pc), int(ph), int(pw)))
            if not valid:
                break

        if not valid or not (4 <= len(pieces) <= 6):
            continue

        supported = {corner: sum(1 for piece in pieces if _piece_supports_corner(piece, corner, RH, RW)) for corner in CORNERS}
        if min(supported.values()) < 1:
            continue
        return pieces
    return None


def _clear_with_padding(mask: np.ndarray, occ: np.ndarray, pad: int = 1) -> bool:
    H, W = occ.shape
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - pad)
        r1 = min(H, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(W, int(c) + pad + 1)
        if occ[r0:r1, c0:c1].any():
            return False
    return True


def _pick_decoy(pieces: list[tuple[int, int, int, int]], true_idx: int, RH: int, RW: int, rng: random.Random) -> tuple[int, str] | None:
    options = []
    for idx, piece in enumerate(pieces):
        if idx == true_idx:
            continue
        for corner in CORNERS:
            if not _piece_supports_corner(piece, corner, RH, RW):
                options.append((idx, corner))
    if not options:
        return None
    return rng.choice(options)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(20, 26)
    GW = rng.randint(20, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    accents = [int(palette[2]), int(palette[3])]
    marker_cols = [int(v) for v in palette[4:8]]

    targets: list[dict] = []
    for gi, accent in enumerate(accents):
        RH = rng.randint(5, min(9, GH - 4))
        RW = rng.randint(5, min(9, GW - 4))
        corner = rng.choice(list(CORNERS))
        rect = _build_target_rect(RH, RW, rect_col, accent, rng)
        if rect is None:
            return None
        pieces = _partition_rect(rect, rect_col, accent, rng)
        if pieces is None:
            return None

        anchor_cell = {
            "TL": (0, 0),
            "TR": (0, RW - 1),
            "BL": (RH - 1, 0),
            "BR": (RH - 1, RW - 1),
        }[corner]
        true_anchor_idx = None
        for i, (pr, pc, ph, pw) in enumerate(pieces):
            if pr <= anchor_cell[0] < pr + ph and pc <= anchor_cell[1] < pc + pw:
                true_anchor_idx = i
                break
        if true_anchor_idx is None:
            return None
        if not _piece_supports_corner(pieces[true_anchor_idx], corner, RH, RW):
            return None

        decoy = _pick_decoy(pieces, true_anchor_idx, RH, RW, rng)
        if decoy is None:
            return None

        piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
        targets.append(
            {
                "accent": int(accent),
                "corner": corner,
                "RH": int(RH),
                "RW": int(RW),
                "rect": rect,
                "pieces": pieces,
                "piece_arrays": piece_arrays,
                "true_anchor_idx": int(true_anchor_idx),
                "decoy_idx": int(decoy[0]),
                "decoy_corner": str(decoy[1]),
                "true_marker_col": int(marker_cols[2 * gi]),
                "decoy_marker_col": int(marker_cols[2 * gi + 1]),
            }
        )

    occ = np.zeros((GH, GW), dtype=bool)
    for target in sorted(targets, key=lambda t: -(int(t["RH"]) * int(t["RW"]))):
        placed = False
        RH = int(target["RH"])
        RW = int(target["RW"])
        corner = str(target["corner"])
        for _ in range(400):
            tr = rng.randint(0, GH - RH)
            tc = rng.randint(0, GW - RW)
            mr, mc = _missing_from_top_left(tr, tc, RH, RW, corner)
            marker_cells = _marker_cells_from_missing(mr, mc, corner)
            if not all(0 <= r < GH and 0 <= c < GW for r, c in marker_cells):
                continue
            footprint = np.zeros((GH, GW), dtype=bool)
            footprint[tr : tr + RH, tc : tc + RW] = True
            for r, c in marker_cells:
                footprint[int(r), int(c)] = True
            if not _clear_with_padding(footprint, occ, pad=1):
                continue
            occ |= footprint
            target["top_left"] = (int(tr), int(tc))
            target["missing"] = (int(mr), int(mc))
            target["true_marker_cells"] = marker_cells
            placed = True
            break
        if not placed:
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    expected_out = np.full((GH, GW), int(bg), dtype=int)

    placed_piece_indices: dict[int, set[int]] = defaultdict(set)
    for ti, target in enumerate(targets):
        tr, tc = map(int, target["top_left"])
        RH = int(target["RH"])
        RW = int(target["RW"])
        expected_out[tr : tr + RH, tc : tc + RW] = target["rect"]
        for r, c in target["true_marker_cells"]:
            canvas[int(r), int(c)] = int(target["true_marker_col"])

        apr, apc, aph, apw = target["pieces"][int(target["true_anchor_idx"])]
        anchor_abs_r = tr + int(apr)
        anchor_abs_c = tc + int(apc)
        paste(canvas, target["piece_arrays"][int(target["true_anchor_idx"])], anchor_abs_r, anchor_abs_c)
        placed_piece_indices[ti].add(int(target["true_anchor_idx"]))

    for ti, target in enumerate(targets):
        idx = int(target["decoy_idx"])
        decoy_corner = str(target["decoy_corner"])
        patch = _make_piece_marker_patch(
            target["piece_arrays"][idx],
            decoy_corner,
            int(target["decoy_marker_col"]),
            int(bg),
        )
        try:
            r0, c0 = random_free_location(canvas, patch, transparent=bg, padding=1, border=1)
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=bg)
        placed_piece_indices[ti].add(idx)

    scatter_pool: list[np.ndarray] = []
    for ti, target in enumerate(targets):
        for idx, piece in enumerate(target["piece_arrays"]):
            if idx in placed_piece_indices[ti]:
                continue
            scatter_pool.append(piece.copy())
    rng.shuffle(scatter_pool)

    for piece in scatter_pool:
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1, border=1)
        except ValueError:
            return None
        paste(canvas, piece, r0, c0)

    counts = Counter(map(int, canvas.flatten()))
    for col in [bg, rect_col] + accents:
        if counts[int(col)] == 3:
            return None

    piece_pixels: list[int] = []
    for target in targets:
        for piece in target["piece_arrays"]:
            piece_pixels.extend(map(int, piece.flatten()))
    if int(Counter(piece_pixels).most_common(1)[0][0]) != int(rect_col):
        return None

    try:
        solved = _solve_groups(canvas)
    except Exception:
        return None

    if not np.array_equal(solved["output"], expected_out):
        return None
    if len(solved["group_results"]) != 2:
        return None
    if any(int(gr["candidate_count"]) != 2 or int(gr["success_count"]) != 1 for gr in solved["group_results"]):
        return None
    if len(_find_candidate_markers(canvas, bg)[0]) != 4:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
