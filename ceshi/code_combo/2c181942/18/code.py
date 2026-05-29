# Combo variant of 2c181942 (index 18)
# combo_spec: D11/2+D14/1
# Name: 覆盖放置+链式贴附

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  fusion_summary: |
    本融合把原题的“按颜色匹配到 hub 某一边”保留下来，但把几何关系从 flush 外贴改成
    cover 覆盖，同时把“每色至多一个 sprite”的独立贴附改成“一条同色链按顺序向外生长”。
    也就是说：
      1) 第一块不是贴在 hub 外一格，而是让 sprite 的接触边直接压在 hub 彩边上。
      2) 第二块以后不再看 hub，而是看前一块暴露在外侧的同色边；它也不是隔一格贴，
         而是直接覆盖那条边。
    于是输出中会出现一条同色链：hub 边 -> 第 1 块 -> 第 2 块，且每次接触都共享同一行/列。

  a_concretization_and_trace: |
    设 hub 顶边颜色为红，N=5，因此 target=N-2=3。
    原题顶边单块外贴的逻辑是：
      - 找到红色 sprite。
      - 旋转到“底边正好有 3 个连续格”。
      - 把这个底边放到 row=r-1，正好紧贴 hub 顶边。
    融合后变成：
      - 第 1 块仍然要求“底边有 3 个连续格”，但它的底边放到 row=r，直接覆盖 hub 顶边。
      - 读取它的暴露顶边长度，例如 2。
      - 在同色剩余分量中找一块旋转后“底边长度也为 2”的 sprite。
      - 这第 2 块的底边不放在 row=(第 1 块顶边)-1，而是直接放在第 1 块的顶边那一行，
        因而继续覆盖上一级的边。
    差异不是“旋转规则变了”，而是锚点几何从“相邻”变成“重合”，并且锚点在步骤 2 以后
    从 hub 边换成前一块的外侧边。

  b_complexity_calibration: |
    为了让融合效果可见而不混乱，生成器固定只有一个链式颜色，链长固定为 2 块：
      - 至少 2 个同色分量，让链式行为一定出现。
      - 可选再加 0-1 个其它颜色的单块覆盖贴附，保留原题“多边匹配”的味道，但避免四边同时
        链化导致画面过密。
    链的长度信息只使用少量可见档位 1/2/3/4；这足以编码顺序，也不会让归纳空间过大。
    如果允许很多同色块、很多方向同时链化，视觉上会太乱，且容易出现多解。

  c_implicit_assumptions_review: |
    原 solver 中与本融合直接冲突的隐式前提有三类：
      - “每个 hub 颜色只取最大连通块”：不成立。链色现在有 2 个相关分量，必须全部提取。
      - “每个 sprite 只对 hub 固定边做一次判断”：不成立。第 2 块以后要对前一块的暴露边判断。
      - “贴附时目标行/列在 hub 外一格”：不成立。cover 规则要求目标行/列就是锚点边本身。
    因此新 main() 改成：
      - 对每个 hub 颜色提取全部 8 连通分量。
      - 用 while 循环逐级消耗同色分量。
      - 旋转合法性由 (incoming edge length, outgoing edge length) 决定。
      - next_fixed 不再是 exposed_edge±1，而是 exposed_edge 本身。
    其它原前提仍保留：
      - 输入里 hub 仍是空心，_find_hub 仍成立。
      - 接触边颜色相同，故允许“已有像素和将贴像素同色时重合”。

  d_placement_legality: |
    生成器仍采用 output-first：
      - 先决定 hub 颜色、链边、链上两块的 in/out edge 长度。
      - 先在输出空间按 cover+chain 规则贴好，逐块检查越界与冲突。
      - 再把这些块拆下来、随机旋转、散落到输入中。
    同色链块在输入散布时使用 random_free_location(..., padding=2, connectivity=8)，
    保证它们仍是互相分离的 8 连通分量。
    额外单块只允许出现在链边的相邻边，不放到正对边，避免在 26x26 上挤不下。

  e_visual_observability: |
    人类从 2-4 组样例中可以直接看到两个稳定线索：
      - 某个 hub 边颜色在输入里出现 2 个分离的同色块，而输出里它们连成一条单色外伸链。
      - 这条链每一级都“压在”前一级边上，而不是像原题那样留出一格缝隙。
    这两个线索都来自可见的空间关系，不依赖颜色编号大小、组件扫描顺序等不可视因素。

  f_visual_cleanliness: |
    只做一条直线链，不转弯，不分叉；其它边至多再来一块单独覆盖贴附。
    输出因此仍是“一个 hub + 一条明显的外伸链 + 少量单块”，不会变成一团同色噪声。

  g_color_inference: |
    颜色规则没有重新映射：hub 四边仍各自用不同非背景色；sprite 颜色与 hub 边颜色一致即配对。
    顺序由边长递推，而不是由颜色值决定，所以解题者能从几组样例中归纳出“同色链”的规则。

  h_variant_metric_for_post_test: |
    量化指标使用两类：
      - input_chain_components: 每个样本中，按 hub 四边颜色统计输入外部同色分量个数的最大值。
        融合样本应恒为 2，证明链式控制变量真实出现。
      - delta_vs_d14 and delta_vs_original: 同一输入上，融合 solver 输出分别与 D14 单变体、
        原始 solver 的像素差异数。若长期为 0，说明样本退化；正常情况应显著大于 0。

  i_new_solver_assumptions_to_guarantee: |
    新 solver 额外依赖以下输入结构，而生成器会显式保证：
      - 恰有一个 hub 边颜色对应 2 个真实分量，其余真实颜色至多 1 个。
      - 对链上每一级，在剩余同色分量中至多有一个候选能匹配当前 required_len。
      - 每个真实 sprite 在其预期 side 上只有一个 rot90 朝向满足 (incoming_len, outgoing_len)。
      - 同色分量在输入中彼此分离，因此 label_mask(..., connectivity=8) 能稳定分块。
"""


OPPOSITE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}
SIDE_TO_NEED_EDGE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}
ADJACENT_SIDES = {
    "top": ["left", "right"],
    "bot": ["left", "right"],
    "left": ["top", "bot"],
    "right": ["top", "bot"],
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _segment_len(mask: np.ndarray, side: str) -> int | None:
    seg = _edge_segment(mask, side)
    if seg is None:
        return None
    if side in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) if _is_contig(rows) else None
    _typ, _edge_r, cols = seg
    return len(cols) if _is_contig(cols) else None


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None  # (score, (r,c,n), (top,bot,left,right))
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + n - 1]) != int(bg)
                    or int(g[r + n - 1, c]) != int(bg)
                    or int(g[r + n - 1, c + n - 1]) != int(bg)
                ):
                    continue

                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (
                    score,
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def _side_specs(r: int, c: int, n: int, colors: tuple[int, int, int, int]):
    top_col, bot_col, left_col, right_col = colors
    return {
        "top": {
            "color": int(top_col),
            "need_edge": "bot",
            "fixed": int(r),
            "span": list(range(int(c + 1), int(c + n - 1))),
        },
        "bot": {
            "color": int(bot_col),
            "need_edge": "top",
            "fixed": int(r + n - 1),
            "span": list(range(int(c + 1), int(c + n - 1))),
        },
        "left": {
            "color": int(left_col),
            "need_edge": "right",
            "fixed": int(c),
            "span": list(range(int(r + 1), int(r + n - 1))),
        },
        "right": {
            "color": int(right_col),
            "need_edge": "left",
            "fixed": int(c + n - 1),
            "span": list(range(int(r + 1), int(r + n - 1))),
        },
    }


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int, color: int):
    m = (g == int(color)) & (~hub_mask)
    labc, nc = label_mask(m, connectivity=8)
    comps = []
    for cid in range(1, int(nc) + 1):
        comp_mask = labc == int(cid)
        if not bool(comp_mask.any()):
            continue
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = int(color)
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        comps.append((arr, comp_mask))
    return comps


def _match_and_place_candidate(
    arr0: np.ndarray,
    mask0: np.ndarray,
    *,
    need_edge: str,
    required_len: int,
    fixed: int,
    span: list[int],
    out: np.ndarray,
    bg: int,
):
    H, W = out.shape
    out_edge = OPPOSITE[need_edge]

    for krot in range(4):
        a = np.rot90(arr0, int(krot))
        msk = np.rot90(mask0, int(krot))
        a, msk = _crop_to_mask(a, msk)

        seg_in = _edge_segment(msk, need_edge)
        seg_out = _edge_segment(msk, out_edge)
        if seg_in is None or seg_out is None:
            continue

        if _segment_len(msk, need_edge) != int(required_len):
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg_in
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg_in
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
        if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
            continue

        ok = True
        for R, C in placed:
            val = int(a[int(R - dr), int(C - dc)])
            if val == int(bg):
                continue
            cur = int(out[int(R), int(C)])
            if cur != int(bg) and cur != val:
                ok = False
                break
        if not ok:
            continue

        out_len = _segment_len(msk, out_edge)
        if out_len is None or int(out_len) <= 0:
            continue

        if out_edge in ("left", "right"):
            _typ2, out_c, out_rows = seg_out
            next_span = [int(x + dr) for x in out_rows]
            next_fixed = int(out_c + dc)
        else:
            _typ2, out_r, out_cols = seg_out
            next_span = [int(x + dc) for x in out_cols]
            next_fixed = int(out_r + dr)

        return {
            "arr": a,
            "mask": msk,
            "dr": int(dr),
            "dc": int(dc),
            "krot": int(krot),
            "out_len": int(out_len),
            "next_fixed": int(next_fixed),
            "next_span": next_span,
            "pixels": int(msk.sum()),
        }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), colors = hub
    hub_mask = np.zeros_like(g, dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, int(bg))
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]

    target = int(n - 2)
    for _side, info in _side_specs(r, c, n, colors).items():
        comps = _extract_components(g, hub_mask, bg, info["color"])
        need_edge = str(info["need_edge"])
        required_len = int(target)
        fixed = int(info["fixed"])
        span = list(info["span"])

        while comps:
            matches = []
            for idx, (arr0, mask0) in enumerate(comps):
                cand = _match_and_place_candidate(
                    arr0,
                    mask0,
                    need_edge=need_edge,
                    required_len=required_len,
                    fixed=fixed,
                    span=span,
                    out=out,
                    bg=bg,
                )
                if cand is not None:
                    matches.append((idx, cand))

            if not matches:
                break

            matches.sort(key=lambda item: (item[1]["krot"], -item[1]["pixels"], item[0]))
            idx, chosen = matches[0]
            paste(out, chosen["arr"], chosen["dr"], chosen["dc"], mask=chosen["mask"], overwrite=True, clip=False)
            fixed = int(chosen["next_fixed"])
            span = list(chosen["next_span"])
            required_len = int(chosen["out_len"])
            comps.pop(idx)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _rot_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _count_valid_rotations(mask: np.ndarray, side: str, in_len: int, out_len: int) -> int:
    need_edge = SIDE_TO_NEED_EDGE[side]
    out_edge = OPPOSITE[need_edge]
    cnt = 0
    for k in range(4):
        rot = np.rot90(mask, k)
        rot, rot_mask = _crop_to_mask(rot.astype(int), rot)
        rot_mask = rot_mask.astype(bool)
        if _segment_len(rot_mask, need_edge) == int(in_len) and _segment_len(rot_mask, out_edge) == int(out_len):
            cnt += 1
    return int(cnt)


def _build_horizontal_link_mask(in_len: int, out_len: int, rng: random.Random) -> np.ndarray | None:
    for _attempt in range(500):
        w = rng.randint(3, 5)
        h = max(int(in_len), int(out_len)) + rng.randint(0, 2)
        if h < max(int(in_len), int(out_len)):
            continue

        left_top = rng.randint(0, h - int(out_len))
        right_top = rng.randint(0, h - int(in_len))
        left_bot = left_top + int(out_len) - 1
        right_bot = right_top + int(in_len) - 1

        grid = np.zeros((h, w), dtype=bool)
        for x in range(w):
            top = int(round((left_top * (w - 1 - x) + right_top * x) / max(1, w - 1)))
            bot = int(round((left_bot * (w - 1 - x) + right_bot * x) / max(1, w - 1)))
            if top > bot:
                top, bot = bot, top
            grid[top : bot + 1, x] = True

        if w >= 4 and rng.random() < 0.6:
            x = rng.randint(1, w - 2)
            filled = np.where(grid[:, x])[0]
            if len(filled) > 0:
                if rng.random() < 0.5 and int(filled[0]) > 0:
                    grid[int(filled[0]) - 1, x] = True
                elif int(filled[-1]) < h - 1:
                    grid[int(filled[-1]) + 1, x] = True

        grid, grid_mask = _crop_to_mask(grid.astype(int), grid)
        grid = grid_mask.astype(bool)

        if _segment_len(grid, "right") != int(in_len):
            continue
        if _segment_len(grid, "left") != int(out_len):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue

        px = int(grid.sum())
        if px < 4 or px > 16:
            continue

        return grid

    return None


def _make_link_mask(in_len: int, out_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    for _attempt in range(600):
        base = _build_horizontal_link_mask(int(in_len), int(out_len), rng)
        if base is None:
            continue
        mask = _rot_to_side(base, side)
        mask, mask_crop = _crop_to_mask(mask.astype(int), mask)
        mask = mask_crop.astype(bool)
        if _count_valid_rotations(mask, side, int(in_len), int(out_len)) != 1:
            continue
        return mask
    return None


def _choose_chain_lengths(start_len: int, rng: random.Random) -> list[int]:
    templates = {
        2: [[2, 3, 1], [2, 4, 1]],
        3: [[3, 2, 1], [3, 4, 1]],
        4: [[4, 2, 1], [4, 3, 1]],
    }
    return list(rng.choice(templates[int(start_len)]))


def _place_hub_frame(out: np.ndarray, hub_r: int, hub_c: int, n: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + n, hub_c : hub_c + n] = int(bg)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(side_colors["top"])
        out[hub_r + n - 1, hub_c + ci] = int(side_colors["bot"])
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(side_colors["left"])
        out[hub_r + ri, hub_c + n - 1] = int(side_colors["right"])


def _place_attached_link(
    out: np.ndarray,
    *,
    bg: int,
    color: int,
    mask: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
):
    arr = np.where(mask, int(color), int(bg)).astype(int)
    seg_in = _edge_segment(mask, need_edge)
    assert seg_in is not None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg_in
        sr = int(span[0] - rows[0])
        sc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg_in
        sr = int(fixed - edge_r)
        sc = int(span[0] - cols[0])

    h, w = mask.shape
    H, W = out.shape
    if sr < 0 or sc < 0 or sr + h > H or sc + w > W:
        return None

    for rr in range(h):
        for cc in range(w):
            if not bool(mask[rr, cc]):
                continue
            cur = int(out[sr + rr, sc + cc])
            if cur != int(bg) and cur != int(color):
                return None

    paste(out, arr, sr, sc, transparent=bg)

    out_edge = OPPOSITE[need_edge]
    seg_out = _edge_segment(mask, out_edge)
    assert seg_out is not None
    out_len = _segment_len(mask, out_edge)
    assert out_len is not None

    if out_edge in ("left", "right"):
        _typ2, out_c, out_rows = seg_out
        next_span = [int(sr + x) for x in out_rows]
        next_fixed = int(sc + out_c)
    else:
        _typ2, out_r, out_cols = seg_out
        next_span = [int(sc + x) for x in out_cols]
        next_fixed = int(sr + out_r)

    return {
        "arr": arr,
        "mask": mask,
        "sr": int(sr),
        "sc": int(sc),
        "next_fixed": int(next_fixed),
        "next_span": next_span,
        "out_len": int(out_len),
    }


def _place_side_sequence(
    out: np.ndarray,
    *,
    bg: int,
    color: int,
    masks: list[np.ndarray],
    need_edge: str,
    fixed: int,
    span: list[int],
    required_len: int,
):
    placed = []
    for mask in masks:
        if _segment_len(mask, need_edge) != int(required_len):
            return None
        item = _place_attached_link(
            out,
            bg=bg,
            color=color,
            mask=mask,
            need_edge=need_edge,
            fixed=fixed,
            span=span,
        )
        if item is None:
            return None
        placed.append(item)
        fixed = int(item["next_fixed"])
        span = list(item["next_span"])
        required_len = int(item["out_len"])
    return placed


def _simulate_extents(n: int, side_colors: dict[str, int], side_masks: dict[str, list[np.ndarray]], bg: int):
    temp = np.full((80, 80), int(bg), dtype=int)
    hub_r = 35
    hub_c = 35
    _place_hub_frame(temp, hub_r, hub_c, n, side_colors, bg)
    specs = _side_specs(
        hub_r,
        hub_c,
        n,
        (side_colors["top"], side_colors["bot"], side_colors["left"], side_colors["right"]),
    )

    for side in ["top", "bot", "left", "right"]:
        masks = side_masks.get(side, [])
        if not masks:
            continue
        info = specs[side]
        placed = _place_side_sequence(
            temp,
            bg=bg,
            color=int(info["color"]),
            masks=masks,
            need_edge=str(info["need_edge"]),
            fixed=int(info["fixed"]),
            span=list(info["span"]),
            required_len=int(n - 2),
        )
        if placed is None:
            return None

    pts = np.argwhere(temp != int(bg))
    if pts.size == 0:
        return None
    min_r = int(pts[:, 0].min())
    max_r = int(pts[:, 0].max())
    min_c = int(pts[:, 1].min())
    max_c = int(pts[:, 1].max())
    return {
        "top": int(hub_r - min_r),
        "bot": int(max_r - (hub_r + n - 1)),
        "left": int(hub_c - min_c),
        "right": int(max_c - (hub_c + n - 1)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = rng.choice([4, 5, 6])
    edge_len = int(n - 2)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    chain_side = rng.choice(["top", "bot", "left", "right"])
    chain_color = int(side_colors[chain_side])
    chain_lengths = _choose_chain_lengths(edge_len, rng)
    link_pairs = [(int(chain_lengths[i]), int(chain_lengths[i + 1])) for i in range(len(chain_lengths) - 1)]

    side_masks: dict[str, list[np.ndarray]] = {}
    chain_masks: list[np.ndarray] = []
    for in_len, out_len in link_pairs:
        mask = _make_link_mask(int(in_len), int(out_len), chain_side, rng)
        if mask is None:
            return None
        chain_masks.append(mask)
    side_masks[chain_side] = chain_masks

    if rng.random() < 0.6:
        extra_side = rng.choice(ADJACENT_SIDES[chain_side])
        extra_out_len = rng.choice([1, 2, 3, 4])
        extra_mask = _make_link_mask(edge_len, int(extra_out_len), extra_side, rng)
        if extra_mask is None:
            return None
        side_masks[extra_side] = [extra_mask]

    ext = _simulate_extents(n, side_colors, side_masks, bg)
    if ext is None:
        return None

    min_h = max(18, n + int(ext["top"]) + int(ext["bot"]) + 6)
    min_w = max(20, n + int(ext["left"]) + int(ext["right"]) + 6)
    if min_h > 26 or min_w > 26:
        return None

    H = rng.randint(max(min_h, 22), 26)
    W = rng.randint(max(min_w, 22), 26)
    hub_r_min = int(ext["top"]) + 2
    hub_r_max = H - n - int(ext["bot"]) - 2
    hub_c_min = int(ext["left"]) + 2
    hub_c_max = W - n - int(ext["right"]) - 2
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    _place_hub_frame(out, hub_r, hub_c, n, side_colors, bg)

    specs = _side_specs(hub_r, hub_c, n, (top_col, bot_col, left_col, right_col))
    attached = []
    for side in ["top", "bot", "left", "right"]:
        masks = side_masks.get(side, [])
        if not masks:
            continue
        info = specs[side]
        seq = _place_side_sequence(
            out,
            bg=bg,
            color=int(info["color"]),
            masks=masks,
            need_edge=str(info["need_edge"]),
            fixed=int(info["fixed"]),
            span=list(info["span"]),
            required_len=edge_len,
        )
        if seq is None:
            return None
        attached.extend(seq)

    inp = np.full((H, W), int(bg), dtype=int)
    _place_hub_frame(inp, hub_r, hub_c, n, side_colors, bg)
    canvas = inp.copy()

    for placed in attached:
        arr = placed["arr"]
        mask = placed["mask"]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, k)
        rot_mask = np.rot90(mask, k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.25:
        free_colors = [c for c in range(10) if c != bg and c not in set(side_colors.values())]
        if free_colors:
            dcol = rng.choice(free_colors)
            dmask = _make_link_mask(rng.choice([2, 3, 4]), rng.choice([1, 2, 3]), rng.choice(["top", "bot", "left", "right"]), rng)
            if dmask is not None:
                darr = np.where(dmask, int(dcol), int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, darr, transparent=bg, padding=2, padding_connectivity=8)
                    paste(canvas, darr, pr, pc, transparent=bg)
                    paste(inp, darr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
