# Combo variant of 2c181942 (index 19)
# combo_spec: D10/0+D15/0
# Name: 多hub竞争+背景由hub决定

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np
from scipy.ndimage import label as _label

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  fusion_summary: |
    本融合把原题的两个核心隐式前提一起拆掉：
      1) D10/0 破坏“离中心最近的空心方块就是真 hub”。
      2) D15/0 破坏“整题背景色固定为 8，且 hub 的内部/角也必须等于全局背景”。
    因此新 solver 不能先全局取一个 bg 再找 hub，而必须：
      - 先枚举所有候选方块；
      - 每个候选自带一个“内部色 = 假设背景色”；
      - 在该背景假设下重跑原题的 sprite 旋转/贴附验证；
      - 以成功贴附数来决定谁是真 hub；
      - 最终把整张输出清成真 hub 的内部色。

  a_concretization_and_trace: |
    具体小例子：
      - 外部画布底色固定是 8。
      - 输入里有两个 4x4 候选 hub。
        * 中央 decoy：四角和 2x2 内部都是 5，四边颜色为 {1,2,6,7}。
        * 偏离中心的 true hub：四角和 2x2 内部都是 3，四边颜色为 {4,9,1,2}。
      - 散落 sprite 只出现颜色 {4,9,2}，并且它们都能在 true hub 的三条边上唯一旋转后贴附。
    原逻辑会怎么做：
      - `bg = get_bg(grid)` 得到 8。
      - `_find_hub()` 只接受“角和内部都是 8”的方块，因此两个候选都不是 hub，求解直接崩。
    融合后逻辑怎么做：
      - 扫描任意候选：只要求“角和内部是同一个局部颜色 k”，不要求 k 等于全局最常见色。
      - 对 decoy 假设 `bg=5`，尝试按其边色去找匹配 sprite，成功数为 0。
      - 对 true hub 假设 `bg=3`，三块 sprite 都能成功贴到对应边，成功数为 3。
      - 输出整张网格先清成 3，再保留 true hub 并贴上三块 sprite。
    差异的关键不只是“多了一个 hub”，而是“候选 hub 的内部色同时决定了该候选的背景解释”。

  b_complexity_calibration: |
    为避免规则退化或过乱，生成器刻意控制自由度：
      - 候选 hub 固定为 2 个：一个真 hub，一个更靠近中心的强干扰 hub。
      - 真 hub 的内部色与外部画布底色总是不同，因此 D15 一定可见且必需。
      - 真 hub 只放 2-3 个 sprite，而不是 4 个。这样可以留下 1-2 个真 hub 边色不出现在 sprite 中，
        便于给 decoy 复用这些“缺席颜色”而又不产生误匹配。
      - decoy 四边仍做成完整候选方块，但其边色集合与真实 sprite 颜色集合不相交。
    这样既能稳定触发 D10 的竞争，也能稳定触发 D15 的局部背景推断。

  c_implicit_assumptions_review: |
    逐段看原 main() 中被破坏的前提：
      - `bg = get_bg(g)`：
        原本假设全局最常见色就是 hub 内部/角/输出背景。融合后不成立，因为外部画布色和真 hub 内部色故意不同。
      - `_find_hub()` 中 “corners must be bg / inner == bg”：
        原本把全局 bg 当作唯一空心判据。融合后必须改成“角和内部彼此相等”，并把这个局部色记为 candidate bg。
      - 只选离中心最近候选：
        D10 直接打破。新逻辑必须给每个候选做 attach 验证并比较得分。
      - `m = (g == col) & (~hub_mask)`：
        在多 hub 情况下会把其他候选 hub 的边当成 sprite。新逻辑必须屏蔽所有候选 hub 的整块 bbox。
      - `out = np.full_like(g, bg)`：
        这里的 bg 不能再是全局 get_bg，而必须是当前候选的内部色；否则生成器中的 D15 线索完全不起作用。
      - “一个颜色只需要最大的那个分量”：
        对真 hub 仍可沿用“找所有分量并取第一个可贴附者”；但不能再默认其他同色像素只有一个来源，因为 decoy 边和真 sprite
        可能共色。为此新逻辑先屏蔽所有 hub bbox，再在剩余区域里按连通块逐个尝试。

  d_placement_legality: |
    生成器仍是 output-first：
      - 先决定真 hub 的内部色和边色。
      - 在输出网格上用“真 hub 内部色”当背景，把 sprite 贴到真 hub 周围，逐块检查越界和冲突。
      - 再构造输入：整张输入先填外部画布色，再把 true/decoy 两个候选 hub 粘上去。
      - 最后把 sprite 随机旋转并散落，且用 hub bbox 作为禁区，保证不重叠、不越界、不掉进候选 hub 内部。
    这保证 solver 的所有新假设都有生成器支撑。

  e_visual_observability: |
    人类从样例里能直接看到两个稳定线索：
      - 输入里不止一个“方框候选”，而且中央那个经常不是最终保留对象。
      - 输出背景并不是输入大面积外部画布色，而是最终那个 hub 的内部填充色。
    这两个线索都来自可见空间结构和颜色复现，不依赖颜色编号大小、扫描顺序等不可视因素。

  f_visual_cleanliness: |
    只保留两个 hub 候选，避免画面充满干扰框。
    sprite 仍保持原题的分散放置与 8 连通单色外形，不叠在 hub 附近，也不混入 hub 内部，因此网格结构清楚。

  g_color_inference: |
    颜色规则仍然低复杂度：
      - 真 hub 四边四种颜色；
      - sprite 只复用其中 2-3 种；
      - decoy 的边色不复用真实 sprite 色；
      - 输出背景只由“被选中的 hub 内部色”决定。
    解题者能从少量样例归纳出“看哪个 hub 的边色能在散落件中找到匹配，再用它的内部色清空背景”。

  h_variant_metric_for_post_test: |
    测试通过后，用三个量化指标检查融合行为是否真的发生：
      - candidate_count：输入里候选 hub 数。这里应稳定为 2。
      - center_decoy_rate：距中心最近的候选是否不是最终真 hub。这里应接近 100%。
      - input_bg_vs_output_bg：`get_bg(input)` 是否不同于 `get_bg(output)`。这里应接近 100%，证明 D15 没有退化回固定外部底色。

  i_new_solver_assumptions_to_guarantee: |
    新 solver 增加了几个原题没有的输入假设，generator 会显式保证：
      - 候选 hub 的四角和内部是统一的局部背景色，且不同候选可有不同内部色。
      - 真 hub 在所有候选中拥有严格更多的可贴附 sprite 匹配数。
      - 所有真实 sprite 都在所有 hub bbox 之外，避免 hub 边被误分成 sprite。
      - 输入外部画布色与真 hub 内部色不同，从而强制 solver 不能依赖 `get_bg(input)`。
"""


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _edge_segment(mask: np.ndarray, side: str) -> tuple[str, int, list[int]] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _hub_candidates(g: np.ndarray) -> list[dict[str, Any]]:
    H, W = g.shape
    out: list[dict[str, Any]] = []
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                corners = [
                    int(g[int(r), int(c)]),
                    int(g[int(r), int(c + n - 1)]),
                    int(g[int(r + n - 1), int(c)]),
                    int(g[int(r + n - 1), int(c + n - 1)]),
                ]
                if len(set(corners)) != 1:
                    continue
                hub_bg = int(corners[0])

                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if inner.size == 0 or not bool((inner == int(hub_bg)).all()):
                    continue

                top = g[int(r), int(c + 1) : int(c + n - 1)]
                bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = g[int(r + 1) : int(r + n - 1), int(c)]
                right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]
                if (
                    bool((top == int(hub_bg)).any())
                    or bool((bot == int(hub_bg)).any())
                    or bool((left == int(hub_bg)).any())
                    or bool((right == int(hub_bg)).any())
                ):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                colors = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                center_score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                out.append(
                    {
                        "r": int(r),
                        "c": int(c),
                        "n": int(n),
                        "bg": int(hub_bg),
                        "colors": colors,
                        "center_score": float(center_score),
                        "all_distinct": len(set(colors)) == 4,
                    }
                )
    return out


def _all_hub_boxes_mask(shape: tuple[int, int], candidates: list[dict[str, Any]]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for cand in candidates:
        r = int(cand["r"])
        c = int(cand["c"])
        n = int(cand["n"])
        mask[r : r + n, c : c + n] = True
    return mask


def _side_specs(cand: dict[str, Any]) -> dict[str, tuple[int, tuple[str, int, list[int]]]]:
    r = int(cand["r"])
    c = int(cand["c"])
    n = int(cand["n"])
    top_col, bot_col, left_col, right_col = map(int, cand["colors"])
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def _try_attach_component(
    g: np.ndarray,
    bg: int,
    comp_mask_full: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    out: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    H, W = g.shape
    spr_arr = np.full_like(g, int(bg))
    spr_arr[comp_mask_full] = g[comp_mask_full]
    spr_arr, spr_mask = _crop_to_mask(spr_arr, comp_mask_full)

    for krot in range(4):
        a = np.rot90(spr_arr, int(krot))
        msk = np.rot90(spr_mask, int(krot))
        a, msk = _crop_to_mask(a, msk)

        if need_edge in ("left", "right"):
            if not bool(np.array_equal(msk, np.flipud(msk))):
                continue
        else:
            if not bool(np.array_equal(msk, np.fliplr(msk))):
                continue

        seg = _edge_segment(msk, need_edge)
        if seg is None:
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
        if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
            continue

        if out is not None:
            ok = True
            for R, C in placed:
                val = int(a[int(R - dr), int(C - dc)])
                if val == int(bg):
                    continue
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
                    ok = False
                    break
            if not ok:
                continue

        return a, msk, int(dr), int(dc), comp_mask_full

    return None


def _find_matching_sprite(
    g: np.ndarray,
    bg: int,
    all_hubs_mask: np.ndarray,
    used_sprite_mask: np.ndarray,
    col: int,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    out: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray] | None:
    m = (g == int(col)) & (~all_hubs_mask) & (~used_sprite_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    comp_ids = sorted(
        range(1, int(nc) + 1),
        key=lambda cid: int((labc == int(cid)).sum()),
        reverse=True,
    )
    for cid in comp_ids:
        comp_mask = labc == int(cid)
        found = _try_attach_component(g, bg, comp_mask, need_edge, fixed, span, target, out)
        if found is not None:
            return found
    return None


def _select_hub(g: np.ndarray, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
    all_hubs_mask = _all_hub_boxes_mask(g.shape, candidates)
    best_cand = None
    best_score = None

    for cand in candidates:
        r = int(cand["r"])
        c = int(cand["c"])
        n = int(cand["n"])
        bg = int(cand["bg"])
        target = int(n - 2)
        out = np.full_like(g, int(bg))
        out[r : r + n, c : c + n] = g[r : r + n, c : c + n]
        used_sprite_mask = np.zeros_like(g, dtype=bool)
        matched = 0
        total_pixels = 0
        side_matches: list[str] = []

        for side, (col, (need_edge, fixed, span)) in _side_specs(cand).items():
            found = _find_matching_sprite(
                g,
                bg,
                all_hubs_mask,
                used_sprite_mask,
                int(col),
                need_edge,
                fixed,
                span,
                target,
                out,
            )
            if found is None:
                continue
            a, msk, dr, dc, comp_mask = found
            paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
            used_sprite_mask |= comp_mask
            matched += 1
            total_pixels += int(comp_mask.sum())
            side_matches.append(side)

        cand_eval = dict(cand)
        cand_eval["matched_count"] = int(matched)
        cand_eval["matched_sides"] = tuple(side_matches)
        score = (
            int(matched),
            int(cand_eval["all_distinct"]),
            int(total_pixels),
            -float(cand_eval["center_score"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_cand = cand_eval

    if best_cand is None:
        raise AssertionError("no hub candidates found")
    return best_cand, all_hubs_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver:
    - There are multiple local-background hollow-square candidates.
    - Each candidate proposes both a hub bbox and a background color via its interior.
    - The true hub is the one whose edge colors can absorb the scattered sprites under the
      original rotation/symmetry/contact rules.
    - Output background is the chosen hub's interior color, not the input's dominant canvas color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    candidates = _hub_candidates(g)
    assert candidates

    hub, all_hubs_mask = _select_hub(g, candidates)
    r = int(hub["r"])
    c = int(hub["c"])
    n = int(hub["n"])
    bg = int(hub["bg"])
    out = np.full((H, W), int(bg), dtype=int)
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]
    used_sprite_mask = np.zeros_like(g, dtype=bool)
    target = int(n - 2)

    for _side, (col, (need_edge, fixed, span)) in _side_specs(hub).items():
        found = _find_matching_sprite(
            g,
            bg,
            all_hubs_mask,
            used_sprite_mask,
            int(col),
            need_edge,
            fixed,
            span,
            target,
            out,
        )
        if found is None:
            continue
        a, msk, dr, dc, comp_mask = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)
        used_sprite_mask |= comp_mask

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Construct a sprite mask in attached orientation for `side`.
    This is the tighter D10 helper: fixed height 4 makes edge_len=2 feasible under flipud
    symmetry while keeping the shape non-square and rotation-disambiguated.
    """
    e = int(edge_len)
    h = 4
    w = rng.randint(2, 4)
    half_h = (h + 1) // 2

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)
        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w):
                val = rng.random() < 0.5
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        right_col = grid[:, -1]
        runs: list[int] = []
        count = 0
        for v in right_col:
            if bool(v):
                count += 1
            else:
                if count > 0:
                    runs.append(int(count))
                count = 0
        if count > 0:
            runs.append(int(count))
        if len(runs) != 1 or int(runs[0]) != int(e):
            continue

        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not bool(rows_any.any()):
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not bool(np.array_equal(grid, np.flipud(grid))):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue
        if bool(np.array_equal(grid, np.fliplr(grid))):
            continue

        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, int(k))
            ra = rot.any(axis=1)
            ca = rot.any(axis=0)
            rr0, rr1 = np.where(ra)[0][[0, -1]]
            cc0, cc1 = np.where(ca)[0][[0, -1]]
            cr = rot[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1]
            if not bool(np.array_equal(cr, np.flipud(cr))):
                continue
            rc = cr[:, -1]
            runs2: list[int] = []
            cnt2 = 0
            for v in rc:
                if bool(v):
                    cnt2 += 1
                else:
                    if cnt2 > 0:
                        runs2.append(int(cnt2))
                    cnt2 = 0
            if cnt2 > 0:
                runs2.append(int(cnt2))
            if len(runs2) == 1 and int(runs2[0]) == int(e):
                valid_for_left += 1

        if int(valid_for_left) != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _build_hub_patch(n: int, colors: tuple[int, int, int, int], hub_bg: int) -> np.ndarray:
    top_col, bot_col, left_col, right_col = map(int, colors)
    patch = np.full((int(n), int(n)), int(hub_bg), dtype=int)
    patch[0, 1 : n - 1] = int(top_col)
    patch[n - 1, 1 : n - 1] = int(bot_col)
    patch[1 : n - 1, 0] = int(left_col)
    patch[1 : n - 1, n - 1] = int(right_col)
    return patch


def _center_score_pos(r: int, c: int, n: int, H: int, W: int) -> float:
    cr = float(r + (n - 1) / 2)
    cc = float(c + (n - 1) / 2)
    return abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))


def _list_bbox_positions(
    blocked: np.ndarray,
    h: int,
    w: int,
    *,
    padding: int = 0,
    r_min: int = 0,
    r_max: int | None = None,
    c_min: int = 0,
    c_max: int | None = None,
) -> list[tuple[int, int]]:
    H, W = blocked.shape
    if r_max is None:
        r_max = H - h
    if c_max is None:
        c_max = W - w
    out: list[tuple[int, int]] = []
    for r in range(int(r_min), int(r_max) + 1):
        for c in range(int(c_min), int(c_max) + 1):
            rr0 = max(0, int(r - padding))
            cc0 = max(0, int(c - padding))
            rr1 = min(H, int(r + h + padding))
            cc1 = min(W, int(c + w + padding))
            if not bool(blocked[rr0:rr1, cc0:cc1].any()):
                out.append((int(r), int(c)))
    return out


def _reserve_bbox(canvas: np.ndarray, r: int, c: int, h: int, w: int, mark: int) -> None:
    canvas[int(r) : int(r + h), int(c) : int(c + w)] = int(mark)


def _attached_sprite_location(hub_r: int, hub_c: int, n: int, side: str, mask: np.ndarray) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + n), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + n)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    outer_bg = 8
    N = 4
    edge_len = N - 2
    sides = ["top", "bot", "left", "right"]

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    palette = [c for c in range(10) if c != int(outer_bg)]
    rng.shuffle(palette)
    true_bg = int(palette[0])
    true_colors = tuple(int(x) for x in palette[1:5])
    decoy_bg = int(palette[5])
    fresh_decoy_colors = [int(x) for x in palette[6:]]
    if len(fresh_decoy_colors) != 3:
        return None

    side_colors = dict(zip(sides, true_colors))
    K = rng.randint(2, 3)
    sprite_sides = rng.sample(sides, K)
    unused_true_colors = [int(side_colors[s]) for s in sides if s not in sprite_sides]
    decoy_pool = fresh_decoy_colors + unused_true_colors
    if len(decoy_pool) < 4:
        return None
    rng.shuffle(decoy_pool)
    decoy_colors = tuple(int(x) for x in decoy_pool[:4])

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(int(mask.shape[0]) for mask in sprite_masks.values())
    max_sprite_w = max(int(mask.shape[1]) for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    blocked = np.zeros((H, W), dtype=bool)
    true_positions = _list_bbox_positions(
        blocked,
        N,
        N,
        padding=0,
        r_min=margin,
        r_max=H - N - margin,
        c_min=margin,
        c_max=W - N - margin,
    )
    if not true_positions:
        return None
    true_positions_sorted = sorted(
        true_positions,
        key=lambda rc: _center_score_pos(rc[0], rc[1], N, H, W),
        reverse=True,
    )
    far_positions = true_positions_sorted[: max(1, len(true_positions_sorted) // 2)]
    hub_r, hub_c = rng.choice(far_positions)
    true_center_score = _center_score_pos(hub_r, hub_c, N, H, W)
    blocked[hub_r : hub_r + N, hub_c : hub_c + N] = True

    decoy_positions = [
        rc
        for rc in _list_bbox_positions(blocked, N, N, padding=1)
        if _center_score_pos(rc[0], rc[1], N, H, W) < float(true_center_score)
    ]
    if not decoy_positions:
        decoy_positions = sorted(
            _list_bbox_positions(blocked, N, N, padding=0),
            key=lambda rc: _center_score_pos(rc[0], rc[1], N, H, W),
        )
    if not decoy_positions:
        return None
    decoy_r, decoy_c = decoy_positions[0]
    decoy_center_score = _center_score_pos(decoy_r, decoy_c, N, H, W)
    if not (float(decoy_center_score) < float(true_center_score)):
        return None

    true_hub_patch = _build_hub_patch(N, true_colors, true_bg)
    decoy_hub_patch = _build_hub_patch(N, decoy_colors, decoy_bg)

    out = np.full((H, W), int(true_bg), dtype=int)
    paste(out, true_hub_patch, hub_r, hub_c)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr_out = np.where(mask, int(color), int(true_bg)).astype(int)
        sr, sc = _attached_sprite_location(hub_r, hub_c, N, side, mask)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[rr, cc]) and int(out[sr + rr, sc + cc]) != int(true_bg):
                    return None
        paste(out, arr_out, sr, sc, transparent=int(true_bg))
        attached_sprites[side] = (arr_out, mask, int(sr), int(sc), int(color))

    inp = np.full((H, W), int(outer_bg), dtype=int)
    paste(inp, true_hub_patch, hub_r, hub_c)
    paste(inp, decoy_hub_patch, decoy_r, decoy_c)

    occupancy = np.full((H, W), int(outer_bg), dtype=int)
    occ_mark = 0 if int(outer_bg) != 0 else 1
    _reserve_bbox(occupancy, hub_r, hub_c, N, N, occ_mark)
    _reserve_bbox(occupancy, decoy_r, decoy_c, N, N, occ_mark)

    for side in sprite_sides:
        _arr_out, mask, _sr, _sc, color = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_mask = np.rot90(mask, int(k))
        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_mask = rotated_mask[int(r0_idx) : int(r1_idx) + 1, int(c0_idx) : int(c1_idx) + 1]
        patch = np.where(cropped_mask, int(color), int(outer_bg)).astype(int)

        try:
            pr, pc = random_free_location(
                occupancy,
                patch,
                transparent=int(outer_bg),
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, patch, pr, pc, transparent=int(outer_bg))
        paste(inp, patch, pr, pc, transparent=int(outer_bg))

    used_colors = {int(outer_bg), int(true_bg), int(decoy_bg), *map(int, true_colors), *map(int, decoy_colors)}
    distractor_colors = [c for c in range(10) if int(c) not in used_colors]
    if distractor_colors and rng.random() < 0.3:
        dcol = int(rng.choice(distractor_colors))
        dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
        if dist_mask is not None:
            dist_patch = np.where(dist_mask, int(dcol), int(outer_bg)).astype(int)
            try:
                pr, pc = random_free_location(
                    occupancy,
                    dist_patch,
                    transparent=int(outer_bg),
                    padding=2,
                    padding_connectivity=8,
                )
                paste(occupancy, dist_patch, pr, pc, transparent=int(outer_bg))
                paste(inp, dist_patch, pr, pc, transparent=int(outer_bg))
            except ValueError:
                pass

    result = main(inp)
    if not bool(np.array_equal(result, out)):
        return None
    if int(get_bg(inp)) == int(get_bg(out)):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
