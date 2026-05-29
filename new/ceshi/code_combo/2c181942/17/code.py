# Combo variant of 2c181942 (index 17)
# combo_spec: D13/1+D14/0
# Name: 移动轨迹+颜色序处理

from __future__ import annotations

import itertools
import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste
from scipy.ndimage import label as _label


"""
design_analysis:
  a_concretization_and_trace:
    original_task: |
      原题先按 hub 边颜色找到同色散落精灵，再在 4 个旋转里找唯一合法朝向，
      最后把精灵直接贴到 hub 对应边上，输出里只保留 hub 和贴附后的精灵。
    fused_task: |
      本组合保留“颜色配边 + 旋转贴附”的核心几何，但把输出渲染改成两阶段：
      1. 先解析出每个真实精灵的最终贴附位；
      2. 再按颜色值升序处理这些精灵：从源位置中心到目标位置中心画一条同色直线轨迹，
         后处理轨迹覆盖先处理轨迹；最后再把最终贴附精灵盖上去。
      这样源精灵本体消失，只留下从源到 hub 的移动痕迹和最终贴附结果。
    concrete_small_trace: |
      例：颜色 1 的精灵在左下，颜色 4 的精灵在右上。原题只会把二者分别旋转后贴到 hub。
      组合后会先处理颜色 1：在左下源位置与其最终贴附位之间画一条 1 色线，再贴 1 色精灵。
      然后处理颜色 4：在右上源位置与其贴附位之间画一条 4 色线；若这条线穿过先前的 1 色线，
      交叉处最终显示为 4，因为“后处理覆盖先处理”。差异不再只是“多了轨迹”，而是轨迹的
      颜色交叉结果依赖于处理顺序。

  b_complexity_calibration:
    choices: |
      为了让 D14 的顺序效应真实发生，生成器固定使用 4 个真实 hub 边色，而不是 2-4 个可变。
      否则颜色升序很容易退化成“只有一两条线，没有交叉，顺序无关”。
      同时颜色集合固定为 {1,2,3,4}，但四种颜色和四条边的对应关系每题随机打乱，
      避免人把“先后顺序”误读成固定边序。
    visible_support: |
      4 条轨迹都存在，且生成器主动搜索能产生可见轨迹交叉的散布布局。
      这样一组样例里通常既能看到“有轨迹”，又能看到“后画颜色压住先画颜色”。

  c_implicit_assumption_audit_of_original_main:
    - _find_hub() 依赖输入 hub 仍是中心附近、四边纯色、角和内部为 bg 的空心方块。
      组合没有改变输入 hub，因此这个前提仍成立。
    - 原版 `out = bg + hub` 然后逐 side 直接 paste 最终精灵，隐含“中间运动过程不可见”。
      组合后这一步不再足够，必须先解析所有精灵的最终放置，再统一渲染轨迹层。
    - 原版 side 循环顺序无语义，因为各 side 最终互不冲突。组合后 side 循环本身仍可用于
      提取几何，但实际绘制顺序必须改成按颜色值升序。
    - 原版冲突检查只关心最终贴附时是否和 hub/已贴精灵冲突。组合后轨迹会跨越大片背景区域，
      所以需要单独定义“轨迹只覆盖背景/旧轨迹，不覆盖 hub 和最终贴附区”，否则 hub 会被路径污染。
    - 原版默认同色外部只有一个主精灵，取最大 8 连通分量即可。组合仍沿用这一点，因此生成器
      不能放置与 hub 边同色的大型干扰物。

  d_placement_legality:
    output_first_core: |
      生成器仍先构造每个 side 的“最终贴附朝向”精灵，检查其与 hub 的贴附合法性，
      再把这些精灵旋转后散布到输入中。
    scatter_legality: |
      散布位置不再纯随机，而是从几组角落候选位置中搜索，保证：
      - 每个散布 patch 不越界；
      - 不与 hub 或其他 patch 重叠；
      - 轨迹长度足够；
      - 轨迹交叉足够多，使顺序效应可见。

  e_visual_observability:
    - 人能直接看到：输出中散落源精灵不见了，取而代之的是通向 hub 的同色线。
    - 多个颜色在交叉处出现覆盖，且哪个颜色压住别人会在不同样例里稳定遵循同一颜色优先级。
    - 颜色优先级虽然内部实现为升序，但生成器固定反复使用少量颜色 {1,2,3,4} 并制造交叉，
      因此解题者可以从样例中归纳出一个稳定的可见颜色先后关系，而不是面对“没有证据的隐藏排序”。

  f_visual_cleanliness:
    - 轨迹使用窄笔刷直线，而不是整块精灵 footprint 的拖影，避免中央区域涂成一团。
    - 轨迹不会覆盖 hub 和最终贴附区，输出层次始终只有三层：
      hub、轨迹、最终贴附精灵。
    - 不再加入干扰物，避免视觉噪声掩盖顺序交叉。

  g_color_rule_inferability:
    - 颜色本身仍先承担原题的“精灵颜色匹配 hub 边颜色”的角色。
    - 新增的顺序信息只通过“哪种颜色的轨迹压住哪种颜色”体现。
      由于颜色种类固定且样例中会重复出现多色交叉，这个规则有结构性线索可归纳。

  h_post_test_metrics:
    trajectory_metric: |
      `trail_visible_cells` = 输出中位于 hub 和最终贴附精灵之外的非背景轨迹格数。
      原题该值恒为 0；本组合要求该值显著大于 0。
    order_metric: |
      `order_sensitive_cells` = 同一输入在“升序轨迹绘制”和“降序轨迹绘制”两种处理顺序下
      的输出差异格数。若该值为 0，则说明样本虽然有轨迹，但 D14 顺序退化为无效。
      生成器只接受 `trail_visible_cells` 和 `order_sensitive_cells` 都达标的样本。

  i_new_solver_assumptions_and_generator_guarantees:
    assumptions:
      - 每个 hub 边颜色 outside hub 至多对应一个主要 8 连通精灵
      - 每个真实精灵存在且只有一个合法旋转能贴到其对应 side
      - 轨迹端点可由源精灵中心和最终贴附精灵中心稳定定义
      - 轨迹可覆盖旧轨迹，但不会要求覆盖 hub 或最终贴附精灵
    generator_guarantees: |
      生成器用原题式 attached-orientation 构造保证合法旋转；固定 4 个真实颜色保证每题都有
      多步顺序；角落候选搜索保证散布合法并筛掉顺序不生效的退化布局；返回前用本 solver
      及顺序敏感指标再次验收。
"""


_SIDES = ("top", "bot", "left", "right")


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[int(r), int(c)]) != int(bg)
                    or int(g[int(r), int(c + n - 1)]) != int(bg)
                    or int(g[int(r + n - 1), int(c)]) != int(bg)
                    or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue

                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[int(r), int(c + 1) : int(c + n - 1)]
                bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = g[int(r + 1) : int(r + n - 1), int(c)]
                right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]

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
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def _edge_segment(mask: np.ndarray, side: str):
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


def _segment_center(mask: np.ndarray, dr: int = 0, dc: int = 0) -> tuple[int, int]:
    pts = np.argwhere(mask)
    center = pts.mean(axis=0)
    return int(round(float(center[0])) + int(dr)), int(round(float(center[1])) + int(dc))


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    points: list[tuple[int, int]] = []

    dr = abs(int(r1 - r0))
    dc = abs(int(c1 - c0))
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    rr = int(r0)
    cc = int(c0)
    while True:
        points.append((int(rr), int(cc)))
        if rr == int(r1) and cc == int(c1):
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            rr += sr
        if e2 < dr:
            err += dr
            cc += sc
    return points


def _trail_brush(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c)),
        (int(r - 1), int(c)),
        (int(r + 1), int(c)),
        (int(r), int(c - 1)),
        (int(r), int(c + 1)),
    ]


def _side_specs(r: int, c: int, n: int, colors: tuple[int, int, int, int]):
    top_col, bot_col, left_col, right_col = colors
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def _resolve_sprite(
    g: np.ndarray,
    bg: int,
    hub_mask: np.ndarray,
    side: str,
    col: int,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
) -> dict[str, object] | None:
    H, W = g.shape
    m = (g == int(col)) & (~hub_mask)
    if not bool(m.any()):
        return None

    labc, nc = label_mask(m, connectivity=8)
    best_id = None
    best_sz = -1
    for cid in range(1, int(nc) + 1):
        sz = int((labc == int(cid)).sum())
        if sz > best_sz:
            best_sz = sz
            best_id = int(cid)
    if best_id is None or best_sz <= 0:
        return None

    spr_mask_full = labc == int(best_id)
    src_pts = np.argwhere(spr_mask_full)
    src_center = (
        int(round(float(src_pts[:, 0].mean()))),
        int(round(float(src_pts[:, 1].mean()))),
    )
    spr_arr = g.copy()
    spr_arr[~spr_mask_full] = int(bg)
    spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask_full)

    found = None
    for krot in range(4):
        arr = np.rot90(spr_arr, int(krot))
        mask = np.rot90(spr_mask, int(krot))
        arr, mask = _crop_to_mask(arr, mask)

        if need_edge in ("left", "right"):
            if not bool(np.array_equal(mask, np.flipud(mask))):
                continue
        else:
            if not bool(np.array_equal(mask, np.fliplr(mask))):
                continue

        seg = _edge_segment(mask, need_edge)
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

        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(mask)}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue
        found = (arr, mask, int(dr), int(dc), placed)
        break

    if found is None:
        return None

    arr, mask, dr, dc, placed = found
    dst_center = _segment_center(mask, dr=dr, dc=dc)
    return {
        "side": side,
        "color": int(col),
        "src_center": src_center,
        "dst_center": dst_center,
        "arr": arr,
        "mask": mask,
        "dr": int(dr),
        "dc": int(dc),
        "placed": placed,
    }


def _solve_scene(
    grid: np.ndarray,
    *,
    order: list[int] | None = None,
    paint_trails: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (r, c, n), colors = hub

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    target = int(n - 2)
    infos: list[dict[str, object]] = []
    for side, (col, (need_edge, fixed, span)) in _side_specs(int(r), int(c), int(n), colors).items():
        info = _resolve_sprite(g, bg, hub_mask, side, int(col), need_edge, int(fixed), span, int(target))
        if info is not None:
            infos.append(info)

    if order is None:
        order = sorted(int(info["color"]) for info in infos)
    rank = {int(col): idx for idx, col in enumerate(order)}
    infos.sort(key=lambda info: (rank.get(int(info["color"]), 10**9), int(info["color"])))

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    protected_mask = hub_mask.copy()
    final_sprite_mask = np.zeros((H, W), dtype=bool)
    for info in infos:
        for rr, cc in info["placed"]:
            final_sprite_mask[int(rr), int(cc)] = True
            protected_mask[int(rr), int(cc)] = True

    trail_mask = np.zeros((H, W), dtype=bool)
    if paint_trails:
        for info in infos:
            col = int(info["color"])
            for rr, cc in _line_points(
                tuple(info["src_center"]),
                tuple(info["dst_center"]),
            ):
                for tr, tc in _trail_brush(int(rr), int(cc)):
                    if 0 <= int(tr) < H and 0 <= int(tc) < W and not bool(protected_mask[int(tr), int(tc)]):
                        out[int(tr), int(tc)] = int(col)
                        trail_mask[int(tr), int(tc)] = True

    for info in infos:
        paste(out, info["arr"], int(info["dr"]), int(info["dc"]), mask=info["mask"], overwrite=True, clip=True)

    meta = {
        "trail_visible_cells": int(np.count_nonzero(trail_mask)),
        "matched_sides": int(len(infos)),
        "final_sprite_cells": int(np.count_nonzero(final_sprite_mask)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to hub sides as in the source task, but render them in color order:
    draw same-color straight trajectories from source to destination first, allowing later
    colors to cover earlier trails, then place the final attached sprites.
    """
    out, _meta = _solve_scene(np.array(grid, dtype=int), paint_trails=True)
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    if int(edge_len) != 2:
        return None

    # Canonical left-attach families:
    # - right edge has exactly two contiguous cells,
    # - flipud symmetric,
    # - not fliplr symmetric,
    # - exactly one rotation is valid for each side.
    bases = [
        np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=bool,
        ),
    ]
    grid = rng.choice(bases)

    labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
    if int(num) != 1:
        return None

    if side == "left":
        return grid.copy()
    if side == "right":
        return np.fliplr(grid)
    if side == "top":
        return np.rot90(grid, -1)
    if side == "bot":
        return np.rot90(grid, 1)
    return None


def _attach_sprite(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int, color: int, bg: int):
    arr = np.where(mask, int(color), int(bg)).astype(int)
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r - sh)
    elif side == "bot":
        filled = np.where(mask[0])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r + n)
    elif side == "left":
        filled = np.where(mask[:, -1])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c - sw)
    else:
        filled = np.where(mask[:, 0])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c + n)
    return arr, mask, int(sr), int(sc)


def _crop_patch(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows_any = mask.any(axis=1)
    cols_any = mask.any(axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    H, W = canvas.shape
    h, w = patch.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != int(bg)
    if bool((((canvas[int(r0) : int(r0 + h), int(c0) : int(c0 + w)] != int(bg)) & mask)).any()):
        return False

    rr0 = max(0, int(r0 - padding))
    cc0 = max(0, int(c0 - padding))
    rr1 = min(H, int(r0 + h + padding))
    cc1 = min(W, int(c0 + w + padding))
    if bool((canvas[int(rr0) : int(rr1), int(cc0) : int(cc1)] != int(bg)).any()):
        local = canvas[int(rr0) : int(rr1), int(cc0) : int(cc1)] != int(bg)
        hole = np.zeros((int(rr1 - rr0), int(cc1 - cc0)), dtype=bool)
        hole[int(r0 - rr0) : int(r0 - rr0 + h), int(c0 - cc0) : int(c0 - cc0 + w)] = True
        if bool((local & (~hole)).any()):
            return False
    return True


def _candidate_positions(side: str, h: int, w: int, H: int, W: int) -> list[tuple[int, int]]:
    top_r = 1
    bot_r = int(H - h - 2)
    left_c = 1
    right_c = int(W - w - 2)
    mid_left_c = max(1, int(W // 4 - w // 2))
    mid_right_c = min(int(W - w - 2), int(3 * W // 4 - w // 2))
    mid_top_r = max(1, int(H // 4 - h // 2))
    mid_bot_r = min(int(H - h - 2), int(3 * H // 4 - h // 2))

    if side == "top":
        return [(bot_r, left_c), (bot_r, right_c), (bot_r, mid_left_c), (bot_r, mid_right_c)]
    if side == "bot":
        return [(top_r, left_c), (top_r, right_c), (top_r, mid_left_c), (top_r, mid_right_c)]
    if side == "left":
        return [(top_r, right_c), (bot_r, right_c), (mid_top_r, right_c), (mid_bot_r, right_c)]
    return [(top_r, left_c), (bot_r, left_c), (mid_top_r, left_c), (mid_bot_r, left_c)]


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    combo_out, combo_meta = _solve_scene(inp, paint_trails=True)
    rev_out, _ = _solve_scene(inp, order=sorted({1, 2, 3, 4}, reverse=True), paint_trails=True)
    no_trail_out, _ = _solve_scene(inp, paint_trails=False)
    return {
        "trail_visible_cells": int(combo_meta["trail_visible_cells"]),
        "order_sensitive_cells": int(np.count_nonzero(combo_out != rev_out)),
        "trail_only_cells": int(np.count_nonzero(combo_out != no_trail_out)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 8
    n = 4
    edge_len = int(n - 2)

    H = rng.randint(18, 24)
    W = rng.randint(20, 24)

    perm = rng.sample([1, 2, 3, 4], 4)
    top_col, bot_col, left_col, right_col = perm
    side_colors = {
        "top": int(top_col),
        "bot": int(bot_col),
        "left": int(left_col),
        "right": int(right_col),
    }

    sprite_masks: dict[str, np.ndarray] = {}
    for side in _SIDES:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 3
    hub_r_min = int(margin)
    hub_r_max = int(H - n - margin)
    hub_c_min = int(margin)
    hub_c_max = int(W - n - margin)
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out_base = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, n - 1):
        out_base[int(hub_r), int(hub_c + ci)] = int(top_col)
        out_base[int(hub_r + n - 1), int(hub_c + ci)] = int(bot_col)
    for ri in range(1, n - 1):
        out_base[int(hub_r + ri), int(hub_c)] = int(left_col)
        out_base[int(hub_r + ri), int(hub_c + n - 1)] = int(right_col)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    occupied = out_base.copy()
    for side in _SIDES:
        arr, mask, sr, sc = _attach_sprite(sprite_masks[side], side, hub_r, hub_c, n, side_colors[side], bg)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(occupied[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None
        paste(occupied, arr, int(sr), int(sc), transparent=bg)
        attached[side] = (arr, mask, int(sr), int(sc))

    patches: dict[str, np.ndarray] = {}
    for side in _SIDES:
        arr, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, int(k))
        rot_mask = np.rot90(mask, int(k))
        patch, patch_mask = _crop_patch(rot_arr, rot_mask)
        patches[side] = np.where(patch_mask, patch, bg).astype(int)

    side_order = list(_SIDES)
    rng.shuffle(side_order)
    candidate_lists = {side: _candidate_positions(side, *patches[side].shape, H, W) for side in _SIDES}
    for side in _SIDES:
        rng.shuffle(candidate_lists[side])

    assignments = []
    for picks in itertools.product(*(candidate_lists[side] for side in side_order)):
        canvas = out_base.copy()
        placed: dict[str, tuple[int, int]] = {}
        ok = True
        for side, (pr, pc) in zip(side_order, picks):
            patch = patches[side]
            if not _can_place(canvas, patch, int(pr), int(pc), bg, padding=1):
                ok = False
                break
            paste(canvas, patch, int(pr), int(pc), transparent=bg)
            placed[side] = (int(pr), int(pc))
        if ok:
            assignments.append(placed)
    rng.shuffle(assignments)

    for placed in assignments[:24]:
        inp = out_base.copy()
        for side in _SIDES:
            pr, pc = placed[side]
            paste(inp, patches[side], int(pr), int(pc), transparent=bg)

        try:
            combo_out = main(inp)
        except Exception:
            continue

        metrics = _behavior_metrics(inp)
        if metrics["trail_visible_cells"] < 10:
            continue
        if metrics["order_sensitive_cells"] < 3:
            continue
        if metrics["trail_only_cells"] < 10:
            continue
        if int(np.count_nonzero(combo_out != out_base)) < 10:
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
