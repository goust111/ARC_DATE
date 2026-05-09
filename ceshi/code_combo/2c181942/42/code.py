# Combo variant of 2c181942 (index 42)
# combo_spec: D11/2+D13/1+D14/0
# Name: 覆盖放置+移动轨迹+颜色序

from __future__ import annotations

import itertools
import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste
from scipy.ndimage import label as _label


"""
design_analysis:
  fusion_summary: |
    原题是“按 hub 边颜色匹配外部单色 sprite，旋转到唯一合法朝向后，flush 贴在 hub 外侧”。
    本融合同时改三处，而且三处共用同一条几何链：
      1. D11/2: 最终贴附不再在 hub 外侧一格，而是 contact 边直接压在 hub 彩边上。
      2. D13/1: 每个 sprite 从输入散布位置移动到目标位时，要留下 1px 直线轨迹。
      3. D14/0: 不再无序独立渲染，而是按颜色值升序逐个执行“画轨迹 -> 覆盖贴附”，
         后处理的轨迹和 sprite 都能覆盖先处理结果。

  a_concretization_and_trace: |
    设 N=4，因此 hub 每条彩边长度 target=2。假设颜色 1 对应顶边，颜色 4 对应右边。
    原题：
      - 颜色 1 的 sprite 旋到“底边刚好 2 格连续”，然后底边放在 row=r-1，紧贴 hub 顶边。
      - 颜色 4 的 sprite 旋到“左边刚好 2 格连续”，然后左边放在 col=c+N，紧贴 hub 右边。
      - 输出只含 hub + 两个最终 sprite。
    融合后：
      - 颜色 1 先处理：从源位置中心到其覆盖贴附后的中心画一条 1 的轨迹淡色，再把 sprite
        的接触边直接放到 row=r 上，覆盖原来的顶边颜色。
      - 颜色 4 后处理：也先画 4 的轨迹淡色，再把 sprite 的接触边直接放到 col=c+N-1 上。
      - 若第二条轨迹穿过第一条轨迹，则交叉处显示后者；若第二个 sprite 覆盖了第一条轨迹，
        最终也显示第二个 sprite 本体颜色。
    核心差异不是单纯“多了轨迹”，而是“覆盖放置”和“顺序覆盖”共享同一个目标位几何。

  b_complexity_calibration: |
    颜色顺序本质上是数值升序，若边色任意抽样，少量样例里人很难看出稳定先后关系。
    因此 generator 固定真实 hub 边色始终是 {1,2,3,4}，但四条边与这四个颜色的对应关系每题打乱。
    这样：
      - 颜色优先级在所有样例里都可见且重复；
      - 颜色仍然不是固定绑定到某一边；
      - 轨迹和最终覆盖都只围绕 4 个可见角色，不会引入过多自由度。
    轨迹色也固定为一个小映射表，而不是每题随机换色，避免“看到颜色变化却无法归纳”。

  c_implicit_assumption_audit: |
    原 main() 中与本融合冲突的隐式前提：
      - `segs` 把目标 fixed 行/列放在 hub 外一格；D11/2 后必须改为 hub 彩边所在行/列本身。
      - 原代码边解析后立刻 paste 到 `out`，默认 side 之间顺序无语义；D14/0 后必须先解析完
        所有 sprite，再按颜色序统一渲染。
      - 原题没有轨迹层，因此不会出现“后一个对象的路径压住前一个对象”的情况；D13/1 + D14/0
        后输出必须允许这种覆盖。
      - 原逻辑只检查最终贴附是否冲突。融合后若仍在解析阶段用 `out` 做碰撞屏障，会把本该允许的
        顺序覆盖误判为非法，所以解析最终目标位时不能依赖先前 side 的绘制结果。
    仍然保留的前提：
      - hub 仍是中心附近空心方框，`_find_hub` 仍可直接复用；
      - 每种 hub 边色在 hub 外只有一个主要真实 sprite，生成器不再放同色大干扰物；
      - 真实 sprite 仍要满足“目标接触边长度 = N-2 且旋转唯一”。

  d_placement_legality: |
    generator 仍采用 output-first 思路，但输出层分成两类：
      - 先在几何上确定 cover attachment 的最终 sprite 位置；
      - 再反推输入散布位置，并用 solver 自检最终带轨迹输出。
    合法性约束包括：
      - 覆盖贴附时允许压住同色 hub 边，但不能压住其它颜色；
      - 不越界；
      - 输入散布 patch 彼此分离且与 hub 分离；
      - 筛掉会让轨迹污染 hub 内部或最终 sprite 本体的布局，保持画面整洁。

  e_visual_observability: |
    人类能从少量样例中看到三件稳定事实：
      - sprite 不是贴在 hub 外面，而是直接压住 hub 彩边；
      - 输出里出现从散布位置通向 hub 的细轨迹；
      - 颜色 4 的轨迹总能压住颜色 3/2/1，颜色 3 又能压住 2/1，形成可见的稳定先后。
    虽然内部实现是“按颜色值升序”，但因为颜色集合固定且每题重复出现，人可以把它归纳为
    可见的颜色优先级，而不是不可观察的隐藏排序。

  f_visual_cleanliness: |
    轨迹只用 1px 直线，不用粗刷；同时 generator 拒绝会把 hub 内部/角点染脏的布局。
    顺序覆盖主要发生在轨迹交叉处，不要求大面积 trail-vs-sprite 互相涂抹，因此画面仍像
    “中心 hub + 四个覆盖贴附 sprite + 几条细线”。

  g_color_inferability: |
    真实边色固定为 {1,2,3,4}，轨迹色固定映射为 {1->5, 2->6, 3->7, 4->9}。
    每条轨迹都直接连接到同色最终 sprite，因此人能从结构上看出“哪条淡色线属于哪种实体色”。
    这种映射是小而稳定的，不会像随机换色那样不可归纳。

  h_post_test_metrics: |
    为防止样本退化，生成器和后验统计都使用三个指标：
      - trail_visible_cells: 融合输出与“同样覆盖贴附但不画轨迹”的输出差异格数。
        原题和无轨迹退化样本该值会很小。
      - order_sensitive_cells: 升序处理与降序处理的输出差异格数。若为 0，D14/0 实际未生效。
      - cover_sensitive_cells: 覆盖贴附版与 flush 外贴版的输出差异格数。若为 0，D11/2 是潜伏的。
    另外要求 hub_pollution_cells=0，保证轨迹不会把 hub 内腔和角点染乱。

  i_new_solver_assumptions_and_generator_guarantees: |
    新 solver 额外假设：
      - 所有真实颜色都来自固定集合 {1,2,3,4} 时，轨迹淡色映射稳定；
      - 每个 side 的最大同色 8 连通分量就是那个 side 的真实 sprite；
      - 每个 sprite 仍只有一个 rot90 朝向能满足对应边的接触长度条件。
    generator 显式保证：
      - 四个真实边色每题都出现；
      - 每个 side 都有一个真实 sprite；
      - 输入散布布局使 order/cover/trail 三项指标都显著非零。
"""


_SIDES = ("top", "bot", "left", "right")
_TRAIL_COLOR = {1: 5, 2: 6, 3: 7, 4: 9}


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


def _side_specs(r: int, c: int, n: int, colors: tuple[int, int, int, int], *, cover: bool):
    top_col, bot_col, left_col, right_col = colors
    if cover:
        return {
            "top": (int(top_col), ("bot", int(r), list(range(int(c + 1), int(c + n - 1))))),
            "bot": (int(bot_col), ("top", int(r + n - 1), list(range(int(c + 1), int(c + n - 1))))),
            "left": (int(left_col), ("right", int(c), list(range(int(r + 1), int(r + n - 1))))),
            "right": (int(right_col), ("left", int(c + n - 1), list(range(int(r + 1), int(r + n - 1))))),
        }
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


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

        dst_pts = np.array(sorted(placed))
        dst_center = (
            int(round(float(dst_pts[:, 0].mean()))),
            int(round(float(dst_pts[:, 1].mean()))),
        )
        found = {
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
        break

    return found


def _trail_color(color: int) -> int:
    return int(_TRAIL_COLOR.get(int(color), int(color)))


def _solve_scene(
    grid: np.ndarray,
    *,
    order: list[int] | None = None,
    cover: bool = True,
    paint_trails: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (r, c, n), colors = hub

    hub_mask = np.zeros_like(g, dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    infos: list[dict[str, object]] = []
    target = int(n - 2)
    for side, (col, (need_edge, fixed, span)) in _side_specs(int(r), int(c), int(n), colors, cover=cover).items():
        info = _resolve_sprite(g, bg, hub_mask, side, int(col), need_edge, int(fixed), span, int(target))
        if info is not None:
            infos.append(info)

    if order is None:
        order = sorted(int(info["color"]) for info in infos)
    rank = {int(col): idx for idx, col in enumerate(order)}
    infos.sort(key=lambda info: (rank.get(int(info["color"]), 10**9), int(info["color"])))

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    final_sprite_mask = np.zeros_like(g, dtype=bool)
    for info in infos:
        for rr, cc in info["placed"]:
            final_sprite_mask[int(rr), int(cc)] = True

    for info in infos:
        if paint_trails:
            trail_col = _trail_color(int(info["color"]))
            for rr, cc in _line_points(tuple(info["src_center"]), tuple(info["dst_center"])):
                if 0 <= int(rr) < g.shape[0] and 0 <= int(cc) < g.shape[1]:
                    out[int(rr), int(cc)] = int(trail_col)
        paste(out, info["arr"], int(info["dr"]), int(info["dc"]), mask=info["mask"], overwrite=True, clip=True)

    meta = {
        "hub_bbox": (int(r), int(c), int(n)),
        "final_sprite_mask": final_sprite_mask,
        "matched_sides": int(len(infos)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each sprite as in the source task, but place it by covering the matching hub edge.
    Sprites are then rendered in ascending color order; each one draws a straight trajectory
    in its trail color and finally pastes its body, so later colors cover earlier work.
    """
    out, _meta = _solve_scene(np.array(grid, dtype=int), cover=True, paint_trails=True)
    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    if int(edge_len) != 2:
        return None

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


def _attach_sprite(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int, color: int, bg: int, *, cover: bool):
    arr = np.where(mask, int(color), int(bg)).astype(int)
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r - sh + 1) if cover else int(hub_r - sh)
    elif side == "bot":
        filled = np.where(mask[0])[0]
        sc = int(hub_c + 1 - int(filled[0]))
        sr = int(hub_r + n - 1) if cover else int(hub_r + n)
    elif side == "left":
        filled = np.where(mask[:, -1])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c - sw + 1) if cover else int(hub_c - sw)
    else:
        filled = np.where(mask[:, 0])[0]
        sr = int(hub_r + 1 - int(filled[0]))
        sc = int(hub_c + n - 1) if cover else int(hub_c + n)
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
    combo_out, combo_meta = _solve_scene(inp, cover=True, paint_trails=True)
    no_trail_out, no_trail_meta = _solve_scene(inp, cover=True, paint_trails=False)
    rev_out, _ = _solve_scene(inp, cover=True, paint_trails=True, order=[4, 3, 2, 1])
    flush_out, _ = _solve_scene(inp, cover=False, paint_trails=True)

    r, c, n = combo_meta["hub_bbox"]
    hub_mask = np.zeros_like(inp, dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True
    final_sprite_mask = no_trail_meta["final_sprite_mask"]

    diff_no_trail = combo_out != no_trail_out
    return {
        "trail_visible_cells": int(np.count_nonzero(diff_no_trail)),
        "order_sensitive_cells": int(np.count_nonzero(combo_out != rev_out)),
        "cover_sensitive_cells": int(np.count_nonzero(combo_out != flush_out)),
        "hub_pollution_cells": int(np.count_nonzero(diff_no_trail & hub_mask)),
        "sprite_pollution_cells": int(np.count_nonzero(diff_no_trail & final_sprite_mask)),
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

    hub_only = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, n - 1):
        hub_only[int(hub_r), int(hub_c + ci)] = int(top_col)
        hub_only[int(hub_r + n - 1), int(hub_c + ci)] = int(bot_col)
    for ri in range(1, n - 1):
        hub_only[int(hub_r + ri), int(hub_c)] = int(left_col)
        hub_only[int(hub_r + ri), int(hub_c + n - 1)] = int(right_col)

    occupied = hub_only.copy()
    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in _SIDES:
        arr, mask, sr, sc = _attach_sprite(sprite_masks[side], side, hub_r, hub_c, n, side_colors[side], bg, cover=True)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if not bool(mask[int(rr), int(cc)]):
                    continue
                cur = int(occupied[int(sr + rr), int(sc + cc)])
                if cur != int(bg) and cur != int(side_colors[side]):
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

    assignments: list[dict[str, tuple[int, int]]] = []
    for picks in itertools.product(*(candidate_lists[side] for side in side_order)):
        canvas = hub_only.copy()
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

    for placed in assignments[:32]:
        inp = hub_only.copy()
        for side in _SIDES:
            pr, pc = placed[side]
            paste(inp, patches[side], int(pr), int(pc), transparent=bg)

        try:
            combo_out = main(inp)
        except Exception:
            continue

        metrics = _behavior_metrics(inp)
        if metrics["trail_visible_cells"] < 8:
            continue
        if metrics["order_sensitive_cells"] < 3:
            continue
        if metrics["cover_sensitive_cells"] < 8:
            continue
        if metrics["hub_pollution_cells"] != 0:
            continue
        if metrics["sprite_pollution_cells"] > 0:
            continue
        if int(np.count_nonzero(combo_out != hub_only)) < 18:
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
