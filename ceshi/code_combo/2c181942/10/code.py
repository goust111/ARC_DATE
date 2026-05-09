# Combo variant of 2c181942 (index 10)
# combo_spec: D3/2+D11/0
# Name: 无旋转+内嵌填充

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location
from scipy.ndimage import label as _label


"""
design_analysis:
  a_concretization_and_trace:
    original_task: |
      原题中，精灵先靠颜色匹配到 hub 某一边，再在 0/90/180/270 四种朝向里找出唯一合法旋转。
      合法朝向要求：接触边长度等于 hub 边长 N-2，且沿接触轴镜像对称。之后精灵贴在 hub 外侧。
    combo_variant: |
      本组合同时改两处：
      1. D3/2 取消旋转搜索。输入中的精灵已经是最终朝向，solver 只检查“当前朝向是否符合该 side”。
      2. D11/0 把贴附位置从 hub 外侧改成 hub 内部。接触边与 hub 彩边重合，精灵主体向内填充。
    concrete_trace: |
      设 hub 为 6x6，顶边颜色是红色，target=N-2=4。输入里红色精灵已经是一个 2x4 形状：
        1111
        .11.
      原题会继续尝试旋转它，并把其“底边”放到 hub 顶边上方一格。
      组合后不会旋转；solver 直接确认它的“顶边”长度正好是 4，且左右镜像对称，
      然后把这条顶边直接覆盖到 hub 的红色顶边上，第二行那两个红点落进 hub 内部。
      差异是双重的：既没有朝向恢复步骤，也不是向外贴，而是向内填。

  b_complexity_calibration:
    keep_visible_structure:
      - 仍保留 2-4 个真实 side、颜色一一配对、随机散落位置、可选干扰物
      - 输出仍只保留 hub 与被放回去的精灵，输入输出差异肉眼明显
    deliberate_simplification: |
      取消旋转后，若继续让 sprite 家族过于自由，容易出现“某形状恰好也能解释成别的 side”。
      因此生成器收敛到一个窄型基元家族，只通过 side 旋转一次生成四个定向版本。
      这不是退化：真正可观察的变化在于“输入已对正 + 输出内嵌”，而不是形状多样性。
    hub_size_choice: |
      为了让内嵌效果稳定且不重叠，hub 固定为 N=6，内部是 4x4。若沿用原题常见的 N=4，
      四侧同时向内填会挤在 2x2 内腔里，空间关系会失真。

  c_implicit_assumptions_review:
    hub_detection: |
      _find_hub 仍假设输入中的 hub 内部全是 bg。组合后这个前提仍必须保留，
      因为输入里只是“散落但已对正”的精灵，不能提前嵌回 hub。
    original_rotation_loop: |
      原 solver 依赖 rot90 枚举来修复朝向。D3/2 使这个前提失效，所以这里必须改成：
      只裁剪当前输入组件，并直接验证其当前朝向是否满足目标 side 的边长与对称约束。
    outer_attachment_geometry: |
      原 segs 使用 opposite side 和 hub 外一格的 fixed 坐标；D11/0 使其失效。
      组合后需要 same side，并把 fixed 设在 hub 彩边本身。
    overlap_policy: |
      原冲突检查允许“已有颜色与将粘贴颜色相同”的重合。组合后这一点继续成立，
      因为接触边需要与 hub 彩边同色重合。
    largest_component_by_color: |
      solver 仍默认每个真实颜色只有一个主精灵，且它是该颜色最大 8 连通分量。
      generator 必须继续避免同色大型干扰物。

  d_placement_legality:
    embedded_family: |
      使用左基元
        10
        11
        11
        10
      其四个定向版本嵌入 6x6 hub 后：
      - top/bot 只占内部中间两列
      - left/right 只占内部中间两行
      因此四侧同时存在时不会互相重叠。
    scatter_legality: |
      输入中的每个 patch 都是裁剪后的单色块，用 random_free_location 以 padding=2 散落，
      保证不越界、不重叠，并与 hub 保持可分辨间距。

  e_visual_observability:
    human_inference: |
      看 2-3 组样例时，人可以看到：同色块不需要再“转正”，它们已经以最终姿态存在，
      只是在输出中被整体平移回同色边，而且不是贴在外面，而是插到方框里面。
      这两个变化会同时反复出现，因此融合规律是可归纳的。
    no_hidden_rule: |
      不使用颜色数值大小、扫描顺序之类的人类不可见因素；side 选择完全由可见颜色决定。

  f_visual_cleanliness:
    layout: |
      输入只含空心 hub、少量散落单色精灵和最多一个异色干扰物。
      输出只含 hub 与内嵌精灵。内嵌基元很窄，不会把 4x4 内腔涂满成一团。

  g_color_inference:
    rule: |
      颜色规则不变：hub 四边四种不同颜色，精灵颜色与边颜色相同即配对。
      没有附加的颜色重映射，因此少量样例足以推断。

  h_variant_metric_for_post_test:
    metric: |
      量化指标 1：输出 hub 内部的非 bg 像素数 inner_fill。
      原题外贴时 inner_fill 恒为 0；本组合只要有真实精灵，inner_fill 必须 > 0。
      量化指标 2：把同一输入送入原 solver 与本组合 solver，比对输出是否不同。
      若多数样本都不同，说明样本确实触发了“无旋转 + 内嵌”而非退化回原题。

  i_new_solver_assumptions_to_guarantee:
    assumptions:
      - 输入 hub 是唯一靠近中心且内部全 bg 的彩边空心方框
      - 每个真实颜色至多有一个主要 8 连通精灵
      - 每个真实精灵在输入中已处于对应 side 的最终朝向
      - 该朝向在对应 side 上有长度为 N-2 的连续接触边，并满足对应轴对称
      - side 专用基元嵌入 hub 后不会与其他 side 的基元冲突
    generator_guarantees: |
      生成器逐条构造这些条件，并在返回前用 main(inp)==out 的 oracle 检查兜底。
"""


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


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


def _mask_valid_for_side(mask: np.ndarray, side: str, target: int) -> bool:
    if side in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False

    seg = _edge_segment(mask, side)
    if seg is None:
        return False

    if side in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) == int(target) and _is_contig(rows)

    _typ, _edge_r, cols = seg
    return len(cols) == int(target) and _is_contig(cols)


def _place_hub_frame(grid: np.ndarray, hub_r: int, hub_c: int, n: int, side_colors: dict[str, int], bg: int) -> None:
    grid[hub_r : hub_r + n, hub_c : hub_c + n] = int(bg)
    for ci in range(1, n - 1):
        grid[hub_r, hub_c + ci] = int(side_colors["top"])
        grid[hub_r + n - 1, hub_c + ci] = int(side_colors["bot"])
    for ri in range(1, n - 1):
        grid[hub_r + ri, hub_c] = int(side_colors["left"])
        grid[hub_r + ri, hub_c + n - 1] = int(side_colors["right"])


def _build_side_mask(edge_len: int, side: str) -> np.ndarray | None:
    if int(edge_len) != 4:
        return None

    left_embed = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=bool,
    )
    labeled, num = _label(left_embed, structure=np.ones((3, 3), dtype=int))
    if int(num) != 1:
        return None

    if side == "left":
        mask = left_embed
    elif side == "right":
        mask = np.fliplr(left_embed)
    elif side == "top":
        mask = np.rot90(left_embed, -1)
    elif side == "bot":
        mask = np.rot90(left_embed, 1)
    else:
        return None

    if not _mask_valid_for_side(mask, side, edge_len):
        return None
    other_valid = sum(int(_mask_valid_for_side(mask, other, edge_len)) for other in ("top", "bot", "left", "right"))
    if other_valid != 1:
        return None
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Match each scattered monochrome sprite to the hub side of the same color.
    - Do not rotate the sprite; the input orientation is already correct.
    - Translate the sprite so its same-named contact edge coincides with the colored hub edge.
    - The sprite fills inward into the hub cavity rather than attaching outside.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c,n), (top,bot,left,right))
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

    hub = _find_hub()
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    target = int(n - 2)
    segs = {
        "top": (int(top_col), ("top", int(r), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("bot", int(r + n - 1), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("left", int(c), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("right", int(c + n - 1), list(range(int(r + 1), int(r + n - 1))))),
    }

    for _side, (col, (need_edge, fixed, span)) in segs.items():
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        labc, nc = label_mask(m, connectivity=8)
        best_id = None
        best_sz = -1
        for cid in range(1, int(nc) + 1):
            sz = int((labc == int(cid)).sum())
            if sz > best_sz:
                best_sz = sz
                best_id = int(cid)
        if best_id is None or best_sz <= 0:
            continue

        spr_mask = labc == int(best_id)
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        if not _mask_valid_for_side(spr_mask, need_edge, target):
            continue

        seg = _edge_segment(spr_mask, need_edge)
        assert seg is not None

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(spr_mask)}
        if any(not (0 <= rr2 < H and 0 <= cc2 < W) for rr2, cc2 in placed):
            continue

        ok = True
        for rr2, cc2 in placed:
            val = int(spr_arr[int(rr2 - dr), int(cc2 - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr2), int(cc2)]) != int(bg) and int(out[int(rr2), int(cc2)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, spr_arr, int(dr), int(dc), mask=spr_mask, overwrite=True, clip=True)

    return out


def _embed_sprite(out: np.ndarray, hub_r: int, hub_c: int, n: int, side: str, color: int, mask: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    sh, sw = mask.shape
    arr = np.where(mask, int(color), int(bg)).astype(int)

    if side == "top":
        filled = np.where(mask[0])[0]
        sr = int(hub_r)
        sc = int(hub_c + 1 - filled[0])
    elif side == "bot":
        filled = np.where(mask[-1])[0]
        sr = int(hub_r + n - sh)
        sc = int(hub_c + 1 - filled[0])
    elif side == "left":
        filled = np.where(mask[:, 0])[0]
        sr = int(hub_r + 1 - filled[0])
        sc = int(hub_c)
    else:
        filled = np.where(mask[:, -1])[0]
        sr = int(hub_r + 1 - filled[0])
        sc = int(hub_c + n - sw)

    H, W = out.shape
    if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
        return None

    for rr in range(sh):
        for cc in range(sw):
            if not bool(mask[rr, cc]):
                continue
            dst = int(out[sr + rr, sc + cc])
            if dst != int(bg) and dst != int(color):
                return None

    paste(out, arr, sr, sc, transparent=int(bg))
    return arr, mask, int(sr), int(sc)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 6
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {
        "top": int(top_col),
        "bot": int(bot_col),
        "left": int(left_col),
        "right": int(right_col),
    }
    sides = ["top", "bot", "left", "right"]

    k = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, k)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _build_side_mask(edge_len, side)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)
    _place_hub_frame(out, hub_r, hub_c, n, side_colors, bg)
    _place_hub_frame(inp, hub_r, hub_c, n, side_colors, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _embed_sprite(out, hub_r, hub_c, n, side, side_colors[side], sprite_masks[side], bg)
        if placed is None:
            return None
        attached[side] = placed

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        patch = np.where(mask, arr, int(bg)).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=int(bg))
        paste(inp, patch, pr, pc, transparent=int(bg))

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dside = rng.choice(sides)
            dist_mask = _build_side_mask(edge_len, dside)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, dist_arr, transparent=int(bg), padding=2, padding_connectivity=8)
                    paste(canvas, dist_arr, pr, pc, transparent=int(bg))
                    paste(inp, dist_arr, pr, pc, transparent=int(bg))
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    inner_fill = int(np.count_nonzero(result[hub_r + 1 : hub_r + n - 1, hub_c + 1 : hub_c + n - 1] != int(bg)))
    if inner_fill <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
