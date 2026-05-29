# Combo variant of 2c181942 (index 39)
# combo_spec: D4/1+D5/0+D11/0
# Name: 中空精灵+矩形hub+内嵌填充

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  a_concretization_and_trace:
    original_rule: |
      原题里 hub 是 4x4 空心正方形，四边颜色各异。每个单色实心 sprite
      通过颜色配对到某一边，再旋转到“对边接触”的朝向，贴在 hub 外侧。
    fused_rule: |
      本融合版把三处改动同时落实：
      1. hub 改成固定 7x9 的空心矩形，因此 top/bot 的接触长度是 7，
         left/right 的接触长度是 5。
      2. sprite 不再是实心块，而是带 1 格内孔的中空框架。
      3. sprite 不再贴在外侧，而是把同名边(top 对 top, left 对 left)直接
         压到 hub 彩边上，并向内部伸入。
    concrete_example: |
      例如顶边颜色是红色，右边颜色是绿色：
      - 输入中红色 sprite 是一个横向中空框架，绿色 sprite 是一个纵向中空框架，
        二者散落在 hub 外部。
      - 输出中红色框架旋转后，其 top 接触边与 hub 顶边 7 格彩边重合，剩余像素
        只落在内部中央上方槽位；绿色框架则把 right 接触边压到 hub 右边 5 格彩边，
        剩余像素落在内部右侧槽位。
      - 与原题相比，变化不只是“矩形长度不同”或“改成内嵌”单独成立，而是：
        矩形带来两类接触长度，中空形状要求保留洞，内嵌要求使用 same-side 对齐。

  b_complexity_calibration:
    reason_for_fixed_geometry: |
      若直接沿用 4x6 hub，再把中空框架塞进内部，top/bot 与 left/right 很容易重叠。
      因此把 hub 固定成 7x9，内部 5x7，给四侧各自预留不相交槽位：
      - top/bot 使用 3x7 横向中空框架
      - left/right 使用 5x3 纵向中空框架
      这两类模板分别对应矩形 hub 的两种边长。
    preserved_freedom: |
      仍保留原题关键自由度：颜色一一配对、rot90 恢复、2-4 个真实 sides、
      非恒等散落旋转、随机散落位置、0-1 个干扰物。
    anti_degeneracy: |
      生成器强制每个样例同时出现至少 1 个水平 side 和 1 个垂直 side，
      这样单个样例里就能看到 7 格与 5 格两种接触长度，矩形 hub 的作用可见。

  c_implicit_assumptions_review:
    solver_lines_to_change: |
      原 solver 的关键隐式假设有三条：
      - hub 是 n x n 正方形，必须改成固定 7x9 矩形搜索。
      - 四边接触长度统一为 n-2，必须拆成 7 和 5 两类。
      - 外贴使用 opposite-side 接触边，必须改成内嵌使用 same-side 接触边。
    assumptions_kept: |
      仍保留并由生成器保证：
      - 输入里 hub 内部全是 bg，因此 hub 检测可继续用“内部全空”。
      - 每个真实颜色最多对应一个主要 sprite，且它是该颜色最大 8 连通分量。
      - 每个模板对目标 side 只有 1 个合法 rot90 朝向。
    new_assumptions: |
      新 solver 还假设横向模板只占内部上/下中槽，纵向模板只占内部左/右中槽，
      因此多侧同时存在时不冲突。生成器使用固定模板并在 output-first 阶段逐个检测重叠，
      显式保证这一点。

  d_placement_legality:
    output_first: |
      先构造真实输出：单独画空心矩形 hub，再把每个中空框架嵌入对应 side。
      由于模板是为 7x9 hub 手工设计的，四侧同现也不会越界或互相压色。
    input_scatter: |
      输入不能直接从输出裁 hub 区域，否则会把内嵌内容一起带回去，破坏 hub 检测。
      因此输入单独重建“空心 hub”，再把 sprites 做随机非恒等旋转后散落到外部。

  e_visual_observability:
    what_humans_can_see: |
      从 2-4 组样例里，人类能直接看到三条线索：
      - hub 明显是矩形，不是方形；
      - 不同方向的 sprite 长短不同，匹配不同边长；
      - 输出时 sprite 进入 hub 内部且保留中空洞。
      这些线索都来自可见像素结构，不依赖颜色数值大小或隐藏状态。

  f_visual_cleanliness:
    cleanliness: |
      模板的内部占位被限制在四个互不相交的槽区：
      top/bot 只占中间三列的上/下带，left/right 只占两侧中带。
      因此输出不会在 hub 内部糊成一团，四个方向同时出现也仍可辨认。

  g_post_test_metric:
    metric_definition: |
      量化融合行为时使用两个指标：
      - inner_fill: hub 内部非 bg 像素数，验证 D11 的“内嵌填充”确实发生。
      - enclosed_bg4: hub 内部满足上下左右四邻都非 bg 的 bg 像素数，
        这些像素就是中空框架保留下来的洞，验证 D4/1 没退化回实心。
      若这两个指标长期都 > 0，则说明生成器确实在产生“中空 + 内嵌”的样本。
"""


HUB_H = 7
HUB_W = 9
HORIZ_TARGET = HUB_W - 2
VERT_TARGET = HUB_H - 2
SIDES = ("top", "bot", "left", "right")

_TOP_CANONICAL = [
    np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
]

_LEFT_CANONICAL = [
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )
]


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


def _target_len_for_side(side: str) -> int:
    return HORIZ_TARGET if side in ("top", "bot") else VERT_TARGET


def _matches_attachment(mask: np.ndarray, need_edge: str, target_len: int) -> bool:
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False

    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return False

    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) == int(target_len) and _is_contig(rows)

    _typ, _edge_r, cols = seg
    return len(cols) == int(target_len) and _is_contig(cols)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target_len: int) -> int:
    count = 0
    arr = mask.astype(int)
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _matches_attachment(rot_mask, need_edge, target_len):
            count += 1
    return count


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bot"):
        base = rng.choice(_TOP_CANONICAL).copy()
        mask = base if side == "top" else np.flipud(base)
    else:
        base = rng.choice(_LEFT_CANONICAL).copy()
        mask = base if side == "left" else np.fliplr(base)

    assert _count_valid_rotations(mask, side, _target_len_for_side(side)) == 1
    labeled, num = _label(mask, structure=np.ones((3, 3), dtype=int))
    assert int(num) == 1
    return mask


def _build_hub(out: np.ndarray, hub_r: int, hub_c: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = int(bg)
    out[hub_r, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["top"])
    out[hub_r + HUB_H - 1, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["bot"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c] = int(side_colors["left"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c + HUB_W - 1] = int(side_colors["right"])


def _attachment_offset(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    if side == "top":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(hub_r - edge_r), int(hub_c + 1 - cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(hub_r + HUB_H - 1 - edge_r), int(hub_c + 1 - cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "left")
        return int(hub_r + 1 - rows[0]), int(hub_c - edge_c)
    _typ, edge_c, rows = _edge_segment(mask, "right")
    return int(hub_r + 1 - rows[0]), int(hub_c + HUB_W - 1 - edge_c)


def _scatter_patch(mask: np.ndarray, color: int, bg: int, rng: random.Random) -> np.ndarray:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    krot = rng.choice([1, 2, 3])
    rot_arr = np.rot90(arr, int(krot))
    rot_mask = np.rot90(mask, int(krot))
    rows_any = rot_mask.any(axis=1)
    cols_any = rot_mask.any(axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return rot_arr[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match same-colored hollow sprites to a central hollow rectangle, rotate them,
    and embed them into the hub interior on the matching side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c), (top,bot,left,right))
        for r in range(0, H - HUB_H + 1):
            for c in range(0, W - HUB_W + 1):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + HUB_W - 1]) != int(bg)
                    or int(g[r + HUB_H - 1, c]) != int(bg)
                    or int(g[r + HUB_H - 1, c + HUB_W - 1]) != int(bg)
                ):
                    continue

                inner = g[r + 1 : r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + HUB_W - 1]
                bot = g[r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                left = g[r + 1 : r + HUB_H - 1, c]
                right = g[r + 1 : r + HUB_H - 1, c + HUB_W - 1]

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

                side_cols = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                if len(set(side_cols)) != 4:
                    continue

                cr = float(r + (HUB_H - 1) / 2)
                cc = float(c + (HUB_W - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (float(score), (int(r), int(c)), side_cols)
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
        return best

    hub = _find_hub()
    assert hub is not None

    _score, (r, c), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + HUB_H, c : c + HUB_W] = True

    out = np.full_like(g, int(bg))
    side_colors = {
        "top": int(top_col),
        "bot": int(bot_col),
        "left": int(left_col),
        "right": int(right_col),
    }
    _build_hub(out, int(r), int(c), side_colors, int(bg))

    segs = {
        "top": (int(top_col), ("top", int(r), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "bot": (int(bot_col), ("bot", int(r + HUB_H - 1), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "left": (int(left_col), ("left", int(c), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
        "right": (int(right_col), ("right", int(c + HUB_W - 1), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
    }

    for _side, (col, (need_edge, fixed, span, target)) in segs.items():
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
        spr_arr = np.full_like(g, int(bg))
        spr_arr[spr_mask] = int(col)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        found = None
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            if not _matches_attachment(msk, need_edge, int(target)):
                continue

            seg = _edge_segment(msk, need_edge)
            if seg is None:
                continue

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                val = int(a[int(rr - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                    ok = False
                    break
            if not ok:
                continue

            found = (a, msk, int(dr), int(dc))
            break

        if found is None:
            continue

        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(22, 28)
    W = rng.randint(24, 30)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    while True:
        k = rng.randint(2, 4)
        sprite_sides = rng.sample(list(SIDES), k)
        if any(side in ("top", "bot") for side in sprite_sides) and any(side in ("left", "right") for side in sprite_sides):
            break

    sprite_masks = {side: _make_sprite_mask(side, rng) for side in sprite_sides}

    hub_r_min = 4
    hub_r_max = H - HUB_H - 4
    hub_c_min = 4
    hub_c_max = W - HUB_W - 4
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    _build_hub(out, hub_r, hub_c, side_colors, bg)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        arr = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        sr, sc = _attachment_offset(mask, side, hub_r, hub_c)
        sh, sw = mask.shape

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if not bool(mask[rr, cc]):
                    continue
                dst_val = int(out[sr + rr, sc + cc])
                if dst_val != int(bg) and dst_val != int(side_colors[side]):
                    return None

        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_colors, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        _arr, mask, _sr, _sc = attached_sprites[side]
        patch = _scatter_patch(mask, int(side_colors[side]), int(bg), rng)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dside = rng.choice(list(SIDES))
            dmask = _make_sprite_mask(dside, rng)
            darr = _scatter_patch(dmask, dcol, int(bg), rng)
            try:
                pr, pc = random_free_location(canvas, darr, transparent=int(bg), padding=2, padding_connectivity=8)
            except ValueError:
                pr = pc = None
            if pr is not None and pc is not None:
                paste(canvas, darr, int(pr), int(pc), transparent=int(bg))
                paste(inp, darr, int(pr), int(pc), transparent=int(bg))

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
