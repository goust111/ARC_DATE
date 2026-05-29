# Variant of 2c181942
# Dimension: D11 — 贴附方向与空间关系 (transform)
# Change: 精灵贴附到hub外侧，接触边紧贴hub边缘（flush），精灵向外延伸... → 精灵嵌入hub内部

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  source_task:
    requested_reference: 5dbc8537
    solver_status: |
      仓库中未找到 5dbc8537 的 solver 源码，只找到了
      ARC-data/ARC-AGI-2/data/evaluation/5dbc8537.json。
      因此无法照搬其 solver 实现细节，只能读取样例，提炼出
      “原本空的腔体/区域在输出中被对象填入”的视觉机制，作为本变体的基线。
    extracted_mechanism: |
      cavity_fill 的核心视觉特征不是“对象仍在原外侧贴附”，而是对象被放进原本空白
      或待填充的内部区域。因此本变体不再让精灵位于 hub 外面，而是让精灵与 hub 对应
      彩边重合后向内伸入，把 hub 的内部空腔部分填满。

  a_concretization_and_trace:
    original_visual: |
      原题输入里，hub 是一个空心方框：四角和内部都是 bg，四条边各有不同颜色。
      若某颜色有对应精灵，输出时该精灵会被旋转后贴到 hub 外侧，与该边紧邻接触，
      接触边在 hub 外面一格，精灵继续向外生长。
    variant_visual: |
      变体中，输入仍然保留“空心 hub + 外部散落精灵”的可读性，方便人类先识别 hub。
      输出则改为：精灵旋转到正确朝向后，不是放在 hub 外，而是把精灵的接触边直接压在
      hub 的对应彩边上，并向 hub 内部伸入。于是 hub 内部从全 bg 变成局部有色填充。
    concrete_trace: |
      例：hub 顶边颜色为红色，N=6，因此边长 target=N-2=4。
      原题：solver 找到红色精灵后，要求其“底边”有 4 个连续像素，然后把这条底边放到
      hub 顶边上一行（row = r-1），精灵主体继续向上伸。
      变体：solver 仍然找红色精灵并旋转，但现在要求其“顶边”有 4 个连续像素，然后把
      这条顶边直接放到 hub 顶边本身（row = r），由于接触边与 hub 边颜色相同，可以重合；
      精灵其余像素则落进 row = r+1, r+2... 的 hub 内部。
      差异的本质是：接触边所对的 sprite side 从 opposite side 改成 same side，平移位置
      从 hub 外一格改成 hub 边框本身。

  b_complexity_calibration:
    keep_from_original:
      - 仍按颜色一一配对 side 与 sprite
      - 仍通过 rot90 四选一
      - 仍要求接触边长度恰为 N-2 且连续
      - 仍依赖镜像对称性来限制合法朝向
      - 仍允许 2-4 个真实精灵以及 0-1 个干扰物
    necessary_adaptation: |
      如果继续用原题常见的 N=4，则 hub 内部只有 2x2，四个方向同时向内填会严重重叠，
      D11 无法稳定成立。因此把 hub 固定为 N=6，内部为 4x4，给四侧精灵留下可观察的
      内嵌空间。这是为了让 D11 在视觉和逻辑上都可实现，而不是引入新的任务机制。
    sprite_family: |
      为了保证四侧同时存在时也不互相覆盖，内嵌态精灵使用一个窄而不对称的 4x2 基型：
      [1,0]
      [1,1]
      [1,1]
      [1,0]
      它在接触轴方向上对称，但向内只伸出一条居中的窄舌。这样 top/bot 只占内部中间两列，
      left/right 只占内部中间两行，四个方向可以同时放入同一个 4x4 腔体。
    why_not_simplified_too_far: |
      虽然精灵家族比原生成器更收敛，但仍保留了关键推理自由度：颜色匹配、旋转恢复、
      2-4 个 side 的子集、随机散落位置、随机非恒等初始旋转、以及干扰色精灵。
      同时变体的核心差异“外贴 vs 内嵌”会在每个样例中非常显眼。

  c_implicit_assumptions_review:
    hub_detection: |
      原 solver 的 _find_hub 强依赖“输入里 hub 内部全是 bg”。这个前提在变体输入中仍然保留，
      因为输入仍是空心 hub，只有输出才会被填充，所以这部分无需改。
    copied_hub_area: |
      原 generator 用 out 的 hub 区域直接复制到 inp。原题里这无害，因为 out 的 hub 内部仍为空。
      变体里若继续这么做，会把已经内嵌好的 sprite 一起拷回输入，直接破坏 _find_hub 的前提。
      因此必须改为“单独重建空心 hub”放进输入，而不是从输出截取 hub 方块。
    segs_geometry: |
      原 solver 的 segs 假设 sprite 在 hub 外侧，所以 top 对应 need_edge='bot' 且 fixed=r-1。
      变体里这个前提不成立，必须改成 top 对应 need_edge='top' 且 fixed=r；其余三侧同理。
    overlap_policy: |
      原 solver 的冲突检查允许“已有像素和将要粘贴的像素颜色相同”时重合。
      这个写法在变体里正好变成必要条件，因为接触边与 hub 彩边需要完全重合。
      因此前提仍成立，不需要改写。
    largest_component_pick: |
      原 solver 对每种匹配颜色只取最大 8 连通分量。变体继续依赖“每个真实颜色至多一个真实精灵，
      且不存在同色更大的干扰块”。generator 继续保证这一点。

  d_placement_legality:
    output_side_placements: |
      采用 N=6 时，内部腔体是 4x4。基型精灵内嵌后：
      - top/bot 只占内部中间两列
      - left/right 只占内部中间两行
      并且接触边仅与各自 hub 彩边重合，不与其他 side 的彩边重合。
      因此最多四个方向同时存在也不会颜色冲突。
    input_scatter: |
      输入中的精灵依然散落在 hub 外，由 random_free_location 保证不重叠且与 hub 保持 padding。
      所有散落 patch 都是裁剪后的单色块，因此 solver 的按色提取和 8 连通标号保持稳定。
    bounds: |
      输出中的内嵌 sprite 完全位于 hub 的 6x6 包围盒内；输入中的散落 sprite 通过
      random_free_location 做边界与间距检查。

  e_visual_observability:
    human_readability: |
      观察 2-3 组样例时，解题者会看到：同色精灵总是被旋转后回到同色边，但最终不是贴在外面，
      而是像插片一样插到方框里面。hub 内部从空白变成有结构的彩色填充，这比原题更醒目。
    no_hidden_numeric_rule: |
      不使用颜色数值大小做排序，也不引入不可见状态；所有配对和变换都由可见颜色和相对位置决定。

  f_visual_cleanliness:
    layout: |
      输出只有 hub 与内嵌后的精灵；输入只有空心 hub、离散精灵、少量可选干扰物。
      由于内嵌形状瘦且居中，不会把 hub 内部糊成一团，四个 side 的贡献仍可分辨。

  g_color_inference:
    rule: |
      颜色规则不变：hub 四边颜色各不相同，精灵颜色与边颜色一致即配对。
      颜色没有重新映射，只承担“哪一块对应哪一边”的可见索引作用，因此从少量样例即可推断。

  h_variant_metric_for_post_test:
    metric: |
      量化指标定义为 solved_output 在 hub 内部 (r+1:r+N-1, c+1:c+N-1) 的非 bg 像素数。
      原题外贴时该值恒为 0；本变体只要至少有一个 sprite 成功内嵌，该值就 > 0。
      另外可比较“新 solver 输出”和“原 solver 输出”是否相同，若大多数样本不同，则说明
      generator 没有退化回原始外贴行为。

  i_new_solver_assumptions_to_guarantee:
    assumptions:
      - hub 仍是输入中唯一靠近中心的空心彩边方框
      - 每个真实 side 颜色至多对应一个主要精灵，且该精灵是该颜色最大 8 连通分量
      - 精灵在正确朝向下，接触边恰好是对应 same side 的边，而不是 opposite side
      - 精灵基型保证对该 side 只有一个 rot90 朝向合法
      - 输入 hub 内部保持全 bg，避免 hub 检测失败
    generator_guarantees: |
      下面的 generate_input() 对以上每条都显式构造或通过 oracle 检查保证。
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


def _rotation_valid_for_side(mask: np.ndarray, side: str, target: int) -> bool:
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


def _place_hub_frame(out: np.ndarray, hub_r: int, hub_c: int, n: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + n, hub_c : hub_c + n] = int(bg)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(side_colors["top"])
        out[hub_r + n - 1, hub_c + ci] = int(side_colors["bot"])
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(side_colors["left"])
        out[hub_r + ri, hub_c + n - 1] = int(side_colors["right"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Match sprites to hub sides by color, as in the original task.
    - Rotate each sprite so the side touching the hub is the SAME named side
      (top->top, left->left, etc.), not the opposite side.
    - Paste that contact edge directly on top of the colored hub edge; the rest of
      the sprite extends inward, filling the hollow square's interior.
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

    # Desired inner-embed geometry: the sprite's contact edge coincides with the hub edge itself.
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

        found = None
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            if not _rotation_valid_for_side(msk, need_edge, target):
                continue

            seg = _edge_segment(msk, need_edge)
            assert seg is not None

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
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
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

from scipy.ndimage import label as _label


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a side-specific sprite already in its embedded orientation.

    Canonical "left-embed" base:
      1 0
      1 1
      1 1
      1 0

    Properties:
    - 8-connected
    - contact edge length = 4 along the left side
    - flipud-symmetric
    - unique valid rotation for each target side under this solver
    - after embedding into a 6x6 hub, the inward pixels stay in a narrow central lane,
      so all four sides can coexist without overlap.
    """
    if int(edge_len) != 4:
        return None

    canonical = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=bool,
    )

    labeled, num = _label(canonical, structure=np.ones((3, 3), dtype=int))
    if int(num) != 1:
        return None

    order = list(range(4))
    rng.shuffle(order)
    valid = []
    for k in order:
        rot = np.rot90(canonical, int(k))
        ra = rot.any(axis=1)
        ca = rot.any(axis=0)
        rr0, rr1 = np.where(ra)[0][[0, -1]]
        cc0, cc1 = np.where(ca)[0][[0, -1]]
        trimmed = rot[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if _rotation_valid_for_side(trimmed, side, edge_len):
            valid.append(trimmed)

    if len(valid) != 1:
        return None
    return valid[0]


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

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(m.shape[0] for m in sprite_masks.values())
    max_sprite_w = max(m.shape[1] for m in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    inp = np.full((H, W), bg, dtype=int)
    _place_hub_frame(out, hub_r, hub_c, n, side_colors, bg)
    _place_hub_frame(inp, hub_r, hub_c, n, side_colors, bg)

    attached_sprites = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[0])[0]
            sr = hub_r
            sc = hub_c + 1 - int(filled[0])
        elif side == "bot":
            filled = np.where(mask[-1])[0]
            sr = hub_r + n - sh
            sc = hub_c + 1 - int(filled[0])
        elif side == "left":
            filled = np.where(mask[:, 0])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c
        else:
            filled = np.where(mask[:, -1])[0]
            sr = hub_r + 1 - int(filled[0])
            sc = hub_c + n - sw

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if not bool(mask[rr, cc]):
                    continue
                dst_val = int(out[sr + rr, sc + cc])
                if dst_val != int(bg) and dst_val != color:
                    return None

        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask, sr, sc)

    result = main(inp)
    if np.count_nonzero(result[hub_r + 1 : hub_r + n - 1, hub_c + 1 : hub_c + n - 1] != bg) != 0:
        return None

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, k)
        rotated_mask = np.rot90(mask, k)

        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        cropped_arr = rotated_arr[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        cropped_mask = rotated_mask[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [c for c in range(10) if c != bg and c not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                k = rng.choice([0, 1, 2, 3])
                dist_arr = np.rot90(dist_arr, k)
                try:
                    pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
                    paste(canvas, dist_arr, pr, pc, transparent=bg)
                    paste(inp, dist_arr, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    inner_fill = int(np.count_nonzero(result[hub_r + 1 : hub_r + n - 1, hub_c + 1 : hub_c + n - 1] != bg))
    if inner_fill <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
