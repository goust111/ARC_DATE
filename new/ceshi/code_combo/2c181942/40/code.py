# Combo variant of 2c181942 (index 40)
# combo_spec: D6/1+D7/2+D16/0
# Name: 同色干扰+非对称+4连通

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

"""
design_analysis:
  a_concretization_and_trace: |
    原题里，每个 hub 边颜色唯一对应一个散落精灵；solver 用同色筛选、8 连通取最大分量、
    再靠镜像对称性从 4 个旋转里找到正确朝向。
    本融合版同时改三件事：
      1) 所有真实精灵都必须是严格 4 连通；
      2) 真实精灵完全非对称，solver 不再检查 flipud/fliplr，而是逐个尝试 4 个旋转，
         只保留那个“接触边长度正好等于 N-2 且位置可对齐”的唯一旋转；
      3) 会额外出现一个与真实精灵同色的斜对角干扰簇。它在视觉上像一团同色碎块，
         用 8 连通看会并成一个更大的整体；但用 4 连通会自动拆成多个小分量。
    具体追踪一个 left 边颜色 c 的例子：
      - 原题:
        a. 取所有颜色 c 的非 hub 像素
        b. 8 连通标记，取最大分量
        c. 旋转直到出现“右边 2 连通 + flipud 对称”的朝向
      - 融合版:
        a. 取所有颜色 c 的非 hub 像素
        b. 4 连通标记，得到一个真实大分量 + 若干同色小碎块
        c. 先过滤面积过小的碎块
        d. 对剩余候选逐个尝试 4 个旋转
        e. 仅接受“恰有一个旋转满足右边 2 连通且能合法贴附”的完全非对称分量
      - 差异:
        颜色只负责指向 hub 边，不再足以唯一选中对象；
        连通性定义从 8 改为 4，且旋转消歧从“整体对称”改为“唯一接触边匹配”。

  b_complexity_calibration: |
    为避免过度复杂，hub 仍保持原题的 4x4 空心方块，接触长度仍固定为 2。
    变化集中在源精灵侧：
      - 真实精灵面积限制在 5-10，足够看出非对称轮廓；
      - 干扰簇由 2-4 个面积 1-3 的小块组成，总体同色，但每个 4 连通分量都太小，
        不会与真实精灵混淆；
      - 干扰簇内部专门做成斜对角链，因此“4 连通提取”这一变化肉眼可见。
    这样既不会退化成原题，也不会引入过多状态分支。

  c_implicit_assumptions_review: |
    原 main() 的关键假设变化如下：
      - `label_mask(..., connectivity=8)` 不再成立，必须改成 4 连通；
      - “取最大同色分量”不再安全，因为同色干扰在 8 连通意义下可能看起来更大，
        但在 4 连通下会拆成多个小块；因此必须遍历所有 4 连通分量；
      - `flipud/fliplr` 对称性假设被移除，改成统计合法旋转数，并要求唯一；
      - 输出仍然只保留 hub 与真实贴附精灵，所以冲突检测和贴附几何仍可沿用。
    新增假设：
      - 真实精灵面积 >= 5，干扰小块面积 < 5；
      - 真实精灵完全非对称，且对目标边只有一个合法旋转。
    generator 会显式保证这些前提。

  d_placement_legality: |
    继续采用 output-first。先构造 hub + 已贴附真实精灵的输出，再把真实精灵旋转后散落回输入，
    最后额外放入同色斜对角干扰簇。输出阶段逐像素检查越界与重叠；输入阶段用
    `random_free_location(..., padding=2, padding_connectivity=8)` 保证不同对象有清晰间隔。

  e_visual_observability: |
    从 2-4 组样例，人类可以看到：
      - 同一种颜色有时会出现一个“大块真实精灵”和一团“小碎块干扰”；
      - 碎块之间常常只是斜对角接触，暗示不能把它们当成一个整体；
      - 真实精灵轮廓明显不对称，必须转到某个特定朝向后，才会露出长度为 2 的平直接触边。
    这些线索都来自可见的形状、接触方式和位置，不依赖颜色数值大小等不可视因素。

  f_visual_cleanliness: |
    干扰采用同色小簇，而不是额外标记或叠层。真实精灵、hub、干扰簇之间保留留白，
    不会发生覆盖糊成一团的情况。输出会清掉所有输入干扰，只保留规整的贴附结果。

  g_inference_of_change: |
    颜色规则没有扩展成复杂映射，仍然只是“同色配同边”。新增可推断性来自三条重复线索：
      - 同色但小且斜对角粘连的碎块不会被搬运；
      - 真正被搬运的是面积更大、轮廓完整的 4 连通非对称对象；
      - 被搬运前后存在显著旋转变化，而且只有一个朝向能露出正确接触边。

  h_post_test_metric: |
    生成后可用两个量化指标确认融合行为没有退化：
      - `split_small_components`: 某个 hub 边颜色在输入中对应的 4 连通小分量个数，
        应经常大于 0，验证“同色干扰 + 4 连通拆分”确实出现；
      - `orig_solver_differs`: 原始 solver 在这些输入上与本融合 solver 的输出是否不同，
        应高频为真，验证样本没有退化成原题可处理的情形。

  i_new_solver_assumptions_guaranteed_by_generator: |
    generator 保证：
      - 输入中只有一个中心附近 hub；
      - hub 四边颜色互异；
      - 每个真实精灵是严格 4 连通、完全非对称，且对自己的目标边恰有一个合法旋转；
      - 每个同色干扰簇由多个面积 < 5 的 4 连通小块组成，并且这些小块只通过对角邻接串联；
      - 输入经本 solver 处理后与预构造输出完全一致。
"""


BG_COLOR = 8
HUB_N = 4
MIN_REAL_AREA = 5
MAX_REAL_AREA = 10
SIDES = ("top", "bot", "left", "right")

DIAGONAL_DISTRACTOR_MASKS = [
    np.array(
        [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
]


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.zeros(mask.shape, dtype=int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


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


def _rotation_attachment(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
):
    arr, mask = _crop_to_mask(arr, mask)
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        dr = int(span[0] - rows[0])
        dc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target) or not _is_contig(cols):
            return None
        dr = int(fixed - edge_r)
        dc = int(span[0] - cols[0])

    return arr, mask, int(dr), int(dc)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target: int) -> int:
    count = 0
    dummy = np.where(mask, 1, 0).astype(int)
    for krot in range(4):
        cand = _rotation_attachment(
            np.rot90(dummy, int(krot)),
            np.rot90(mask, int(krot)),
            need_edge=need_edge,
            fixed=0,
            span=list(range(int(target))),
            target=int(target),
        )
        if cand is not None:
            count += 1
    return int(count)


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    base = _crop_mask(mask)
    variants = [
        np.rot90(base, 1),
        np.rot90(base, 2),
        np.rot90(base, 3),
        np.fliplr(base),
        np.flipud(base),
        base.T,
        np.fliplr(base.T),
    ]
    return all(not np.array_equal(base, other) for other in variants)


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _rotate_canonical_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _attach_specs(
    r: int,
    c: int,
    n: int,
    top_col: int,
    bot_col: int,
    left_col: int,
    right_col: int,
) -> dict[str, tuple[int, tuple[str, int, list[int]]]]:
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def _find_hub(grid: np.ndarray, bg: int):
    H, W = grid.shape
    best = None
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(grid[int(r), int(c)]) != int(bg)
                    or int(grid[int(r), int(c + n - 1)]) != int(bg)
                    or int(grid[int(r + n - 1), int(c)]) != int(bg)
                    or int(grid[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue

                inner = grid[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue

                top = grid[int(r), int(c + 1) : int(c + n - 1)]
                bot = grid[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = grid[int(r + 1) : int(r + n - 1), int(c)]
                right = grid[int(r + 1) : int(r + n - 1), int(c + n - 1)]

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


def _same_color_component_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    if hub is None:
        return {"split_small_components": 0, "colors_with_small_components": 0}

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros(g.shape, dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    total_small = 0
    colors_with_small = 0
    for col in (top_col, bot_col, left_col, right_col):
        labc, nc = label_mask((g == int(col)) & (~hub_mask), connectivity=4)
        small_here = 0
        for cid in range(1, int(nc) + 1):
            area = int((labc == int(cid)).sum())
            if 0 < area < MIN_REAL_AREA:
                small_here += 1
        total_small += int(small_here)
        if small_here > 0:
            colors_with_small += 1

    return {
        "split_small_components": int(total_small),
        "colors_with_small_components": int(colors_with_small),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused solver:
    - extract same-color source components with 4-connectivity
    - ignore small same-color fragments
    - require the surviving sprite to be fully asymmetric
    - try all 4 rotations and keep only a unique contact-edge match
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    target = int(n - 2)
    segs = _attach_specs(r, c, n, top_col, bot_col, left_col, right_col)

    for side in SIDES:
        col, (need_edge, fixed, span) = segs[side]
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        labc, nc = label_mask(m, connectivity=4)
        candidates = []
        for cid in range(1, int(nc) + 1):
            comp_mask = labc == int(cid)
            area = int(comp_mask.sum())
            if area < MIN_REAL_AREA:
                continue

            comp_arr = np.where(comp_mask, int(col), int(bg)).astype(int)
            comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)
            if not _is_fully_asymmetric(comp_mask):
                continue

            valid_rots = []
            for krot in range(4):
                cand = _rotation_attachment(
                    np.rot90(comp_arr, int(krot)),
                    np.rot90(comp_mask, int(krot)),
                    need_edge=need_edge,
                    fixed=int(fixed),
                    span=span,
                    target=int(target),
                )
                if cand is not None:
                    valid_rots.append(cand)

            if len(valid_rots) != 1:
                continue

            arr, mask, dr, dc = valid_rots[0]
            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(mask)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                val = int(arr[int(rr - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                    ok = False
                    break
            if not ok:
                continue

            candidates.append((int(area), arr, mask, int(dr), int(dc)))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _area, arr, mask, dr, dc = candidates[0]
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(800):
        h = rng.choice([4, 5])
        w = rng.choice([3, 4, 5])
        run_start = rng.randint(0, int(h - edge_len))
        target_pixels = rng.randint(MIN_REAL_AREA, min(MAX_REAL_AREA, int(h * w - max(0, h - edge_len - 1))))

        mask = np.zeros((h, w), dtype=bool)
        mask[int(run_start) : int(run_start + edge_len), int(w - 1)] = True
        anchor_row = rng.choice(list(range(int(run_start), int(run_start + edge_len))))
        mask[int(anchor_row), int(w - 2)] = True

        while int(mask.sum()) < int(target_pixels):
            candidates: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(mask[int(nr), int(nc)]):
                        continue
                    if nc == int(w - 1) and not (int(run_start) <= nr < int(run_start + edge_len)):
                        continue
                    candidates.add((int(nr), int(nc)))
            if not candidates:
                break

            cand_list = list(candidates)
            weights = []
            for nr, nc in cand_list:
                weight = 1 + 3 * (int(w - 1) - int(nc))
                if nc <= int(w - 3):
                    weight += 2
                if not (int(run_start) <= nr < int(run_start + edge_len)):
                    weight += 2
                weights.append(float(weight))
            nr, nc = rng.choices(cand_list, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if not (MIN_REAL_AREA <= int(mask.sum()) <= MAX_REAL_AREA):
            continue
        _lab, nc = label_mask(mask, connectivity=4)
        if int(nc) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_rotations(mask, "right", edge_len) != 1:
            continue
        return mask

    return None


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    need_edge = _contact_side_for_hub_side(side)
    for _ in range(200):
        canonical = _make_canonical_mask(edge_len, rng)
        if canonical is None:
            continue
        mask = _rotate_canonical_to_side(canonical, side)
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_rotations(mask, need_edge, edge_len) != 1:
            continue
        return mask
    return None


def _make_same_color_distractor(real_area: int, rng: random.Random) -> np.ndarray:
    masks = [mask.copy() for mask in DIAGONAL_DISTRACTOR_MASKS]
    masks.sort(key=lambda item: int(item.sum()))
    bigger = [mask for mask in masks if int(mask.sum()) > int(real_area)]
    chosen = rng.choice(bigger if bigger else masks)
    return np.rot90(chosen, rng.choice([0, 1, 2, 3]))


def _scatter_rotated_patch(arr: np.ndarray, mask: np.ndarray, bg: int, rng: random.Random) -> np.ndarray:
    krot = rng.choice([1, 2, 3])
    rotated_arr = np.rot90(arr, int(krot))
    rotated_mask = np.rot90(mask, int(krot))
    cropped_arr, cropped_mask = _crop_to_mask(rotated_arr, rotated_mask)
    return np.where(cropped_mask, cropped_arr, int(bg)).astype(int)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    n = HUB_N
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [col for col in range(10) if col != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}
    k = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), k)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_dist_h = max(mask.shape[0] for mask in DIAGONAL_DISTRACTOR_MASKS)
    max_dist_w = max(mask.shape[1] for mask in DIAGONAL_DISTRACTOR_MASKS)
    margin = max(max_sprite_h, max_sprite_w, max_dist_h, max_dist_w) + 2

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + n - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + n - 1] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    segs = _attach_specs(hub_r, hub_c, n, top_col, bot_col, left_col, right_col)
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)
        _col, (need_edge, fixed, span) = segs[side]
        placed = _rotation_attachment(
            arr,
            mask,
            need_edge=need_edge,
            fixed=int(fixed),
            span=span,
            target=int(edge_len),
        )
        if placed is None:
            return None
        placed_arr, placed_mask, sr, sc = placed
        sh, sw = placed_mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr, cc in np.argwhere(placed_mask):
            if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(out, placed_arr, int(sr), int(sc), mask=placed_mask, overwrite=True, clip=True)
        attached_sprites[side] = (placed_arr, placed_mask)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask = attached_sprites[side]
        patch = _scatter_rotated_patch(arr, mask, bg, rng)
        try:
            pr, pc = random_free_location(
                canvas,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    dside = rng.choice(sprite_sides)
    dcol = int(side_colors[dside])
    dist_mask = _make_same_color_distractor(int(sprite_masks[dside].sum()), rng)
    dist_arr = np.where(dist_mask, dcol, bg).astype(int)
    try:
        pr, pc = random_free_location(
            canvas,
            dist_arr,
            transparent=bg,
            padding=2,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(canvas, dist_arr, pr, pc, transparent=bg)
    paste(inp, dist_arr, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _same_color_component_metrics(inp)
    if metrics["split_small_components"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
