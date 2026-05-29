# Combo variant of 2c181942 (index 21)
# combo_spec: D6/1+D7/2
# Name: 同色干扰+非对称唯一旋转

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  fusion_summary: |
    这个融合把原题的两个关键前提同时去掉：
      1) 不能再按“hub 边颜色 -> 最大同色连通块”直接选 sprite，因为输入里会出现一个
         与真 sprite 同色但几何不合法的干扰块。
      2) 不能再按镜像对称性判断旋转，因为真 sprite 被改成完全非对称；正确旋转只能靠
         穷举 4 个旋转后检查“目标接触边是否恰好暴露出 N-2 个连续格”。
    因此新规则是：对每个 hub 边颜色，把所有同色 8 连通分量都拿出来；对每个分量枚举
    4 个旋转，只保留“轮廓完全非对称且对该边恰好只有 1 个旋转可贴附”的那个分量，把它
    贴到对应 hub 边上，其他同色分量全部丢弃。

  a_concretization_and_trace: |
    设 hub 是 4x4，所以 target=N-2=2。左边颜色为红。输入里出现两个红色分量：
      - 真分量 A：一个歪斜的不对称小形状。它在 4 个旋转里，只有 1 个旋转会让“右边界”
        暴露出恰好 2 个连续红格，因此能贴到 hub 左边。
      - 干扰分量 B：颜色也红，而且面积更大，但它要么有 0 个旋转满足该条件，要么轮廓带
        镜像/旋转对称，不满足“完全非对称+唯一旋转”约束。
    原题逻辑会：
      - 先取最大红色分量 B。
      - 再用对称性检查旋转；这在本融合里不再可靠。
    融合后逻辑会：
      - 把 A 和 B 都单独取出。
      - 对 A/B 分别枚举 4 个旋转，统计能否唯一合法贴附。
      - 只接受 A，把 B 排除。
    差异是协同的：如果只做 D6 而仍用对称真精灵，唯一旋转搜索不会成为核心；如果只做 D7
    而没有同色干扰，就不需要遍历同色所有分量。这里两者必须同时成立。

  b_complexity_calibration: |
    为了让变化肉眼可见但不失控，生成器保留原题 2-4 个有效边，只额外强制加入 1 个同色
    干扰块，而且这个干扰块面积严格大于对应真块。这样：
      - 颜色复用在样例中一定可见。
      - “最大分量”启发会被明确打破。
      - 每个颜色组仍然只有少量候选，不会产生过多组合爆炸。
    真精灵来自一小族 5-12 像素、完全非对称的轮廓，避免四边都变得过于混乱。

  c_implicit_assumptions_review: |
    原 main() 中与本融合直接冲突的假设有：
      - “同色块里取最大分量就是目标精灵”：失效。现在必须逐分量检查。
      - “贴附方向的镜像对称性可用来消歧”：失效。现在要统计 4 个旋转中合法者的数量。
      - “找到第一个合法旋转即可”：失效。现在必须要求合法旋转数恰好为 1。
    仍然保留的前提有：
      - hub 仍是中央附近的空心方框，_find_hub() 仍可用。
      - 真精灵和干扰精灵都仍是单色 8 连通对象。
      - 输出语义不变，仍然只保留 hub 和正确贴附后的真精灵。
    新 main() 额外引入的显式假设是：
      - 对每个实际存在的 hub 边颜色，输入里至多有一个分量满足“完全非对称且唯一旋转有效”。
      - generate_input() 会构造并自检这一点。

  d_placement_legality: |
    生成器继续采用 output-first：
      - 先生成 hub 与所有真精灵的贴附输出，逐个检查越界和重叠。
      - 再把真精灵拆下、随机非零旋转后散放到输入。
      - 最后再放一个同色干扰块，并保持 padding=2 的 8 邻域间隔，确保不会与 hub 或其他
        分量连成一个连通块。
    干扰块面积大于对应真块，但仍限制在小 bbox 内，避免大面积挤占画面。

  e_visual_observability: |
    人类在 2-4 个样例上能看到两个稳定线索：
      - 某个 hub 边颜色在输入中会重复出现两次，但输出里只有其中一个轮廓被用上。
      - 被用上的轮廓明显是不对称的，且需要转到某个唯一方向后才能把一条长度为 2 的边
        对齐到 hub 上。
    这些线索都来自可见的几何与颜色重复，不依赖颜色编号大小、扫描顺序或面积排序这种
    不可视因素。

  f_visual_cleanliness: |
    所有对象都保持单色、彼此隔开至少 2 格，不会叠成一团。
    干扰块只增加 1 个，而且复用已有颜色，不引入新的标记系统，所以画面仍然干净。

  g_inferability_of_change: |
    颜色规则没有被复杂化：hub 四边仍是四个不同颜色，颜色仍然只决定目标边。
    变化发生在“同色里谁是真的”与“如何找正确旋转”这两个几何层面，且这两点在少量样例中
    都能反复出现。生成器还会强制至少一个颜色组满足：
      - 同色分量数 >= 2
      - 恰有 1 个分量满足完全非对称 + 唯一有效旋转
      - 干扰分量面积更大
    这样本融合不会退化回原题。
"""


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
    variants = [
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(mask.T),
    ]
    return all(not np.array_equal(mask, other) for other in variants)


def _need_edge_for_side(side: str) -> str:
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


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    lab, ncomp = label_mask(mask, connectivity=8)
    return [lab == int(cid) for cid in range(1, int(ncomp) + 1)]


def _valid_component_candidates(
    g: np.ndarray,
    out: np.ndarray,
    hub_mask: np.ndarray,
    *,
    color: int,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
    bg: int,
) -> list[tuple[int, np.ndarray, np.ndarray, int, int]]:
    all_mask = (g == int(color)) & (~hub_mask)
    if not bool(all_mask.any()):
        return []

    candidates: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    for comp_mask_full in _component_masks(all_mask):
        comp_arr = np.where(comp_mask_full, int(color), int(bg)).astype(int)
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask_full)

        if not _is_fully_asymmetric(comp_mask):
            continue

        placements = []
        for krot in range(4):
            cand = _rotation_attachment(
                np.rot90(comp_arr, int(krot)),
                np.rot90(comp_mask, int(krot)),
                need_edge=need_edge,
                fixed=int(fixed),
                span=span,
                target=int(target),
            )
            if cand is None:
                continue

            arr2, mask2, dr, dc = cand
            placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(mask2)}
            if any(not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]) for rr, cc in placed):
                continue

            ok = True
            for rr, cc in placed:
                val = int(arr2[int(rr - dr), int(cc - dc)])
                if val == int(bg):
                    continue
                if int(out[int(rr), int(cc)]) != int(bg) and int(out[int(rr), int(cc)]) != val:
                    ok = False
                    break
            if ok:
                placements.append((arr2, mask2, int(dr), int(dc)))

        if len(placements) != 1:
            continue

        arr2, mask2, dr, dc = placements[0]
        candidates.append((int(comp_mask.sum()), arr2, mask2, int(dr), int(dc)))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - same-color distractors may exist for a hub-edge color
    - true sprites are fully asymmetric
    - for the target side, the true sprite has exactly one valid rotation whose
      contact edge exposes N-2 contiguous cells on the needed side
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

    for side in ("top", "bot", "left", "right"):
        color, (need_edge, fixed, span) = segs[side]
        candidates = _valid_component_candidates(
            g,
            out,
            hub_mask,
            color=int(color),
            need_edge=need_edge,
            fixed=int(fixed),
            span=span,
            target=int(target),
            bg=int(bg),
        )
        if len(candidates) != 1:
            continue

        _area, arr, mask, dr, dc = candidates[0]
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    """
    Canonical frame = left attachment:
    - need_edge="right"
    - fully asymmetric silhouette
    - exactly one valid rotation for that need_edge
    """
    for _ in range(800):
        h = rng.choice([4, 5])
        w = rng.choice([3, 4, 5])
        run_start = rng.randint(0, int(h - edge_len))
        target_pixels = rng.randint(5, min(12, h * w - max(0, h - edge_len - 1)))

        mask = np.zeros((h, w), dtype=bool)
        mask[int(run_start) : int(run_start + edge_len), int(w - 1)] = True
        anchor_row = rng.choice(list(range(int(run_start), int(run_start + edge_len))))
        mask[int(anchor_row), int(w - 2)] = True

        while int(mask.sum()) < int(target_pixels):
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if bool(mask[int(nr), int(nc)]):
                            continue
                        if nc == int(w - 1) and not (int(run_start) <= nr < int(run_start + edge_len)):
                            continue
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break

            cells = list(frontier)
            weights = []
            for nr, nc in cells:
                weight = 1 + (int(w - 1) - int(nc))
                if int(nc) <= int(w - 3):
                    weight += 2
                if not (int(run_start) <= int(nr) < int(run_start + edge_len)):
                    weight += 1
                weights.append(float(weight))
            nr, nc = rng.choices(cells, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) > 12:
            continue
        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_rotations(mask, "right", edge_len) != 1:
            continue
        return mask

    return None


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    need_edge = _need_edge_for_side(side)
    for _ in range(250):
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


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr = rng.randrange(h)
    sc = rng.randrange(w)
    mask[int(sr), int(sc)] = True

    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[int(nr), int(nc)]):
                        frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[int(nr), int(nc)] = True

    return _crop_mask(mask)


def _make_same_color_distractor_mask(
    edge_len: int,
    side: str,
    min_area: int,
    rng: random.Random,
) -> np.ndarray | None:
    """
    Build a same-color component that fails the fused matching rule for this side.
    It must not simultaneously satisfy:
      - fully asymmetric silhouette
      - exactly one valid rotation for the target side
    """
    need_edge = _need_edge_for_side(side)
    lo = max(int(min_area), 6)

    for _ in range(600):
        h = rng.randint(4, 6)
        w = rng.randint(3, 6)
        hi = min(14, h * w - 1)
        if lo > hi:
            continue
        area = rng.randint(lo, hi)
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < int(lo):
            continue
        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue

        if _is_fully_asymmetric(mask) and _count_valid_rotations(mask, need_edge, edge_len) == 1:
            continue

        return mask

    return None


def _analyze_input_behavior(grid: np.ndarray) -> dict[str, int]:
    """
    Quantifies the fusion signal directly from one generated input.
    Key metric: the number of hub colors whose same-color group contains
    exactly one valid fully-asymmetric unique-rotation component plus at least
    one invalid component.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    if hub is None:
        return {"qualifying_groups": 0, "repeated_groups": 0, "largest_invalid_groups": 0}

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros(g.shape, dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True
    target = int(n - 2)
    segs = _attach_specs(r, c, n, top_col, bot_col, left_col, right_col)

    qualifying = 0
    repeated = 0
    largest_invalid = 0

    for side in ("top", "bot", "left", "right"):
        color, (need_edge, _fixed, _span) = segs[side]
        comps = _component_masks((g == int(color)) & (~hub_mask))
        if len(comps) < 2:
            continue

        repeated += 1
        stats = []
        for comp in comps:
            cropped = _crop_mask(comp)
            area = int(cropped.sum())
            valid = _is_fully_asymmetric(cropped) and _count_valid_rotations(cropped, need_edge, target) == 1
            stats.append((area, valid))

        n_valid = sum(1 for _area, valid in stats if valid)
        n_invalid = sum(1 for _area, valid in stats if not valid)
        if n_valid == 1 and n_invalid >= 1:
            qualifying += 1
        largest_area = max(area for area, _valid in stats)
        valid_area = max(area for area, valid in stats if valid) if n_valid else -1
        if n_valid == 1 and largest_area > valid_area:
            largest_invalid += 1

    return {
        "qualifying_groups": int(qualifying),
        "repeated_groups": int(repeated),
        "largest_invalid_groups": int(largest_invalid),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [col for col in range(10) if col != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
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

    out = np.full((H, W), bg, dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + n - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + n - 1] = int(right_col)

    segs = _attach_specs(hub_r, hub_c, n, top_col, bot_col, left_col, right_col)
    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
        arr2, mask2, sr, sc = placed
        if sr < 0 or sc < 0 or sr + arr2.shape[0] > H or sc + arr2.shape[1] > W:
            return None
        for rr, cc in np.argwhere(mask2):
            if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(out, arr2, int(sr), int(sc), mask=mask2, overwrite=True, clip=False)
        attached_sprites[side] = (arr2, mask2)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask = attached_sprites[side]
        rot_k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(rot_k))
        rotated_mask = np.rot90(mask, int(rot_k))
        cropped_arr, cropped_mask = _crop_to_mask(rotated_arr, rotated_mask)
        patch = np.where(cropped_mask, int(side_colors[side]), bg).astype(int)
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
    real_area = int(sprite_masks[dside].sum())
    dist_mask = _make_same_color_distractor_mask(edge_len, dside, real_area + 1, rng)
    if dist_mask is None:
        return None
    dist_mask = _crop_mask(np.rot90(dist_mask, rng.choice([0, 1, 2, 3])))
    dist_arr = np.where(dist_mask, int(side_colors[dside]), bg).astype(int)
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

    metrics = _analyze_input_behavior(inp)
    if metrics["qualifying_groups"] < 1 or metrics["largest_invalid_groups"] < 1:
        return None

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
