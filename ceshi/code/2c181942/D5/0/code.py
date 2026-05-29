# Variant of 2c181942
# Dimension: D5 — 目标结构：hub的形状与设计 (dst)
# Change: NxN空心正方形(N=4)，角为背景色，内部为背景色，四条边各为一种不同颜色，位于网格中心附近... → 空心矩形hub替代正方形

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   preserved_rule:
#     - 仍然是“中央 hub + 按颜色匹配的单色 sprite + 旋转后贴到对应边上”。
#     - 输出仍然只保留 hub 和已贴附的匹配 sprite，其余清空为背景色。
#   core_change:
#     - hub 不再是 4x4 空心正方形，而是固定 4x6 空心矩形。
#     - 因此四边的接触长度不再统一：
#       top/bot 需要 4 格接触段；left/right 需要 2 格接触段。
#   concreteization:
#     - hub 外框尺寸固定为 4x6，四角为 bg，内部 2x4 为空。
#     - 上下边是 4 格纯色水平线，左右边是 2 格纯色竖线，四边颜色两两不同。
#     - sprite 仍是单色、8 连通、散落输入；贴附时：
#       top/bot sprite 需左右镜像对称，且接触边长度为 4；
#       left/right sprite 需上下镜像对称，且接触边长度为 2。
#   behavior_trace:
#     - 原题小例子：4x4 方形 hub 时，所有边的 target 都是 n-2=2。
#     - 本变体：4x6 矩形 hub 时，
#       top side 会寻找“底边 4 连通”的旋转；
#       left side 会寻找“右边 2 连通”的旋转。
#     - 差异点不在颜色配对，而在“同一 solver 里按 side 使用不同 target 长度”。
#   complexity_calibration:
#     - 只改 D5，不引入新的匹配维度；颜色仍是唯一配对键。
#     - 为了让矩形差异在 2-4 个样例中肉眼可见，生成器强制每个样例至少同时出现：
#       1 个水平边 sprite（接触长度 4）和 1 个垂直边 sprite（接触长度 2）。
#     - hub 尺寸固定成 4x6，而不是再开放很多矩形档位；这样人类更容易归纳“规则变了，但主体逻辑没变”。
#   implicit_assumptions_review:
#     - 原 `_find_hub()` 枚举 `n x n`，默认 hub 是正方形；这里必须改成矩形搜索。
#     - 原 `target = n - 2` 把四边接触长度绑定成同一个数；这里必须拆成 horizontal_target / vertical_target。
#     - 原 `segs` 默认四边 span 长度一致；这里 top/bot 的 span 长度为 4，left/right 为 2。
#     - 原 generator 用单一 `edge_len` 造所有 sprite；这里必须分两类模板，否则无法表达矩形带来的长度差。
#   placement_legality:
#     - 生成器仍采用 output-first：先搭建 hub+attached sprites 的真输出，再旋转散落回输入。
#     - hub 四周预留 margin，attached sprite 先做越界/重叠检查，scatter 时用 `random_free_location(..., padding=2)` 保证间距。
#   visual_observability:
#     - 每个样例都能同时看到 4 格接触和 2 格接触两种长度，矩形 hub 与方形 hub 的不同直接可见。
#     - 颜色只用于配对，不参与排序、大小比较或任何不可见规则。
#   new_solver_assumptions_checked_against_generator:
#     - solver 假设 hub 是唯一一个中心附近的 4x6 空心矩形；generator 明确只放一个这样的对象。
#     - solver 假设同色 sprite 的最大 8 连通分量才是目标；generator 不会把同色块拆成多个竞争主体。
#     - solver 假设每个目标 sprite 针对其 side 只有 1 个有效旋转；generator 只使用经过该条件验证的模板。

HUB_H = 4
HUB_W = 6
HORIZ_TARGET = HUB_W - 2
VERT_TARGET = HUB_H - 2
SIDES = ("top", "bot", "left", "right")

_TOP_CANONICAL = [
    np.array(
        [
            [0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

_LEFT_CANONICAL = [
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
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
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
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _matches_attachment(rot_mask, need_edge, target_len):
            count += 1
    return count


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bot"):
        base = rng.choice(_TOP_CANONICAL).copy()
        mask = base if side == "top" else np.flipud(base)
        need_edge = "bot" if side == "top" else "top"
    else:
        base = rng.choice(_LEFT_CANONICAL).copy()
        mask = base if side == "left" else np.fliplr(base)
        need_edge = "right" if side == "left" else "left"

    assert _count_valid_rotations(mask, need_edge, _target_len_for_side(side)) == 1

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
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(hub_r - edge_r - 1), int(hub_c + 1 - cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(hub_r + HUB_H - edge_r), int(hub_c + 1 - cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "right")
        return int(hub_r + 1 - rows[0]), int(hub_c - edge_c - 1)
    _typ, edge_c, rows = _edge_segment(mask, "left")
    return int(hub_r + 1 - rows[0]), int(hub_c + HUB_W - edge_c)


def _scatter_patch(mask: np.ndarray, color: int, bg: int, rng: random.Random) -> np.ndarray:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    krot = rng.choice([1, 2, 3])
    rot_arr = np.rot90(arr, krot)
    rot_mask = np.rot90(mask, krot)
    rows_any = rot_mask.any(axis=1)
    cols_any = rot_mask.any(axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return rot_arr[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach monochrome sprites to the matching sides of a central hollow rectangle.

    Variant rule:
    - The hub is a hollow 4x6 rectangle, not a square.
    - The top/bottom edge contact length is 4, while the left/right edge contact length is 2.
    - Matching still happens by edge color, and only the correctly rotated sprite fits the
      required side-specific symmetry + contact-length constraints.
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
    out[r : r + HUB_H, c : c + HUB_W] = g[r : r + HUB_H, c : c + HUB_W]

    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "bot": (int(bot_col), ("top", int(r + HUB_H), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
        "right": (int(right_col), ("left", int(c + HUB_W), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
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
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
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

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    while True:
        K = rng.randint(2, 4)
        sprite_sides = rng.sample(list(SIDES), K)
        if any(side in ("top", "bot") for side in sprite_sides) and any(
            side in ("left", "right") for side in sprite_sides
        ):
            break

    sprite_masks = {side: _make_sprite_mask(side, rng) for side in sprite_sides}

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_H - margin
    hub_c_min = margin
    hub_c_max = W - HUB_W - margin
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
                if bool(mask[rr, cc]) and int(out[sr + rr, sc + cc]) != int(bg):
                    return None

        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_colors, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        _arr, mask, _sr, _sc = attached_sprites[side]
        patch = _scatter_patch(mask, side_colors[side], bg, rng)
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
            dcol = rng.choice(distractor_colors)
            dside = rng.choice(list(SIDES))
            dmask = _make_sprite_mask(dside, rng)
            darr = _scatter_patch(dmask, dcol, bg, rng)
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
